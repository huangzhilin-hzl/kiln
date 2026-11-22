"""Analytical cost model for parallelism selection (TKCC §4.3 / App. D).

    T_pass   = sum_L [ max(T_c^L, D^L / BW) + T_comm^L ]
    tau      = N * T_pass * n_pass / C          (chip-seconds per request, per phase)
    TPM      = 60 * (ISL + OSL) / (tau_P + tau_D)  (tokens / minute / chip)

C (batch) is the largest batch that fits in HBM next to weights + KV cache AND
meets the phase latency target (TTFT for prefill, per-user TPS for decode).

This is level 1 of the paper's 3 fidelity levels (hardware spec). To get to levels 2-3,
replace `ChipSpec` numbers with *measured* GEMM / HBM / collective throughput
(`efficiency` fields) and calibrate against end-to-end forward timings.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

GB = 1e9


@dataclass
class EmpiricalCurves:
    """Measured hardware curves from `kiln probe` (TKCC §4.3 Level 2)."""
    bf16_tflops_by_m: list[tuple[int, float]] = field(default_factory=list)
    fp8_tflops_by_m: list[tuple[int, float]] = field(default_factory=list)
    fp4_tflops_by_m: list[tuple[int, float]] = field(default_factory=list)
    fp4_native_tc: bool = False
    moe_roofline_eff_by_tokens: list[tuple[int, float]] = field(default_factory=list)
    allreduce_by_k: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    alltoall_by_k: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    cudagraph_per_kernel_s: float = 0.0


@dataclass
class ChipSpec:
    name: str
    flops: float              # dense bf16 FLOP/s
    hbm_bw: float             # bytes/s
    hbm_bytes: float
    link_bw: float            # per-GPU interconnect bytes/s (NVLink within node)
    link_latency: float = 5e-6  # per collective call (the term hardware specs miss, Fig. 4b)
    flops_eff: float = 0.6    # measured / peak; calibrate!
    bw_eff: float = 0.8
    link_eff: float = 0.7
    max_chips: int = 8        # chips reachable at link_bw (e.g. one NVL8 node)
    fp8_scale: float = 2.0    # FP8 / BF16 Tensor Core ratio (datasheet=2.0, or calibrated)
    empirical: EmpiricalCurves | None = None


CHIPS = {
    "H100": ChipSpec("H100", 989e12, 3.35e12, 80 * GB, 450 * GB),
    "H20-3e": ChipSpec("H20-3e", 148e12, 4.0e12, 141 * GB, 450 * GB),
    "B200": ChipSpec("B200", 2250e12, 8.0e12, 180 * GB, 900 * GB),
    "B300": ChipSpec("B300", 2250e12, 8.0e12, 288 * GB, 900 * GB),
    "MI355X": ChipSpec("MI355X", 2500e12, 8.0e12, 288 * GB, 538 * GB),
}
CHIPS["h20-3e"] = CHIPS["H20-3e"]


@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    ffn_inter: int = 0          # dense FFN intermediate (0 if all-MoE)
    experts: int = 0
    top_k: int = 0
    expert_inter: int = 0
    shared_experts: int = 0
    bytes_per_param: float = 2.0
    kv_bytes: float = 2.0
    mtp_accept: float = 1.0     # tokens per decode step (speculative / MTP acceptance)
    # Optional architectural overrides (e.g. DeepSeek-V4.1-Flash CED + CSA2 + Engram):
    total_weight_bytes: float | None = None
    prefill_layer_frac: float = 1.0
    kv_bytes_per_tok: float | None = None

    @property
    def is_moe(self) -> bool:
        return self.experts > 0

    def attn_params(self) -> float:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        return q + kv + o

    def ffn_params_total(self) -> float:
        if self.is_moe:
            return 3 * self.hidden * self.expert_inter * (self.experts + self.shared_experts)
        return 3 * self.hidden * self.ffn_inter

    def ffn_params_active(self) -> float:
        if self.is_moe:
            return 3 * self.hidden * self.expert_inter * (self.top_k + self.shared_experts)
        return self.ffn_params_total()

    def kv_bytes_per_token(self) -> float:
        if self.kv_bytes_per_tok is not None:
            return self.kv_bytes_per_tok
        return self.layers * 2 * self.kv_heads * self.head_dim * self.kv_bytes


@dataclass(frozen=True)
class ParallelConfig:
    attn_tp: int
    attn_dp: int
    moe_tp: int
    moe_ep: int

    @property
    def n(self) -> int:
        return self.attn_tp * self.attn_dp

    def label(self) -> str:
        a = f"TP{self.attn_tp}" + (f"·DP{self.attn_dp}" if self.attn_dp > 1 else "")
        m = f"TP{self.moe_tp}" + (f"·EP{self.moe_ep}" if self.moe_ep > 1 else "")
        return f"attn[{a}] ffn[{m}]"


@dataclass
class Workload:
    isl: int = 8192
    osl: int = 1024
    cache_hit: float = 0.0      # fraction of prefill tokens served from prefix cache
    ttft_sla_s: float = 1.5
    decode_tps_sla: float = 40.0


# --------------------------------------------------------------------------- search space
def search_space(m: ModelSpec, max_chips: int) -> list[ParallelConfig]:
    pows = [2 ** i for i in range(0, max_chips.bit_length()) if 2 ** i <= max_chips]
    out = []
    for n in pows:
        for tp, ep in itertools.product(pows, pows):
            if n % tp:
                continue
            dp = n // tp
            if m.is_moe:
                if ep > n or n % ep:
                    continue
                cfg = ParallelConfig(tp, dp, n // ep, ep)
            else:
                if ep != 1:
                    continue
                cfg = ParallelConfig(tp, dp, n, 1)
            out.append(cfg)
    return sorted(set(out), key=lambda c: (c.n, c.attn_tp, c.moe_ep))


def prune(m: ModelSpec, c: ParallelConfig, chip: ChipSpec) -> str | None:
    """Return a reason string if the config is inadmissible, else None."""
    if m.heads % c.attn_tp:
        return "heads not divisible by attn TP"
    if c.attn_tp > m.kv_heads and c.attn_tp % m.kv_heads:
        return "kv heads cannot be replicated evenly"
    if m.is_moe and m.experts % c.moe_ep:
        return "experts not divisible by EP"
    inter = m.expert_inter if m.is_moe else m.ffn_inter
    if inter % c.moe_tp:
        return "ffn intermediate not divisible by TP"
    if weight_bytes_per_chip(m, c) > 0.9 * chip.hbm_bytes:
        return "weights do not fit"
    return None


def weight_bytes_per_chip(m: ModelSpec, c: ParallelConfig) -> float:
    attn = m.attn_params() / c.attn_tp
    emb = 2 * m.vocab * m.hidden / c.attn_tp
    if m.total_weight_bytes is not None:
        attn_emb_total = (m.layers * m.attn_params() + 2 * m.vocab * m.hidden) * m.bytes_per_param
        routed_bytes = max(0.0, m.total_weight_bytes - attn_emb_total)
        return (m.layers * attn + emb) * m.bytes_per_param + routed_bytes / c.n
    ffn = m.ffn_params_total() / c.n
    return (m.layers * (attn + ffn) + emb) * m.bytes_per_param


# --------------------------------------------------------------------------- per-pass time
def _interp_linear(pts: list[tuple[int, float]], x: float) -> float:
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if x <= x1:
            return y0 + (x - x0) / max(x1 - x0, 1e-12) * (y1 - y0)
    return pts[-1][1]


def _interp_comm(pts: list[tuple[int, float]], bytes_: float, asymp_factor: float, eff_bw: float) -> float:
    if bytes_ <= pts[0][0]:
        return pts[0][1]
    if bytes_ <= pts[-1][0]:
        return _interp_linear(pts, bytes_)
    return pts[-1][1] + asymp_factor * (bytes_ - pts[-1][0]) / max(eff_bw, 1.0)


def _allreduce(bytes_: float, k: int, chip: ChipSpec) -> float:
    if k <= 1:
        return 0.0
    pts = chip.empirical.allreduce_by_k.get(k) if chip.empirical else None
    if pts:
        return _interp_comm(pts, bytes_, 2 * (k - 1) / k, chip.link_bw * chip.link_eff)
    return 2 * (k - 1) / k * bytes_ / (chip.link_bw * chip.link_eff) + 2 * chip.link_latency


def _alltoall(bytes_: float, k: int, chip: ChipSpec) -> float:
    if k <= 1:
        return 0.0
    pts = chip.empirical.alltoall_by_k.get(k) if chip.empirical else None
    if pts:
        return _interp_comm(pts, bytes_, (k - 1) / k, chip.link_bw * chip.link_eff)
    return (k - 1) / k * bytes_ / (chip.link_bw * chip.link_eff) + chip.link_latency


def _compute_ceiling(m: ModelSpec, chip: ChipSpec, tok_attn: int, prefill: bool) -> float:
    emp = chip.empirical
    if emp is not None:
        curve = emp.bf16_tflops_by_m
        if m.bytes_per_param <= 0.6 and emp.fp4_native_tc and emp.fp4_tflops_by_m:
            curve = emp.fp4_tflops_by_m
        elif m.bytes_per_param <= 1.0 and emp.fp8_tflops_by_m:
            curve = emp.fp8_tflops_by_m
        if curve:
            tflops = _interp_linear(curve, tok_attn) if prefill else curve[-1][1]
            return max(1e9, tflops * 1e12)
    fp_scale = chip.fp8_scale if m.bytes_per_param <= 1.0 else 1.0   # FP8 Tensor Core speedup over BF16
    return chip.flops * fp_scale * chip.flops_eff


def pass_time(m: ModelSpec, c: ParallelConfig, chip: ChipSpec, batch: int,
              tokens_per_seq: int, ctx_len: int, prefill: bool) -> float:
    """Wall time of one forward pass for `batch` sequences across the replica."""
    seqs_per_dp = max(1, -(-batch // c.attn_dp))
    tok_attn = seqs_per_dp * tokens_per_seq           # tokens one attention group processes
    tok_total = batch * tokens_per_seq
    act = max(1.0, m.bytes_per_param)
    F, BW = _compute_ceiling(m, chip, tok_attn, prefill), chip.hbm_bw * chip.bw_eff

    # attention projections + score/value matmuls
    attn_flops = 2 * tok_attn * m.attn_params() / c.attn_tp
    attn_flops += 4 * tok_attn * ctx_len * m.heads * m.head_dim / c.attn_tp
    attn_bytes = m.attn_params() * m.bytes_per_param / c.attn_tp
    kv_per_layer = m.kv_bytes_per_token() / max(m.layers, 1)
    attn_bytes += seqs_per_dp * ctx_len * kv_per_layer / max(1, min(c.attn_tp, m.kv_heads))

    # FFN / MoE (per chip share of total routed work)
    ffn_flops = 2 * tok_total * m.ffn_params_active() / c.n
    if m.is_moe:
        # experts touched this pass (cap at all experts) -> weight bytes actually streamed
        touched = min(m.experts, tok_total * m.top_k) + m.shared_experts
        ffn_bytes = 3 * m.hidden * m.expert_inter * touched * m.bytes_per_param / c.n
    else:
        ffn_bytes = m.ffn_params_total() * m.bytes_per_param / c.n

    emp = chip.empirical
    moe_eff = 1.0
    if m.is_moe and emp is not None and emp.moe_roofline_eff_by_tokens:
        moe_eff = max(0.05, _interp_linear(emp.moe_roofline_eff_by_tokens, tok_total))

    if prefill:
        t_layer = attn_flops / F + (ffn_flops / F) / moe_eff        # compute-bound
    else:
        t_layer = max(attn_flops / F, attn_bytes / BW) + max(ffn_flops / F, ffn_bytes / BW) / moe_eff
    if emp is not None and emp.cudagraph_per_kernel_s > 0:
        t_layer = max(t_layer, 8.0 * emp.cudagraph_per_kernel_s)

    comm = _allreduce(tok_attn * m.hidden * act, c.attn_tp, chip)
    if m.is_moe and c.moe_ep > 1:
        comm += 2 * _alltoall(tok_total * m.top_k * m.hidden * act / c.moe_ep, c.moe_ep, chip)
    comm += _allreduce(tok_total * m.hidden * act / c.moe_ep, c.moe_tp, chip)
    if c.n > chip.max_chips:          # crude cross-node penalty
        comm *= 4

    eff_layers = m.layers * (m.prefill_layer_frac if prefill else 1.0)
    lm_head = 2 * tok_attn * m.vocab * m.hidden / c.attn_tp / F
    return eff_layers * (t_layer + comm) + lm_head


# --------------------------------------------------------------------------- throughput
@dataclass
class Estimate:
    config: str
    n: int
    prefill_batch: int
    decode_batch: int
    ttft_s: float
    decode_tps_user: float
    tpm_per_chip: float


def _max_batch_memory(m: ModelSpec, c: ParallelConfig, chip: ChipSpec, seq_len: int) -> int:
    free = 0.9 * chip.hbm_bytes * c.n - weight_bytes_per_chip(m, c) * c.n
    per_seq = m.kv_bytes_per_token() * seq_len / max(1, min(c.attn_tp, m.kv_heads)) * c.attn_tp
    return max(0, int(free // per_seq))


def estimate(m: ModelSpec, c: ParallelConfig, chip: ChipSpec, w: Workload) -> Estimate | None:
    ctx = w.isl + w.osl
    c_mem = _max_batch_memory(m, c, chip, ctx)
    if c_mem < 1:
        return None
    new_tokens = max(1, int(w.isl * (1 - w.cache_hit)))

    # prefill: largest batch meeting TTFT
    cp, tp_ = 0, None
    b = 1
    while b <= c_mem:
        t = pass_time(m, c, chip, b, new_tokens, w.isl, prefill=True)
        if t > w.ttft_sla_s:
            break
        cp, tp_ = b, t
        b *= 2
    # decode: largest batch meeting per-user TPS
    cd, td = 0, None
    b = 1
    while b <= c_mem:
        t = pass_time(m, c, chip, b, 1, ctx, prefill=False)
        if m.mtp_accept / t < w.decode_tps_sla:
            break
        cd, td = b, t
        b *= 2
    if not cp or not cd:
        return None
    tau_p = c.n * tp_ * 1 / cp
    tau_d = c.n * td * (w.osl / m.mtp_accept) / cd
    tpm = 60 * (w.isl + w.osl) / (tau_p + tau_d)
    return Estimate(c.label(), c.n, cp, cd, tp_, m.mtp_accept / td, tpm)


def rank(m: ModelSpec, chip: ChipSpec, w: Workload, max_chips: int = 64, k: int = 5) -> list[dict]:
    rows = []
    for c in search_space(m, max_chips):
        if prune(m, c, chip):
            continue
        e = estimate(m, c, chip, w)
        if e:
            rows.append(asdict(e))
    rows.sort(key=lambda r: r["tpm_per_chip"], reverse=True)
    return rows[:k]


# --------------------------------------------------------------------------- PD disaggregation (xPyD)
@dataclass
class DisaggEstimate:
    ratio: str
    prefill_config: str
    decode_config: str
    prefill_replicas: int
    decode_replicas: int
    prefill_gpus_per_replica: int
    decode_gpus_per_replica: int
    total_gpus: int
    prefill_batch: int
    decode_batch: int
    ttft_s: float
    kv_xfer_ms: float
    decode_tps_user: float
    tpm_per_chip: float
    tpm_ideal_per_chip: float


def estimate_disagg(
    m: ModelSpec,
    c_p: ParallelConfig,
    c_d: ParallelConfig,
    chip: ChipSpec,
    w: Workload,
    *,
    kv_bw_per_gpu: float = 50 * GB,
    kv_eff: float = 0.75,
    cluster_gpus: int | None = None,
    max_replicas_per_role: int = 8,
) -> DisaggEstimate | None:
    """Analytical throughput & `xPyD` sizing for asymmetric Prefill (`c_p`) + Decode (`c_d`)."""
    ctx = w.isl + w.osl
    c_mem_p = _max_batch_memory(m, c_p, chip, max(1, w.isl))
    c_mem_d = _max_batch_memory(m, c_d, chip, ctx)
    if c_mem_p < 1 or c_mem_d < 1:
        return None

    new_tokens = max(1, int(w.isl * (1 - w.cache_hit)))
    eff_kv_bw = max(1.0, min(c_p.n, c_d.n) * kv_bw_per_gpu * kv_eff)

    # Prefill: largest batch whose compute + Mooncake/RDMA KV transfer meets TTFT SLA.
    cp, tp_, tkv_ = 0, None, 0.0
    b = 1
    while b <= c_mem_p:
        t_comp = pass_time(m, c_p, chip, b, new_tokens, w.isl, prefill=True)
        t_kv = (b * new_tokens * m.kv_bytes_per_token()) / eff_kv_bw
        if t_comp + t_kv > w.ttft_sla_s:
            break
        cp, tp_, tkv_ = b, t_comp, t_kv
        b *= 2

    # Decode: largest batch meeting per-user interactivity SLA.
    cd, td = 0, None
    b = 1
    while b <= c_mem_d:
        t = pass_time(m, c_d, chip, b, 1, ctx, prefill=False)
        if m.mtp_accept / t < w.decode_tps_sla:
            break
        cd, td = b, t
        b *= 2

    if not cp or not cd or tp_ is None or td is None:
        return None

    # Steady-state replica request rates (async KV DMA overlaps subsequent prefill compute).
    p_step = max(tp_, tkv_)
    d_total = td * (w.osl / m.mtp_accept)
    r_p = cp / max(p_step, 1e-12)
    r_d = cd / max(d_total, 1e-12)

    tau_p = c_p.n / r_p
    tau_d = c_d.n / r_d
    tpm_ideal = 60.0 * (w.isl + w.osl) / (tau_p + tau_d)

    best_xy: tuple[int, int] | None = None
    best_tpm = -1.0
    best_gpus = 10**9
    max_x = max_replicas_per_role if cluster_gpus is None else max(1, cluster_gpus // c_p.n)
    max_y = max_replicas_per_role if cluster_gpus is None else max(1, cluster_gpus // c_d.n)
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            tot_g = x * c_p.n + y * c_d.n
            if cluster_gpus is not None and tot_g > cluster_gpus:
                continue
            sys_rps = min(x * r_p, y * r_d)
            tpm_disc = 60.0 * (w.isl + w.osl) * sys_rps / tot_g
            # Prefer higher TPM/chip; break <0.5% ties toward smaller clusters.
            if tpm_disc > best_tpm * 1.005 or (tpm_disc >= best_tpm * 0.995 and tot_g < best_gpus):
                best_xy = (x, y)
                best_tpm = tpm_disc
                best_gpus = tot_g

    if best_xy is None:
        return None
    x, y = best_xy
    return DisaggEstimate(
        ratio=f"{x}P:{y}D",
        prefill_config=c_p.label(),
        decode_config=c_d.label(),
        prefill_replicas=x,
        decode_replicas=y,
        prefill_gpus_per_replica=c_p.n,
        decode_gpus_per_replica=c_d.n,
        total_gpus=best_gpus,
        prefill_batch=cp,
        decode_batch=cd,
        ttft_s=tp_ + tkv_,
        kv_xfer_ms=tkv_ * 1000.0,
        decode_tps_user=m.mtp_accept / td,
        tpm_per_chip=best_tpm,
        tpm_ideal_per_chip=tpm_ideal,
    )


def rank_disagg(
    m: ModelSpec,
    chip: ChipSpec,
    w: Workload,
    max_chips: int = 64,
    k: int = 5,
    *,
    kv_bw_per_gpu: float = 50 * GB,
    cluster_gpus: int | None = None,
) -> list[dict]:
    """Rank asymmetric `(c_prefill, c_decode)` disaggregated configurations by `tpm_per_chip`."""
    admissible = [c for c in search_space(m, max_chips) if not prune(m, c, chip)]
    rows: list[dict] = []
    for c_p in admissible:
        for c_d in admissible:
            est = estimate_disagg(
                m,
                c_p,
                c_d,
                chip,
                w,
                kv_bw_per_gpu=kv_bw_per_gpu,
                cluster_gpus=cluster_gpus,
            )
            if est is not None:
                rows.append(asdict(est))
    rows.sort(key=lambda r: (r["tpm_per_chip"], r["tpm_ideal_per_chip"], -r["total_gpus"]), reverse=True)
    return rows[:k]


# --------------------------------------------------------------------------- presets
MODELS = {
    "llama-3.2-1b": ModelSpec("llama-3.2-1b", layers=16, hidden=2048, heads=32, kv_heads=8,
                              head_dim=64, vocab=128256, ffn_inter=8192),
    "llama-3.1-70b": ModelSpec("llama-3.1-70b", layers=80, hidden=8192, heads=64, kv_heads=8,
                               head_dim=128, vocab=128256, ffn_inter=28672),
    "qwen3-30b-a3b": ModelSpec("qwen3-30b-a3b", layers=48, hidden=2048, heads=32, kv_heads=4,
                               head_dim=128, vocab=151936, experts=128, top_k=8, expert_inter=768),
    # 552B MoE (MXFP4 experts + FP8 dense) + 196.6B FP8 Engram (~510 GB checkpoint),
    # 20 causal encoder + 20 decoder layers, CSA2 + FP4 KV (~890 B/tok), DSpark 5-tok spec decode.
    "deepseek-v4.1-flash": ModelSpec(
        "deepseek-v4.1-flash", layers=40, hidden=5120, heads=64, kv_heads=1, head_dim=128,
        vocab=129280, experts=256, top_k=8, expert_inter=2304, shared_experts=1,
        bytes_per_param=0.7, kv_bytes=0.5, mtp_accept=3.2,
        total_weight_bytes=510 * GB, prefill_layer_frac=0.5, kv_bytes_per_tok=890.0,
    ),
}
MODELS["deepseek4.1-flash"] = MODELS["deepseek-v4.1-flash"]
MODELS["dsv41-flash"] = MODELS["deepseek-v4.1-flash"]
