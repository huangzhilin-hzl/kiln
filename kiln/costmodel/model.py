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
from dataclasses import asdict, dataclass

GB = 1e9


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


CHIPS = {
    "H100": ChipSpec("H100", 989e12, 3.35e12, 80 * GB, 450 * GB),
    "B200": ChipSpec("B200", 2250e12, 8.0e12, 180 * GB, 900 * GB),
    "MI355X": ChipSpec("MI355X", 2500e12, 8.0e12, 288 * GB, 538 * GB),
}


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
    ffn = m.ffn_params_total() / c.n
    emb = 2 * m.vocab * m.hidden / c.attn_tp
    return (m.layers * (attn + ffn) + emb) * m.bytes_per_param


# --------------------------------------------------------------------------- per-pass time
def _allreduce(bytes_: float, k: int, chip: ChipSpec) -> float:
    if k <= 1:
        return 0.0
    return 2 * (k - 1) / k * bytes_ / (chip.link_bw * chip.link_eff) + 2 * chip.link_latency


def _alltoall(bytes_: float, k: int, chip: ChipSpec) -> float:
    if k <= 1:
        return 0.0
    return (k - 1) / k * bytes_ / (chip.link_bw * chip.link_eff) + chip.link_latency


def pass_time(m: ModelSpec, c: ParallelConfig, chip: ChipSpec, batch: int,
              tokens_per_seq: int, ctx_len: int, prefill: bool) -> float:
    """Wall time of one forward pass for `batch` sequences across the replica."""
    seqs_per_dp = max(1, -(-batch // c.attn_dp))
    tok_attn = seqs_per_dp * tokens_per_seq           # tokens one attention group processes
    tok_total = batch * tokens_per_seq
    act = m.bytes_per_param
    F, BW = chip.flops * chip.flops_eff, chip.hbm_bw * chip.bw_eff

    # attention projections + score/value matmuls
    attn_flops = 2 * tok_attn * m.attn_params() / c.attn_tp
    attn_flops += 4 * tok_attn * ctx_len * m.heads * m.head_dim / c.attn_tp
    attn_bytes = m.attn_params() * m.bytes_per_param / c.attn_tp
    attn_bytes += seqs_per_dp * ctx_len * 2 * m.kv_heads * m.head_dim * m.kv_bytes / max(1, min(c.attn_tp, m.kv_heads))

    # FFN / MoE (per chip share of total routed work)
    ffn_flops = 2 * tok_total * m.ffn_params_active() / c.n
    if m.is_moe:
        # experts touched this pass (cap at all experts) -> weight bytes actually streamed
        touched = min(m.experts, tok_total * m.top_k) + m.shared_experts
        ffn_bytes = 3 * m.hidden * m.expert_inter * touched * m.bytes_per_param / c.n
    else:
        ffn_bytes = m.ffn_params_total() * m.bytes_per_param / c.n

    if prefill:
        t_layer = (attn_flops + ffn_flops) / F                      # compute-bound
    else:
        t_layer = max(attn_flops / F, attn_bytes / BW) + max(ffn_flops / F, ffn_bytes / BW)

    comm = _allreduce(tok_attn * m.hidden * act, c.attn_tp, chip)
    if m.is_moe and c.moe_ep > 1:
        comm += 2 * _alltoall(tok_total * m.top_k * m.hidden * act / c.moe_ep, c.moe_ep, chip)
    comm += _allreduce(tok_total * m.hidden * act / c.moe_ep, c.moe_tp, chip)
    if c.n > chip.max_chips:          # crude cross-node penalty
        comm *= 4

    lm_head = 2 * tok_attn * m.vocab * m.hidden / c.attn_tp / F
    return m.layers * (t_layer + comm) + lm_head


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


# --------------------------------------------------------------------------- presets
MODELS = {
    "llama-3.2-1b": ModelSpec("llama-3.2-1b", layers=16, hidden=2048, heads=32, kv_heads=8,
                              head_dim=64, vocab=128256, ffn_inter=8192),
    "llama-3.1-70b": ModelSpec("llama-3.1-70b", layers=80, hidden=8192, heads=64, kv_heads=8,
                               head_dim=128, vocab=128256, ffn_inter=28672),
    "qwen3-30b-a3b": ModelSpec("qwen3-30b-a3b", layers=48, hidden=2048, heads=32, kv_heads=4,
                               head_dim=128, vocab=151936, experts=128, top_k=8, expert_inter=768),
}
