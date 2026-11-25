"""Shared torch.profiler analysis, baseline trace diff, stream bubble & roofline attribution."""
from __future__ import annotations

import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("collective_mhc", ("nccl", "all_reduce", "allreduce", "reduce_scatter", "all_gather", "all_to_all", "alltoall", "multicast", "mhc", "lamport", "symm")),
    ("moe", ("moe", "humming", "topk", "router", "expert", "grouped_gemm")),
    ("attention_mla", ("mla", "flash_attn", "flashinfer", "fmha", "paged_attention", "attention", "attn")),
    ("gemm_gemv", ("gemm", "gemv", "cutlass", "wgmma", "mma", "cublas", "nvjet", "matmul")),
    ("norm_rope_act", ("rmsnorm", "layernorm", "rope", "rotary", "silu", "swiglu", "gelu", "norm", "softmax")),
)


def classify_kernel_category(name: str) -> str:
    """Map a CUDA/Triton kernel name to a coarse execution category."""
    low = name.lower()
    for category, needles in _CATEGORY_RULES:
        if any(tok in low for tok in needles):
            return category
    return "other"


@dataclass
class KernelStat:
    name: str
    dur_us: float
    launches: int
    category: str
    streams: list[str] = field(default_factory=list)


@dataclass
class TraceAnalysis:
    trace_path: str
    total_kernel_us: float
    total_launches: int
    span_us: float
    busy_us: float
    bubble_us: float
    bubble_pct: float
    multi_stream_overlap_us: float
    overlap_pct: float
    comm_contention_us: float
    comm_contention_pct: float
    num_streams: int
    category_us: dict[str, float]
    kernels: list[KernelStat]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "total_kernel_us": round(self.total_kernel_us, 3),
            "total_launches": int(self.total_launches),
            "span_us": round(self.span_us, 3),
            "busy_us": round(self.busy_us, 3),
            "bubble_us": round(self.bubble_us, 3),
            "bubble_pct": round(self.bubble_pct, 3),
            "multi_stream_overlap_us": round(self.multi_stream_overlap_us, 3),
            "overlap_pct": round(self.overlap_pct, 3),
            "comm_contention_us": round(self.comm_contention_us, 3),
            "comm_contention_pct": round(self.comm_contention_pct, 3),
            "num_streams": int(self.num_streams),
            "category_us": {k: round(v, 3) for k, v in self.category_us.items()},
            "kernels": [asdict(k) for k in self.kernels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceAnalysis":
        kernels = [
            KernelStat(
                name=str(k["name"]),
                dur_us=float(k["dur_us"]),
                launches=int(k.get("launches", 1)),
                category=str(k.get("category") or classify_kernel_category(str(k["name"]))),
                streams=[str(s) for s in k.get("streams", ())],
            )
            for k in data.get("kernels", ())
        ]
        return cls(
            trace_path=str(data.get("trace_path", "<memory>")),
            total_kernel_us=float(data.get("total_kernel_us", 0.0)),
            total_launches=int(data.get("total_launches", 0)),
            span_us=float(data.get("span_us", 0.0)),
            busy_us=float(data.get("busy_us", 0.0)),
            bubble_us=float(data.get("bubble_us", 0.0)),
            bubble_pct=float(data.get("bubble_pct", 0.0)),
            multi_stream_overlap_us=float(data.get("multi_stream_overlap_us", 0.0)),
            overlap_pct=float(data.get("overlap_pct", 0.0)),
            comm_contention_us=float(data.get("comm_contention_us", 0.0)),
            comm_contention_pct=float(data.get("comm_contention_pct", 0.0)),
            num_streams=int(data.get("num_streams", 0)),
            category_us={str(k): float(v) for k, v in (data.get("category_us") or {}).items()},
            kernels=kernels,
        )


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _intervals_len(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


def _interval_intersection_len(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    if not a or not b:
        return 0.0
    i = j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            overlap += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def _multi_stream_overlap(per_stream: dict[str, list[tuple[float, float]]]) -> float:
    if len(per_stream) < 2:
        return 0.0
    points: list[tuple[float, int]] = []
    for ints in per_stream.values():
        for s, e in _merge_intervals(ints):
            if e > s:
                points.append((s, 1))
                points.append((e, -1))
    if not points:
        return 0.0
    # Process ends (-1) before starts (+1) at identical timestamps.
    points.sort(key=lambda item: (item[0], item[1]))
    active = 0
    prev_t = points[0][0]
    overlap = 0.0
    for t, delta in points:
        if t > prev_t and active >= 2:
            overlap += t - prev_t
        active += delta
        prev_t = t
    return overlap


def _event_stream(ev: dict[str, Any]) -> str:
    args = ev.get("args")
    if isinstance(args, dict):
        if "stream" in args and args["stream"] is not None:
            return str(args["stream"])
        if "cuda_stream" in args and args["cuda_stream"] is not None:
            return str(args["cuda_stream"])
    if ev.get("tid") is not None:
        return str(ev["tid"])
    return "0"


def analyze_trace_events(
    events: Iterable[dict[str, Any]],
    trace_path: str = "<memory>",
) -> TraceAnalysis | None:
    """Aggregate kernel durations, launch counts, bubbles, and multi-stream contention."""
    dur_by_kernel: dict[str, float] = defaultdict(float)
    launches_by_kernel: dict[str, int] = defaultdict(int)
    streams_by_kernel: dict[str, set[str]] = defaultdict(set)
    category_us: dict[str, float] = defaultdict(float)

    all_intervals: list[tuple[float, float]] = []
    stream_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stream_dur: dict[str, float] = defaultdict(float)
    stream_compute_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stream_comm_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "kernel" or "dur" not in ev:
            continue
        try:
            dur = float(ev["dur"])
        except (TypeError, ValueError):
            continue
        if dur < 0:
            continue
        name = str(ev.get("name", "unknown"))
        cat = classify_kernel_category(name)
        stream = _event_stream(ev)

        dur_by_kernel[name] += dur
        launches_by_kernel[name] += 1
        streams_by_kernel[name].add(stream)
        category_us[cat] += dur
        stream_dur[stream] += dur

        ts = ev.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and dur > 0:
            start = float(ts)
            end = start + dur
            iv = (start, end)
            all_intervals.append(iv)
            stream_intervals[stream].append(iv)
            if cat == "collective_mhc":
                stream_comm_intervals[stream].append(iv)
            else:
                stream_compute_intervals[stream].append(iv)

    total_kernel_us = sum(dur_by_kernel.values())
    total_launches = sum(launches_by_kernel.values())
    if not dur_by_kernel or total_launches == 0:
        return None

    if all_intervals:
        merged_all = _merge_intervals(all_intervals)
        span_us = max(0.0, merged_all[-1][1] - merged_all[0][0])
        busy_us = _intervals_len(merged_all)
        bubble_us = max(0.0, span_us - busy_us)
        bubble_pct = (100.0 * bubble_us / span_us) if span_us > 0 else 0.0
        overlap_us = _multi_stream_overlap(stream_intervals)
        overlap_pct = (100.0 * overlap_us / busy_us) if busy_us > 0 else 0.0

        main_stream = max(stream_dur.items(), key=lambda kv: (kv[1], kv[0]))[0]
        main_busy_us = _intervals_len(_merge_intervals(stream_intervals[main_stream]))
        main_compute = _merge_intervals(stream_compute_intervals.get(main_stream, []))
        other_comm = _merge_intervals([
            iv
            for st, ivs in stream_comm_intervals.items()
            if st != main_stream
            for iv in ivs
        ])
        comm_contention_us = _interval_intersection_len(main_compute, other_comm)
        comm_contention_pct = (100.0 * comm_contention_us / main_busy_us) if main_busy_us > 0 else 0.0
    else:
        span_us = total_kernel_us
        busy_us = total_kernel_us
        bubble_us = 0.0
        bubble_pct = 0.0
        overlap_us = 0.0
        overlap_pct = 0.0
        comm_contention_us = 0.0
        comm_contention_pct = 0.0

    kernels = [
        KernelStat(
            name=name,
            dur_us=dur_by_kernel[name],
            launches=launches_by_kernel[name],
            category=classify_kernel_category(name),
            streams=sorted(streams_by_kernel[name]),
        )
        for name in sorted(dur_by_kernel, key=lambda k: dur_by_kernel[k], reverse=True)
    ]

    return TraceAnalysis(
        trace_path=trace_path,
        total_kernel_us=total_kernel_us,
        total_launches=total_launches,
        span_us=span_us,
        busy_us=busy_us,
        bubble_us=bubble_us,
        bubble_pct=bubble_pct,
        multi_stream_overlap_us=overlap_us,
        overlap_pct=overlap_pct,
        comm_contention_us=comm_contention_us,
        comm_contention_pct=comm_contention_pct,
        num_streams=max(len(stream_dur), 1),
        category_us=dict(sorted(category_us.items(), key=lambda kv: kv[1], reverse=True)),
        kernels=kernels,
    )


def analyze_trace(trace_dir: str | Path) -> TraceAnalysis | None:
    """Load the newest `.trace.json(.gz)` in `trace_dir` and return `TraceAnalysis`."""
    p = Path(trace_dir)
    files = sorted(p.glob("*.trace.json*"), key=lambda f: f.stat().st_mtime)
    if not files:
        return None
    latest = files[-1]
    try:
        opener = gzip.open if latest.name.endswith(".gz") else open
        with opener(latest, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return analyze_trace_events(data.get("traceEvents", ()), trace_path=str(latest))
    except Exception:
        return None


def diff_traces(
    candidate: TraceAnalysis,
    baseline: TraceAnalysis | None,
    top_k: int = 12,
) -> dict[str, Any]:
    """Compare candidate trace against baseline trace by kernel, launch count, category, and bubbles."""
    if baseline is None:
        return {
            "has_baseline": False,
            "cand_kernel_ms": round(candidate.total_kernel_us / 1000.0, 3),
            "cand_union_busy_ms": round(candidate.busy_us / 1000.0, 3),
            "cand_span_ms": round(candidate.span_us / 1000.0, 3),
            "cand_launches": candidate.total_launches,
            "cand_bubble_pct": round(candidate.bubble_pct, 2),
            "overlap_pct": round(candidate.overlap_pct, 2),
            "comm_contention_pct": round(candidate.comm_contention_pct, 2),
            "kernel_deltas": [],
            "category_delta_ms": {},
        }

    cand_map = {k.name: k for k in candidate.kernels}
    base_map = {k.name: k for k in baseline.kernels}
    all_names = set(cand_map) | set(base_map)
    deltas: list[dict[str, Any]] = []
    for name in all_names:
        c = cand_map.get(name)
        b = base_map.get(name)
        c_ms = (c.dur_us / 1000.0) if c else 0.0
        b_ms = (b.dur_us / 1000.0) if b else 0.0
        d_ms = c_ms - b_ms
        c_launches = c.launches if c else 0
        b_launches = b.launches if b else 0
        if b is None:
            status = "new"
        elif c is None:
            status = "eliminated"
        elif d_ms < 0:
            status = "faster"
        elif d_ms > 0:
            status = "slower"
        else:
            status = "unchanged"
        deltas.append({
            "name": name,
            "category": (c or b).category,
            "status": status,
            "cand_ms": round(c_ms, 3),
            "base_ms": round(b_ms, 3),
            "delta_ms": round(d_ms, 3),
            "cand_launches": c_launches,
            "base_launches": b_launches,
            "delta_launches": c_launches - b_launches,
        })
    deltas.sort(key=lambda r: abs(r["delta_ms"]), reverse=True)

    cats = set(candidate.category_us) | set(baseline.category_us)
    cat_delta_ms = {
        cat: round((candidate.category_us.get(cat, 0.0) - baseline.category_us.get(cat, 0.0)) / 1000.0, 3)
        for cat in sorted(cats)
    }
    cand_ms_tot = candidate.total_kernel_us / 1000.0
    base_ms_tot = baseline.total_kernel_us / 1000.0
    k_speedup = (base_ms_tot / cand_ms_tot) if cand_ms_tot > 0 and base_ms_tot > 0 else None
    cand_busy_ms = candidate.busy_us / 1000.0
    base_busy_ms = baseline.busy_us / 1000.0
    busy_speedup = (base_busy_ms / cand_busy_ms) if cand_busy_ms > 0 and base_busy_ms > 0 else None

    return {
        "has_baseline": True,
        "cand_kernel_ms": round(cand_ms_tot, 3),
        "base_kernel_ms": round(base_ms_tot, 3),
        "delta_kernel_ms": round(cand_ms_tot - base_ms_tot, 3),
        "kernel_speedup": round(k_speedup, 4) if k_speedup is not None else None,
        "cand_union_busy_ms": round(cand_busy_ms, 3),
        "base_union_busy_ms": round(base_busy_ms, 3),
        "delta_union_busy_ms": round(cand_busy_ms - base_busy_ms, 3),
        "union_busy_speedup": round(busy_speedup, 4) if busy_speedup is not None else None,
        "cand_span_ms": round(candidate.span_us / 1000.0, 3),
        "base_span_ms": round(baseline.span_us / 1000.0, 3),
        "cand_launches": candidate.total_launches,
        "base_launches": baseline.total_launches,
        "delta_launches": candidate.total_launches - baseline.total_launches,
        "cand_bubble_pct": round(candidate.bubble_pct, 2),
        "base_bubble_pct": round(baseline.bubble_pct, 2),
        "delta_bubble_pct": round(candidate.bubble_pct - baseline.bubble_pct, 2),
        "overlap_pct": round(candidate.overlap_pct, 2),
        "comm_contention_pct": round(candidate.comm_contention_pct, 2),
        "category_delta_ms": cat_delta_ms,
        "kernel_deltas": deltas[:top_k],
    }


def _norm_model_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _match_model_spec(model_name: str | None):
    if not model_name:
        return None
    from kiln.costmodel.model import MODELS

    tail = Path(model_name.rstrip("/")).name or model_name
    low = tail.lower()
    if low in MODELS:
        return MODELS[low]
    norm = _norm_model_key(low)
    best = None
    best_len = 0
    for key, spec in MODELS.items():
        k_norm = _norm_model_key(key)
        if norm == k_norm:
            return spec
        if k_norm in norm and len(k_norm) > best_len:
            best = spec
            best_len = len(k_norm)
    return best


def _interp_curve(rows: list[dict[str, Any]], x_key: str, y_key: str, x: float) -> float | None:
    pts = sorted((float(r[x_key]), float(r[y_key])) for r in rows if x_key in r and y_key in r)
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x <= x1:
            return y0 + (x - x0) / max(x1 - x0, 1e-12) * (y1 - y0)
    return pts[-1][1]


def compute_roofline_attribution(
    analysis: TraceAnalysis | None,
    hw_profile: Any | None,
    *,
    model_name: str | None = None,
    tp_size: int = 1,
    decode_tps: float | None = None,
    prefill_tps: float | None = None,
    batch_size: int | None = None,
) -> dict[str, Any] | None:
    """Attribute trace and throughput metrics against measured `hw_profile.json` hardware ceilings."""
    if hw_profile is None:
        return None
    prof = hw_profile.to_dict() if hasattr(hw_profile, "to_dict") else dict(hw_profile)
    bw = prof.get("bandwidth") or {}
    gemm = prof.get("gemm") or {}
    moe = prof.get("moe") or {}
    comm = prof.get("comm") or {}
    ov = prof.get("overhead") or {}
    cc = prof.get("calibrated_chip") or {}
    topo = prof.get("topology") or {}

    hbm_peak_gbps = bw.get("hbm_peak_achieved_gbps") or cc.get("hbm_achieved_gbps") or cc.get("hbm_bw_gbps")
    hbm_peak_gbps = float(hbm_peak_gbps) if hbm_peak_gbps else None

    m_eff = max(1, int(batch_size or 1))
    gemm_dtypes = gemm.get("dtypes") or {}
    gemm_at_m: dict[str, float] = {}
    for dtype_key in ("bf16", "fp8_e4m3", "fp4"):
        info = gemm_dtypes.get(dtype_key) or {}
        if info.get("supported") and info.get("by_m"):
            val = _interp_curve(info["by_m"], "m", "tflops", m_eff)
            if val is not None:
                gemm_at_m[dtype_key] = round(val, 2)

    moe_eff_at_m = None
    if moe.get("supported") and moe.get("by_tokens"):
        val = _interp_curve(moe["by_tokens"], "tokens", "roofline_eff", m_eff)
        if val is not None:
            moe_eff_at_m = round(val, 4)

    groups = comm.get("groups") or {}
    grp = groups.get(str(tp_size)) or groups.get(tp_size) or {}
    if not grp and groups:
        grp = groups[str(max(map(int, groups)))]
    link_latency_us = grp.get("link_latency_us") or cc.get("link_latency_us")
    link_latency_us = float(link_latency_us) if link_latency_us is not None else None
    peak_busbw_gbps = grp.get("peak_busbw_gbps") or cc.get("link_achieved_gbps")
    peak_busbw_gbps = float(peak_busbw_gbps) if peak_busbw_gbps is not None else None

    cg_per_kernel_us = float(ov["cudagraph_per_kernel_us"]) if ov.get("cudagraph_per_kernel_us") is not None else None
    eager_launch_us = float(ov["eager_launch_us"]) if ov.get("eager_launch_us") is not None else None

    achieved_decode_hbm_gbps: float | None = None
    hbm_roofline_pct: float | None = None
    hbm_headroom_pct: float | None = None
    m_spec = _match_model_spec(model_name)
    if m_spec is not None and decode_tps and decode_tps > 0 and batch_size and batch_size > 0 and hbm_peak_gbps:
        tp = max(1, int(tp_size))
        attn_bytes = m_spec.attn_params() * m_spec.bytes_per_param / tp
        if m_spec.is_moe:
            touched = min(m_spec.experts, batch_size * m_spec.top_k) + m_spec.shared_experts
            ffn_bytes = 3 * m_spec.hidden * m_spec.expert_inter * touched * m_spec.bytes_per_param / tp
        else:
            ffn_bytes = m_spec.ffn_params_total() * m_spec.bytes_per_param / tp
        active_bytes_per_rank = m_spec.layers * (attn_bytes + ffn_bytes)
        steps_per_s = float(decode_tps) / max(1.0, float(batch_size) * float(getattr(m_spec, "mtp_accept", 1.0)))
        achieved_decode_hbm_gbps = (active_bytes_per_rank * steps_per_s) / 1e9
        hbm_roofline_pct = 100.0 * achieved_decode_hbm_gbps / hbm_peak_gbps
        hbm_headroom_pct = max(0.0, 100.0 - hbm_roofline_pct)

    bottleneck_flags: list[str] = []
    hints: list[str] = []

    if analysis is not None and analysis.total_kernel_us > 0:
        launch_us_per_op = cg_per_kernel_us if cg_per_kernel_us is not None else eager_launch_us
        if launch_us_per_op is not None and launch_us_per_op > 0:
            est_launch_us = analysis.total_launches * launch_us_per_op
            launch_share_pct = 100.0 * est_launch_us / analysis.total_kernel_us
            avg_kernel_us = analysis.total_kernel_us / max(analysis.total_launches, 1)
            if launch_share_pct >= 15.0 or (avg_kernel_us < 5.0 and analysis.total_launches >= 10):
                bottleneck_flags.append("LAUNCH_BOUND")
                hints.append(
                    f"Estimated launch overhead {est_launch_us:.1f}us ({launch_share_pct:.1f}% of kernel time, "
                    f"avg {avg_kernel_us:.1f}us/kernel): fuse small elementwise/norm/RoPE ops."
                )
        if analysis.num_streams >= 2 and analysis.comm_contention_pct >= 3.0:
            bottleneck_flags.append("STREAM_CONTENTION")
            hints.append(
                f"Aux/collective stream overlaps main-stream compute for {analysis.comm_contention_us / 1000.0:.2f}ms "
                f"({analysis.comm_contention_pct:.1f}% SM contention risk)."
            )
        if analysis.bubble_pct >= 15.0:
            bottleneck_flags.append("LAUNCH_OR_SYNC_BUBBLE")
            hints.append(
                f"GPU idle bubbles account for {analysis.bubble_pct:.1f}% of trace span; check host syncs or graph breaks."
            )
        comm_us = analysis.category_us.get("collective_mhc", 0.0)
        comm_pct = 100.0 * comm_us / analysis.total_kernel_us
        if comm_pct >= 20.0:
            bottleneck_flags.append("COLLECTIVE_LATENCY_BOUND")
            floor_txt = f" (NVLink floor {link_latency_us:.1f}us)" if link_latency_us is not None else ""
            hints.append(
                f"Collectives/mHC take {comm_pct:.1f}% of kernel time{floor_txt}; consider 1-shot/LL multicast or GEMV fusion."
            )

    if hbm_roofline_pct is not None:
        if hbm_roofline_pct >= 75.0:
            bottleneck_flags.append("HBM_BANDWIDTH_BOUND")
            hints.append(
                f"Decode HBM utilization ~{hbm_roofline_pct:.1f}% of {hbm_peak_gbps:,.0f} GB/s ceiling; reduce weight/KV bytes moved."
            )
        elif hbm_roofline_pct < 45.0:
            bottleneck_flags.append("UNDERUTILIZED_HBM")
            hints.append(
                f"Decode HBM utilization ~{hbm_roofline_pct:.1f}% ({hbm_headroom_pct:.1f}% headroom below {hbm_peak_gbps:,.0f} GB/s ceiling)."
            )

    return {
        "device_name": topo.get("device_name") or cc.get("name"),
        "tp_size": int(tp_size),
        "m_eff": m_eff,
        "hbm_peak_gbps": round(hbm_peak_gbps, 2) if hbm_peak_gbps is not None else None,
        "achieved_decode_hbm_gbps": round(achieved_decode_hbm_gbps, 2) if achieved_decode_hbm_gbps is not None else None,
        "hbm_roofline_pct": round(hbm_roofline_pct, 2) if hbm_roofline_pct is not None else None,
        "hbm_headroom_pct": round(hbm_headroom_pct, 2) if hbm_headroom_pct is not None else None,
        "gemm_tflops_at_m": gemm_at_m,
        "moe_roofline_eff_at_m": moe_eff_at_m,
        "link_latency_us": round(link_latency_us, 2) if link_latency_us is not None else None,
        "peak_busbw_gbps": round(peak_busbw_gbps, 2) if peak_busbw_gbps is not None else None,
        "cudagraph_per_kernel_us": round(cg_per_kernel_us, 2) if cg_per_kernel_us is not None else None,
        "eager_launch_us": round(eager_launch_us, 2) if eager_launch_us is not None else None,
        "prefill_tps": round(float(prefill_tps), 2) if prefill_tps else None,
        "bottleneck_flags": bottleneck_flags,
        "hints": hints,
    }


def format_trace_report(
    analysis: TraceAnalysis,
    baseline: TraceAnalysis | None = None,
    roofline: dict[str, Any] | None = None,
    top_k: int = 12,
) -> str:
    """Format a compact kernel + diff + stream-bubble + roofline attribution report."""
    total = analysis.total_kernel_us or 1.0
    lines = [f"trace: {analysis.trace_path}"]
    for k in analysis.kernels[:top_k]:
        lines.append(
            f"{100.0 * k.dur_us / total:5.1f}%  {k.dur_us / 1000.0:8.2f} ms  "
            f"x{k.launches:<4d}  [{k.category}]  {k.name[:80]}"
        )

    if analysis.num_streams > 1 or analysis.bubble_pct > 0.0:
        lines.append(
            f"[Streams & Bubbles] streams={analysis.num_streams} span={analysis.span_us / 1000.0:.2f}ms "
            f"union_busy={analysis.busy_us / 1000.0:.2f}ms "
            f"bubble={analysis.bubble_pct:.1f}% overlap={analysis.overlap_pct:.1f}% "
            f"comm_contention={analysis.comm_contention_pct:.1f}%"
        )

    if baseline is not None:
        td = diff_traces(analysis, baseline, top_k=min(top_k, 8))
        sp_txt = f" ({td['kernel_speedup']:.3f}x)" if td.get("kernel_speedup") else ""
        ub_txt = ""
        if td.get("cand_union_busy_ms") is not None and td.get("delta_union_busy_ms") is not None:
            ub_txt = f", union_busy={td['cand_union_busy_ms']:.2f}ms (delta {td['delta_union_busy_ms']:+.2f}ms)"
        lines.append(
            f"[Trace Diff vs Baseline] total={td['cand_kernel_ms']:.2f}ms "
            f"(delta {td['delta_kernel_ms']:+.2f}ms{sp_txt}){ub_txt}, "
            f"launches={td['cand_launches']} (delta {td['delta_launches']:+d}), "
            f"bubble_delta={td['delta_bubble_pct']:+.2f}%"
        )
        for kd in td.get("kernel_deltas", ())[:5]:
            lines.append(
                f"  delta {kd['delta_ms']:+7.2f} ms  ({kd['delta_launches']:+d} launches, {kd['status']})  {kd['name'][:72]}"
            )

    if roofline is not None:
        parts = []
        if roofline.get("hbm_roofline_pct") is not None:
            parts.append(
                f"decode_hbm={roofline['achieved_decode_hbm_gbps']:.0f}/{roofline['hbm_peak_gbps']:.0f} GB/s "
                f"({roofline['hbm_roofline_pct']:.1f}%, headroom {roofline['hbm_headroom_pct']:.1f}%)"
            )
        elif roofline.get("hbm_peak_gbps") is not None:
            parts.append(f"hbm_ceil={roofline['hbm_peak_gbps']:.0f} GB/s")
        if roofline.get("link_latency_us") is not None:
            parts.append(f"link_floor={roofline['link_latency_us']:.1f}us")
        if roofline.get("bottleneck_flags"):
            parts.append("flags=" + ",".join(roofline["bottleneck_flags"]))
        if parts:
            lines.append("[Roofline Gap] " + " | ".join(parts))
        for h in roofline.get("hints", ())[:3]:
            lines.append(f"  hint: {h}")

    return "\n".join(lines)


def configure_torch_profiler(kwargs: dict, tier_name: str) -> str:
    config = {**(kwargs.get("profiler_config") or {})}
    directory = (config.get("torch_profiler_dir") or os.environ.get("VLLM_TORCH_PROFILER_DIR")
                 or Path("runs/traces") / tier_name)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config.update(profiler="torch", torch_profiler_dir=str(directory))
    kwargs["profiler_config"] = config
    return str(directory)


def summarize_trace(trace_dir: str | Path, top_k: int = 12) -> str:
    p = Path(trace_dir)
    files = sorted(p.glob("*.trace.json*"), key=lambda f: f.stat().st_mtime)
    if not files:
        return f"trace written to {trace_dir}"
    latest = files[-1]
    analysis = analyze_trace(trace_dir)
    if analysis is None:
        return f"trace written to {latest}"
    return format_trace_report(analysis, top_k=top_k)


def build_trace_payload(trace_dir: str | Path, top_k: int = 12) -> tuple[str | None, dict[str, Any] | None]:
    """Return `(formatted_report, serialized_analysis)` for the newest trace in `trace_dir`."""
    analysis = analyze_trace(trace_dir)
    if analysis is None:
        return summarize_trace(trace_dir, top_k=top_k), None
    return format_trace_report(analysis, top_k=top_k), analysis.to_dict()
