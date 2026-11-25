"""AgentX (SemiAnalysis InferenceX) workload for kiln.

Two ways to use the same public traces (HF: semianalysisai/cc-traces-weka-062126, 393 agentic-coding
sessions with subagents; the dataset TKCC used, seed 42):

1. Slow tier, exact: run the real `agentx profile` harness against a `vllm serve` endpoint.
   Methodology is unchanged: pinned harness, public dataset, seed, timing, scoring.
   See `run_agentx_profile` and kiln/engine/vllm_server.py.

2. Fast tier, approximate: `export_offline` turns the same traces into kiln request JSONL for
   offline replay. Exact: per-request input/output lengths and hash-block prefix sharing.
   Approximate: block content is real Python source tokenized with the target tokenizer;
   AgentX uses its own synthetic coding text. Not reproduced: inter-turn timing, subagent DAG
   concurrency, chat template. The export needs only huggingface_hub and transformers.

The slow tier needs `pip install 'agentx-harness==1.0.12'` (the version pinned in the TKCC paper).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_DATASET = "semianalysis_cc_traces_weka_062126"
DEFAULT_HF_REPO = "semianalysisai/cc-traces-weka-062126"


# --------------------------------------------------------------------------- offline export
def _iter_requests(trace: dict) -> list[dict]:
    """Main-agent requests followed by each subagent's requests (in trace order)."""
    out = []
    for r in trace.get("requests", []):
        kind = r.get("type")
        if kind in ("n", "s"):
            out.append(r)
        elif kind == "subagent":
            out.extend(sr for sr in r.get("requests", []) if sr.get("type") in ("n", "s"))
    return out


def _code_corpus(tok, max_tokens: int = 4_000_000) -> list[int]:
    """Real source code from the Python stdlib, tokenized with the target tokenizer.

    Offline and identical on every machine with the same Python version. Used to fill the
    trace hash blocks with code-like content (AgentX fills them with synthetic coding text).
    """
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    ids: list[int] = []
    for f in sorted(root.rglob("*.py")):
        if "test" in f.parts or "idlelib" in f.parts:
            continue
        try:
            ids.extend(tok.encode(f.read_text(errors="ignore"), add_special_tokens=False))
        except Exception:
            continue
        if len(ids) >= max_tokens:
            break
    return ids


def _block_tokens(corpus: list[int], trace_id: str, hash_id: int, bs: int) -> list[int]:
    """Deterministic block for (trace, hash_id): equal ids => identical tokens => prefix hits."""
    import hashlib
    h = int.from_bytes(hashlib.blake2b(f"{trace_id}:{hash_id}".encode(), digest_size=8).digest(), "little")
    start = h % len(corpus)
    blk = corpus[start:start + bs]
    return blk + corpus[: bs - len(blk)] if len(blk) < bs else blk


def traces_path(hf_repo: str = DEFAULT_HF_REPO) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(hf_repo, "traces.jsonl", repo_type="dataset")   # ~1.9 GB, cached


def iter_sampled_traces(path: str, seed: int):
    """Yield traces in a seeded random order, parsing one line at a time (the file is ~1.9 GB)."""
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
    random.Random(seed).shuffle(offsets)
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            yield json.loads(f.readline())


def export_offline(out: str | Path, tokenizer: str, hf_repo: str = DEFAULT_HF_REPO,
                   num_traces: int = 32, turns_per_trace: int = 4, seed: int = 42,
                   max_context: int | None = None, trust_remote_code: bool = True) -> dict:
    """Write kiln JSONL: {"prompt_ids", "output_len", "trace_id", "turn", "t"} per request.

    Prompt i of a trace is the concatenation of its hash blocks (block_size tokens each, from
    the trace), trimmed or padded to the recorded input length. Requests in a trace that
    share hash ids therefore share token prefixes, the same way AgentX does. Takes
    `turns_per_trace` consecutive turns from a seeded random offset in each of `num_traces`
    seeded-random traces.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)
    corpus = _code_corpus(tok)
    rng = random.Random(seed + 1)

    isl, osl, n_tr = [], [], 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for tr in iter_sampled_traces(traces_path(hf_repo), seed):
            if n_tr >= num_traces:
                break
            bs = int(tr.get("block_size", 64))
            reqs = [r for r in _iter_requests(tr) if r.get("hash_ids")]
            if max_context:
                reqs = [r for r in reqs if r["in"] + r["out"] <= max_context]
            if not reqs:
                continue
            start = rng.randrange(max(1, len(reqs) - turns_per_trace + 1))
            for i, r in enumerate(reqs[start:start + turns_per_trace]):
                ids: list[int] = []
                for h in r["hash_ids"]:
                    ids.extend(_block_tokens(corpus, tr["id"], h, bs))
                n_in = int(r["in"])
                if len(ids) < n_in:      # tail beyond the last full block: unique filler
                    ids.extend(_block_tokens(corpus, tr["id"], -1 - i, n_in - len(ids)))
                ids = ids[:n_in]
                f.write(json.dumps({"prompt_ids": ids, "output_len": int(max(1, r["out"])),
                                    "trace_id": tr["id"], "turn": start + i, "t": r.get("t")}) + "\n")
                isl.append(len(ids))
                osl.append(int(r["out"]))
            n_tr += 1

    def _pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    summary = {"traces": n_tr, "requests": len(isl),
               "isl_p50": _pct(isl, .5), "isl_p90": _pct(isl, .9), "isl_max": max(isl, default=0),
               "osl_p50": _pct(osl, .5), "osl_p90": _pct(osl, .9)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- real harness
@dataclass
class AgentXConfig:
    concurrency: int = 8
    public_dataset: str = DEFAULT_DATASET
    random_seed: int = 42
    benchmark_duration_s: int | None = 900      # None uses the harness default
    max_context_length: int | None = None
    unsafe_override: bool = False
    extra_args: list[str] = field(default_factory=list)
    binary: str = "agentx"
    spec_collapse_min_ratio: float | None = 0.30
    spec_collapse_window_s: float = 120.0
    spec_collapse_poll_s: float = 15.0
    spec_collapse_min_drafts: int = 100

    def __post_init__(self) -> None:
        if self.benchmark_duration_s is not None:
            if type(self.benchmark_duration_s) is not int or self.benchmark_duration_s < 1:
                raise ValueError("AgentX benchmark_duration_s must be a positive integer or null")
            if not self.unsafe_override and self.benchmark_duration_s < 900:
                raise ValueError(
                    "AgentX benchmark_duration_s must be at least 900 seconds or null "
                    "(or set unsafe_override: true)"
                )


class SpecAcceptanceCollapsed(RuntimeError):
    """Raised when rolling speculative-decode acceptance collapses far below baseline."""

    def __init__(self, message: str, snapshot: dict[str, Any] | None = None):
        super().__init__(message)
        self.snapshot: dict[str, Any] = dict(snapshot or {})


@dataclass
class AgentXResult:
    total_tput_per_gpu: float          # InferenceX: (input incl. cached + output) tok/s / GPUs
    output_tput: float
    itl_p50_ms: float | None
    itl_p90_ms: float | None
    submission_valid: bool | None
    raw_path: str
    active_decode_tps: float | None = None
    effective_concurrency: float | None = None
    request_count: int | None = None
    decode_duration_sum_ms: float | None = None
    ttft_p50_ms: float | None = None
    benchmark_duration_s: float | None = None
    error_count: int | None = None
    was_cancelled: bool | None = None
    spec_decode_accepted_tokens: float | None = None
    spec_decode_num_drafts: float | None = None
    spec_decode_draft_tokens: float | None = None
    spec_decode_acc_per_draft: float | None = None
    spec_decode_mean_accept_len: float | None = None
    spec_accept_rate: float | None = None
    num_preemptions: float | None = None
    kv_cache_usage_avg: float | None = None
    spec_acceptance_collapsed: bool | None = None
    spec_collapse_detail: str | None = None

    @property
    def interactivity_p90(self) -> float | None:   # InferenceX: 1 / P90 ITL (tok/s/user)
        return 1000.0 / self.itl_p90_ms if self.itl_p90_ms else None


def _num(val) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        return f if f == f and abs(f) != float("inf") else None
    return None


def _metric(data: dict, name: str, key: str = "avg") -> float | None:
    sec = data.get(name)
    if not isinstance(sec, dict):
        return None
    v = sec.get(key)
    if v is None and key == "avg":
        v = sec.get("value")
    return _num(v)


def _error_count(data: dict) -> int | None:
    if "error_summary" not in data:
        return None
    es = data["error_summary"]
    if isinstance(es, list):
        total = 0
        for item in es:
            if isinstance(item, dict) and item.get("count") is not None:
                total += int(item["count"])
            else:
                total += 1
        return total
    if isinstance(es, dict):
        if es.get("total") is not None:
            return int(es["total"])
        return sum(int(v) for v in es.values() if isinstance(v, (int, float)))
    return None


def _csv_float(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _norm_prom_metric_name(name: str) -> str:
    base = name.strip().split("{", 1)[0].strip()
    if base.endswith("_total"):
        base = base[: -len("_total")]
    return base


def parse_server_metrics_csv(export_path: str | Path) -> dict[str, float]:
    """Parse sibling `server_metrics_export.csv` for vLLM spec-decode, preemption, and KV metrics."""
    p = Path(export_path)
    csv_path = p if p.name.endswith(".csv") else p.with_name("server_metrics_export.csv")
    if not csv_path.is_file():
        return {}

    raw_text = csv_path.read_text(encoding="utf-8", errors="ignore")
    if not raw_text.strip():
        return {}

    # AIPerf server_metrics_export.csv can contain multiple blank-line-separated sections.
    sections: list[list[str]] = []
    cur: list[str] = []
    for line in raw_text.splitlines():
        if not line.strip():
            if cur:
                sections.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        sections.append(cur)

    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}

    counter_targets = {
        "vllm:spec_decode_num_accepted_tokens": "spec_decode_accepted_tokens",
        "vllm:spec_decode_num_drafts": "spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens": "spec_decode_draft_tokens",
        "vllm:num_preemptions": "num_preemptions",
    }
    gauge_targets = {
        "vllm:spec_decode_mean_acceptance_length": "spec_decode_mean_accept_len",
        "vllm:spec_decode_avg_acceptance_length": "spec_decode_mean_accept_len",
        "vllm:spec_decode_draft_acceptance_rate": "spec_accept_rate",
        "vllm:kv_cache_usage_perc": "kv_cache_usage_avg",
        "vllm:gpu_cache_usage_perc": "kv_cache_usage_avg",
    }

    for sec_lines in sections:
        rows = list(csv.reader(sec_lines))
        if len(rows) < 2:
            continue
        headers = [h.strip() for h in rows[0]]
        lower_headers = [h.lower() for h in headers]

        name_idx: int | None = None
        for cand_col in ("metric", "metric_name", "metric name", "name"):
            if cand_col in lower_headers:
                name_idx = lower_headers.index(cand_col)
                break
        if name_idx is None:
            continue

        col_idx = {lh: i for i, lh in enumerate(lower_headers)}

        def _get(row: list[str], *keys: str) -> float | None:
            for k in keys:
                idx = col_idx.get(k)
                if idx is not None and idx < len(row):
                    val = _csv_float(row[idx])
                    if val is not None:
                        return val
            return None

        for row in rows[1:]:
            if name_idx >= len(row):
                continue
            mname = _norm_prom_metric_name(row[name_idx])
            if not mname:
                continue

            if mname in counter_targets:
                out_k = counter_targets[mname]
                val = _get(row, "delta", "total", "sum", "change", "increase")
                if val is None:
                    first_v = _get(row, "first", "first_value", "min")
                    last_v = _get(row, "last", "last_value", "max")
                    if first_v is not None and last_v is not None and last_v >= first_v:
                        val = last_v - first_v
                    else:
                        val = _get(row, "value", "avg", "mean", "max")
                if val is not None:
                    counters[out_k] = counters.get(out_k, 0.0) + val
            elif mname in gauge_targets:
                out_k = gauge_targets[mname]
                val = _get(row, "avg", "mean", "value", "p50", "last", "max")
                if val is not None:
                    gauges[out_k] = val

    out: dict[str, float] = {**counters, **gauges}
    acc = out.get("spec_decode_accepted_tokens")
    drafts = out.get("spec_decode_num_drafts")
    draft_toks = out.get("spec_decode_draft_tokens")
    if acc is not None and drafts is not None and drafts > 0:
        acc_per_draft = acc / drafts
        out["spec_decode_acc_per_draft"] = round(acc_per_draft, 6)
        out.setdefault("spec_decode_mean_accept_len", round(1.0 + acc_per_draft, 6))
    elif "spec_decode_mean_accept_len" in out:
        out["spec_decode_acc_per_draft"] = round(max(0.0, out["spec_decode_mean_accept_len"] - 1.0), 6)
    if acc is not None and draft_toks is not None and draft_toks > 0:
        out.setdefault("spec_accept_rate", round(acc / draft_toks, 6))
    return out


def detect_spec_acceptance_collapse(
    cand_acc_per_draft: float | None,
    baseline_acc_per_draft: float | None,
    min_ratio: float | None = 0.30,
) -> tuple[bool, str | None]:
    """Return `(collapsed, detail)` if candidate spec acceptance drops below `min_ratio` of baseline."""
    if (
        min_ratio is None
        or min_ratio <= 0
        or cand_acc_per_draft is None
        or baseline_acc_per_draft is None
        or not math.isfinite(cand_acc_per_draft)
        or not math.isfinite(baseline_acc_per_draft)
        or baseline_acc_per_draft <= 0
    ):
        return False, None
    ratio = cand_acc_per_draft / baseline_acc_per_draft
    if ratio < min_ratio:
        return (
            True,
            f"Spec-decode acceptance collapsed: {cand_acc_per_draft:.3f} vs baseline "
            f"{baseline_acc_per_draft:.3f} acc/draft ({100.0 * ratio:.1f}% < {100.0 * min_ratio:.0f}%)",
        )
    return False, None


def parse_agentx_export(path: str | Path, num_gpus: int) -> AgentXResult:
    data = json.loads(Path(path).read_text())
    # Per-run exports keep measured metrics at the root. Warmup metrics
    # cannot substitute for a missing measurement.
    tot = _metric(data, "total_token_throughput", "avg") or 0.0
    out = _metric(data, "output_token_throughput", "avg") or 0.0
    itl = data.get("inter_token_latency") or {}
    req_cnt = _metric(data, "request_count", "avg")
    if req_cnt is None:
        req_cnt = _metric(data, "decode_duration", "count")
    cancelled = data.get("was_cancelled") if isinstance(data.get("was_cancelled"), bool) else None
    srv = parse_server_metrics_csv(path)
    return AgentXResult(
        total_tput_per_gpu=float(tot) / max(num_gpus, 1),
        output_tput=float(out),
        itl_p50_ms=_num(itl.get("p50")),
        itl_p90_ms=_num(itl.get("p90")),
        submission_valid=(data.get("metadata") or {}).get("submission_valid", data.get("submission_valid")),
        raw_path=str(path),
        active_decode_tps=_metric(data, "active_decode_throughput", "avg"),
        effective_concurrency=_metric(data, "effective_concurrency", "avg"),
        request_count=int(round(req_cnt)) if req_cnt is not None else None,
        decode_duration_sum_ms=_metric(data, "decode_duration", "sum"),
        ttft_p50_ms=_metric(data, "time_to_first_token", "p50"),
        benchmark_duration_s=_metric(data, "benchmark_duration", "avg"),
        error_count=_error_count(data),
        was_cancelled=cancelled,
        spec_decode_accepted_tokens=srv.get("spec_decode_accepted_tokens"),
        spec_decode_num_drafts=srv.get("spec_decode_num_drafts"),
        spec_decode_draft_tokens=srv.get("spec_decode_draft_tokens"),
        spec_decode_acc_per_draft=srv.get("spec_decode_acc_per_draft"),
        spec_decode_mean_accept_len=srv.get("spec_decode_mean_accept_len"),
        spec_accept_rate=srv.get("spec_accept_rate"),
        num_preemptions=srv.get("num_preemptions"),
        kv_cache_usage_avg=srv.get("kv_cache_usage_avg"),
    )


def parse_agentx_requests(export_path: str | Path) -> dict[str, dict]:
    """Read sibling `profile_export.jsonl` keyed by `(conversation_id, turn_index)` (excluding warmup/cancelled)."""
    jsonl_path = Path(export_path).with_name("profile_export.jsonl")
    if not jsonl_path.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in jsonl_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = obj.get("metadata") or {}
        metrics = obj.get("metrics") or {}
        if meta.get("benchmark_phase") == "warmup" or meta.get("was_cancelled") is True:
            continue
        conv = meta.get("conversation_id")
        turn = meta.get("turn_index")
        if conv is not None and turn is not None:
            key = f"{conv}:{turn}"
        elif meta.get("source_trace_id") is not None and turn is not None:
            key = f"{meta['source_trace_id']}:{turn}"
        elif meta.get("x_correlation_id") is not None:
            key = str(meta["x_correlation_id"])
        else:
            continue

        def _val(v):
            return _num(v.get("value") if isinstance(v, dict) else v)

        dec_ms = _val(metrics.get("decode_duration"))
        tok = _val(metrics.get("output_token_count"))
        if tok is None:
            tok = _val(metrics.get("usage_completion_tokens"))
        if dec_ms is None and tok is None:
            continue
        out[key] = {
            "decode_ms": dec_ms or 0.0,
            "output_tokens": tok or 0.0,
        }
    return out


def compare_agentx_paired(baseline_export: str | Path, candidate_export: str | Path) -> dict | None:
    """Compare completed requests shared between baseline and candidate AgentX exports."""
    base = parse_agentx_requests(baseline_export)
    cand = parse_agentx_requests(candidate_export)
    common = sorted(base.keys() & cand.keys())
    if not common:
        return None
    base_ms = sum(base[k]["decode_ms"] for k in common)
    cand_ms = sum(cand[k]["decode_ms"] for k in common)
    base_tok = sum(base[k]["output_tokens"] for k in common)
    cand_tok = sum(cand[k]["output_tokens"] for k in common)
    base_s = base_ms / 1000.0
    cand_s = cand_ms / 1000.0
    decode_sp = (base_s / cand_s) if cand_s > 0 and base_s > 0 else None
    base_tps = (base_tok / base_s) if base_s > 0 else None
    cand_tps = (cand_tok / cand_s) if cand_s > 0 else None
    tput_sp = (cand_tps / base_tps) if (cand_tps is not None and base_tps and base_tps > 0) else None
    return {
        "paired_requests": len(common),
        "baseline_decode_s": round(base_s, 4),
        "candidate_decode_s": round(cand_s, 4),
        "decode_duration_speedup": round(decode_sp, 4) if decode_sp is not None else None,
        "baseline_output_tokens": int(round(base_tok)),
        "candidate_output_tokens": int(round(cand_tok)),
        "paired_output_tps_speedup": round(tput_sp, 4) if tput_sp is not None else None,
    }


def agentx_result_to_extra(
    last: AgentXResult, repeats: list[dict], unsafe_override: bool
) -> dict:
    """Build normalized `MeasureResult.extra` payload for an AgentX evaluation."""
    itl = last.itl_p90_ms
    extra: dict = {
        "metric": "agentx_total_tput_per_gpu",
        "repeats": repeats,
        "output_tput": last.output_tput,
        "itl_p50_ms": last.itl_p50_ms,
        "itl_p90_ms": last.itl_p90_ms,
        "submission_valid": (last.submission_valid is True),
        "unsafe_override": unsafe_override,
        "export": last.raw_path,
        "interactivity_p90": 1000.0 / itl if itl and itl > 0 else None,
    }
    for attr in (
        "active_decode_tps",
        "effective_concurrency",
        "request_count",
        "decode_duration_sum_ms",
        "ttft_p50_ms",
        "benchmark_duration_s",
        "error_count",
        "was_cancelled",
        "spec_decode_accepted_tokens",
        "spec_decode_num_drafts",
        "spec_decode_draft_tokens",
        "spec_decode_acc_per_draft",
        "spec_decode_mean_accept_len",
        "spec_accept_rate",
        "num_preemptions",
        "kv_cache_usage_avg",
        "spec_acceptance_collapsed",
        "spec_collapse_detail",
    ):
        val = getattr(last, attr, None)
        if val is not None:
            extra[attr] = val
    if last.spec_decode_mean_accept_len is not None:
        extra.setdefault("spec_accept_length", last.spec_decode_mean_accept_len)
    return extra


def _default_spec_counter_reader(url: str) -> Callable[[], dict[str, float]]:
    import httpx
    from kiln.engine.base import parse_prometheus_metric

    target = f"{url.rstrip('/')}/metrics"

    def _read() -> dict[str, float]:
        try:
            r = httpx.get(target, trust_env=False, timeout=3.0)
            if r.status_code != 200 or not r.text:
                return {}
            out: dict[str, float] = {}
            acc = parse_prometheus_metric(
                r.text,
                ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
            )
            drafts = parse_prometheus_metric(
                r.text,
                ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"),
            )
            if acc is not None:
                out["accepted_tokens"] = acc
            if drafts is not None:
                out["num_drafts"] = drafts
            return out
        except Exception:
            return {}

    return _read


def run_agentx_profile(
    url: str,
    model: str,
    tokenizer: str,
    cfg: AgentXConfig,
    artifact_dir: str | Path,
    num_gpus: int,
    timeout_s: float,
    *,
    baseline_spec_acc_per_draft: float | None = None,
    metrics_reader: Callable[[], dict[str, float]] | None = None,
) -> AgentXResult:
    from kiln.engine.base import shutdown_process_group

    venv_bin = str(Path(sys.executable).parent)
    path_env = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    binary = shutil.which(cfg.binary, path=path_env)
    if binary is None:
        raise RuntimeError("agentx not found: pip install 'agentx-harness==1.0.12'")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "profile", "--url", url, "--endpoint", "/v1/chat/completions",
           "--model", model, "--tokenizer", tokenizer, "--tokenizer-trust-remote-code",
           "--concurrency", str(cfg.concurrency), "--public-dataset", cfg.public_dataset,
           "--random-seed", str(cfg.random_seed), "--output-artifact-dir", str(artifact_dir)]
    if cfg.benchmark_duration_s:
        cmd += ["--benchmark-duration", str(cfg.benchmark_duration_s)]
    if cfg.max_context_length:
        cmd += ["--max-context-length", str(cfg.max_context_length)]
    if cfg.unsafe_override:
        cmd.append("--unsafe-override")
    cmd += cfg.extra_args
    env = {**os.environ, "PATH": path_env}

    monitor_spec = (
        baseline_spec_acc_per_draft is not None
        and baseline_spec_acc_per_draft > 0
        and cfg.spec_collapse_min_ratio is not None
        and cfg.spec_collapse_min_ratio > 0
    )
    reader = metrics_reader if metrics_reader is not None else (_default_spec_counter_reader(url) if monitor_spec else None)
    init_snap = reader() if (monitor_spec and reader is not None) else {}
    init_acc = init_snap.get("accepted_tokens")
    init_drafts = init_snap.get("num_drafts")

    t0 = time.monotonic()
    with open(artifact_dir / "agentx.log", "w") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            poll_s = max(0.05, float(cfg.spec_collapse_poll_s)) if monitor_spec else 1.0
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                elapsed = time.monotonic() - t0
                if timeout_s and elapsed >= float(timeout_s):
                    shutdown_process_group(proc, timeout=10.0)
                    raise subprocess.TimeoutExpired(cmd, timeout_s)
                if monitor_spec and reader is not None and elapsed >= float(cfg.spec_collapse_window_s):
                    cur = reader()
                    c_acc = cur.get("accepted_tokens")
                    c_drafts = cur.get("num_drafts")
                    if c_acc is not None and c_drafts is not None:
                        if init_acc is None or init_drafts is None:
                            init_acc, init_drafts = c_acc, c_drafts
                        else:
                            d_acc = max(0.0, c_acc - init_acc)
                            d_drafts = c_drafts - init_drafts
                            if d_drafts >= max(1, int(cfg.spec_collapse_min_drafts)):
                                obs_rate = d_acc / d_drafts
                                collapsed, detail = detect_spec_acceptance_collapse(
                                    obs_rate,
                                    baseline_spec_acc_per_draft,
                                    min_ratio=cfg.spec_collapse_min_ratio,
                                )
                                if collapsed:
                                    shutdown_process_group(proc, timeout=10.0)
                                    snap = {
                                        "baseline_acc_per_draft": round(float(baseline_spec_acc_per_draft), 4),
                                        "observed_acc_per_draft": round(obs_rate, 4),
                                        "ratio": round(obs_rate / float(baseline_spec_acc_per_draft), 4),
                                        "window_drafts": int(round(d_drafts)),
                                        "window_accepted_tokens": int(round(d_acc)),
                                        "elapsed_s": round(elapsed, 2),
                                    }
                                    raise SpecAcceptanceCollapsed(detail or "Spec-decode acceptance collapsed", snap)
                wait_slice = min(poll_s, max(0.05, float(timeout_s) - elapsed)) if timeout_s else poll_s
                try:
                    proc.wait(timeout=wait_slice)
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except BaseException:
            if proc.poll() is None:
                shutdown_process_group(proc, timeout=10.0)
            raise

    hits = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if not hits:
        raise RuntimeError(f"no profile_export_aiperf.json under {artifact_dir}")
    res = parse_agentx_export(hits[-1], num_gpus)
    if monitor_spec and res.spec_decode_acc_per_draft is not None:
        collapsed, detail = detect_spec_acceptance_collapse(
            res.spec_decode_acc_per_draft,
            baseline_spec_acc_per_draft,
            min_ratio=cfg.spec_collapse_min_ratio,
        )
        res.spec_acceptance_collapsed = collapsed
        res.spec_collapse_detail = detail
    return res
