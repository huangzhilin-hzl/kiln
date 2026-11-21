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

import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

    def __post_init__(self) -> None:
        if self.benchmark_duration_s is not None:
            if type(self.benchmark_duration_s) is not int or self.benchmark_duration_s < 1:
                raise ValueError("AgentX benchmark_duration_s must be a positive integer or null")
            if not self.unsafe_override and self.benchmark_duration_s < 900:
                raise ValueError(
                    "AgentX benchmark_duration_s must be at least 900 seconds or null "
                    "(or set unsafe_override: true)"
                )


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
    ):
        val = getattr(last, attr, None)
        if val is not None:
            extra[attr] = val
    return extra


def run_agentx_profile(url: str, model: str, tokenizer: str, cfg: AgentXConfig,
                       artifact_dir: str | Path, num_gpus: int, timeout_s: float) -> AgentXResult:
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
    with open(artifact_dir / "agentx.log", "w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout_s)
    hits = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if not hits:
        raise RuntimeError(f"no profile_export_aiperf.json under {artifact_dir}")
    return parse_agentx_export(hits[-1], num_gpus)
