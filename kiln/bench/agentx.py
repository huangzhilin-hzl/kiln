"""AgentX (SemiAnalysis InferenceX) workload for kiln.

Two ways to use the same public traces (HF: semianalysisai/cc-traces-weka-062126, 393 agentic-coding
sessions with subagents; the dataset TKCC used, seed 42):

1. Slow tier, exact: run the real `agentx profile` harness against a `vllm serve` endpoint.
   Methodology is unchanged: pinned harness, public dataset, seed, timing, scoring.
   See `run_agentx_profile` and kiln/engine/vllm_server.py.

2. Fast tier, approximate: `export_offline` turns traces into kiln request JSONL for offline
   replay. Each prompt is built by AgentX's own content generator
   (aiperf CodingContentGenerator.generate(mean=in, hash_ids=..., block_size)), so it has the
   same lengths, the same hash-block prefix sharing and the same synthetic coding text.
   It is then tokenized with the target model's tokenizer. Not reproduced here: inter-turn
   timing, subagent DAG concurrency and the chat template.

Requires `pip install 'agentx-harness==1.0.12'` (the version pinned in the TKCC paper).
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
from typing import Any

DEFAULT_DATASET = "semianalysis_cc_traces_weka_062126"
DEFAULT_HF_REPO = "semianalysisai/cc-traces-weka-062126"


# --------------------------------------------------------------------------- offline export
def _iter_requests(trace) -> list[Any]:
    """Main-agent requests followed by subagent requests (in trace order)."""
    out = []
    for r in trace.requests:
        kind = getattr(r, "type", None)
        if kind in ("n", "s"):
            out.append(r)
        elif kind == "subagent":
            out.extend(sr for sr in r.requests if getattr(sr, "type", None) in ("n", "s"))
    return out


def export_offline(out: str | Path, tokenizer: str, hf_repo: str = DEFAULT_HF_REPO,
                   num_traces: int = 32, turns_per_trace: int = 4, seed: int = 42,
                   max_context: int | None = None, trust_remote_code: bool = True) -> dict:
    """Write kiln JSONL: {"prompt_ids", "output_len", "trace_id", "turn", "t"} per request.

    Takes `turns_per_trace` consecutive turns from a seeded random offset in each of
    `num_traces` sampled traces. Consecutive turns share prefixes, which exercises the
    prefix cache the way AgentX does.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from aiperf.common import random_generator as aiperf_rng
    from aiperf.common.tokenizer import Tokenizer as AiperfTokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.coding_content import CodingContentGenerator
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    aiperf_rng.init(seed)
    gen = CodingContentGenerator(config=PromptConfig(),
                                 tokenizer=AiperfTokenizer.from_pretrained(
                                     tokenizer, trust_remote_code=trust_remote_code))
    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)

    rows = load_dataset(hf_repo, split="train")
    traces = [WekaTrace.model_validate(r) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(traces)

    stats = {"traces": 0, "requests": 0, "skipped_too_long": 0, "isl": [], "osl": []}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for tr in traces:
            if stats["traces"] >= num_traces:
                break
            reqs = [r for r in _iter_requests(tr) if r.hash_ids]
            if max_context:
                reqs = [r for r in reqs if r.input_length + r.output_length <= max_context]
            if not reqs:
                continue
            # AgentX scope contract: hash ids are trace-local.
            gen._cache.clear()
            gen._hash_id_corpus_rng.set_trace_id(tr.id)
            start = rng.randrange(max(1, len(reqs) - turns_per_trace + 1))
            for i, r in enumerate(reqs[start:start + turns_per_trace]):
                text = gen.generate(mean=r.input_length, hash_ids=list(r.hash_ids),
                                    block_size=tr.block_size)
                ids = tok.encode(text, add_special_tokens=False)
                f.write(json.dumps({"prompt_ids": ids, "output_len": int(max(1, r.output_length)),
                                    "trace_id": tr.id, "turn": start + i, "t": r.t}) + "\n")
                stats["requests"] += 1
                stats["isl"].append(len(ids))
                stats["osl"].append(r.output_length)
            stats["traces"] += 1

    def _pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    summary = {"traces": stats["traces"], "requests": stats["requests"],
               "isl_p50": _pct(stats["isl"], .5), "isl_p90": _pct(stats["isl"], .9),
               "osl_p50": _pct(stats["osl"], .5), "osl_p90": _pct(stats["osl"], .9)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- real harness
@dataclass
class AgentXConfig:
    concurrency: int = 8
    public_dataset: str = DEFAULT_DATASET
    random_seed: int = 42
    benchmark_duration_s: int | None = 600      # None = full corpus replay (slowest, exact)
    max_context_length: int | None = None
    extra_args: list[str] = field(default_factory=list)
    binary: str = "agentx"


@dataclass
class AgentXResult:
    total_tput_per_gpu: float          # InferenceX: (input incl. cached + output) tok/s / GPUs
    output_tput: float
    itl_p50_ms: float | None
    itl_p90_ms: float | None
    submission_valid: bool | None
    raw_path: str

    @property
    def interactivity_p90(self) -> float | None:   # InferenceX: 1 / P90 ITL (tok/s/user)
        return 1000.0 / self.itl_p90_ms if self.itl_p90_ms else None


def _find_metric(d: Any, tag: str) -> dict | None:
    """Locate a metric block by tag anywhere in AIPerf's export JSON."""
    if isinstance(d, dict):
        if tag in d and isinstance(d[tag], dict):
            return d[tag]
        if d.get("tag") == tag:
            return d
        for v in d.values():
            if (hit := _find_metric(v, tag)) is not None:
                return hit
    elif isinstance(d, list):
        for v in d:
            if (hit := _find_metric(v, tag)) is not None:
                return hit
    return None


def _find_key(d: Any, key: str) -> Any:
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            if (hit := _find_key(v, key)) is not None:
                return hit
    elif isinstance(d, list):
        for v in d:
            if (hit := _find_key(v, key)) is not None:
                return hit
    return None


def parse_agentx_export(path: str | Path, num_gpus: int) -> AgentXResult:
    data = json.loads(Path(path).read_text())
    total = _find_metric(data, "total_token_throughput") or {}
    out = _find_metric(data, "output_token_throughput") or {}
    itl = _find_metric(data, "inter_token_latency") or {}
    tot = total.get("avg") or total.get("value") or 0.0
    return AgentXResult(
        total_tput_per_gpu=float(tot) / max(num_gpus, 1),
        output_tput=float(out.get("avg") or out.get("value") or 0.0),
        itl_p50_ms=itl.get("p50"), itl_p90_ms=itl.get("p90"),
        submission_valid=_find_key(data, "submission_valid"),
        raw_path=str(path),
    )


def run_agentx_profile(url: str, model: str, tokenizer: str, cfg: AgentXConfig,
                       artifact_dir: str | Path, num_gpus: int, timeout_s: float) -> AgentXResult:
    venv_bin = str(Path(sys.executable).resolve().parent)
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
    cmd += cfg.extra_args
    env = {**os.environ, "PATH": path_env}
    with open(artifact_dir / "agentx.log", "w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout_s)
    hits = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if not hits:
        raise RuntimeError(f"no profile_export_aiperf.json under {artifact_dir}")
    return parse_agentx_export(hits[-1], num_gpus)
