"""Shared torch.profiler helpers for persistent vLLM engines (`vllm_offline` and `vllm_server`)."""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


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
    files = sorted(p.glob("*.pt.trace.json*"), key=lambda f: f.stat().st_mtime)
    if not files:
        return f"trace written to {trace_dir}"
    latest = files[-1]
    try:
        opener = gzip.open if latest.name.endswith(".gz") else open
        with opener(latest, "rt", encoding="utf-8") as f:
            data = json.load(f)
        dur_by_kernel: dict[str, float] = defaultdict(float)
        for ev in data.get("traceEvents", ()):
            if ev.get("cat") == "kernel" and "dur" in ev:
                dur_by_kernel[str(ev.get("name", "unknown"))] += float(ev["dur"])
        total = sum(dur_by_kernel.values())
        if not total:
            return f"trace written to {latest}"
        top = sorted(dur_by_kernel.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        lines = [f"{100.0 * us / total:5.1f}%  {us / 1000.0:8.2f} ms  {name[:90]}" for name, us in top]
        return f"trace: {latest}\n" + "\n".join(lines)
    except Exception:
        return f"trace written to {latest}"
