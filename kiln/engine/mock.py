"""CPU-only mock engine so the whole loop runs on a laptop.

A candidate file may define `SPEEDUP = 1.07` (pretend speedup), `PREFILL_SPEEDUP`,
`BUCKET_SPEEDUPS = {"decode_small": 1.2, ...}`, `MOCK_TRACE_EVENTS`, and `CORRUPT = True`.
"""
from __future__ import annotations

import random
import runpy
import time

from kiln.bench.traffic import Request
from kiln.config import TierConfig

from .base import Engine, MeasureResult
from .vllm_profiler import analyze_trace_events, format_trace_report

BASE_DECODE_TPS = 1000.0
BASE_PREFILL_TPS = 20_000.0


class MockEngine(Engine):
    def __init__(self, tier: TierConfig):
        self.tier = tier
        self.speedup = 1.0
        self.prefill_speedup = 1.0
        self.bucket_speedups: dict[str, float] = {}
        self.mock_trace_events: list[dict] | None = None
        self.corrupt = False
        self.logit_diff = 0.0

    def start(self) -> None:
        time.sleep(0.05)

    def hot_reload(self, candidate_path: str) -> None:
        ns = runpy.run_path(candidate_path)
        if ns.get("CRASH"):
            raise RuntimeError("illegal memory access (mock)")
        self.speedup = float(ns.get("SPEEDUP", 1.0))
        self.prefill_speedup = float(ns.get("PREFILL_SPEEDUP", self.speedup))
        raw_buckets = ns.get("BUCKET_SPEEDUPS")
        self.bucket_speedups = (
            {str(k): float(v) for k, v in raw_buckets.items()}
            if isinstance(raw_buckets, dict)
            else {}
        )
        raw_trace = ns.get("MOCK_TRACE_EVENTS")
        self.mock_trace_events = list(raw_trace) if isinstance(raw_trace, list) else None
        self.corrupt = bool(ns.get("CORRUPT", False))
        self.logit_diff = float(ns.get("LOGIT_DIFF", 0.0))

    def teardown_candidate(self) -> None:
        self.speedup = 1.0
        self.prefill_speedup = 1.0
        self.bucket_speedups = {}
        self.mock_trace_events = None
        self.corrupt = False
        self.logit_diff = 0.0

    def _trace_events(self) -> list[dict]:
        if self.mock_trace_events is not None:
            return self.mock_trace_events
        inv = 1.0 / max(self.speedup, 0.05)
        return [
            {"cat": "kernel", "name": "mock_gemm", "ts": 0.0, "dur": 410.0 * inv, "args": {"stream": 7}},
            {"cat": "kernel", "name": "mock_attn", "ts": 430.0 * inv, "dur": 330.0 * inv, "args": {"stream": 7}},
            {"cat": "kernel", "name": "mock_allreduce", "ts": 760.0 * inv, "dur": 120.0 * inv, "args": {"stream": 8}},
        ]

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        noise = 0.002 if self.tier.name == "slow" else 0.005
        runs = [BASE_DECODE_TPS * self.speedup * (1 + random.gauss(0, noise)) for _ in range(repeats)]
        decode_tps = sum(runs) / len(runs)
        prefill_tps = BASE_PREFILL_TPS * self.bucket_speedups.get("prefill_large", self.prefill_speedup)
        bucket_tps = {
            "decode_small": BASE_DECODE_TPS * self.bucket_speedups.get("decode_small", self.speedup),
            "decode_mid": BASE_DECODE_TPS * self.bucket_speedups.get("decode_mid", self.speedup),
            "prefill_large": prefill_tps,
        }
        kernel_report = None
        trace_analysis = None
        if trace:
            ana = analyze_trace_events(self._trace_events(), trace_path="mock://trace")
            if ana is not None:
                trace_analysis = ana.to_dict()
                kernel_report = format_trace_report(ana)
        return MeasureResult(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            ttft_s=0.05 / max(self.prefill_speedup, 1e-6),
            runs_decode_tps=runs,
            kernel_report=kernel_report,
            trace_analysis=trace_analysis,
            extra={"bucket_tps": bucket_tps},
        )

    def capture_snapshots(self, snapshot_dir: str, m_buckets: list[int]) -> dict | None:
        from kiln.micro.snapshots import write_mock_snapshots

        return write_mock_snapshots(snapshot_dir, m_buckets)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        out = []
        for p in prompts:
            seq = [(sum(p) + i) % 32000 for i in range(max_new_tokens)]
            if self.corrupt:
                seq = [(t + 7) % 32000 if i % 3 == 0 else t for i, t in enumerate(seq)]
            out.append(seq)
        return out

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        tokens = self.greedy_outputs(prompts, max_new_tokens)
        logprobs = [[{tok: -0.1 - self.logit_diff} for tok in seq] for seq in tokens]
        return tokens, logprobs

    def shutdown(self) -> None:
        pass
