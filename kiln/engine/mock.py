"""CPU-only mock engine so the whole loop runs on a laptop.

A candidate file may define `SPEEDUP = 1.07` (pretend speedup) and
`CORRUPT = True` (pretend wrong outputs). Real candidates are ignored here.
"""
from __future__ import annotations

import random
import runpy
import time

from kiln.bench.traffic import Request
from kiln.config import TierConfig

from .base import Engine, MeasureResult

BASE_DECODE_TPS = 1000.0


class MockEngine(Engine):
    def __init__(self, tier: TierConfig):
        self.tier = tier
        self.speedup = 1.0
        self.corrupt = False

    def start(self) -> None:
        time.sleep(0.05)

    def hot_reload(self, candidate_path: str) -> None:
        ns = runpy.run_path(candidate_path)
        if ns.get("CRASH"):
            raise RuntimeError("illegal memory access (mock)")
        self.speedup = float(ns.get("SPEEDUP", 1.0))
        self.corrupt = bool(ns.get("CORRUPT", False))

    def teardown_candidate(self) -> None:
        self.speedup, self.corrupt = 1.0, False

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        noise = 0.002 if self.tier.name == "slow" else 0.005
        runs = [BASE_DECODE_TPS * self.speedup * (1 + random.gauss(0, noise)) for _ in range(repeats)]
        return MeasureResult(
            decode_tps=sum(runs) / len(runs),
            prefill_tps=20_000.0 * self.speedup,
            ttft_s=0.05 / self.speedup,
            runs_decode_tps=runs,
            kernel_report="mock_gemm 41%\nmock_attn 33%\nmock_allreduce 12%" if trace else None,
        )

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        out = []
        for p in prompts:
            seq = [(sum(p) + i) % 32000 for i in range(max_new_tokens)]
            if self.corrupt:
                seq = [(t + 7) % 32000 if i % 3 == 0 else t for i, t in enumerate(seq)]
            out.append(seq)
        return out

    def shutdown(self) -> None:
        pass
