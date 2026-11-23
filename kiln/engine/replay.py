"""Request-level latency measurements for traffic replay."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean

from kiln.bench.traffic import Request


def mean_ttft(latencies: Iterable[float | None]) -> float:
    """A missing request timestamp makes the batch's mean TTFT unavailable."""
    samples = list(latencies)
    if not samples or any(t is None or not math.isfinite(t) or t <= 0 for t in samples):
        return 0.0
    return fmean(samples)


def time_http_prefill(
    pool: ThreadPoolExecutor, requests: list[Request], generate: Callable[[list[int], int], int]
) -> tuple[float, float]:
    """Time the batch for throughput and each one-token response for mean TTFT."""
    def time_one(req: Request) -> float:
        start = time.perf_counter()
        generate(req.prompt_ids, 1)
        return time.perf_counter() - start

    start = time.perf_counter()
    latencies = list(pool.map(time_one, requests))
    elapsed = max(time.perf_counter() - start, 1e-9)
    return elapsed, mean_ttft(latencies)
