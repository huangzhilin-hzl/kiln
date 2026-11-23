"""Request-level latency measurements for traffic replay."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean

import httpx

from kiln.bench.traffic import Request

from .sglang_ext import extract_output_ids


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


def read_completion(client: httpx.Client, url: str, payload: dict, *, sglang: bool) -> tuple[dict, int, float | None]:
    """Read a single completion and its mean client-observed inter-token latency in ms."""
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", ""):
            response.read()
            result = response.json()
            if sglang:
                count = (result.get("meta_info") or {}).get("completion_tokens")
                if count is None:
                    ids = extract_output_ids(result)
                    count = len(ids) if ids is not None else payload["sampling_params"]["max_new_tokens"]
            else:
                count = (result.get("usage") or {}).get("completion_tokens")
                if count is None:
                    ids = (result.get("choices") or [{}])[0].get("token_ids")
                    count = len(ids) if ids is not None else payload["max_tokens"]
            # A buffered reply contains no token arrival timestamps.
            return result, int(count), None

        first = last = None
        tokens = 0
        usage_tokens = None
        latest = {}
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            arrived = time.perf_counter()
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Inference stream failed: {event['error']}")
            latest = event
            if sglang:
                # Native /generate reports cumulative counts, even for incremental output_ids.
                count = (event.get("meta_info") or {}).get("completion_tokens")
                if count is None:
                    continue
            else:
                # With return_token_ids, vLLM sends delta IDs and a final usage-only event.
                count = tokens + sum(len(c.get("token_ids") or []) for c in event.get("choices", []))
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
            if count > tokens:
                tokens = count
                if first is None:
                    first = arrived
                last = arrived
        else:
            raise RuntimeError("Inference stream ended before [DONE]")

        if usage_tokens is not None and usage_tokens != tokens:
            # A server can supply accurate final usage without streaming token IDs.
            return latest, int(usage_tokens), None
        if tokens == 0:
            raise RuntimeError("Inference stream reported no output tokens")
        itl_ms = None
        if tokens > 1 and first is not None and last is not None and last > first:
            itl_ms = 1000.0 * (last - first) / (tokens - 1)
        return latest, tokens, itl_ms
