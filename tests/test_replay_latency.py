from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, current_thread, local, main_thread
from types import SimpleNamespace

import pytest

from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig
from kiln.engine import replay
from kiln.engine.sglang_offline import SGLangOfflineEngine


def test_http_prefill_tracks_each_concurrent_request(monkeypatch):
    batch_clock = iter([0.0, 0.1])
    clock = local()
    started = Barrier(2)

    def now():
        if current_thread() is main_thread():
            return next(batch_clock)
        return getattr(clock, "now", 0.0)

    def generate(prompt, max_tokens):
        assert max_tokens == 1
        started.wait(timeout=3)
        clock.now = prompt[0] / 1000
        return 1

    monkeypatch.setattr(replay, "time", SimpleNamespace(perf_counter=now))
    with ThreadPoolExecutor(max_workers=2) as pool:
        elapsed, mean_ttft = replay.time_http_prefill(
            pool, [Request([80], 5), Request([100], 5)], generate
        )
    assert elapsed == pytest.approx(0.1)
    assert mean_ttft == pytest.approx(0.09)


def test_offline_prefill_uses_backend_first_token_timings():
    engine = SGLangOfflineEngine(EngineConfig(), TierConfig(name="p", phase="prefill"))
    engine.engine = SimpleNamespace(generate=lambda **kwargs: [
        {"meta_info": {"e2e_latency": 0.8, "ttft_latency": 0.5}},
        {"meta_info": {"e2e_latency": 1.0}},
    ])
    result = engine.measure([Request([1], 2), Request([2], 2)], repeats=2, trace=False)
    assert result.ttft_s == pytest.approx(0.75)


def test_missing_timestamp_does_not_produce_partial_batch_ttft():
    assert replay.mean_ttft([1.0, None]) == 0.0


def test_mock_prefill_reports_ttft_for_additional_timed_repeats():
    from kiln.engine.mock import MockEngine

    engine = MockEngine(TierConfig(name="p", phase="prefill"))
    result = engine.measure_more([Request([1, 2], 2)], repeats=2, trace=False)
    assert result.ttft_s > 0
    assert len(result.runs_prefill_tps) == 2


def test_ttft_screening_weights_all_prefill_repeats():
    from kiln.engine.base import MeasureResult
    from kiln.server.jobs import TierWorker

    first = MeasureResult(0, 30, 0.6, runs_prefill_tps=[30])
    more = MeasureResult(0, 10, 0.9, runs_prefill_tps=[10, 10])
    merged = TierWorker._merge_stage_results(first, more, [])
    assert merged.ttft_s == pytest.approx(0.8)


@pytest.mark.parametrize("sglang", [True, False])
@pytest.mark.parametrize("n_tokens", [1, 3])
def test_itl_excludes_first_token_wait_and_final_metadata(monkeypatch, sglang, n_tokens):
    import json
    import httpx

    # First token at 10 s, two more at 10.04 s, trailing usage/stats at 12 s.
    clock = iter((10.0, 10.04, 12.0))
    monkeypatch.setattr(replay, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    if sglang:
        counts = [1, 3, 3] if n_tokens == 3 else [1, 1]
        events = [{"meta_info": {"completion_tokens": count}} for count in counts]
    else:
        events = [{"choices": [{"token_ids": [1]}]}]
        if n_tokens == 3:
            events.append({"choices": [{"token_ids": [2, 3]}]})
        events.append({"choices": [], "usage": {"completion_tokens": n_tokens}})
    chunks = [f"data: {json.dumps(event)}\n\n" for event in events]
    response = httpx.Response(200, text="".join(chunks) + "data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    with httpx.Client(transport=httpx.MockTransport(lambda req: response)) as client:
        _, tokens, itl = replay.read_completion(client, "http://engine", {}, sglang=sglang)
    assert tokens == n_tokens
    if n_tokens == 1:
        assert itl is None
    else:
        assert itl == pytest.approx(20.0)


def test_merge_itl_percentiles_over_both_screening_stages():
    from kiln.engine.base import MeasureResult
    from kiln.server.jobs import TierWorker

    first = MeasureResult(100, 0, 0, itl_samples_ms=[10, 20])
    more = MeasureResult(100, 0, 0, itl_samples_ms=[30, 40])
    merged = TierWorker._merge_stage_results(first, more, [])
    assert merged.extra["itl_p50_ms"] == pytest.approx(25)
    assert merged.extra["itl_p90_ms"] == pytest.approx(37)
