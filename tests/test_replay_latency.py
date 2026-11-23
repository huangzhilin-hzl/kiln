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
