from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig
from kiln.engine.sglang_offline import SGLangOfflineEngine
from kiln.engine.sglang_server import SGLangServerEngine


@pytest.mark.parametrize("backend", ["offline", "http"])
def test_prefill_profiler_replays_cold_prompts(tmp_path, backend):
    prompt = [11, 12, 13]
    cached = {tuple(prompt)}
    cache_hits = []

    def flush():
        cached.clear()
        return True

    def generate(prompt_ids):
        key = tuple(prompt_ids)
        cache_hits.append(key in cached)
        cached.add(key)

    tier = TierConfig(name="p", phase="prefill")
    if backend == "offline":
        engine = SGLangOfflineEngine(EngineConfig(), tier)
        engine._trace_dir = str(tmp_path)
        engine.engine = SimpleNamespace(
            flush_cache=flush,
            generate=lambda input_ids, **kw: generate(input_ids[0]),
            start_profile=lambda **kw: None,
            stop_profile=lambda: None,
        )
        engine._kernel_report([prompt], max_new_tokens=1)
    else:
        engine = SGLangServerEngine(EngineConfig(), tier)
        engine._trace_dir = str(tmp_path)
        engine.http.close()

        def handle(request):
            if request.url.path == "/flush_cache":
                flush()
            if request.url.path == "/generate":
                generate(prompt)
                return httpx.Response(200, json={"output_ids": [1]})
            return httpx.Response(200, json={"success": True})

        with httpx.Client(transport=httpx.MockTransport(handle)) as engine.http, ThreadPoolExecutor(1) as pool:
            engine._kernel_report(pool, [Request(prompt, 8)], max_new_tokens=1)
    assert cache_hits == [False]
