import json

import httpx

from kiln.config import EngineConfig, TierConfig
from kiln.engine.base import Request
from kiln.engine.sglang_server import SGLangServerEngine


def test_server_prefill_replay_returns_baseline_metric():
    def handle(request):
        if request.url.path == "/flush_cache":
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/generate"
        assert json.loads(request.content)["sampling_params"]["max_new_tokens"] == 1
        return httpx.Response(200, json={"output_ids": [7], "meta_info": {"completion_tokens": 1}})

    engine = SGLangServerEngine(EngineConfig(), TierConfig(name="prefill", phase="prefill"))
    engine.http.close()
    with httpx.Client(transport=httpx.MockTransport(handle)) as engine.http:
        result = engine.measure([Request(prompt_ids=[2, 3], output_len=8)], repeats=2, trace=False)
    assert len(result.runs_prefill_tps) == 2
    assert result.prefill_tps > 0
    assert result.extra["bucket_tps"]["prefill_large"] == result.prefill_tps
    assert result.decode_tps == 0
