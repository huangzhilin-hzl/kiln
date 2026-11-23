import json
import math

import httpx
import pytest

from kiln.bench.correctness import collect_probe_logprobs, evaluate_correctness
from kiln.config import CorrectnessConfig, EngineConfig, TierConfig
from kiln.engine.sglang_ext import extract_output_logprobs
from kiln.engine.vllm_server import VLLMServerEngine


@pytest.mark.parametrize("candidate", [
    None,
    [[]],
    [[{1: -0.2}]],
    [[{1: -0.2}, {99: -0.1}]],
    [[{1: -0.2}, {2: math.nan, 99: 0.0}]],
])
def test_candidate_must_keep_baseline_logprob_coverage(candidate):
    tokens = [[1, 2]]
    baseline = [[{1: -0.2}, {2: -0.1}]]
    result = evaluate_correctness(tokens, tokens, CorrectnessConfig(), baseline, candidate)
    assert not result.ok and result.token_match == 1.0


def test_token_only_backend_and_tolerated_different_greedy_token():
    tokens = [[1, 2]]
    cfg = CorrectnessConfig(min_token_match=0.5)
    assert evaluate_correctness(tokens, tokens, cfg, None, None).ok
    assert evaluate_correctness(tokens, [[1, 3]], cfg, [[{1: -0.2}, {2: -0.1}]],
                                [[{1: -0.2}, {3: -0.1}]]).ok


def test_invalid_selected_sglang_logprob_survives_top_alternative():
    seq = extract_output_logprobs({"meta_info": {
        "output_token_logprobs": [["nan", 2, "chosen"]],
        "output_top_logprobs": [[[-0.1, 2, "chosen"], [-0.4, 8, "other"]]],
    }})
    assert seq is not None and math.isnan(seq[0][2])
    assert not evaluate_correctness([[2]], [[2]], CorrectnessConfig(), [[{2: -0.1}]], [seq]).ok


def test_partial_vllm_probe_cannot_silently_disable_the_gate():
    def handler(request):
        prompt = json.loads(request.content)["prompt"]
        choice = {"token_ids": [4]}
        if prompt == [1]:
            choice["logprobs"] = {"token_logprobs": [-0.25]}
        return httpx.Response(200, json={"choices": [choice]})

    engine = VLLMServerEngine(EngineConfig(), TierConfig(name="server"))
    engine.http.close()
    with httpx.Client(transport=httpx.MockTransport(handler)) as engine.http:
        ids, probs = engine.greedy_probe([[1], [2]], 1)
        assert probs == [[{4: -0.25}], []]
        assert not evaluate_correctness(ids, ids, CorrectnessConfig(), probs, probs).ok
        assert engine.greedy_probe([[2]], 1)[1] is None
    assert collect_probe_logprobs([None, None]) is None
