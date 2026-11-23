from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from kiln.bench.agentx import AgentXResult
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, PDClusterConfig, PDNodeConfig, RunConfig, TierConfig
from kiln.engine import make_engine
from kiln.engine.sglang_ext import decode_rpc
from kiln.engine.sglang_pd import (
    SGLangPDServerEngine,
    build_router_cmd,
    validate_sglang_pd_args,
)


def test_pd_config_yaml_and_pd_node_cli_dry_run(capsys):
    import kiln.cli as cli_mod

    cfg_path = "configs/dsv41_flash_h20_sglang_pd.yaml"
    run = RunConfig.load(cfg_path)
    assert [t.name for t in run.tiers] == ["fast_prefill", "fast_decode", "slow"]
    slow = run.tiers[2]
    assert slow.engine_kind == "sglang_pd_server"
    assert slow.pd is not None and slow.pd.transfer_backend == "mooncake"

    eng = make_engine(run.engine, slow)
    assert isinstance(eng, SGLangPDServerEngine)
    assert eng.total_gpus == 16  # 1P (8 GPUs) + 1D (8 GPUs)
    assert eng.prefill_urls == ["http://kiln-prefill:31401"]
    assert eng.decode_urls == ["http://kiln-decode:31402"]

    cli_mod.main(["pd-node", "-c", cfg_path, "--role", "prefill", "--host", "0.0.0.0", "--dry-run"])
    p_spec = json.loads(capsys.readouterr().out)
    assert p_spec["role"] == "prefill"
    assert "--disaggregation-mode" in p_spec["cmd"]
    idx = p_spec["cmd"].index("--disaggregation-mode")
    assert p_spec["cmd"][idx + 1] == "prefill"
    assert p_spec["cmd"][p_spec["cmd"].index("--port") + 1] == "31401"
    assert p_spec["cmd"][p_spec["cmd"].index("--disaggregation-bootstrap-port") + 1] == "8998"
    assert p_spec["env_overrides"]["KILN_SGLANG_HOOK"] == "1"
    assert p_spec["env_overrides"]["KILN_PD_ROLE"] == "prefill"

    cli_mod.main(["pd-node", "-c", cfg_path, "--role", "decode", "--host", "0.0.0.0", "--dry-run"])
    d_spec = json.loads(capsys.readouterr().out)
    idx_d = d_spec["cmd"].index("--disaggregation-mode")
    assert d_spec["cmd"][idx_d + 1] == "decode"
    assert d_spec["cmd"][d_spec["cmd"].index("--port") + 1] == "31402"
    assert d_spec["env_overrides"]["KILN_PD_ROLE"] == "decode"


def test_deepseek_v41_pd_validation_and_router_cmd():
    base_p = {"tp_size": 8, "speculative_algorithm": "DSPARK", "speculative_dspark_block_size": 5}
    base_d = {"tp_size": 8, "speculative_algorithm": "DSPARK", "speculative_dspark_block_size": 5}
    validate_sglang_pd_args("/models/DeepSeek-V4.1-Flash", "mooncake", [base_p], [base_d])
    validate_sglang_pd_args("/models/DeepSeek-V4.1-Flash", "mooncake_tcp", [base_p], [base_d])

    # Non-mooncake backend with DSpark PD must fail
    with pytest.raises(ValueError, match="mooncake"):
        validate_sglang_pd_args("/models/DeepSeek-V4.1-Flash", "nixl", [base_p], [base_d])

    # Asymmetric DSpark enablement across P and D must fail
    with pytest.raises(ValueError, match="both prefill and decode"):
        validate_sglang_pd_args("/models/DeepSeek-V4.1-Flash", "mooncake", [{"tp_size": 8}], [base_d])

    # Mismatched block size or TP size must fail
    with pytest.raises(ValueError, match="block_size"):
        validate_sglang_pd_args(
            "/models/DeepSeek-V4.1-Flash",
            "mooncake",
            [base_p],
            [{**base_d, "speculative_dspark_block_size": 3}],
        )

    # Decode radix cache + speculative decoding must fail
    with pytest.raises(ValueError, match="disaggregation_decode_enable_radix_cache"):
        validate_sglang_pd_args(
            "/models/DeepSeek-V4.1-Flash",
            "mooncake",
            [base_p],
            [{**base_d, "disaggregation_decode_enable_radix_cache": True}],
        )

    # Encoder SWA bounded replay or trtllm dsv4 backend must fail
    with pytest.raises(ValueError, match="enable-encoder-swa-bounded-replay"):
        validate_sglang_pd_args(
            "/models/DeepSeek-V4.1-Flash",
            "mooncake",
            [{**base_p, "enable_encoder_swa_bounded_replay": True}],
            [base_d],
        )

    pd = PDClusterConfig(
        transfer_backend="mooncake",
        prefill=[PDNodeConfig(url="http://10.0.0.1:30000", bootstrap_port=8998)],
        decode=[PDNodeConfig(url="http://10.0.0.2:30000")],
        router_policy="cache_aware",
        router_command="sglang-router",
    )
    rcmd = build_router_cmd(pd, ["http://10.0.0.1:30000"], ["http://10.0.0.2:30000"], "0.0.0.0", 30080)
    assert rcmd == [
        "sglang-router",
        "--pd-disaggregation",
        "--host", "0.0.0.0",
        "--port", "30080",
        "--prefill", "http://10.0.0.1:30000", "8998",
        "--decode", "http://10.0.0.2:30000",
        "--policy", "cache_aware",
    ]


def test_sglang_pd_server_engine_broadcast_rollback_and_agentx(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock = threading.Lock()
    rpcs_by_host: dict[str, list[dict]] = {}
    flushes_by_host: dict[str, int] = {}
    router_generates: list[dict] = []
    fail_host: dict[str, bool] = {"d2": False}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/update_weights_from_disk":
            assert host in ("p1", "p2", "d1", "d2")  # never sent to router!
            rpc = decode_rpc(json.loads(request.content.decode())["model_path"])
            assert rpc is not None
            with lock:
                rpcs_by_host.setdefault(host, []).append(rpc)
            if rpc["method"] == "kiln_hot_reload_source" and fail_host.get(host):
                return httpx.Response(500, json={"success": False, "message": "simulated compile crash"})
            res = {"engine": "sglang", "pd_role": "prefill" if host.startswith("p") else "decode"}
            if rpc["method"] != "kiln_info":
                res = 0.15
            return httpx.Response(200, json={"success": True, "message": json.dumps({"ok": True, "result": res})})
        if path == "/flush_cache":
            assert host in ("p1", "p2", "d1", "d2")  # never sent to router!
            with lock:
                flushes_by_host[host] = flushes_by_host.get(host, 0) + 1
            return httpx.Response(200, json={"success": True})
        if path == "/generate":
            assert host == "router"  # inference traffic ALWAYS goes through router!
            body = json.loads(request.content.decode())
            with lock:
                router_generates.append(body)
            mx = int(body["sampling_params"]["max_new_tokens"])
            toks = [50 + i for i in range(mx)]
            return httpx.Response(
                200,
                json={
                    "output_ids": toks,
                    "meta_info": {
                        "completion_tokens": mx,
                        "output_token_logprobs": [[-0.05, t, "tok"] for t in toks],
                    },
                },
            )
        return httpx.Response(404)

    cfg = EngineConfig(kind="sglang_pd_server", model="dummy/dsv41", tensor_parallel_size=8)
    pd_cfg = PDClusterConfig(
        transfer_backend="mooncake",
        router_url="http://router:30080",
        spawn_router=False,
        prefill=[
            PDNodeConfig(url="http://p1:30000", num_gpus=8),
            PDNodeConfig(url="http://p2:30000", num_gpus=8),
        ],
        decode=[
            PDNodeConfig(url="http://d1:30000", num_gpus=8),
            PDNodeConfig(url="http://d2:30000", num_gpus=8),
        ],
    )
    tier = TierConfig(
        name="slow",
        engine_kind="sglang_pd_server",
        pd=pd_cfg,
        repeats=1,
        agentx={"benchmark_duration_s": 900},
    )
    eng = SGLangPDServerEngine(cfg, tier)
    eng.http = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    assert eng.total_gpus == 32  # 2P * 8 + 2D * 8

    eng.start()
    assert eng.is_alive() is True
    for h in ("p1", "p2", "d1", "d2"):
        assert rpcs_by_host[h][-1]["method"] == "kiln_info"

    # Hot reload ships source inline with explicit role ('prefill' to p1/p2, 'decode' to d1/d2)
    cand = tmp_path / "cand_pd.py"
    cand.write_text("def make_forward(model):\n    return None\n")
    eng.hot_reload(str(cand))
    for p_host in ("p1", "p2"):
        last = rpcs_by_host[p_host][-1]
        assert last["method"] == "kiln_hot_reload_source"
        assert last["args"][2] == "prefill"
    for d_host in ("d1", "d2"):
        last = rpcs_by_host[d_host][-1]
        assert last["method"] == "kiln_hot_reload_source"
        assert last["args"][2] == "decode"

    # Greedy probe flushes all 4 backends directly and sends /generate via router
    ids, lps = eng.greedy_probe([[1, 2, 3]], 3)
    assert ids == [[50, 51, 52]]
    assert lps == [[{50: -0.05}, {51: -0.05}, {52: -0.05}]]
    assert len(router_generates) == 1

    # AgentX measurement passes total_gpus == 32 across all P + D instances
    seen_agentx: list[tuple[str, int]] = []

    def fake_agentx(url, model, tokenizer, ax_cfg, artifact_dir, num_gpus, timeout_s):
        seen_agentx.append((url, num_gpus))
        return AgentXResult(5120.0, 640.0, 4.2, 6.8, True, "pd_ok.json")

    monkeypatch.setattr("kiln.engine.sglang_pd.run_agentx_profile", fake_agentx)
    m = eng.measure([Request(prompt_ids=[1, 2, 3], output_len=8)], repeats=1, trace=False)
    assert seen_agentx == [("http://router:30080", 32)]
    assert m.decode_tps == 5120.0
    assert m.extra["pd_topology"]["total_gpus"] == 32
    assert m.extra["pd_topology"]["prefill_instances"] == 2
    assert m.extra["pd_topology"]["decode_instances"] == 2

    # Partial failure on d2 triggers automatic cluster-wide kiln_teardown rollback!
    fail_host["d2"] = True
    with pytest.raises(RuntimeError, match="simulated compile crash"):
        eng.hot_reload(str(cand))
    for h in ("p1", "p2", "d1", "d2"):
        assert rpcs_by_host[h][-1]["method"] == "kiln_teardown"

    eng.shutdown()
    eng.http.close()


def test_pd_router_fallback_and_2p1d_bootstrap_pairing(monkeypatch):
    import sys
    from fastapi.testclient import TestClient
    from kiln.engine.pd_router import create_pd_router_app

    pd_auto = PDClusterConfig(
        transfer_backend="mooncake_tcp",
        prefill=[
            PDNodeConfig(url="http://127.0.0.1:31500", bootstrap_port=31501),
            PDNodeConfig(url="http://127.0.0.1:31502", bootstrap_port=31503),
        ],
        decode=[PDNodeConfig(url="http://127.0.0.1:31504")],
        router_policy="round_robin",
    )
    monkeypatch.setattr("kiln.engine.sglang_pd._has_rust_sglang_router", lambda: False)
    monkeypatch.setattr("kiln.engine.sglang_pd.shutil.which", lambda _n: None)
    cmd_fb = build_router_cmd(
        pd_auto,
        ["http://127.0.0.1:31500", "http://127.0.0.1:31502"],
        ["http://127.0.0.1:31504"],
        "127.0.0.1",
        30080,
    )
    assert cmd_fb[:3] == [sys.executable, "-m", "kiln.engine.pd_router"]

    monkeypatch.setattr("kiln.engine.sglang_pd._has_rust_sglang_router", lambda: True)
    cmd_rs = build_router_cmd(
        pd_auto,
        ["http://127.0.0.1:31500", "http://127.0.0.1:31502"],
        ["http://127.0.0.1:31504"],
        "127.0.0.1",
        30080,
    )
    assert cmd_rs[:3] == [sys.executable, "-m", "sglang_router.launch_router"]

    seen: list[tuple[int, dict]] = []

    def backend_handler(request: httpx.Request) -> httpx.Response:
        port = request.url.port
        body = json.loads(request.content.decode()) if request.content else {}
        seen.append((port, body))
        if port == 31500:
            assert body["bootstrap_port"] == 31501
        elif port == 31502:
            assert body["bootstrap_port"] == 31503
        elif port == 31504:
            assert body["bootstrap_port"] in (31501, 31503)
        return httpx.Response(
            200,
            json={
                "output_ids": [101, 102],
                "meta_info": {
                    "completion_tokens": 2,
                    "input_token_logprobs": None,
                    "output_token_logprobs": [[-0.1, 101, "a"], [-0.2, 102, "b"]],
                },
            },
        )

    router_app = create_pd_router_app(
        [("http://127.0.0.1:31500", 31501), ("http://127.0.0.1:31502", 31503)],
        ["http://127.0.0.1:31504"],
        policy="round_robin",
        transport=httpx.MockTransport(backend_handler),
    )
    with TestClient(router_app) as tc:
        for _ in range(4):
            r = tc.post(
                "/generate",
                json={
                    "input_ids": [1, 2, 3],
                    "sampling_params": {"max_new_tokens": 2, "temperature": 0.0},
                    "return_logprob": True,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            assert r.json()["output_ids"] == [101, 102]
        stats = tc.get("/router_stats").json()
        assert stats["prefill_counts"] == {
            "http://127.0.0.1:31500": 2,
            "http://127.0.0.1:31502": 2,
        }
        assert stats["decode_counts"] == {"http://127.0.0.1:31504": 4}

