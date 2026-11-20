import time
from pathlib import Path

from kiln.bench.traffic import synth_traffic
from kiln.config import RunConfig
from kiln.costmodel.model import CHIPS, MODELS, Workload, rank
from kiln.server.jobs import JobManager
from kiln.bench.traffic import load_traffic


def _wait(job, timeout=10):
    t0 = time.time()
    while (job.status in ("queued", "running") or job.finished_at is None) and time.time() - t0 < timeout:
        time.sleep(0.05)
    return job


def test_mock_loop(tmp_path: Path):
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=8, isl=64, osl=16)
    run = RunConfig(name="t", traffic_file=str(traffic), workdir=str(tmp_path))
    mgr = JobManager(run, load_traffic(traffic))
    mgr.start()

    good = mgr.submit("a1", "fast", "SPEEDUP = 1.10\n", "faster")
    bad = mgr.submit("a1", "fast", "SPEEDUP = 2.0\nCORRUPT = True\n", "wrong")
    crash = mgr.submit("a2", "fast", "CRASH = True\n", "ima")
    slow = mgr.submit("a2", "slow", "SPEEDUP = 1.05\n", "promote")

    assert _wait(good).status == "done" and good.speedup > 1.05
    assert _wait(bad).status == "incorrect"
    assert _wait(crash).status == "crashed"
    assert _wait(slow).status == "done"
    assert mgr.tiers["fast"].crashes == 1
    assert mgr.leaderboard("slow")[0]["id"] == slow.id


def test_costmodel_ranks_something():
    rows = rank(MODELS["qwen3-30b-a3b"], CHIPS["B200"], Workload(isl=4096, osl=512), max_chips=16)
    assert rows and rows[0]["tpm_per_chip"] > 0


def test_costmodel_high_tps_prefers_more_tp():
    m, chip = MODELS["llama-3.1-70b"], CHIPS["H100"]
    lo = rank(m, chip, Workload(decode_tps_sla=20), max_chips=8, k=1)[0]
    hi = rank(m, chip, Workload(decode_tps_sla=120), max_chips=8, k=1)
    assert hi, "a high-TPS SLA should still be feasible at TP8"
    assert hi[0]["decode_batch"] <= lo["decode_batch"]


def test_parse_agentx_export(tmp_path: Path):
    import json
    from kiln.bench.agentx import parse_agentx_export
    p = tmp_path / "profile_export_aiperf.json"
    p.write_text(json.dumps({
        "warmup_metrics": {"total_token_throughput": {"unit": "tokens/sec", "avg": 1.0}},
        "total_token_throughput": {"unit": "tokens/sec", "avg": 80000.0},
        "output_token_throughput": {"unit": "tokens/sec", "avg": 2000.0},
        "inter_token_latency": {"unit": "ms", "avg": 9.0, "p50": 8.0, "p90": 12.5},
        "submission_valid": True,
    }))
    r = parse_agentx_export(p, num_gpus=4)
    assert r.total_tput_per_gpu == 20000.0
    assert r.itl_p90_ms == 12.5 and abs(r.interactivity_p90 - 80.0) < 1e-9
    assert r.submission_valid is True


def test_dsv41_h20_config_and_costmodel():
    run = RunConfig.load("configs/dsv41_flash_h20.yaml")
    assert run.engine.tensor_parallel_size == 8
    assert run.engine.extra_args["engram_config"] == {"cpu_offload": False}
    assert "agy" in run.agents.command and "{model}" in run.agents.command
    assert "{prompt}" in run.agents.command
    assert run.agents.models == [
        "gemini-4-neon-high",
        "gemini-4-neon-high",
        "gemini-4-carbon-high",
        "claude-opus-5-5-high",
    ]
    resolved_models = [run.agents.resolve_slot_command(i)[1] for i in range(4)]
    assert resolved_models == run.agents.models
    assert run.client_url == "http://127.0.0.1:8765"
    fast, slow = run.tiers
    assert fast.max_requests == 16 and fast.num_layers is None
    assert fast.engine_kind == "vllm_offline"
    assert slow.engine_kind == "vllm_server"

    rows = rank(MODELS["deepseek4.1-flash"], CHIPS["h20-3e"],
                Workload(isl=8192, osl=1024, ttft_sla_s=3.0, decode_tps_sla=30), max_chips=16)
    assert rows, "8x or 16x H20-3e should fit DeepSeek-V4.1-Flash (~510 GB)"
    assert all(r["n"] >= 8 for r in rows), "4x H20-3e (564 GB) should be pruned for ~510 GB checkpoint"


def test_submodule_forward_snapshot_restore():
    import types
    from kiln.engine.vllm_ext import _restore_forwards, _snapshot_forwards

    class FakeMLP:
        def forward(self, x):
            return x + 1

    class FakeModel:
        def __init__(self):
            self.mlp = FakeMLP()

        def named_modules(self):
            yield "", self
            yield "mlp", self.mlp

        def forward(self, x):
            return self.mlp.forward(x)

    m = FakeModel()
    assert m.forward(10) == 11
    snap = _snapshot_forwards(m)

    # Simulate an agent patching both model.forward and a child submodule without defining restore()
    m.mlp.forward = types.MethodType(lambda self, x: x * 100, m.mlp)
    m.forward = types.MethodType(lambda self, x: self.mlp.forward(x) + 5, m)
    assert m.forward(10) == 1005

    _restore_forwards(snap)
    assert "forward" not in m.mlp.__dict__
    assert "forward" not in m.__dict__
    assert m.forward(10) == 11


def test_warm_recovery_vs_fatal_restart(tmp_path: Path):
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=8, isl=64, osl=16)
    run = RunConfig(name="warm", traffic_file=str(traffic), workdir=str(tmp_path))
    mgr = JobManager(run, load_traffic(traffic), local_tiers={"fast"})
    mgr.start()
    mgr.tiers["fast"].ready.wait(timeout=5)
    eng_initial = mgr.tiers["fast"].engine

    # SyntaxError in candidate should fail job, increment crashes, but keep warm engine alive
    syntax_bad = mgr.submit("a1", "fast", "def broken(:\n", "typo")
    assert _wait(syntax_bad).status == "crashed"
    assert mgr.tiers["fast"].crashes == 1
    assert mgr.tiers["fast"].engine is eng_initial

    # Fatal IMA error must replace the engine instance
    ima = mgr.submit("a1", "fast", "CRASH = True\n", "ima")
    assert _wait(ima).status == "crashed"
    assert mgr.tiers["fast"].crashes == 2
    assert mgr.tiers["fast"].engine is not eng_initial


def test_two_node_remote_worker_endpoints(tmp_path: Path):
    from fastapi.testclient import TestClient
    from kiln.server.app import create_app

    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=6, isl=32, osl=8)
    run = RunConfig(name="twonode", traffic_file=str(traffic), workdir=str(tmp_path))
    # Node 0 only boots 'fast' locally; 'slow' is served by Node 1 over HTTP
    app = create_app(run, local_tiers={"fast"})
    with TestClient(app) as c:
        st = c.get("/status").json()
        modes = {t["tier"]: t["mode"] for t in st["tiers"]}
        assert modes == {"fast": "local", "slow": "remote"}

        # Node 1 fetches traffic & registers heartbeat
        reqs = c.get("/workers/slow/requests").json()
        assert len(reqs) == 6
        c.post("/workers/slow/heartbeat",
               json={"worker_id": "node1", "ready": True,
                     "baseline_tps": 1000.0, "crashes": 0}).raise_for_status()

        # Agent submits slow-tier candidate to Node 0
        jid = c.post("/jobs", json={"agent": "codex_1", "tier": "slow",
                                    "source": "SPEEDUP = 1.12\n", "note": "fused mla"}).json()["id"]

        # Node 1 claims, evaluates, and completes over HTTP without shared filesystem
        claimed = c.post("/workers/slow/claim",
                         params={"worker_id": "node1", "wait_s": 0.5}).json()["job"]
        assert claimed["id"] == jid and "SPEEDUP = 1.12" in claimed["source"]
        c.post("/workers/slow/complete", json={
            "id": jid, "worker_id": "node1", "attempt_id": claimed["attempt_id"],
            "status": "done", "speedup": 1.12, "decode_tps": 1120.0,
            "prefill_tps": 22400.0, "variance_pct": 0.15, "token_match": 1.0, "crashes": 0,
        }).raise_for_status()

        lb = c.get("/leaderboard", params={"tier": "slow"}).json()
        assert lb and lb[0]["id"] == jid and lb[0]["speedup"] == 1.12
        assert app.state.mgr.tiers["fast"].ready.wait(timeout=5)


def test_single_node_configs_load():
    r1 = RunConfig.load("configs/dsv41_flash_h20_1node.yaml")
    assert r1.engine.kind == "vllm_server" and r1.engine.tensor_parallel_size == 8
    f1, s1 = r1.tiers
    assert f1.share_engine == "slow" and f1.agentx is None and f1.max_requests == 16
    assert s1.engine_kind == "vllm_server" and s1.agentx is not None

    r2 = RunConfig.load("configs/dsv41_flash_h20_1node_tp4.yaml")
    assert r2.engine.tensor_parallel_size == 4
    assert r2.engine.extra_args["engram_config"] == {"cpu_offload": True}
    f2, s2 = r2.tiers
    assert f2.gpus == [0, 1, 2, 3] and s2.gpus == [4, 5, 6, 7]


def test_shared_engine_single_node_loop(tmp_path: Path):
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=8, isl=64, osl=16)
    run = RunConfig.load("configs/mock.yaml")
    run.name = "shared1node"
    run.traffic_file = str(traffic)
    run.workdir = str(tmp_path)
    run.tiers[0].share_engine = "slow"
    run.tiers[0].max_requests = 4

    mgr = JobManager(run, load_traffic(traffic))
    mgr.start()
    assert mgr.tiers["fast"].ready.wait(timeout=5)
    assert mgr.tiers["slow"].ready.wait(timeout=5)
    assert mgr.tiers["fast"].stats()["mode"] == "shared:slow"

    # Only one underlying engine instance should be created
    owner_eng_initial = mgr.tiers["slow"].engine
    assert mgr.tiers["fast"].engine.owner.engine is owner_eng_initial

    good_fast = mgr.submit("a1", "fast", "SPEEDUP = 1.15\n", "fast ok")
    bad_fast = mgr.submit("a1", "fast", "SPEEDUP = 2.0\nCORRUPT = True\n", "fast bad")
    good_slow = mgr.submit("a2", "slow", "SPEEDUP = 1.07\n", "slow ok")

    assert _wait(good_fast).status == "done" and good_fast.speedup > 1.10
    assert _wait(bad_fast).status == "incorrect"
    assert _wait(good_slow).status == "done" and good_slow.speedup > 1.03

    # Fatal crash on fast tier must restart the shared owning engine cleanly
    crash = mgr.submit("a1", "fast", "CRASH = True\n", "fatal ima")
    assert _wait(crash).status == "crashed"
    assert mgr.tiers["slow"].engine is not None
    assert mgr.tiers["slow"].engine is not owner_eng_initial

    after = mgr.submit("a2", "slow", "SPEEDUP = 1.09\n", "after recovery")
    assert _wait(after).status == "done" and after.speedup > 1.05


def test_vllm_server_measure_requests_fallback():
    import json
    import httpx
    from kiln.bench.traffic import Request
    from kiln.config import EngineConfig, TierConfig
    from kiln.engine.vllm_server import VLLMServerEngine

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": True})
        if request.url.path == "/v1/completions":
            body = json.loads(request.content.decode())
            max_tok = int(body["max_tokens"])
            return httpx.Response(200, json={
                "choices": [{"token_ids": list(range(max_tok))}],
                "usage": {"completion_tokens": max_tok},
            })
        return httpx.Response(404)

    cfg = EngineConfig(kind="vllm_server", model="dummy-model", tensor_parallel_size=8)
    tier = TierConfig(name="fast", share_engine="slow", agentx=None, repeats=2, trace_kernels=False)
    eng = VLLMServerEngine(cfg, tier)
    eng.http = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)

    reqs = [Request(prompt_ids=[1, 2, 3, 4], output_len=8) for _ in range(4)]
    res = eng.measure(reqs, repeats=2, trace=False)
    assert res.decode_tps > 0 and res.prefill_tps > 0
    assert len(res.runs_decode_tps) == 2


def test_engine_factory_vllm_offline_and_server():
    from kiln.config import EngineConfig, TierConfig
    from kiln.engine import make_engine
    from kiln.engine.vllm_offline import VLLMOfflineEngine
    from kiln.engine.vllm_server import VLLMServerEngine

    tier = TierConfig(name="fast")
    assert isinstance(make_engine(EngineConfig(kind="vllm_offline"), tier), VLLMOfflineEngine)
    assert isinstance(make_engine(EngineConfig(kind="vllm"), tier), VLLMOfflineEngine)
    assert isinstance(make_engine(EngineConfig(kind="vllm_server"), tier), VLLMServerEngine)

