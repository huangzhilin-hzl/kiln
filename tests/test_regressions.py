from types import SimpleNamespace

import pytest

from kiln.bench.agentx import AgentXResult
from kiln.bench.correctness import compare_tokens
from kiln.bench.traffic import Request
from kiln.config import CorrectnessConfig, EngineConfig, RunConfig, TierConfig
from kiln.engine.base import MeasureResult
from kiln.engine.vllm_ext import HotReloadExt
from kiln.engine.vllm_server import VLLMServerEngine
from kiln.server import jobs


def test_agentx_duration_validated_before_engine_start():
    with pytest.raises(ValueError, match="at least 900 seconds"):
        RunConfig.model_validate({"tiers": [{"name": "slow", "agentx": {"benchmark_duration_s": 600}}]})
    assert TierConfig(name="slow", agentx={"benchmark_duration_s": 900}).agentx
    assert TierConfig(name="slow", agentx={"benchmark_duration_s": None}).agentx
    assert TierConfig(name="slow", agentx={"benchmark_duration_s": 180, "unsafe_override": True}).agentx


def test_unsafe_override_agentx_enters_leaderboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tier = TierConfig(name="slow", agentx={"benchmark_duration_s": 180, "unsafe_override": True}, repeats=1)
    eng = VLLMServerEngine(EngineConfig(), tier)
    monkeypatch.setattr(eng, "hot_reload", lambda path: None)
    monkeypatch.setattr(eng, "teardown_candidate", lambda: None)
    monkeypatch.setattr(eng, "_reset_cache", lambda: None)
    monkeypatch.setattr(eng, "greedy_outputs", lambda prompts, n: [[42]])
    monkeypatch.setattr(
        "kiln.engine.vllm_server.run_agentx_profile",
        lambda *a, **kw: AgentXResult(5000.0, 120.0, 2.0, 3.0, False, "short.json"),
    )
    manager = jobs.JobManager(RunConfig(tiers=[tier], workdir=str(tmp_path)), [Request([1], 1)])
    worker = manager.tiers["slow"]
    worker.engine = eng
    worker.baseline_tokens = [[42]]
    worker.baseline_tps = 1000.0
    job = manager.submit("a", "slow", "pass")
    try:
        worker.execute_job(job)
    finally:
        eng.http.close()
    assert job.status == "done"
    assert job.extra["unsafe_override"] is True
    assert manager.leaderboard("slow")[0]["id"] == job.id


@pytest.mark.parametrize("valid, throughput", [(False, 2000.0), (None, 2000.0), (True, 0.0), (True, float("nan"))])
def test_unscorable_agentx_repeat_does_not_enter_leaderboard(tmp_path, monkeypatch, valid, throughput):
    monkeypatch.chdir(tmp_path)
    tier = TierConfig(name="slow", agentx={}, repeats=2)
    eng = VLLMServerEngine(EngineConfig(), tier)
    teardown_calls = []
    monkeypatch.setattr(eng, "hot_reload", lambda path: None)
    monkeypatch.setattr(eng, "teardown_candidate", lambda: teardown_calls.append(True))
    monkeypatch.setattr(eng, "_reset_cache", lambda: None)
    monkeypatch.setattr(eng, "greedy_outputs", lambda prompts, n: [[42]])
    results = iter([AgentXResult(1000.0, 20.0, 8.0, 10.0, True, "good.json"),
                    AgentXResult(throughput, 10.0, 8.0, 10.0, valid, "bad.json")])
    monkeypatch.setattr("kiln.engine.vllm_server.run_agentx_profile", lambda *a, **kw: next(results))
    manager = jobs.JobManager(RunConfig(tiers=[tier], workdir=str(tmp_path)), [Request([1], 1)])
    worker = manager.tiers["slow"]
    worker.engine = eng
    worker.baseline_tokens = [[42]]
    worker.baseline_tps = 1000.0
    job = manager.submit("a", "slow", "pass")
    try:
        worker.execute_job(job)
    finally:
        eng.http.close()
    assert job.status == "invalid"
    assert job.speedup is None
    assert [r["raw_path"] for r in job.extra["repeats"]] == ["good.json", "bad.json"]
    assert manager.leaderboard("slow") == []
    assert teardown_calls == [True]


@pytest.mark.parametrize("restore_fails", [False, True])
def test_failed_candidate_factory_rolls_back_or_requires_restart(tmp_path, restore_fails):
    class Model:
        state = "baseline"

        def forward(self):
            return self.state

    model = Model()
    ext = HotReloadExt()
    ext.model_runner = SimpleNamespace(get_model=lambda: model)
    ext._kiln_rebuild = lambda: None
    candidate = tmp_path / "broken.py"
    restore_body = 'raise ValueError("cannot restore")' if restore_fails else 'model.state = "baseline"'
    candidate.write_text(
        'def make_forward(model):\n'
        '    model.state = "candidate"\n'
        '    model.forward = lambda: "candidate forward"\n'
        '    raise ValueError("candidate setup failed")\n'
        f'def restore(model):\n    {restore_body}\n'
    )
    if restore_fails:
        with pytest.raises(RuntimeError, match="rollback failed"):
            ext.kiln_hot_reload(str(candidate))
        with pytest.raises(RuntimeError, match="restart the engine"):
            ext.kiln_teardown()
        assert "forward" not in vars(model)
    else:
        with pytest.raises(ValueError, match="candidate setup failed"):
            ext.kiln_hot_reload(str(candidate))
        ext.kiln_teardown()
        assert model.forward() == "baseline"


def test_remote_claim_retry_lease_expiry_and_restart(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: now[0])
    run = RunConfig(workdir=str(tmp_path), tiers=[TierConfig(name="slow")])

    def open_coordinator():
        return jobs.JobManager(run, [Request([1], 1)], local_tiers=set())

    manager = open_coordinator()
    job = manager.submit("agent", "slow", "SPEEDUP = 1.12")
    waiting = manager.submit("agent", "slow", "SPEEDUP = 1.03")
    claim = manager.claim_remote("slow", "worker1", wait_s=0)
    # The retry of a claim whose response was lost must get the same job and attempt.
    assert manager.claim_remote("slow", "worker1", wait_s=0) == claim
    assert claim["id"] == job.id
    now[0] += jobs.REMOTE_LEASE_S - 1
    manager.heartbeat_remote("slow", "worker1", True, 1000.0, 0, claim["attempt_id"])
    now[0] += 2

    manager = open_coordinator()
    assert manager.get_job(waiting.id)["status"] == "queued"
    assert manager.claim_remote("slow", "worker1", wait_s=0) == claim
    now[0] += jobs.REMOTE_LEASE_S + 1
    assert manager.get_job(job.id)["status"] == "queued"
    assert manager.status()["tiers"][0]["ready"] is False

    # The queued job can run independently of the expired attempt.
    assert manager.claim_remote("slow", "worker2", wait_s=0)["id"] == waiting.id
    new_claim = manager.claim_remote("slow", "worker3", wait_s=0)
    assert new_claim["id"] == job.id
    assert new_claim["attempt_id"] != claim["attempt_id"]
    result = {"id": job.id, "worker_id": "worker1", "attempt_id": claim["attempt_id"],
              "status": "done", "speedup": 1.12}
    with pytest.raises(ValueError, match="stale"):
        manager.complete_remote("slow", result)

    result.update(worker_id="worker3", attempt_id=new_claim["attempt_id"])
    manager.complete_remote("slow", result)
    finished = manager.get_job(job.id)["finished_at"]
    now[0] += 1
    manager = open_coordinator()
    manager.complete_remote("slow", result)
    assert manager.get_job(job.id)["finished_at"] == finished
    assert manager.leaderboard("slow")[0]["id"] == job.id


@pytest.mark.parametrize("other", [[[1]], [[1], [2], [3]]])
def test_correctness_rejects_wrong_number_of_outputs(other):
    result = compare_tokens([[1], [2]], other, CorrectnessConfig())
    assert not result.ok
    assert "expected 2 outputs" in result.detail


def test_smoke_test_fails_if_hot_reload_has_no_effect():
    from scripts.smoke_vllm import run_checks

    engine = SimpleNamespace(
        greedy_outputs=lambda probe, n: [[1] * n for _ in probe],
        hot_reload=lambda path: None,
        teardown_candidate=lambda: None,
        measure=lambda requests, repeats, trace: MeasureResult(100.0, 100.0, 1.0, [100.0]),
    )
    with pytest.raises(RuntimeError, match="Broken candidate passed"):
        run_checks(engine, [[1]], [Request([1], 1)], CorrectnessConfig(), 1, 1)


def test_launcher_preserves_venv_directory_and_stops_failing_agents(tmp_path, monkeypatch):
    from kiln.agents import launcher

    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    monkeypatch.setattr(launcher, "REPO", repo)

    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(launcher.sys.executable)
    monkeypatch.setattr(launcher.sys, "executable", str(python))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    launched = []

    def spawn(cmd, ws, env, prompt):
        import io
        launched.append(env["PATH"])
        return SimpleNamespace(poll=lambda: 1, pid=123), io.BytesIO()

    monkeypatch.setattr(launcher, "_spawn_agent", spawn)
    run = RunConfig(workdir=str(tmp_path))
    run.agents.num_agents = 1
    launcher.launch(run)
    assert len(launched) == 3
    assert all(path.split(launcher.os.pathsep)[0] == str(venv_bin) for path in launched)


def test_bearer_token_required_when_configured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from kiln.server.app import create_app

    traffic = tmp_path / "traffic.jsonl"
    traffic.write_text('{"prompt_ids":[1],"output_len":1}\n')
    monkeypatch.setenv("KILN_TOKEN", "test-token")
    run = RunConfig(workdir=str(tmp_path), traffic_file=str(traffic))
    with TestClient(create_app(run, local_tiers=set())) as client:
        assert client.get("/status").status_code == 401
        assert client.get("/status", headers={"Authorization": "Bearer test-token"}).status_code == 200


def test_vllm_server_cli_preserves_explicit_false():
    from kiln.engine.vllm_server import _cli_args

    assert _cli_args({"enable_prefix_caching": False, "speculative_config": None}) == [
        "--no-enable-prefix-caching",
    ]


def test_agentx_export_cannot_substitute_warmup_for_measured_score(tmp_path):
    from kiln.bench.agentx import parse_agentx_export

    export = tmp_path / "profile_export_aiperf.json"
    export.write_text('{"metadata":{"submission_valid":true},'
                      '"warmup_metrics":{"total_token_throughput":{"avg":8000}}}')
    assert parse_agentx_export(export, num_gpus=1).total_tput_per_gpu == 0


def test_tier_retries_engine_boot_failure(tmp_path, monkeypatch):
    from kiln.engine.mock import MockEngine

    tier = TierConfig(name="fast")
    worker = jobs.TierWorker(RunConfig(tiers=[tier]), tier, [Request([1], 1)], tmp_path)
    original_start = MockEngine.start
    starts = []

    def start_once_failing(engine):
        starts.append(engine)
        if len(starts) == 1:
            raise RuntimeError("temporary startup failure")
        original_start(engine)

    monkeypatch.setattr(MockEngine, "start", start_once_failing)
    monkeypatch.setattr(jobs.time, "sleep", lambda seconds: None)
    worker.wait_until_ready()
    assert worker.ready.is_set()
    assert worker.boot_error is None
    assert worker.engine is starts[1]


def test_remote_worker_retries_completion_beyond_five_attempts(tmp_path, monkeypatch):
    import json

    import httpx

    traffic = tmp_path / "traffic.jsonl"
    traffic.write_text('{"prompt_ids":[1,2],"output_len":1}\n')
    run = RunConfig(workdir=str(tmp_path), traffic_file=str(traffic), tiers=[TierConfig(name="slow")])
    manager = jobs.JobManager(run, [Request([1, 2], 1)], local_tiers=set())
    job = manager.submit("agent", "slow", "SPEEDUP = 1.10")
    delivered = []
    received = 0

    def coordinator(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "heartbeat":
            manager.heartbeat_remote("slow", **json.loads(request.content))
        elif endpoint == "claim":
            if delivered:
                raise KeyboardInterrupt
            claim = manager.claim_remote("slow", request.url.params["worker_id"], wait_s=0)
            return httpx.Response(200, json={"job": claim})
        elif endpoint == "complete":
            nonlocal received
            received += 1
            assert received <= 7
            payload = json.loads(request.content)
            manager.complete_remote("slow", payload)
            if received < 7:
                raise httpx.ReadTimeout("completion response lost", request=request)
            delivered.append(payload)
        else:
            raise AssertionError(endpoint)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(coordinator))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(jobs.time, "sleep", lambda seconds: None)
    with pytest.raises(KeyboardInterrupt):
        jobs.run_remote_worker(run, "slow", "http://local")
    assert received == 7
    assert manager.get_job(job.id)["status"] == "done"
    assert len(manager.tiers["slow"].durations) == 1


def test_vllm_reload_restores_live_kv_allocation(monkeypatch):
    import sys
    from contextlib import nullcontext

    workspace = []
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", SimpleNamespace(
        unlock_workspace=lambda: workspace.append("unlocked"),
        lock_workspace=lambda: workspace.append("locked"),
    ))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.eplb_utils", SimpleNamespace(
        preserve_serving_state=lambda runner: nullcontext(),
    ))
    cache = SimpleNamespace(num_committed_blocks=4096)

    def commit(n):
        assert workspace == ["unlocked"]
        cache.num_committed_blocks = n

    cache.commit = commit
    ext = HotReloadExt()
    kernel_cfg = SimpleNamespace(enable_flashinfer_autotune=True)
    ext.vllm_config = SimpleNamespace(kernel_config=kernel_cfg)
    ext.model_runner = SimpleNamespace(extensible_kv_cache=cache, max_num_tokens=8,
                                       _dummy_run=lambda *a, **kw: None)
    ext._kiln_release_graphs = lambda: None
    seen_autotune = []

    def warmup():
        seen_autotune.append(kernel_cfg.enable_flashinfer_autotune)
        commit(2)  # vLLM startup warmup shrinks the cache

    ext.compile_or_warm_up_model = warmup
    ext._kiln_rebuild()
    assert cache.num_committed_blocks == 4096
    assert workspace == ["unlocked", "locked"]
    assert seen_autotune == [False]
    assert kernel_cfg.enable_flashinfer_autotune is True


def test_spawn_agent_substitutes_prompt_placeholder(tmp_path):
    import os
    import shlex
    import sys
    from kiln.agents.launcher import _spawn_agent

    cmd = shlex.split(f"{sys.executable} -c 'import sys; print(sys.argv[1], sys.stdin.read() == \"\")' {{prompt}}")
    proc, log = _spawn_agent(cmd, tmp_path, dict(os.environ), "hello\nmulti-line $prompt")
    assert proc.wait(timeout=5) == 0
    log.close()
    assert (tmp_path / "agent.log").read_text() == "hello\nmulti-line $prompt True\n"


def test_parse_quota_backoff(tmp_path):
    from kiln.agents.launcher import _parse_quota_backoff

    log = tmp_path / "agent.log"
    assert _parse_quota_backoff(log) is None

    log.write_text("ok run\nerror: Individual quota reached. Resets in 1h2m12s.\n")
    assert _parse_quota_backoff(log, slot_idx=0) == 3732.0
    assert _parse_quota_backoff(log, slot_idx=2) == 3762.0

    log.write_text("AGY_ERROR: {\"error_code\":429,\"message\":\"Resets in 55m50s.\"}\n")
    assert _parse_quota_backoff(log, slot_idx=0) == 3350.0

    log.write_text("AGY_ERROR: RESOURCE_EXHAUSTED (no reset timer)\n")
    assert _parse_quota_backoff(log, slot_idx=1) == 315.0

    log.write_text(
        "old RESOURCE_EXHAUSTED Resets in 50m0s\n[kiln] respawned\nnormal non-429 crash\n"
    )
    assert _parse_quota_backoff(log, slot_idx=0) is None


def test_agent_config_multi_model_and_commands():
    from kiln.config import AgentConfig

    cfg = AgentConfig(
        num_agents=4,
        command="agy --model {model} -p {prompt}",
        models=["m-a", "m-b"],
    )
    assert cfg.resolve_slot_command(0) == (["agy", "--model", "m-a", "-p", "{prompt}"], "m-a")
    assert cfg.resolve_slot_command(1) == (["agy", "--model", "m-b", "-p", "{prompt}"], "m-b")
    assert cfg.resolve_slot_command(2) == (["agy", "--model", "m-a", "-p", "{prompt}"], "m-a")

    cfg_override = AgentConfig(
        num_agents=3,
        commands=[
            "agy --model default-m -p {prompt}",
            "claude -p --model=old-m",
        ],
        models=["m1", "m2", "m3"],
    )
    assert cfg_override.resolve_slot_command(0) == (["agy", "--model", "m1", "-p", "{prompt}"], "m1")
    assert cfg_override.resolve_slot_command(1) == (["claude", "-p", "--model=m2"], "m2")
    assert cfg_override.resolve_slot_command(2) == (["agy", "--model", "m3", "-p", "{prompt}"], "m3")

    with pytest.raises(ValueError, match="models"):
        AgentConfig(num_agents=1, command="agy --model {model} -p {prompt}")
    with pytest.raises(ValueError, match="requires"):
        AgentConfig(num_agents=1, command="claude -p", models=["opus"])
    with pytest.raises(ValueError, match="blank"):
        AgentConfig(num_agents=1, command="agy --model {model}", models=["  "])


def test_launcher_per_slot_models_commands_respawn_and_refs_index(tmp_path, monkeypatch):
    import io
    from pathlib import Path
    from kiln.agents import launcher
    from kiln.config import AgentConfig

    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    model_dir = repo / "refs" / "vllm" / "models" / "demo_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "model.py").write_text(
        "class DemoModel:\n"
        + "".join(f"    x{i} = {i}\n" for i in range(65))
        + "    def forward(self, x):\n        return x\n"
    )

    monkeypatch.setattr(launcher, "REPO", repo)
    monkeypatch.setattr(launcher.time, "sleep", lambda s: None)

    spawned = []
    stop_flag = [False]

    class FakeProc:
        def __init__(self, pid, exits):
            self.pid = pid
            self._exits = iter(exits)

        def poll(self):
            if stop_flag[0]:
                return 0
            nxt = next(self._exits, "interrupt")
            if nxt == "interrupt":
                stop_flag[0] = True
                raise KeyboardInterrupt
            return nxt

    slot_runs = {0: 0, 1: 0}

    def fake_spawn(cmd, ws, env, prompt):
        slot_idx = int(env["KILN_AGENT"].split("_")[1]) - 1
        run_no = slot_runs[slot_idx]
        slot_runs[slot_idx] += 1
        spawned.append((env["KILN_AGENT"], list(cmd), env.get("KILN_AGENT_MODEL"), prompt))
        if slot_idx == 0 and run_no == 0:
            return FakeProc(1000 + slot_idx, [0]), io.BytesIO()
        return FakeProc(2000 + slot_idx, []), io.BytesIO()

    monkeypatch.setattr(launcher, "_spawn_agent", fake_spawn)

    run = RunConfig(
        name="mixed",
        workdir=str(tmp_path / "runs"),
        engine=EngineConfig(kind="vllm_offline", model="Demo-v1"),
        agents=AgentConfig(
            num_agents=2,
            commands=[
                "agy --model {model} -p {prompt}",
                "claude -p --model {model}",
            ],
            models=["neon-high", "opus-high"],
            respawn_on_exit=True,
        ),
    )
    launcher.launch(run)

    # refs/INDEX.md auto-generated and symlinked
    index_text = (repo / "refs" / "INDEX.md").read_text()
    assert "DemoModel" in index_text
    assert (Path(run.workdir) / "mixed" / "agents" / "agent_1" / "refs" / "INDEX.md").exists()

    # Initial spawns: agent_1 (agy) and agent_2 (claude) both receive --add-dir <refs>
    a1_init, a2_init, a1_respawn = spawned[0], spawned[1], spawned[2]
    assert a1_init[0] == "agent_1" and a1_init[2] == "neon-high"
    assert a1_init[1][:4] == ["agy", "--add-dir", str(repo / "refs"), "--model"]
    assert "neon-high" in a1_init[1]

    assert a2_init[0] == "agent_2" and a2_init[2] == "opus-high"
    assert a2_init[1] == ["claude", "--add-dir", str(repo / "refs"), "-p", "--model", "opus-high"]

    # Respawn preserves slot 0's own cmd & model and includes INDEX.md reminder
    assert a1_respawn[0] == "agent_1"
    assert a1_respawn[1] == a1_init[1]
    assert a1_respawn[2] == "neon-high"
    assert "refs/INDEX.md" in a1_respawn[3]


def test_refs_index_ast_budget_preserve_and_cli(tmp_path, capsys):
    from kiln.agents.refs_index import GENERATED_PREFIX, build_refs_index, ensure_refs_index
    from kiln.cli import main

    refs = tmp_path / "refs"
    pkg = refs / "vllm" / "models" / "deepseek_v41"
    (pkg / "amd").mkdir(parents=True)
    (pkg / "nvidia" / "ops").mkdir(parents=True)
    layers = refs / "vllm" / "model_executor" / "layers"
    layers.mkdir(parents=True)

    (pkg / "amd" / "model.py").write_text("class AmdOnly:\n" + "    pass\n" * 80)
    (pkg / "nvidia" / "model.py").write_text(
        "from vllm.model_executor.layers.custom_mhc import MHCBenchLayer\n"
        "class DeepseekV4Model:\n"
        + "".join(f"    a{i} = {i}\n" for i in range(60))
        + "    def forward(self, x):\n        return x\n"
    )
    (pkg / "attention.py").write_text(
        "class DeepseekV4Attention:\n"
        + "".join(f"    b{i} = {i}\n" for i in range(60))
        + "    def forward(self, q, k, v):\n        return q\n"
    )
    (pkg / "nvidia" / "ops" / "mhc.py").write_text(
        "def mhc_op(x):\n" + "".join(f"    y{i} = x\n" for i in range(60)) + "    return x\n"
    )
    (layers / "custom_mhc.py").write_text(
        "class MHCBenchLayer:\n"
        + "".join(f"    c{i} = {i}\n" for i in range(85))
        + "    def forward(self, x):\n        return x\n"
    )

    out = build_refs_index(refs, model_hint="/models/DeepSeek-V4.1-Flash", engine_hint="vllm_offline")
    assert out.startswith(GENERATED_PREFIX)
    assert "vllm/models/deepseek_v41/nvidia/model.py" in out
    assert "DeepseekV4Model" in out and "`forward`:" in out
    assert "vllm/models/deepseek_v41/attention.py" in out
    assert "vllm/model_executor/layers/custom_mhc.py" in out
    assert "AmdOnly" not in out

    # Strict max_lines enforcement
    tight = build_refs_index(refs, model_hint="DeepSeek-V4.1-Flash", max_lines=10)
    assert len(tight.splitlines()) <= 10

    # Hand-written INDEX.md must be preserved when force=False
    idx_path = refs / "INDEX.md"
    idx_path.write_text("# My custom manual index\n")
    ensure_refs_index(refs, model_hint="DeepSeek-V4.1-Flash", force=False)
    assert idx_path.read_text() == "# My custom manual index\n"

    # CLI --force overwrites and prints; subsequent ensure without force keeps matching model, updates on model change
    main(["index-refs", "--refs", str(refs), "--model", "DeepSeek-V4.1-Flash", "--force", "--print"])
    captured = capsys.readouterr().out
    assert "DeepseekV4Model" in captured
    assert "model=DeepSeek-V4.1-Flash -->" in idx_path.read_text()

    idx_path.write_text(idx_path.read_text() + "<!-- marker -->\n")
    ensure_refs_index(refs, model_hint="DeepSeek-V4.1-Flash", force=False)
    assert "<!-- marker -->" in idx_path.read_text()

    ensure_refs_index(refs, model_hint="Other-Model", force=False)
    assert "model=Other-Model -->" in idx_path.read_text()


def test_agent_slots_multi_provider_config():
    from kiln.config import AgentConfig, AgentSlotConfig

    cfg = AgentConfig(
        slots=[
            AgentSlotConfig(provider="agy", model="gemini-4-neon-high", count=2),
            AgentSlotConfig(
                provider="codex",
                model="o3",
                extra_args=["-c", "model_reasoning_effort=high"],
                env={"OPENAI_BASE_URL": "https://proxy.example/v1"},
            ),
            AgentSlotConfig(
                provider="claude",
                model="claude-opus-4-5",
                extra_args=["--max-turns", "25"],
                env={"ANTHROPIC_API_KEY": "sk-ant-test"},
            ),
            AgentSlotConfig(
                provider="custom",
                command="my-agent -m {model} --run",
                model="deepseek-r1",
            ),
        ]
    )
    assert cfg.num_agents == 5

    s0 = cfg.resolve_slot(0)
    s1 = cfg.resolve_slot(1)
    assert s0.provider == "agy" and s0.model == "gemini-4-neon-high"
    assert s1.provider == "agy" and s1.model == "gemini-4-neon-high"
    assert s0.argv[-2:] == ["-p", "{prompt}"]
    assert s0.argv[s0.argv.index("--model") + 1] == "gemini-4-neon-high"

    s2 = cfg.resolve_slot(2)
    assert s2.provider == "codex" and s2.model == "o3"
    assert s2.argv[:2] == ["codex", "exec"]
    assert s2.argv[-1] == "-"
    assert s2.argv[s2.argv.index("--model") + 1] == "o3"
    assert s2.argv[-3:-1] == ["model_reasoning_effort=high", "-"][:1] or "model_reasoning_effort=high" in s2.argv[:-1]
    assert s2.env == {"OPENAI_BASE_URL": "https://proxy.example/v1"}

    s3 = cfg.resolve_slot(3)
    assert s3.provider == "claude" and s3.model == "claude-opus-4-5"
    assert s3.argv[:3] == ["claude", "-p", "--dangerously-skip-permissions"]
    assert s3.argv[s3.argv.index("--model") + 1] == "claude-opus-4-5"
    assert s3.argv[-2:] == ["--max-turns", "25"]
    assert s3.env == {"ANTHROPIC_API_KEY": "sk-ant-test"}

    s4 = cfg.resolve_slot(4)
    assert s4.provider == "custom" and s4.model == "deepseek-r1"
    assert s4.argv == ["my-agent", "-m", "deepseek-r1", "--run"]

    # Explicit num_agents overrides sum(count), cycling slots; -m flag replacement works too
    cfg_short = AgentConfig(
        num_agents=3,
        slots=[
            AgentSlotConfig(command="codex exec -m old-model -", model="gpt-5"),
            AgentSlotConfig(provider="claude", model="sonnet"),
        ],
    )
    assert cfg_short.num_agents == 3
    assert cfg_short.resolve_slot(0).argv == ["codex", "exec", "-m", "gpt-5", "-"]
    assert cfg_short.resolve_slot(2).argv == ["codex", "exec", "-m", "gpt-5", "-"]

    with pytest.raises(ValueError, match="requires 'provider'"):
        AgentSlotConfig(model="o3")
    with pytest.raises(ValueError, match="custom"):
        AgentSlotConfig(provider="custom", model="o3")


def test_launcher_mixed_providers_add_dir_env_and_missing_cli_cleanup(tmp_path, monkeypatch):
    import signal
    from kiln.agents import launcher
    from kiln.config import AgentConfig, AgentSlotConfig

    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    (repo / "refs").mkdir(parents=True)
    monkeypatch.setattr(launcher, "REPO", repo)
    monkeypatch.setattr(launcher.time, "sleep", lambda s: None)

    spawned = []
    signals = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0

        def send_signal(self, sig):
            signals.append((self.pid, sig))

    def fake_spawn(cmd, ws, env, prompt):
        spawned.append((env["KILN_AGENT"], list(cmd), dict(env)))
        return FakeProc(3000 + len(spawned)), open(ws / "agent.log", "ab")

    monkeypatch.setattr(launcher, "_spawn_agent", fake_spawn)

    run = RunConfig(
        name="multi-provider",
        workdir=str(tmp_path / "runs"),
        agents=AgentConfig(
            respawn_on_exit=False,
            slots=[
                AgentSlotConfig(provider="agy", model="gemini-4-neon-high"),
                AgentSlotConfig(
                    provider="codex",
                    model="o3",
                    env={"OPENAI_API_KEY": "sk-openai-slot"},
                ),
                AgentSlotConfig(
                    provider="claude",
                    model="claude-opus-4-5",
                    env={"ANTHROPIC_API_KEY": "sk-claude-slot"},
                ),
            ],
        ),
    )
    launcher.launch(run)

    assert len(spawned) == 3
    refs_str = str(repo / "refs")

    # agent_1: agy gets `--add-dir <refs>` right after binary
    a1_name, a1_cmd, a1_env = spawned[0]
    assert a1_name == "agent_1"
    assert a1_cmd[:3] == ["agy", "--add-dir", refs_str]
    assert a1_env["KILN_AGENT_PROVIDER"] == "agy"
    assert a1_env["KILN_AGENT_MODEL"] == "gemini-4-neon-high"

    # agent_2: codex exec gets `--add-dir <refs>` after `exec` and before trailing `-`
    a2_name, a2_cmd, a2_env = spawned[1]
    assert a2_name == "agent_2"
    assert a2_cmd[:4] == ["codex", "exec", "--add-dir", refs_str]
    assert a2_cmd[-1] == "-"
    assert a2_env["KILN_AGENT_PROVIDER"] == "codex"
    assert a2_env["KILN_AGENT_MODEL"] == "o3"
    assert a2_env["OPENAI_API_KEY"] == "sk-openai-slot"

    # agent_3: claude gets `--add-dir <refs>` right after binary
    a3_name, a3_cmd, a3_env = spawned[2]
    assert a3_name == "agent_3"
    assert a3_cmd[:3] == ["claude", "--add-dir", refs_str]
    assert a3_env["KILN_AGENT_PROVIDER"] == "claude"
    assert a3_env["KILN_AGENT_MODEL"] == "claude-opus-4-5"
    assert a3_env["ANTHROPIC_API_KEY"] == "sk-claude-slot"

    # If a later slot's CLI is missing, earlier running slots must be terminated via SIGTERM
    running_signals = []

    class StillRunningProc:
        pid = 4001

        def poll(self):
            return None

        def send_signal(self, sig):
            running_signals.append((self.pid, sig))

    def spawn_second_missing(cmd, ws, env, prompt):
        if cmd[0] == "codex":
            raise FileNotFoundError(cmd[0])
        return StillRunningProc(), open(ws / "agent.log", "ab")

    monkeypatch.setattr(launcher, "_spawn_agent", spawn_second_missing)
    run_fail = RunConfig(
        name="missing-cli",
        workdir=str(tmp_path / "runs2"),
        agents=AgentConfig(
            respawn_on_exit=False,
            slots=[
                AgentSlotConfig(provider="agy", model="gemini-4-neon-high"),
                AgentSlotConfig(provider="codex", model="o3"),
            ],
        ),
    )
    with pytest.raises(RuntimeError, match="not found on PATH"):
        launcher.launch(run_fail)
    assert running_signals == [(4001, signal.SIGTERM)]


def test_parse_quota_backoff_codex_and_claude(tmp_path):
    from kiln.agents.launcher import _parse_quota_backoff

    log = tmp_path / "agent.log"

    log.write_text("openai.RateLimitError: rate_limit_exceeded. Please try again in 2m30s.\n")
    assert _parse_quota_backoff(log, slot_idx=0) == 150.0
    assert _parse_quota_backoff(log, slot_idx=1) == 165.0

    log.write_text("Error: insufficient_quota (HTTP 429)\n")
    assert _parse_quota_backoff(log, slot_idx=0) == 300.0

    log.write_text('{"type":"error","error":{"type":"rate_limit_error"}}\nretry-after: 45\n')
    assert _parse_quota_backoff(log, slot_idx=0) == 45.0

    log.write_text("Claude error: You've hit your limit · resets in 1h5m\n")
    assert _parse_quota_backoff(log, slot_idx=0) == 3900.0

    log.write_text("API Error: 529 overloaded_error\n")
    assert _parse_quota_backoff(log, slot_idx=2) == 330.0


def test_cli_agents_dry_run(tmp_path, capsys):
    import json
    from kiln.cli import main

    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "name: dryrun\n"
        "agents:\n"
        "  slots:\n"
        "    - {provider: agy, model: gemini-4-neon-high}\n"
        "    - {provider: codex, model: o3}\n"
        "    - {provider: claude, model: claude-opus-4-5}\n"
    )
    main(["agents", "-c", str(cfg_path), "--dry-run"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["agent"] for r in rows] == ["agent_1", "agent_2", "agent_3"]
    assert [r["provider"] for r in rows] == ["agy", "codex", "claude"]
    assert [r["model"] for r in rows] == ["gemini-4-neon-high", "o3", "claude-opus-4-5"]


def test_codex_default_model_and_reasoning_effort_override():
    from kiln.config import AgentConfig, AgentSlotConfig

    cfg_default = AgentConfig(slots=[AgentSlotConfig(provider="codex")])
    s0 = cfg_default.resolve_slot(0)
    assert s0.provider == "codex"
    assert s0.model == "gpt-6.2-sol"
    assert s0.argv[s0.argv.index("--model") + 1] == "gpt-6.2-sol"
    assert "-c" in s0.argv and "model_reasoning_effort=high" in s0.argv

    cfg_override = AgentConfig(
        slots=[AgentSlotConfig(provider="codex", extra_args=["-c", "model_reasoning_effort=ultra"])]
    )
    s_ultra = cfg_override.resolve_slot(0)
    effort_args = [tok for tok in s_ultra.argv if tok.startswith("model_reasoning_effort=")]
    assert effort_args == ["model_reasoning_effort=ultra"]

    run_sglang = RunConfig.load("configs/dsv41_flash_h20_sglang.yaml")
    assert run_sglang.agents.num_agents == 5
    s4 = run_sglang.agents.resolve_slot(4)
    assert s4.provider == "codex"
    assert s4.model == "gpt-6.2-sol"
    assert [tok for tok in s4.argv if tok.startswith("model_reasoning_effort=")] == ["model_reasoning_effort=high"]


def test_refs_index_prefers_engine_root_and_includes_sglang_siblings(tmp_path):
    from kiln.agents.refs_index import ensure_refs_index

    refs = tmp_path / "refs"
    vllm_model = refs / "vllm" / "models" / "deepseek_v41" / "nvidia" / "model.py"
    vllm_model.parent.mkdir(parents=True)
    vllm_model.write_text("class DeepseekV41LLMForCausalLM:\n    def forward(self, x):\n        return x\n" + "\n" * 65)

    sg_models = refs / "sglang" / "srt" / "models"
    sg_models.mkdir(parents=True)
    (sg_models / "deepseek_v4.py").write_text(
        "class DeepseekV4ForCausalLM:\n    def forward(self, input_ids):\n        return input_ids\n"
    )
    (sg_models / "deepseek_v4_dspark.py").write_text(
        "class DeepseekV4DSparkForCausalLM:\n    def forward(self, input_ids):\n        return input_ids\n"
    )

    idx_vllm = ensure_refs_index(refs, model_hint="DeepSeek-V4.1-Flash-1017", engine_hint="vllm_offline")
    assert idx_vllm is not None
    vllm_text = idx_vllm.read_text()
    assert "engine=vllm_offline; model=DeepSeek-V4.1-Flash-1017 -->" in vllm_text
    assert "vllm/models/deepseek_v41/nvidia/model.py" in vllm_text

    # Switching engine_hint on the same model refreshes INDEX.md automatically even with force=False.
    idx_sg = ensure_refs_index(refs, model_hint="DeepSeek-V4.1-Flash-1017", engine_hint="sglang_offline", force=False)
    assert idx_sg is not None
    sg_text = idx_sg.read_text()
    assert "engine=sglang_offline; model=DeepSeek-V4.1-Flash-1017 -->" in sg_text
    assert "sglang/srt/models/deepseek_v4.py" in sg_text
    assert "sglang/srt/models/deepseek_v4_dspark.py" in sg_text
    assert "vllm/models/deepseek_v41" not in sg_text


def test_normalize_proxy_env():
    from kiln.agents.launcher import _normalize_proxy_env

    # Lowercase mirrors to uppercase and merges loopback into NO_PROXY / no_proxy without duplicates.
    env_lo = {
        "http_proxy": "http://127.0.0.1:7897",
        "https_proxy": "http://127.0.0.1:7897",
        "no_proxy": "10.0.0.0/8,localhost",
    }
    out_lo = _normalize_proxy_env(env_lo)
    assert out_lo["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert out_lo["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert "all_proxy" not in out_lo and "ALL_PROXY" not in out_lo
    assert out_lo["NO_PROXY"].split(",") == ["10.0.0.0/8", "localhost", "127.0.0.1", "::1"]
    assert out_lo["no_proxy"] == out_lo["NO_PROXY"]

    # Uppercase mirrors to lowercase without overwriting distinct existing pairs.
    env_hi = {
        "HTTP_PROXY": "http://proxy.internal:8080",
        "ALL_PROXY": "socks5://127.0.0.1:7897",
        "NO_PROXY": "127.0.0.1,*.internal",
    }
    out_hi = _normalize_proxy_env(env_hi)
    assert out_hi["http_proxy"] == "http://proxy.internal:8080"
    assert out_hi["all_proxy"] == "socks5://127.0.0.1:7897"
    assert out_hi["NO_PROXY"].split(",") == ["127.0.0.1", "*.internal", "localhost", "::1"]
    assert out_hi["no_proxy"] == out_hi["NO_PROXY"]


def test_correctness_logprob_gate_and_mock_engine_rejection(tmp_path):
    import time
    from kiln.bench.correctness import evaluate_correctness
    from kiln.config import CorrectnessConfig

    cfg = CorrectnessConfig(min_token_match=0.85, max_logit_abs_diff=0.5)
    ok_res = evaluate_correctness([[1, 2]], [[1, 2]], cfg, [[{1: -0.1}, {2: -0.2}]], [[{1: -0.3}, {2: -0.25}]])
    assert ok_res.ok is True and abs(ok_res.max_logit_abs_diff - 0.2) < 1e-6

    drift_res = evaluate_correctness([[1, 2]], [[1, 2]], cfg, [[{1: -0.1}, {2: -0.2}]], [[{1: -0.9}, {2: -0.2}]])
    assert drift_res.ok is False and abs(drift_res.max_logit_abs_diff - 0.8) < 1e-6
    assert "max logit abs diff" in drift_res.detail

    nan_res = evaluate_correctness([[1]], [[1]], cfg, [[{1: -0.1}]], [[{1: float("nan")}]])
    assert nan_res.ok is False and nan_res.max_logit_abs_diff is None

    traffic = tmp_path / "t.jsonl"
    traffic.write_text('{"prompt_ids":[1,2,3],"output_len":4}\n')
    run = RunConfig(name="logit_gate", traffic_file=str(traffic), workdir=str(tmp_path))
    mgr = jobs.JobManager(run, [Request([1, 2, 3], 4)], local_tiers={"fast"})
    mgr.start()
    j = mgr.submit("a1", "fast", "SPEEDUP = 1.25\nLOGIT_DIFF = 0.8\n", "drifted logits")
    t0 = time.time()
    while j.status in ("queued", "running") and time.time() - t0 < 5:
        time.sleep(0.05)
    assert j.status == "incorrect"
    assert j.token_match == 1.0
    assert abs(j.max_logit_abs_diff - 0.8) < 1e-6
    assert mgr.leaderboard("fast") == []


def test_job_invalidate_cli_and_cross_tier_cascade(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from kiln.cli import main
    from kiln.server.app import create_app

    traffic = tmp_path / "t.jsonl"
    traffic.write_text('{"prompt_ids":[1,2],"output_len":2}\n')
    run = RunConfig(name="cascade", traffic_file=str(traffic), workdir=str(tmp_path))
    app = create_app(run, local_tiers=set())
    cand_src = "SPEEDUP = 1.32\n"

    with TestClient(app) as client:
        # 1. Submit bogus candidate on both fast and slow, mark fast as done
        j_fast = client.post("/jobs", json={"agent": "a4", "tier": "fast", "source": cand_src, "note": "bogus"}).json()["id"]
        j_slow = client.post("/jobs", json={"agent": "a4", "tier": "slow", "source": cand_src, "note": "promote"}).json()["id"]
        j_other = client.post("/jobs", json={"agent": "a1", "tier": "fast", "source": "SPEEDUP = 1.20\n", "note": "clean"}).json()["id"]

        client.post("/workers/fast/heartbeat", json={"worker_id": "w0", "ready": True, "baseline_tps": 400.0, "crashes": 0})
        for jid, sp in ((j_fast, 1.32), (j_other, 1.20)):
            cl = client.post("/workers/fast/claim", params={"worker_id": "w0", "wait_s": 0}).json()["job"]
            client.post("/workers/fast/complete", json={
                "id": cl["id"], "worker_id": "w0", "attempt_id": cl["attempt_id"],
                "status": "done", "speedup": sp, "decode_tps": 400.0 * sp, "token_match": 0.94, "crashes": 0,
            }).raise_for_status()
        assert [r["id"] for r in client.get("/leaderboard", params={"tier": "fast"}).json()] == [j_fast, j_other]

        # 2. Slow fails severely (< 0.5 token_match) -> auto-cascades invalidation to matching fast sha256!
        client.post("/workers/slow/heartbeat", json={"worker_id": "w1", "ready": True, "baseline_tps": 900.0, "crashes": 0})
        cl_s = client.post("/workers/slow/claim", params={"worker_id": "w1", "wait_s": 0}).json()["job"]
        assert cl_s["id"] == j_slow
        client.post("/workers/slow/complete", json={
            "id": j_slow, "worker_id": "w1", "attempt_id": cl_s["attempt_id"],
            "status": "incorrect", "token_match": 0.1406, "error": "token match 0.1406", "crashes": 0,
        }).raise_for_status()

        assert client.get(f"/jobs/{j_fast}").json()["status"] == "invalidated"
        assert [r["id"] for r in client.get("/leaderboard", params={"tier": "fast"}).json()] == [j_other]

        # 3. Manual withdrawal via `kiln withdraw` removes j_other too
        import kiln.cli as cli_mod
        monkeypatch.setattr(cli_mod, "http", client)
        monkeypatch.setenv("KILN_SERVER", "http://testserver")
        main(["withdraw", j_other, "--reason", "manual cleanup"])
        assert client.get(f"/jobs/{j_other}").json()["status"] == "invalidated"
        assert client.get("/leaderboard", params={"tier": "fast"}).json() == []


def test_agentx_paired_compare_and_multi_metric_leaderboard_and_promote(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from fastapi.testclient import TestClient
    from kiln.bench.agentx import compare_agentx_paired
    from kiln.cli import main
    from kiln.server.app import create_app

    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    base_dir.mkdir()
    cand_dir.mkdir()
    (base_dir / "profile_export_aiperf.json").write_text("{}")
    (cand_dir / "profile_export_aiperf.json").write_text("{}")

    (base_dir / "profile_export.jsonl").write_text(
        json.dumps({"metadata": {"benchmark_phase": "warmup", "conversation_id": "c0", "turn_index": 0}, "metrics": {"decode_duration": 9999.0, "output_token_count": 100}}) + "\n"
        + json.dumps({"metadata": {"benchmark_phase": "profile", "conversation_id": "c1", "turn_index": 0, "was_cancelled": False}, "metrics": {"decode_duration": 2000.0, "output_token_count": 500}}) + "\n"
        + json.dumps({"metadata": {"benchmark_phase": "profile", "conversation_id": "c1", "turn_index": 1, "was_cancelled": False}, "metrics": {"decode_duration": 3000.0, "output_token_count": 700}}) + "\n"
    )
    (cand_dir / "profile_export.jsonl").write_text(
        json.dumps({"metadata": {"benchmark_phase": "profile", "conversation_id": "c1", "turn_index": 0, "was_cancelled": False}, "metrics": {"decode_duration": 1600.0, "output_token_count": 500}}) + "\n"
        + json.dumps({"metadata": {"benchmark_phase": "profile", "conversation_id": "c1", "turn_index": 1, "was_cancelled": False}, "metrics": {"decode_duration": 2400.0, "output_token_count": 700}}) + "\n"
        + json.dumps({"metadata": {"benchmark_phase": "profile", "conversation_id": "c2", "turn_index": 0, "was_cancelled": True}, "metrics": {"decode_duration": 20000.0, "output_token_count": 0}}) + "\n"
    )
    paired = compare_agentx_paired(base_dir / "profile_export_aiperf.json", cand_dir / "profile_export_aiperf.json")
    assert paired is not None
    assert paired["paired_requests"] == 2
    assert paired["baseline_decode_s"] == 5.0
    assert paired["candidate_decode_s"] == 4.0
    assert paired["decode_duration_speedup"] == 1.25
    assert paired["paired_output_tps_speedup"] == 1.25

    traffic = tmp_path / "t.jsonl"
    traffic.write_text('{"prompt_ids":[1,2],"output_len":2}\n')
    run = RunConfig(name="multimetric", traffic_file=str(traffic), workdir=str(tmp_path))
    app = create_app(run, local_tiers=set())
    with TestClient(app) as client:
        client.post("/workers/fast/heartbeat", json={"worker_id": "wf", "ready": True, "baseline_tps": 400.0, "crashes": 0})
        client.post("/workers/slow/heartbeat", json={"worker_id": "ws", "ready": True, "baseline_tps": 1000.0, "crashes": 0})

        j_fast = client.post("/jobs", json={"agent": "a1", "tier": "fast", "source": "SPEEDUP = 1.29\n", "note": "fast #1"}).json()["id"]
        cl_f = client.post("/workers/fast/claim", params={"worker_id": "wf", "wait_s": 0}).json()["job"]
        client.post("/workers/fast/complete", json={
            "id": j_fast, "worker_id": "wf", "attempt_id": cl_f["attempt_id"],
            "status": "done", "speedup": 1.29, "decode_tps": 516.0, "prefill_tps": 29500.0, "token_match": 0.98, "crashes": 0,
        }).raise_for_status()

        j_tput = client.post("/jobs", json={"agent": "a1", "tier": "slow", "source": "SPEEDUP = 1.10\n", "note": "tput"}).json()["id"]
        j_lat = client.post("/jobs", json={"agent": "a2", "tier": "slow", "source": "SPEEDUP = 0.99\n", "note": "low itl"}).json()["id"]
        j_nolat = client.post("/jobs", json={"agent": "a3", "tier": "slow", "source": "SPEEDUP = 1.05\n", "note": "no itl"}).json()["id"]

        for jid, sp, extra in (
            (j_tput, 1.10, {"metric": "agentx_total_tput_per_gpu", "unsafe_override": True, "repeats": [{}], "output_tput": 195.0, "itl_p90_ms": 5.7, "interactivity_p90": 175.4, "paired": {"decode_duration_speedup": 1.08}}),
            (j_lat, 0.99, {"metric": "agentx_total_tput_per_gpu", "unsafe_override": True, "repeats": [{}], "output_tput": 182.0, "itl_p90_ms": 4.0, "interactivity_p90": 250.0, "paired": {"decode_duration_speedup": 1.16}}),
            (j_nolat, 1.05, {"metric": "agentx_total_tput_per_gpu", "unsafe_override": True, "repeats": [{}], "output_tput": 188.0}),
        ):
            cl = client.post("/workers/slow/claim", params={"worker_id": "ws", "wait_s": 0}).json()["job"]
            client.post("/workers/slow/complete", json={
                "id": cl["id"], "worker_id": "ws", "attempt_id": cl["attempt_id"],
                "status": "done", "speedup": sp, "decode_tps": 1000.0 * sp, "token_match": 0.91, "extra": extra, "crashes": 0,
            }).raise_for_status()

        by_itl = client.get("/leaderboard", params={"tier": "slow", "sort_by": "itl_p90_ms"}).json()
        # None sorts last even in ascending mode!
        assert [r["id"] for r in by_itl] == [j_lat, j_tput, j_nolat]
        assert by_itl[0]["interactivity_p90"] == 250.0
        assert by_itl[0]["paired_decode_speedup"] == 1.16

        by_paired = client.get("/leaderboard", params={"tier": "slow", "sort_by": "paired_speedup"}).json()
        assert [r["id"] for r in by_paired] == [j_lat, j_tput, j_nolat]

        assert client.get("/leaderboard", params={"tier": "slow", "sort_by": "bogus_key"}).status_code == 400

        # Promote fast winner to slow & generate official 900s config
        import kiln.cli as cli_mod
        monkeypatch.setattr(cli_mod, "http", client)
        monkeypatch.setenv("KILN_SERVER", "http://testserver")
        official_yaml = tmp_path / "official_900s.yaml"
        main([
            "promote", "--top", "1", "--from-tier", "fast", "--to-tier", "slow",
            "-c", "configs/dsv41_flash_h20_sglang.yaml",
            "--write-official-config", str(official_yaml),
        ])
        assert official_yaml.exists()
        off_cfg = RunConfig.load(official_yaml)
        slow_cfg = next(t for t in off_cfg.tiers if t.name == "slow")
        assert slow_cfg.agentx["benchmark_duration_s"] == 900
        assert slow_cfg.agentx["unsafe_override"] is False
        cl_promoted = client.post("/workers/slow/claim", params={"worker_id": "ws", "wait_s": 0}).json()["job"]
        assert "SPEEDUP = 1.29" in cl_promoted["source"]
        assert "Promote fast job" in cl_promoted["note"]


def test_tier_watchdog_timeout_and_dead_engine_restart_and_shutdown(tmp_path, monkeypatch):
    import signal
    import subprocess
    import threading
    import time
    from kiln.engine.base import shutdown_process_group
    from kiln.engine.mock import MockEngine

    traffic = tmp_path / "t.jsonl"
    traffic.write_text('{"prompt_ids":[1,2],"output_len":2}\n')
    tier = TierConfig(name="fast", timeout_s=0.15)
    run = RunConfig(name="watchdog", traffic_file=str(traffic), workdir=str(tmp_path), tiers=[tier])
    mgr = jobs.JobManager(run, [Request([1, 2], 2)], local_tiers={"fast"})
    worker = mgr.tiers["fast"]
    worker.wait_until_ready()
    eng0 = worker.engine

    # 1. Hung measure hits watchdog TimeoutError and restarts engine
    release = threading.Event()
    monkeypatch.setattr(eng0, "measure", lambda reqs, repeats, trace: release.wait(timeout=2.0) or MeasureResult(1.0, 1.0, 1.0))
    hung_job = mgr.submit("a1", "fast", "SPEEDUP = 1.1\n", "hangs")
    worker.execute_job(hung_job)
    release.set()
    assert hung_job.status == "crashed"
    assert "timed out" in (hung_job.error or "").lower()
    assert worker.engine is not eng0

    # 2. Dead engine skips warm recovery and triggers restart
    eng1 = worker.engine
    alive = [True]
    monkeypatch.setattr(eng1, "is_alive", lambda: alive[0])
    restarts: list[object] = []
    orig_restart = worker._restart_engine

    def tracked_restart():
        restarts.append(worker.engine)
        orig_restart()

    monkeypatch.setattr(worker, "_restart_engine", tracked_restart)
    def boom_then_die(path):
        alive[0] = False
        raise RuntimeError("minor non-fatal exception")

    monkeypatch.setattr(eng1, "hot_reload", boom_then_die)
    dead_job = mgr.submit("a1", "fast", "SPEEDUP = 1.2\n", "dead")
    worker.execute_job(dead_job)
    assert dead_job.status == "crashed"
    assert restarts == [eng1]
    assert worker.engine is not eng1

    # 3. shutdown_process_group escalates SIGTERM -> SIGKILL, waits, and survives ProcessLookupError
    signals_sent: list[tuple[int, int]] = []
    wait_calls: list[float | None] = []

    class FakeProc:
        pid = 999999
        _polled = None

        def poll(self):
            return self._polled

        def wait(self, timeout=None):
            wait_calls.append(timeout)
            if len(wait_calls) == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            self._polled = -9
            return -9

    monkeypatch.setattr("kiln.engine.base.os.killpg", lambda pid, sig: signals_sent.append((pid, sig)))
    shutdown_process_group(FakeProc(), timeout=1.0)
    assert signals_sent == [(999999, signal.SIGTERM), (999999, signal.SIGKILL)]
    assert len(wait_calls) == 2

    def raise_lookup(pid, sig):
        raise ProcessLookupError

    wait_calls.clear()
    monkeypatch.setattr("kiln.engine.base.os.killpg", raise_lookup)
    shutdown_process_group(FakeProc(), timeout=1.0)
    assert len(wait_calls) == 2


def test_strip_sensitive_env_and_launcher_isolation(tmp_path, monkeypatch):
    from kiln.agents.launcher import _strip_sensitive_env, launch
    from kiln.config import AgentConfig, AgentSlotConfig

    env = {
        "PATH": "/usr/bin",
        "KUBECONFIG": "/Users/me/.kube/config",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "SCHED_KUBECONFIG": "/tmp/sched",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }
    stripped = _strip_sensitive_env(dict(env))
    assert stripped["KUBECONFIG"] == "/dev/null"
    assert "SSH_AUTH_SOCK" not in stripped
    assert "SCHED_KUBECONFIG" not in stripped
    assert "AWS_SECRET_ACCESS_KEY" not in stripped
    assert stripped["PATH"] == "/usr/bin"

    kept = _strip_sensitive_env(dict(env), allowed_keys={"KUBECONFIG", "SSH_AUTH_SOCK"})
    assert kept["KUBECONFIG"] == "/Users/me/.kube/config"
    assert kept["SSH_AUTH_SOCK"] == "/tmp/ssh-agent.sock"
    assert "SCHED_KUBECONFIG" not in kept

    # Verify launch() respects isolate_credentials=True vs False
    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    monkeypatch.setattr("kiln.agents.launcher.REPO", repo)
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh.sock")
    monkeypatch.setenv("KUBECONFIG", "/tmp/host_kube")

    captured_envs: list[dict[str, str]] = []

    class FakeP:
        pid = 123
        def poll(self): return 0
        def send_signal(self, sig): pass

    def fake_spawn(cmd, ws, slot_env, prompt_text):
        captured_envs.append(slot_env)
        (ws / "agent.log").touch()
        return FakeP(), open(ws / "agent.log", "ab")

    monkeypatch.setattr("kiln.agents.launcher._spawn_agent", fake_spawn)
    run_iso = RunConfig(
        name="iso", workdir=str(tmp_path),
        agents=AgentConfig(slots=[AgentSlotConfig(provider="codex")], isolate_credentials=True, respawn_on_exit=False),
    )
    launch(run_iso)
    assert captured_envs[-1]["KUBECONFIG"] == "/dev/null"
    assert "SSH_AUTH_SOCK" not in captured_envs[-1]

    run_no_iso = RunConfig(
        name="no_iso", workdir=str(tmp_path),
        agents=AgentConfig(slots=[AgentSlotConfig(provider="codex")], isolate_credentials=False, respawn_on_exit=False),
    )
    launch(run_no_iso)
    assert captured_envs[-1]["KUBECONFIG"] == "/tmp/host_kube"
    assert captured_envs[-1]["SSH_AUTH_SOCK"] == "/tmp/ssh.sock"
