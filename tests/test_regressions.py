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
