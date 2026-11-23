import threading

from fastapi.testclient import TestClient

from kiln.config import RunConfig, TierConfig
from kiln.engine.mock import MockEngine
from kiln.server import jobs
from kiln.server.app import create_app


def test_lifespan_stops_shared_workers_and_closes_each_local_owner_once(tmp_path, monkeypatch):
    run = RunConfig(workdir=str(tmp_path), tiers=[
        TierConfig(name="p", phase="prefill"),
        TierConfig(name="d", phase="decode", share_engine="p"),
        TierConfig(name="remote"),
    ], micro={"enabled": False})
    app = create_app(run, local_tiers={"p", "d"})
    workers = app.state.mgr.tiers
    closed = []
    monkeypatch.setattr(MockEngine, "shutdown", lambda engine: closed.append(engine.tier.name))
    with TestClient(app):
        assert workers["p"].ready.wait(2)
        assert workers["d"].ready.wait(2)
    assert closed == ["p"]
    assert all(not w.thread.is_alive() and not w.ready.is_set() for w in workers.values())
    assert all(w.engine is None for w in workers.values())


def test_shutdown_wakes_boot_retry_and_keeps_queued_jobs(tmp_path, monkeypatch):
    attempted = threading.Event()

    def fail_start(self):
        attempted.set()
        raise RuntimeError("unavailable")

    monkeypatch.setattr(MockEngine, "start", fail_start)
    cfg = RunConfig(workdir=str(tmp_path), tiers=[TierConfig(name="fast")], micro={"enabled": False})
    app = create_app(cfg)
    mgr = app.state.mgr
    with TestClient(app):
        assert attempted.wait(2)
        job = mgr.submit("agent", "fast", "SPEEDUP = 1")
    assert not mgr.tiers["fast"].thread.is_alive()
    restored = jobs.JobManager(cfg, mgr.requests)
    assert restored.tiers["fast"].q.get_nowait().id == job.id


def test_shutdown_finishes_current_job_without_starting_next(tmp_path, monkeypatch):
    entered = threading.Event()
    original_reload = MockEngine.hot_reload
    cfg = RunConfig(workdir=str(tmp_path), tiers=[TierConfig(name="fast")], micro={"enabled": False})
    app = create_app(cfg)
    mgr = app.state.mgr
    worker = mgr.tiers["fast"]

    def wait_for_shutdown(engine, path):
        entered.set()
        assert worker.stop_requested.wait(3)
        original_reload(engine, path)

    monkeypatch.setattr(MockEngine, "hot_reload", wait_for_shutdown)
    with TestClient(app):
        first = mgr.submit("agent", "fast", "SPEEDUP = 1")
        assert entered.wait(2)
        second = mgr.submit("agent", "fast", "SPEEDUP = 2")
    restored = jobs.JobManager(cfg, mgr.requests)
    assert restored.get_job(first.id)["status"] == "done"
    assert restored.get_job(second.id)["status"] == "queued"
    assert restored.tiers["fast"].q.get_nowait().id == second.id
