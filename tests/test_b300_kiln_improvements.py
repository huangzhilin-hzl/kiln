"""Regression & feature tests for Kiln B300 / DSpark-7 framework improvements."""
from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path

import pytest

from kiln.config import MicroConfig, RunConfig
from kiln.micro.counters import (
    detect_ptxas_toolchain,
    get_ptxas_version,
    inspect_ptxas_and_spills,
    resolve_ptxas_binary,
)
from kiln.micro.sandbox import MicroRunner


def _make_fake_ptxas(path: Path, version_line: str = "Cuda compilation tools, release 13.0, V13.0.48",
                     fail_msg: str | None = None) -> Path:
    body = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import sys
        if "--version" in sys.argv:
            print("ptxas: NVIDIA (R) Ptx optimizing assembler")
            print({version_line!r})
            raise SystemExit(0)
        fail = {fail_msg!r}
        if fail:
            sys.stderr.write(fail + "\\n")
            raise SystemExit(1)
        sys.stderr.write("ptxas info    : Compiling entry function 'fake_k' for 'sm_103a'\\n")
        sys.stderr.write("ptxas info    : Used 48 registers, 2048 bytes smem\\n")
        raise SystemExit(0)
    """)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_ptxas_resolution_precedence_and_version(tmp_path: Path):
    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    cuda_ptxas = _make_fake_ptxas(cuda_bin / "ptxas", "Cuda compilation tools, release 12.8, V12.8.1")
    triton_ptxas = _make_fake_ptxas(tmp_path / "triton_ptxas", "Cuda compilation tools, release 13.0, V13.0.48")
    explicit_ptxas = _make_fake_ptxas(tmp_path / "explicit_ptxas", "Cuda compilation tools, release 13.1, V13.1.0")

    env = {"CUDA_HOME": str(tmp_path / "cuda"), "TRITON_PTXAS_PATH": str(triton_ptxas), "PATH": ""}
    assert resolve_ptxas_binary(env=env) == str(triton_ptxas.resolve())
    assert resolve_ptxas_binary(ptxas_path=str(explicit_ptxas), env=env) == str(explicit_ptxas.resolve())
    assert resolve_ptxas_binary(env={"CUDA_HOME": str(tmp_path / "cuda"), "PATH": ""}) == str(cuda_ptxas.resolve())

    ver = get_ptxas_version(str(triton_ptxas))
    assert ver is not None and "release 13.0" in ver
    meta = detect_ptxas_toolchain(env=env)
    assert meta["ptxas_path"] == str(triton_ptxas.resolve())
    assert meta["triton_ptxas_path"] == str(triton_ptxas)
    assert "release 13.0" in (meta["ptxas_version"] or "")


def test_inspect_ptxas_reports_metadata_and_arch_failure_warning(tmp_path: Path):
    good_ptxas = _make_fake_ptxas(tmp_path / "ptxas_ok")
    bad_ptxas = _make_fake_ptxas(
        tmp_path / "ptxas_old",
        version_line="Cuda compilation tools, release 12.4, V12.4.99",
        fail_msg="ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'",
    )
    sample_ptx = ".version 8.7\n.target sm_103a\n.visible .entry fake_k() { .reg .f32 %f<16>; ret; }\n"

    ok_res = inspect_ptxas_and_spills(ptx_texts=[sample_ptx], ptxas_path=str(good_ptxas))
    assert ok_res["available"] is True
    assert ok_res["used_ptxas_bin"] is True
    assert ok_res["ptxas_path"] == str(good_ptxas.resolve())
    assert "release 13.0" in (ok_res["ptxas_version"] or "")
    assert ok_res["max_registers"] == 48

    bad_res = inspect_ptxas_and_spills(ptx_texts=[sample_ptx], ptxas_path=str(bad_ptxas))
    assert bad_res["used_ptxas_bin"] is False
    assert bad_res["ptxas_path"] == str(bad_ptxas.resolve())
    assert any("sm_103a" in w for w in bad_res["warnings"])


def test_micro_runner_propagates_triton_ptxas_path_and_records_toolchain(tmp_path: Path, monkeypatch):
    fake_ptxas = _make_fake_ptxas(tmp_path / "ptxas_b300")
    monkeypatch.setenv("KILN_EXT_CACHE_ROOT", str(tmp_path / "ext_cache"))
    run = RunConfig(
        name="micro-ptxas-test",
        workdir=str(tmp_path),
        micro=MicroConfig(ptxas_path=str(fake_ptxas), enable_ptxas=True),
    )
    runner = MicroRunner(run, tmp_path / "run_root")
    src = textwrap.dedent("""\
        import os
        def micro_bench(device):
            seen = os.environ.get("TRITON_PTXAS_PATH", "")
            ok = seen == %r
            return [{"m": 1, "correct": ok, "ref_ms": 2.0, "cand_ms": 1.0, "detail": seen}]
    """ % str(fake_ptxas.resolve()))
    rec = runner.run_candidate("agent_1", src, note="check ptxas parity")
    assert rec["status"] == "done"
    assert rec["all_correct"] is True
    assert rec["toolchain"]["ptxas_path"] == str(fake_ptxas.resolve())
    assert rec["toolchain"]["triton_ptxas_path"] == str(fake_ptxas.resolve())
    assert "release 13.0" in (rec["toolchain"]["ptxas_version"] or "")
    assert rec["ptxas"]["ptxas_path"] == str(fake_ptxas.resolve())


# --------------------------------------------------------------------------- Commit 2: Micro RW lock, timeout, ABBA
def test_micro_rw_lock_allows_parallel_numeric_and_exclusive_timing():
    import threading
    import time

    from kiln.micro.sandbox import _MicroRWLock

    lock = _MicroRWLock(max_readers=2)
    events: list[tuple[str, str, float]] = []
    ev_lock = threading.Lock()
    both_readers_in = threading.Barrier(2, timeout=2.0)
    release_readers = threading.Event()
    timing_started = threading.Event()

    def _reader(idx: int):
        with lock.acquire("numeric"):
            with ev_lock:
                events.append((f"r{idx}", "enter", time.monotonic()))
            both_readers_in.wait()
            release_readers.wait(timeout=2.0)
            with ev_lock:
                events.append((f"r{idx}", "exit", time.monotonic()))

    def _writer():
        with lock.acquire("timing"):
            timing_started.set()
            with ev_lock:
                events.append(("w", "enter", time.monotonic()))
            time.sleep(0.05)
            with ev_lock:
                events.append(("w", "exit", time.monotonic()))

    t1 = threading.Thread(target=_reader, args=(1,))
    t2 = threading.Thread(target=_reader, args=(2,))
    t1.start()
    t2.start()
    # Both readers enter concurrently; barrier releases once both are inside.
    time.sleep(0.05)
    tw = threading.Thread(target=_writer)
    tw.start()
    time.sleep(0.05)
    assert not timing_started.is_set(), "timing lock must wait for active numeric readers"
    release_readers.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    tw.join(timeout=2.0)
    assert timing_started.is_set()

    r_exits = [t for name, ev, t in events if name.startswith("r") and ev == "exit"]
    w_enter = next(t for name, ev, t in events if name == "w" and ev == "enter")
    assert max(r_exits) <= w_enter + 1e-4


def test_micro_abba_helper_and_per_call_timeout_clamping(tmp_path: Path, monkeypatch):
    import numpy as np
    from kiln.micro.sandbox import abba_time_callables, cold_l2_flush

    cold_l2_flush(1024, device="cpu")
    x = np.ones((16, 16), dtype=np.float32)
    order: list[str] = []

    def ref_fn(arr):
        order.append("A")
        return arr * 2.0

    def cand_fn(arr):
        order.append("B")
        return arr + arr

    res = abba_time_callables(ref_fn, cand_fn, x, device="cpu", warmup=0, pairs=2, flush_l2=True)
    assert order == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert res["ref_ms"] > 0 and res["cand_ms"] > 0 and res["speedup"] > 0

    monkeypatch.setenv("KILN_EXT_CACHE_ROOT", str(tmp_path / "ext_cache"))
    run = RunConfig(
        name="micro-timeout-test",
        workdir=str(tmp_path),
        micro=MicroConfig(timeout_s=5.0, max_timeout_s=12.0, enable_ptxas=False),
    )
    runner = MicroRunner(run, tmp_path / "run_root")

    # Timeout override is clamped to max_timeout_s and timing mode is propagated.
    fast_src = textwrap.dedent("""\
        import os, numpy as np
        def micro_bench(device):
            assert os.environ.get("KILN_MICRO_MODE") == "timing"
            x = np.ones((8, 8), dtype=np.float32)
            return [{"m": 8, "inputs": x, "ref_fn": lambda a: a * 2.0, "cand_fn": lambda a: a + a}]
    """)
    rec = runner.run_candidate("agent_1", fast_src, mode="timing", timeout_s=999.0)
    assert rec["status"] == "done"
    assert rec["mode"] == "timing"
    assert rec["timeout_s"] == 12.0

    # Short per-call timeout triggers timeout error cleanly.
    slow_src = textwrap.dedent("""\
        import time
        def micro_bench(device):
            time.sleep(3.0)
            return [{"m": 1, "correct": True}]
    """)
    rec_to = runner.run_candidate("agent_1", slow_src, mode="numeric", timeout_s=0.4)
    assert rec_to["status"] == "crashed"
    assert "0.4s" in (rec_to["error"] or "")


def test_micro_http_endpoint_accepts_mode_and_timeout(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient
    from kiln.bench.traffic import synth_traffic
    from kiln.server.app import create_app

    monkeypatch.setenv("KILN_EXT_CACHE_ROOT", str(tmp_path / "ext_cache"))
    traffic = tmp_path / "traffic.jsonl"
    synth_traffic(str(traffic), n=2, isl=16, osl=8)
    run = RunConfig(
        name="micro-api-test",
        workdir=str(tmp_path / "runs"),
        traffic_file=str(traffic),
        micro=MicroConfig(timeout_s=30.0, max_timeout_s=90.0, enable_ptxas=False),
    )
    app = create_app(run, local_tiers=set())
    with TestClient(app) as client:
        r = client.post(
            "/micro",
            json={
                "agent": "agent_1",
                "source": "def micro_bench(device): return [{'m': 1, 'correct': True, 'ref_ms': 2.0, 'cand_ms': 1.0}]",
                "mode": "timing",
                "timeout_s": 45.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert body["mode"] == "timing"
        assert body["timeout_s"] == 45.0


# --------------------------------------------------------------------------- Commit 3: AgentX spec-decode acceptance & collapse abort
def test_agentx_parses_server_metrics_csv_and_detects_collapse(tmp_path: Path):
    import json
    from kiln.bench.agentx import (
        agentx_result_to_extra,
        detect_spec_acceptance_collapse,
        parse_agentx_export,
        parse_server_metrics_csv,
    )

    exp_dir = tmp_path / "run_healthy"
    exp_dir.mkdir()
    export_json = exp_dir / "profile_export_aiperf.json"
    export_json.write_text(json.dumps({
        "total_token_throughput": {"avg": 79907.2},
        "output_token_throughput": {"avg": 676.2},
        "inter_token_latency": {"p50": 4.41, "p90": 8.6},
        "time_to_first_token": {"p50": 265.0},
        "active_decode_throughput": {"avg": 850.0},
        "benchmark_duration": {"avg": 900.0},
        "metadata": {"submission_valid": True},
    }), encoding="utf-8")
    (exp_dir / "server_metrics_export.csv").write_text(
        "Metric,Type,Unit,First,Last,Avg,Total\n"
        "vllm:spec_decode_num_accepted_tokens_total,counter,tokens,1000,22200,0,21200\n"
        "vllm:spec_decode_num_drafts_total,counter,drafts,500,10500,0,10000\n"
        "vllm:spec_decode_num_draft_tokens_total,counter,tokens,3500,73500,0,70000\n"
        "vllm:num_preemptions_total,counter,count,0,3,0,3\n"
        "vllm:kv_cache_usage_perc,gauge,ratio,0.1,0.4,0.28,0\n",
        encoding="utf-8",
    )

    srv = parse_server_metrics_csv(export_json)
    assert srv["spec_decode_accepted_tokens"] == pytest.approx(21200.0)
    assert srv["spec_decode_num_drafts"] == pytest.approx(10000.0)
    assert srv["spec_decode_acc_per_draft"] == pytest.approx(2.12)
    assert srv["spec_decode_mean_accept_len"] == pytest.approx(3.12)
    assert srv["spec_accept_rate"] == pytest.approx(21200.0 / 70000.0, rel=1e-4)
    assert srv["num_preemptions"] == pytest.approx(3.0)
    assert srv["kv_cache_usage_avg"] == pytest.approx(0.28)

    res = parse_agentx_export(export_json, num_gpus=1)
    assert res.spec_decode_acc_per_draft == pytest.approx(2.12)
    assert res.spec_decode_mean_accept_len == pytest.approx(3.12)
    extra = agentx_result_to_extra(res, [], unsafe_override=False)
    assert extra["spec_decode_acc_per_draft"] == pytest.approx(2.12)
    assert extra["spec_accept_length"] == pytest.approx(3.12)

    collapsed_ok, detail_ok = detect_spec_acceptance_collapse(2.05, 2.12, min_ratio=0.30)
    assert not collapsed_ok and detail_ok is None
    collapsed_bad, detail_bad = detect_spec_acceptance_collapse(0.401, 2.12, min_ratio=0.30)
    assert collapsed_bad and "0.401" in (detail_bad or "")


def test_agentx_rolling_spec_collapse_watchdog_aborts_early(tmp_path: Path):
    import time
    from kiln.bench.agentx import AgentXConfig, SpecAcceptanceCollapsed, run_agentx_profile

    fake_agentx = tmp_path / "fake_agentx"
    fake_agentx.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n", encoding="utf-8")
    fake_agentx.chmod(fake_agentx.stat().st_mode | stat.S_IXUSR)

    cfg = AgentXConfig(
        binary=str(fake_agentx),
        benchmark_duration_s=10,
        unsafe_override=True,
        spec_collapse_min_ratio=0.30,
        spec_collapse_window_s=0.1,
        spec_collapse_poll_s=0.05,
        spec_collapse_min_drafts=50,
    )
    calls = {"n": 0}

    def fake_metrics():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"accepted_tokens": 1000.0, "num_drafts": 500.0}
        # Collapsed delta: 40 accepted over 100 drafts = 0.40 vs 2.12 baseline (<30%)
        return {"accepted_tokens": 1040.0, "num_drafts": 600.0}

    t0 = time.monotonic()
    with pytest.raises(SpecAcceptanceCollapsed) as ei:
        run_agentx_profile(
            "http://127.0.0.1:19999",
            "dummy-model",
            "dummy-tok",
            cfg,
            tmp_path / "ax_out",
            num_gpus=1,
            timeout_s=15.0,
            baseline_spec_acc_per_draft=2.12,
            metrics_reader=fake_metrics,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"expected early abort, took {elapsed:.2f}s"
    assert ei.value.snapshot["observed_acc_per_draft"] == pytest.approx(0.4, abs=1e-3)
    assert ei.value.snapshot["window_drafts"] == 100


# --------------------------------------------------------------------------- Commit 4: Tunnel active health probe
def test_tunnel_loop_restarts_wedged_port_forward_on_failed_probes():
    from kiln.cli import run_tunnel_loop

    class _FakeProc:
        def __init__(self, idx: int):
            self.idx = idx
            self.terminated = False
            self._rc: int | None = None

        def poll(self):
            return self._rc

        def terminate(self):
            self.terminated = True
            self._rc = -15

        def wait(self, timeout=None):
            return self._rc

        def kill(self):
            self._rc = -9

    spawned: list[_FakeProc] = []
    clock = {"t": 0.0}
    probes = [True, False, False, False, True, False, False, False]
    probe_idx = {"i": 0}

    def fake_popen(cmd, env):
        p = _FakeProc(len(spawned))
        spawned.append(p)
        return p

    def fake_probe(port: int) -> bool:
        assert port == 18766
        i = probe_idx["i"]
        probe_idx["i"] += 1
        return probes[i] if i < len(probes) else False

    def fake_sleep(dt: float):
        clock["t"] += dt

    restarts = run_tunnel_loop(
        ["kubectl", "port-forward", "pod/x", "18766:18766"],
        {},
        local_port=18766,
        reconnect_delay=1.0,
        health_check=True,
        health_interval=2.0,
        health_failures=3,
        startup_grace_s=1.0,
        max_restarts=2,
        popen_fn=fake_popen,
        probe_fn=fake_probe,
        sleep_fn=fake_sleep,
        clock_fn=lambda: clock["t"],
    )
    assert restarts == 2
    assert len(spawned) == 2
    assert all(p.terminated for p in spawned), "both wedged port-forward processes must be terminated"


# --------------------------------------------------------------------------- Commit 5: Function-level AST certification & --cascade-ast
def test_extract_function_ast_hashes_is_docstring_invariant_and_body_sensitive():
    from kiln.patches import extract_function_ast_hashes

    src_a = textwrap.dedent("""\
        def quant_kernel(x):
            '''Docstring v1'''
            return x * 2  # comment 1
    """)
    src_b = textwrap.dedent("""\
        def quant_kernel(x):
            '''Docstring v2 changed'''
            return x * 2  # comment 2
    """)
    src_c = textwrap.dedent("""\
        def quant_kernel(x):
            return x * 3
    """)
    ha = extract_function_ast_hashes(src_a)
    hb = extract_function_ast_hashes(src_b)
    hc = extract_function_ast_hashes(src_c)
    assert ha["quant_kernel"] == hb["quant_kernel"]
    assert ha["quant_kernel"] != hc["quant_kernel"]


def test_certify_and_cascade_ast_invalidation_via_api(tmp_path: Path):
    import time
    from fastapi.testclient import TestClient
    from kiln.bench.traffic import synth_traffic
    from kiln.config import EngineConfig, TierConfig
    from kiln.server.app import create_app

    traffic = tmp_path / "traffic.jsonl"
    synth_traffic(str(traffic), n=4, isl=16, osl=8)
    run = RunConfig(
        name="ast-audit-test",
        engine=EngineConfig(kind="mock"),
        tiers=[TierConfig(name="fast", repeats=1), TierConfig(name="slow", repeats=1)],
        traffic_file=str(traffic),
        workdir=str(tmp_path / "runs"),
    )
    app = create_app(run, local_tiers={"fast"})

    buggy_fn = "def bad_fp4_quant(x):\n    return x * 0.5\n"
    fixed_fn = "def bad_fp4_quant(x):\n    return x * 1.0\n"
    src_base = buggy_fn + "SPEEDUP = 1.10\ndef make_forward(model): return model.forward\n"
    src_sibling = buggy_fn + "SPEEDUP = 1.15\ndef extra_win(): return 1\ndef make_forward(model): return model.forward\n"
    src_clean = fixed_fn + "SPEEDUP = 1.08\ndef make_forward(model): return model.forward\n"

    with TestClient(app) as client:
        # Submit 3 jobs on fast and wait for completion
        ids = []
        for idx, src in enumerate((src_base, src_sibling, src_clean)):
            r = client.post("/jobs", json={"agent": f"agent_{idx+1}", "tier": "fast", "source": src})
            assert r.status_code == 200
            ids.append(r.json()["id"])
        j_base, j_sib, j_clean = ids

        deadline = time.time() + 10.0
        while time.time() < deadline:
            statuses = [client.get(f"/jobs/{jid}").json()["status"] for jid in ids]
            if all(st == "done" for st in statuses):
                break
            time.sleep(0.05)
        assert [client.get(f"/jobs/{jid}").json()["status"] for jid in ids] == ["done", "done", "done"]

        # Queue a sibling job on slow (remote/unclaimed so it stays queued!)
        r_q = client.post("/jobs", json={"agent": "agent_2", "tier": "slow", "source": src_sibling})
        assert r_q.status_code == 200
        j_queued_slow = r_q.json()["id"]

        # Certify the clean job's function AST
        rc = client.post(
            f"/jobs/{j_clean}/certify",
            json={"agent": "auditor_1", "func": "bad_fp4_quant", "scope": "M=1..1024", "note": "bit-exact"},
        )
        assert rc.status_code == 200
        cert_data = rc.json()
        assert cert_data["certificate"]["func"] == "bad_fp4_quant"
        assert len(cert_data["certificate"]["ast_sha256"]) == 64

        lb = client.get("/leaderboard", params={"tier": "fast"}).json()
        clean_row = next(row for row in lb if row["id"] == j_clean)
        assert clean_row["certified"] is True
        assert clean_row["certified_count"] == 1

        # Withdraw j_base with --cascade-ast bad_fp4_quant: must invalidate j_sib & cancel j_queued_slow, sparing j_clean!
        ri = client.post(
            f"/jobs/{j_base}/invalidate",
            json={"reason": "bad_fp4_quant drops upper bits", "cascade": True, "cascade_ast": ["bad_fp4_quant"]},
        )
        assert ri.status_code == 200
        inv_ids = set(ri.json()["invalidated"])
        assert {j_base, j_sib, j_queued_slow}.issubset(inv_ids)
        assert j_clean not in inv_ids
        assert client.get(f"/jobs/{j_sib}").json()["status"] == "invalidated"
        assert client.get(f"/jobs/{j_queued_slow}").json()["status"] == "cancelled"
        assert client.get(f"/jobs/{j_clean}").json()["status"] == "done"


# --------------------------------------------------------------------------- Commit 6: Slow Pareto leaderboard & multi-metric sorting
def test_leaderboard_pareto_frontier_and_extended_sort_keys(tmp_path: Path):
    from fastapi.testclient import TestClient
    from kiln.bench.traffic import synth_traffic
    from kiln.config import EngineConfig, TierConfig
    from kiln.server.app import create_app
    from kiln.server.jobs import Job

    traffic = tmp_path / "traffic.jsonl"
    synth_traffic(str(traffic), n=2, isl=16, osl=8)
    run = RunConfig(
        name="pareto-lb-test",
        engine=EngineConfig(kind="mock"),
        tiers=[TierConfig(name="slow", repeats=1)],
        traffic_file=str(traffic),
        workdir=str(tmp_path / "runs"),
    )
    app = create_app(run, local_tiers=set())
    mgr = app.state.mgr

    def _add_slow_job(jid: str, speedup: float, extra: dict):
        jdir = mgr.jobs_dir / jid
        jdir.mkdir(parents=True, exist_ok=True)
        cpath = jdir / "candidate.py"
        cpath.write_text("def make_forward(m): return m.forward\n", encoding="utf-8")
        job = Job(
            id=jid,
            agent="agent_1",
            tier="slow",
            note=jid,
            candidate_path=str(cpath),
            status="done",
            speedup=speedup,
            decode_tps=78000.0 * speedup,
            token_match=1.0,
            max_logit_abs_diff=0.0,
            extra={
                "metric": "agentx_total_tput_per_gpu",
                "submission_valid": True,
                "repeats": [{"total_tput_per_gpu": 78000.0 * speedup}],
                "benchmark_duration_s": 900.0,
                **extra,
            },
        )
        mgr.jobs[jid] = job

    # j_tput: best overall speedup/output_tput
    _add_slow_job("j_tput", 1.0238, {
        "output_tput": 676.2, "active_decode_tps": 845.0, "itl_p50_ms": 4.46, "itl_p90_ms": 8.65,
        "interactivity_p90": 115.6, "ttft_p50_ms": 270.0, "spec_decode_acc_per_draft": 2.123,
        "paired": {"decode_duration_speedup": 1.03, "paired_output_tps_speedup": 1.035},
    })
    # j_itl: Pareto winner on ITL p50 / p90 & active_decode_tps
    _add_slow_job("j_itl", 1.0150, {
        "output_tput": 668.0, "active_decode_tps": 867.2, "itl_p50_ms": 4.387, "itl_p90_ms": 8.414,
        "interactivity_p90": 118.85, "ttft_p50_ms": 275.0, "spec_decode_acc_per_draft": 2.110,
        "paired": {"decode_duration_speedup": 1.05, "paired_output_tps_speedup": 1.06},
    })
    # j_ttft: Pareto winner on TTFT p50
    _add_slow_job("j_ttft", 1.0204, {
        "output_tput": 672.0, "active_decode_tps": 840.0, "itl_p50_ms": 4.50, "itl_p90_ms": 8.70,
        "interactivity_p90": 114.9, "ttft_p50_ms": 259.27, "spec_decode_acc_per_draft": 2.135,
        "paired": {"decode_duration_speedup": 1.02, "paired_output_tps_speedup": 1.025},
    })
    # j_dominated: strictly worse than j_tput on EVERY objective
    _add_slow_job("j_dominated", 1.0050, {
        "output_tput": 650.0, "active_decode_tps": 820.0, "itl_p50_ms": 4.80, "itl_p90_ms": 9.10,
        "interactivity_p90": 109.9, "ttft_p50_ms": 290.0, "spec_decode_acc_per_draft": 2.08,
        "paired": {"decode_duration_speedup": 1.00, "paired_output_tps_speedup": 1.00},
    })

    with TestClient(app) as client:
        by_ttft = client.get("/leaderboard", params={"tier": "slow", "sort_by": "ttft_p50_ms"}).json()
        assert by_ttft[0]["id"] == "j_ttft"
        assert by_ttft[0]["ttft_p50_ms"] == pytest.approx(259.27)
        assert by_ttft[0]["benchmark_duration_s"] == pytest.approx(900.0)

        by_spec = client.get("/leaderboard", params={"tier": "slow", "sort_by": "spec_acc_per_draft"}).json()
        assert by_spec[0]["id"] == "j_ttft"
        assert by_spec[0]["spec_decode_acc_per_draft"] == pytest.approx(2.135)

        by_paired_out = client.get("/leaderboard", params={"tier": "slow", "sort_by": "paired_output_tps_speedup"}).json()
        assert by_paired_out[0]["id"] == "j_itl"
        assert by_paired_out[0]["paired_output_tps_speedup"] == pytest.approx(1.06)

        pareto_rows = client.get("/leaderboard", params={"tier": "slow", "pareto": "true"}).json()
        pareto_ids = {r["id"] for r in pareto_rows}
        assert pareto_ids == {"j_tput", "j_itl", "j_ttft"}
        assert "j_dominated" not in pareto_ids
        best_map = {r["id"]: set(r["pareto_best_on"]) for r in pareto_rows}
        assert "speedup" in best_map["j_tput"]
        assert "itl_p50_ms" in best_map["j_itl"] and "active_decode_tps" in best_map["j_itl"]
        assert "ttft_p50_ms" in best_map["j_ttft"]


# --------------------------------------------------------------------------- Commit 7: Profiler union_busy, warmup/steady split, UUID warning, idle auto-promote
def test_profiler_diff_exposes_union_busy_and_bubble_delta():
    from kiln.engine.vllm_profiler import analyze_trace_events, diff_traces, format_trace_report

    # Two overlapped streams: sum = 2000us (1000+1000), union busy = 1500us (0..1500), span = 2000us -> bubble = 25%
    base_events = [
        {"ph": "X", "cat": "kernel", "name": "k_a", "ts": 0.0, "dur": 1000.0, "tid": 1},
        {"ph": "X", "cat": "kernel", "name": "k_b", "ts": 500.0, "dur": 1000.0, "tid": 2},
        {"ph": "X", "cat": "kernel", "name": "k_c", "ts": 1900.0, "dur": 100.0, "tid": 1},
    ]
    cand_events = [
        {"ph": "X", "cat": "kernel", "name": "k_a", "ts": 0.0, "dur": 800.0, "tid": 1},
        {"ph": "X", "cat": "kernel", "name": "k_b", "ts": 400.0, "dur": 800.0, "tid": 2},
        {"ph": "X", "cat": "kernel", "name": "k_c", "ts": 1300.0, "dur": 100.0, "tid": 1},
    ]
    base_tr = analyze_trace_events(base_events, trace_path="base.json")
    cand_tr = analyze_trace_events(cand_events, trace_path="cand.json")
    td = diff_traces(cand_tr, base_tr)
    assert td["base_union_busy_ms"] == pytest.approx(1.6)
    assert td["cand_union_busy_ms"] == pytest.approx(1.3)
    assert td["delta_union_busy_ms"] == pytest.approx(-0.3)
    assert td["union_busy_speedup"] == pytest.approx(1.6 / 1.3, rel=1e-3)
    assert td["delta_bubble_pct"] < 0
    rep = format_trace_report(cand_tr, base_tr)
    assert "union_busy=" in rep and "bubble_delta=" in rep


def test_warmup_vs_steady_split_and_dynamic_torch_library_warning(tmp_path: Path):
    from kiln.engine.base import split_warmup_steady_runs
    from kiln.patches import detect_dynamic_torch_library_namespaces, inspect_candidate_source

    ws = split_warmup_steady_runs([2200.0, 2500.0, 2520.0])
    assert ws["cold_repeat_tps"] == pytest.approx(2200.0)
    assert ws["steady_decode_tps"] == pytest.approx(2510.0)
    assert ws["warmup_gap_pct"] == pytest.approx(100.0 * (2510.0 - 2200.0) / 2510.0, abs=0.05)

    bad_src = textwrap.dedent("""\
        import uuid, torch
        _lib = torch.library.Library(f"kiln_{uuid.uuid4().hex}", "DEF")
        def make_forward(m): return m.forward
    """)
    good_src = textwrap.dedent("""\
        import torch
        _lib = torch.library.Library("kiln_static_ops", "DEF")
        def make_forward(m): return m.forward
    """)
    warns_bad = detect_dynamic_torch_library_namespaces(bad_src)
    assert len(warns_bad) == 1 and "FXGraphCache" in warns_bad[0]
    assert inspect_candidate_source(bad_src)["warnings"] == warns_bad
    assert detect_dynamic_torch_library_namespaces(good_src) == []


def test_idle_slow_tier_auto_promotes_top_unpromoted_fast_candidate(tmp_path: Path):
    import time
    from kiln.bench.traffic import synth_traffic
    from kiln.config import EngineConfig, TierConfig
    from kiln.server.jobs import JobManager

    traffic = tmp_path / "traffic.jsonl"
    synth_traffic(str(traffic), n=4, isl=16, osl=8)
    run = RunConfig(
        name="idle-autopromote-test",
        engine=EngineConfig(kind="mock"),
        tiers=[
            TierConfig(name="fast", repeats=2),
            TierConfig(
                name="slow",
                repeats=1,
                idle_auto_promote_from="fast",
                idle_auto_promote_after_s=10.0,
                idle_auto_promote_require_certified=True,
            ),
        ],
        traffic_file=str(traffic),
        workdir=str(tmp_path / "runs"),
    )
    mgr = JobManager(run, [], local_tiers={"fast"})
    # Seed two completed fast jobs: j1 higher speedup (uncertified), j2 certified
    src1 = "SPEEDUP = 1.25\ndef make_forward(m): return m.forward\n"
    src2 = "SPEEDUP = 1.18\ndef kernel_ok(x): return x\ndef make_forward(m): return m.forward\n"
    j1 = mgr.submit("agent_1", "fast", src1, note="uncertified high")
    j2 = mgr.submit("agent_2", "fast", src2, note="certified clean")
    for j, sp in ((j1, 1.25), (j2, 1.18)):
        j.status = "done"
        j.speedup = sp
        j.decode_tps = 1000.0 * sp
    mgr.certify_job(j2.id, agent="auditor_1", func="kernel_ok", scope="all M")

    # Before idle threshold: nothing promoted
    now0 = mgr.tiers["slow"]._last_job_finished_mono
    assert mgr.maybe_auto_promote_idle_tier("slow", now_mono=now0 + 2.0) is None

    # Past idle threshold: certified j2 gets promoted to slow (j1 skipped because require_certified=True)
    promoted = mgr.maybe_auto_promote_idle_tier("slow", now_mono=now0 + 15.0)
    assert promoted is not None
    assert promoted.tier == "slow"
    assert promoted.parent_id == j2.id
    assert promoted.extra.get("auto_promoted_from") == j2.id

    # Second call while promoted is queued must NOT duplicate-promote
    assert mgr.maybe_auto_promote_idle_tier("slow", now_mono=now0 + 30.0) is None


# --------------------------------------------------------------------------- Commit 8: Launcher doctor checks & stale-agent watchdog
def test_doctor_validates_codex_model_cache_and_offline_paths(tmp_path: Path, monkeypatch):
    import json
    from types import SimpleNamespace
    import httpx
    from kiln.agents.launcher import _validate_offline_model_paths, doctor_slots
    from kiln.config import AgentConfig, AgentSlotConfig, EngineConfig

    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/usr/bin/env python3\nimport sys\nprint('codex 0.1')\n", encoding="utf-8")
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)

    cache_file = tmp_path / "models_cache.json"
    cache_file.write_text(json.dumps({"models": [{"slug": "gpt-5.4"}, {"slug": "gpt-5.3-codex"}]}), encoding="utf-8")

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None))
    run_bad_model = RunConfig(
        name="doc-codex",
        workdir=str(tmp_path),
        agents=AgentConfig(slots=[
            AgentSlotConfig(provider="codex", command=f"{fake_codex} exec --model gpt-5.4 -", model="gpt-5.4"),
            AgentSlotConfig(provider="codex", command=f"{fake_codex} exec --model gpt-6.2-sol -", model="gpt-6.2-sol"),
        ]),
    )
    rep = doctor_slots(run_bad_model, server_url="http://127.0.0.1:18765", probe_cli=False, codex_models_cache=cache_file)
    assert rep["ok"] is False
    assert rep["slots"][0]["ok"] is True and rep["slots"][0]["model_cache_ok"] is True
    assert rep["slots"][1]["ok"] is False and rep["slots"][1]["model_cache_ok"] is False
    assert "gpt-6.2-sol" in (rep["slots"][1]["error"] or "")

    # Offline / local path validation
    run_missing_local = RunConfig(
        name="doc-local",
        engine=EngineConfig(kind="vllm_server", model="/nonexistent/local/model/dir"),
    )
    errs = _validate_offline_model_paths(run_missing_local, env={})
    assert len(errs) == 1 and "does not exist" in errs[0]

    run_offline_hub = RunConfig(
        name="doc-offline",
        engine=EngineConfig(kind="vllm_server", model="thinkingmachines/Inkling-2-Mini"),
    )
    errs_off = _validate_offline_model_paths(
        run_offline_hub, env={"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(tmp_path / "empty_hf")}
    )
    assert len(errs_off) == 1 and "HF_HUB_OFFLINE" in errs_off[0]

    cached_dir = tmp_path / "hf_ok" / "models--thinkingmachines--Inkling-2-Mini"
    cached_dir.mkdir(parents=True)
    assert _validate_offline_model_paths(
        run_offline_hub, env={"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(tmp_path / "hf_ok")}
    ) == []


def test_stale_agent_watchdog_terminates_and_respawns_hung_slot(tmp_path: Path):
    import json
    from kiln.agents.launcher import _check_and_handle_stale_slot

    class _HungProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.terminated = False
            self._rc: int | None = None

        def poll(self):
            return self._rc

        def terminate(self):
            self.terminated = True
            self._rc = -15

        def wait(self, timeout=None):
            return self._rc

        def kill(self):
            self._rc = -9

    ws = tmp_path / "agent_1"
    ws.mkdir()
    log_path = ws / "agent.log"
    old_log = open(log_path, "ab")
    old_proc = _HungProc(101)
    new_proc = _HungProc(202)
    spawned: list[str] = []

    def fake_spawn(cmd, ws_arg, env, prompt_text):
        spawned.append(prompt_text)
        return new_proc, open(ws_arg / "agent.log", "ab")

    slot = {
        "name": "agent_1",
        "ws": ws,
        "cmd": ["codex", "exec", "-"],
        "provider": "codex",
        "model": "gpt-5.4",
        "role": "explorer",
        "env": {},
        "prompt": "BASE_PROMPT",
        "proc": old_proc,
        "log": old_log,
        "started_at": 1000.0,
        "last_activity_at": 1000.0,
        "last_heartbeat_at": 1000.0,
        "last_log_bytes": 0,
        "failures": 0,
        "quota_retries": 0,
        "stale_restarts": 0,
        "wake_at": None,
        "active": True,
    }

    # Not yet stale
    assert _check_and_handle_stale_slot(
        slot, now=1500.0, stale_timeout_s=1800.0, respawn_on_exit=True, deadline=9999.0, spawn_fn=fake_spawn
    ) is None
    assert not old_proc.terminated

    # Past stale_timeout_s -> terminated and respawned
    action = _check_and_handle_stale_slot(
        slot, now=2900.0, stale_timeout_s=1800.0, respawn_on_exit=True, deadline=9999.0, spawn_fn=fake_spawn
    )
    assert action == "respawned"
    assert old_proc.terminated is True
    assert old_log.closed is True
    assert slot["proc"] is new_proc
    assert slot["stale_restarts"] == 1
    assert len(spawned) == 1 and "Continuation" in spawned[0]
    st = json.loads((ws / "status.json").read_text(encoding="utf-8"))
    assert st["pid"] == 202 and st["stale_restarts"] == 1
    slot["log"].close()
