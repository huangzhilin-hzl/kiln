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
