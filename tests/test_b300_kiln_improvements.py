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
