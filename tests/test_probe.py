import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kiln import cli
from kiln.agents import launcher
from kiln.config import RunConfig
from kiln.costmodel.model import MODELS, ParallelConfig, Workload, pass_time, rank
from kiln.costmodel.probe import HardwareProfile, probe_hardware
from kiln.server.app import create_app


def test_probe_suites_and_roundtrip(tmp_path: Path):
    h20 = probe_hardware(suite="deep", mock_chip="H20-3e", progress=False)
    assert h20.is_mock is True
    assert "hostname" not in h20.topology
    assert h20.gemm["dtypes"]["fp4"]["supported"] is True
    assert h20.gemm["dtypes"]["fp4"]["native_tc"] is False
    assert h20.moe["supported"] is True and len(h20.moe["by_tokens"]) == 6
    assert "8" in h20.comm["groups"]
    assert h20.overhead["cudagraph_per_kernel_us"] > 0

    out = h20.save(tmp_path / "hw_profile.json")
    loaded = HardwareProfile.load(out)
    assert loaded.to_dict() == h20.to_dict()

    md = loaded.to_markdown()
    assert "Ridge Point (FLOP/Byte)" in md
    assert "Grouped MoE GEMM" in md
    assert "Multi-GPU Collectives" in md
    assert "CUDA Graph replay" in md

    b200 = probe_hardware(suite="deep", mock_chip="B200", progress=False)
    assert b200.gemm["dtypes"]["fp4"]["native_tc"] is True

    quick = probe_hardware(suite="quick", mock_chip="H100", progress=False)
    assert quick.moe["supported"] is False
    assert quick.comm["groups"] == {}
    assert quick.overhead == {}


def test_probe_cpu_fallback_and_require_gpu(monkeypatch):
    from kiln.costmodel import probe as probe_mod

    monkeypatch.setattr(probe_mod, "_has_cuda", lambda: False)
    prof = probe_hardware(suite="quick", progress=False)
    assert prof.is_mock is True
    with pytest.raises(RuntimeError, match="No CUDA/ROCm GPU"):
        probe_hardware(suite="quick", require_gpu=True, progress=False)


def test_empirical_curves_affect_costmodel():
    prof = probe_hardware(suite="deep", mock_chip="H20-3e", progress=False)
    chip = prof.to_chip_spec()
    rows = rank(
        MODELS["deepseek-v4.1-flash"],
        chip,
        Workload(isl=8192, osl=1024, ttft_sla_s=3.5, decode_tps_sla=25),
        max_chips=16,
    )
    assert rows and all(r["n"] >= 8 for r in rows)

    # Inflating empirical TP8 all_reduce latency must penalize TP8 without changing TP1·DP8
    qwen = MODELS["qwen3-30b-a3b"]
    tp8 = ParallelConfig(attn_tp=8, attn_dp=1, moe_tp=8, moe_ep=1)
    dp8_ep8 = ParallelConfig(attn_tp=1, attn_dp=8, moe_tp=1, moe_ep=8)
    t_tp8_base = pass_time(qwen, tp8, chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    t_ep8_base = pass_time(qwen, dp8_ep8, chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)

    slow_ar_chip = prof.to_chip_spec()
    slow_ar_chip.empirical.allreduce_by_k[8] = [
        (b, lat + 500e-6) for b, lat in slow_ar_chip.empirical.allreduce_by_k[8]
    ]
    t_tp8_slow = pass_time(qwen, tp8, slow_ar_chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    t_ep8_slow = pass_time(qwen, dp8_ep8, slow_ar_chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    assert t_tp8_slow > t_tp8_base + 0.04   # 48 layers * 2 allreduces * 500 us ≈ +48 ms
    assert t_ep8_slow == pytest.approx(t_ep8_base)

    # Lowering MoE roofline efficiency slows MoE pass_time while leaving dense models unchanged
    low_moe_chip = prof.to_chip_spec()
    low_moe_chip.empirical.moe_roofline_eff_by_tokens = [(1, 0.25), (2048, 0.25)]
    t_moe_low = pass_time(qwen, tp8, low_moe_chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    assert t_moe_low > t_tp8_base

    dense = MODELS["llama-3.2-1b"]
    t_dense_base = pass_time(dense, tp8, chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    t_dense_low = pass_time(dense, tp8, low_moe_chip, batch=8, tokens_per_seq=1, ctx_len=4096, prefill=False)
    assert t_dense_low == pytest.approx(t_dense_base)


def test_cli_probe_costmodel_and_hw(tmp_path: Path, capsys):
    prof_path = tmp_path / "hw_profile.json"
    md_path = tmp_path / "HARDWARE.md"

    cli.main(["probe", "--mock", "B200", "-o", str(prof_path), "--md", str(md_path)])
    out = capsys.readouterr().out
    assert "# Hardware & Roofline Profile" in out
    assert prof_path.exists() and md_path.exists()

    cli.main(["costmodel", "--model", "qwen3-30b-a3b", "--profile", str(prof_path), "-k", "3"])
    cm_out = capsys.readouterr().out
    assert "[profile: B200 (mock)" in cm_out
    assert "TPM/chip=" in cm_out

    cli.main(["hw", "--profile", str(prof_path), "--json"])
    hw_json = json.loads(capsys.readouterr().out)
    assert hw_json["calibrated_chip"]["name"] == "B200"


def test_launcher_injects_hardware_into_agent_workspace(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    monkeypatch.setattr(launcher, "REPO", repo)

    prof = probe_hardware(suite="deep", mock_chip="H20-3e", progress=False)
    prof.save(tmp_path / "hw_profile.json")

    seen_prompts = []

    def fake_spawn(cmd, ws, env, prompt):
        seen_prompts.append(prompt)
        return SimpleNamespace(poll=lambda: 0, pid=999), io.BytesIO()

    monkeypatch.setattr(launcher, "_spawn_agent", fake_spawn)
    run = RunConfig(name="hwtest", workdir=str(tmp_path))
    run.agents.num_agents = 1
    run.agents.respawn_on_exit = False
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)

    launcher.launch(run)
    ws = tmp_path / "hwtest" / "agents" / "agent_1"
    assert (ws / "HARDWARE.md").exists()
    assert (ws / "hw_profile.json").exists()
    prompt_md = (ws / "PROMPT.md").read_text(encoding="utf-8")
    assert "## Hardware & Roofline Reference" in prompt_md
    assert "ridge" in prompt_md
    assert seen_prompts and "## Hardware & Roofline Reference" in seen_prompts[0]


def test_server_hardware_endpoint(tmp_path: Path):
    traffic = tmp_path / "t.jsonl"
    traffic.write_text('{"prompt_ids":[1,2,3],"output_len":2}\n')
    run = RunConfig(name="srv", workdir=str(tmp_path), traffic_file=str(traffic))
    with TestClient(create_app(run, local_tiers=set())) as client:
        r0 = client.get("/hardware").json()
        assert r0 == {"available": False, "profile": None}

        prof = probe_hardware(suite="quick", mock_chip="H100", progress=False)
        prof.save(tmp_path / "hw_profile.json")

        r1 = client.get("/hardware").json()
        assert r1["available"] is True
        assert r1["profile"]["calibrated_chip"]["name"] == "H100"


def test_parse_nvlink_topology_with_ansi_and_many_nics(monkeypatch):
    from kiln.costmodel import probe as probe_mod

    topo_sample = (
        "\x1b[4m\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tNIC0\tNIC1\x1b[0m\n"
        "GPU0\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tPXB\tSYS\n"
        "GPU10\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\n"
    )
    monkeypatch.setattr(probe_mod.subprocess, "check_output", lambda *a, **kw: topo_sample)
    kind, clique = probe_mod._parse_nvlink_topology(8)
    assert kind == "NVLink/NVSwitch (NV18)"
    assert clique == 8

    b300 = probe_hardware(suite="deep", mock_chip="B300", progress=False)
    assert b300.calibrated_chip["name"] == "B300"
    assert b300.gemm["dtypes"]["fp4"]["native_tc"] is True


def test_checked_in_b300_profile():
    path = Path("profiles/b300_hw_profile.json")
    raw = path.read_text(encoding="utf-8")
    for secret in ("hostname", "molou", "sigma", "aidc", "antgroup"):
        assert secret not in raw.lower()
    prof = HardwareProfile.load(path)
    assert prof.is_mock is False
    cc = prof.calibrated_chip
    assert cc["name"] == "B300"
    assert set(cc["peak_tflops"]) == {"bf16", "fp8", "fp4"}
    assert cc["peak_tflops"]["fp4"] > cc["peak_tflops"]["fp8"] > cc["peak_tflops"]["bf16"] > 0
    assert cc["spec_tflops"] == {"bf16": 2250.0, "fp8": 4500.0, "fp4": 13500.0}
    assert cc["link_bw_gbps"] == 900.0 and cc["link_bidir_bw_gbps"] == 1800.0
    assert all(0 < cc["flops_eff"][dt] <= 1.0 for dt in ("bf16", "fp8", "fp4"))
    assert prof.topology["interconnect"] == "NVLink/NVSwitch (NV18)"
    rows = rank(MODELS["deepseek-v4.1-flash"], prof.to_chip_spec(), Workload(), max_chips=8)
    assert rows and rows[0]["tpm_per_chip"] > 0
