import json
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kiln.agents.launcher import describe_slots, launch
from kiln.bench.traffic import synth_traffic
from kiln.config import AgentConfig, AgentSlotConfig, RunConfig, TierConfig
from kiln.costmodel.probe import HardwareProfile
from kiln.engine.vllm_profiler import (
    analyze_trace_events,
    compute_roofline_attribution,
    diff_traces,
    format_trace_report,
)
from kiln.micro.counters import (
    inspect_ptxas_and_spills,
    parse_ncu_csv,
    parse_ptx_or_ptxas_text,
)
from kiln.micro.sandbox import execute_candidate_file
from kiln.micro.snapshots import SnapshotCollector, load_snapshots
from kiln.patches import (
    PatchConflictError,
    compose_candidates,
    inspect_candidate_source,
    kiln_patch,
    make_composite_forward,
)
from kiln.server.app import create_app


def _wait_job(client: TestClient, jid: str, timeout: float = 10.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        row = client.get(f"/jobs/{jid}").json()
        if row["status"] not in ("queued", "running"):
            return row
        time.sleep(0.05)
    raise TimeoutError(f"Job {jid} did not finish within {timeout}s")


def test_lineage_checkout_single_arg_diff_and_archive(tmp_path: Path, monkeypatch, capsys):
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=8, isl=64, osl=16)
    run = RunConfig(name="evo", traffic_file=str(traffic), workdir=str(tmp_path))
    app = create_app(run, local_tiers={"fast", "slow"})

    cand_a = (
        'KILN_TAGS = ["gemv"]\n'
        "SPEEDUP = 1.15\n"
        'BUCKET_SPEEDUPS = {"decode_small": 1.30, "decode_mid": 1.15, "prefill_large": 1.00}\n'
        "def patch_gemv(x):\n"
        "    return x\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    cand_b = (
        'KILN_TAGS = ["mla_rope"]\n'
        "SPEEDUP = 1.10\n"
        'BUCKET_SPEEDUPS = {"decode_small": 1.00, "decode_mid": 1.05, "prefill_large": 1.40}\n'
        "def patch_mla(x):\n"
        "    return x\n"
        "def make_forward(model):\n"
        "    return None\n"
    )

    with TestClient(app) as client:
        assert app.state.mgr.tiers["fast"].ready.wait(timeout=5)
        assert app.state.mgr.tiers["slow"].ready.wait(timeout=5)

        import kiln.cli as cli_mod
        monkeypatch.setattr(cli_mod, "http", client)
        monkeypatch.setenv("KILN_SERVER", "http://testserver")

        ja = client.post("/jobs", json={"agent": "a1", "tier": "fast", "source": cand_a, "note": "gemv opt"}).json()["id"]
        jb = client.post("/jobs", json={"agent": "a2", "tier": "fast", "source": cand_b, "note": "mla rope"}).json()["id"]
        assert _wait_job(client, ja)["status"] == "done"
        assert _wait_job(client, jb)["status"] == "done"

        # Checkout candidate A -> edit -> submit automatically picks up parent lineage
        ws = tmp_path / "agent_ws"
        ws.mkdir()
        fwd_path = ws / "forward.py"
        cli_mod.main(["checkout", ja, "-o", str(fwd_path)])
        capsys.readouterr()
        assert (ws / ".kiln_checkout.json").exists()
        assert "SPEEDUP = 1.15" in fwd_path.read_text()

        fwd_path.write_text(cand_a.replace("SPEEDUP = 1.15", "SPEEDUP = 1.22").replace("1.30", "1.35"))
        cli_mod.main(["submit", str(fwd_path), "--tier", "fast", "--note", "tuned gemv", "--wait", "--agent", "a1"])
        child = json.loads(capsys.readouterr().out)
        jc = child["id"]
        assert child["status"] == "done"
        assert child["parent_id"] == ja
        assert ja in child["parent_ids"]
        assert "gemv" in child["tags"]

        # Single-arg `kiln diff <child>` compares against parent
        cli_mod.main(["diff", jc])
        diff_single = capsys.readouterr().out
        assert "-SPEEDUP = 1.15" in diff_single and "+SPEEDUP = 1.22" in diff_single

        # Root job single-arg diff fails cleanly
        with pytest.raises(SystemExit):
            cli_mod.main(["diff", ja])

        # Lineage endpoints & CLI
        cli_mod.main(["lineage", jc])
        lin_c = json.loads(capsys.readouterr().out)
        assert lin_c["depth"] == 1
        assert [a["id"] for a in lin_c["ancestors"]] == [ja]
        lin_a = client.get(f"/jobs/{ja}/lineage").json()
        assert [c["id"] for c in lin_a["children"]] == [jc]

        # Promote preserves parent lineage & tags
        prom = client.post(f"/jobs/{jc}/promote", json={"target_tier": "slow"}).json()
        jp = prom["id"]
        prom_row = _wait_job(client, jp)
        assert prom_row["status"] == "done"
        assert prom_row["parent_id"] == jc
        assert "gemv" in prom_row["tags"]

        # Multi-island archive surfaces regime & submodule stepping stones + orthogonal merge suggestions
        cli_mod.main(["archive", "--tier", "fast", "-k", "3"])
        arch = json.loads(capsys.readouterr().out)
        assert arch["regimes"]["decode_small"]["top"][0]["id"] == jc
        assert arch["regimes"]["prefill_large"]["top"][0]["id"] == jb
        assert "gemv" in arch["submodules"] and "mla_rope" in arch["submodules"]
        assert arch["submodules"]["mla_rope"][0]["id"] == jb
        stone_ids = {s["id"] for s in arch["stepping_stones"]}
        assert {jc, jb}.issubset(stone_ids)
        assert arch["merge_suggestions"]
        top_pair = arch["merge_suggestions"][0]
        assert set(top_pair["job_ids"]) == {jc, jb}


def test_kiln_patch_composite_merge_conflicts_and_synthesizer_slot(tmp_path: Path, monkeypatch, capsys):
    # 1. Test @kiln_patch + make_composite_forward + LIFO rollback
    events: list[str] = []

    class GemvLayer:
        def forward(self, x):
            return x + 1

    class MlaLayer:
        def forward(self, x):
            return x * 2

    class FakeModel:
        def __init__(self):
            self.gemv = GemvLayer()
            self.mla = MlaLayer()

        def named_modules(self):
            yield "", self
            yield "gemv", self.gemv
            yield "mla", self.mla

        def forward(self, x):
            return self.mla.forward(self.gemv.forward(x))

    @kiln_patch("gemv", regime="M<=48", targets=["GemvLayer"])
    def _p_gemv(model):
        events.append("apply:gemv")
        orig = model.gemv.forward
        model.gemv.forward = types.MethodType(lambda self, x: x + 10, model.gemv)
        return lambda m: (setattr(m.gemv, "forward", orig), events.append("undo:gemv"))

    @kiln_patch("mla", regime="M>512", targets=["MlaLayer"])
    def _p_mla_fail(model):
        events.append("apply:mla_fail")
        raise RuntimeError("boom during second patch")

    mk_fail, _ = make_composite_forward([_p_gemv, _p_mla_fail])
    fm = FakeModel()
    with pytest.raises(RuntimeError, match="boom"):
        mk_fail(fm)
    assert events == ["apply:gemv", "apply:mla_fail", "undo:gemv"]
    assert fm.forward(1) == 4  # (1 + 1) * 2 restored cleanly

    # 2. Test compose_candidates isolation, multiplicative speedup, conflict detection, and CLI merge
    src_a = (
        "import types\n"
        "from kiln.patches import kiln_patch\n"
        "SPEEDUP = 1.10\n"
        'BUCKET_SPEEDUPS = {"decode_small": 1.25}\n'
        "def _helper(v):\n"
        "    return v + 10\n"
        "@kiln_patch('gemv', regime='M<=48', targets=['GemvLayer'])\n"
        "def apply_gemv(model):\n"
        "    for _, m in model.named_modules():\n"
        "        if isinstance(m, type(model.gemv)):\n"
        "            m.forward = types.MethodType(lambda self, x: _helper(x), m)\n"
        "    return None\n"
        "def make_forward(model):\n"
        "    apply_gemv(model)\n"
        "    return None\n"
        "def restore(model):\n"
        "    if hasattr(model.gemv, 'forward') and 'forward' in model.gemv.__dict__:\n"
        "        del model.gemv.__dict__['forward']\n"
    )
    src_b = (
        "import types\n"
        "from kiln.patches import kiln_patch\n"
        "SPEEDUP = 1.08\n"
        'BUCKET_SPEEDUPS = {"prefill_large": 1.30}\n'
        "def _helper(v):\n"
        "    return v * 5\n"
        "@kiln_patch('mla_rope', regime='M>512', targets=['MlaLayer'])\n"
        "def apply_mla(model):\n"
        "    for _, m in model.named_modules():\n"
        "        if isinstance(m, type(model.mla)):\n"
        "            m.forward = types.MethodType(lambda self, x: _helper(x), m)\n"
        "    return None\n"
        "def make_forward(model):\n"
        "    apply_mla(model)\n"
        "    return None\n"
        "def restore(model):\n"
        "    if hasattr(model.mla, 'forward') and 'forward' in model.mla.__dict__:\n"
        "        del model.mla.__dict__['forward']\n"
    )
    src_conflict = src_a.replace("SPEEDUP = 1.10", "SPEEDUP = 1.12")

    info_a = inspect_candidate_source(src_a)
    assert "gemv" in info_a["tags"] and "GemvLayer" in info_a["targets"]

    with pytest.raises(PatchConflictError):
        compose_candidates([("j1", src_a), ("j2", src_conflict)])

    composed = compose_candidates([("j1", src_a), ("j2", src_b)])
    assert set(composed["tags"]) >= {"gemv", "mla_rope"}
    ns: dict = {}
    exec(compile(composed["source"], "<composed>", "exec"), ns)
    assert abs(ns["SPEEDUP"] - (1.10 * 1.08)) < 1e-6
    assert abs(ns["BUCKET_SPEEDUPS"]["decode_small"] - (1.25 * 1.08)) < 1e-6
    assert abs(ns["BUCKET_SPEEDUPS"]["prefill_large"] - (1.10 * 1.30)) < 1e-6

    live_model = FakeModel()
    ns["make_forward"](live_model)
    assert live_model.forward(2) == (2 + 10) * 5
    ns["restore"](live_model)
    assert live_model.forward(2) == (2 + 1) * 2

    # End-to-end HTTP 409 conflict & `kiln merge --submit --wait`
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=6, isl=32, osl=8)
    run = RunConfig(name="merge", traffic_file=str(traffic), workdir=str(tmp_path))
    app = create_app(run, local_tiers={"fast"})
    with TestClient(app) as client:
        assert app.state.mgr.tiers["fast"].ready.wait(timeout=5)
        import kiln.cli as cli_mod
        monkeypatch.setattr(cli_mod, "http", client)
        monkeypatch.setenv("KILN_SERVER", "http://testserver")

        id_a = client.post("/jobs", json={"agent": "a1", "tier": "fast", "source": src_a}).json()["id"]
        id_b = client.post("/jobs", json={"agent": "a2", "tier": "fast", "source": src_b}).json()["id"]
        id_c = client.post("/jobs", json={"agent": "a3", "tier": "fast", "source": src_conflict}).json()["id"]
        for jid in (id_a, id_b, id_c):
            assert _wait_job(client, jid)["status"] == "done"

        r_409 = client.post("/ensembles/compose", json={"job_ids": [id_a, id_c]})
        assert r_409.status_code == 409

        merged_file = tmp_path / "merged_forward.py"
        capsys.readouterr()
        cli_mod.main(["merge", id_a, id_b, "-o", str(merged_file), "--submit", "--tier", "fast", "--wait"])
        out_json = json.loads(capsys.readouterr().out)
        assert merged_file.is_file()
        sub_job = out_json["submitted"]
        assert sub_job["status"] == "done"
        assert sub_job["speedup"] > 1.15
        assert set(sub_job["parent_ids"]) == {id_a, id_b}

    # 3. Test synthesizer_slot auto-assignment and prompt injection (while default remains False)
    cfg_default = AgentConfig(
        slots=[AgentSlotConfig(provider="codex", count=3)],
    )
    assert [cfg_default.resolve_slot(i).role for i in range(3)] == [None, None, None]

    cfg_synth = AgentConfig(
        slots=[AgentSlotConfig(provider="codex", count=4)],
        synthesizer_slot=True,
    )
    assert [cfg_synth.resolve_slot(i).role for i in range(4)] == [None, None, None, "synthesizer"]

    repo = tmp_path / "repo"
    (repo / "candidates").mkdir(parents=True)
    (repo / "candidates" / "baseline_forward.py").write_text("def make_forward(m): return None\n")
    monkeypatch.setattr("kiln.agents.launcher.REPO", repo)

    class FakeP:
        pid = 101
        def poll(self): return 0
        def send_signal(self, sig): pass

    monkeypatch.setattr(
        "kiln.agents.launcher._spawn_agent",
        lambda cmd, ws, env, pt: ((ws / "agent.log").touch(), (FakeP(), open(ws / "agent.log", "ab")))[1],
    )
    synth_run = RunConfig(
        name="synth_run",
        workdir=str(tmp_path),
        agents=AgentConfig(
            slots=[AgentSlotConfig(provider="codex", count=3)],
            synthesizer_slot=True,
            respawn_on_exit=False,
        ),
    )
    assert describe_slots(synth_run)[-1]["role"] == "synthesizer"
    launch(synth_run)
    p3 = (tmp_path / "synth_run" / "agents" / "agent_3" / "PROMPT.md").read_text()
    assert "Role: `synthesizer`" in p3 and "kiln merge" in p3


def test_trace_diff_stream_bubbles_and_roofline_attribution(tmp_path: Path):
    base_events = [
        {"cat": "kernel", "name": "cutlass_fp8_gemm", "ts": 0.0, "dur": 400.0, "args": {"stream": 7}},
        {"cat": "kernel", "name": "flashinfer_mla", "ts": 450.0, "dur": 300.0, "args": {"stream": 7}},
        {"cat": "kernel", "name": "nccl_all_reduce", "ts": 750.0, "dur": 250.0, "args": {"stream": 7}},
        {"cat": "kernel", "name": "tiny_rmsnorm", "ts": 1000.0, "dur": 10.0, "args": {"stream": 7}},
        {"cat": "kernel", "name": "tiny_rmsnorm", "ts": 1010.0, "dur": 10.0, "args": {"stream": 7}},
    ]
    cand_events = [
        {"cat": "kernel", "name": "cutlass_fp8_gemm", "ts": 0.0, "dur": 300.0, "args": {"stream": 7}},
        {"cat": "kernel", "name": "flashinfer_mla", "ts": 310.0, "dur": 250.0, "args": {"stream": 7}},
        # Concurrent collective on stream 8 overlapping main-stream compute
        {"cat": "kernel", "name": "mhc_multicast_allreduce", "ts": 200.0, "dur": 180.0, "args": {"stream": 8}},
    ]

    base_ana = analyze_trace_events(base_events, trace_path="base.trace.json")
    cand_ana = analyze_trace_events(cand_events, trace_path="cand.trace.json")
    assert base_ana is not None and cand_ana is not None
    assert base_ana.total_launches == 5
    assert cand_ana.total_launches == 3
    assert cand_ana.num_streams == 2
    assert cand_ana.multi_stream_overlap_us == pytest.approx(170.0)
    assert cand_ana.comm_contention_us == pytest.approx(170.0)
    assert cand_ana.comm_contention_pct > 25.0

    td = diff_traces(cand_ana, base_ana)
    assert td["has_baseline"] is True
    assert td["delta_launches"] == -2
    assert td["delta_kernel_ms"] < 0
    statuses = {k["name"]: k["status"] for k in td["kernel_deltas"]}
    assert statuses["mhc_multicast_allreduce"] == "new"
    assert statuses["nccl_all_reduce"] == "eliminated"
    assert statuses["cutlass_fp8_gemm"] == "faster"

    hw = HardwareProfile.load(Path(__file__).resolve().parents[1] / "profiles" / "h20_3e_hw_profile.json")
    rf = compute_roofline_attribution(
        cand_ana,
        hw,
        model_name="deepseek-v4.1-flash",
        tp_size=8,
        decode_tps=900.0,
        prefill_tps=50_000.0,
        batch_size=16,
    )
    assert rf is not None
    assert rf["hbm_peak_gbps"] == pytest.approx(4371.91)
    assert rf["achieved_decode_hbm_gbps"] > 0
    assert rf["hbm_roofline_pct"] > 0
    assert rf["link_latency_us"] == pytest.approx(12.96)
    assert "STREAM_CONTENTION" in rf["bottleneck_flags"]
    assert "COLLECTIVE_LATENCY_BOUND" in rf["bottleneck_flags"]

    rep = format_trace_report(cand_ana, base_ana, rf)
    assert "[Trace Diff vs Baseline]" in rep
    assert "[Streams & Bubbles]" in rep
    assert "[Roofline Gap]" in rep

    # End-to-end TierWorker populates extra.trace_diff & extra.roofline on MockEngine
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=8, isl=64, osl=16)
    hw_copy = tmp_path / "hw.json"
    hw.save(hw_copy)
    run = RunConfig(
        name="traced",
        traffic_file=str(traffic),
        workdir=str(tmp_path),
        hw_profile=str(hw_copy),
        engine={"model": "deepseek-v4.1-flash", "tensor_parallel_size": 8},
        tiers=[TierConfig(name="fast", repeats=2, trace_kernels=True)],
    )
    app = create_app(run, local_tiers={"fast"})
    with TestClient(app) as client:
        assert app.state.mgr.tiers["fast"].ready.wait(timeout=5)
        src = (
            "SPEEDUP = 1.20\n"
            f"MOCK_TRACE_EVENTS = {cand_events!r}\n"
            "def make_forward(model):\n    return None\n"
        )
        jid = client.post("/jobs", json={"agent": "a1", "tier": "fast", "source": src}).json()["id"]
        done = _wait_job(client, jid)
        assert done["status"] == "done"
        assert done["extra"]["trace_diff"]["has_baseline"] is True
        assert done["extra"]["roofline"]["hbm_peak_gbps"] == pytest.approx(4371.91)
        assert "[Trace Diff vs Baseline]" in done["kernel_report"]
        assert "[Roofline Gap]" in done["kernel_report"]


@pytest.mark.parametrize("use_reference_kernel", [False, True])
@pytest.mark.parametrize("candidate_increment", [1, 2])
def test_micro_in_place_kernels_start_each_trial_from_the_same_input(use_reference_kernel, candidate_increment):
    from kiln.micro.sandbox import _evaluate_case

    original = np.zeros(1, dtype=np.float32)
    reference_inputs = []
    candidate_inputs = []

    def reference(x):
        reference_inputs.append(x.item())
        x += 1
        return x

    def candidate(x):
        candidate_inputs.append(x.item())
        x += candidate_increment
        return x

    case = {"inputs": {"x": original}, "cand_fn": candidate}
    if use_reference_kernel:
        case["ref_fn"] = reference
    else:
        case["ref_output"] = np.ones_like(original)
    row = _evaluate_case(case, "cpu", atol=0, rtol=0)
    assert row["correct"] == (candidate_increment == 1)
    assert row["max_abs_diff"] == candidate_increment - 1
    assert original.item() == 0
    assert len(candidate_inputs) > 1
    assert set(candidate_inputs) == {0}
    if use_reference_kernel:
        assert reference_inputs == candidate_inputs


def test_micro_preserves_overlapping_strided_array_inputs():
    from kiln.micro.sandbox import _copy_inputs

    base = np.arange(16)
    a, b = base[14:1:-2], base[6:12:2]
    broadcast = np.broadcast_to(base[6:7], (4,))
    x, y, z = _copy_inputs((a, b, broadcast))
    assert [v.strides for v in (x, y, z)] == [v.strides for v in (a, b, broadcast)]
    assert np.array_equal(x, a) and np.array_equal(y, b)
    x[4] = 99
    assert y[0] == 99 and np.all(z == 99)
    assert not z.flags.writeable and base[6] == 6


def test_micro_preserves_nonleaf_tensor_views_and_inference_mode():
    torch = pytest.importorskip("torch")
    from kiln.micro.sandbox import _copy_inputs

    base = torch.arange(10.0, requires_grad=True) * 3
    sliced = base[1::2]
    x, y = _copy_inputs((base, sliced))
    assert not x.is_leaf and x.requires_grad and y.stride() == sliced.stride()
    x[1] += 2
    assert y[0].item() == 5 and base[1].item() == 3
    with torch.inference_mode():
        inference = torch.ones(3)
        assert _copy_inputs(inference).is_inference()
    assert _copy_inputs(inference).is_inference()


def test_micro_snapshots_lock_free_sandbox_padded_nan_ptxas_ncu(tmp_path: Path):
    # 1. Cold-boot SnapshotCollector on duck-typed multi-layer model + replay
    class DummyAttn:
        def forward(self, x, w):
            return x @ w

    class DummyMoE:
        def forward(self, x):
            return x * 2.0

    class DummyModel:
        def __init__(self):
            self.l0_attn = DummyAttn()
            self.l1_attn = DummyAttn()
            self.l0_moe = DummyMoE()

        def named_modules(self):
            yield "", self
            yield "layers.0.self_attn", self.l0_attn
            yield "layers.1.self_attn", self.l1_attn
            yield "layers.0.mlp.experts", self.l0_moe

        def forward(self, x, w):
            return self.l0_moe.forward(self.l1_attn.forward(self.l0_attn.forward(x, w), w))

    snap_dir = tmp_path / "snaps"
    m = DummyModel()
    col = SnapshotCollector(snap_dir, m_buckets=[1, 8, 64, 1024], max_snapshot_mb=8.0)
    attached = col.attach(m)
    assert {a["tag"] for a in attached} == {"attention", "moe"}
    # Only 1 representative layer per tag is hooked (layers.0, not layers.1)
    assert "forward" in m.l0_attn.__dict__
    assert "forward" not in m.l1_attn.__dict__

    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 16)).astype(np.float32)
    for mb in (1, 8, 64, 1024):
        x = rng.standard_normal((mb, 16)).astype(np.float32)
        m.forward(x, w)
    manifest = col.close()
    # Hooks removed cleanly on close()
    assert "forward" not in m.l0_attn.__dict__
    assert "forward" not in m.l0_moe.__dict__
    attn_snaps = load_snapshots(snap_dir, tag="attention")
    assert [s["m_bucket"] for s in attn_snaps] == [1, 8, 64, 1024]

    replay_cand = tmp_path / "replay_cand.py"
    replay_cand.write_text(
        'MICRO_SNAPSHOT = "attention"\n'
        "def micro_kernel(x, w):\n"
        "    return x @ w\n"
    )
    rep_res = execute_candidate_file(replay_cand, snapshot_dir=snap_dir)
    assert rep_res["all_correct"] is True
    assert [b["m"] for b in rep_res["buckets"]] == [1, 8, 64, 1024]

    # 2. Lock-free POST /micro completes immediately even while fast engine_lock is held
    traffic = tmp_path / "t.jsonl"
    synth_traffic(traffic, n=4, isl=32, osl=8)
    hw_src = Path(__file__).resolve().parents[1] / "profiles" / "h20_3e_hw_profile.json"
    run = RunConfig(
        name="microrun",
        traffic_file=str(traffic),
        workdir=str(tmp_path),
        hw_profile=str(hw_src),
        tiers=[TierConfig(name="fast", repeats=1)],
    )
    app = create_app(run, local_tiers={"fast"})
    with TestClient(app) as client:
        assert app.state.mgr.tiers["fast"].ready.wait(timeout=5)
        # MockEngine cold-boot should have populated micro_snapshots
        snaps_resp = client.get("/micro/snapshots").json()
        assert snaps_resp["count"] > 0

        fast_worker = app.state.mgr.tiers["fast"]
        locked_acquired = threading.Event()
        release_lock = threading.Event()

        def _hold_engine_lock():
            with fast_worker.engine_lock:
                locked_acquired.set()
                release_lock.wait(timeout=10.0)

        holder = threading.Thread(target=_hold_engine_lock, daemon=True)
        holder.start()
        assert locked_acquired.wait(timeout=2.0)

        clean_micro_src = (
            "import time\n"
            "import numpy as np\n"
            "def micro_bench(device):\n"
            "    x = np.ones((8, 64), dtype=np.float32)\n"
            "    def ref_fn(a):\n"
            "        time.sleep(0.004)\n"
            "        return a * 2.0\n"
            "    def cand_fn(a):\n"
            "        return a * 2.0\n"
            "    return [{'m': 8, 'inputs': x, 'ref_fn': ref_fn, 'cand_fn': cand_fn, 'bytes_moved': x.nbytes * 2}]\n"
            "def micro_padded_kernel(x, valid_rows):\n"
            "    out = np.zeros_like(x)\n"
            "    out[:valid_rows] = x[:valid_rows] * 2.0\n"
            "    return out\n"
        )
        t0 = time.monotonic()
        r_micro = client.post("/micro", json={"agent": "a1", "source": clean_micro_src, "note": "clean"})
        elapsed = time.monotonic() - t0
        release_lock.set()
        holder.join(timeout=2.0)

        assert r_micro.status_code == 200
        mres = r_micro.json()
        assert elapsed < 5.0
        assert mres["status"] == "done"
        assert mres["all_correct"] is True
        assert mres["padded_nan_ok"] is True
        assert mres["geomean_speedup"] > 1.0
        assert mres["ready_for_fast"] is True

        # 3. Padded-row NaN guard catches `0.0 * NaN = NaN` contamination
        nan_bug_src = (
            "import numpy as np\n"
            "def micro_bench(device):\n"
            "    return [{'m': 8, 'correct': True, 'ref_ms': 1.0, 'cand_ms': 0.5}]\n"
            "def micro_padded_kernel(x, valid_rows):\n"
            "    # Buggy: multiplies padded NaN rows by 0.0 and reduces into valid rows\n"
            "    bad_sum = (x * 0.0).sum(axis=0, keepdims=True)\n"
            "    return x + bad_sum\n"
        )
        r_nan = client.post("/micro", json={"agent": "a1", "source": nan_bug_src, "note": "nan bug"}).json()
        assert r_nan["padded_nan_ok"] is False
        assert r_nan["status"] == "incorrect"
        assert r_nan["ready_for_fast"] is False

        # 4. PTX spill blocks `ready_for_fast`
        spill_ptx_src = (
            "import numpy as np\n"
            'PTX_SOURCE = ".visible .entry kern() { .reg .b32 %r<48>; .local .align 4 .b8 __local_depot[64]; ret; }"\n'
            "def micro_bench(device):\n"
            "    return [{'m': 8, 'correct': True, 'ref_ms': 2.0, 'cand_ms': 0.5}]\n"
        )
        r_spill = client.post("/micro", json={"agent": "a1", "source": spill_ptx_src, "note": "spill"}).json()
        assert r_spill["ptxas"]["any_spills"] is True
        assert r_spill["ready_for_fast"] is False

    # 5. ptxas -v parser & Nsight Compute CSV parser (SwizzlePerf bank conflict hint)
    ptxas_diag = (
        "ptxas info    : Compiling entry function 'fused_mxfp8_gemv' for 'sm_90'\n"
        "ptxas info    : Function properties for fused_mxfp8_gemv\n"
        "    128 bytes stack frame, 32 bytes spill stores, 16 bytes spill loads\n"
        "ptxas info    : Used 160 registers, 16384 bytes smem\n"
    )
    parsed_ptxas = parse_ptx_or_ptxas_text(ptxas_diag)
    assert len(parsed_ptxas) == 1
    assert parsed_ptxas[0]["kernel"] == "fused_mxfp8_gemv"
    assert parsed_ptxas[0]["registers"] == 160
    assert parsed_ptxas[0]["spill_stores_bytes"] == 32
    assert parsed_ptxas[0]["spill_loads_bytes"] == 16
    assert parsed_ptxas[0]["has_spills"] is True

    agg = inspect_ptxas_and_spills(ptx_texts=[ptxas_diag], run_ptxas_bin=False)
    assert agg["any_spills"] is True and agg["max_registers"] == 160

    ncu_csv = "\n".join([
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"',
        '"0","fused_gemv_k","dram__bytes.sum.per_second","Gbyte/second","3420.5"',
        '"0","fused_gemv_k","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active","%","68.4"',
        '"0","fused_gemv_k","sm__warps_active.avg.pct_of_peak_sustained_active","%","74.2"',
        '"0","fused_gemv_k","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum","conflict","4096"',
        '"0","fused_gemv_k","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum","conflict","512"',
    ])
    ncu = parse_ncu_csv(ncu_csv)
    assert ncu["available"] is True
    assert ncu["hbm_gbps"] == pytest.approx(3420.5)
    assert ncu["tensor_core_pct"] == pytest.approx(68.4)
    assert ncu["achieved_occupancy_pct"] == pytest.approx(74.2)
    assert ncu["shared_bank_conflicts_total"] == pytest.approx(4608.0)
    assert "Swizzle" in (ncu["swizzle_hint"] or "")
