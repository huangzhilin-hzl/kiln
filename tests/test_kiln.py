import time
from pathlib import Path

from kiln.bench.traffic import synth_traffic
from kiln.config import RunConfig
from kiln.costmodel.model import CHIPS, MODELS, Workload, rank
from kiln.server.jobs import JobManager
from kiln.bench.traffic import load_traffic


def _wait(job, timeout=10):
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < timeout:
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
