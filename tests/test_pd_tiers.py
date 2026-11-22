from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from kiln.bench.traffic import Request, synth_traffic
from kiln.config import EngineConfig, MicroConfig, RunConfig, TierConfig
from kiln.engine.sglang_ext import decode_rpc
from kiln.engine.sglang_offline import SGLangOfflineEngine
from kiln.engine.sglang_server import SGLangServerEngine
from kiln.server.app import create_app


def _wait_job(client: TestClient, jid: str, timeout: float = 10.0) -> dict:
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        row = client.get(f"/jobs/{jid}").json()
        if row["status"] not in ("queued", "running"):
            return row
        time.sleep(0.03)
    raise TimeoutError(f"Job {jid} did not finish in {timeout}s")


def test_tier_config_phase_metric_and_validation():
    t_p = TierConfig(name="fast_prefill", phase="prefill")
    assert t_p.resolved_metric == "prefill_tps"

    t_d = TierConfig(name="fast_decode", phase="decode")
    assert t_d.resolved_metric == "decode_tps"

    t_ttft = TierConfig(name="prefill_sla", phase="prefill", metric="ttft")
    assert t_ttft.resolved_metric == "ttft"

    with pytest.raises(ValueError, match="reserved"):
        TierConfig(name="micro")

    with pytest.raises(ValueError, match="Invalid tier name"):
        TierConfig(name="bad name!")

    with pytest.raises(ValueError, match="Duplicate tier name"):
        RunConfig(tiers=[TierConfig(name="fast_p"), TierConfig(name="fast_p")])


def test_sglang_offline_and_server_phase_only_measurement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen_max_tokens: list[int] = []
    seen_rpcs: list[dict] = []

    class FakeSGLEngine:
        def __init__(self, **kwargs):
            pass

        def update_weights_from_disk(self, model_path: str, load_format=None):
            rpc = decode_rpc(model_path)
            assert rpc is not None
            seen_rpcs.append(rpc)
            return True, json.dumps({"ok": True, "result": 0.05})

        def flush_cache(self):
            return SimpleNamespace(success=True)

        def generate(self, input_ids=None, sampling_params=None, return_logprob=False):
            sp = sampling_params[0] if isinstance(sampling_params, list) else sampling_params
            mx = int(sp["max_new_tokens"])
            seen_max_tokens.append(mx)
            if isinstance(input_ids[0], int):
                return {"text": "ok", "output_ids": [1] * mx, "meta_info": {"completion_tokens": mx}}
            return [
                {"text": "ok", "output_ids": [1] * mx, "meta_info": {"completion_tokens": mx}}
                for _ in input_ids
            ]

        def shutdown(self):
            pass

    fake_sglang = types.ModuleType("sglang")
    fake_sglang.Engine = FakeSGLEngine
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)

    cfg = EngineConfig(kind="sglang_offline", model="dummy/m", tensor_parallel_size=2)
    reqs = [Request(prompt_ids=[1, 2, 3, 4, 5, 6], output_len=16) for _ in range(3)]

    # 1. Offline prefill-only tier skips multi-token decode and passes 'prefill' role on hot_reload
    p_eng = SGLangOfflineEngine(cfg, TierConfig(name="fast_prefill", phase="prefill", repeats=3))
    p_eng.start()
    cand = tmp_path / "cand.py"
    cand.write_text("pass\n")
    p_eng.hot_reload(str(cand))
    assert seen_rpcs[-1] == {"method": "kiln_hot_reload", "args": [str(cand), "prefill"], "kwargs": {}}
    seen_max_tokens.clear()
    mp = p_eng.measure(reqs, repeats=3, trace=False)
    assert seen_max_tokens == [1, 1, 1, 1]  # 1 warmup + 3 prefill repeats, zero 16-tok decode calls!
    assert mp.decode_tps == 0.0
    assert mp.prefill_tps > 0.0
    assert len(mp.runs_prefill_tps) == 3
    assert mp.variance_pct >= 0.0
    p_eng.shutdown()

    # 2. HTTP server decode-only tier skips separate prefill timing and passes 'decode' role
    srv_rpcs: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/update_weights_from_disk":
            body = json.loads(request.content.decode())
            srv_rpcs.append(decode_rpc(body["model_path"]))
            return httpx.Response(200, json={"success": True, "message": json.dumps({"ok": True, "result": 0.1})})
        if request.url.path == "/flush_cache":
            return httpx.Response(200, json={"success": True})
        if request.url.path == "/generate":
            body = json.loads(request.content.decode())
            mx = int(body["sampling_params"]["max_new_tokens"])
            return httpx.Response(200, json={"output_ids": [9] * mx, "meta_info": {"completion_tokens": mx}})
        return httpx.Response(404)

    d_srv = SGLangServerEngine(
        EngineConfig(kind="sglang_server", model="dummy/m"),
        TierConfig(name="fast_decode", phase="decode", repeats=2),
    )
    d_srv.http = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    d_srv.hot_reload(str(cand))
    assert srv_rpcs[-1] == {"method": "kiln_hot_reload", "args": [str(cand), "decode"], "kwargs": {}}
    md = d_srv.measure(reqs, repeats=2, trace=False)
    assert md.decode_tps > 0.0
    assert md.prefill_tps == 0.0
    assert md.ttft_s == 0.0
    d_srv.http.close()


def test_end_to_end_decoupled_pd_tiers_and_cli_merge(tmp_path: Path, monkeypatch, capsys):
    traffic = tmp_path / "traffic.jsonl"
    synth_traffic(traffic, n=8, isl=128, osl=16)
    run = RunConfig(
        name="pd_search",
        traffic_file=str(traffic),
        workdir=str(tmp_path),
        micro=MicroConfig(enabled=False),
        tiers=[
            TierConfig(name="fast_prefill", phase="prefill", repeats=2, canary_every_n_jobs=1),
            TierConfig(name="fast_decode", phase="decode", repeats=2),
            TierConfig(name="prefill_ttft", phase="prefill", metric="ttft", repeats=2),
            TierConfig(name="slow", phase="both", repeats=2),
        ],
    )
    app = create_app(run, local_tiers={"fast_prefill", "fast_decode", "prefill_ttft", "slow"})

    cand_prefill = (
        'KILN_TAGS = ["mla"]\n'
        "SPEEDUP = 1.00\n"
        "PREFILL_SPEEDUP = 1.45\n"
        "def make_forward(model):\n"
        "    model.x = 1\n"
        "    return None\n"
    )
    cand_decode = (
        'KILN_TAGS = ["gemv"]\n'
        "SPEEDUP = 1.30\n"
        "PREFILL_SPEEDUP = 1.00\n"
        "def make_forward(model):\n"
        "    model.x = 2\n"  # intentionally overlaps `model.x` with cand_prefill!
        "    return None\n"
    )

    with TestClient(app) as client:
        for tname in ("fast_prefill", "fast_decode", "prefill_ttft", "slow"):
            assert app.state.mgr.tiers[tname].ready.wait(timeout=5)

        import kiln.cli as cli_mod

        monkeypatch.setattr(cli_mod, "http", client)
        monkeypatch.setenv("KILN_SERVER", "http://testserver")

        st = client.get("/status").json()
        by_name = {t["tier"]: t for t in st["tiers"]}
        assert by_name["fast_prefill"]["phase"] == "prefill"
        assert by_name["fast_prefill"]["metric"] == "prefill_tps"
        assert by_name["prefill_ttft"]["metric"] == "ttft"

        jp = client.post("/jobs", json={"agent": "ap", "tier": "fast_prefill", "source": cand_prefill}).json()["id"]
        jd = client.post("/jobs", json={"agent": "ad", "tier": "fast_decode", "source": cand_decode}).json()["id"]
        jt = client.post("/jobs", json={"agent": "ap", "tier": "prefill_ttft", "source": cand_prefill}).json()["id"]

        rp = _wait_job(client, jp)
        rd = _wait_job(client, jd)
        rt = _wait_job(client, jt)
        assert rp["status"] == "done" and rp["decode_tps"] == 0.0 and rp["prefill_tps"] > 0.0
        assert 1.38 <= rp["speedup"] <= 1.52
        assert by_name["fast_prefill"]["baseline_decode_tps"] > 10_000  # prefill baseline tok/s
        assert app.state.mgr.tiers["fast_prefill"].last_canary is not None

        assert rd["status"] == "done" and rd["prefill_tps"] == 0.0 and rd["decode_tps"] > 0.0
        assert 1.24 <= rd["speedup"] <= 1.36

        assert rt["status"] == "done" and 1.38 <= rt["speedup"] <= 1.52
        lb_ttft = client.get("/leaderboard", params={"tier": "prefill_ttft", "sort_by": "ttft_s"}).json()
        assert lb_ttft[0]["id"] == jt

        # Archive exposes ready-to-merge asymmetric P+D pair suggestions
        arch = client.get("/archive", params={"tier": "fast_prefill"}).json()
        assert arch["pd_pair_suggestions"]
        assert arch["pd_pair_suggestions"][0]["prefill_job_id"] == jp
        assert arch["pd_pair_suggestions"][0]["decode_job_id"] == jd

        # Merge P + D winners via CLI (even though both assign `model.x`) and submit to slow!
        capsys.readouterr()
        merged_out = tmp_path / "merged_pd.py"
        cli_mod.main([
            "merge",
            "--prefill", jp,
            "--decode", jd,
            "-o", str(merged_out),
            "--submit",
            "--tier", "slow",
            "--wait",
            "--agent", "synth",
        ])
        summary = json.loads(capsys.readouterr().out)
        assert summary["pd_pair"] == {"prefill": jp, "decode": jd}
        assert summary["parent_ids"] == [jp, jd]
        assert summary["submitted"]["status"] == "done"
        assert 1.24 <= summary["submitted"]["speedup"] <= 1.36
        assert summary["submitted"]["extra"]["bucket_speedups"]["prefill_large"] >= 1.40
