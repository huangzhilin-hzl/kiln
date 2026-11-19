"""kiln CLI (used by humans and by agents)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

from kiln.server.auth import auth_headers


# Talk to the kiln server directly; ignore HTTP(S)_PROXY env vars (they break localhost).
http = httpx.Client(trust_env=False, timeout=30, headers=auth_headers())


def _url() -> str:
    return os.environ.get("KILN_SERVER", "http://127.0.0.1:8765").replace("://0.0.0.0:", "://127.0.0.1:")


def _agent(a) -> str:
    return a.agent or os.environ.get("KILN_AGENT", "human")


def _print(x) -> None:
    print(json.dumps(x, indent=2))


def cmd_serve(a):
    import uvicorn
    from kiln.config import RunConfig
    from kiln.server.app import create_app
    run = RunConfig.load(a.config)
    cfg_host, port = run.server_url.split("//")[1].rstrip("/").split(":")
    host = a.host or cfg_host
    if host not in ("localhost", "127.0.0.1", "::1") and not auth_headers():
        raise SystemExit("Set KILN_TOKEN before serving on a non-loopback address")
    local_tiers = set(a.tiers.split(",")) if a.tiers else None
    uvicorn.run(create_app(run, local_tiers=local_tiers), host=host, port=int(port), log_level="warning")


def cmd_worker(a):
    from kiln.config import RunConfig
    from kiln.server.jobs import run_remote_worker
    run = RunConfig.load(a.config)
    server = a.server or os.environ.get("KILN_SERVER") or run.client_url
    run_remote_worker(run, tier_name=a.tier, server_url=server)


def cmd_submit(a):
    src = open(a.file).read()
    r = http.post(f"{_url()}/jobs", json={"agent": _agent(a), "tier": a.tier,
                                           "source": src, "note": a.note})
    r.raise_for_status()
    jid = r.json()["id"]
    if not a.wait:
        _print(r.json())
        return
    while True:
        j = http.get(f"{_url()}/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            j.pop("candidate_path", None)
            _print(j)
            return
        time.sleep(2)


def cmd_job(a):
    _print(http.get(f"{_url()}/jobs/{a.id}").json())


def cmd_status(a):
    _print(http.get(f"{_url()}/status").json())


def cmd_leaderboard(a):
    _print(http.get(f"{_url()}/leaderboard", params={"tier": a.tier, "k": a.k}).json())


def cmd_board(a):
    if a.post:
        _print(http.post(f"{_url()}/board", json={"agent": _agent(a), "text": a.post}).json())
        return
    for m in http.get(f"{_url()}/board", params={"since": a.since, "limit": a.limit}).json():
        print(f"[{time.strftime('%H:%M:%S', time.localtime(m['ts']))}] {m['agent']}: {m['text']}")


def cmd_agents(a):
    from kiln.agents.launcher import launch
    from kiln.config import RunConfig
    launch(RunConfig.load(a.config))


def cmd_costmodel(a):
    from kiln.costmodel.model import CHIPS, MODELS, Workload, rank
    if a.profile:
        from kiln.costmodel.probe import HardwareProfile
        prof = HardwareProfile.load(a.profile)
        chip = prof.to_chip_spec()
        cc = prof.calibrated_chip
        tag = "mock" if prof.is_mock else "measured"
        peaks = cc.get("peak_tflops") if isinstance(cc.get("peak_tflops"), dict) else {}
        effs = cc.get("flops_eff") if isinstance(cc.get("flops_eff"), dict) else {"bf16": cc.get("flops_eff")}
        t_parts = []
        for k_dt, lbl in (("bf16", "BF16"), ("fp8", "FP8"), ("fp4", "FP4")):
            if peaks.get(k_dt):
                e_val = effs.get(k_dt)
                t_parts.append(f"{lbl}={peaks[k_dt]:,.0f}" + (f"({e_val:.2f})" if e_val is not None else ""))
        if not t_parts:
            bf16 = (prof.gemm.get("dtypes", {}).get("bf16") or {}).get("peak_tflops", 0.0)
            t_parts.append(f"BF16={bf16:,.1f}")
        hbm = prof.bandwidth.get("hbm_peak_achieved_gbps", 0.0)
        print(f"[profile: {chip.name} ({tag}) | {' '.join(t_parts)} TFLOP/s "
              f"| HBM={hbm:,.0f} GB/s (eff={cc.get('bw_eff', 0):.2f}) "
              f"| link_lat={chip.link_latency * 1e6:.1f}us]")
    else:
        chip = CHIPS[a.chip]
    w = Workload(isl=a.isl, osl=a.osl, cache_hit=a.cache_hit,
                 ttft_sla_s=a.ttft, decode_tps_sla=a.tps)
    rows = rank(MODELS[a.model], chip, w, max_chips=a.max_chips, k=a.k)
    if not rows:
        print("no admissible config meets the SLA")
    for r in rows:
        print(f"{r['config']:<28} N={r['n']:<3} TPM/chip={r['tpm_per_chip']:>12,.0f} "
              f"Cp={r['prefill_batch']:<4} Cd={r['decode_batch']:<5} "
              f"TTFT={r['ttft_s']:.3f}s TPS/user={r['decode_tps_user']:.0f}")


def cmd_probe(a):
    from pathlib import Path
    from kiln.costmodel.probe import HardwareProfile, probe_hardware
    if a.show:
        prof = HardwareProfile.load(a.show)
    else:
        gpus = [int(x) for x in a.gpus.split(",") if x.strip()] if a.gpus else None
        prof = probe_hardware(
            suite=a.suite, gpus=gpus, no_comm=a.no_comm,
            mock_chip=a.mock, require_gpu=a.require_gpu,
        )
        out = prof.save(a.out)
        print(f"[kiln probe] saved {out}", file=sys.stderr)
        if a.md:
            md_path = Path(a.md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(prof.to_markdown(), encoding="utf-8")
            print(f"[kiln probe] saved {md_path}", file=sys.stderr)
    if a.json:
        _print(prof.to_dict())
    else:
        print(prof.to_markdown(), end="")


def cmd_hw(a):
    from pathlib import Path
    from kiln.costmodel.probe import HardwareProfile
    prof = None
    if a.profile:
        prof = HardwareProfile.load(a.profile)
    else:
        try:
            res = http.get(f"{_url()}/hardware").json()
            if res.get("available") and res.get("profile"):
                prof = HardwareProfile.from_dict(res["profile"])
        except Exception:
            pass
        if prof is None and Path("runs/hw_profile.json").exists():
            prof = HardwareProfile.load("runs/hw_profile.json")
    if prof is None:
        raise SystemExit("No hardware profile found (run `kiln probe` first or pass --profile)")
    if a.json:
        _print(prof.to_dict())
    else:
        print(prof.to_markdown(), end="")


def cmd_agentx_export(a):
    from kiln.bench.agentx import export_offline
    s = export_offline(a.out, tokenizer=a.tokenizer, hf_repo=a.hf_repo, num_traces=a.traces,
                       turns_per_trace=a.turns, seed=a.seed, max_context=a.max_context)
    print(json.dumps(s, indent=2))


def cmd_synth(a):
    from kiln.bench.traffic import synth_traffic
    synth_traffic(a.out, n=a.n, isl=a.isl, osl=a.osl)
    print(f"wrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kiln")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("serve", help="start persistent engines + job server")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--tiers", help="comma-separated local tiers to run here (e.g. 'fast' on node 0)")
    s.add_argument("--host", help="override bind host (e.g. 0.0.0.0 for multi-node)")
    s.set_defaults(f=cmd_serve)

    s = sp.add_parser("worker", help="run a persistent tier worker on a second GPU node")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--tier", default="slow", help="tier to serve on this node")
    s.add_argument("--server", help="central kiln server URL, e.g. http://<node0-ip>:8765")
    s.set_defaults(f=cmd_worker)

    s = sp.add_parser("submit", help="submit a candidate forward.py")
    s.add_argument("file"); s.add_argument("--tier", default="fast")
    s.add_argument("--note", default=""); s.add_argument("--agent")
    s.add_argument("--wait", action="store_true"); s.set_defaults(f=cmd_submit)

    s = sp.add_parser("job"); s.add_argument("id"); s.set_defaults(f=cmd_job)
    s = sp.add_parser("status"); s.set_defaults(f=cmd_status)

    s = sp.add_parser("leaderboard"); s.add_argument("--tier", default="slow")
    s.add_argument("-k", type=int, default=10); s.set_defaults(f=cmd_leaderboard)

    s = sp.add_parser("board"); s.add_argument("--post"); s.add_argument("--agent")
    s.add_argument("--since", type=float, default=0.0); s.add_argument("--limit", type=int, default=50)
    s.set_defaults(f=cmd_board)

    s = sp.add_parser("agents", help="launch N parallel coding agents")
    s.add_argument("-c", "--config", required=True); s.set_defaults(f=cmd_agents)

    s = sp.add_parser("costmodel", help="rank parallelism configs")
    s.add_argument("--model", default="qwen3-30b-a3b"); s.add_argument("--chip", default="B200")
    s.add_argument("-p", "--profile", help="calibrated hw_profile.json from `kiln probe` (overrides --chip)")
    s.add_argument("--isl", type=int, default=8192); s.add_argument("--osl", type=int, default=1024)
    s.add_argument("--cache-hit", type=float, default=0.0)
    s.add_argument("--ttft", type=float, default=1.5); s.add_argument("--tps", type=float, default=40)
    s.add_argument("--max-chips", type=int, default=64); s.add_argument("-k", type=int, default=8)
    s.set_defaults(f=cmd_costmodel)

    s = sp.add_parser("probe", help="measure GPU/node compute, bandwidth, MoE, launch & NCCL roofline")
    s.add_argument("-o", "--out", default="runs/hw_profile.json", help="output JSON profile path")
    s.add_argument("--md", help="optional path to also write Markdown report (e.g. runs/HARDWARE.md)")
    s.add_argument("--suite", choices=("quick", "standard", "deep"), default="deep")
    s.add_argument("--gpus", help="comma-separated GPU indices (sets CUDA_VISIBLE_DEVICES)")
    s.add_argument("--no-comm", action="store_true", help="skip multi-GPU NCCL benchmarks")
    s.add_argument("--mock", nargs="?", const="H20-3e", metavar="CHIP",
                   help="generate synthetic profile from CHIPS (e.g. H20-3e, B200, H100)")
    s.add_argument("--require-gpu", action="store_true", help="fail instead of CPU mock fallback if no GPU")
    s.add_argument("--show", metavar="FILE", help="render an existing hw_profile.json without re-probing")
    s.add_argument("--json", action="store_true", help="print raw JSON instead of Markdown")
    s.set_defaults(f=cmd_probe)

    s = sp.add_parser("hw", help="show active hardware & roofline profile (from server or local file)")
    s.add_argument("-p", "--profile", help="path to hw_profile.json")
    s.add_argument("--json", action="store_true")
    s.set_defaults(f=cmd_hw)

    s = sp.add_parser("synth-traffic", help="write a synthetic placeholder traffic file")
    s.add_argument("--out", default="traffic/sample.jsonl"); s.add_argument("-n", type=int, default=32)
    s.add_argument("--isl", type=int, default=1024); s.add_argument("--osl", type=int, default=256)
    s.set_defaults(f=cmd_synth)

    s = sp.add_parser("agentx-export", help="AgentX traces -> offline replay JSONL (fast tier)")
    s.add_argument("--tokenizer", required=True, help="target model path / HF id")
    s.add_argument("--out", default="traffic/agentx.jsonl")
    s.add_argument("--hf-repo", default="semianalysisai/cc-traces-weka-062126")
    s.add_argument("--traces", type=int, default=32); s.add_argument("--turns", type=int, default=4)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--max-context", type=int)
    s.set_defaults(f=cmd_agentx_export)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
