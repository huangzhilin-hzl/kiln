"""kiln CLI (used by humans and by agents)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx


def _url() -> str:
    return os.environ.get("KILN_SERVER", "http://127.0.0.1:8765")


def _agent(a) -> str:
    return a.agent or os.environ.get("KILN_AGENT", "human")


def _print(x) -> None:
    print(json.dumps(x, indent=2))


def cmd_serve(a):
    import uvicorn
    from kiln.config import RunConfig
    from kiln.server.app import create_app
    run = RunConfig.load(a.config)
    host, port = run.server_url.split("//")[1].split(":")
    uvicorn.run(create_app(run), host=host, port=int(port), log_level="warning")


def cmd_submit(a):
    src = open(a.file).read()
    r = httpx.post(f"{_url()}/jobs", json={"agent": _agent(a), "tier": a.tier,
                                           "source": src, "note": a.note}, timeout=30)
    r.raise_for_status()
    jid = r.json()["id"]
    if not a.wait:
        _print(r.json())
        return
    while True:
        j = httpx.get(f"{_url()}/jobs/{jid}", timeout=30).json()
        if j["status"] not in ("queued", "running"):
            j.pop("candidate_path", None)
            _print(j)
            return
        time.sleep(2)


def cmd_job(a):
    _print(httpx.get(f"{_url()}/jobs/{a.id}", timeout=30).json())


def cmd_status(a):
    _print(httpx.get(f"{_url()}/status", timeout=30).json())


def cmd_leaderboard(a):
    _print(httpx.get(f"{_url()}/leaderboard", params={"tier": a.tier, "k": a.k}, timeout=30).json())


def cmd_board(a):
    if a.post:
        _print(httpx.post(f"{_url()}/board", json={"agent": _agent(a), "text": a.post}, timeout=30).json())
        return
    for m in httpx.get(f"{_url()}/board", params={"since": a.since, "limit": a.limit}, timeout=30).json():
        print(f"[{time.strftime('%H:%M:%S', time.localtime(m['ts']))}] {m['agent']}: {m['text']}")


def cmd_agents(a):
    from kiln.agents.launcher import launch
    from kiln.config import RunConfig
    launch(RunConfig.load(a.config))


def cmd_costmodel(a):
    from kiln.costmodel.model import CHIPS, MODELS, Workload, rank
    w = Workload(isl=a.isl, osl=a.osl, cache_hit=a.cache_hit,
                 ttft_sla_s=a.ttft, decode_tps_sla=a.tps)
    rows = rank(MODELS[a.model], CHIPS[a.chip], w, max_chips=a.max_chips, k=a.k)
    if not rows:
        print("no admissible config meets the SLA")
    for r in rows:
        print(f"{r['config']:<28} N={r['n']:<3} TPM/chip={r['tpm_per_chip']:>12,.0f} "
              f"Cp={r['prefill_batch']:<4} Cd={r['decode_batch']:<5} "
              f"TTFT={r['ttft_s']:.3f}s TPS/user={r['decode_tps_user']:.0f}")


def cmd_synth(a):
    from kiln.bench.traffic import synth_traffic
    synth_traffic(a.out, n=a.n, isl=a.isl, osl=a.osl)
    print(f"wrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kiln")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("serve", help="start persistent engines + job server")
    s.add_argument("-c", "--config", required=True); s.set_defaults(f=cmd_serve)

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
    s.add_argument("--isl", type=int, default=8192); s.add_argument("--osl", type=int, default=1024)
    s.add_argument("--cache-hit", type=float, default=0.0)
    s.add_argument("--ttft", type=float, default=1.5); s.add_argument("--tps", type=float, default=40)
    s.add_argument("--max-chips", type=int, default=64); s.add_argument("-k", type=int, default=8)
    s.set_defaults(f=cmd_costmodel)

    s = sp.add_parser("synth-traffic", help="write a synthetic placeholder traffic file")
    s.add_argument("--out", default="traffic/sample.jsonl"); s.add_argument("-n", type=int, default=32)
    s.add_argument("--isl", type=int, default=1024); s.add_argument("--osl", type=int, default=256)
    s.set_defaults(f=cmd_synth)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
