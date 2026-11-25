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
    extra = [x.strip() for x in (a.tiers or "").split(",") if x.strip()] or None
    run_remote_worker(
        run,
        tier_name=a.tier,
        server_url=server,
        extra_tiers=extra,
        include_shared=not a.no_shared,
    )


def cmd_pd_node(a):
    from kiln.config import RunConfig
    from kiln.engine.sglang_pd import build_pd_node_launch_spec

    run = RunConfig.load(a.config)
    cmd, full_env, overrides = build_pd_node_launch_spec(
        run,
        tier_name=a.tier,
        role=a.role,
        index=a.index,
        host_override=a.host,
        port_override=a.port,
    )
    if a.dry_run:
        _print({
            "tier": a.tier,
            "role": a.role,
            "index": a.index,
            "cmd": cmd,
            "env_overrides": overrides,
        })
        return
    os.execvpe(cmd[0], cmd, full_env)


def _read_checkout_lineage(file_path: str) -> list[str]:
    from pathlib import Path

    p = Path(file_path).resolve()
    meta_path = p.parent / ".kiln_checkout.json"
    if not meta_path.is_file():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        tgt = data.get("target")
        if tgt and Path(tgt).resolve() != p and Path(tgt).name != p.name:
            return []
        pids = data.get("parent_ids")
        if isinstance(pids, list):
            return [str(x).strip() for x in pids if str(x).strip()]
    except Exception:
        pass
    return []


def _write_checkout_meta(out_path, parent_ids: list[str], sha256: str | None = None, tags: list[str] | None = None) -> None:
    from pathlib import Path

    out = Path(out_path)
    meta_path = out.parent / ".kiln_checkout.json"
    meta_path.write_text(
        json.dumps(
            {
                "parent_ids": list(parent_ids),
                "target": str(out.resolve()),
                "sha256": sha256,
                "tags": list(tags or ()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _wait_for_job(jid: str) -> dict:
    while True:
        j = http.get(f"{_url()}/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            j.pop("candidate_path", None)
            return j
        time.sleep(2)


def cmd_submit(a):
    src = open(a.file).read()
    if a.tier == "micro":
        r = http.post(f"{_url()}/micro", json={"agent": _agent(a), "source": src, "note": a.note})
        r.raise_for_status()
        _print(r.json())
        return
    parent_ids = [x.strip() for x in (a.parent or "").split(",") if x.strip()]
    if not parent_ids:
        parent_ids = _read_checkout_lineage(a.file)
    tags = [x.strip() for x in (a.tags or "").split(",") if x.strip()]
    payload: dict = {"agent": _agent(a), "tier": a.tier, "source": src, "note": a.note}
    if parent_ids:
        payload["parent_id"] = parent_ids[0]
        payload["parent_ids"] = parent_ids
    if tags:
        payload["tags"] = tags
    if getattr(a, "trace", False):
        payload["force_trace"] = True
    if getattr(a, "supersede", False):
        payload["supersede"] = True
    r = http.post(f"{_url()}/jobs", json=payload)
    r.raise_for_status()
    jid = r.json()["id"]
    if not a.wait:
        _print(r.json())
        return
    _print(_wait_for_job(jid))


def cmd_job(a):
    _print(http.get(f"{_url()}/jobs/{a.id}").json())


def cmd_jobs(a):
    params: dict = {"limit": a.limit}
    if a.tier:
        params["tier"] = a.tier
    if a.status:
        params["status"] = a.status
    if a.agent:
        params["agent"] = a.agent
    r = http.get(f"{_url()}/jobs", params=params)
    r.raise_for_status()
    _print(r.json())


def cmd_summary(a):
    from collections import Counter

    st = http.get(f"{_url()}/status")
    st.raise_for_status()
    lb = http.get(f"{_url()}/leaderboard", params={"tier": a.tier, "k": a.k, "sort_by": a.sort})
    lb.raise_for_status()
    jl = http.get(f"{_url()}/jobs", params={"limit": 500})
    jl.raise_for_status()
    rows = jl.json()
    by_tier_status: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    for (t, s_val), cnt in Counter((r["tier"], r["status"]) for r in rows).items():
        by_tier_status[f"{t}:{s_val}"] = cnt
    for ag, cnt in Counter(r["agent"] for r in rows).items():
        by_agent[ag] = cnt
    _print({
        "status": st.json(),
        "job_counts": by_tier_status,
        "by_agent": by_agent,
        "leaderboard": lb.json(),
    })


def cmd_pull(a):
    from pathlib import Path
    r = http.get(f"{_url()}/jobs/{a.id}/source")
    r.raise_for_status()
    data = r.json()
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(data["source"])
        _print({k: v for k, v in data.items() if k != "source"} | {"saved": str(out)})
    else:
        print(data["source"], end="" if data["source"].endswith("\n") else "\n")


def cmd_checkout(a):
    from pathlib import Path

    r = http.get(f"{_url()}/jobs/{a.id}/source")
    r.raise_for_status()
    data = r.json()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data["source"], encoding="utf-8")
    _write_checkout_meta(out, [a.id], sha256=data.get("source_sha256"), tags=data.get("tags"))
    _print({
        "id": data["id"],
        "tier": data["tier"],
        "agent": data["agent"],
        "tags": data.get("tags", []),
        "parent_ids": [a.id],
        "saved": str(out),
    })


def cmd_diff(a):
    import difflib

    id_a = a.id_a
    id_b = a.id_b
    if id_b is None:
        rj = http.get(f"{_url()}/jobs/{id_a}")
        rj.raise_for_status()
        meta = rj.json()
        parent = meta.get("parent_id") or (meta.get("parent_ids") or [None])[0]
        if not parent:
            raise SystemExit(f"Job {id_a} has no parent_id; provide two job IDs (`kiln diff <id_a> <id_b>`)")
        id_b = id_a
        id_a = parent

    ra = http.get(f"{_url()}/jobs/{id_a}/source")
    ra.raise_for_status()
    rb = http.get(f"{_url()}/jobs/{id_b}/source")
    rb.raise_for_status()
    diff = difflib.unified_diff(
        ra.json()["source"].splitlines(keepends=True),
        rb.json()["source"].splitlines(keepends=True),
        fromfile=id_a,
        tofile=id_b,
    )
    sys.stdout.write("".join(diff))


def cmd_archive(a):
    r = http.get(f"{_url()}/archive", params={"tier": a.tier, "k": a.k})
    r.raise_for_status()
    data = r.json()
    if a.island:
        if a.island.startswith("regime:"):
            rkey = a.island.split(":", 1)[1]
            _print(data.get("regimes", {}).get(rkey))
        elif a.island.startswith("submodule:"):
            skey = a.island.split(":", 1)[1]
            _print(data.get("submodules", {}).get(skey))
        else:
            _print(data.get(a.island))
        return
    _print(data)


def cmd_lineage(a):
    r = http.get(f"{_url()}/jobs/{a.id}/lineage")
    r.raise_for_status()
    _print(r.json())


def cmd_merge(a):
    from pathlib import Path

    pd_mode = bool(getattr(a, "prefill_id", None) or getattr(a, "decode_id", None))
    if pd_mode:
        if not (a.prefill_id and a.decode_id) or a.ids:
            raise SystemExit("PD merge requires both --prefill <id> and --decode <id> (without positional job IDs)")
        r = http.post(
            f"{_url()}/ensembles/compose_pd",
            json={"prefill_job_id": a.prefill_id, "decode_job_id": a.decode_id},
        )
    else:
        if len(a.ids) < 2:
            raise SystemExit("Provide two or more job IDs to merge, or pass --prefill <id> --decode <id>")
        r = http.post(
            f"{_url()}/ensembles/compose",
            json={"job_ids": a.ids, "allow_target_overlap": a.allow_overlap},
        )
    r.raise_for_status()
    data = r.json()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data["source"], encoding="utf-8")
    _write_checkout_meta(out, data["parent_ids"], tags=data.get("tags"))
    summary: dict = {
        "saved": str(out),
        "parent_ids": data["parent_ids"],
        "tags": data.get("tags", []),
        "conflicts": data.get("conflicts", data.get("colocated_conflicts", [])),
    }
    if pd_mode:
        summary["pd_pair"] = {"prefill": a.prefill_id, "decode": a.decode_id}
    if a.submit:
        default_note = (
            f"PD pair prefill={a.prefill_id} + decode={a.decode_id}"
            if pd_mode
            else f"Merge {' + '.join(data['parent_ids'])}"
        )
        note = a.note.strip() or default_note
        sr = http.post(
            f"{_url()}/jobs",
            json={
                "agent": _agent(a),
                "tier": a.tier,
                "source": data["source"],
                "note": note,
                "parent_id": data["parent_ids"][0],
                "parent_ids": data["parent_ids"],
                "tags": data.get("tags", []),
            },
        )
        sr.raise_for_status()
        sub = sr.json()
        summary["submitted"] = _wait_for_job(sub["id"]) if a.wait else sub
    _print(summary)


def cmd_micro(a):
    from pathlib import Path

    src = Path(a.file).read_text(encoding="utf-8")
    if a.local:
        from kiln.config import RunConfig
        from kiln.micro.sandbox import MicroRunner

        run = RunConfig.load(a.config) if a.config else RunConfig()
        runner = MicroRunner(run, Path(run.workdir) / run.name)
        res = runner.run_candidate(
            _agent(a),
            src,
            note=a.note,
            snapshot_tag=a.snapshot,
            enable_ncu=True if a.ncu else None,
            enable_ptxas=False if a.no_ptxas else None,
            mode=a.mode,
            timeout_s=a.timeout,
        )
    else:
        payload = {
            "agent": _agent(a),
            "source": src,
            "note": a.note,
            "snapshot_tag": a.snapshot,
            "enable_ncu": True if a.ncu else None,
            "enable_ptxas": False if a.no_ptxas else None,
            "mode": a.mode,
            "timeout_s": a.timeout,
        }
        http_timeout = max(180.0, float(a.timeout) + 60.0) if a.timeout else 180.0
        r = http.post(f"{_url()}/micro", json=payload, timeout=http_timeout)
        r.raise_for_status()
        res = r.json()

    if a.promote_fast and res.get("ready_for_fast"):
        parent_ids = _read_checkout_lineage(a.file)
        target_tier = getattr(a, "promote_tier", None) or "fast"
        sub_payload: dict = {"agent": _agent(a), "tier": target_tier, "source": src, "note": a.note or f"micro:{res['id']}"}
        if parent_ids:
            sub_payload["parent_id"] = parent_ids[0]
            sub_payload["parent_ids"] = parent_ids
        sr = http.post(f"{_url()}/jobs", json=sub_payload)
        sr.raise_for_status()
        sub = sr.json()
        res["promoted_fast"] = _wait_for_job(sub["id"]) if a.wait else sub

    _print(res)


def cmd_micro_snapshots(a):
    params = {"tag": a.tag} if a.tag else {}
    r = http.get(f"{_url()}/micro/snapshots", params=params)
    r.raise_for_status()
    _print(r.json())


def cmd_cancel(a):
    r = http.post(f"{_url()}/jobs/{a.id}/cancel")
    r.raise_for_status()
    _print(r.json())


def cmd_withdraw(a):
    r = http.post(
        f"{_url()}/jobs/{a.id}/invalidate",
        json={"reason": a.reason, "cascade": not a.no_cascade},
    )
    r.raise_for_status()
    _print(r.json())


def cmd_status(a):
    _print(http.get(f"{_url()}/status").json())


def cmd_leaderboard(a):
    _print(http.get(f"{_url()}/leaderboard", params={"tier": a.tier, "k": a.k, "sort_by": a.sort}).json())


def write_official_agentx_config(src_path: str | Path, out_path: str | Path, duration_s: int = 900) -> Path:
    """Create an official `submission_valid=true` (`unsafe_override: false`, `>=900s`) copy of a RunConfig YAML."""
    import yaml
    from pathlib import Path
    from kiln.config import RunConfig

    if duration_s < 900:
        raise ValueError("Official AgentX duration must be at least 900 seconds")
    raw = yaml.safe_load(Path(src_path).read_text()) or {}
    updated_tiers = 0
    for tier in raw.get("tiers") or []:
        if isinstance(tier, dict) and isinstance(tier.get("agentx"), dict):
            tier["agentx"]["benchmark_duration_s"] = int(duration_s)
            tier["agentx"]["unsafe_override"] = False
            tier["timeout_s"] = max(float(tier.get("timeout_s") or 600.0), float(duration_s * 3))
            updated_tiers += 1
    if updated_tiers == 0:
        raise ValueError(f"No agentx tier found in {src_path}")
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(raw, sort_keys=False))
    RunConfig.load(dest)
    return dest


def cmd_promote(a):
    out: dict = {}
    if a.write_official_config:
        if not a.config:
            raise SystemExit("--write-official-config requires -c/--config")
        dest = write_official_agentx_config(a.config, a.write_official_config, duration_s=a.duration)
        out["official_config"] = str(dest)

    target_ids = list(a.ids or [])
    if not target_ids and a.top:
        r = http.get(
            f"{_url()}/leaderboard",
            params={"tier": a.from_tier, "k": a.top, "sort_by": a.sort},
        )
        r.raise_for_status()
        target_ids = [row["id"] for row in r.json()]

    if target_ids:
        promoted = []
        for jid in target_ids:
            r = http.post(
                f"{_url()}/jobs/{jid}/promote",
                json={"target_tier": a.to_tier, "agent": a.agent, "note": a.note},
            )
            r.raise_for_status()
            promoted.append(r.json())
        out["promoted"] = promoted

    if not out:
        raise SystemExit("Provide job IDs, --top K, or --write-official-config")
    _print(out)


def cmd_board(a):
    if a.post:
        _print(http.post(f"{_url()}/board", json={"agent": _agent(a), "text": a.post}).json())
        return
    for m in http.get(f"{_url()}/board", params={"since": a.since, "limit": a.limit}).json():
        print(f"[{time.strftime('%H:%M:%S', time.localtime(m['ts']))}] {m['agent']}: {m['text']}")


def cmd_agents(a):
    from kiln.agents.launcher import describe_slots, doctor_slots, launch
    from kiln.config import RunConfig
    run = RunConfig.load(a.config)
    if a.dry_run:
        _print(describe_slots(run))
        return
    if a.doctor:
        res = doctor_slots(run, probe_cli=not a.no_probe)
        _print(res)
        if not res["ok"]:
            raise SystemExit(1)
        return
    launch(run)


def build_tunnel_command(
    pod: str,
    namespace: str = "infer-forge-tasks",
    kubeconfig: str | None = None,
    local_port: int = 18765,
    remote_port: int = 18765,
) -> list[str]:
    """Build `kubectl port-forward` argv targeting the coordinator pod."""
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += ["--kubeconfig", os.path.expanduser(kubeconfig)]
    cmd += ["port-forward", "-n", namespace, f"pod/{pod}", f"{local_port}:{remote_port}"]
    return cmd


def tunnel_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Strip HTTP/SOCKS proxy variables that break Kubernetes API SPDY port-forwarding."""
    env = dict(os.environ if base_env is None else base_env)
    for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(k, None)
    return env


def probe_tunnel_health(local_port: int, path: str = "/status", timeout_s: float = 3.0) -> bool:
    """Return True if the forwarded local port responds over HTTP (200/401/403)."""
    try:
        r = httpx.get(
            f"http://127.0.0.1:{local_port}{path}",
            headers=auth_headers(),
            trust_env=False,
            timeout=timeout_s,
        )
        return r.status_code in (200, 401, 403)
    except Exception:
        return False


def _stop_tunnel_proc(proc) -> None:
    import subprocess as _subproc

    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        if isinstance(proc, _subproc.Popen):
            from kiln.engine.base import shutdown_process_group
            shutdown_process_group(proc, timeout=5.0)
            return
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_tunnel_loop(
    cmd: list[str],
    env: dict[str, str],
    *,
    local_port: int,
    reconnect_delay: float = 3.0,
    health_check: bool = True,
    health_interval: float = 15.0,
    health_failures: int = 3,
    health_timeout: float = 3.0,
    startup_grace_s: float = 5.0,
    max_restarts: int | None = None,
    popen_fn=None,
    probe_fn=None,
    sleep_fn=time.sleep,
    clock_fn=time.monotonic,
) -> int:
    """Maintain a `kubectl port-forward` child, restarting on exit or consecutive failed `/status` probes."""
    import subprocess as _subproc

    spawn = popen_fn or (lambda c, env: _subproc.Popen(c, env=env, start_new_session=True))
    probe = probe_fn or (lambda port: probe_tunnel_health(port, timeout_s=health_timeout))
    restarts = 0
    proc = None
    try:
        while True:
            print(f"[kiln tunnel] starting {' '.join(cmd)}", flush=True)
            proc = spawn(cmd, env=env)
            started_at = clock_fn()
            last_probe_at = started_at
            consec_failures = 0
            poll_slice = min(1.0, max(0.02, float(health_interval) / 4.0)) if health_check else 1.0

            while True:
                rc = proc.poll()
                if rc is not None:
                    print(f"[kiln tunnel] exited rc={rc}; reconnecting in {reconnect_delay}s", flush=True)
                    break
                now = clock_fn()
                if (
                    health_check
                    and (now - started_at) >= startup_grace_s
                    and (now - last_probe_at) >= health_interval
                ):
                    last_probe_at = now
                    if probe(local_port):
                        consec_failures = 0
                    else:
                        consec_failures += 1
                        if consec_failures >= max(1, int(health_failures)):
                            print(
                                f"[kiln tunnel] /status probe failed {consec_failures}x on 127.0.0.1:{local_port}; "
                                f"restarting wedged port-forward",
                                flush=True,
                            )
                            _stop_tunnel_proc(proc)
                            break
                sleep_fn(poll_slice)

            proc = None
            restarts += 1
            if max_restarts is not None and restarts >= max_restarts:
                return restarts
            sleep_fn(reconnect_delay)
    except KeyboardInterrupt:
        _stop_tunnel_proc(proc)
        return restarts


def cmd_tunnel(a):
    cmd = build_tunnel_command(
        pod=a.pod,
        namespace=a.namespace,
        kubeconfig=a.kubeconfig,
        local_port=a.local_port,
        remote_port=a.remote_port,
    )
    env = tunnel_env()
    run_tunnel_loop(
        cmd,
        env,
        local_port=a.local_port,
        reconnect_delay=a.reconnect_delay,
        health_check=not a.no_health_check,
        health_interval=a.health_interval,
        health_failures=a.health_failures,
        health_timeout=a.health_timeout,
    )


def cmd_costmodel(a):
    from kiln.costmodel.model import CHIPS, GB, MODELS, Workload, rank, rank_disagg
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
        if not a.json:
            print(f"[profile: {chip.name} ({tag}) | {' '.join(t_parts)} TFLOP/s "
                  f"| HBM={hbm:,.0f} GB/s (eff={cc.get('bw_eff', 0):.2f}) "
                  f"| link_lat={chip.link_latency * 1e6:.1f}us]")
    else:
        chip = CHIPS[a.chip]
    w = Workload(isl=a.isl, osl=a.osl, cache_hit=a.cache_hit,
                 ttft_sla_s=a.ttft, decode_tps_sla=a.tps)
    model_spec = MODELS[a.model]
    if a.disagg:
        max_chips = a.max_chips if a.cluster_gpus is None else min(a.max_chips, a.cluster_gpus)
        coloc = rank(model_spec, chip, w, max_chips=max_chips, k=1)
        best_coloc = coloc[0]["tpm_per_chip"] if coloc else None
        rows = rank_disagg(
            model_spec,
            chip,
            w,
            max_chips=max_chips,
            k=a.k,
            kv_bw_per_gpu=float(a.kv_bw_gbps) * GB,
            cluster_gpus=a.cluster_gpus,
        )
        if a.json:
            _print({"mode": "disagg", "best_colocated_tpm_per_chip": best_coloc, "rankings": rows})
            return
        if not rows:
            print("no admissible disaggregated config meets the SLA")
            return
        for r in rows:
            vs = f" ({r['tpm_per_chip'] / best_coloc:.2f}x vs coloc)" if best_coloc else ""
            print(
                f"{r['ratio']:<6} GPUs={r['total_gpus']:<3} "
                f"P[{r['prefill_config']}] D[{r['decode_config']}] "
                f"TPM/chip={r['tpm_per_chip']:>11,.0f}{vs} "
                f"Cp={r['prefill_batch']:<3} Cd={r['decode_batch']:<4} "
                f"TTFT={r['ttft_s']:.3f}s(kv={r['kv_xfer_ms']:.1f}ms) TPS/u={r['decode_tps_user']:.0f}"
            )
        return
    rows = rank(model_spec, chip, w, max_chips=a.max_chips, k=a.k)
    if a.json:
        _print({"mode": "colocated", "rankings": rows})
        return
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


def cmd_index_refs(a):
    from pathlib import Path
    from kiln.agents.refs_index import ensure_refs_index
    from kiln.config import RunConfig

    model = a.model
    engine = a.engine
    if a.config:
        run = RunConfig.load(a.config)
        model = model or run.engine.model
        engine = engine or run.engine.kind

    out = ensure_refs_index(
        a.refs,
        model_hint=model,
        engine_hint=engine,
        out_path=a.out,
        max_lines=a.max_lines,
        force=a.force,
    )
    if out is None:
        raise SystemExit(f"Refs directory does not exist: {a.refs}")
    if a.print_index:
        print(Path(out).read_text(encoding="utf-8"), end="")
    else:
        print(out)


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
    s.add_argument("--tiers", help="comma-separated additional tiers to serve on this node, e.g. mid,slow")
    s.add_argument("--no-shared", action="store_true",
                   help="serve only --tier/--tiers without auto-including shared-engine tiers")
    s.add_argument("--server", help="central kiln server URL, e.g. http://<node0-ip>:8765")
    s.set_defaults(f=cmd_worker)

    s = sp.add_parser("pd-node", help="launch one SGLang Prefill or Decode worker with kiln hot-reload hooks")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--role", choices=("prefill", "decode"), required=True)
    s.add_argument("--tier", default="slow", help="PD tier name in config (default: slow)")
    s.add_argument("--index", type=int, default=0, help="replica index within pd.prefill / pd.decode")
    s.add_argument("--host", help="override bind host (e.g. 0.0.0.0 on multi-pod K8s)")
    s.add_argument("--port", type=int, help="override bind port")
    s.add_argument("--dry-run", action="store_true", help="print resolved launch command & env JSON and exit")
    s.set_defaults(f=cmd_pd_node)

    s = sp.add_parser("submit", help="submit a candidate forward.py")
    s.add_argument("file"); s.add_argument("--tier", default="fast")
    s.add_argument("--note", default=""); s.add_argument("--agent")
    s.add_argument("--parent", help="parent job id(s), comma-separated (auto-read from .kiln_checkout.json if omitted)")
    s.add_argument("--tags", help="comma-separated submodule/regime tags")
    s.add_argument("--trace", action="store_true", help="force full repeats + kernel trace even under fast screening")
    s.add_argument("--supersede", action="store_true", help="cancel your older queued job(s) on this tier before queueing")
    s.add_argument("--wait", action="store_true"); s.set_defaults(f=cmd_submit)

    s = sp.add_parser("job"); s.add_argument("id"); s.set_defaults(f=cmd_job)
    s = sp.add_parser("jobs", help="list and filter submitted jobs")
    s.add_argument("--tier")
    s.add_argument("--status")
    s.add_argument("--agent")
    s.add_argument("-n", "--limit", type=int, default=30)
    s.set_defaults(f=cmd_jobs)
    s = sp.add_parser("summary", help="show cluster status, job breakdown, and leaderboard summary")
    s.add_argument("--tier", default="slow")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--sort", default="speedup")
    s.set_defaults(f=cmd_summary)
    s = sp.add_parser("pull", help="fetch a job's candidate forward.py source")
    s.add_argument("id")
    s.add_argument("-o", "--out", help="optional path to save candidate source (e.g. forward.py)")
    s.set_defaults(f=cmd_pull)
    s = sp.add_parser("checkout", help="fetch a job into forward.py and record its parent lineage")
    s.add_argument("id")
    s.add_argument("-o", "--out", default="forward.py", help="destination file path (default: forward.py)")
    s.set_defaults(f=cmd_checkout)
    s = sp.add_parser("diff", help="unified diff between two jobs (or parent -> child with 1 job ID)")
    s.add_argument("id_a"); s.add_argument("id_b", nargs="?"); s.set_defaults(f=cmd_diff)
    s = sp.add_parser("lineage", help="show ancestor and descendant lineage for a job")
    s.add_argument("id"); s.set_defaults(f=cmd_lineage)
    s = sp.add_parser("archive", help="show Multi-Island Archive (regimes, submodules, stepping stones, merge suggestions)")
    s.add_argument("--tier", default="fast")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--island", help="filter output (overall, regimes, submodules, stepping_stones, merge_suggestions, or regime:<k>)")
    s.set_defaults(f=cmd_archive)
    s = sp.add_parser("merge", help="compose multiple orthogonal candidate jobs (or asymmetric --prefill/--decode pair) into one forward.py")
    s.add_argument("ids", nargs="*", help="two or more job IDs to merge")
    s.add_argument("--prefill", dest="prefill_id", help="prefill job ID for asymmetric PD pair bundle")
    s.add_argument("--decode", dest="decode_id", help="decode job ID for asymmetric PD pair bundle")
    s.add_argument("-o", "--out", default="forward.py", help="output merged file (default: forward.py)")
    s.add_argument("--allow-overlap", action="store_true", help="merge even if static patch targets overlap")
    s.add_argument("--submit", action="store_true", help="immediately submit merged candidate")
    s.add_argument("--tier", default="fast")
    s.add_argument("--note", default="")
    s.add_argument("--agent")
    s.add_argument("--wait", action="store_true")
    s.set_defaults(f=cmd_merge)
    s = sp.add_parser("micro", help="run lock-free isolated GPU kernel sandbox (snapshots, ptxas spills, optional ncu)")
    s.add_argument("file")
    s.add_argument("--snapshot", help="snapshot tag to replay (e.g. attention, moe, mhc)")
    s.add_argument("--ncu", action="store_true", help="collect Nsight Compute HBM/bank-conflict counters")
    s.add_argument("--no-ptxas", action="store_true", help="skip ptxas spill inspection")
    s.add_argument("--mode", choices=("numeric", "timing"), default="numeric",
                   help="numeric (shared lock) or timing (exclusive GPU lock + ABBA cold-L2 timing)")
    s.add_argument("--timeout", type=float, help="per-run sandbox timeout in seconds (clamped to micro.max_timeout_s)")
    s.add_argument("--local", action="store_true", help="run sandbox locally without kiln serve")
    s.add_argument("-c", "--config", help="optional RunConfig YAML when running with --local")
    s.add_argument("--promote-fast", action="store_true", help="auto-submit to fast tier if micro passes ready_for_fast")
    s.add_argument("--promote-tier", default="fast", help="target tier for --promote-fast (e.g. fast_prefill, fast_decode)")
    s.add_argument("--wait", action="store_true", help="wait for fast-tier job when --promote-fast is set")
    s.add_argument("--note", default="")
    s.add_argument("--agent")
    s.set_defaults(f=cmd_micro)
    s = sp.add_parser("micro-snapshots", help="list cold-boot captured tensor snapshots available for `kiln micro`")
    s.add_argument("--tag")
    s.set_defaults(f=cmd_micro_snapshots)
    s = sp.add_parser("cancel", help="cancel a queued job before it runs")
    s.add_argument("id"); s.set_defaults(f=cmd_cancel)
    s = sp.add_parser("withdraw", help="invalidate a bogus or superseded job from leaderboards")
    s.add_argument("id")
    s.add_argument("--reason", default="Withdrawn")
    s.add_argument("--no-cascade", action="store_true", help="do not invalidate sibling jobs sharing source sha256")
    s.set_defaults(f=cmd_withdraw)
    s = sp.add_parser("status"); s.set_defaults(f=cmd_status)

    s = sp.add_parser("leaderboard"); s.add_argument("--tier", default="slow")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--sort", default="speedup",
                   help="sort key: speedup, decode_tps, prefill_tps, ttft_s, decode_small, decode_mid, prefill_large, "
                        "output_tput, active_decode_tps, itl_p50_ms, itl_p90_ms, interactivity_p90, "
                        "spec_accept_length, spec_accept_rate, paired_speedup")
    s.set_defaults(f=cmd_leaderboard)

    s = sp.add_parser("promote", help="promote winning jobs to another tier or generate 900s official config")
    s.add_argument("ids", nargs="*", help="specific job IDs to promote")
    s.add_argument("--top", type=int, help="promote top K jobs from --from-tier leaderboard")
    s.add_argument("--from-tier", default="fast")
    s.add_argument("--to-tier", default="slow")
    s.add_argument("--sort", default="speedup")
    s.add_argument("--agent")
    s.add_argument("--note", default="")
    s.add_argument("-c", "--config", help="source YAML config for --write-official-config")
    s.add_argument("--write-official-config", metavar="OUT_YAML",
                   help="write an official submission_valid=true (>=900s, unsafe_override=false) YAML config")
    s.add_argument("--duration", type=int, default=900, help="official AgentX benchmark_duration_s (>=900)")
    s.set_defaults(f=cmd_promote)

    s = sp.add_parser("board"); s.add_argument("--post"); s.add_argument("--agent")
    s.add_argument("--since", type=float, default=0.0); s.add_argument("--limit", type=int, default=50)
    s.set_defaults(f=cmd_board)

    s = sp.add_parser("agents", help="launch N parallel coding agents")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--dry-run", action="store_true", help="print resolved per-slot CLI commands and exit")
    s.add_argument("--doctor", action="store_true", help="verify server reachability and probe each slot CLI")
    s.add_argument("--no-probe", action="store_true", help="with --doctor, skip launching CLI smoke prompt")
    s.set_defaults(f=cmd_agents)

    s = sp.add_parser("tunnel", help="maintain an auto-reconnecting proxy-free kubectl port-forward")
    s.add_argument("--pod", required=True)
    s.add_argument("--namespace", default="infer-forge-tasks")
    s.add_argument("--kubeconfig")
    s.add_argument("--local-port", type=int, default=18765)
    s.add_argument("--remote-port", type=int, default=18765)
    s.add_argument("--reconnect-delay", type=float, default=3.0)
    s.add_argument("--health-interval", type=float, default=15.0,
                   help="seconds between active HTTP /status probes (default: 15)")
    s.add_argument("--health-failures", type=int, default=3,
                   help="consecutive failed /status probes before killing wedged port-forward (default: 3)")
    s.add_argument("--health-timeout", type=float, default=3.0,
                   help="timeout per /status health probe in seconds (default: 3)")
    s.add_argument("--no-health-check", action="store_true",
                   help="disable active HTTP /status health probing")
    s.set_defaults(f=cmd_tunnel)

    s = sp.add_parser("costmodel", help="rank colocated or PD-disaggregated (xPyD) parallelism configs")
    s.add_argument("--model", default="qwen3-30b-a3b"); s.add_argument("--chip", default="B200")
    s.add_argument("-p", "--profile", help="calibrated hw_profile.json from `kiln probe` (overrides --chip)")
    s.add_argument("--isl", type=int, default=8192); s.add_argument("--osl", type=int, default=1024)
    s.add_argument("--cache-hit", type=float, default=0.0)
    s.add_argument("--ttft", type=float, default=1.5); s.add_argument("--tps", type=float, default=40)
    s.add_argument("--max-chips", type=int, default=64); s.add_argument("-k", type=int, default=8)
    s.add_argument("--disagg", action="store_true", help="search asymmetric xPyD Prefill/Decode disaggregated configs")
    s.add_argument("--kv-bw-gbps", type=float, default=50.0, help="inter-node KV transfer bandwidth per GPU in GB/s (default: 50)")
    s.add_argument("--cluster-gpus", type=int, help="optional total GPU budget to fit integer xPyD replicas (e.g. 16, 32)")
    s.add_argument("--json", action="store_true", help="output rankings as JSON")
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

    s = sp.add_parser("index-refs", help="generate a compact source index (refs/INDEX.md) for agents")
    s.add_argument("--refs", default="refs", help="refs directory path")
    s.add_argument("-c", "--config", help="optional run config YAML to infer model and engine")
    s.add_argument("--model", help="model name or path hint (overrides --config)")
    s.add_argument("--engine", help="engine kind hint (overrides --config)")
    s.add_argument("-o", "--out", help="output index path (defaults to <refs>/INDEX.md)")
    s.add_argument("--max-lines", type=int, default=85, help="maximum lines in generated INDEX.md")
    s.add_argument("--force", action="store_true", help="overwrite even an existing or manual INDEX.md")
    s.add_argument("--print", dest="print_index", action="store_true", help="print generated INDEX.md")
    s.set_defaults(f=cmd_index_refs)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
