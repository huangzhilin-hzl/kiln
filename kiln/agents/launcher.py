"""Spawn N parallel coding agents, each in its own workspace dir.

Supports mixing providers in one fleet (`agy`, `codex`, `claude`, or custom CLIs),
either via `agents.slots` or `agents.command` / `agents.commands` + `agents.models`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from kiln.agents.refs_index import ensure_refs_index
from kiln.config import RunConfig
from kiln.engine.base import shutdown_process_group

PROMPT = Path(__file__).with_name("prompt.md")
REPO = Path(__file__).resolve().parents[2]

ROLE_PROMPTS: dict[str, str] = {
    "explorer": (
        "Focus on bold, high-upside algorithmic and kernel optimizations (novel operator fusions, "
        "memory-hierarchy / SM wave scheduling, and speculative draft acceleration). Post clear hypotheses and trace gains."
    ),
    "auditor": (
        "Focus on bitwise/numerical verification, tight shape specialization, isolating standalone gains "
        "via ablation, catching numerical drift across widths, and withdrawing/reporting invalid candidates."
    ),
    "integrator": (
        "Focus on pulling verified winning candidates via `kiln checkout <job_id>` / `kiln pull <job_id>` and "
        "`kiln diff [id_a] <id_b>`, inspecting `kiln archive --tier fast` for orthogonal stepping stones, "
        "combining orthogonal wins cleanly via `kiln merge <id_a> <id_b> -o forward.py`, and checking stacked end-to-end speedups."
    ),
    "synthesizer": (
        "Dedicated Synthesizer / Ensembler slot: continuously monitor `kiln archive --tier fast` and `kiln board` "
        "for orthogonal stepping stones across regimes (`M<=48`, `65..256`, `M>512`) and submodule tags "
        "(`mhc`, `gemv`, `moe_router`, `mla_rope`). Synthesize combined candidates via "
        "`kiln merge <id_a> <id_b> -o forward.py` (or `@kiln_patch` composition), resolve any target overlaps, "
        "verify stacked speedups on `fast`, and promote verified composite winners to `slow`."
    ),
}

CONTINUE_NOTE = (
    "\n\n## Continuation\n"
    "You are resuming an ongoing optimization run in this workspace. Inspect your current `forward.py`, "
    "check `kiln board` and `kiln leaderboard`, and continue iterating on new speedups.\n"
    "Avoid re-reading full large reference files in `./refs/`; check `./refs/INDEX.md` and read only needed line ranges.\n"
)

_QUOTA_MARKERS = (
    "resource_exhausted",
    '"error_code":429',
    "individual quota reached",
    "rate_limit_exceeded",
    "insufficient_quota",
    "too many requests",
    "rate_limit_error",
    "overloaded_error",
    "hit your limit",
    "http 429",
    "status 429",
    "api error: 429",
    "api error: 529",
)
_RESET_RE = re.compile(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", re.IGNORECASE)
_TRY_AGAIN_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE)
_RETRY_AFTER_RE = re.compile(r"retry[- ]after[:\s]+(\d+)", re.IGNORECASE)


def _parse_quota_backoff(log_path: Path, slot_idx: int = 0) -> float | None:
    """Return backoff seconds if the latest agent run ended with a 429/rate-limit/quota error."""
    if not log_path.exists():
        return None
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 8192))
        tail = f.read().decode("utf-8", errors="ignore")
    segment = tail.rsplit("[kiln]", 1)[-1][-4096:] if "[kiln]" in tail else tail[-4096:]
    if not any(m in segment.lower() for m in _QUOTA_MARKERS):
        return None
    base_wait = 300.0
    m = _RESET_RE.search(segment)
    if m and any(g is not None for g in m.groups()):
        hours = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = int(m.group(3) or 0)
        base_wait = max(60.0, min(3900.0, float(hours * 3600 + mins * 60 + secs)))
    else:
        m = _TRY_AGAIN_RE.search(segment)
        if m and any(g is not None for g in m.groups()):
            hours = int(m.group(1) or 0)
            mins = int(m.group(2) or 0)
            secs_f = float(m.group(3) or 0.0)
            base_wait = max(15.0, min(3900.0, hours * 3600.0 + mins * 60.0 + secs_f))
        elif (m := _RETRY_AFTER_RE.search(segment)) is not None:
            base_wait = max(15.0, min(3900.0, float(int(m.group(1)))))
    return base_wait + 15.0 * slot_idx


def _inject_refs_dir(argv: list[str], provider: str, refs_dir: Path) -> list[str]:
    """Attach `--add-dir <refs_dir>` for CLIs that support extra readable directories."""
    if not argv or not refs_dir.exists():
        return list(argv)
    if any(tok == "--add-dir" or tok.startswith("--add-dir=") for tok in argv):
        return list(argv)
    exe = Path(argv[0]).name
    kind = provider if provider in ("agy", "codex", "claude") else exe
    if kind in ("agy", "claude"):
        return [argv[0], "--add-dir", str(refs_dir), *argv[1:]]
    if kind == "codex" and len(argv) >= 2 and argv[1] in ("exec", "e"):
        return [argv[0], argv[1], "--add-dir", str(refs_dir), *argv[2:]]
    return list(argv)


def _normalize_proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Mirror proxy env vars between lower/uppercase and bypass proxy for loopback."""
    for lo, hi in (
        ("http_proxy", "HTTP_PROXY"),
        ("https_proxy", "HTTPS_PROXY"),
        ("all_proxy", "ALL_PROXY"),
    ):
        if env.get(lo) and not env.get(hi):
            env[hi] = env[lo]
        elif env.get(hi) and not env.get(lo):
            env[lo] = env[hi]

    merged: list[str] = []
    seen: set[str] = set()
    for raw in (env.get("NO_PROXY", ""), env.get("no_proxy", "")):
        for item in raw.split(","):
            entry = item.strip()
            if entry and entry not in seen:
                seen.add(entry)
                merged.append(entry)
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in seen:
            seen.add(host)
            merged.append(host)
    joined = ",".join(merged)
    env["NO_PROXY"] = joined
    env["no_proxy"] = joined
    return env


_SENSITIVE_ENV_VARS = (
    "SSH_AUTH_SOCK",
    "SCHED_KUBECONFIG",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DOCKER_HOST",
)


def _strip_sensitive_env(
    env: dict[str, str], allowed_keys: set[str] | None = None
) -> dict[str, str]:
    """Strip host cluster/cloud credentials from spawned agent environments unless allowed per-slot."""
    allowed = allowed_keys or set()
    for key in _SENSITIVE_ENV_VARS:
        if key not in allowed:
            env.pop(key, None)
    if "KUBECONFIG" not in allowed:
        env["KUBECONFIG"] = "/dev/null"
    return env


def describe_slots(run: RunConfig) -> list[dict]:
    """Return resolved per-slot CLI metadata without spawning any agent processes."""
    refs_dir = REPO / "refs"
    out: list[dict] = []
    for i in range(1, run.agents.num_agents + 1):
        slot = run.agents.resolve_slot(i - 1)
        argv = _inject_refs_dir(slot.argv, slot.provider, refs_dir)
        exe = argv[0] if argv else ""
        cli_found = bool(exe and (Path(exe).is_file() or shutil.which(exe)))
        out.append({
            "agent": f"agent_{i}",
            "provider": slot.provider,
            "model": slot.model,
            "label": slot.label,
            "role": slot.role,
            "argv": argv,
            "env_keys": sorted(slot.env.keys()),
            "cli_found": cli_found,
        })
    return out


def _build_slot_env(
    run: RunConfig,
    slot,
    agent_name: str,
    venv_bin: str,
    server_url: str,
    triton_dir: Path | None = None,
) -> dict[str, str]:
    base_path = slot.env.get("PATH", os.environ.get("PATH", ""))
    env = {
        **os.environ,
        **slot.env,
        "PATH": f"{venv_bin}{os.pathsep}{base_path}",
        "PYTHONUNBUFFERED": "1",
        "KILN_AGENT": agent_name,
        "KILN_AGENT_PROVIDER": slot.provider,
        "KILN_SERVER": server_url,
    }
    if triton_dir is not None:
        env["TRITON_CACHE_DIR"] = str(triton_dir)
    if slot.model:
        env["KILN_AGENT_MODEL"] = slot.model
    _normalize_proxy_env(env)
    if run.agents.isolate_credentials:
        _strip_sensitive_env(env, allowed_keys=set(slot.env.keys()))
    return env


def _write_slot_status(
    ws: Path,
    slot: dict,
    state: str,
    now: float | None = None,
    rc: int | None = None,
) -> Path:
    ts = time.time() if now is None else now
    log_path = ws / "agent.log"
    log_bytes = log_path.stat().st_size if log_path.exists() else 0
    started_at = slot.get("started_at")
    payload = {
        "agent": slot["name"],
        "provider": slot["provider"],
        "model": slot.get("model"),
        "role": slot.get("role"),
        "state": state,
        "pid": slot["proc"].pid if slot.get("proc") is not None else None,
        "started_at": started_at,
        "updated_at": ts,
        "elapsed_s": round(max(0.0, ts - started_at), 1) if started_at else 0.0,
        "log_bytes": log_bytes,
        "last_rc": rc,
        "failures": slot.get("failures", 0),
        "quota_retries": slot.get("quota_retries", 0),
        "stale_restarts": slot.get("stale_restarts", 0),
        "last_activity_at": slot.get("last_activity_at"),
        "wake_at": slot.get("wake_at"),
    }
    status_path = ws / "status.json"
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return status_path


def _load_codex_cached_models(cache_path: Path | None = None) -> set[str] | None:
    """Load available Codex model slugs/IDs from `~/.codex/models_cache.json` when present."""
    if cache_path is None:
        codex_home = os.environ.get("CODEX_HOME")
        base = Path(codex_home).expanduser() if codex_home else (Path.home() / ".codex")
        cache_path = base / "models_cache.json"
    if not cache_path.is_file():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return None
    models: set[str] = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            models.add(item.strip())
        elif isinstance(item, dict):
            for key in ("slug", "id", "model", "name"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    models.add(val.strip())
    return models or None


def _looks_like_local_path(val: str) -> bool:
    s = val.strip()
    return s.startswith(("/", "./", "../", "~"))


def _validate_offline_model_paths(
    run: RunConfig, env: Mapping[str, str] | None = None
) -> list[str]:
    """Validate local filesystem model/tokenizer paths and offline HF cache readiness."""
    if run.engine.kind == "mock":
        return []
    src_env = os.environ if env is None else env
    offline = any(
        str(src_env.get(k, "")).strip().lower() in ("1", "true", "yes")
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    hf_cache_root = Path(
        src_env.get("HF_HUB_CACHE")
        or (Path(src_env["HF_HOME"]).expanduser() / "hub" if src_env.get("HF_HOME") else Path.home() / ".cache" / "huggingface" / "hub")
    )
    errors: list[str] = []
    for field_name, raw_val in (("engine.model", run.engine.model), ("engine.tokenizer", run.engine.tokenizer)):
        if not raw_val or not str(raw_val).strip():
            continue
        val = str(raw_val).strip()
        expanded = Path(val).expanduser()
        if _looks_like_local_path(val):
            if not expanded.exists():
                errors.append(f"{field_name} local path does not exist: {val}")
        elif offline and not expanded.exists():
            # Check standard HF Hub cache layout (`models--org--repo`).
            cache_dir = hf_cache_root / ("models--" + val.replace("/", "--"))
            if not cache_dir.is_dir():
                errors.append(
                    f"HF_HUB_OFFLINE is enabled and {field_name}={val!r} is neither a local path "
                    f"nor present in HF cache ({cache_dir})"
                )
    return errors


def doctor_slots(
    run: RunConfig,
    server_url: str | None = None,
    probe_cli: bool = True,
    probe_timeout_s: float = 20.0,
    codex_models_cache: Path | None = None,
) -> dict:
    """Validate server connectivity, CLI availability, Codex model cache, offline paths, and optional CLI probe."""
    import httpx
    from kiln.server.auth import auth_headers

    target_url = server_url or os.environ.get("KILN_SERVER") or run.client_url
    server_ok = False
    server_error: str | None = None
    try:
        r = httpx.get(
            f"{target_url.rstrip('/')}/status",
            headers=auth_headers(),
            trust_env=False,
            timeout=5.0,
        )
        r.raise_for_status()
        server_ok = True
    except Exception as exc:
        server_error = str(exc)

    offline_errors = _validate_offline_model_paths(run)
    codex_cached = _load_codex_cached_models(codex_models_cache)

    refs_dir = REPO / "refs"
    venv_bin = str(Path(sys.executable).parent)
    reports: list[dict] = []
    for i in range(1, run.agents.num_agents + 1):
        name = f"agent_{i}"
        slot = run.agents.resolve_slot(i - 1)
        argv = _inject_refs_dir(slot.argv, slot.provider, refs_dir)
        env = _build_slot_env(run, slot, name, venv_bin, target_url)
        exe = argv[0] if argv else ""
        resolved_exe = (
            str(Path(exe))
            if exe and (os.sep in exe or (os.altsep and os.altsep in exe)) and Path(exe).is_file()
            else shutil.which(exe, path=env["PATH"])
        )
        cli_found = bool(resolved_exe)
        version_str: str | None = None
        probe_ok: bool | None = None
        model_cache_ok: bool | None = None
        err: str | None = None

        if slot.provider == "codex" and slot.model and codex_cached is not None:
            model_cache_ok = slot.model in codex_cached
            if not model_cache_ok:
                sample = ", ".join(sorted(codex_cached)[:8])
                err = f"Codex model {slot.model!r} not found in models_cache.json (available: {sample})"

        if not cli_found:
            err = f"CLI {exe!r} not found on PATH"
        elif model_cache_ok is not False:
            try:
                v = subprocess.run(
                    [resolved_exe, "--version"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                first = (v.stdout or v.stderr or "").strip().splitlines()
                if v.returncode == 0 and first:
                    version_str = first[0][:120]
            except Exception:
                pass

            if probe_cli:
                probe_text = "Reply with exactly OK"
                try:
                    if any("{prompt}" in tok for tok in argv):
                        probe_cmd = [resolved_exe, *(tok.replace("{prompt}", probe_text) for tok in argv[1:])]
                        p_res = subprocess.run(
                            probe_cmd,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            capture_output=True,
                            text=True,
                            timeout=probe_timeout_s,
                        )
                    elif slot.provider in ("codex", "claude") or (argv and argv[-1] == "-"):
                        probe_cmd = [resolved_exe, *argv[1:]]
                        p_res = subprocess.run(
                            probe_cmd,
                            env=env,
                            input=probe_text + "\n",
                            capture_output=True,
                            text=True,
                            timeout=probe_timeout_s,
                        )
                    else:
                        p_res = subprocess.run(
                            [resolved_exe, "--help"],
                            env=env,
                            stdin=subprocess.DEVNULL,
                            capture_output=True,
                            text=True,
                            timeout=probe_timeout_s,
                        )
                    probe_ok = p_res.returncode == 0
                    if not probe_ok:
                        out_tail = ((p_res.stderr or "") + "\n" + (p_res.stdout or "")).strip()
                        err = out_tail[-400:] if out_tail else f"probe exited rc={p_res.returncode}"
                except Exception as exc:
                    probe_ok = False
                    err = str(exc)

        ok = cli_found and (model_cache_ok is not False) and (True if not probe_cli else bool(probe_ok))
        reports.append({
            "agent": name,
            "provider": slot.provider,
            "model": slot.model,
            "role": slot.role,
            "binary": resolved_exe,
            "cli_found": cli_found,
            "model_cache_ok": model_cache_ok,
            "version": version_str,
            "probe_ok": probe_ok,
            "ok": ok,
            "error": err,
        })

    return {
        "server_url": target_url,
        "server_ok": server_ok,
        "server_error": server_error,
        "offline_path_errors": offline_errors,
        "ok": server_ok and (not offline_errors) and all(r["ok"] for r in reports),
        "slots": reports,
    }


def _terminate_slot_proc(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    if isinstance(proc, subprocess.Popen):
        shutdown_process_group(proc, timeout=5.0)
        return
    try:
        proc.terminate()
    except Exception:
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _spawn_agent(cmd: list[str], ws: Path, env: dict[str, str], prompt_text: str):
    log = open(ws / "agent.log", "ab")
    try:
        if any("{prompt}" in tok for tok in cmd):
            argv = [tok.replace("{prompt}", prompt_text) for tok in cmd]
            p = subprocess.Popen(argv, cwd=ws, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            return p, log
        p = subprocess.Popen(cmd, cwd=ws, env=env,
                             stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    except Exception:
        log.close()
        raise
    assert p.stdin is not None
    try:
        with p.stdin:
            p.stdin.write(prompt_text.encode())
    except BrokenPipeError:
        pass  # The supervisor below handles the child exit.
    return p, log


def _check_and_handle_stale_slot(
    slot: dict,
    now: float,
    stale_timeout_s: float | None,
    *,
    respawn_on_exit: bool,
    deadline: float,
    spawn_fn=_spawn_agent,
) -> str | None:
    """Terminate and optionally respawn a running agent slot whose log stays silent past `stale_timeout_s`."""
    if stale_timeout_s is None or stale_timeout_s <= 0:
        return None
    proc = slot.get("proc")
    if proc is None or proc.poll() is not None:
        return None
    last_act = float(slot.get("last_activity_at") or slot.get("started_at") or now)
    silent_s = now - last_act
    if silent_s < float(stale_timeout_s):
        return None

    ws: Path = slot["ws"]
    print(
        f"[kiln] {slot['name']} log silent for {silent_s:.0f}s (>= stale_timeout_s={stale_timeout_s:.0f}s); "
        f"terminating hung process"
    )
    _terminate_slot_proc(proc)
    if slot.get("log") is not None and not slot["log"].closed:
        slot["log"].close()
    slot["proc"] = None
    slot["stale_restarts"] = int(slot.get("stale_restarts", 0)) + 1

    with open(ws / "agent.log", "a", encoding="utf-8") as lf:
        lf.write(
            f"\n[kiln] {slot['name']} terminated after {silent_s:.0f}s of silence "
            f"(stale_timeout_s={stale_timeout_s:.0f}s)\n"
        )

    if not respawn_on_exit or now >= deadline:
        slot["active"] = False
        _write_slot_status(ws, slot, "stopped", now=now)
        return "stopped"

    p, log = spawn_fn(slot["cmd"], ws, slot["env"], slot["prompt"] + CONTINUE_NOTE)
    log_bytes = (ws / "agent.log").stat().st_size if (ws / "agent.log").exists() else 0
    slot.update(
        proc=p,
        log=log,
        wake_at=None,
        started_at=now,
        last_activity_at=now,
        last_heartbeat_at=now,
        last_log_bytes=log_bytes,
    )
    _write_slot_status(ws, slot, "running", now=now)
    info = f"provider={slot['provider']}" + (f", model={slot['model']}" if slot.get("model") else "")
    print(f"[kiln] respawned stale {slot['name']} (pid {p.pid}, {info})")
    return "respawned"


def launch(run: RunConfig, baseline: str = "candidates/baseline_forward.py") -> None:
    root = Path(run.workdir) / run.name / "agents"
    root.mkdir(parents=True, exist_ok=True)
    n = run.agents.num_agents
    venv_bin = str(Path(sys.executable).parent)
    server_url = os.environ.get("KILN_SERVER") or run.client_url
    hw_profile = None
    if hw_path := run.resolve_hw_profile():
        from kiln.costmodel.probe import HardwareProfile
        hw_profile = HardwareProfile.load(hw_path)
    else:
        try:
            import httpx
            from kiln.costmodel.probe import HardwareProfile
            from kiln.server.auth import auth_headers
            r = httpx.get(f"{server_url.rstrip('/')}/hardware", headers=auth_headers(),
                          trust_env=False, timeout=10.0)
            if r.status_code == 200 and r.json().get("available") and r.json().get("profile"):
                hw_profile = HardwareProfile.from_dict(r.json()["profile"])
        except Exception:
            pass

    refs_dir = REPO / "refs"
    if refs_dir.exists():
        ensure_refs_index(refs_dir, model_hint=run.engine.model, engine_hint=run.engine.kind)
    slots: list[dict] = []
    try:
        for i in range(1, n + 1):
            if i > 1:
                time.sleep(2.0)
            name = f"agent_{i}"
            ws = root / name
            ws.mkdir(exist_ok=True)
            if refs_dir.exists() and not (ws / "refs").exists():
                (ws / "refs").symlink_to(refs_dir)
            triton_dir = (ws / ".triton").resolve()
            triton_dir.mkdir(exist_ok=True)
            if not (ws / "forward.py").exists():
                shutil.copy(REPO / baseline, ws / "forward.py")
            slot = run.agents.resolve_slot(i - 1)
            prompt = PROMPT.read_text().format(agent=name, n=n, model=run.engine.model,
                                               engine=run.engine.kind)
            if hw_profile is not None:
                hw_profile.save(ws / "hw_profile.json")
                (ws / "HARDWARE.md").write_text(hw_profile.to_markdown(), encoding="utf-8")
                prompt = prompt.rstrip() + "\n" + hw_profile.to_prompt_summary()
            if slot.role or slot.prompt_extra:
                role_lines = ["\n\n## Your Assigned Role"]
                if slot.role:
                    role_lines.append(f"- Role: `{slot.role}` — {ROLE_PROMPTS[slot.role]}")
                if slot.prompt_extra and slot.prompt_extra.strip():
                    role_lines.append(f"- Guidance: {slot.prompt_extra.strip()}")
                prompt = prompt.rstrip() + "\n" + "\n".join(role_lines) + "\n"
            (ws / "PROMPT.md").write_text(prompt, encoding="utf-8")
            slot_cmd = _inject_refs_dir(slot.argv, slot.provider, refs_dir)
            env = _build_slot_env(run, slot, name, venv_bin, server_url, triton_dir=triton_dir)
            try:
                p, log = _spawn_agent(slot_cmd, ws, env, prompt)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Agent CLI {slot_cmd[0]!r} not found on PATH (slot {name}, provider={slot.provider})"
                ) from exc
            now_spawn = time.time()
            log_size = (ws / "agent.log").stat().st_size if (ws / "agent.log").exists() else 0
            slot_state = {
                "idx": i - 1, "name": name, "ws": ws, "cmd": slot_cmd,
                "provider": slot.provider, "model": slot.model, "role": slot.role,
                "env": env, "prompt": prompt, "proc": p, "log": log,
                "started_at": now_spawn, "last_activity_at": now_spawn,
                "last_heartbeat_at": now_spawn, "last_log_bytes": log_size,
                "failures": 0, "quota_retries": 0, "stale_restarts": 0, "wake_at": None, "active": True,
            }
            _write_slot_status(ws, slot_state, "running", now=now_spawn)
            slots.append(slot_state)
            info = f"provider={slot.provider}"
            if slot.model:
                info += f", model={slot.model}"
            if slot.role:
                info += f", role={slot.role}"
            print(f"[kiln] started {name} (pid {p.pid}, {info}) in {ws}")

        deadline = time.time() + run.agents.budget_hours * 3600
        while time.time() < deadline and any(s["active"] for s in slots):
            time.sleep(5)
            now = time.time()
            for s in slots:
                if not s["active"]:
                    continue
                if s["proc"] is None:
                    if now >= deadline:
                        s["active"] = False
                        _write_slot_status(s["ws"], s, "stopped", now=now)
                        continue
                    if s["wake_at"] is not None and now >= s["wake_at"]:
                        p, log = _spawn_agent(s["cmd"], s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                        log_bytes = (s["ws"] / "agent.log").stat().st_size if (s["ws"] / "agent.log").exists() else 0
                        s.update(
                            proc=p, log=log, wake_at=None,
                            started_at=now, last_activity_at=now, last_heartbeat_at=now, last_log_bytes=log_bytes,
                        )
                        _write_slot_status(s["ws"], s, "running", now=now)
                        info = f"provider={s['provider']}" + (f", model={s['model']}" if s.get("model") else "")
                        print(f"[kiln] respawned {s['name']} (pid {p.pid}, {info})")
                    else:
                        _write_slot_status(s["ws"], s, "backing_off", now=now)
                    continue
                rc = s["proc"].poll()
                if rc is None:
                    cur_bytes = (s["ws"] / "agent.log").stat().st_size if (s["ws"] / "agent.log").exists() else 0
                    if cur_bytes != s["last_log_bytes"]:
                        s["last_log_bytes"] = cur_bytes
                        s["last_activity_at"] = now
                        s["last_heartbeat_at"] = now
                    else:
                        action = _check_and_handle_stale_slot(
                            s,
                            now,
                            run.agents.stale_timeout_s,
                            respawn_on_exit=run.agents.respawn_on_exit,
                            deadline=deadline,
                        )
                        if action is not None:
                            continue
                        if now - s["last_heartbeat_at"] >= 300.0:
                            elapsed = now - (s.get("started_at") or now)
                            print(f"[kiln] {s['name']} running ({elapsed:.0f}s elapsed, log silent at {cur_bytes}B)")
                            s["last_heartbeat_at"] = now
                    _write_slot_status(s["ws"], s, "running", now=now)
                    continue
                s["log"].close()
                s["proc"] = None
                if not run.agents.respawn_on_exit or now >= deadline:
                    s["active"] = False
                    _write_slot_status(s["ws"], s, "exited", now=now, rc=rc)
                    continue
                if rc != 0:
                    wait_s = _parse_quota_backoff(s["ws"] / "agent.log", slot_idx=s["idx"])
                    if wait_s is not None:
                        s["quota_retries"] += 1
                        if s["quota_retries"] > 24:
                            print(f"[kiln] {s['name']} exceeded 24 quota retries; stopping")
                            s["active"] = False
                            _write_slot_status(s["ws"], s, "stopped", now=now, rc=rc)
                            continue
                        with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                            lf.write(f"\n[kiln] {s['name']} hit 429 quota limit; backing off {wait_s:.0f}s\n")
                        print(f"[kiln] {s['name']} hit 429 quota limit; backing off {wait_s:.0f}s")
                        s["wake_at"] = now + wait_s
                        _write_slot_status(s["ws"], s, "backing_off", now=now, rc=rc)
                        continue
                    s["failures"] += 1
                    if s["failures"] >= 3:
                        print(f"[kiln] {s['name']} failed {s['failures']} times in a row (rc={rc}); "
                              f"see {s['ws'] / 'agent.log'}")
                        s["active"] = False
                        _write_slot_status(s["ws"], s, "stopped", now=now, rc=rc)
                        continue
                    wait_s = 5.0 * s["failures"]
                    with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                        lf.write(f"\n[kiln] {s['name']} exited (rc={rc}); retrying in {wait_s:.0f}s\n")
                    s["wake_at"] = now + wait_s
                    _write_slot_status(s["ws"], s, "backing_off", now=now, rc=rc)
                    continue
                s["failures"] = 0
                s["quota_retries"] = 0
                with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                    lf.write(f"\n[kiln] {s['name']} exited cleanly; respawning\n")
                p, log = _spawn_agent(s["cmd"], s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                log_bytes = (s["ws"] / "agent.log").stat().st_size if (s["ws"] / "agent.log").exists() else 0
                s.update(
                    proc=p, log=log, wake_at=None,
                    started_at=now, last_activity_at=now, last_heartbeat_at=now, last_log_bytes=log_bytes,
                )
                _write_slot_status(s["ws"], s, "running", now=now, rc=rc)
                info = f"provider={s['provider']}" + (f", model={s['model']}" if s.get("model") else "")
                print(f"[kiln] respawned {s['name']} (pid {p.pid}, {info})")
    except KeyboardInterrupt:
        pass
    finally:
        for s in slots:
            p = s["proc"]
            if p is not None and p.poll() is None:
                _terminate_slot_proc(p)
                print(f"[kiln] stopped {s['name']}")
                s["proc"] = None
                _write_slot_status(s["ws"], s, "stopped")
            elif s["active"] or not (s["ws"] / "status.json").exists():
                _write_slot_status(s["ws"], s, "stopped")
            if s.get("log") is not None and not s["log"].closed:
                s["log"].close()
