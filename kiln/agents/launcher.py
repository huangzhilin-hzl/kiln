"""Spawn N parallel coding agents, each in its own workspace dir.

Supports mixing providers in one fleet (`agy`, `codex`, `claude`, or custom CLIs),
either via `agents.slots` or `agents.command` / `agents.commands` + `agents.models`.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from kiln.agents.refs_index import ensure_refs_index
from kiln.config import RunConfig

PROMPT = Path(__file__).with_name("prompt.md")
REPO = Path(__file__).resolve().parents[2]

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
            "argv": argv,
            "env_keys": sorted(slot.env.keys()),
            "cli_found": cli_found,
        })
    return out


def _spawn_agent(cmd: list[str], ws: Path, env: dict[str, str], prompt_text: str):
    log = open(ws / "agent.log", "ab")
    try:
        if any("{prompt}" in tok for tok in cmd):
            argv = [tok.replace("{prompt}", prompt_text) for tok in cmd]
            p = subprocess.Popen(argv, cwd=ws, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            return p, log
        p = subprocess.Popen(cmd, cwd=ws, env=env,
                             stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
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
            prompt = PROMPT.read_text().format(agent=name, n=n, model=run.engine.model,
                                               engine=run.engine.kind)
            if hw_profile is not None:
                hw_profile.save(ws / "hw_profile.json")
                (ws / "HARDWARE.md").write_text(hw_profile.to_markdown(), encoding="utf-8")
                prompt = prompt.rstrip() + "\n" + hw_profile.to_prompt_summary()
            (ws / "PROMPT.md").write_text(prompt, encoding="utf-8")
            slot = run.agents.resolve_slot(i - 1)
            slot_cmd = _inject_refs_dir(slot.argv, slot.provider, refs_dir)
            base_path = slot.env.get("PATH", os.environ.get("PATH", ""))
            env = {
                **os.environ,
                **slot.env,
                "PATH": f"{venv_bin}{os.pathsep}{base_path}",
                "KILN_AGENT": name,
                "KILN_AGENT_PROVIDER": slot.provider,
                "KILN_SERVER": server_url,
                "TRITON_CACHE_DIR": str(triton_dir),
            }
            if slot.model:
                env["KILN_AGENT_MODEL"] = slot.model
            _normalize_proxy_env(env)
            try:
                p, log = _spawn_agent(slot_cmd, ws, env, prompt)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Agent CLI {slot_cmd[0]!r} not found on PATH (slot {name}, provider={slot.provider})"
                ) from exc
            slots.append({
                "idx": i - 1, "name": name, "ws": ws, "cmd": slot_cmd,
                "provider": slot.provider, "model": slot.model,
                "env": env, "prompt": prompt, "proc": p, "log": log,
                "failures": 0, "quota_retries": 0, "wake_at": None, "active": True,
            })
            info = f"provider={slot.provider}" + (f", model={slot.model}" if slot.model else "")
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
                        continue
                    if s["wake_at"] is not None and now >= s["wake_at"]:
                        p, log = _spawn_agent(s["cmd"], s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                        s.update(proc=p, log=log, wake_at=None)
                        info = f"provider={s['provider']}" + (f", model={s['model']}" if s.get("model") else "")
                        print(f"[kiln] respawned {s['name']} (pid {p.pid}, {info})")
                    continue
                rc = s["proc"].poll()
                if rc is None:
                    continue
                s["log"].close()
                s["proc"] = None
                if not run.agents.respawn_on_exit or now >= deadline:
                    s["active"] = False
                    continue
                if rc != 0:
                    wait_s = _parse_quota_backoff(s["ws"] / "agent.log", slot_idx=s["idx"])
                    if wait_s is not None:
                        s["quota_retries"] += 1
                        if s["quota_retries"] > 24:
                            print(f"[kiln] {s['name']} exceeded 24 quota retries; stopping")
                            s["active"] = False
                            continue
                        with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                            lf.write(f"\n[kiln] {s['name']} hit 429 quota limit; backing off {wait_s:.0f}s\n")
                        print(f"[kiln] {s['name']} hit 429 quota limit; backing off {wait_s:.0f}s")
                        s["wake_at"] = now + wait_s
                        continue
                    s["failures"] += 1
                    if s["failures"] >= 3:
                        print(f"[kiln] {s['name']} failed {s['failures']} times in a row (rc={rc}); "
                              f"see {s['ws'] / 'agent.log'}")
                        s["active"] = False
                        continue
                    wait_s = 5.0 * s["failures"]
                    with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                        lf.write(f"\n[kiln] {s['name']} exited (rc={rc}); retrying in {wait_s:.0f}s\n")
                    s["wake_at"] = now + wait_s
                    continue
                s["failures"] = 0
                s["quota_retries"] = 0
                with open(s["ws"] / "agent.log", "a", encoding="utf-8") as lf:
                    lf.write(f"\n[kiln] {s['name']} exited cleanly; respawning\n")
                p, log = _spawn_agent(s["cmd"], s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                s.update(proc=p, log=log, wake_at=None)
                info = f"provider={s['provider']}" + (f", model={s['model']}" if s.get("model") else "")
                print(f"[kiln] respawned {s['name']} (pid {p.pid}, {info})")
    except KeyboardInterrupt:
        pass
    finally:
        for s in slots:
            p = s["proc"]
            if p is not None and p.poll() is None:
                p.send_signal(signal.SIGTERM)
                print(f"[kiln] stopped {s['name']}")
            if s.get("log") is not None and not s["log"].closed:
                s["log"].close()
