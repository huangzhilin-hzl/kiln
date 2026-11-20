"""Spawn N parallel coding agents, each in its own workspace dir.

Any CLI agent that accepts a prompt on stdin (Codex `codex exec ... -`, Claude Code `claude -p`)
or via an argv `{prompt}` placeholder (Antigravity `agy ... -p {prompt}`) works. Configure via
`agents.command` in the run YAML.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from kiln.config import RunConfig

PROMPT = Path(__file__).with_name("prompt.md")
REPO = Path(__file__).resolve().parents[2]

CONTINUE_NOTE = (
    "\n\n## Continuation\n"
    "You are resuming an ongoing optimization run in this workspace. Inspect your current `forward.py`, "
    "check `kiln board` and `kiln leaderboard`, and continue iterating on new speedups.\n"
)

_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", '"error_code":429', "Individual quota reached")
_RESET_RE = re.compile(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def _parse_quota_backoff(log_path: Path, slot_idx: int = 0) -> float | None:
    """Return backoff seconds if the latest agent run ended with a 429 quota error."""
    if not log_path.exists():
        return None
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 8192))
        tail = f.read().decode("utf-8", errors="ignore")
    segment = tail.rsplit("[kiln]", 1)[-1][-4096:] if "[kiln]" in tail else tail[-4096:]
    if not any(m in segment for m in _QUOTA_MARKERS):
        return None
    base_wait = 300.0
    m = _RESET_RE.search(segment)
    if m and any(g is not None for g in m.groups()):
        hours = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = int(m.group(3) or 0)
        base_wait = max(60.0, min(3900.0, float(hours * 3600 + mins * 60 + secs)))
    return base_wait + 15.0 * slot_idx


def _spawn_agent(cmd: list[str], ws: Path, env: dict[str, str], prompt_text: str):
    log = open(ws / "agent.log", "ab")
    if any("{prompt}" in tok for tok in cmd):
        argv = [tok.replace("{prompt}", prompt_text) for tok in cmd]
        p = subprocess.Popen(argv, cwd=ws, env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        return p, log
    p = subprocess.Popen(cmd, cwd=ws, env=env,
                         stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
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
    cmd = shlex.split(run.agents.command)
    venv_bin = str(Path(sys.executable).parent)
    path_env = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
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
    if refs_dir.exists() and cmd and Path(cmd[0]).name == "agy" and "--add-dir" not in cmd:
        cmd = [cmd[0], "--add-dir", str(refs_dir), *cmd[1:]]
    slots: list[dict] = []
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
        env = {
            **os.environ,
            "PATH": path_env,
            "KILN_AGENT": name,
            "KILN_SERVER": server_url,
            "TRITON_CACHE_DIR": str(triton_dir),
        }
        p, log = _spawn_agent(cmd, ws, env, prompt)
        slots.append({
            "idx": i - 1, "name": name, "ws": ws, "env": env, "prompt": prompt,
            "proc": p, "log": log, "failures": 0, "quota_retries": 0,
            "wake_at": None, "active": True,
        })
        print(f"[kiln] started {name} (pid {p.pid}) in {ws}")

    deadline = time.time() + run.agents.budget_hours * 3600
    try:
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
                        p, log = _spawn_agent(cmd, s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                        s.update(proc=p, log=log, wake_at=None)
                        print(f"[kiln] respawned {s['name']} (pid {p.pid})")
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
                p, log = _spawn_agent(cmd, s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                s.update(proc=p, log=log, wake_at=None)
                print(f"[kiln] respawned {s['name']} (pid {p.pid})")
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
