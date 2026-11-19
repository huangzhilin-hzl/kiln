"""Spawn N parallel coding agents, each in its own workspace dir.

Any CLI agent that accepts a prompt on stdin (Codex `codex exec ... -`, Claude Code `claude -p`)
or via an argv `{prompt}` placeholder (Antigravity `agy ... -p {prompt}`) works. Configure via
`agents.command` in the run YAML.
"""
from __future__ import annotations

import os
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
        slots.append({"name": name, "ws": ws, "env": env, "prompt": prompt,
                      "proc": p, "log": log, "failures": 0, "active": True})
        print(f"[kiln] started {name} (pid {p.pid}) in {ws}")

    deadline = time.time() + run.agents.budget_hours * 3600
    try:
        while time.time() < deadline and any(s["active"] for s in slots):
            time.sleep(5)
            for s in slots:
                if not s["active"]:
                    continue
                rc = s["proc"].poll()
                if rc is None:
                    continue
                s["log"].close()
                if not run.agents.respawn_on_exit or time.time() >= deadline:
                    s["active"] = False
                    continue
                if rc != 0:
                    s["failures"] += 1
                    if s["failures"] >= 3:
                        print(f"[kiln] {s['name']} failed {s['failures']} times in a row (rc={rc}); "
                              f"see {s['ws'] / 'agent.log'}")
                        s["active"] = False
                        continue
                else:
                    s["failures"] = 0
                p, log = _spawn_agent(cmd, s["ws"], s["env"], s["prompt"] + CONTINUE_NOTE)
                s.update(proc=p, log=log)
                print(f"[kiln] respawned {s['name']} (pid {p.pid})")
    except KeyboardInterrupt:
        pass
    finally:
        for s in slots:
            p = s["proc"]
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                print(f"[kiln] stopped {s['name']}")
            s["log"].close()
