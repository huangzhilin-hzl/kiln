"""Spawn N parallel coding agents, each in its own workspace dir.

Any CLI agent that accepts a prompt on stdin works (Claude Code `claude -p`, Codex `codex exec`,
Gemini CLI, ...). Configure via `agents.command` in the run YAML.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from kiln.config import RunConfig

PROMPT = Path(__file__).with_name("prompt.md")
REPO = Path(__file__).resolve().parents[2]


def launch(run: RunConfig, baseline: str = "candidates/baseline_forward.py") -> None:
    root = Path(run.workdir) / run.name / "agents"
    root.mkdir(parents=True, exist_ok=True)
    n = run.agents.num_agents
    procs: list[tuple[str, subprocess.Popen]] = []

    for i in range(1, n + 1):
        name = f"agent_{i}"
        ws = root / name
        ws.mkdir(exist_ok=True)
        if not (ws / "forward.py").exists():
            shutil.copy(REPO / baseline, ws / "forward.py")
        prompt = PROMPT.read_text().format(agent=name, n=n, model=run.engine.model,
                                           engine=run.engine.kind)
        (ws / "PROMPT.md").write_text(prompt)
        env = {**os.environ, "KILN_AGENT": name, "KILN_SERVER": run.server_url}
        log = open(ws / "agent.log", "ab")
        p = subprocess.Popen(shlex.split(run.agents.command), cwd=ws, env=env,
                             stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        p.stdin.write(prompt.encode())
        p.stdin.close()
        procs.append((name, p))
        print(f"[kiln] started {name} (pid {p.pid}) in {ws}")

    deadline = time.time() + run.agents.budget_hours * 3600
    try:
        while time.time() < deadline and any(p.poll() is None for _, p in procs):
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        for name, p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                print(f"[kiln] stopped {name}")
