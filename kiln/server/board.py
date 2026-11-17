"""Shared message board for agents (TKCC Fig. 5c: frequent communication -> 4.7x faster).

Append-only JSONL. Agents post what they tried, what worked, and what they're
working on now, so they don't duplicate each other's experiments.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Board:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def post(self, agent: str, text: str, kind: str = "note") -> dict:
        msg = {"ts": time.time(), "agent": agent, "kind": kind, "text": text}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return msg

    def read(self, since: float = 0.0, limit: int = 200) -> list[dict]:
        rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        return [r for r in rows if r["ts"] > since][-limit:]
