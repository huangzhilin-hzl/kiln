"""Persistent engine interface.

Paper idea (TKCC §4.2): engine start-up (weights, KV sizing, autotune, graph capture)
is decoupled from the forward pass. We start the engine ONCE, then hot-reload
agent-written forward passes into it and measure.
"""
from __future__ import annotations

import abc
import os
import signal
import subprocess
from dataclasses import dataclass, field

from kiln.bench.traffic import Request


def reap_zombies() -> int:
    """Reap any finished child processes without blocking; return count reaped."""
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid <= 0:
            break
        reaped += 1
    return reaped


def shutdown_process_group(proc: subprocess.Popen | None, timeout: float = 120.0) -> None:
    """Terminate a process group cleanly, escalate to SIGKILL if needed, wait, and reap zombies."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                pass
    reap_zombies()


class InvalidBenchmark(ValueError):
    """A completed measurement that cannot be scored."""

    def __init__(self, message: str, extra: dict):
        super().__init__(message)
        self.extra = extra


@dataclass
class MeasureResult:
    decode_tps: float                  # output tokens / s during pure-decode phase
    prefill_tps: float                 # input tokens / s during pure-prefill phase
    ttft_s: float                      # mean time-to-first-token
    runs_decode_tps: list[float] = field(default_factory=list)
    kernel_report: str | None = None
    trace_analysis: dict | None = None
    extra: dict = field(default_factory=dict)   # e.g. AgentX P90 interactivity, export path

    @property
    def variance_pct(self) -> float:
        r = self.runs_decode_tps
        if len(r) < 2:
            return 0.0
        mean = sum(r) / len(r)
        return 100.0 * (max(r) - min(r)) / mean


def bucket_tps_for_requests(
    requests: list[Request], decode_tps: float, prefill_tps: float
) -> dict[str, float]:
    """Classify offline measurement into M-regime buckets (`decode_small`, `decode_mid`, `prefill_large`)."""
    out: dict[str, float] = {}
    n_req = len(requests)
    if decode_tps > 0:
        if n_req <= 48:
            out["decode_small"] = decode_tps
        elif n_req <= 256:
            out["decode_mid"] = decode_tps
    if prefill_tps > 0:
        out["prefill_large"] = prefill_tps
    return out


class Engine(abc.ABC):
    """One persistent engine == one measurement tier."""

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def hot_reload(self, candidate_path: str) -> None:
        """Swap the model forward for the candidate in `candidate_path`, re-capture graphs."""

    @abc.abstractmethod
    def teardown_candidate(self) -> None:
        """Restore the baseline forward."""

    @abc.abstractmethod
    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult: ...

    @abc.abstractmethod
    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Greedy token ids, used for correctness checks."""

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        """Return `(token_ids, per_step_logprobs)` when supported by the backend."""
        return self.greedy_outputs(prompts, max_new_tokens), None

    def is_alive(self) -> bool:
        """Return whether the backing engine/worker processes are healthy and alive."""
        return True

    def capture_snapshots(self, snapshot_dir: str, m_buckets: list[int]) -> dict | None:
        """Optionally capture representative submodule tensor snapshots for `kiln micro`."""
        return None

    @abc.abstractmethod
    def shutdown(self) -> None: ...
