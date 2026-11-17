"""Persistent engine interface.

Paper idea (TKCC §4.2): engine start-up (weights, KV sizing, autotune, graph capture)
is decoupled from the forward pass. We start the engine ONCE, then hot-reload
agent-written forward passes into it and measure.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from kiln.bench.traffic import Request


@dataclass
class MeasureResult:
    decode_tps: float                  # output tokens / s during pure-decode phase
    prefill_tps: float                 # input tokens / s during pure-prefill phase
    ttft_s: float                      # mean time-to-first-token
    runs_decode_tps: list[float] = field(default_factory=list)
    kernel_report: str | None = None
    extra: dict = field(default_factory=dict)   # e.g. AgentX P90 interactivity, export path

    @property
    def variance_pct(self) -> float:
        r = self.runs_decode_tps
        if len(r) < 2:
            return 0.0
        mean = sum(r) / len(r)
        return 100.0 * (max(r) - min(r)) / mean


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

    @abc.abstractmethod
    def shutdown(self) -> None: ...
