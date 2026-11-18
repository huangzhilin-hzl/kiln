"""Correctness gate: candidate greedy outputs vs. baseline greedy outputs."""
from __future__ import annotations

from dataclasses import dataclass

from kiln.config import CorrectnessConfig


@dataclass
class CorrectnessResult:
    ok: bool
    token_match: float
    detail: str


def compare_tokens(baseline: list[list[int]], candidate: list[list[int]],
                   cfg: CorrectnessConfig) -> CorrectnessResult:
    if len(candidate) != len(baseline):
        return CorrectnessResult(False, 0.0,
                                 f"expected {len(baseline)} outputs, received {len(candidate)}")
    total = match = 0
    for b, c in zip(baseline, candidate):
        n = min(len(b), len(c))
        total += max(len(b), len(c))
        match += sum(1 for i in range(n) if b[i] == c[i])
    frac = match / total if total else 0.0
    ok = total > 0 and frac >= cfg.min_token_match
    return CorrectnessResult(ok=ok, token_match=frac,
                             detail=f"token match {frac:.4f} (min {cfg.min_token_match})")
