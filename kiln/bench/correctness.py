"""Correctness gate: candidate greedy outputs + optional logprob diff vs. baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass

from kiln.config import CorrectnessConfig


@dataclass
class CorrectnessResult:
    ok: bool
    token_match: float
    detail: str
    max_logit_abs_diff: float | None = None


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


def compare_logprobs(
    baseline_logprobs: list[list[dict[int, float]]] | None,
    candidate_logprobs: list[list[dict[int, float]]] | None,
    cfg: CorrectnessConfig,
) -> tuple[bool, float | None, str]:
    """Compare per-step `{token_id: logprob}` dicts on matching tokens against `max_logit_abs_diff`."""
    if baseline_logprobs is None or candidate_logprobs is None:
        return True, None, ""
    if len(baseline_logprobs) != len(candidate_logprobs):
        return (
            False,
            None,
            f"logprob sequence count mismatch: {len(candidate_logprobs)} vs {len(baseline_logprobs)}",
        )

    max_diff = 0.0
    compared = 0
    for b_seq, c_seq in zip(baseline_logprobs, candidate_logprobs):
        for b_step, c_step in zip(b_seq, c_seq):
            if not b_step or not c_step:
                continue
            for tid, b_val in b_step.items():
                if tid not in c_step:
                    continue
                try:
                    fb = float(b_val)
                    fc = float(c_step[tid])
                except (TypeError, ValueError):
                    return False, None, "non-numeric logprob detected"
                if not (math.isfinite(fb) and math.isfinite(fc)):
                    return False, None, "non-finite logprob detected"
                diff = abs(fb - fc)
                if diff > max_diff:
                    max_diff = diff
                compared += 1

    if compared == 0:
        return True, None, ""
    ok = max_diff <= cfg.max_logit_abs_diff
    return ok, max_diff, f"max logit abs diff {max_diff:.4f} (max {cfg.max_logit_abs_diff})"


def evaluate_correctness(
    baseline_tokens: list[list[int]],
    candidate_tokens: list[list[int]],
    cfg: CorrectnessConfig,
    baseline_logprobs: list[list[dict[int, float]]] | None = None,
    candidate_logprobs: list[list[dict[int, float]]] | None = None,
) -> CorrectnessResult:
    """Evaluate greedy token overlap and, when available, per-step logprob magnitude drift."""
    tok = compare_tokens(baseline_tokens, candidate_tokens, cfg)
    lp_ok, max_diff, lp_detail = compare_logprobs(baseline_logprobs, candidate_logprobs, cfg)
    ok = tok.ok and lp_ok
    detail = tok.detail if not lp_detail else f"{tok.detail}; {lp_detail}"
    return CorrectnessResult(
        ok=ok,
        token_match=tok.token_match,
        detail=detail,
        max_logit_abs_diff=max_diff,
    )
