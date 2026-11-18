"""GPU smoke test: run this FIRST on the target box, before `kiln serve`.

    python scripts/smoke_vllm.py -c configs/dsv41_flash_h20.yaml --tier fast

Checks, in order:
  1. Engine boots with the kiln worker extension (prints kiln_info).
  2. Baseline greedy tokens + decode TPS.
  3. Hot reload the baseline candidate -> tokens identical, speedup ~1.00, reload time.
  4. Teardown -> tokens identical again.
  5. Hot reload a deliberately broken candidate -> must be rejected by the correctness check.
  6. Repeat 3 a few times -> measure run-to-run variance (your noise floor).
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from kiln.bench.correctness import compare_tokens
from kiln.bench.traffic import load_traffic
from kiln.config import RunConfig
from kiln.engine import make_engine

BROKEN = '''
CORRUPT = True
_original_instance_logits = None
_had_instance_logits = False

def make_forward(model):
    import torch
    global _original_instance_logits, _had_instance_logits
    _had_instance_logits = "compute_logits" in vars(model)
    _original_instance_logits = vars(model).get("compute_logits")
    original = model.compute_logits

    def incorrect_logits(*args, **kwargs):
        logits = original(*args, **kwargs)
        if logits is None:
            return None
        forced = torch.full_like(logits, -float("inf"))
        forced[..., REPLACEMENT_TOKEN] = 0
        return forced

    model.compute_logits = incorrect_logits

def restore(model):
    del model.compute_logits
    if _had_instance_logits:
        model.compute_logits = _original_instance_logits
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--tier", default="fast")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    if a.rounds < 1:
        ap.error("--rounds must be at least 1")

    run = RunConfig.load(a.config)
    tier = next(t for t in run.tiers if t.name == a.tier)
    tok_name = None if (tier.engine_kind or run.engine.kind) == "mock" else (run.engine.tokenizer or run.engine.model)
    reqs = load_traffic(run.traffic_file, limit=tier.max_requests, model_name=tok_name)
    cc = run.correctness
    probe = [r.prompt_ids for r in reqs[: cc.num_prompts]]

    t0 = time.time()
    eng = make_engine(run.engine, tier)
    try:
        eng.start()
        print(f"[1] boot OK in {time.time() - t0:.0f}s")
        run_checks(eng, probe, reqs, cc, tier.repeats, a.rounds)
    finally:
        eng.shutdown()


def run_checks(eng, probe, reqs, cc, repeats, rounds):
    base_tok = eng.greedy_outputs(probe, cc.max_new_tokens)
    if len(base_tok) != len(probe) or not base_tok or not all(base_tok):
        raise RuntimeError("Baseline returned incomplete smoke-test outputs")
    base = eng.measure(reqs, repeats, trace=False)
    print(f"[2] baseline decode {base.decode_tps:.1f} tok/s, variance {base.variance_pct:.2f}%")

    speedups = []
    for i in range(rounds):
        t = time.time()
        eng.hot_reload("candidates/baseline_forward.py")
        reload_s = time.time() - t
        tok = eng.greedy_outputs(probe, cc.max_new_tokens)
        corr = compare_tokens(base_tok, tok, cc)
        m = eng.measure(reqs, repeats, trace=False)
        speedups.append(m.decode_tps / base.decode_tps)
        print(f"[3.{i}] reload {reload_s:.1f}s | {corr.detail} | speedup {speedups[-1]:.4f}")
        if corr.token_match != 1.0:
            raise RuntimeError("Baseline candidate changed outputs")
        eng.teardown_candidate()
        tok = eng.greedy_outputs(probe, cc.max_new_tokens)
        if compare_tokens(base_tok, tok, cc).token_match != 1.0:
            raise RuntimeError("Teardown did not restore baseline")
    print("[4] teardown restores baseline: OK")

    seen_tokens = {token for tokens in base_tok for token in tokens}
    replacement = next(token for token in range(len(seen_tokens) + 1) if token not in seen_tokens)
    with tempfile.TemporaryDirectory() as directory:
        broken = Path(directory) / "broken.py"
        broken.write_text(BROKEN.replace("REPLACEMENT_TOKEN", str(replacement)))
        eng.hot_reload(str(broken))
        try:
            tok = eng.greedy_outputs(probe, cc.max_new_tokens)
            corr = compare_tokens(base_tok, tok, cc)
            if corr.ok:
                raise RuntimeError("Broken candidate passed the correctness check")
            print(f"[5] broken candidate rejected: {corr.detail}")
        finally:
            eng.teardown_candidate()

    spread = 100 * (max(speedups) - min(speedups)) / statistics.mean(speedups)
    print(f"[6] noise floor across reloads: {spread:.2f}% (gains below this are not real)")


if __name__ == "__main__":
    main()
