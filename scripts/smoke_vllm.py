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

from kiln.bench.correctness import compare_tokens
from kiln.bench.traffic import load_traffic
from kiln.config import RunConfig
from kiln.engine import make_engine

BROKEN = '''
import types, torch
def make_forward(model):
    orig = type(model).forward
    def fwd(self, *a, **k):
        out = orig(self, *a, **k)
        return out * 0.5 if isinstance(out, torch.Tensor) else out
    return types.MethodType(fwd, model)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--tier", default="fast")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    run = RunConfig.load(a.config)
    tier = next(t for t in run.tiers if t.name == a.tier)
    reqs = load_traffic(run.traffic_file)
    cc = run.correctness
    probe = [r.prompt_ids for r in reqs[: cc.num_prompts]]

    t0 = time.time()
    eng = make_engine(run.engine, tier)
    eng.start()
    print(f"[1] boot OK in {time.time() - t0:.0f}s")

    base_tok = eng.greedy_outputs(probe, cc.max_new_tokens)
    base = eng.measure(reqs, tier.repeats, trace=False)
    print(f"[2] baseline decode {base.decode_tps:.1f} tok/s, variance {base.variance_pct:.2f}%")

    speedups = []
    for i in range(a.rounds):
        t = time.time()
        eng.hot_reload("candidates/baseline_forward.py")
        reload_s = time.time() - t
        tok = eng.greedy_outputs(probe, cc.max_new_tokens)
        corr = compare_tokens(base_tok, tok, cc)
        m = eng.measure(reqs, tier.repeats, trace=False)
        speedups.append(m.decode_tps / base.decode_tps)
        print(f"[3.{i}] reload {reload_s:.1f}s | {corr.detail} | speedup {speedups[-1]:.4f}")
        assert corr.token_match == 1.0, "baseline candidate changed outputs -> reload path is broken"
        eng.teardown_candidate()
        tok = eng.greedy_outputs(probe, cc.max_new_tokens)
        assert compare_tokens(base_tok, tok, cc).token_match == 1.0, "teardown did not restore baseline"
    print("[4] teardown restores baseline: OK")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(BROKEN)
    try:
        eng.hot_reload(f.name)
        tok = eng.greedy_outputs(probe, cc.max_new_tokens)
        corr = compare_tokens(base_tok, tok, cc)
        print(f"[5] broken candidate: {corr.detail} -> {'REJECTED (good)' if not corr.ok else 'ACCEPTED (BAD: patch not taking effect!)'}")
    finally:
        eng.teardown_candidate()

    spread = 100 * (max(speedups) - min(speedups)) / statistics.mean(speedups)
    print(f"[6] noise floor across reloads: {spread:.2f}% (gains below this are not real)")
    eng.shutdown()


if __name__ == "__main__":
    main()
