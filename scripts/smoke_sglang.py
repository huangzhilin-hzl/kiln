"""GPU smoke test for SGLang engines (`sglang_offline` and `sglang_server`).

    python -m scripts.smoke_sglang -c configs/llama1b_sglang.yaml --tier fast
    python -m scripts.smoke_sglang -c configs/llama1b_sglang.yaml --tier slow

Runs the same 6-step verification as `scripts/smoke_vllm.py`:
  1. Engine boots with kiln's SGLang hot-reload hook (prints kiln_info).
  2. Baseline greedy tokens + decode TPS.
  3. Hot reload `candidates/baseline_forward.py` -> tokens identical, speedup ~1.00.
  4. Teardown -> tokens identical again.
  5. Hot reload a deliberately broken candidate -> rejected by correctness check.
  6. Measure noise floor across reloads.
"""
from __future__ import annotations

from scripts.smoke_vllm import main, run_checks

__all__ = ["main", "run_checks"]

if __name__ == "__main__":
    main()
