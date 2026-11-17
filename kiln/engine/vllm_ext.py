"""vLLM worker extension that adds hot reload (TKCC Fig. 3).

Loaded into every worker process via `LLM(..., worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt")`.
Its methods are invoked on all ranks with `llm.collective_rpc("kiln_hot_reload", args=(path,))`.

Candidate contract (see candidates/baseline_forward.py):
    def make_forward(model) -> Callable   # same signature as model.forward

NOTE: vLLM internals move fast. Things to verify against your installed version:
  * `self.model_runner.model` – the nn.Module (may be wrapped, e.g. CUDAGraphWrapper in recent V1).
  * `self.model_runner.capture_model()` – re-captures CUDA graphs.
  * If torch.compile is enabled, patching `forward` may be bypassed by the compiled
    callable; start with compilation disabled (see configs/llama1b_vllm.yaml) and add
    compile back once the loop works.
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import types


def _unwrap(model):
    # Peel off common wrappers (CUDAGraphWrapper, etc.) to reach the real nn.Module.
    seen = 0
    while hasattr(model, "runnable") and seen < 4:
        model = model.runnable
        seen += 1
    return model


class HotReloadExt:
    _kiln_baseline = None
    _kiln_module: types.ModuleType | None = None

    def kiln_hot_reload(self, path: str) -> str:
        import torch

        model = _unwrap(self.model_runner.model)
        if self._kiln_baseline is None:
            self._kiln_baseline = model.forward

        name = f"kiln_candidate_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

        model.forward = mod.make_forward(model)   # <- the patch
        self._kiln_module = mod
        torch.cuda.synchronize()
        self._kiln_recapture()
        return "ok"

    def kiln_teardown(self) -> str:
        model = _unwrap(self.model_runner.model)
        if self._kiln_baseline is not None:
            model.forward = self._kiln_baseline
        if self._kiln_module is not None:
            sys.modules.pop(self._kiln_module.__name__, None)
            self._kiln_module = None
        gc.collect()
        self._kiln_recapture()
        return "ok"

    def kiln_recapture(self) -> str:
        self._kiln_recapture()
        return "ok"

    def _kiln_recapture(self) -> None:
        import torch

        runner = self.model_runner
        if getattr(runner, "model_config", None) is not None and runner.model_config.enforce_eager:
            return
        if hasattr(runner, "capture_model"):
            runner.capture_model()
        torch.cuda.synchronize()
