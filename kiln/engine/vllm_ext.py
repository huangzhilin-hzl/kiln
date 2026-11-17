"""vLLM worker extension that adds hot reload (TKCC Fig. 3).

Loaded into every worker via `LLM(..., worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt")`.
vLLM mixes this class into the Worker, so `self` is the GPU Worker. Calls go through
`llm.collective_rpc("kiln_hot_reload", args=(path,))` and run on all TP/EP ranks.

Candidate contract (see candidates/baseline_forward.py):
    def make_forward(model) -> Callable   # same signature as model.forward
`make_forward` may also patch submodules in place (e.g. swap an MoE/attention forward).

Re-capture follows vLLM's own "drop graphs, recompile, recapture in a live engine"
recipe from elastic EP (vllm/distributed/elastic_ep/elastic_execute.py,
`_release_cuda_graphs` + `_switch_to_fresh_graph_pool` + `warm_and_capture`).
Verified against vLLM main @ 20d8574 (2026-11-16). Re-check these names on upgrade:
  Worker.model_runner, GPUModelRunner.{get_model, cudagraph_manager, speculator, _dummy_run},
  CudaGraphManager.release_graphs, reset_compile_wrapper, {lock,unlock}_workspace,
  preserve_serving_state, Worker.compile_or_warm_up_model.

NOTE: Calling `capture_model()` on its own does NOT work. Graphs that were already
captured trip `assert desc not in self.graphs`, and the workspace is locked after
the first capture.
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import time
import types


class HotReloadExt:
    # Attribute names must not collide with anything on vLLM's Worker (it asserts).
    _kiln_baseline_forward = None
    _kiln_patched: list | None = None
    _kiln_module: types.ModuleType | None = None

    # ------------------------------------------------------------------ public RPCs
    def kiln_hot_reload(self, path: str) -> float:
        """Install candidate `path`, recompile + recapture. Returns seconds taken."""
        t0 = time.perf_counter()
        model = self.model_runner.get_model()
        if self._kiln_baseline_forward is None:
            self._kiln_baseline_forward = model.__dict__.get("forward")  # None => class method
        name = f"kiln_candidate_{abs(hash((path, t0)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

        new_fwd = mod.make_forward(model)
        if new_fwd is not None:
            model.forward = new_fwd
        self._kiln_module = mod
        self._kiln_rebuild()
        return time.perf_counter() - t0

    def kiln_teardown(self) -> float:
        """Restore the baseline forward (and any submodule patches the candidate undid)."""
        t0 = time.perf_counter()
        model = self.model_runner.get_model()
        mod = self._kiln_module
        if mod is not None and hasattr(mod, "restore"):
            mod.restore(model)             # candidates that patch submodules should define this
        if self._kiln_baseline_forward is None:
            model.__dict__.pop("forward", None)
        else:
            model.forward = self._kiln_baseline_forward
        if mod is not None:
            sys.modules.pop(mod.__name__, None)
            self._kiln_module = None
        self._kiln_rebuild()
        return time.perf_counter() - t0

    def kiln_info(self) -> dict:
        r = self.model_runner
        return {
            "model_cls": type(r.get_model()).__name__,
            "enforce_eager": self.model_config.enforce_eager,
            "compile_mode": str(self.vllm_config.compilation_config.mode),
            "cudagraph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
            "has_speculator": r.speculator is not None,
            "num_graphs": len(getattr(r.cudagraph_manager, "graphs", {}) or {}),
        }

    # ------------------------------------------------------------------ internals
    def _kiln_release_graphs(self) -> None:
        import torch
        from vllm.compilation.wrapper import reset_compile_wrapper
        from vllm.config import set_current_vllm_config
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        runner = self.model_runner
        if runner.cudagraph_manager is not None:
            runner.cudagraph_manager.release_graphs()
        spec = runner.speculator
        for attr in vars(spec).values() if spec else ():
            if isinstance(attr, CudaGraphManager):
                attr.release_graphs()
        # Model-specific inner graphs (e.g. DeepSeek-V4.1 decoder-replay graphs).
        replay = getattr(getattr(runner, "model_state", None), "replay_cudagraphs", None)
        if isinstance(replay, CudaGraphManager):
            replay.release_graphs()

        # Drop compiled code so torch.compile re-traces the *patched* modules.
        torch.compiler.reset()
        with set_current_vllm_config(self.vllm_config):
            reset_compile_wrapper(runner.get_model())
            if draft := getattr(spec, "model", None):
                reset_compile_wrapper(draft)

        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        self._kiln_fresh_graph_pool()

    def _kiln_fresh_graph_pool(self) -> None:
        """Capture into a fresh pool (the old one may still be referenced)."""
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.platforms import current_platform
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        platform_cls = type(current_platform)
        if platform_cls._global_graph_pool is None:
            return
        new_pool = current_platform.graph_pool_handle()
        platform_cls._global_graph_pool = new_pool
        runner = self.model_runner
        managers = [runner.cudagraph_manager]
        if runner.speculator:
            managers += [a for a in vars(runner.speculator).values() if isinstance(a, CudaGraphManager)]
        replay = getattr(getattr(runner, "model_state", None), "replay_cudagraphs", None)
        if isinstance(replay, CudaGraphManager):
            managers.append(replay)
        for m in managers:
            if m is not None and getattr(m, "pool", None) is not None:
                m.pool = new_pool
        for w in [*CUDAGraphWrapper._all_instances, *BreakableCUDAGraphWrapper._all_instances]:
            w.graph_pool = new_pool

    def _kiln_rebuild(self) -> None:
        """Release graphs/compiled code, re-warm (recompiles), re-capture."""
        from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
        from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

        self._kiln_release_graphs()
        unlock_workspace()
        runner = self.model_runner
        with preserve_serving_state(runner):   # warmup writes go to the null KV block
            runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
            self.compile_or_warm_up_model()
        lock_workspace()
