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


def _snapshot_forwards(model) -> list[tuple]:
    """Capture original instance and class `.forward` bindings across `model` and all submodules."""
    snap = []
    it = model.named_modules() if hasattr(model, "named_modules") else [("", model)]
    for _, sub in it:
        had_inst = "forward" in sub.__dict__
        inst_fwd = sub.__dict__.get("forward")
        cls = type(sub)
        cls_fwd = getattr(cls, "forward", None)
        snap.append((sub, had_inst, inst_fwd, cls, cls_fwd))
    return snap


def _restore_forwards(snap: list[tuple] | None) -> None:
    if not snap:
        return
    for sub, had_inst, inst_fwd, cls, cls_fwd in snap:
        if had_inst:
            sub.__dict__["forward"] = inst_fwd
        else:
            sub.__dict__.pop("forward", None)
        if cls_fwd is not None and getattr(cls, "forward", None) is not cls_fwd:
            cls.forward = cls_fwd


class HotReloadExt:
    # Attribute names must not collide with anything on vLLM's Worker (it asserts).
    _kiln_patched: list | None = None
    _kiln_module: types.ModuleType | None = None
    _kiln_dirty: bool = False

    # ------------------------------------------------------------------ public RPCs
    def kiln_hot_reload(self, path: str) -> float:
        """Install candidate `path`, recompile + recapture. Returns seconds taken."""
        t0 = time.perf_counter()
        model = self.model_runner.get_model()
        self.kiln_teardown()

        name = f"kiln_candidate_{abs(hash((path, t0)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        self._kiln_patched = _snapshot_forwards(model)
        self._kiln_module = mod
        sys.modules[name] = mod
        rebuild_started = False
        try:
            spec.loader.exec_module(mod)
            new_fwd = mod.make_forward(model)
            if new_fwd is not None:
                model.forward = new_fwd
            rebuild_started = True
            self._kiln_rebuild()
        except Exception:
            self._kiln_restore_baseline(rebuild_started)
            raise
        return time.perf_counter() - t0

    def kiln_teardown(self) -> float:
        """Restore baseline `.forward` across `model` and all submodules, then re-capture."""
        t0 = time.perf_counter()
        if self._kiln_dirty:
            raise RuntimeError("Kiln candidate rollback failed; restart the engine")
        if self._kiln_patched is not None:
            self._kiln_restore_baseline(rebuild=True)
        return time.perf_counter() - t0

    def _kiln_restore_baseline(self, rebuild: bool) -> None:
        mod = self._kiln_module
        try:
            try:
                if mod is not None and hasattr(mod, "restore"):
                    mod.restore(self.model_runner.get_model())
            finally:
                _restore_forwards(self._kiln_patched)
            if rebuild:
                self._kiln_rebuild()
        except Exception as exc:
            self._kiln_dirty = True
            raise RuntimeError("Kiln candidate rollback failed; restart the engine") from exc
        finally:
            if mod is not None:
                sys.modules.pop(mod.__name__, None)
            self._kiln_module = None
            self._kiln_patched = None

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
        kv_cache = runner.extensible_kv_cache
        committed_blocks = kv_cache.num_committed_blocks if kv_cache is not None else None
        try:
            with preserve_serving_state(runner):   # warmup writes go to the null KV block
                runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
                self.compile_or_warm_up_model()
                # Startup warmup shrinks the physical KV allocation. The live
                # scheduler still owns the original block range after recapture.
                if committed_blocks is not None:
                    kv_cache.commit(committed_blocks)
        finally:
            lock_workspace()
