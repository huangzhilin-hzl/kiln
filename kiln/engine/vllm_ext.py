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
import os
import sys
import time
import types
from pathlib import Path

from kiln.patches import inspect_candidate_source, normalize_pd_role

from .patching import restore_forwards as _restore_forwards
from .patching import snapshot_forwards as _snapshot_forwards


class HotReloadExt:
    # Attribute names must not collide with anything on vLLM's Worker (it asserts).
    _kiln_patched: list | None = None
    _kiln_module: types.ModuleType | None = None
    _kiln_dirty: bool = False
    _kiln_baseline_graphs_valid: bool = True
    _kiln_graphs_released: bool = False
    _kiln_last_skipped_graph_rebuild: bool = False
    _kiln_snapshot_collector: object | None = None

    # ------------------------------------------------------------------ phase / graph guards
    def _kiln_is_decode_only_graph_engine(self) -> bool:
        vllm_cfg = getattr(self, "vllm_config", None)
        model_cfg = getattr(self, "model_config", None)
        runner = getattr(self, "model_runner", None)
        if vllm_cfg is None or runner is None:
            return False
        comp_cfg = getattr(vllm_cfg, "compilation_config", None)
        if comp_cfg is None:
            return False
        mode_val = getattr(comp_cfg, "mode", None)
        is_no_compile = (
            mode_val in (0, "0", None)
            or str(mode_val).upper().endswith("NONE")
        )
        if not is_no_compile:
            return False
        if bool(getattr(model_cfg, "enforce_eager", False)):
            return True
        cg_mode = str(getattr(comp_cfg, "cudagraph_mode", "") or "").upper()
        if not (
            cg_mode.endswith("FULL_DECODE_ONLY")
            or cg_mode.endswith("DECODE_ONLY")
            or cg_mode.endswith("NONE")
        ):
            return False
        if getattr(runner, "speculator", None) is not None:
            return False
        if getattr(getattr(runner, "model_state", None), "replay_cudagraphs", None) is not None:
            return False
        return True

    def _kiln_can_skip_graph_rebuild(self, source: str | None, pd_role: str = "both") -> bool:
        if not source:
            return False
        info = inspect_candidate_source(source)
        if not info.get("syntax_ok"):
            return False
        eff_phase = info.get("effective_phase", "all")
        role = normalize_pd_role(pd_role)
        if (
            info.get("only_kiln_patches")
            and role in ("prefill", "decode")
            and eff_phase in ("prefill", "decode")
            and eff_phase != role
        ):
            return True
        if eff_phase == "prefill" and self._kiln_is_decode_only_graph_engine():
            return True
        return False

    def _kiln_rebuild_baseline(self) -> None:
        try:
            self._kiln_rebuild()
            self._kiln_baseline_graphs_valid = True
            self._kiln_graphs_released = False
        except Exception as exc:
            self._kiln_dirty = True
            raise RuntimeError("Kiln baseline graph rebuild failed; restart the engine") from exc

    # ------------------------------------------------------------------ public RPCs
    def kiln_hot_reload(self, path: str, pd_role: str | None = None) -> float:
        """Install candidate `path`, recompile + recapture when needed. Returns seconds taken."""
        t0 = time.perf_counter()
        if self._kiln_dirty:
            raise RuntimeError("Kiln candidate rollback failed; restart the engine")
        if self._kiln_patched is not None:
            self.kiln_teardown_lazy()

        model = self.model_runner.get_model()
        role = normalize_pd_role(pd_role if pd_role is not None else os.environ.get("KILN_PD_ROLE"))
        p = Path(path)
        source = p.read_text(encoding="utf-8") if p.is_file() else None
        skip_rebuild = self._kiln_can_skip_graph_rebuild(source, role)
        if skip_rebuild and not self._kiln_baseline_graphs_valid:
            self._kiln_rebuild_baseline()

        name = f"kiln_candidate_{abs(hash((path, t0)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load candidate module from {path}")
        mod = importlib.util.module_from_spec(spec)
        self._kiln_patched = _snapshot_forwards(model)
        self._kiln_module = mod
        try:
            model._kiln_pd_role = role
        except Exception:
            pass
        sys.modules[name] = mod
        rebuild_started = False
        had_env = "KILN_PD_ROLE" in os.environ
        prev_env = os.environ.get("KILN_PD_ROLE")
        try:
            os.environ["KILN_PD_ROLE"] = role
            try:
                spec.loader.exec_module(mod)
                new_fwd = mod.make_forward(model)
                if new_fwd is not None:
                    model.forward = new_fwd
            finally:
                if had_env and prev_env is not None:
                    os.environ["KILN_PD_ROLE"] = prev_env
                else:
                    os.environ.pop("KILN_PD_ROLE", None)
            if skip_rebuild:
                self._kiln_last_skipped_graph_rebuild = True
            else:
                self._kiln_last_skipped_graph_rebuild = False
                rebuild_started = True
                self._kiln_baseline_graphs_valid = False
                self._kiln_rebuild()
                self._kiln_graphs_released = False
        except Exception:
            self._kiln_restore_baseline(rebuild=rebuild_started)
            raise
        return time.perf_counter() - t0

    def kiln_teardown_lazy(self) -> dict:
        """Restore baseline `.forward` and drop candidate graphs without recapturing baseline graphs."""
        t0 = time.perf_counter()
        if self._kiln_dirty:
            raise RuntimeError("Kiln candidate rollback failed; restart the engine")
        if self._kiln_patched is not None:
            self._kiln_restore_baseline(rebuild=False, release_if_invalid=True)
        return {
            "secs": time.perf_counter() - t0,
            "baseline_graphs_valid": self._kiln_baseline_graphs_valid,
        }

    def kiln_ensure_baseline(self) -> dict:
        """Ensure `.forward` is restored and baseline CUDA graphs are captured and valid."""
        t0 = time.perf_counter()
        if self._kiln_dirty:
            raise RuntimeError("Kiln candidate rollback failed; restart the engine")
        rebuilt = False
        if self._kiln_patched is not None:
            need_rebuild = not self._kiln_baseline_graphs_valid
            self._kiln_restore_baseline(rebuild=need_rebuild)
            rebuilt = need_rebuild
        elif not self._kiln_baseline_graphs_valid:
            self._kiln_rebuild_baseline()
            rebuilt = True
        return {
            "secs": time.perf_counter() - t0,
            "rebuilt": rebuilt,
            "baseline_graphs_valid": self._kiln_baseline_graphs_valid,
        }

    def kiln_teardown(self) -> float:
        """Restore baseline `.forward` across `model` and all submodules, then ensure baseline graphs."""
        return float(self.kiln_ensure_baseline()["secs"])

    def _kiln_restore_baseline(self, rebuild: bool, *, release_if_invalid: bool = False) -> None:
        mod = self._kiln_module
        model = None
        try:
            model = self.model_runner.get_model()
            try:
                if mod is not None and hasattr(mod, "restore"):
                    mod.restore(model)
            finally:
                _restore_forwards(self._kiln_patched)
            if rebuild:
                self._kiln_rebuild()
                self._kiln_baseline_graphs_valid = True
                self._kiln_graphs_released = False
            elif release_if_invalid and not self._kiln_baseline_graphs_valid and not self._kiln_graphs_released:
                if getattr(self, "vllm_config", None) is not None or "_kiln_release_graphs" in self.__dict__:
                    self._kiln_release_graphs()
                self._kiln_graphs_released = True
        except Exception as exc:
            self._kiln_dirty = True
            raise RuntimeError("Kiln candidate rollback failed; restart the engine") from exc
        finally:
            if model is not None and hasattr(model, "_kiln_pd_role"):
                try:
                    delattr(model, "_kiln_pd_role")
                except Exception:
                    pass
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
            "baseline_graphs_valid": self._kiln_baseline_graphs_valid,
            "last_skipped_graph_rebuild": self._kiln_last_skipped_graph_rebuild,
        }

    def kiln_enable_snapshot_capture(self, snapshot_dir: str, m_buckets=None, max_snapshot_mb: float = 64.0) -> dict:
        import json as _json
        from kiln.micro.snapshots import DEFAULT_M_BUCKETS, SnapshotCollector

        rank = getattr(self, "rank", 0)
        if rank not in (0, None):
            return {"rank": rank, "skipped": True}
        if self._kiln_snapshot_collector is not None:
            self._kiln_snapshot_collector.close()
        if isinstance(m_buckets, str):
            buckets = [int(x) for x in _json.loads(m_buckets)]
        elif m_buckets is not None:
            buckets = [int(x) for x in m_buckets]
        else:
            buckets = list(DEFAULT_M_BUCKETS)
        model = self.model_runner.get_model()
        collector = SnapshotCollector(snapshot_dir, m_buckets=buckets, max_snapshot_mb=float(max_snapshot_mb))
        attached = collector.attach(model)
        self._kiln_snapshot_collector = collector
        return {"rank": 0, "attached": attached}

    def kiln_finish_snapshot_capture(self) -> dict:
        collector = self._kiln_snapshot_collector
        self._kiln_snapshot_collector = None
        if collector is None:
            return {"snapshots": []}
        return collector.close()

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
        if not self._kiln_graphs_released:
            self._kiln_fresh_graph_pool()
        self._kiln_graphs_released = True

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
        vllm_cfg = getattr(self, "vllm_config", None)
        kernel_cfg = getattr(vllm_cfg, "kernel_config", None)
        has_autotune = kernel_cfg is not None and hasattr(kernel_cfg, "enable_flashinfer_autotune")
        prev_autotune = kernel_cfg.enable_flashinfer_autotune if has_autotune else None
        if has_autotune:
            kernel_cfg.enable_flashinfer_autotune = False
        try:
            with preserve_serving_state(runner):   # warmup writes go to the null KV block
                runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
                self.compile_or_warm_up_model()
                # Startup warmup shrinks the physical KV allocation. The live
                # scheduler still owns the original block range after recapture.
                if committed_blocks is not None:
                    kv_cache.commit(committed_blocks)
        finally:
            if has_autotune:
                kernel_cfg.enable_flashinfer_autotune = prev_autotune
            lock_workspace()
