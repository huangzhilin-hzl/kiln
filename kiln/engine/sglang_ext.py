"""SGLang worker hot-reload controller and control-plane hooks (TKCC Fig. 3).

SGLang does not expose a native `--worker-extension-cls` + `/collective_rpc` HTTP route
that broadcasts reliably across TP, PP, EP, and DP-attention ranks. Instead, kiln hooks
`Scheduler.update_weights_from_disk` (which SGLang's TokenizerManager, DataParallelController,
and Scheduler already broadcast synchronously to every rank in lockstep) when `model_path`
carries the `KILN_RPC_PREFIX` sentinel.

The hook is activated in spawned SGLang worker processes via `PYTHONPATH` pointing to
`kiln/engine/_sglang_bootstrap` + `KILN_SGLANG_HOOK=1`, with zero edits to SGLang itself.

Candidate contract (identical to vLLM, see `candidates/baseline_forward.py`):
    def make_forward(model) -> Callable | None
    def restore(model) -> None                  # optional, for state outside `.forward`
"""
from __future__ import annotations

import gc
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import types
from pathlib import Path

from .patching import restore_forwards, snapshot_forwards

KILN_RPC_PREFIX = "__kiln_rpc__:"

_TARGET_MODULES = {
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.scheduler_components.weight_updater",
    "sglang.srt.managers.tokenizer_manager",
}

_MODEL_RUNNER_GRAPH_ATTRS = (
    "decode_cuda_graph_runner",
    "prefill_cuda_graph_runner",
    "graph_runner",
    "decode_graph_runner",
    "prefill_graph_runner",
    "piecewise_cuda_graph_runner",
    "cuda_graph_runner",
)

_SPEC_WORKER_GRAPH_ATTRS = (
    "cuda_graph_runner",
    "cuda_graph_runner_for_draft_extend",
    "draft_runner",
    "graph_runner",
)

_GRAPH_CONTAINER_ATTRS = (
    "graphs",
    "_graphs",
    "cuda_graphs",
    "graph_buffers",
    "output_buffers",
    "_outputs",
    "_capture_inputs",
    "compile_graphs",
    "decode_graphs",
    "prefill_graphs",
)

_GLOBAL_POOL_MODULES = (
    "sglang.srt.model_executor.runner_utils.pool",
    "sglang.srt.model_executor.runner_utils",
    "sglang.srt.model_executor.runner",
    "sglang.srt.model_executor.cuda_graph_runner",
    "sglang.srt.model_executor.base_cuda_graph_runner",
    "sglang.srt.model_executor.runner.cuda_graph_runner",
    "sglang.srt.model_executor.runner.base_cuda_graph_runner",
    "sglang.srt.model_executor.cuda_graph_runner_utils",
    "sglang.srt.model_executor.piecewise_cuda_graph_runner",
)


# --------------------------------------------------------------------------- RPC codec & helpers
def encode_rpc(method: str, *args, **kwargs) -> str:
    return KILN_RPC_PREFIX + json.dumps({"method": method, "args": list(args), "kwargs": kwargs})


def decode_rpc(model_path: object) -> dict | None:
    if not isinstance(model_path, str) or not model_path.startswith(KILN_RPC_PREFIX):
        return None
    return json.loads(model_path[len(KILN_RPC_PREFIX):])


def parse_rpc_message(message: str | None):
    if not message:
        return None
    try:
        data = json.loads(message)
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data
    except (TypeError, json.JSONDecodeError):
        return message


def bootstrap_dir() -> Path:
    return Path(__file__).resolve().parent / "_sglang_bootstrap"


def bootstrap_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env dict that installs kiln's SGLang import hook in spawned workers."""
    env = dict(os.environ if base_env is None else base_env)
    bdir = str(bootstrap_dir())
    pkg_root = str(Path(__file__).resolve().parents[2])
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    parts: list[str] = []
    for p in (bdir, pkg_root, *existing):
        if p and p not in parts:
            parts.append(p)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["KILN_SGLANG_HOOK"] = "1"
    # Candidates may patch child `.forward` without changing Inductor cache keys.
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    return env


def merge_json_model_override_args(existing, overrides: dict) -> str:
    if existing is None or existing == "":
        base: dict = {}
    elif isinstance(existing, str):
        base = json.loads(existing)
    elif isinstance(existing, dict):
        base = dict(existing)
    else:
        raise TypeError(f"Unsupported json_model_override_args type: {type(existing).__name__}")
    base.update(overrides)
    return json.dumps(base)


def extract_output_ids(out: dict) -> list[int] | None:
    """Extract generated token IDs from an SGLang `/generate` or `Engine.generate` response."""
    ids = out.get("output_ids")
    if ids is not None:
        return [int(t) for t in ids]
    meta = out.get("meta_info") or {}
    ids = meta.get("output_ids")
    if ids is not None:
        return [int(t) for t in ids]
    logprobs = meta.get("output_token_logprobs")
    if isinstance(logprobs, list) and logprobs:
        extracted = []
        for entry in logprobs:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[1] is not None:
                extracted.append(int(entry[1]))
            else:
                return None
        return extracted
    return None


def extract_output_logprobs(out: dict) -> list[dict[int, float]] | None:
    """Extract per-step `{token_id: logprob}` dicts from an SGLang response when present."""
    meta = out.get("meta_info") or {}
    entries = meta.get("output_token_logprobs")
    if not isinstance(entries, list) or not entries:
        return None
    top_entries = meta.get("output_top_logprobs")
    steps: list[dict[int, float]] = []
    for idx, entry in enumerate(entries):
        step: dict[int, float] = {}
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] is not None and entry[1] is not None:
            try:
                lp = float(entry[0])
                if math.isfinite(lp):
                    step[int(entry[1])] = lp
            except (TypeError, ValueError):
                pass
        if isinstance(top_entries, list) and idx < len(top_entries) and isinstance(top_entries[idx], list):
            for item in top_entries[idx]:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] is not None and item[1] is not None:
                    try:
                        lp = float(item[0])
                        if math.isfinite(lp):
                            step.setdefault(int(item[1]), lp)
                    except (TypeError, ValueError):
                        pass
        if not step:
            return None
        steps.append(step)
    return steps


# --------------------------------------------------------------------------- device / graph helpers
def _sync_device() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "is_available"):
        try:
            if torch.accelerator.is_available():
                torch.accelerator.synchronize()
                return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "is_available"):
        try:
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()
                return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_torch_compiler() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
        torch._dynamo.reset()


def _reset_global_graph_pool() -> None:
    for mod_name in _GLOBAL_POOL_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        teardown = getattr(mod, "_teardown_borrow_pool", None)
        if callable(teardown):
            try:
                teardown()
            except Exception:
                pass
        setter = getattr(mod, "set_global_graph_memory_pool", None)
        if callable(setter):
            try:
                setter(None)
                continue
            except Exception:
                pass
        if hasattr(mod, "global_graph_memory_pool"):
            try:
                setattr(mod, "global_graph_memory_pool", None)
            except Exception:
                pass


def _is_eager_runner(obj, gr) -> bool:
    if gr is None:
        return False
    eager = getattr(obj, "eager_runner", None)
    return (eager is not None and gr is eager) or type(gr).__name__ == "EagerRunner"


def _clear_containers(target) -> None:
    for container_attr in _GRAPH_CONTAINER_ATTRS:
        container = getattr(target, container_attr, None)
        if isinstance(container, (dict, list)):
            container.clear()


def _release_graph_attr(obj, attr: str) -> bool:
    gr = getattr(obj, attr, None)
    if gr is None or _is_eager_runner(obj, gr):
        return False
    backend = getattr(gr, "backend", None)
    if backend is not None:
        cleanup = getattr(backend, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
        _clear_containers(backend)
    _clear_containers(gr)
    destroy = getattr(gr, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass
    fallback = getattr(obj, "eager_runner", None) if attr == "prefill_cuda_graph_runner" else None
    setattr(obj, attr, fallback)
    return True


def _count_graphs(obj, attrs: tuple[str, ...]) -> int:
    total = 0
    if obj is None:
        return 0
    seen_ids: set[int] = set()
    for attr in attrs:
        gr = getattr(obj, attr, None)
        if gr is None or _is_eager_runner(obj, gr) or id(gr) in seen_ids:
            continue
        seen_ids.add(id(gr))
        for target in (gr, getattr(gr, "backend", None)):
            if target is None:
                continue
            for c_attr in ("graphs", "_graphs", "cuda_graphs", "decode_graphs", "prefill_graphs"):
                container = getattr(target, c_attr, None)
                if isinstance(container, (dict, list)):
                    total += len(container)
    return total


def _unwrap_worker(worker):
    """Unwrap overlap-scheduling wrapper (e.g. `TpModelWorkerClient`) to the underlying worker."""
    seen = set()
    cur = worker
    while cur is not None and id(cur) not in seen and hasattr(cur, "worker"):
        seen.add(id(cur))
        nxt = getattr(cur, "worker", None)
        if nxt is None:
            break
        cur = nxt
    return cur


def _get_model_runner(worker):
    if worker is None:
        return None
    getter = getattr(worker, "get_model_runner", None)
    if callable(getter):
        runner = getter()
        if runner is not None:
            return runner
    return getattr(worker, "model_runner", None)


def _get_raw_model(model_runner):
    getter = getattr(model_runner, "get_model", None)
    model = getter() if callable(getter) else getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("SGLang ModelRunner has no `model` attribute")
    return getattr(model, "_orig_mod", model)


def _collect_draft_runners(spec_worker, target_runner) -> list:
    runners: list = []
    seen = {id(target_runner)}
    if spec_worker is None:
        return runners
    for cand in (
        list(getattr(spec_worker, "draft_runners", ()) or ()),
        [getattr(spec_worker, "draft_model_runner", None)],
        [_get_model_runner(_unwrap_worker(getattr(spec_worker, "draft_worker", None)))],
        [_get_model_runner(_unwrap_worker(getattr(spec_worker, "_draft_worker", None)))],
        [_get_model_runner(spec_worker)],
    ):
        for r in cand:
            if r is not None and id(r) not in seen:
                seen.add(id(r))
                runners.append(r)
    return runners


def _resolve_workers(owner):
    """Return `(target_worker, target_runner, spec_worker, draft_runners)` for an SGLang Scheduler or WeightUpdaterManager."""
    scheduler = getattr(owner, "scheduler", None) or owner
    raw_tp = getattr(owner, "tp_worker", None) or getattr(scheduler, "tp_worker", None)
    if raw_tp is None:
        raise RuntimeError("SGLang Scheduler has no `tp_worker`")
    inner = _unwrap_worker(raw_tp)

    spec_worker = None
    if hasattr(inner, "target_worker") and getattr(inner, "target_worker", None) is not None:
        spec_worker = inner
        target_worker = _unwrap_worker(inner.target_worker)
    else:
        target_worker = inner
        raw_draft = getattr(owner, "draft_worker", None) or getattr(scheduler, "draft_worker", None)
        if raw_draft is not None:
            spec_worker = _unwrap_worker(raw_draft)

    target_runner = _get_model_runner(target_worker)
    if target_runner is None:
        raise RuntimeError(f"Could not locate SGLang ModelRunner on {type(target_worker).__name__}")
    draft_runners = _collect_draft_runners(spec_worker, target_runner)
    return target_worker, target_runner, spec_worker, draft_runners


# --------------------------------------------------------------------------- HotReload controller
class SGLangHotReloadController:
    """Attached lazily to each SGLang `Scheduler` rank to manage candidate hot reload."""

    def __init__(self, owner):
        self.owner = owner
        self.scheduler = getattr(owner, "scheduler", None) or owner
        self._kiln_patched: list | None = None
        self._kiln_module: types.ModuleType | None = None
        self._kiln_dirty: bool = False
        self._kiln_snapshot_collector: object | None = None

    def dispatch(self, method: str, args: list, kwargs: dict):
        if method == "kiln_hot_reload":
            return self.kiln_hot_reload(*args, **kwargs)
        if method == "kiln_teardown":
            return self.kiln_teardown(*args, **kwargs)
        if method == "kiln_info":
            return self.kiln_info(*args, **kwargs)
        if method == "kiln_enable_snapshot_capture":
            return self.kiln_enable_snapshot_capture(*args, **kwargs)
        if method == "kiln_finish_snapshot_capture":
            return self.kiln_finish_snapshot_capture(*args, **kwargs)
        raise ValueError(f"Unknown kiln SGLang RPC method: {method!r}")

    def kiln_hot_reload(self, path: str) -> float:
        """Install candidate `path`, reset compile/graphs, and recapture. Returns seconds taken."""
        t0 = time.perf_counter()
        _, target_runner, _, _ = _resolve_workers(self.owner)
        model = _get_raw_model(target_runner)
        self.kiln_teardown()

        name = f"kiln_candidate_{abs(hash((path, t0)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load candidate module from {path}")
        mod = importlib.util.module_from_spec(spec)
        self._kiln_patched = snapshot_forwards(model)
        self._kiln_module = mod
        sys.modules[name] = mod
        rebuild_started = False
        try:
            spec.loader.exec_module(mod)
            new_fwd = mod.make_forward(model)
            if new_fwd is not None:
                model.forward = new_fwd
            rebuild_started = True
            self._rebuild()
        except Exception:
            self._restore_baseline(rebuild=rebuild_started)
            raise
        return time.perf_counter() - t0

    def kiln_teardown(self) -> float:
        """Restore baseline `.forward` across `model` and all submodules, then recapture."""
        t0 = time.perf_counter()
        if self._kiln_dirty:
            raise RuntimeError("Kiln candidate rollback failed; restart the engine")
        if self._kiln_patched is not None:
            self._restore_baseline(rebuild=True)
        return time.perf_counter() - t0

    def _restore_baseline(self, rebuild: bool) -> None:
        mod = self._kiln_module
        try:
            _, target_runner, _, _ = _resolve_workers(self.owner)
            model = _get_raw_model(target_runner)
            try:
                if mod is not None and hasattr(mod, "restore"):
                    mod.restore(model)
            finally:
                restore_forwards(self._kiln_patched)
            if rebuild:
                self._rebuild()
        except Exception as exc:
            self._kiln_dirty = True
            raise RuntimeError("Kiln candidate rollback failed; restart the engine") from exc
        finally:
            if mod is not None:
                sys.modules.pop(mod.__name__, None)
            self._kiln_module = None
            self._kiln_patched = None

    def kiln_info(self) -> dict:
        _, target_runner, spec_worker, draft_runners = _resolve_workers(self.owner)
        model = _get_raw_model(target_runner)
        server_args = getattr(target_runner, "server_args", None) or getattr(self.scheduler, "server_args", None)
        num_graphs = (
            _count_graphs(target_runner, _MODEL_RUNNER_GRAPH_ATTRS)
            + _count_graphs(spec_worker, _SPEC_WORKER_GRAPH_ATTRS)
            + sum(_count_graphs(dr, _MODEL_RUNNER_GRAPH_ATTRS) for dr in draft_runners)
        )
        return {
            "engine": "sglang",
            "model_cls": type(model).__name__,
            "disable_cuda_graph": bool(getattr(server_args, "disable_cuda_graph", False)),
            "enable_torch_compile": bool(getattr(server_args, "enable_torch_compile", False)),
            "enable_piecewise_cuda_graph": bool(getattr(server_args, "enable_piecewise_cuda_graph", False)),
            "has_speculator": spec_worker is not None,
            "num_graphs": num_graphs,
            "tp_rank": getattr(target_runner, "tp_rank", getattr(self.scheduler, "tp_rank", 0)),
            "dp_rank": getattr(self.scheduler, "dp_rank", 0),
        }

    def kiln_enable_snapshot_capture(
        self, snapshot_dir: str, m_buckets=None, max_snapshot_mb: float = 64.0
    ) -> dict:
        from kiln.micro.snapshots import DEFAULT_M_BUCKETS, SnapshotCollector

        _, target_runner, _, _ = _resolve_workers(self.owner)
        tp_rank = getattr(target_runner, "tp_rank", getattr(self.scheduler, "tp_rank", 0))
        dp_rank = getattr(self.scheduler, "dp_rank", 0)
        if tp_rank not in (0, None) or dp_rank not in (0, None):
            return {"tp_rank": tp_rank, "dp_rank": dp_rank, "skipped": True}
        if self._kiln_snapshot_collector is not None:
            self._kiln_snapshot_collector.close()
        if isinstance(m_buckets, str):
            buckets = [int(x) for x in json.loads(m_buckets)]
        elif m_buckets is not None:
            buckets = [int(x) for x in m_buckets]
        else:
            buckets = list(DEFAULT_M_BUCKETS)
        model = _get_raw_model(target_runner)
        collector = SnapshotCollector(snapshot_dir, m_buckets=buckets, max_snapshot_mb=float(max_snapshot_mb))
        attached = collector.attach(model)
        self._kiln_snapshot_collector = collector
        return {"tp_rank": 0, "attached": attached}

    def kiln_finish_snapshot_capture(self) -> dict:
        collector = self._kiln_snapshot_collector
        self._kiln_snapshot_collector = None
        if collector is None:
            return {"snapshots": []}
        return collector.close()

    def _rebuild(self) -> None:
        """Drop CUDA graphs, reset torch.compile & global graph pool, and recapture."""
        _, target_runner, spec_worker, draft_runners = _resolve_workers(self.owner)

        _sync_device()

        # Best-effort radix/KV cache flush before warmup/recapture.
        flush = getattr(self.owner, "flush_cache", None) or getattr(self.scheduler, "flush_cache", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass

        had_piecewise = getattr(target_runner, "piecewise_cuda_graph_runner", None) is not None
        released_attrs = {a: _release_graph_attr(target_runner, a) for a in _MODEL_RUNNER_GRAPH_ATTRS}
        had_prefill_cg = released_attrs.get("prefill_cuda_graph_runner", False)
        had_decode_cg = released_attrs.get("decode_cuda_graph_runner", False)
        had_legacy_cg = any(
            v for k, v in released_attrs.items()
            if k not in ("prefill_cuda_graph_runner", "decode_cuda_graph_runner")
        )
        had_spec_graphs = (
            spec_worker is not None
            and any([_release_graph_attr(spec_worker, a) for a in _SPEC_WORKER_GRAPH_ATTRS])
        )
        had_draft_runner_graphs = any(
            [any([_release_graph_attr(dr, a) for a in _MODEL_RUNNER_GRAPH_ATTRS]) for dr in draft_runners]
        )
        if spec_worker is not None:
            for dr in draft_runners:
                hooks = getattr(dr, "capture_tail_hooks", None)
                if isinstance(hooks, list):
                    hooks.clear()
            if hasattr(spec_worker, "_draft_sampler"):
                spec_worker._draft_sampler = None

        _reset_torch_compiler()
        _reset_global_graph_pool()

        gc.collect()
        _sync_device()
        _empty_cache()

        # Recapture target graphs first (matches SGLang startup order for shared memory pool).
        if had_prefill_cg or had_decode_cg:
            if had_prefill_cg and hasattr(target_runner, "init_prefill_cuda_graph"):
                target_runner.init_prefill_cuda_graph()
            if had_decode_cg and hasattr(target_runner, "init_decode_cuda_graph"):
                target_runner.init_decode_cuda_graph()
            if not hasattr(target_runner, "init_prefill_cuda_graph") and not hasattr(target_runner, "init_decode_cuda_graph"):
                if hasattr(target_runner, "init_cuda_graphs"):
                    target_runner.init_cuda_graphs()
        elif had_legacy_cg:
            if hasattr(target_runner, "init_device_graphs"):
                target_runner.init_device_graphs()
            elif hasattr(target_runner, "init_cuda_graphs"):
                target_runner.init_cuda_graphs()
            if (
                had_piecewise
                and getattr(target_runner, "piecewise_cuda_graph_runner", None) is None
                and hasattr(target_runner, "init_piecewise_cuda_graphs")
            ):
                target_runner.init_piecewise_cuda_graphs()

        # Recapture speculative / draft graphs second.
        if had_spec_graphs or had_draft_runner_graphs:
            if spec_worker is not None and hasattr(spec_worker, "init_cuda_graphs"):
                spec_worker.init_cuda_graphs()
            elif spec_worker is not None and hasattr(spec_worker, "_capture_cuda_graphs"):
                spec_worker._capture_cuda_graphs()
            else:
                for dr in draft_runners:
                    if hasattr(dr, "init_device_graphs"):
                        dr.init_device_graphs()
                    elif hasattr(dr, "init_cuda_graphs"):
                        dr.init_cuda_graphs()

        _sync_device()


# --------------------------------------------------------------------------- SGLang class patches
def _make_update_output(success: bool, message: str):
    try:
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqOutput
    except ImportError:
        return types.SimpleNamespace(success=success, message=message, num_paused_requests=0)
    sig = inspect.signature(UpdateWeightFromDiskReqOutput)
    kwargs: dict = {"success": success, "message": message}
    if "num_paused_requests" in sig.parameters:
        kwargs["num_paused_requests"] = 0
    return UpdateWeightFromDiskReqOutput(**kwargs)


_CONTROLLERS_BY_ID: dict[int, tuple[object, SGLangHotReloadController]] = {}


def _get_controller(owner: object) -> SGLangHotReloadController:
    key = id(owner)
    entry = _CONTROLLERS_BY_ID.get(key)
    if entry is not None and entry[0] is owner:
        return entry[1]
    ctrl = SGLangHotReloadController(owner)
    _CONTROLLERS_BY_ID[key] = (owner, ctrl)
    return ctrl


def patch_scheduler_cls(scheduler_cls) -> None:
    if getattr(scheduler_cls, "_kiln_hooked", False):
        return
    orig = getattr(scheduler_cls, "update_weights_from_disk", None)
    if orig is None:
        return

    def _kiln_update_weights_from_disk(self, recv_req):
        rpc = decode_rpc(getattr(recv_req, "model_path", None))
        if rpc is None:
            return orig(self, recv_req)
        try:
            ctrl = _get_controller(self)
            res = ctrl.dispatch(rpc["method"], rpc.get("args", []), rpc.get("kwargs", {}))
            return _make_update_output(True, json.dumps({"ok": True, "result": res}))
        except Exception as exc:
            return _make_update_output(False, f"{type(exc).__name__}: {exc}")

    scheduler_cls.update_weights_from_disk = _kiln_update_weights_from_disk
    scheduler_cls._kiln_hooked = True


def patch_tokenizer_manager_cls(tm_cls) -> None:
    if getattr(tm_cls, "_kiln_hooked", False):
        return
    orig = getattr(tm_cls, "update_weights_from_disk", None)
    if orig is None:
        return

    orig_update_info = getattr(tm_cls, "_update_model_path_info", None)
    if orig_update_info is not None:
        def _kiln_update_model_path_info(self, model_path: str, *args, **kwargs):
            if decode_rpc(model_path) is not None:
                return None
            return orig_update_info(self, model_path, *args, **kwargs)
        tm_cls._update_model_path_info = _kiln_update_model_path_info

    def _snapshot_tm_paths(tm):
        sa = getattr(tm, "server_args", None)
        return (
            sa,
            getattr(sa, "model_path", None) if sa is not None else None,
            getattr(tm, "model_path", None),
            getattr(tm, "served_model_name", None),
        )

    def _restore_tm_paths(tm, snap) -> None:
        sa, orig_sa_path, orig_tm_path, orig_served_name = snap
        if sa is not None and orig_sa_path is not None and getattr(sa, "model_path", None) != orig_sa_path:
            sa.model_path = orig_sa_path
        if orig_tm_path is not None and getattr(tm, "model_path", None) != orig_tm_path:
            tm.model_path = orig_tm_path
        if orig_served_name is not None and getattr(tm, "served_model_name", None) != orig_served_name:
            tm.served_model_name = orig_served_name

    if inspect.iscoroutinefunction(orig):
        async def _kiln_tm_update_weights(self, obj, request=None):
            if decode_rpc(getattr(obj, "model_path", None)) is None:
                return await orig(self, obj, request)
            snap = _snapshot_tm_paths(self)
            try:
                return await orig(self, obj, request)
            finally:
                _restore_tm_paths(self, snap)
    else:
        def _kiln_tm_update_weights(self, obj, request=None):
            if decode_rpc(getattr(obj, "model_path", None)) is None:
                return orig(self, obj, request)
            snap = _snapshot_tm_paths(self)
            try:
                return orig(self, obj, request)
            finally:
                _restore_tm_paths(self, snap)

    tm_cls.update_weights_from_disk = _kiln_tm_update_weights
    tm_cls._kiln_hooked = True


def patch_module(module: types.ModuleType) -> None:
    name = getattr(module, "__name__", "")
    if name == "sglang.srt.managers.scheduler":
        cls = getattr(module, "Scheduler", None)
        if cls is not None:
            patch_scheduler_cls(cls)
    elif name == "sglang.srt.managers.scheduler_components.weight_updater":
        cls = getattr(module, "SchedulerWeightUpdaterManager", None)
        if cls is not None:
            patch_scheduler_cls(cls)
    elif name == "sglang.srt.managers.tokenizer_manager":
        cls = getattr(module, "TokenizerManager", None)
        if cls is not None:
            patch_tokenizer_manager_cls(cls)


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self._orig = orig_loader

    def create_module(self, spec):
        if hasattr(self._orig, "create_module"):
            return self._orig.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig.exec_module(module)
        patch_module(module)


class _KilnSGLangImportHook(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._active: set[str] = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGET_MODULES or fullname in self._active:
            return None
        self._active.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _PostImportLoader(spec.loader)
                    return spec
            return None
        finally:
            self._active.discard(fullname)


def install_sglang_hooks() -> None:
    """Install kiln's SGLang scheduler/tokenizer hooks (idempotent)."""
    for mod_name in _TARGET_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            patch_module(mod)
    if not any(isinstance(f, _KilnSGLangImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _KilnSGLangImportHook())
