"""Lock-free isolated subprocess kernel sandbox (`kiln micro`) with ptxas/ncu & padded-NaN guards."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Sequence

import numpy as np

from kiln.config import RunConfig
from kiln.micro.counters import collect_ncu_counters, inspect_ptxas_and_spills
from kiln.micro.snapshots import DEFAULT_M_BUCKETS, load_snapshot_payload, load_snapshots

try:
    import torch as _torch
except ImportError:
    _torch = None


def _to_numpy(val: Any) -> np.ndarray:
    if _torch is not None and isinstance(val, _torch.Tensor):
        t = val.detach().cpu()
        if t.dtype in (getattr(_torch, "bfloat16", None), getattr(_torch, "float16", None)):
            t = t.float()
        return t.numpy()
    return np.asarray(val)


def _flatten_arrays(obj: Any) -> list[np.ndarray]:
    if _torch is not None and isinstance(obj, _torch.Tensor):
        return [_to_numpy(obj)]
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, (tuple, list)):
        out: list[np.ndarray] = []
        for item in obj:
            out.extend(_flatten_arrays(item))
        return out
    if isinstance(obj, dict):
        out = []
        for item in obj.values():
            out.extend(_flatten_arrays(item))
        return out
    if isinstance(obj, (int, float)):
        return [np.asarray([obj], dtype=np.float64)]
    return []


def _check_close(ref_out: Any, cand_out: Any, atol: float, rtol: float) -> tuple[bool, float | None, str]:
    ref_arrs = _flatten_arrays(ref_out)
    cand_arrs = _flatten_arrays(cand_out)
    if len(ref_arrs) != len(cand_arrs) or not ref_arrs:
        return False, None, f"output arity mismatch ({len(cand_arrs)} vs {len(ref_arrs)})"
    max_abs = 0.0
    for r, c in zip(ref_arrs, cand_arrs):
        if r.shape != c.shape:
            return False, None, f"shape mismatch {list(c.shape)} vs {list(r.shape)}"
        rf = r.astype(np.float64, copy=False)
        cf = c.astype(np.float64, copy=False)
        if not (np.all(np.isfinite(rf)) and np.all(np.isfinite(cf))):
            return False, None, "non-finite (NaN/Inf) output detected"
        diff = float(np.max(np.abs(rf - cf))) if rf.size else 0.0
        if diff > max_abs:
            max_abs = diff
        if not np.allclose(rf, cf, atol=atol, rtol=rtol):
            return False, max_abs, f"max_abs_diff={max_abs:.6g} exceeds tolerance (atol={atol}, rtol={rtol})"
    return True, max_abs, "ok"


def _sync_if_cuda(device: str) -> None:
    if device == "cuda" and _torch is not None and _torch.cuda.is_available():
        _torch.cuda.synchronize()


def _time_callable(fn: Callable[[], Any], device: str, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync_if_cuda(device)
    samples_ms: list[float] = []
    if device == "cuda" and _torch is not None and _torch.cuda.is_available():
        for _ in range(iters):
            s_ev = _torch.cuda.Event(enable_timing=True)
            e_ev = _torch.cuda.Event(enable_timing=True)
            s_ev.record()
            fn()
            e_ev.record()
            _torch.cuda.synchronize()
            samples_ms.append(float(s_ev.elapsed_time(e_ev)))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples_ms))


def _move_to_device(obj: Any, device: str) -> Any:
    if device != "cuda" or _torch is None:
        return obj
    if isinstance(obj, _torch.Tensor):
        return obj.to("cuda")
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _call_with_inputs(fn: Callable, inputs: Any):
    if isinstance(inputs, tuple):
        return fn(*inputs)
    if isinstance(inputs, dict):
        return fn(**inputs)
    return fn(inputs)


def _evaluate_case(case: dict[str, Any], device: str, atol: float, rtol: float) -> dict[str, Any]:
    m_val = int(case.get("m", 0))
    if "inputs_fn" in case and callable(case["inputs_fn"]):
        try:
            inputs = case["inputs_fn"](device)
        except TypeError:
            inputs = case["inputs_fn"]()
    else:
        inputs = case.get("inputs")

    ref_fn = case.get("ref_fn")
    cand_fn = case.get("cand_fn")
    bytes_moved = case.get("bytes_moved")
    flops = case.get("flops")

    if callable(ref_fn) and callable(cand_fn) and inputs is not None:
        ref_out = _call_with_inputs(ref_fn, inputs)
        cand_out = _call_with_inputs(cand_fn, inputs)
        _sync_if_cuda(device)
        ok, max_diff, detail = _check_close(ref_out, cand_out, atol, rtol)
        ref_ms = _time_callable(lambda: _call_with_inputs(ref_fn, inputs), device)
        cand_ms = _time_callable(lambda: _call_with_inputs(cand_fn, inputs), device)
        speedup = (ref_ms / cand_ms) if cand_ms > 0 and ref_ms > 0 else None
    elif callable(cand_fn) and inputs is not None and "ref_output" in case:
        cand_out = _call_with_inputs(cand_fn, inputs)
        _sync_if_cuda(device)
        ok, max_diff, detail = _check_close(case["ref_output"], cand_out, atol, rtol)
        ref_ms = float(case["ref_ms"]) if case.get("ref_ms") else None
        cand_ms = _time_callable(lambda: _call_with_inputs(cand_fn, inputs), device)
        speedup = (ref_ms / cand_ms) if (ref_ms and cand_ms > 0) else None
    else:
        ok = bool(case.get("correct", True))
        max_diff = case.get("max_abs_diff")
        detail = str(case.get("detail", "ok" if ok else "failed"))
        ref_ms = float(case["ref_ms"]) if case.get("ref_ms") else None
        cand_ms = float(case["cand_ms"]) if case.get("cand_ms") else None
        speedup = (
            float(case["speedup"])
            if case.get("speedup") is not None
            else ((ref_ms / cand_ms) if (ref_ms and cand_ms and cand_ms > 0) else None)
        )

    achieved_gbps = (float(bytes_moved) / (cand_ms * 1e-3) / 1e9) if (bytes_moved and cand_ms and cand_ms > 0) else None
    achieved_tflops = (float(flops) / (cand_ms * 1e-3) / 1e12) if (flops and cand_ms and cand_ms > 0) else None

    return {
        "m": m_val,
        "correct": bool(ok),
        "max_abs_diff": round(float(max_diff), 8) if max_diff is not None else None,
        "detail": detail,
        "ref_ms": round(ref_ms, 5) if ref_ms is not None else None,
        "cand_ms": round(cand_ms, 5) if cand_ms is not None else None,
        "speedup": round(speedup, 4) if speedup is not None else None,
        "achieved_gbps": round(achieved_gbps, 2) if achieved_gbps is not None else None,
        "achieved_tflops": round(achieved_tflops, 2) if achieved_tflops is not None else None,
    }


def _run_padded_nan_guard(mod: Any, cases: Sequence[dict[str, Any]], device: str) -> tuple[bool, str]:
    """Detect `0.0 * NaN = NaN` padding contamination across padded tensor rows."""
    padded_fns: list[Callable] = []
    if callable(getattr(mod, "micro_padded_kernel", None)):
        padded_fns.append(mod.micro_padded_kernel)
    for c in cases:
        if callable(c.get("padded_fn")):
            padded_fns.append(c["padded_fn"])
    if not padded_fns:
        return True, "not_applicable"

    valid_rows = 8
    total_rows = 16
    dim = 64
    for fn in padded_fns:
        if device == "cuda" and _torch is not None:
            x = _torch.randn((total_rows, dim), device="cuda", dtype=_torch.float32)
            x[valid_rows:] = float("nan")
        elif _torch is not None and getattr(mod, "MICRO_USE_TORCH", False):
            x = _torch.randn((total_rows, dim), dtype=_torch.float32)
            x[valid_rows:] = float("nan")
        else:
            rng = np.random.default_rng(7)
            x = rng.standard_normal((total_rows, dim)).astype(np.float32)
            x[valid_rows:] = np.nan
        out = fn(x, valid_rows)
        arrs = _flatten_arrays(out)
        if not arrs:
            return False, "padded kernel returned no array output"
        for arr in arrs:
            if arr.shape[0] < valid_rows:
                return False, f"padded kernel returned fewer than {valid_rows} rows"
            if not np.all(np.isfinite(arr[:valid_rows])):
                return (
                    False,
                    "Padded-row NaN guard failed: dirty NaN/Inf in padding rows leaked into valid rows "
                    "(e.g. `0.0 * workspace_NaN = NaN`).",
                )
    return True, "ok"


def execute_candidate_file(
    candidate_path: Path,
    *,
    snapshot_dir: Path | None = None,
    snapshot_tag: str | None = None,
    m_buckets: Sequence[int] = DEFAULT_M_BUCKETS,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    padded_nan_check: bool = True,
) -> dict[str, Any]:
    """Execute one micro candidate in-process (intended to be called inside a fresh subprocess)."""
    device = "cuda" if (_torch is not None and _torch.cuda.is_available()) else "cpu"
    name = f"kiln_micro_{abs(hash((str(candidate_path), time.perf_counter())))}"
    spec = importlib.util.spec_from_file_location(name, str(candidate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load micro candidate from {candidate_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)

        inline_ptx: list[str] = []
        raw_ptx = getattr(mod, "PTX_SOURCE", None)
        if isinstance(raw_ptx, str):
            inline_ptx.append(raw_ptx)
        elif isinstance(raw_ptx, (list, tuple)):
            inline_ptx.extend(str(x) for x in raw_ptx if x)

        cases: list[dict[str, Any]] = []
        if callable(getattr(mod, "micro_bench", None)):
            res = mod.micro_bench(device)
            if isinstance(res, dict) and isinstance(res.get("buckets"), list):
                cases = list(res["buckets"])
            elif isinstance(res, list):
                cases = list(res)
            elif isinstance(res, dict):
                cases = [res]
        elif isinstance(getattr(mod, "MICRO_CASES", None), list):
            cases = list(mod.MICRO_CASES)

        tag = snapshot_tag or getattr(mod, "MICRO_SNAPSHOT", None)
        kernel_fn = getattr(mod, "micro_kernel", None)
        maker_fn = getattr(mod, "make_micro_kernel", None)
        ref_fn = getattr(mod, "micro_ref_kernel", None)
        if not cases and tag and snapshot_dir is not None and (callable(kernel_fn) or callable(maker_fn)):
            snaps = load_snapshots(snapshot_dir, tag=str(tag))
            for snap in snaps:
                payload = load_snapshot_payload(snap["path"])
                args = _move_to_device(payload.get("args", ()), device)
                kwargs = _move_to_device(payload.get("kwargs", {}), device)
                ref_out = payload.get("output")
                fn = maker_fn(snap) if callable(maker_fn) else kernel_fn
                cases.append({
                    "m": int(snap["m_bucket"]),
                    "inputs": (args, kwargs),
                    "ref_output": ref_out,
                    "cand_fn": lambda a, kw, _fn=fn: _fn(*a, **kw),
                    "ref_fn": (lambda a, kw, _rf=ref_fn: _rf(*a, **kw)) if callable(ref_fn) else None,
                    "bytes_moved": snap.get("bytes"),
                })
                # Normalize inputs when ref_fn is absent so _evaluate_case matches signature
                if not callable(ref_fn):
                    cases[-1].pop("ref_fn")
                else:
                    cases[-1].pop("ref_output")

        if not cases:
            raise ValueError(
                "Micro candidate must define `micro_bench(device)`, `MICRO_CASES`, "
                "or `MICRO_SNAPSHOT` + `micro_kernel(...)` matching an available snapshot."
            )

        buckets = [_evaluate_case(c, device, atol=atol, rtol=rtol) for c in cases]
        all_correct = all(b["correct"] for b in buckets)

        padded_ok, padded_detail = (
            _run_padded_nan_guard(mod, cases, device) if padded_nan_check else (True, "skipped")
        )
        if not padded_ok:
            all_correct = False

        speedups = [b["speedup"] for b in buckets if isinstance(b.get("speedup"), (int, float)) and b["speedup"] > 0]
        geomean_speedup = (
            round(math.exp(sum(math.log(s) for s in speedups) / len(speedups)), 4)
            if speedups
            else None
        )

        return {
            "status": "done" if all_correct else "incorrect",
            "device": device,
            "all_correct": all_correct,
            "padded_nan_ok": padded_ok,
            "padded_nan_detail": padded_detail,
            "geomean_speedup": geomean_speedup,
            "buckets": buckets,
            "inline_ptx": inline_ptx,
            "error": None if all_correct else (padded_detail if not padded_ok else "Numerical mismatch in micro bucket(s)"),
        }
    finally:
        sys.modules.pop(name, None)


def _sandbox_main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kiln-micro-sandbox")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snapshot-dir")
    p.add_argument("--snapshot-tag")
    p.add_argument("--m-buckets", default="")
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--no-padded-nan-check", action="store_true")
    a = p.parse_args(argv)

    buckets = [int(x) for x in a.m_buckets.split(",") if x.strip()] if a.m_buckets else list(DEFAULT_M_BUCKETS)
    snap_dir = Path(a.snapshot_dir) if a.snapshot_dir else None
    try:
        res = execute_candidate_file(
            Path(a.candidate),
            snapshot_dir=snap_dir,
            snapshot_tag=a.snapshot_tag,
            m_buckets=buckets,
            atol=a.atol,
            rtol=a.rtol,
            padded_nan_check=not a.no_padded_nan_check,
        )
    except Exception as exc:
        res = {
            "status": "crashed",
            "all_correct": False,
            "padded_nan_ok": False,
            "padded_nan_detail": "not_run",
            "geomean_speedup": None,
            "buckets": [],
            "inline_ptx": [],
            "error": f"{exc}\n{traceback.format_exc()[-2000:]}",
        }
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0 if res.get("status") != "crashed" else 2


class MicroRunner:
    """Lock-free concurrent micro-kernel validator completely decoupled from `TierWorker.engine_lock`."""

    def __init__(self, run: RunConfig, root: Path):
        self.run = run
        self.root = Path(root)
        self.micro_dir = self.root / "micro"
        self.micro_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.root / "micro_snapshots"
        self.sem = threading.Semaphore(run.micro.max_concurrent)

    def _load_hw_profile(self) -> dict[str, Any] | None:
        try:
            hw_path = self.run.resolve_hw_profile()
            if hw_path is not None and hw_path.exists():
                return json.loads(hw_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def list_snapshots(self, tag: str | None = None) -> dict[str, Any]:
        rows = load_snapshots(self.snapshot_dir, tag=tag)
        return {
            "snapshot_dir": str(self.snapshot_dir),
            "count": len(rows),
            "snapshots": rows,
        }

    def run_candidate(
        self,
        agent: str,
        source: str,
        note: str = "",
        snapshot_tag: str | None = None,
        enable_ncu: bool | None = None,
        enable_ptxas: bool | None = None,
    ) -> dict[str, Any]:
        cfg = self.run.micro
        if not cfg.enabled:
            raise ValueError("Micro sandbox is disabled in RunConfig (`micro.enabled: false`)")

        mid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        mdir = self.micro_dir / mid
        mdir.mkdir(parents=True, exist_ok=True)
        cand_path = mdir / "candidate.py"
        cand_path.write_text(source, encoding="utf-8")
        out_path = mdir / "sandbox_out.json"
        triton_cache = (mdir / ".triton").resolve()
        triton_cache.mkdir(parents=True, exist_ok=True)

        env = {**os.environ, "TRITON_CACHE_DIR": str(triton_cache)}
        if cfg.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus))

        cmd = [
            sys.executable,
            "-m",
            "kiln.micro.sandbox",
            "--candidate",
            str(cand_path),
            "--out",
            str(out_path),
            "--atol",
            str(cfg.atol),
            "--rtol",
            str(cfg.rtol),
        ]
        if self.snapshot_dir.is_dir():
            cmd.extend(["--snapshot-dir", str(self.snapshot_dir)])
        if snapshot_tag:
            cmd.extend(["--snapshot-tag", snapshot_tag])
        if cfg.m_buckets:
            cmd.extend(["--m-buckets", ",".join(str(b) for b in cfg.m_buckets)])
        if not cfg.padded_nan_check:
            cmd.append("--no-padded-nan-check")

        t0 = time.time()
        with self.sem:
            timed_out = False
            stderr_tail = ""
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_s,
                )
                stderr_tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-1500:]
            except subprocess.TimeoutExpired:
                timed_out = True

            if timed_out:
                raw_out: dict[str, Any] = {
                    "status": "crashed",
                    "all_correct": False,
                    "padded_nan_ok": False,
                    "padded_nan_detail": "timeout",
                    "geomean_speedup": None,
                    "buckets": [],
                    "inline_ptx": [],
                    "error": f"micro sandbox timed out after {cfg.timeout_s:.1f}s",
                }
            elif out_path.is_file():
                raw_out = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                raw_out = {
                    "status": "crashed",
                    "all_correct": False,
                    "padded_nan_ok": False,
                    "padded_nan_detail": "missing_output",
                    "geomean_speedup": None,
                    "buckets": [],
                    "inline_ptx": [],
                    "error": stderr_tail or "micro sandbox exited without output",
                }

            inline_ptx = raw_out.pop("inline_ptx", [])
            use_ptxas = cfg.enable_ptxas if enable_ptxas is None else bool(enable_ptxas)
            ptxas_diag = (
                inspect_ptxas_and_spills(cache_dir=triton_cache, ptx_texts=inline_ptx)
                if use_ptxas
                else {"available": False, "any_spills": False, "max_registers": 0, "kernels": [], "warnings": []}
            )

            use_ncu = cfg.enable_ncu if enable_ncu is None else bool(enable_ncu)
            ncu_diag: dict[str, Any] | None = None
            if use_ncu and raw_out.get("status") == "done":
                ncu_diag = collect_ncu_counters(cmd, env=env, timeout_s=cfg.ncu_timeout_s)

        hw = self._load_hw_profile()
        roofline_summary = _micro_roofline(raw_out.get("buckets", ()), hw, ncu_diag)

        any_spills = bool(ptxas_diag.get("any_spills"))
        sp = raw_out.get("geomean_speedup")
        ready_for_fast = bool(
            raw_out.get("status") == "done"
            and raw_out.get("all_correct")
            and raw_out.get("padded_nan_ok", True)
            and not any_spills
            and (sp is None or float(sp) >= 1.0)
        )

        record = {
            "id": mid,
            "agent": agent,
            "note": note,
            "snapshot_tag": snapshot_tag,
            "status": raw_out.get("status", "crashed"),
            "device": raw_out.get("device"),
            "all_correct": bool(raw_out.get("all_correct")),
            "padded_nan_ok": bool(raw_out.get("padded_nan_ok", True)),
            "padded_nan_detail": raw_out.get("padded_nan_detail"),
            "geomean_speedup": sp,
            "buckets": raw_out.get("buckets", []),
            "ptxas": ptxas_diag,
            "ncu": ncu_diag,
            "roofline": roofline_summary,
            "ready_for_fast": ready_for_fast,
            "elapsed_s": round(time.time() - t0, 3),
            "created_at": t0,
            "error": raw_out.get("error"),
        }
        self._save_result(mdir / "result.json", record)
        return record

    @staticmethod
    def _save_result(path: Path, data: dict[str, Any]) -> None:
        with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
            try:
                json.dump(data, f, indent=2)
                f.write("\n")
                f.close()
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)

    def get_result(self, mid: str) -> dict[str, Any]:
        p = self.micro_dir / mid / "result.json"
        if not p.is_file():
            raise KeyError(f"unknown micro id {mid!r}")
        return json.loads(p.read_text(encoding="utf-8"))

    def list_results(self, limit: int = 30, agent: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for p in self.micro_dir.glob("*/result.json"):
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if agent is not None and row.get("agent") != agent:
                continue
            rows.append(row)
        rows.sort(key=lambda r: (float(r.get("created_at") or 0.0), str(r.get("id", ""))), reverse=True)
        return rows[: max(1, limit)]


def _micro_roofline(
    buckets: Sequence[dict[str, Any]],
    hw: dict[str, Any] | None,
    ncu: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if hw is None and not ncu:
        return None
    bw = (hw or {}).get("bandwidth") or {}
    hbm_peak = bw.get("hbm_peak_achieved_gbps")
    peak_gbps_vals = [float(b["achieved_gbps"]) for b in buckets if b.get("achieved_gbps")]
    achieved_gbps = max(peak_gbps_vals) if peak_gbps_vals else (ncu.get("hbm_gbps") if ncu else None)
    hbm_pct = (
        round(100.0 * float(achieved_gbps) / float(hbm_peak), 2)
        if (achieved_gbps and hbm_peak and float(hbm_peak) > 0)
        else None
    )
    return {
        "hbm_peak_gbps": round(float(hbm_peak), 2) if hbm_peak else None,
        "best_achieved_gbps": round(float(achieved_gbps), 2) if achieved_gbps else None,
        "hbm_roofline_pct": hbm_pct,
        "swizzle_hint": ncu.get("swizzle_hint") if ncu else None,
    }


if __name__ == "__main__":
    raise SystemExit(_sandbox_main())
