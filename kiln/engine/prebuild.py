"""Out-of-lock candidate prebuild runner (`kiln.engine.prebuild`).

Warms `TORCH_EXTENSIONS_DIR`, `KILN_CACHE_DIR`, and `TRITON_CACHE_DIR` in an isolated
process group *before* acquiring `TierWorker.engine_lock`, so slow `nvcc`/`ninja`/`gcc`
compilations overlap with in-flight GPU benchmarks and broken C++/CUDA/Python syntax
candidates fail fast without stalling or dirtying the persistent GPU engine.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kiln.patches import inspect_candidate_source, normalize_pd_role

from .base import reap_zombies, shutdown_process_group

_COMPILER_BINS = frozenset({"gcc", "g++", "cc", "c++", "clang", "clang++", "nvcc", "ninja"})


@dataclass(frozen=True)
class PrebuildResult:
    ok: bool
    elapsed_s: float = 0.0
    skipped: bool = False
    reason: str = ""
    triggers: tuple[str, ...] = ()
    extensions_built: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "elapsed_s": round(self.elapsed_s, 4),
            "skipped": self.skipped,
            "reason": self.reason,
            "triggers": list(self.triggers),
            "extensions_built": self.extensions_built,
            "error": self.error,
        }


def ensure_extension_cache_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Ensure `TORCH_EXTENSIONS_DIR` and `KILN_CACHE_DIR` point to shared persistent directories."""
    env = dict(os.environ if base_env is None else base_env)
    raw_root = (env.get("KILN_EXT_CACHE_ROOT") or os.environ.get("KILN_EXT_CACHE_ROOT") or "").strip()
    cache_root = Path(raw_root).expanduser() if raw_root else (Path.home() / ".cache" / "kiln")
    if not env.get("TORCH_EXTENSIONS_DIR", "").strip():
        torch_ext_dir = (cache_root / "torch_extensions").resolve()
        torch_ext_dir.mkdir(parents=True, exist_ok=True)
        env["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)
    else:
        Path(env["TORCH_EXTENSIONS_DIR"]).expanduser().mkdir(parents=True, exist_ok=True)

    if not env.get("KILN_CACHE_DIR", "").strip():
        kiln_ext_dir = (cache_root / "ext").resolve()
        kiln_ext_dir.mkdir(parents=True, exist_ok=True)
        env["KILN_CACHE_DIR"] = str(kiln_ext_dir)
    else:
        Path(env["KILN_CACHE_DIR"]).expanduser().mkdir(parents=True, exist_ok=True)
    return env


class _DummyPrebuildModel:
    """Minimal stand-in passed to `make_forward` solely to trigger top-of-function `load_inline` calls."""

    def modules(self):
        return iter(())

    def named_modules(self):
        yield "", self

    def parameters(self):
        return iter(())

    def forward(self, *args, **kwargs):
        return None


def _is_compiler_cmd(cmd: Any) -> bool:
    if isinstance(cmd, (list, tuple)) and cmd:
        exe = Path(str(cmd[0])).name.lower()
        return exe in _COMPILER_BINS
    if isinstance(cmd, str) and cmd.strip():
        exe = Path(cmd.strip().split()[0]).name.lower()
        return exe in _COMPILER_BINS
    return False


def _format_called_proc_error(exc: subprocess.CalledProcessError) -> str:
    parts = [str(exc)]
    for stream_name, val in (("stdout", exc.stdout), ("stderr", exc.stderr)):
        if val:
            txt = val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
            if txt.strip():
                parts.append(f"--- {stream_name} ---\n{txt.strip()[-1500:]}")
    return "\n".join(parts)


def _is_compile_exception(exc: BaseException) -> bool:
    if isinstance(exc, (SyntaxError, subprocess.CalledProcessError)):
        return True
    cls_name = type(exc).__name__
    msg = str(exc)
    return (
        "CompileError" in cls_name
        or "CompilationError" in cls_name
        or "Error building extension" in msg
        or "ninja: build stopped" in msg
        or "nvcc fatal" in msg
    )


def _missing_uninstalled_engine_pkg(exc: BaseException, engine_kind: str) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    root = (getattr(exc, "name", None) or "").split(".")[0]
    if engine_kind == "mock" and root in ("vllm", "sglang", "cutlass", "tilelang", "triton"):
        return True
    if root in ("vllm", "sglang") and importlib.util.find_spec(root) is None:
        return True
    return False


def execute_prebuild_in_process(
    candidate_path: Path,
    *,
    pd_role: str = "both",
    engine_kind: str = "mock",
    triggers: Sequence[str] = (),
) -> dict[str, Any]:
    """Import candidate and warm any native extension caches inside an isolated child process."""
    t0 = time.perf_counter()
    source = candidate_path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(candidate_path))
    except SyntaxError:
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "skipped": False,
            "reason": "syntax_error",
            "triggers": list(triggers),
            "extensions_built": 0,
            "error": traceback.format_exc()[-2000:],
        }

    extensions_built = 0
    compile_errors: list[str] = []

    orig_subproc_run = subprocess.run
    orig_check_call = subprocess.check_call
    orig_check_output = subprocess.check_output

    def _wrapped_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        is_cc = _is_compiler_cmd(cmd)
        try:
            res = orig_subproc_run(*args, **kwargs)
        except subprocess.CalledProcessError as exc:
            if is_cc:
                compile_errors.append(_format_called_proc_error(exc))
            raise
        if is_cc and res.returncode == 0:
            nonlocal extensions_built
            extensions_built += 1
        return res

    def _wrapped_check_call(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        is_cc = _is_compiler_cmd(cmd)
        try:
            out = orig_check_call(*args, **kwargs)
        except subprocess.CalledProcessError as exc:
            if is_cc:
                compile_errors.append(_format_called_proc_error(exc))
            raise
        if is_cc:
            nonlocal extensions_built
            extensions_built += 1
        return out

    def _wrapped_check_output(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        is_cc = _is_compiler_cmd(cmd)
        try:
            out = orig_check_output(*args, **kwargs)
        except subprocess.CalledProcessError as exc:
            if is_cc:
                compile_errors.append(_format_called_proc_error(exc))
            raise
        if is_cc:
            nonlocal extensions_built
            extensions_built += 1
        return out

    subprocess.run = _wrapped_run
    subprocess.check_call = _wrapped_check_call
    subprocess.check_output = _wrapped_check_output

    cpp_ext_mod = None
    orig_load_inline = None
    orig_load = None
    if any(t in triggers for t in ("load_inline", "cpp_extension")):
        try:
            import torch.utils.cpp_extension as _cpp_ext

            cpp_ext_mod = _cpp_ext
            orig_load_inline = _cpp_ext.load_inline
            orig_load = _cpp_ext.load

            def _hook_load_inline(*args, **kwargs):
                nonlocal extensions_built
                try:
                    mod_ext = orig_load_inline(*args, **kwargs)
                    extensions_built += 1
                    return mod_ext
                except BaseException as exc:
                    compile_errors.append(f"{exc}\n{traceback.format_exc()[-2000:]}")
                    raise

            def _hook_load(*args, **kwargs):
                nonlocal extensions_built
                try:
                    mod_ext = orig_load(*args, **kwargs)
                    extensions_built += 1
                    return mod_ext
                except BaseException as exc:
                    compile_errors.append(f"{exc}\n{traceback.format_exc()[-2000:]}")
                    raise

            _cpp_ext.load_inline = _hook_load_inline
            _cpp_ext.load = _hook_load
        except ImportError:
            pass

    role = normalize_pd_role(pd_role)
    os.environ["KILN_PD_ROLE"] = role
    os.environ["KILN_PREBUILD"] = "1"
    mod_name = f"kiln_prebuild_{abs(hash((str(candidate_path), t0)))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(candidate_path))
    if spec is None or spec.loader is None:
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "skipped": False,
            "reason": "import_spec_failed",
            "triggers": list(triggers),
            "extensions_built": 0,
            "error": f"Cannot load module spec from {candidate_path}",
        }
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        try:
            spec.loader.exec_module(mod)
        except BaseException as exc:
            if _missing_uninstalled_engine_pkg(exc, engine_kind) and not compile_errors:
                return {
                    "ok": True,
                    "elapsed_s": time.perf_counter() - t0,
                    "skipped": True,
                    "reason": "missing_engine_pkg",
                    "triggers": list(triggers),
                    "extensions_built": extensions_built,
                    "error": None,
                }
            err_msg = compile_errors[-1] if compile_errors else f"{exc}\n{traceback.format_exc()[-2000:]}"
            return {
                "ok": False,
                "elapsed_s": time.perf_counter() - t0,
                "skipped": False,
                "reason": "compile_error" if (compile_errors or _is_compile_exception(exc)) else "import_error",
                "triggers": list(triggers),
                "extensions_built": extensions_built,
                "error": err_msg,
            }

        build_fn = getattr(mod, "kiln_build", None)
        if callable(build_fn):
            try:
                build_fn()
            except BaseException as exc:
                if _missing_uninstalled_engine_pkg(exc, engine_kind) and not compile_errors:
                    return {
                        "ok": True,
                        "elapsed_s": time.perf_counter() - t0,
                        "skipped": True,
                        "reason": "missing_engine_pkg",
                        "triggers": list(triggers),
                        "extensions_built": extensions_built,
                        "error": None,
                    }
                err_msg = compile_errors[-1] if compile_errors else f"{exc}\n{traceback.format_exc()[-2000:]}"
                return {
                    "ok": False,
                    "elapsed_s": time.perf_counter() - t0,
                    "skipped": False,
                    "reason": "kiln_build_failed",
                    "triggers": list(triggers),
                    "extensions_built": extensions_built,
                    "error": err_msg,
                }
        elif extensions_built == 0 and any(
            t in triggers for t in ("load_inline", "cpp_extension", "c_compiler", "tvm_ffi")
        ):
            mk = getattr(mod, "make_forward", None)
            if callable(mk):
                dummy = _DummyPrebuildModel()
                dummy._kiln_pd_role = role
                try:
                    mk(dummy)
                except BaseException as exc:
                    if compile_errors or _is_compile_exception(exc):
                        err_msg = compile_errors[-1] if compile_errors else f"{exc}\n{traceback.format_exc()[-2000:]}"
                        return {
                            "ok": False,
                            "elapsed_s": time.perf_counter() - t0,
                            "skipped": False,
                            "reason": "compile_error",
                            "triggers": list(triggers),
                            "extensions_built": extensions_built,
                            "error": err_msg,
                        }
                finally:
                    restore_fn = getattr(mod, "restore", None)
                    if callable(restore_fn):
                        try:
                            restore_fn(dummy)
                        except BaseException:
                            pass

        return {
            "ok": True,
            "elapsed_s": time.perf_counter() - t0,
            "skipped": False,
            "reason": "prebuilt",
            "triggers": list(triggers),
            "extensions_built": extensions_built,
            "error": None,
        }
    finally:
        subprocess.run = orig_subproc_run
        subprocess.check_call = orig_check_call
        subprocess.check_output = orig_check_output
        if cpp_ext_mod is not None and orig_load_inline is not None and orig_load is not None:
            cpp_ext_mod.load_inline = orig_load_inline
            cpp_ext_mod.load = orig_load
        sys.modules.pop(mod_name, None)


def prebuild_candidate(
    candidate_path: str | Path,
    *,
    mode: str = "auto",
    timeout_s: float = 300.0,
    pd_role: str = "both",
    engine_kind: str = "mock",
    gpus: Sequence[int] = (),
    base_env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen | None], None] | None = None,
) -> PrebuildResult:
    """Validate syntax in-process and run out-of-lock native extension prebuild in a subprocess."""
    t0 = time.perf_counter()
    cpath = Path(candidate_path)
    source = cpath.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(cpath))
    except SyntaxError:
        return PrebuildResult(
            ok=False,
            elapsed_s=time.perf_counter() - t0,
            skipped=False,
            reason="syntax_error",
            error=traceback.format_exc()[-2000:],
        )

    if mode == "off":
        return PrebuildResult(ok=True, elapsed_s=time.perf_counter() - t0, skipped=True, reason="disabled")

    info = inspect_candidate_source(source)
    triggers = tuple(info.get("build_triggers") or ())
    if mode == "auto" and not info.get("needs_prebuild"):
        return PrebuildResult(
            ok=True,
            elapsed_s=time.perf_counter() - t0,
            skipped=True,
            reason="no_build_triggers",
            triggers=triggers,
        )

    if cancel_event is not None and cancel_event.is_set():
        return PrebuildResult(
            ok=False,
            elapsed_s=time.perf_counter() - t0,
            skipped=False,
            reason="cancelled",
            triggers=triggers,
            error="Prebuild cancelled before start",
        )

    env = ensure_extension_cache_env(base_env)
    pkg_root = str(Path(__file__).resolve().parents[2])
    existing_pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if pkg_root not in existing_pp:
        env["PYTHONPATH"] = os.pathsep.join([pkg_root, *existing_pp])
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["KILN_PD_ROLE"] = normalize_pd_role(pd_role)
    env["KILN_PREBUILD"] = "1"
    env.pop("KILN_SGLANG_HOOK", None)

    with tempfile.TemporaryDirectory(prefix="kiln_prebuild_") as tmpdir:
        out_path = Path(tmpdir) / "prebuild_result.json"
        cmd = [
            sys.executable,
            "-m",
            "kiln.engine.prebuild",
            "--candidate",
            str(cpath.resolve()),
            "--out",
            str(out_path),
            "--role",
            normalize_pd_role(pd_role),
            "--engine-kind",
            engine_kind,
            "--triggers",
            ",".join(triggers),
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        if on_proc is not None:
            on_proc(proc)
        try:
            deadline = (time.monotonic() + float(timeout_s)) if (timeout_s and timeout_s > 0) else None
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    shutdown_process_group(proc, timeout=5.0)
                    return PrebuildResult(
                        ok=False,
                        elapsed_s=time.perf_counter() - t0,
                        skipped=False,
                        reason="cancelled",
                        triggers=triggers,
                        error="Candidate prebuild cancelled",
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    shutdown_process_group(proc, timeout=5.0)
                    return PrebuildResult(
                        ok=False,
                        elapsed_s=time.perf_counter() - t0,
                        skipped=False,
                        reason="timeout",
                        triggers=triggers,
                        error=f"Candidate prebuild timed out after {timeout_s:.1f}s",
                    )
                time.sleep(0.03)

            stdout_txt, stderr_txt = proc.communicate(timeout=5.0)
            reap_zombies()
            if out_path.is_file():
                data = json.loads(out_path.read_text(encoding="utf-8"))
                return PrebuildResult(
                    ok=bool(data.get("ok", False)),
                    elapsed_s=time.perf_counter() - t0,
                    skipped=bool(data.get("skipped", False)),
                    reason=str(data.get("reason") or ""),
                    triggers=tuple(data.get("triggers") or triggers),
                    extensions_built=int(data.get("extensions_built") or 0),
                    error=data.get("error"),
                )
            combined_tail = ((stderr_txt or "") + "\n" + (stdout_txt or "")).strip()[-2000:]
            return PrebuildResult(
                ok=(proc.returncode == 0),
                elapsed_s=time.perf_counter() - t0,
                skipped=False,
                reason="subprocess_exit",
                triggers=triggers,
                error=None if proc.returncode == 0 else (combined_tail or f"Prebuild exited with code {proc.returncode}"),
            )
        finally:
            if on_proc is not None:
                on_proc(None)
            if proc.poll() is None:
                shutdown_process_group(proc, timeout=5.0)


def _cli_main(argv: Sequence[str] | None = None) -> int:
    def _on_sigterm(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _on_sigterm)
    p = argparse.ArgumentParser(prog="kiln-prebuild")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--role", default="both")
    p.add_argument("--engine-kind", default="mock")
    p.add_argument("--triggers", default="")
    args = p.parse_args(argv)

    triggers = [t.strip() for t in args.triggers.split(",") if t.strip()]
    res = execute_prebuild_in_process(
        Path(args.candidate),
        pd_role=args.role,
        engine_kind=args.engine_kind,
        triggers=triggers,
    )
    out = Path(args.out)
    with tempfile.NamedTemporaryFile("w", dir=out.parent, delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        try:
            json.dump(res, f, indent=2)
            f.flush()
            f.close()
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    return 0 if res.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
