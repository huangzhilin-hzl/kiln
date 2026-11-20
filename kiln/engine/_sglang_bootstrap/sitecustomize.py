"""Bootstrap sitecustomize injected via PYTHONPATH for spawned SGLang worker processes.

Chain-loads any pre-existing `sitecustomize` on `sys.path`, then installs kiln's lazy
SGLang import hook ONLY when `KILN_SGLANG_HOOK=1` is set in the environment.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _chain_existing_sitecustomize() -> None:
    this_dir = str(Path(__file__).resolve().parent)
    other_paths = [p for p in sys.path if p and str(Path(p).resolve()) != this_dir]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", other_paths)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


_chain_existing_sitecustomize()

if os.environ.get("KILN_SGLANG_HOOK") == "1":
    from kiln.engine.sglang_ext import install_sglang_hooks

    install_sglang_hooks()
