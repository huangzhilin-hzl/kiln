"""Modular patch contract, AST metadata/conflict inspector, and orthogonal candidate composer."""
from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


VALID_PATCH_PHASES = ("all", "prefill", "decode")


def normalize_pd_role(role: str | None) -> str:
    """Normalize PD role strings to `'prefill'`, `'decode'`, or `'both'`."""
    val = (role or "").strip().lower()
    if val in ("prefill", "prefill_only", "kv_producer"):
        return "prefill"
    if val in ("decode", "decode_only", "kv_consumer"):
        return "decode"
    return "both"


def current_pd_role(model: Any = None) -> str:
    """Return active PD execution role (`'prefill'`, `'decode'`, or `'both'`)."""
    if model is not None:
        attr = getattr(model, "_kiln_pd_role", None)
        if isinstance(attr, str) and attr.strip():
            return normalize_pd_role(attr)
    return normalize_pd_role(os.environ.get("KILN_PD_ROLE"))


def _resolve_patch_phase(phase: str, regime: str) -> str:
    p = str(phase or "all").strip().lower() or "all"
    if p == "both":
        p = "all"
    if p == "all":
        reg = str(regime or "").strip().lower()
        if reg in ("prefill", "prefill_only"):
            p = "prefill"
        elif reg in ("decode", "decode_only"):
            p = "decode"
    if p not in VALID_PATCH_PHASES:
        raise ValueError(f"Unsupported patch phase {phase!r}; expected one of {VALID_PATCH_PHASES}")
    return p


class PatchConflictError(ValueError):
    """Raised when merging candidates that target overlapping patches or symbols."""

    def __init__(self, conflicts: list[dict]):
        self.conflicts = conflicts
        summary = ", ".join(
            f"{c['kind']}:{c['target']} ({c['job_a']} vs {c['job_b']})" for c in conflicts
        )
        super().__init__(f"Patch conflict(s) detected: {summary}")


@dataclass(frozen=True)
class PatchSpec:
    """Declarative metadata attached to a modular patch function via `@kiln_patch`."""

    name: str
    regime: str = "all"
    targets: tuple[str, ...] = ()
    description: str = ""
    returns_forward: bool = False
    phase: str = "all"


def patch_matches_pd_role(spec: PatchSpec | None, pd_role: str) -> bool:
    """Return True if `spec` should be installed on an instance serving `pd_role`."""
    role = normalize_pd_role(pd_role)
    if spec is None or role == "both" or spec.phase == "all":
        return True
    return spec.phase == role


def kiln_patch(
    name: str,
    *,
    regime: str = "all",
    phase: str = "all",
    targets: Sequence[str] = (),
    description: str = "",
    returns_forward: bool = False,
) -> Callable[[Callable], Callable]:
    """Mark a modular patch callable with its orthogonal name, regime, PD phase, and patched targets."""
    clean_regime = str(regime).strip() or "all"
    resolved_phase = _resolve_patch_phase(phase, clean_regime)
    spec = PatchSpec(
        name=str(name).strip(),
        regime=clean_regime,
        targets=tuple(str(t).strip() for t in targets if str(t).strip()),
        description=str(description).strip(),
        returns_forward=bool(returns_forward),
        phase=resolved_phase,
    )

    def _decorator(fn: Callable) -> Callable:
        fn.__kiln_patch__ = spec
        return fn

    return _decorator


def _discover_patches(namespace_or_patches: Mapping[str, Any] | Sequence[Callable]) -> list[Callable]:
    if isinstance(namespace_or_patches, Mapping):
        explicit = namespace_or_patches.get("KILN_PATCHES")
        if isinstance(explicit, (list, tuple)):
            return [fn for fn in explicit if callable(fn)]
        return [
            val
            for val in namespace_or_patches.values()
            if callable(val) and isinstance(getattr(val, "__kiln_patch__", None), PatchSpec)
        ]
    return [fn for fn in namespace_or_patches if callable(fn)]


def make_composite_forward(
    namespace_or_patches: Mapping[str, Any] | Sequence[Callable],
) -> tuple[Callable[[Any], Any], Callable[[Any], None]]:
    """Build `(make_forward, restore)` from `@kiln_patch` functions with LIFO rollback."""
    patches = _discover_patches(namespace_or_patches)
    applied_undos: list[Callable[[Any], Any]] = []

    def _run_undos(model: Any) -> list[BaseException]:
        errors: list[BaseException] = []
        while applied_undos:
            undo = applied_undos.pop()
            try:
                undo(model)
            except BaseException as exc:
                errors.append(exc)
        return errors

    def make_forward(model: Any):
        applied_undos.clear()
        root_forward = None
        active_role = current_pd_role(model)
        try:
            for patch_fn in patches:
                spec = getattr(patch_fn, "__kiln_patch__", None)
                if isinstance(spec, PatchSpec) and not patch_matches_pd_role(spec, active_role):
                    continue
                res = patch_fn(model)
                undo_fn = None
                if isinstance(spec, PatchSpec) and spec.returns_forward:
                    if isinstance(res, tuple) and len(res) == 2:
                        cand_fwd, undo_fn = res
                    else:
                        cand_fwd = res
                    if cand_fwd is not None:
                        model.forward = cand_fwd
                        root_forward = cand_fwd
                elif isinstance(res, tuple) and len(res) == 2 and (res[0] is None or callable(res[0])):
                    cand_fwd, undo_fn = res
                    if cand_fwd is not None:
                        model.forward = cand_fwd
                        root_forward = cand_fwd
                elif callable(res):
                    undo_fn = res
                if callable(undo_fn):
                    applied_undos.append(undo_fn)
        except BaseException:
            _run_undos(model)
            raise
        return root_forward

    def restore(model: Any) -> None:
        errors = _run_undos(model)
        if errors:
            raise errors[0]

    return make_forward, restore


# --------------------------------------------------------------------------- Static AST inspection
_CANONICAL_TAG_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mhc", ("mhc", "post_collapse", "coefficients", "all_reduce_mhc")),
    ("gemv", ("gemv", "mxfp8_gemv", "wo_a", "wo_b")),
    ("moe_router", ("router", "topk", "dsv4_topk")),
    ("moe", ("moe", "humming", "experts", "grouped_gemm")),
    ("mla_rope", ("mla_rope", "inverse_rope", "rotary")),
    ("mla", ("mla", "flashmla", "sparse_mla")),
    ("attention", ("attention", "attn", "flashinfer", "flash_attn")),
    ("all_reduce", ("allreduce", "all_reduce", "multicast", "symm_mem", "nccl")),
    ("embed", ("embed", "lm_head", "vocab_parallel")),
    ("engram", ("engram",)),
    ("norm", ("rmsnorm", "layernorm")),
)


def _literal_str_seq(node: ast.AST) -> list[str]:
    try:
        val = ast.literal_eval(node)
    except Exception:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if isinstance(x, str) and x.strip()]
    return []


def _literal_num(node: ast.AST) -> float | None:
    try:
        val = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        return f if math.isfinite(f) and f > 0 else None
    return None


def _literal_bucket_map(node: ast.AST) -> dict[str, float]:
    try:
        val = ast.literal_eval(node)
    except Exception:
        return {}
    if not isinstance(val, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in val.items():
        if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if math.isfinite(fv) and fv > 0:
                out[k] = fv
    return out


def _attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_kiln_patch_decorator(dec: ast.AST) -> ast.Call | None:
    if not isinstance(dec, ast.Call):
        return None
    fn = dec.func
    if isinstance(fn, ast.Name) and fn.id == "kiln_patch":
        return dec
    if isinstance(fn, ast.Attribute) and fn.attr == "kiln_patch":
        return dec
    return None


def _extract_decorator_patch(call: ast.Call) -> dict[str, Any]:
    name = ""
    if call.args:
        strs = _literal_str_seq(call.args[0])
        if strs:
            name = strs[0]
    regime = "all"
    phase = "all"
    targets: list[str] = []
    for kw in call.keywords:
        if kw.arg == "name":
            strs = _literal_str_seq(kw.value)
            if strs:
                name = strs[0]
        elif kw.arg == "regime":
            strs = _literal_str_seq(kw.value)
            if strs:
                regime = strs[0]
        elif kw.arg == "phase":
            strs = _literal_str_seq(kw.value)
            if strs:
                phase = strs[0]
        elif kw.arg == "targets":
            targets.extend(_literal_str_seq(kw.value))
    try:
        resolved_phase = _resolve_patch_phase(phase, regime)
    except ValueError:
        resolved_phase = "all"
    return {"name": name, "regime": regime, "phase": resolved_phase, "targets": targets}


def _infer_tags(tokens: Sequence[str]) -> list[str]:
    joined = " ".join(tokens).lower()
    out: list[str] = []
    for tag, keywords in _CANONICAL_TAG_KEYWORDS:
        if any(kw in joined for kw in keywords):
            out.append(tag)
    return out


def _func_tail_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_composite_forward_rhs(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Call):
        return _func_tail_name(node.func) == "make_composite_forward"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        return _func_tail_name(node.value.func) == "make_composite_forward"
    return False


def _iter_toplevel_exec_nodes(stmts: Sequence[ast.stmt]):
    """Yield AST nodes executed at module import time (stopping at def/class boundaries)."""
    stack = list(reversed(stmts))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield cur
        for child in reversed(list(ast.iter_child_nodes(cur))):
            stack.append(child)


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_target_names(elt))
        return out
    return []


def _strip_docstring_from_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _normalized_func_ast_sha256(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Hash a function AST ignoring line numbers, formatting, comments, and leading docstrings."""
    cloned = copy.deepcopy(node)
    cloned.body = _strip_docstring_from_body(list(cloned.body))
    dumped = ast.dump(cloned, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def extract_function_ast_hashes(source: str) -> dict[str, str]:
    """Return `{func_or_patch_key: sha256}` for top-level functions, class methods, and `@kiln_patch` targets."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    out: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            digest = _normalized_func_ast_sha256(stmt)
            out[stmt.name] = digest
            for dec in stmt.decorator_list:
                call = _is_kiln_patch_decorator(dec)
                if call is not None:
                    pinfo = _extract_decorator_patch(call)
                    if pinfo.get("name"):
                        out[f"patch:{pinfo['name']}"] = digest
        elif isinstance(stmt, ast.ClassDef):
            for child in stmt.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{stmt.name}.{child.name}"] = _normalized_func_ast_sha256(child)
    return out


def inspect_candidate_source(source: str) -> dict[str, Any]:
    """Statically parse candidate source for tags, parent IDs, regimes, patches, targets, phases, and build triggers."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "syntax_ok": False,
            "tags": [],
            "parents": [],
            "regimes": [],
            "phases": [],
            "effective_phase": "all",
            "only_kiln_patches": False,
            "needs_prebuild": False,
            "build_triggers": [],
            "patches": [],
            "targets": [],
            "func_ast_hashes": {},
            "mock_speedup": None,
            "mock_prefill_speedup": None,
            "mock_bucket_speedups": {},
        }

    tags: list[str] = []
    parents: list[str] = []
    regimes: list[str] = []
    phases: list[str] = []
    patches: list[dict[str, Any]] = []
    targets: list[str] = []
    build_triggers: list[str] = []
    seen_tags: set[str] = set()
    seen_parents: set[str] = set()
    seen_regimes: set[str] = set()
    seen_phases: set[str] = set()
    seen_targets: set[str] = set()
    seen_triggers: set[str] = set()
    symbol_tokens: list[str] = []
    imported_aliases: dict[str, str] = {}
    mock_speedup: float | None = None
    mock_prefill_speedup: float | None = None
    mock_bucket_speedups: dict[str, float] = {}
    explicit_phase: str | None = None
    explicit_only_patches: bool | None = None
    explicit_prebuild: bool | None = None
    has_custom_make_forward = False
    has_toplevel_mutation = False
    imports_subprocess = False
    string_literals: set[str] = set()

    def _add_tag(val: str) -> None:
        v = val.strip()
        if v and v not in seen_tags:
            seen_tags.add(v)
            tags.append(v)

    def _add_parent(val: str) -> None:
        v = val.strip()
        if v and v not in seen_parents:
            seen_parents.add(v)
            parents.append(v)

    def _add_regime(val: str) -> None:
        v = val.strip()
        if v and v not in seen_regimes:
            seen_regimes.add(v)
            regimes.append(v)

    def _add_target(val: str) -> None:
        v = val.strip()
        if v and v not in seen_targets:
            seen_targets.add(v)
            targets.append(v)

    def _add_trigger(val: str) -> None:
        if val and val not in seen_triggers:
            seen_triggers.add(val)
            build_triggers.append(val)

    # Top-level imports & metadata constants
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                imported_aliases[local] = alias.name
                symbol_tokens.append(alias.name)
                if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                    imports_subprocess = True
                if "cpp_extension" in alias.name:
                    _add_trigger("cpp_extension")
                if "tvm_ffi" in alias.name:
                    _add_trigger("tvm_ffi")
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            symbol_tokens.append(mod)
            if mod == "subprocess" or mod.startswith("subprocess."):
                imports_subprocess = True
            if "cpp_extension" in mod:
                _add_trigger("cpp_extension")
            if "tvm_ffi" in mod:
                _add_trigger("tvm_ffi")
            for alias in stmt.names:
                local = alias.asname or alias.name
                full = f"{mod}.{alias.name}" if mod else alias.name
                imported_aliases[local] = full
                symbol_tokens.append(alias.name)
                if alias.name == "load_inline":
                    _add_trigger("load_inline")
                elif alias.name in ("cpp_extension", "CUDAExtension", "CppExtension", "compile_extension"):
                    _add_trigger("cpp_extension")
                elif alias.name == "tvm_ffi":
                    _add_trigger("tvm_ffi")
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "kiln_build":
            _add_trigger("kiln_build")
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                if tgt.id == "KILN_TAGS":
                    for item in _literal_str_seq(stmt.value):
                        _add_tag(item)
                elif tgt.id == "KILN_PARENTS":
                    for item in _literal_str_seq(stmt.value):
                        _add_parent(item)
                elif tgt.id == "KILN_PHASE":
                    vals = _literal_str_seq(stmt.value)
                    if vals:
                        try:
                            explicit_phase = _resolve_patch_phase(vals[0], "all")
                        except ValueError:
                            explicit_phase = "all"
                elif tgt.id == "KILN_ONLY_PATCHES":
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, bool):
                        explicit_only_patches = stmt.value.value
                elif tgt.id == "KILN_PREBUILD":
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, bool):
                        explicit_prebuild = stmt.value.value
                elif tgt.id == "SPEEDUP":
                    num = _literal_num(stmt.value)
                    if num is not None:
                        mock_speedup = num
                elif tgt.id == "PREFILL_SPEEDUP":
                    num = _literal_num(stmt.value)
                    if num is not None:
                        mock_prefill_speedup = num
                elif tgt.id == "BUCKET_SPEEDUPS":
                    mock_bucket_speedups = _literal_bucket_map(stmt.value)

    for top_node in _iter_toplevel_exec_nodes(tree.body):
        if isinstance(top_node, ast.Assign):
            for tgt in top_node.targets:
                if isinstance(tgt, (ast.Attribute, ast.Subscript)):
                    has_toplevel_mutation = True
                if "make_forward" in _target_names(tgt) and not _is_composite_forward_rhs(top_node.value):
                    has_custom_make_forward = True
        elif isinstance(top_node, ast.AnnAssign) and top_node.target is not None:
            if isinstance(top_node.target, (ast.Attribute, ast.Subscript)):
                has_toplevel_mutation = True
            if "make_forward" in _target_names(top_node.target) and not _is_composite_forward_rhs(top_node.value):
                has_custom_make_forward = True
        elif isinstance(top_node, ast.AugAssign) and isinstance(top_node.target, (ast.Attribute, ast.Subscript)):
            has_toplevel_mutation = True

    # Walk AST for @kiln_patch decorators, class/module patching targets, build calls, and symbol hints
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_tokens.append(node.name)
            if node.name == "make_forward":
                has_custom_make_forward = True
            elif node.name == "kiln_build":
                _add_trigger("kiln_build")
            for dec in node.decorator_list:
                call = _is_kiln_patch_decorator(dec)
                if call is None:
                    continue
                pinfo = _extract_decorator_patch(call)
                pinfo["fn"] = node.name
                patches.append(pinfo)
                if pinfo["name"]:
                    _add_tag(pinfo["name"])
                if pinfo["regime"] and pinfo["regime"] != "all":
                    _add_regime(pinfo["regime"])
                if pinfo["phase"] and pinfo["phase"] != "all" and pinfo["phase"] not in seen_phases:
                    seen_phases.add(pinfo["phase"])
                    phases.append(pinfo["phase"])
                for t in pinfo["targets"]:
                    _add_target(t)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                    imports_subprocess = True
                if "cpp_extension" in alias.name:
                    _add_trigger("cpp_extension")
                if "tvm_ffi" in alias.name:
                    _add_trigger("tvm_ffi")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "subprocess" or mod.startswith("subprocess."):
                imports_subprocess = True
            if "cpp_extension" in mod:
                _add_trigger("cpp_extension")
            if "tvm_ffi" in mod:
                _add_trigger("tvm_ffi")
            for alias in node.names:
                if alias.name == "load_inline":
                    _add_trigger("load_inline")
                elif alias.name in ("cpp_extension", "CUDAExtension", "CppExtension", "compile_extension"):
                    _add_trigger("cpp_extension")
                elif alias.name == "tvm_ffi":
                    _add_trigger("tvm_ffi")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 64:
            string_literals.add(node.value)
        elif isinstance(node, ast.ClassDef):
            symbol_tokens.append(node.name)
        elif isinstance(node, ast.Call):
            tail = _func_tail_name(node.func)
            if tail == "load_inline":
                _add_trigger("load_inline")
            elif tail in ("compile_extension", "CUDAExtension", "CppExtension"):
                _add_trigger("cpp_extension")
            # isinstance(x, ClassName) or isinstance(x, (A, B))
            if isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) >= 2:
                cls_arg = node.args[1]
                cls_nodes = cls_arg.elts if isinstance(cls_arg, (ast.Tuple, ast.List)) else [cls_arg]
                for cnode in cls_nodes:
                    chain = _attr_chain(cnode)
                    if chain and chain not in ("dict", "list", "tuple", "str", "int", "float", "bool"):
                        _add_target(chain.split(".")[-1])
            # type(m).__name__ == "Foo" or type(m).__name__.endswith("MLP")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "endswith" and node.args:
                recv = node.func.value
                if (
                    isinstance(recv, ast.Attribute)
                    and recv.attr == "__name__"
                    and isinstance(recv.value, ast.Call)
                    and isinstance(recv.value.func, ast.Name)
                    and recv.value.func.id == "type"
                ):
                    for s in _literal_str_seq(node.args[0]):
                        _add_target(f"*{s}")
        elif isinstance(node, ast.Compare):
            left = node.left
            if (
                isinstance(left, ast.Attribute)
                and left.attr == "__name__"
                and isinstance(left.value, ast.Call)
                and isinstance(left.value.func, ast.Name)
                and left.value.func.id == "type"
            ):
                for comp in node.comparators:
                    for s in _literal_str_seq(comp):
                        _add_target(s)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                chain = _attr_chain(tgt)
                if not chain or "." not in chain:
                    continue
                root_name, rest = chain.split(".", 1)
                if root_name in imported_aliases:
                    _add_target(f"{imported_aliases[root_name]}.{rest}")
                elif root_name == "model" and rest != "forward":
                    _add_target(chain)

    if imports_subprocess and (
        "nvcc" in string_literals
        or ("-shared" in string_literals and "-fPIC" in string_literals)
    ):
        _add_trigger("c_compiler")

    if explicit_only_patches is not None:
        only_kiln_patches = bool(explicit_only_patches) and not has_toplevel_mutation
    else:
        only_kiln_patches = bool(patches) and (not has_custom_make_forward) and (not has_toplevel_mutation)

    if explicit_phase is not None:
        effective_phase = explicit_phase
    elif only_kiln_patches:
        patch_phases = {p["phase"] for p in patches}
        if patch_phases == {"prefill"}:
            effective_phase = "prefill"
        elif patch_phases == {"decode"}:
            effective_phase = "decode"
        else:
            effective_phase = "all"
    else:
        effective_phase = "all"

    if explicit_prebuild is False:
        needs_prebuild = False
    elif explicit_prebuild is True:
        _add_trigger("explicit")
        needs_prebuild = True
    else:
        needs_prebuild = bool(build_triggers)

    for inferred in _infer_tags([*symbol_tokens, *targets]):
        _add_tag(inferred)

    return {
        "syntax_ok": True,
        "tags": tags,
        "parents": parents,
        "regimes": regimes,
        "phases": phases,
        "effective_phase": effective_phase,
        "only_kiln_patches": only_kiln_patches,
        "needs_prebuild": needs_prebuild,
        "build_triggers": build_triggers,
        "patches": patches,
        "targets": targets,
        "func_ast_hashes": extract_function_ast_hashes(source),
        "mock_speedup": mock_speedup,
        "mock_prefill_speedup": mock_prefill_speedup,
        "mock_bucket_speedups": mock_bucket_speedups,
    }


def detect_candidate_conflicts(inspected: Sequence[tuple[str, dict[str, Any]]]) -> list[dict[str, str]]:
    """Find overlapping `@kiln_patch` names or target symbols across pairs of candidates."""
    conflicts: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for i in range(len(inspected)):
        jid_a, info_a = inspected[i]
        names_a = {p["name"] for p in info_a.get("patches", ()) if p.get("name")}
        targets_a = set(info_a.get("targets", ()))
        for j in range(i + 1, len(inspected)):
            jid_b, info_b = inspected[j]
            names_b = {p["name"] for p in info_b.get("patches", ()) if p.get("name")}
            targets_b = set(info_b.get("targets", ()))
            for pname in sorted(names_a & names_b):
                key = ("patch_name", pname, jid_a, jid_b)
                if key not in seen_keys:
                    seen_keys.add(key)
                    conflicts.append({
                        "kind": "patch_name",
                        "target": pname,
                        "job_a": jid_a,
                        "job_b": jid_b,
                    })
            for tname in sorted(targets_a & targets_b):
                key = ("target", tname, jid_a, jid_b)
                if key not in seen_keys:
                    seen_keys.add(key)
                    conflicts.append({
                        "kind": "target",
                        "target": tname,
                        "job_a": jid_a,
                        "job_b": jid_b,
                    })
    return conflicts


def _safe_ident(jid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", jid)


def compose_candidates(
    items: Sequence[tuple[str, str]],
    *,
    allow_target_overlap: bool = False,
) -> dict[str, Any]:
    """Compose two or more candidate sources into a self-contained orthogonal forward module."""
    if len(items) < 2:
        raise ValueError("compose_candidates requires at least two (job_id, source) items")

    inspected: list[tuple[str, dict[str, Any]]] = []
    merged_tags: list[str] = []
    seen_tags: set[str] = set()
    parent_ids: list[str] = []

    for jid, src in items:
        info = inspect_candidate_source(src)
        if not info["syntax_ok"]:
            raise ValueError(f"Candidate {jid} has invalid Python syntax")
        parent_ids.append(jid)
        inspected.append((jid, info))
        for tag in info["tags"]:
            if tag not in seen_tags:
                seen_tags.add(tag)
                merged_tags.append(tag)

    conflicts = detect_candidate_conflicts(inspected)
    if conflicts and not allow_target_overlap:
        raise PatchConflictError(conflicts)

    # Combine mock speedups multiplicatively for CPU dry runs.
    has_mock_speedup = any(info["mock_speedup"] is not None for _, info in inspected)
    combined_speedup = 1.0
    for _, info in inspected:
        if info["mock_speedup"] is not None:
            combined_speedup *= info["mock_speedup"]

    all_bucket_keys: list[str] = []
    for _, info in inspected:
        for bkey in info["mock_bucket_speedups"]:
            if bkey not in all_bucket_keys:
                all_bucket_keys.append(bkey)
    combined_buckets: dict[str, float] = {}
    for bkey in all_bucket_keys:
        prod = 1.0
        for _, info in inspected:
            bmap = info["mock_bucket_speedups"]
            if bkey in bmap:
                prod *= bmap[bkey]
            elif info["mock_speedup"] is not None:
                prod *= info["mock_speedup"]
        combined_buckets[bkey] = round(prod, 6)

    parent_phases = {info["effective_phase"] for _, info in inspected}
    merged_phase = next(iter(parent_phases)) if len(parent_phases) == 1 else "all"
    merged_only_patches = all(bool(info.get("only_kiln_patches")) for _, info in inspected)
    merged_needs_prebuild = any(bool(info.get("needs_prebuild")) for _, info in inspected)

    lines: list[str] = [
        '"""Auto-composed orthogonal candidate synthesized by `kiln merge`."""',
        "from __future__ import annotations",
        "",
        "import linecache",
        "import sys",
        "import types",
        "",
        f"KILN_PARENTS = {parent_ids!r}",
        f"KILN_TAGS = {merged_tags!r}",
    ]
    if merged_phase in ("prefill", "decode"):
        lines.append(f"KILN_PHASE = {merged_phase!r}")
    if merged_only_patches:
        lines.append("KILN_ONLY_PATCHES = True")
    if merged_needs_prebuild:
        lines.append("KILN_PREBUILD = True")
    if has_mock_speedup:
        lines.append(f"SPEEDUP = {round(combined_speedup, 6)!r}")
    if combined_buckets:
        lines.append(f"BUCKET_SPEEDUPS = {combined_buckets!r}")

    lines.extend([
        "",
        "_PARENT_SPECS: list[tuple[str, str]] = [",
    ])
    for jid, src in items:
        lines.append(f"    ({jid!r}, {src!r}),")
    lines.extend([
        "]",
        "",
        "_LOADED_PARENTS: list[tuple[str, types.ModuleType]] = []",
        "",
        "",
        "def _load_parent_module(idx: int, jid: str, src: str) -> types.ModuleType:",
        "    safe_id = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in jid)",
        "    mod_name = f'_kiln_merged_{idx}_{safe_id}'",
        "    mod = types.ModuleType(mod_name)",
        "    mod.__file__ = f'<kiln_merged:{jid}>'",
        "    linecache.cache[mod.__file__] = (len(src), None, src.splitlines(keepends=True), mod.__file__)",
        "    sys.modules[mod_name] = mod",
        "    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)",
        "    return mod",
        "",
        "",
        "def _cleanup_loaded() -> None:",
        "    for mod_name, mod in reversed(_LOADED_PARENTS):",
        "        linecache.cache.pop(getattr(mod, '__file__', None), None)",
        "        sys.modules.pop(mod_name, None)",
        "    _LOADED_PARENTS.clear()",
        "",
        "",
        "def _restore_applied(applied: list[tuple[str, types.ModuleType]], model) -> None:",
        "    first_err = None",
        "    for mod_name, mod in reversed(applied):",
        "        restore_fn = getattr(mod, 'restore', None)",
        "        if callable(restore_fn):",
        "            try:",
        "                restore_fn(model)",
        "            except BaseException as exc:",
        "                if first_err is None:",
        "                    first_err = exc",
        "        linecache.cache.pop(getattr(mod, '__file__', None), None)",
        "        sys.modules.pop(mod_name, None)",
        "    if first_err is not None:",
        "        raise first_err",
        "",
        "",
        "def make_forward(model):",
        "    restore(model)",
        "    root_forward = None",
        "    applied: list[tuple[str, types.ModuleType]] = []",
        "    try:",
        "        for idx, (jid, src) in enumerate(_PARENT_SPECS):",
        "            mod = _load_parent_module(idx, jid, src)",
        "            applied.append((mod.__name__, mod))",
        "            _LOADED_PARENTS.append((mod.__name__, mod))",
        "            mk = getattr(mod, 'make_forward', None)",
        "            if not callable(mk):",
        "                from kiln.patches import make_composite_forward",
        "                mk, auto_restore = make_composite_forward(vars(mod))",
        "                if not hasattr(mod, 'restore'):",
        "                    mod.restore = auto_restore",
        "            new_fwd = mk(model)",
        "            if new_fwd is not None:",
        "                model.forward = new_fwd",
        "                root_forward = new_fwd",
        "    except BaseException:",
        "        _LOADED_PARENTS.clear()",
        "        _restore_applied(applied, model)",
        "        raise",
        "    return root_forward",
        "",
        "",
        "def restore(model) -> None:",
        "    applied = list(_LOADED_PARENTS)",
        "    _LOADED_PARENTS.clear()",
        "    if applied:",
        "        _restore_applied(applied, model)",
        "",
    ])
    if merged_needs_prebuild:
        lines.extend([
            "def kiln_build() -> None:",
            "    loaded: list[tuple[str, types.ModuleType]] = []",
            "    try:",
            "        for idx, (jid, src) in enumerate(_PARENT_SPECS):",
            "            mod = _load_parent_module(idx, jid, src)",
            "            loaded.append((mod.__name__, mod))",
            "            build_fn = getattr(mod, 'kiln_build', None)",
            "            if callable(build_fn):",
            "                build_fn()",
            "    finally:",
            "        for mod_name, mod in reversed(loaded):",
            "            linecache.cache.pop(getattr(mod, '__file__', None), None)",
            "            sys.modules.pop(mod_name, None)",
            "",
        ])

    return {
        "source": "\n".join(lines),
        "parent_ids": parent_ids,
        "tags": merged_tags,
        "conflicts": conflicts,
    }


def compose_pd_pair(
    prefill_item: tuple[str, str],
    decode_item: tuple[str, str],
) -> dict[str, Any]:
    """Bundle asymmetric `(prefill_id, prefill_src)` + `(decode_id, decode_src)` into one PD-aware candidate.

    At hot-reload time, `make_forward(model)` inspects `current_pd_role(model)`:
    - `'prefill'` instances execute ONLY `prefill_src`.
    - `'decode'` instances execute ONLY `decode_src`.
    - `'both'` (colocated fallback) executes both if their targets are disjoint, else `decode_src`.
    Because P and D workers run in isolated processes under PD disaggregation, overlapping targets
    between `prefill_src` and `decode_src` are safe and never raise `PatchConflictError`.
    """
    p_id, p_src = prefill_item
    d_id, d_src = decode_item
    info_p = inspect_candidate_source(p_src)
    if not info_p["syntax_ok"]:
        raise ValueError(f"Prefill candidate {p_id} has invalid Python syntax")
    info_d = inspect_candidate_source(d_src)
    if not info_d["syntax_ok"]:
        raise ValueError(f"Decode candidate {d_id} has invalid Python syntax")

    parent_ids: list[str] = []
    for jid in (p_id, d_id):
        if jid and jid not in parent_ids:
            parent_ids.append(jid)

    merged_tags: list[str] = []
    for tag in [*info_p["tags"], *info_d["tags"], "pd_pair"]:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)

    colocated_conflicts = (
        [] if p_id == d_id else detect_candidate_conflicts([(p_id, info_p), (d_id, info_d)])
    )
    has_colocated_overlap = bool(colocated_conflicts)

    mock_decode_sp = info_d["mock_speedup"]
    mock_prefill_sp = (
        info_p["mock_prefill_speedup"]
        or info_p["mock_bucket_speedups"].get("prefill_large")
        or info_p["mock_speedup"]
    )
    combined_buckets: dict[str, float] = {}
    for k in ("decode_small", "decode_mid"):
        if k in info_d["mock_bucket_speedups"]:
            combined_buckets[k] = round(float(info_d["mock_bucket_speedups"][k]), 6)
        elif mock_decode_sp is not None:
            combined_buckets[k] = round(float(mock_decode_sp), 6)
    if "prefill_large" in info_p["mock_bucket_speedups"]:
        combined_buckets["prefill_large"] = round(float(info_p["mock_bucket_speedups"]["prefill_large"]), 6)
    elif mock_prefill_sp is not None:
        combined_buckets["prefill_large"] = round(float(mock_prefill_sp), 6)

    pd_needs_prebuild = bool(info_p.get("needs_prebuild") or info_d.get("needs_prebuild"))

    lines: list[str] = [
        '"""Asymmetric Prefill/Decode candidate pair synthesized by `kiln merge --prefill --decode`."""',
        "from __future__ import annotations",
        "",
        "import linecache",
        "import os",
        "import sys",
        "import types",
        "",
        f"KILN_PARENTS = {parent_ids!r}",
        f"KILN_TAGS = {merged_tags!r}",
        f"KILN_PD_PAIR = {{'prefill': {p_id!r}, 'decode': {d_id!r}}}",
    ]
    if pd_needs_prebuild:
        lines.append("KILN_PREBUILD = True")
    if mock_decode_sp is not None:
        lines.append(f"SPEEDUP = {round(float(mock_decode_sp), 6)!r}")
    if mock_prefill_sp is not None:
        lines.append(f"PREFILL_SPEEDUP = {round(float(mock_prefill_sp), 6)!r}")
    if combined_buckets:
        lines.append(f"BUCKET_SPEEDUPS = {combined_buckets!r}")

    lines.extend([
        "",
        f"_PREFILL_SPEC: tuple[str, str] = ({p_id!r}, {p_src!r})",
        f"_DECODE_SPEC: tuple[str, str] = ({d_id!r}, {d_src!r})",
        f"_HAS_COLOCATED_OVERLAP: bool = {has_colocated_overlap!r}",
        "_LOADED: list[tuple[str, types.ModuleType]] = []",
        "",
        "",
        "def _active_role(model) -> str:",
        "    attr = getattr(model, '_kiln_pd_role', None) if model is not None else None",
        "    raw = (attr if isinstance(attr, str) and attr.strip() else os.environ.get('KILN_PD_ROLE', '')).strip().lower()",
        "    if raw in ('prefill', 'prefill_only', 'kv_producer'):",
        "        return 'prefill'",
        "    if raw in ('decode', 'decode_only', 'kv_consumer'):",
        "        return 'decode'",
        "    return 'both'",
        "",
        "",
        "def _load_role_module(role_tag: str, jid: str, src: str) -> types.ModuleType:",
        "    safe_id = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in jid)",
        "    mod_name = f'_kiln_pd_{role_tag}_{safe_id}'",
        "    mod = types.ModuleType(mod_name)",
        "    mod.__file__ = f'<kiln_pd:{role_tag}:{jid}>'",
        "    linecache.cache[mod.__file__] = (len(src), None, src.splitlines(keepends=True), mod.__file__)",
        "    sys.modules[mod_name] = mod",
        "    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)",
        "    return mod",
        "",
        "",
        "def _restore_applied(applied: list[tuple[str, types.ModuleType]], model) -> None:",
        "    first_err = None",
        "    for mod_name, mod in reversed(applied):",
        "        restore_fn = getattr(mod, 'restore', None)",
        "        if callable(restore_fn):",
        "            try:",
        "                restore_fn(model)",
        "            except BaseException as exc:",
        "                if first_err is None:",
        "                    first_err = exc",
        "        linecache.cache.pop(getattr(mod, '__file__', None), None)",
        "        sys.modules.pop(mod_name, None)",
        "    if first_err is not None:",
        "        raise first_err",
        "",
        "",
        "def make_forward(model):",
        "    restore(model)",
        "    role = _active_role(model)",
        "    if role == 'prefill':",
        "        specs = [('prefill', *_PREFILL_SPEC)]",
        "    elif role == 'decode':",
        "        specs = [('decode', *_DECODE_SPEC)]",
        "    elif _PREFILL_SPEC[0] == _DECODE_SPEC[0] or _HAS_COLOCATED_OVERLAP:",
        "        specs = [('decode', *_DECODE_SPEC)]",
        "    else:",
        "        specs = [('prefill', *_PREFILL_SPEC), ('decode', *_DECODE_SPEC)]",
        "    root_forward = None",
        "    applied: list[tuple[str, types.ModuleType]] = []",
        "    try:",
        "        for role_tag, jid, src in specs:",
        "            mod = _load_role_module(role_tag, jid, src)",
        "            applied.append((mod.__name__, mod))",
        "            _LOADED.append((mod.__name__, mod))",
        "            mk = getattr(mod, 'make_forward', None)",
        "            if not callable(mk):",
        "                from kiln.patches import make_composite_forward",
        "                mk, auto_restore = make_composite_forward(vars(mod))",
        "                if not hasattr(mod, 'restore'):",
        "                    mod.restore = auto_restore",
        "            new_fwd = mk(model)",
        "            if new_fwd is not None:",
        "                model.forward = new_fwd",
        "                root_forward = new_fwd",
        "    except BaseException:",
        "        _LOADED.clear()",
        "        _restore_applied(applied, model)",
        "        raise",
        "    return root_forward",
        "",
        "",
        "def restore(model) -> None:",
        "    applied = list(_LOADED)",
        "    _LOADED.clear()",
        "    if applied:",
        "        _restore_applied(applied, model)",
        "",
    ])
    if pd_needs_prebuild:
        lines.extend([
            "def kiln_build() -> None:",
            "    role = _active_role(None)",
            "    specs = [('prefill', *_PREFILL_SPEC)] if role == 'prefill' else (",
            "        [('decode', *_DECODE_SPEC)] if role == 'decode' else [('prefill', *_PREFILL_SPEC), ('decode', *_DECODE_SPEC)]",
            "    )",
            "    loaded: list[tuple[str, types.ModuleType]] = []",
            "    try:",
            "        for role_tag, jid, src in specs:",
            "            mod = _load_role_module(role_tag, jid, src)",
            "            loaded.append((mod.__name__, mod))",
            "            build_fn = getattr(mod, 'kiln_build', None)",
            "            if callable(build_fn):",
            "                build_fn()",
            "    finally:",
            "        for mod_name, mod in reversed(loaded):",
            "            linecache.cache.pop(getattr(mod, '__file__', None), None)",
            "            sys.modules.pop(mod_name, None)",
            "",
        ])

    return {
        "source": "\n".join(lines),
        "parent_ids": parent_ids,
        "prefill_id": p_id,
        "decode_id": d_id,
        "tags": merged_tags,
        "colocated_conflicts": colocated_conflicts,
    }
