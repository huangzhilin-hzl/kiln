"""Modular patch contract, AST metadata/conflict inspector, and orthogonal candidate composer."""
from __future__ import annotations

import ast
import math
import re
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


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


def kiln_patch(
    name: str,
    *,
    regime: str = "all",
    targets: Sequence[str] = (),
    description: str = "",
    returns_forward: bool = False,
) -> Callable[[Callable], Callable]:
    """Mark a modular patch callable with its orthogonal name, regime, and patched targets."""
    spec = PatchSpec(
        name=str(name).strip(),
        regime=str(regime).strip() or "all",
        targets=tuple(str(t).strip() for t in targets if str(t).strip()),
        description=str(description).strip(),
        returns_forward=bool(returns_forward),
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
        try:
            for patch_fn in patches:
                spec = getattr(patch_fn, "__kiln_patch__", None)
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
        elif kw.arg == "targets":
            targets.extend(_literal_str_seq(kw.value))
    return {"name": name, "regime": regime, "targets": targets}


def _infer_tags(tokens: Sequence[str]) -> list[str]:
    joined = " ".join(tokens).lower()
    out: list[str] = []
    for tag, keywords in _CANONICAL_TAG_KEYWORDS:
        if any(kw in joined for kw in keywords):
            out.append(tag)
    return out


def inspect_candidate_source(source: str) -> dict[str, Any]:
    """Statically parse candidate source for tags, parent IDs, regimes, patches, targets, and mock speedups."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "syntax_ok": False,
            "tags": [],
            "parents": [],
            "regimes": [],
            "patches": [],
            "targets": [],
            "mock_speedup": None,
            "mock_bucket_speedups": {},
        }

    tags: list[str] = []
    parents: list[str] = []
    regimes: list[str] = []
    patches: list[dict[str, Any]] = []
    targets: list[str] = []
    seen_tags: set[str] = set()
    seen_parents: set[str] = set()
    seen_regimes: set[str] = set()
    seen_targets: set[str] = set()
    symbol_tokens: list[str] = []
    imported_aliases: dict[str, str] = {}
    mock_speedup: float | None = None
    mock_bucket_speedups: dict[str, float] = {}

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

    # Top-level imports & metadata constants
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                imported_aliases[local] = alias.name
                symbol_tokens.append(alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            symbol_tokens.append(mod)
            for alias in stmt.names:
                local = alias.asname or alias.name
                full = f"{mod}.{alias.name}" if mod else alias.name
                imported_aliases[local] = full
                symbol_tokens.append(alias.name)
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
                elif tgt.id == "SPEEDUP":
                    num = _literal_num(stmt.value)
                    if num is not None:
                        mock_speedup = num
                elif tgt.id == "BUCKET_SPEEDUPS":
                    mock_bucket_speedups = _literal_bucket_map(stmt.value)

    # Walk AST for @kiln_patch decorators, class/module patching targets, and symbol hints
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_tokens.append(node.name)
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
                for t in pinfo["targets"]:
                    _add_target(t)
        elif isinstance(node, ast.ClassDef):
            symbol_tokens.append(node.name)
        elif isinstance(node, ast.Call):
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

    for inferred in _infer_tags([*symbol_tokens, *targets]):
        _add_tag(inferred)

    return {
        "syntax_ok": True,
        "tags": tags,
        "parents": parents,
        "regimes": regimes,
        "patches": patches,
        "targets": targets,
        "mock_speedup": mock_speedup,
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

    lines: list[str] = [
        '"""Auto-composed orthogonal candidate synthesized by `kiln merge`."""',
        "from __future__ import annotations",
        "",
        "import sys",
        "import types",
        "",
        f"KILN_PARENTS = {parent_ids!r}",
        f"KILN_TAGS = {merged_tags!r}",
    ]
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
        "    sys.modules[mod_name] = mod",
        "    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)",
        "    return mod",
        "",
        "",
        "def _cleanup_loaded() -> None:",
        "    for mod_name, _ in reversed(_LOADED_PARENTS):",
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

    return {
        "source": "\n".join(lines),
        "parent_ids": parent_ids,
        "tags": merged_tags,
        "conflicts": conflicts,
    }
