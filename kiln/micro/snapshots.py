"""Cold-boot submodule tensor snapshot capture and replay for lock-free `kiln micro`."""
from __future__ import annotations

import json
import pickle
import types
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import numpy as np

DEFAULT_M_BUCKETS: tuple[int, ...] = (1, 8, 32, 64, 256, 1024)

_TAG_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mhc", ("mhc",)),
    ("mla", ("mla", "flashmla", "sparse_mla")),
    ("attention", ("attention", "attn")),
    ("moe", ("moe", "experts")),
    ("mlp", ("mlp", "ffn")),
    ("norm", ("rmsnorm", "layernorm")),
)


def classify_submodule_tag(qualname: str, mod: Any) -> str | None:
    """Classify a model submodule into a canonical kernel family tag."""
    if not qualname:
        return None
    haystack = f"{qualname}.{type(mod).__name__}".lower()
    for tag, pats in _TAG_PATTERNS:
        if any(p in haystack for p in pats):
            return tag
    return None


def nearest_m_bucket(m: int, buckets: Sequence[int] = DEFAULT_M_BUCKETS) -> int | None:
    """Return the nearest M bucket within a 2x tolerance, or `None` if outside range."""
    if m <= 0 or not buckets:
        return None
    best = min(buckets, key=lambda b: (abs(int(b) - m), int(b)))
    if m <= 2 * best and best <= 2 * m:
        return int(best)
    return None


def _is_torch_tensor(obj: Any) -> bool:
    return hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numel")


def _to_cpu_value(obj: Any) -> tuple[Any, int, list[str], list[list[int]], bool]:
    """Detach tensors to CPU and report `(cpu_obj, nbytes, dtypes, shapes, has_torch)`."""
    if _is_torch_tensor(obj):
        t = obj.detach().cpu().contiguous()
        nbytes = int(t.numel() * t.element_size())
        return t, nbytes, [str(t.dtype)], [list(t.shape)], True
    if isinstance(obj, np.ndarray):
        arr = np.array(obj, copy=True)
        return arr, int(arr.nbytes), [str(arr.dtype)], [list(arr.shape)], False
    if isinstance(obj, tuple):
        items, total_b, dts, shps, has_t = [], 0, [], [], False
        for it in obj:
            co, nb, dt, sp, ht = _to_cpu_value(it)
            items.append(co)
            total_b += nb
            dts.extend(dt)
            shps.extend(sp)
            has_t = has_t or ht
        return tuple(items), total_b, dts, shps, has_t
    if isinstance(obj, list):
        items, total_b, dts, shps, has_t = [], 0, [], [], False
        for it in obj:
            co, nb, dt, sp, ht = _to_cpu_value(it)
            items.append(co)
            total_b += nb
            dts.extend(dt)
            shps.extend(sp)
            has_t = has_t or ht
        return items, total_b, dts, shps, has_t
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        total_b, dts, shps, has_t = 0, [], [], False
        for k, v in obj.items():
            co, nb, dt, sp, ht = _to_cpu_value(v)
            out[k] = co
            total_b += nb
            dts.extend(dt)
            shps.extend(sp)
            has_t = has_t or ht
        return out, total_b, dts, shps, has_t
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj, 0, [], [], False
    return None, 0, [], [], False


def _infer_m_from_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    for val in (*args, *kwargs.values()):
        if _is_torch_tensor(val) or isinstance(val, np.ndarray):
            shape = tuple(int(s) for s in val.shape)
            if not shape:
                continue
            if len(shape) >= 3:
                return int(shape[0] * shape[1])
            return int(shape[0])
    return None


class SnapshotCollector:
    """Attach forward hooks to representative layers to capture real `(inputs, output)` snapshots."""

    def __init__(
        self,
        out_dir: str | Path,
        m_buckets: Sequence[int] = DEFAULT_M_BUCKETS,
        max_snapshot_mb: float = 64.0,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.m_buckets = tuple(int(b) for b in m_buckets)
        self.max_bytes = int(max(1.0, float(max_snapshot_mb)) * 1024 * 1024)
        self.total_bytes = 0
        self.captured_keys: set[tuple[str, int]] = set()
        self.records: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._wrapped_fwds: list[tuple[Any, bool, Any]] = []

    def attach(self, model: Any) -> list[dict[str, str]]:
        """Attach capture hooks to one representative submodule per canonical tag."""
        attached: list[dict[str, str]] = []
        picked_tags: set[str] = set()
        it = model.named_modules() if hasattr(model, "named_modules") else []
        for qualname, mod in it:
            tag = classify_submodule_tag(qualname, mod)
            if tag is None or tag in picked_tags:
                continue
            picked_tags.add(tag)
            cls_name = type(mod).__name__
            if callable(getattr(mod, "register_forward_hook", None)):
                handle = mod.register_forward_hook(
                    self._make_torch_hook(tag, qualname, cls_name),
                    with_kwargs=True,
                )
                self._handles.append(handle)
            elif callable(getattr(mod, "forward", None)):
                had_inst = "forward" in getattr(mod, "__dict__", {})
                orig_inst = mod.__dict__.get("forward") if had_inst else None
                orig_fn = mod.forward

                def _make_wrapper(_orig, _tag, _q, _cls):
                    def _wrapped(this, *args, **kwargs):
                        out = _orig(*args, **kwargs)
                        self._record(_tag, _q, _cls, args, kwargs, out)
                        return out

                    return _wrapped

                mod.forward = types.MethodType(_make_wrapper(orig_fn, tag, qualname, cls_name), mod)
                self._wrapped_fwds.append((mod, had_inst, orig_inst))
            attached.append({"tag": tag, "module": qualname, "cls": cls_name})
        return attached

    def _make_torch_hook(self, tag: str, qualname: str, cls_name: str):
        def _hook(_mod, args, kwargs, output):
            self._record(tag, qualname, cls_name, args, kwargs or {}, output)

        return _hook

    def _record(
        self,
        tag: str,
        qualname: str,
        cls_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        m_obs = _infer_m_from_inputs(args, kwargs)
        if m_obs is None:
            return
        bucket = nearest_m_bucket(m_obs, self.m_buckets)
        if bucket is None or (tag, bucket) in self.captured_keys:
            return
        if self.total_bytes >= self.max_bytes:
            return

        c_args, b_a, dt_a, sp_a, ht_a = _to_cpu_value(args)
        c_kw, b_k, dt_k, sp_k, ht_k = _to_cpu_value(kwargs)
        c_out, b_o, dt_o, sp_o, ht_o = _to_cpu_value(output)
        payload_bytes = b_a + b_k + b_o
        if self.total_bytes + payload_bytes > self.max_bytes:
            return

        has_torch = ht_a or ht_k or ht_o
        sid = f"{tag}_M{bucket}"
        fname = f"{sid}.pt" if has_torch else f"{sid}.pkl"
        fpath = self.out_dir / fname
        payload = {"args": c_args, "kwargs": c_kw, "output": c_out}
        if has_torch:
            import torch

            torch.save(payload, fpath)
        else:
            with open(fpath, "wb") as f:
                pickle.dump(payload, f)

        self.total_bytes += payload_bytes
        self.captured_keys.add((tag, bucket))
        self.records.append({
            "id": sid,
            "tag": tag,
            "m_bucket": bucket,
            "observed_m": m_obs,
            "module": qualname,
            "cls": cls_name,
            "file": fname,
            "bytes": payload_bytes,
            "dtypes": [*dt_a, *dt_k, *dt_o],
            "input_shapes": [*sp_a, *sp_k],
            "output_shapes": sp_o,
            "backend": "torch" if has_torch else "numpy",
        })

    def close(self) -> dict[str, Any]:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        for mod, had_inst, orig_inst in self._wrapped_fwds:
            if had_inst:
                mod.__dict__["forward"] = orig_inst
            else:
                mod.__dict__.pop("forward", None)
        self._wrapped_fwds.clear()

        manifest = {
            "schema_version": 1,
            "m_buckets": list(self.m_buckets),
            "total_bytes": self.total_bytes,
            "snapshots": sorted(self.records, key=lambda r: (r["tag"], r["m_bucket"])),
        }
        _write_manifest(self.out_dir / "manifest.json", manifest)
        return manifest


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        try:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def write_mock_snapshots(
    snapshot_dir: str | Path,
    m_buckets: Sequence[int] = DEFAULT_M_BUCKETS,
    dim: int = 64,
) -> dict[str, Any]:
    """Generate deterministic NumPy snapshots for CPU laptop tests and mock runs."""
    out_dir = Path(snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for tag, mod_name in (
        ("attention", "model.layers.0.self_attn"),
        ("moe", "model.layers.0.mlp.experts"),
        ("mhc", "model.layers.0.mhc"),
    ):
        w = (rng.standard_normal((dim, dim)) * 0.05).astype(np.float32)
        for m in m_buckets:
            m_int = int(m)
            x = rng.standard_normal((m_int, dim)).astype(np.float32)
            y = x @ w
            sid = f"{tag}_M{m_int}"
            fname = f"{sid}.pkl"
            with open(out_dir / fname, "wb") as f:
                pickle.dump({"args": (x, w), "kwargs": {}, "output": y}, f)
            nbytes = int(x.nbytes + w.nbytes + y.nbytes)
            total_bytes += nbytes
            records.append({
                "id": sid,
                "tag": tag,
                "m_bucket": m_int,
                "observed_m": m_int,
                "module": mod_name,
                "cls": f"Mock{tag.upper()}",
                "file": fname,
                "bytes": nbytes,
                "dtypes": ["float32", "float32", "float32"],
                "input_shapes": [[m_int, dim], [dim, dim]],
                "output_shapes": [[m_int, dim]],
                "backend": "numpy",
            })
    manifest = {
        "schema_version": 1,
        "m_buckets": [int(b) for b in m_buckets],
        "total_bytes": total_bytes,
        "snapshots": records,
    }
    _write_manifest(out_dir / "manifest.json", manifest)
    return manifest


def load_snapshots(snapshot_dir: str | Path, tag: str | None = None) -> list[dict[str, Any]]:
    """Load snapshot manifest records from `snapshot_dir`, optionally filtered by `tag`."""
    base = Path(snapshot_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in data.get("snapshots", ()):
        if tag is not None and row.get("tag") != tag:
            continue
        item = dict(row)
        item["path"] = str((base / row["file"]).resolve())
        rows.append(item)
    rows.sort(key=lambda r: (str(r.get("tag", "")), int(r.get("m_bucket", 0))))
    return rows


def load_snapshot_payload(path: str | Path) -> dict[str, Any]:
    """Load serialized snapshot payload (`.pt`, `.pkl`, or `.npz`)."""
    p = Path(path)
    if p.suffix == ".pt":
        import torch

        return torch.load(p, map_location="cpu", weights_only=False)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    with open(p, "rb") as f:
        return pickle.load(f)
