"""Engine-agnostic forward-pass snapshot and restore helpers for candidate hot reload."""
from __future__ import annotations


def snapshot_forwards(model) -> list[tuple]:
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


def restore_forwards(snap: list[tuple] | None) -> None:
    """Restore `.forward` bindings captured by `snapshot_forwards`."""
    if not snap:
        return
    for sub, had_inst, inst_fwd, cls, cls_fwd in snap:
        if had_inst:
            sub.__dict__["forward"] = inst_fwd
        else:
            sub.__dict__.pop("forward", None)
        if cls_fwd is not None and getattr(cls, "forward", None) is not cls_fwd:
            cls.forward = cls_fwd
