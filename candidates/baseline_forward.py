"""Baseline candidate: identical to the engine's own forward.

Contract: define `make_forward(model)` returning a callable with the same signature
as `model.forward`. Agents copy this file to forward.py and optimize from here.

Example of a real optimization: replace a submodule's forward with a fused Triton
kernel, then delegate everything else to the original:

    def make_forward(model):
        orig = type(model).forward
        for layer in model.model.layers:
            layer.mlp.forward = types.MethodType(my_fused_mlp, layer.mlp)
        return types.MethodType(orig, model)
"""
import types


def make_forward(model):
    return types.MethodType(type(model).forward, model)


# Used only by the mock engine (laptop dry-runs):
SPEEDUP = 1.0
