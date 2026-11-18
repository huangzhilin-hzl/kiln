"""Baseline candidate: identical to the engine's own forward.

Contract: define `make_forward(model)` returning a callable with the same signature
as `model.forward` (or `None` if patching submodules in place). Teardown automatically
restores `.forward` on `model` and all submodules; never mutate weights in place.

Example of a real optimization: replace a submodule's forward with a fused Triton
kernel (registered via `torch.library.custom_op` so `torch.compile` + CUDA graphs trace it),
then delegate everything else to the original:

    def make_forward(model):
        orig = type(model).forward
        for _, m in model.named_modules():
            if type(m).__name__.endswith("MLP"):
                m.forward = types.MethodType(my_fused_mlp, m)
        return types.MethodType(orig, model)
"""
import types


def make_forward(model):
    return types.MethodType(type(model).forward, model)


# Used only by the mock engine (laptop dry-runs):
SPEEDUP = 1.0
