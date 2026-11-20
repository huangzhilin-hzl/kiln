from __future__ import annotations

from kiln.config import EngineConfig, TierConfig

from .base import Engine


def make_engine(cfg: EngineConfig, tier: TierConfig) -> Engine:
    kind = tier.engine_kind or cfg.kind
    if kind == "mock":
        from .mock import MockEngine
        return MockEngine(tier)
    if kind in ("vllm_offline", "vllm"):
        from .vllm_offline import VLLMOfflineEngine
        return VLLMOfflineEngine(cfg, tier)
    if kind == "vllm_server":
        from .vllm_server import VLLMServerEngine
        return VLLMServerEngine(cfg, tier)
    if kind in ("sglang_offline", "sglang"):
        from .sglang_offline import SGLangOfflineEngine
        return SGLangOfflineEngine(cfg, tier)
    if kind == "sglang_server":
        from .sglang_server import SGLangServerEngine
        return SGLangServerEngine(cfg, tier)
    raise ValueError(f"unknown engine kind {kind}")
