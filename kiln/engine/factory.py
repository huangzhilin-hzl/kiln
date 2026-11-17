from __future__ import annotations

from kiln.config import EngineConfig, TierConfig

from .base import Engine


def make_engine(cfg: EngineConfig, tier: TierConfig) -> Engine:
    kind = tier.engine_kind or cfg.kind
    if kind == "mock":
        from .mock import MockEngine
        return MockEngine(tier)
    if kind == "vllm":
        from .vllm_engine import VLLMEngine
        return VLLMEngine(cfg, tier)
    if kind == "vllm_server":
        from .vllm_server import VLLMServerEngine
        return VLLMServerEngine(cfg, tier)
    raise ValueError(f"unknown engine kind {kind}")
