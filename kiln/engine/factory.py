from __future__ import annotations

from kiln.config import EngineConfig, TierConfig

from .base import Engine


def make_engine(cfg: EngineConfig, tier: TierConfig) -> Engine:
    if cfg.kind == "mock":
        from .mock import MockEngine
        return MockEngine(tier)
    if cfg.kind == "vllm":
        from .vllm_engine import VLLMEngine
        return VLLMEngine(cfg, tier)
    raise ValueError(f"unknown engine kind {cfg.kind}")
