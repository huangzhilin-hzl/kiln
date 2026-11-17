"""Run configuration (loaded from configs/*.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class TierConfig(BaseModel):
    name: Literal["fast", "slow"]
    # Number of decoder layers kept in this tier's engine (None = all layers).
    num_layers: int | None = None
    gpus: list[int] = Field(default_factory=list)
    repeats: int = 3               # back-to-back measurements per job
    trace_kernels: bool = False    # CUPTI / torch.profiler kernel report
    timeout_s: float = 600.0


class EngineConfig(BaseModel):
    kind: Literal["mock", "vllm"] = "mock"
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    extra_args: dict = Field(default_factory=dict)


class CorrectnessConfig(BaseModel):
    # Greedy-decoded tokens must match baseline for at least this fraction of positions.
    min_token_match: float = 0.98
    max_logit_abs_diff: float = 0.5
    num_prompts: int = 8
    max_new_tokens: int = 64


class AgentConfig(BaseModel):
    num_agents: int = 4
    command: str = "claude -p --dangerously-skip-permissions"   # any CLI agent that reads a prompt from stdin
    budget_hours: float = 12.0


class RunConfig(BaseModel):
    name: str = "default"
    engine: EngineConfig = Field(default_factory=EngineConfig)
    tiers: list[TierConfig] = Field(
        default_factory=lambda: [TierConfig(name="fast", num_layers=4, trace_kernels=True),
                                 TierConfig(name="slow")]
    )
    traffic_file: str = "traffic/sample.jsonl"
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    workdir: str = "runs"
    server_url: str = "http://127.0.0.1:8765"

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
