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
    max_requests: int | None = None  # cap offline traffic replay count (useful when keeping all layers)
    repeats: int = 3                 # back-to-back measurements per job
    trace_kernels: bool = False      # CUPTI / torch.profiler kernel report
    # Extra LLM(...) kwargs for this tier only, merged over engine.extra_args.
    # e.g. fast tier: {speculative_config: null} -- truncated models break MTP/DSpark layers.
    engine_overrides: dict = Field(default_factory=dict)
    # Per-tier engine kind (defaults to engine.kind). "vllm_server" = `vllm serve` + AgentX harness.
    engine_kind: Literal["mock", "vllm", "vllm_server"] | None = None
    port: int | None = None                      # vllm_server only
    agentx: dict | None = None                   # vllm_server only: AgentXConfig fields
    timeout_s: float = 600.0


class EngineConfig(BaseModel):
    kind: Literal["mock", "vllm", "vllm_server"] = "mock"
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer: str | None = None                 # defaults to model
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
    command: str = "codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c shell_environment_policy.inherit=all -"
    budget_hours: float = 12.0
    respawn_on_exit: bool = True     # re-launch single-turn CLI agents (e.g. `codex exec`) until budget_hours


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

    @property
    def client_url(self) -> str:
        return self.server_url.replace("://0.0.0.0:", "://127.0.0.1:")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
