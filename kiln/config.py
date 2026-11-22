"""Run configuration (loaded from configs/*.yaml)."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from kiln.bench.agentx import AgentXConfig


EngineKind = Literal[
    "mock",
    "vllm_offline",
    "vllm_server",
    "vllm",
    "sglang_offline",
    "sglang_server",
    "sglang_pd_server",
    "sglang",
]

TierPhase = Literal["both", "prefill", "decode"]
TierMetric = Literal["decode_tps", "prefill_tps", "ttft"]
PDTransferBackend = Literal["mooncake", "mooncake_tcp", "nixl", "ascend", "mori", "fake"]
_TIER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class PDNodeConfig(BaseModel):
    """One Prefill or Decode SGLang server replica in an `xPyD` disaggregated cluster."""

    url: str | None = None                       # e.g. "http://10.0.0.11:30000" (if remote/pre-launched)
    host: str = "127.0.0.1"
    port: int | None = None
    bootstrap_port: int | None = None            # prefill `--disaggregation-bootstrap-port` (default 8998)
    gpus: list[int] = Field(default_factory=list)
    num_gpus: int | None = Field(default=None, ge=1)
    spawn: bool | None = None                    # True = spawn locally; False = connect to existing URL
    engine_overrides: dict = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    def should_spawn(self) -> bool:
        if self.spawn is not None:
            return bool(self.spawn)
        return self.url is None

    def resolved_url(self, default_port: int) -> str:
        if self.url:
            return self.url.rstrip("/")
        port = self.port or default_port
        bind_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{bind_host}:{port}"

    def resolved_num_gpus(self, cfg: "EngineConfig", tier_overrides: dict) -> int:
        if self.num_gpus is not None:
            return int(self.num_gpus)
        if self.gpus:
            return len(self.gpus)
        merged = {**cfg.extra_args, **tier_overrides, **self.engine_overrides}
        tp = int(merged.get("tp_size", cfg.tensor_parallel_size))
        dp = int(merged.get("dp_size", 1))
        dp_attn = bool(merged.get("enable_dp_attention", False))
        return tp if dp_attn else tp * dp


class PDClusterConfig(BaseModel):
    """Topology and router settings for `sglang_pd_server` (`xPyD` disaggregation)."""

    transfer_backend: PDTransferBackend = "mooncake"
    ib_device: str | None = None
    prefill: list[PDNodeConfig] = Field(default_factory=list)
    decode: list[PDNodeConfig] = Field(default_factory=list)
    router_url: str | None = None                # connect to pre-running router when set (unless spawn_router=True)
    spawn_router: bool | None = None
    router_host: str = "127.0.0.1"
    router_port: int | None = None
    router_policy: str | None = None             # e.g. "cache_aware", "round_robin", "power_of_two"
    router_command: str | None = None            # override router CLI prefix
    router_extra_args: list[str] = Field(default_factory=list)

    def should_spawn_router(self) -> bool:
        if self.spawn_router is not None:
            return bool(self.spawn_router)
        return self.router_url is None

    @model_validator(mode="after")
    def _validate_topology(self) -> "PDClusterConfig":
        if not self.prefill:
            raise ValueError("PDClusterConfig requires at least one prefill instance")
        if not self.decode:
            raise ValueError("PDClusterConfig requires at least one decode instance")
        if self.transfer_backend == "fake":
            raise ValueError(
                "PDClusterConfig transfer_backend='fake' is not supported for full xPyD "
                "(SGLang prefill servers reject 'fake'); use 'mooncake', 'mooncake_tcp', or 'nixl'"
            )
        return self


class TierConfig(BaseModel):
    name: str
    phase: TierPhase = "both"
    metric: TierMetric | None = None
    # Number of decoder layers kept in this tier's engine (None = all layers).
    num_layers: int | None = None
    gpus: list[int] = Field(default_factory=list)
    max_requests: int | None = None  # cap offline traffic replay count (useful when keeping all layers)
    repeats: int = Field(default=3, ge=1)  # back-to-back measurements per job
    trace_kernels: bool = False      # CUPTI / torch.profiler kernel report
    # Extra LLM(...) kwargs for this tier only, merged over engine.extra_args.
    # e.g. fast tier: {speculative_config: null} -- truncated models break MTP/DSpark layers.
    engine_overrides: dict = Field(default_factory=dict)
    # Per-tier engine kind (defaults to engine.kind): "vllm_offline", "vllm_server", "sglang_offline", "sglang_server", or "sglang_pd_server".
    engine_kind: EngineKind | None = None
    share_engine: str | None = None              # reuse another local tier's engine (single-node mode)
    port: int | None = None                      # *_server only
    pd: PDClusterConfig | None = None            # sglang_pd_server topology
    agentx: dict | None = None                   # *_server only: AgentXConfig fields
    timeout_s: float = 600.0
    max_queued_per_agent: int | None = Field(default=None, ge=1)
    max_inflight_per_agent: int | None = Field(default=None, ge=1)
    deduplicate_queued: bool = False
    supersede_queued: bool = False
    canary_every_n_jobs: int | None = Field(default=None, ge=1)
    screen_repeats: int | None = Field(default=None, ge=1)
    screen_min_speedup: float = Field(default=1.0, gt=0.0)
    screen_top_k: int | None = Field(default=None, ge=1)
    trace_policy: Literal["always", "on_improve", "never"] = "always"
    lazy_baseline: bool = False

    @model_validator(mode="after")
    def _validate_screening(self) -> "TierConfig":
        if self.screen_repeats is not None and self.screen_repeats > self.repeats:
            raise ValueError("screen_repeats must be <= repeats")
        return self

    @model_validator(mode="after")
    def _validate_pd_tier(self) -> "TierConfig":
        if self.engine_kind == "sglang_pd_server" and self.pd is None:
            raise ValueError(f"Tier {self.name!r} with engine_kind='sglang_pd_server' requires a 'pd' block")
        return self

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        clean = value.strip()
        if not clean or not _TIER_NAME_RE.match(clean):
            raise ValueError(f"Invalid tier name {value!r}; use alphanumeric, '_' or '-'")
        if clean == "micro":
            raise ValueError("Tier name 'micro' is reserved for the lock-free micro sandbox")
        return clean

    @property
    def resolved_metric(self) -> TierMetric:
        if self.metric is not None:
            return self.metric
        return "prefill_tps" if self.phase == "prefill" else "decode_tps"

    @field_validator("agentx")
    @classmethod
    def validate_agentx(cls, value: dict | None) -> dict | None:
        if value is not None:
            try:
                AgentXConfig(**value)
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        return value


class EngineConfig(BaseModel):
    kind: EngineKind = "mock"
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


AgentProvider = Literal["agy", "codex", "claude", "custom"]
AgentRole = Literal["explorer", "auditor", "integrator", "synthesizer"]

PROVIDER_COMMANDS: dict[str, str] = {
    "agy": (
        "agy --dangerously-skip-permissions --disable-slash-commands "
        "--settings '{\"enableTerminalSandbox\":false}' -p {prompt}"
    ),
    "codex": (
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check --model gpt-6.2-sol "
        "-c model_reasoning_effort=high -c shell_environment_policy.inherit=all -"
    ),
    "claude": "claude -p --dangerously-skip-permissions",
}


def _infer_provider(argv: list[str]) -> str:
    exe = Path(argv[0]).name if argv else ""
    return exe if exe in ("agy", "codex", "claude") else "custom"


def _insert_provider_args(argv: list[str], provider: str, extra: list[str]) -> list[str]:
    if not extra:
        return list(argv)
    if provider == "agy":
        for i, tok in enumerate(argv):
            if tok == "-p" and i + 1 < len(argv) and "{prompt}" in argv[i + 1]:
                return [*argv[:i], *extra, *argv[i:]]
        for i, tok in enumerate(argv):
            if "{prompt}" in tok:
                return [*argv[:i], *extra, *argv[i:]]
    if provider == "codex" or (argv and argv[-1] == "-"):
        if argv and argv[-1] == "-":
            return [*argv[:-1], *extra, "-"]
    return [*argv, *extra]


def _apply_model_to_argv(
    argv: list[str],
    model: str,
    provider: str | None = None,
    allow_insert: bool = False,
) -> list[str]:
    if any("{model}" in tok for tok in argv):
        return [tok.replace("{model}", model) for tok in argv]
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--model", "-m") and i + 1 < len(argv):
            out.extend([tok, model])
            replaced = True
            i += 2
            continue
        if tok.startswith("--model="):
            out.append(f"--model={model}")
            replaced = True
            i += 1
            continue
        out.append(tok)
        i += 1
    if replaced:
        return out
    if allow_insert and provider in ("agy", "codex", "claude"):
        return _insert_provider_args(argv, provider, ["--model", model])
    raise ValueError(
        "agents.models requires '{model}' placeholder or '--model'/'-m' flag in agents.command(s)"
    )


def _extract_model(argv: list[str]) -> str | None:
    for i, tok in enumerate(argv):
        if tok in ("--model", "-m") and i + 1 < len(argv):
            return argv[i + 1] or None
        if tok.startswith("--model="):
            return tok.split("=", 1)[1] or None
    return None


@dataclass(frozen=True)
class ResolvedAgentSlot:
    argv: list[str]
    provider: str
    model: str | None
    env: dict[str, str]
    label: str | None = None
    role: str | None = None
    prompt_extra: str | None = None


class AgentSlotConfig(BaseModel):
    provider: AgentProvider | None = None
    model: str | None = None
    command: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    count: int = Field(default=1, ge=1)
    name: str | None = None
    role: AgentRole | None = None
    prompt_extra: str | None = None

    @model_validator(mode="after")
    def _validate_slot(self) -> "AgentSlotConfig":
        if self.provider is None and not (self.command and self.command.strip()):
            raise ValueError("AgentSlotConfig requires 'provider' or non-empty 'command'")
        if self.provider == "custom" and not (self.command and self.command.strip()):
            raise ValueError("AgentSlotConfig with provider='custom' requires non-empty 'command'")
        if self.command is not None and not self.command.strip():
            raise ValueError("AgentSlotConfig.command must not be blank")
        if self.model is not None and not self.model.strip():
            raise ValueError("AgentSlotConfig.model must not be blank")
        if any(not a.strip() for a in self.extra_args):
            raise ValueError("AgentSlotConfig.extra_args entries must not be blank")
        return self


def _strip_codex_reasoning_effort(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-c", "--config") and i + 1 < len(argv) and "model_reasoning_effort=" in argv[i + 1]:
            i += 2
            continue
        if (tok.startswith("-c=") or tok.startswith("--config=")) and "model_reasoning_effort=" in tok:
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


class AgentConfig(BaseModel):
    num_agents: int = Field(default=4, ge=1)
    command: str = (
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check --model gpt-6.2-sol "
        "-c model_reasoning_effort=high -c shell_environment_policy.inherit=all -"
    )
    models: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    slots: list[AgentSlotConfig] = Field(default_factory=list)
    budget_hours: float = 12.0
    respawn_on_exit: bool = True     # re-launch single-turn CLI agents (e.g. `codex exec`) until budget_hours
    isolate_credentials: bool = True # strip host cluster/cloud creds (KUBECONFIG, SSH_AUTH_SOCK, etc.)
    synthesizer_slot: bool = False   # auto-assign the final slot as `synthesizer` when no integrator/synthesizer exists

    def _has_explicit_synthesizer(self) -> bool:
        return any(s.role in ("synthesizer", "integrator") for s in self.slots)

    def resolve_slot(self, slot_idx: int) -> ResolvedAgentSlot:
        """Return resolved CLI argv, provider, model, and env overrides for `slot_idx`."""
        auto_synth = (
            self.synthesizer_slot
            and self.num_agents >= 2
            and slot_idx == self.num_agents - 1
            and not self._has_explicit_synthesizer()
        )
        if self.slots:
            expanded = [s for s in self.slots for _ in range(s.count)]
            spec = expanded[slot_idx % len(expanded)]
            raw_cmd = spec.command or PROVIDER_COMMANDS[spec.provider]  # type: ignore[index]
            argv = shlex.split(raw_cmd)
            if not argv:
                raise ValueError("agents slot command must not be empty")
            provider = spec.provider or _infer_provider(argv)
            if spec.model is not None:
                argv = _apply_model_to_argv(argv, spec.model, provider=provider, allow_insert=True)
                slot_model: str | None = spec.model
            else:
                if any("{model}" in tok for tok in argv):
                    raise ValueError("'{model}' placeholder in slot command requires 'model'")
                slot_model = _extract_model(argv)
            if spec.extra_args:
                if provider == "codex" and any("model_reasoning_effort=" in a for a in spec.extra_args):
                    argv = _strip_codex_reasoning_effort(argv)
                argv = _insert_provider_args(argv, provider, spec.extra_args)
            role = spec.role or ("synthesizer" if auto_synth else None)
            return ResolvedAgentSlot(
                argv=argv,
                provider=provider,
                model=slot_model,
                env=dict(spec.env),
                label=spec.name,
                role=role,
                prompt_extra=spec.prompt_extra,
            )

        raw_cmd = self.commands[slot_idx % len(self.commands)] if self.commands else self.command
        argv = shlex.split(raw_cmd)
        if not argv:
            raise ValueError("agents command must not be empty")
        provider = _infer_provider(argv)
        slot_model = self.models[slot_idx % len(self.models)] if self.models else None
        if slot_model is not None:
            argv = _apply_model_to_argv(argv, slot_model, provider=provider, allow_insert=False)
        else:
            if any("{model}" in tok for tok in argv):
                raise ValueError("'{model}' placeholder in agents.command(s) requires non-empty agents.models")
            slot_model = _extract_model(argv)
        return ResolvedAgentSlot(
            argv=argv,
            provider=provider,
            model=slot_model,
            env={},
            role="synthesizer" if auto_synth else None,
        )

    def resolve_slot_command(self, slot_idx: int) -> tuple[list[str], str | None]:
        """Return (argv, model) for agent slot `slot_idx`, cycling through `commands` and `models`."""
        slot = self.resolve_slot(slot_idx)
        return slot.argv, slot.model

    @model_validator(mode="after")
    def _validate_slots(self) -> "AgentConfig":
        if self.slots:
            expanded = [s for s in self.slots for _ in range(s.count)]
            if "num_agents" not in self.model_fields_set:
                self.num_agents = len(expanded)
            for i in range(max(self.num_agents, len(expanded))):
                self.resolve_slot(i)
            return self
        if any(not m.strip() for m in self.models):
            raise ValueError("agents.models entries must not be blank")
        if any(not c.strip() for c in self.commands):
            raise ValueError("agents.commands entries must not be blank")
        for i in range(max(self.num_agents, len(self.models), len(self.commands))):
            self.resolve_slot(i)
        return self


class MicroConfig(BaseModel):
    enabled: bool = True
    gpus: list[int] = Field(default_factory=list)
    max_concurrent: int = Field(default=2, ge=1)
    timeout_s: float = 60.0
    m_buckets: list[int] = Field(default_factory=lambda: [1, 8, 32, 64, 256, 1024])
    capture_on_boot: bool = True
    max_snapshot_mb: float = 64.0
    enable_ptxas: bool = True
    enable_ncu: bool = False
    ncu_timeout_s: float = 45.0
    atol: float = 1e-2
    rtol: float = 1e-2
    padded_nan_check: bool = True


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
    micro: MicroConfig = Field(default_factory=MicroConfig)
    hw_profile: str | None = None    # optional `kiln probe` JSON (auto-discovered in workdir if unset)
    workdir: str = "runs"
    server_url: str = "http://127.0.0.1:8765"

    @model_validator(mode="after")
    def _validate_unique_tiers(self) -> "RunConfig":
        seen: set[str] = set()
        for t in self.tiers:
            if t.name in seen:
                raise ValueError(f"Duplicate tier name {t.name!r} in tiers")
            seen.add(t.name)
            eff_kind = t.engine_kind or self.engine.kind
            if eff_kind == "sglang_pd_server" and t.pd is None:
                raise ValueError(f"Tier {t.name!r} resolved to sglang_pd_server requires a 'pd' block")
        return self

    @property
    def client_url(self) -> str:
        return self.server_url.replace("://0.0.0.0:", "://127.0.0.1:")

    def resolve_hw_profile(self) -> Path | None:
        if self.hw_profile:
            p = Path(self.hw_profile)
            if not p.exists():
                raise FileNotFoundError(f"Configured hw_profile does not exist: {p}")
            return p
        for cand in (Path(self.workdir) / self.name / "hw_profile.json",
                     Path(self.workdir) / "hw_profile.json"):
            if cand.exists():
                return cand
        return None

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
