# kiln 🔥

**Continual agentic compilation for LLM inference** — a DIY implementation of [Together Compiler (TKCC)](https://cdn.prod.website-files.com/69654e88dce9154b5f1206dd/6aeb3e27b0a7e915c3091fde_Together-AI_Together-Compiler.pdf).

Parallel coding agents rewrite a model's forward pass; each candidate is **hot-reloaded into a persistent vLLM or SGLang engine**, checked for greedy correctness, and benchmarked on **real traffic** across two tiers.

```
             ┌──────────── kiln agents (N in parallel) ─────────────┐
             │ agent_1   agent_2   agent_3 ...   ⇄  shared board    │
             └──────┬──────────────────────────────────────────────┘
                    │ kiln submit forward.py --tier fast|slow
             ┌──────▼──────────── kiln serve (job server) ──────────┐
             │  fast tier queue           slow tier queue           │
             │  ┌────────────────┐        ┌────────────────┐        │
             │  │ persistent eng │        │ persistent eng │        │
             │  │ N layers+trace │        │ all layers     │        │
             │  └────────────────┘        └────────────────┘        │
             │  hot_reload → correctness → measure → teardown       │
             └──────────────────────────────────────────────────────┘
   kiln probe / costmodel ──> calibrates hardware & picks TP/EP/DP config
```

## Quickstart (laptop, mock engine)

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/kiln synth-traffic
.venv/bin/kiln serve -c configs/mock.yaml &
echo 'SPEEDUP = 1.08' > /tmp/c.py
.venv/bin/kiln submit /tmp/c.py --tier fast --wait
.venv/bin/kiln status && .venv/bin/kiln leaderboard --tier fast
.venv/bin/pytest -q
```

## GPU Workflow

### 1. Install

```bash
uv venv --python 3.12
# vLLM engine
uv pip install --extra-index-url https://wheels.vllm.ai/20d8574efbca8d3e22ab1bc497c71f31ad2a7153 -e '.[gpu,dev]'
# or SGLang engine
uv pip install -e '.[sglang,dev]'

# AgentX harness in a separate venv (pinned `agentx-harness==1.0.12` conflicts with vLLM's Pillow)
uv venv --python 3.12 .venv-agentx
uv pip install --python .venv-agentx/bin/python -e '.[agentx]'
export PATH="$PWD/.venv/bin:$PWD/.venv-agentx/bin:$PATH"
```

Keep `.venv-agentx/bin` on `PATH` (or set `agentx.binary` in the slow tier config) so the server can invoke `agentx`.

### 2. Pick a config & workload

| Config | Setup |
|---|---|
| `configs/mock.yaml` | CPU dry run (no GPU) |
| `configs/llama1b_vllm.yaml` / `configs/llama1b_sglang.yaml` | 2-GPU vLLM or SGLang (`fast` + `slow`) |
| `configs/dsv41_flash_h20_1node.yaml` | 1×8 H20-3e, shared TP8 `vllm serve` (recommended single-node) |
| `configs/dsv41_flash_h20_1node_tp4.yaml` | 1×8 H20-3e, concurrent TP4+TP4 w/ Engram CPU offload (≥400 GB free RAM) |
| `configs/dsv41_flash_h20.yaml` | 2×8 H20-3e, parallel TP8 `fast` (Node 0) + TP8 `slow` (Node 1) |
| `configs/dsv41_flash_h20_sglang_pd.yaml` | SGLang PD disaggregation (`fast_prefill` + `fast_decode` + `1P1D` Mooncake + `sglang_router`) |

Both engines share the AgentX workload (`semianalysisai/cc-traces-weka-062126`, seed 42): `fast` replays offline requests exported with `kiln agentx-export`, while `slow` runs live `agentx profile` against `vllm_server` or `sglang_server`.

```bash
kiln agentx-export --tokenizer /path/to/model --max-context 262144
```

### 3. Smoke-test hot reload

```bash
python -m scripts.smoke_vllm -c configs/dsv41_flash_h20_1node.yaml --tier fast
python -m scripts.smoke_sglang -c configs/llama1b_sglang.yaml --tier fast
```

### 4. Serve & launch agents

```bash
# Single-node (serves both tiers)
kiln serve -c configs/dsv41_flash_h20_1node.yaml

# Or 2-node (Node 0 runs fast + server; Node 1 runs slow worker)
kiln serve -c configs/dsv41_flash_h20.yaml --tiers fast --host 0.0.0.0
kiln worker -c configs/dsv41_flash_h20.yaml --tier slow --server http://<NODE0_IP>:8765

# Verify readiness, run baseline, and launch parallel agents
kiln status
kiln submit candidates/baseline_forward.py --tier fast --wait
kiln agents -c configs/dsv41_flash_h20_1node.yaml
```

- Multi-node requires a shared `KILN_TOKEN` on a private network (submissions execute Python over HTTP).
- `agents.slots` mixes `agy`, `codex`, `claude`, and custom CLIs (or use `agents.command` + `models`, with `synthesizer_slot: true` or `role: synthesizer` for dedicated ensembling); preview with `kiln agents -c <config> --dry-run`. Agents coordinate via `kiln submit / status / board / leaderboard / archive / checkout / merge / micro / hw`, and run state persists under `runs/<name>/`.

## Lock-Free Micro Sandbox, Lineage & Orthogonal Ensembling

```bash
# 1. Pre-screen custom kernels without waiting on the fast/slow engine lock
kiln micro-snapshots
kiln micro forward.py --snapshot moe --ncu --promote-fast --wait

# 2. Multi-island archive (regimes M<=48 / 65..256 / M>512 & submodule islands) + lineage
kiln archive --tier fast
kiln checkout <job_id> -o forward.py
kiln diff <child_job_id>
kiln lineage <job_id>

# 3. Synthesize orthogonal winners (or asymmetric Prefill + Decode winners) into one composite forward.py
kiln merge <job_a> <job_b> -o forward.py --submit --tier fast --wait
kiln merge --prefill <prefill_job> --decode <decode_job> -o forward.py --submit --tier slow --wait
```

## Prefill-Decode (PD) Disaggregation (SGLang + Mooncake)

The four-pod example uses three DNS names. Point `kiln-control` at the Kiln server on port 18765, `kiln-prefill` at the prefill pod on ports 31401 and 8998, and `kiln-decode` at the decode pod on port 31402. Use a separate name per replica when scaling. The launch commands in the config can also use your own pod addresses.

The example selects `round_robin`, which both the native SGLang router and the built-in fallback support. The fallback also supports `random`. Install the native router for `cache_aware` or `power_of_two`.

```bash
# Launch hook-enabled Prefill and Decode servers on their respective GPU pods
kiln pd-node -c configs/dsv41_flash_h20_sglang_pd.yaml --role prefill --host 0.0.0.0
kiln pd-node -c configs/dsv41_flash_h20_sglang_pd.yaml --role decode  --host 0.0.0.0

# Search asymmetric xPyD topologies & KV transfer overhead before deployment
kiln costmodel --model deepseek-v4.1-flash -p profiles/h20_3e_hw_profile.json --disagg --cluster-gpus 16
```

## Hardware Probe & Cost Model

```bash
kiln probe -o runs/hw_profile.json --md runs/HARDWARE.md
kiln costmodel --model deepseek-v4.1-flash -p profiles/h20_3e_hw_profile.json --isl 8192 --osl 1024 --tps 40
kiln hw
```

Pre-measured profiles live in `profiles/` (`h20_3e_hw_profile.json`, `b300_hw_profile.json`).

## Candidate Contract & Results

Candidates implement `make_forward(model)` returning a replacement forward callable (or `None` when patching submodules in place) and an optional `restore(model)` for extra state; never mutate weights or buffers in place. See `candidates/baseline_forward.py`.

Archived winner `candidates/dsv41_flash_h20_tp8_forward.py` (DeepSeek-V4.1-Flash, 8× H20-3e TP8) reached `1.20x` peak on fast tier, `1.115x` on 180s slow AgentX, and `+16.7%` P90 user interactivity (`1.026x` total throughput/GPU) on official 900s InferenceX AgentX.

## Layout

| Path | Role | TKCC |
|---|---|---|
| `kiln/engine/` | Hot-reloadable vLLM (`vllm_offline`, `vllm_server`), SGLang (`sglang_offline`, `sglang_server`), CPU `mock`, baseline trace diff & roofline attribution | §4.2, Fig. 3 |
| `kiln/server/` | FastAPI job queues, remote workers, token auth, crash recovery, multi-island archive, lineage, and shared board | §4.2, Fig. 5c |
| `kiln/micro/` | Lock-free isolated kernel sandbox, cold-boot layer tensor snapshots, `ptxas` spill & `ncu` bank-conflict counters | — |
| `kiln/patches.py` | `@kiln_patch` modular patch contract, static AST target/conflict analysis, and orthogonal candidate composer | — |
| `kiln/bench/` | Traffic replay, AgentX export & harness runner, greedy-token + logprob correctness check | §3, §4.1 |
| `kiln/costmodel/` | Hardware roofline prober (`kiln probe`) and parallelism ranker (`kiln costmodel`) | §4.3, App. D |
| `kiln/agents/` | Parallel CLI agent supervisor (respawn, 429 backoff, synthesizer slot) and minimal prompt | §4.4 |
| `configs/`, `profiles/` | Ready-to-run YAML configs and measured H20-3e / B300 hardware profiles | §4.3 |
| `candidates/`, `scripts/` | Baseline contract, archived DeepSeek-V4.1-Flash winner, and vLLM/SGLang smoke tests | — |

## Limitations

- **Engines:** vLLM (checked against `20d8574`) and SGLang are supported; TRT-LLM is not yet implemented.
- **Final verification:** greedy-token + logprob matching gates search candidates; run full eval suites, perplexity, and KV-cache checks before deploying.
- **Cost model:** `kiln probe` calibrates Level 2 hardware curves; Level 3 end-to-end forward calibration is manual.
- **Ensembling:** static AST target inspection + `@kiln_patch` composition merges disjoint winners automatically (`kiln merge`), while overlapping targets still require `--allow-overlap` or manual conflict resolution.
