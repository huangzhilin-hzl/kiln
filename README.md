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
- `agents.command` supports any stdin (`codex`, `claude`) or `{prompt}` (`agy`) CLI; agents coordinate via `kiln submit / status / board / leaderboard / hw`, and run state persists under `runs/<name>/` across restarts.

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
| `kiln/engine/` | Hot-reloadable vLLM (`vllm_offline`, `vllm_server`), SGLang (`sglang_offline`, `sglang_server`), and CPU `mock` engines | §4.2, Fig. 3 |
| `kiln/server/` | FastAPI job queues, remote workers, token auth, crash recovery, leaderboard, and shared board | §4.2, Fig. 5c |
| `kiln/bench/` | Traffic replay, AgentX export & harness runner, greedy-token correctness check | §3, §4.1 |
| `kiln/costmodel/` | Hardware roofline prober (`kiln probe`) and parallelism ranker (`kiln costmodel`) | §4.3, App. D |
| `kiln/agents/` | Parallel CLI agent supervisor (respawn, 429 backoff) and minimal prompt | §4.4 |
| `configs/`, `profiles/` | Ready-to-run YAML configs and measured H20-3e / B300 hardware profiles | §4.3 |
| `candidates/`, `scripts/` | Baseline contract, archived DeepSeek-V4.1-Flash winner, and vLLM/SGLang smoke tests | — |

## Limitations

- **Engines:** vLLM (checked against `20d8574`) and SGLang are supported; TRT-LLM is not yet implemented.
- **Final verification:** greedy-token matching gates search candidates; run full eval suites, perplexity, and KV-cache checks before deploying.
- **Cost model:** `kiln probe` calibrates Level 2 hardware curves; Level 3 end-to-end forward calibration is manual.
- **Ensembling:** automatic merging of multiple winning candidates is not implemented yet.
