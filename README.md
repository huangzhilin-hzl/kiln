# kiln 🔥

**Continual agentic compilation for LLM inference.** It's a do-it-yourself version of the design in
[Together Compiler (TKCC)](https://cdn.prod.website-files.com/69654e88dce9154b5f1206dd/6aeb3e27b0a7e915c3091fde_Together-AI_Together-Compiler.pdf).

Parallel coding agents rewrite a model's forward pass. Each candidate is **hot-reloaded into a
persistent inference engine**, checked for correctness, and measured on **real traffic** in two
tiers. Winners get promoted. When traffic shifts, you run the loop again.

```
             ┌──────────── kiln agents (N in parallel) ─────────────┐
             │ agent_1   agent_2   agent_3 ...   ⇄  shared board    │
             └──────┬──────────────────────────────────────────────┘
                    │ kiln submit forward.py --tier fast|slow
             ┌──────▼──────────── kiln serve (job server) ──────────┐
             │  fast tier queue           slow tier queue           │
             │  ┌────────────────┐        ┌────────────────┐        │
             │  │ persistent eng │        │ persistent eng │        │
             │  │ N layers + trace│       │ all layers     │        │
             │  └────────────────┘        └────────────────┘        │
             │  hot_reload → correctness → measure → teardown       │
             └──────────────────────────────────────────────────────┘
   kiln costmodel ──> chooses the TP/EP/DP config the engines start with
```

## Layout

| Path | What it is | TKCC section |
|---|---|---|
| `kiln/engine/vllm_ext.py` | vLLM worker extension: `kiln_hot_reload` / `kiln_teardown` (swap `forward`, re-capture CUDA graphs) | §4.2, Fig 3 |
| `kiln/engine/vllm_offline.py` | Persistent offline `vllm.LLM` engine. Fast tier can truncate layers with `hf_overrides`; prefill and decode are measured separately. | §4.2, Table 1 |
| `kiln/engine/vllm_server.py` | Persistent `vllm serve` HTTP engine measured with AgentX (or `/v1/completions` replay when shared). | §4.2 |
| `kiln/engine/mock.py` | CPU mock engine, so the whole loop runs on a laptop | — |
| `kiln/server/jobs.py` | One queue per tier, a worker thread per engine, auto-restart on crash, leaderboard | §4.2 |
| `kiln/server/board.py` | Shared message board for agents | Fig 5c |
| `kiln/bench/traffic.py` | Real-traffic replay (JSONL) | §3 |
| `kiln/bench/correctness.py` | Greedy-token match against baseline | §4.1 |
| `kiln/costmodel/model.py` | Roofline + comm cost model, search space, pruning, TPM ranking | §4.3, App. D |
| `kiln/costmodel/probe.py` | Hardware & roofline microbenchmarks (HBM/L2/PCIe, BF16/FP8/FP4 GEMM, MoE, launch, NCCL) | §4.3 (Level 2) |
| `profiles/b300_hw_profile.json` | Measured 8× NVIDIA B300 SXM6 AC (`sm_103`) hardware & roofline profile | §4.3 (Level 2) |
| `kiln/agents/` | Agent launcher + the deliberately minimal prompt | §4.4 |
| `candidates/baseline_forward.py` | The candidate contract: `make_forward(model)` | — |

## Quickstart (laptop, mock engine)

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/kiln synth-traffic                       # placeholder traffic
.venv/bin/kiln serve -c configs/mock.yaml &        # terminal 1
echo 'SPEEDUP = 1.08' > /tmp/c.py
.venv/bin/kiln submit /tmp/c.py --tier fast --wait
.venv/bin/kiln status; .venv/bin/kiln leaderboard --tier fast
.venv/bin/pytest -q
```

## Workload: AgentX (SemiAnalysis InferenceX)

kiln uses the same workload as the TKCC paper: AgentX's public agentic-coding traces
(`semianalysisai/cc-traces-weka-062126`, 393 sessions with subagents, seed 42).

| Tier | How AgentX is used | Fidelity |
|---|---|---|
| fast (`engine_kind: vllm_offline`) | `kiln agentx-export` builds offline requests from the same traces. Request lengths and hash-block prefix sharing are exact; block content is real Python code (AgentX uses synthetic coding text); timing and subagent concurrency are not replayed. | Approximate, quick |
| slow (`engine_kind: vllm_server`) | `vllm serve` (with `VLLM_SERVER_DEV_MODE=1`) is measured by the real `agentx profile` harness. Hot reload goes through `/collective_rpc`. Score = InferenceX throughput/GPU; P90 interactivity is recorded in `extra`. | Exact protocol |

```bash
uv pip install -e '.[agentx]'          # agentx-harness==1.0.12, the version pinned in the paper
kiln agentx-export --tokenizer /path/to/DeepSeek-V4.1-Flash --traces 32 --turns 4
```

## Real run (2×8 or 1×8 H20-3e GPUs + DeepSeek-V4.1-Flash + Codex)

First install dependencies, set `engine.model` in your chosen config, export offline AgentX traffic, and smoke-test hot reload:

```bash
uv venv --python 3.12
uv pip install --extra-index-url https://wheels.vllm.ai/20d8574efbca8d3e22ab1bc497c71f31ad2a7153 -e '.[gpu,dev]'
uv venv --python 3.12 .venv-agentx
uv pip install --python .venv-agentx/bin/python -e '.[agentx]'
export PATH="$PWD/.venv/bin:$PWD/.venv-agentx/bin:$PATH"
# 1. Export offline AgentX traffic (Node 0)
.venv-agentx/bin/kiln agentx-export --tokenizer /path/to/DeepSeek-V4.1-Flash --max-context 262144
# 2. (Optional) Probe node roofline & check parallelism tradeoffs in the calibrated cost model
kiln probe -o runs/hw_profile.json --md runs/HARDWARE.md
kiln costmodel --model deepseek-v4.1-flash --profile runs/hw_profile.json --isl 8192 --osl 1024 --tps 40
# 3. Smoke-test hot reload on one tier first (boot, reload, teardown, broken-candidate rejection, noise)
.venv/bin/python -m scripts.smoke_vllm -c configs/dsv41_flash_h20.yaml --tier fast
```

The pinned vLLM and AgentX releases require different Pillow versions. The slow worker runs `agentx` from the separate environment. Keep that executable on `PATH` or set `agentx.binary` to its absolute path in the slow tier.

### Option A: 2×8 H20-3e (two nodes, parallel TP8 fast + TP8 slow)

Set the same `KILN_TOKEN` in the coordinator, worker, and client environments. Run this setup on a private network since submissions execute Python and the examples use HTTP. Update the coordinator and worker together when changing the worker protocol.

```bash
# Node 0 (8× H20-3e, TP8): job server + persistent fast-tier engine
kiln serve -c configs/dsv41_flash_h20.yaml --tiers fast --host 0.0.0.0
# Node 1 (8× H20-3e, TP8): persistent slow-tier engine (`vllm serve` + AgentX)
kiln worker -c configs/dsv41_flash_h20.yaml --tier slow --server http://<NODE0_IP>:8765
```

### Option B: 1×8 H20-3e (single node)
- **Shared TP8 (`configs/dsv41_flash_h20_1node.yaml`, recommended):** keeps all 510 GB + Engram in HBM3e on one TP8 `vllm serve` instance (`fast` and `slow` serialize via lock):
  ```bash
  python scripts/smoke_vllm.py -c configs/dsv41_flash_h20_1node.yaml --tier fast
  kiln serve -c configs/dsv41_flash_h20_1node.yaml
  ```
- **Concurrent TP4 + TP4 with Engram CPU offload (`configs/dsv41_flash_h20_1node_tp4.yaml`):** runs `fast` (GPUs 0–3) and `slow` (GPUs 4–7) simultaneously if the host has >= 400 GB free RAM:
  ```bash
  kiln serve -c configs/dsv41_flash_h20_1node_tp4.yaml
  ```

### Verify & launch Codex agents
```bash
kiln status                                                      # wait until both tiers show ready=true
kiln submit candidates/baseline_forward.py --tier fast --wait    # speedup ≈ 1.0
kiln agents -c configs/dsv41_flash_h20.yaml                      # or the 1-node yaml you used above
```

Agents get `KILN_AGENT` / `KILN_SERVER` env vars and use `kiln submit / status / board / leaderboard`.
Run artifacts go to `runs/<name>/` (`jobs/<id>/candidate.py` and `result.json`, `board.jsonl`, `agents/`).
The coordinator restores jobs from these records on restart. Expired worker claims are requeued.
AgentX jobs require a valid export for every repeat to appear on the leaderboard.

## Known gaps and TODOs

- **vLLM internals:** hot reload reuses vLLM's elastic-EP recipe for "release graphs, reset
  torch.compile, recapture" (checked against vLLM main @ 20d8574, 2026-11-16). On a GPU box, run
  `python scripts/smoke_vllm.py -c <config> --tier fast` before anything else.
- **Noise:** tune `fast.num_layers` and `repeats` until `variance_pct` is below the gains you're
  looking for (TKCC got ~0.1%).
- **Final verifier:** before you deploy, add perplexity / eval-suite and KV-cache contamination checks.
- **Cost model:** `kiln probe` calibrates Level 2 hardware curves (GEMM/MoE/HBM/NCCL/launch); Level 3 end-to-end forward-pass calibration is still manual.
- Other engines (SGLang, TRT-LLM) go behind the same `Engine` interface.
- **Ensembling:** merging several winning candidates automatically is not implemented yet.
