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
| `kiln/engine/vllm_engine.py` | Persistent engine for one tier. The fast tier truncates layers with `hf_overrides`. Prefill and decode are measured separately. | §4.2, Table 1 |
| `kiln/engine/mock.py` | CPU mock engine, so the whole loop runs on a laptop | — |
| `kiln/server/jobs.py` | One queue per tier, a worker thread per engine, auto-restart on crash, leaderboard | §4.2 |
| `kiln/server/board.py` | Shared message board for agents | Fig 5c |
| `kiln/bench/traffic.py` | Real-traffic replay (JSONL) | §3 |
| `kiln/bench/correctness.py` | Greedy-token match against baseline | §4.1 |
| `kiln/costmodel/model.py` | Roofline + comm cost model, search space, pruning, TPM ranking | §4.3, App. D |
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

## Real run (GPU box)

```bash
uv pip install -e '.[gpu,dev]'
# 1. Record real traffic to traffic/sample.jsonl: {"prompt_ids": [...], "output_len": N}
# 2. Pick a parallel config
kiln costmodel --model qwen3-30b-a3b --chip B200 --isl 8192 --osl 1024 --tps 40
# 3. Smoke-test hot reload on one tier (boot, reload, teardown, broken-candidate rejection, noise)
python scripts/smoke_vllm.py -c configs/dsv41_flash_h20.yaml --tier fast
# 4. Start the persistent engines (the slow startup happens once)
kiln serve -c configs/dsv41_flash_h20.yaml
kiln submit candidates/baseline_forward.py --tier fast --wait     # speedup ≈ 1.0
# 5. Release the agents
kiln agents -c configs/dsv41_flash_h20.yaml
```

Agents get `KILN_AGENT` / `KILN_SERVER` env vars and use `kiln submit / status / board / leaderboard`.
Run artifacts go to `runs/<name>/` (`jobs/<id>/candidate.py` and `result.json`, `board.jsonl`, `agents/`).

## Known gaps and TODOs

- **vLLM internals:** hot reload reuses vLLM's elastic-EP recipe for "release graphs, reset
  torch.compile, recapture" (checked against vLLM main @ 20d8574, 2026-11-16). On a GPU box, run
  `python scripts/smoke_vllm.py -c <config> --tier fast` before anything else.
- **Noise:** tune `fast.num_layers` and `repeats` until `variance_pct` is below the gains you're
  looking for (TKCC got ~0.1%).
- **Final verifier:** before you deploy, add perplexity / eval-suite and KV-cache contamination checks.
- **Cost model:** fill in measured `flops_eff` / `bw_eff` / `link_latency` per chip (levels 2–3).
- Other engines (SGLang, TRT-LLM) go behind the same `Engine` interface.
- **Ensembling:** merging several winning candidates automatically is not implemented yet.
