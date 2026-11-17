"""Persistent vLLM engine (one per tier).

Fast tier: `num_layers` set -> model truncated via hf_overrides (fewer layers = faster
start, faster jobs, still real kernels and real data). Slow tier: full model.
"""
from __future__ import annotations

import os
import threading
import time

from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import Engine, MeasureResult


# Both tiers live in one process; CUDA_VISIBLE_DEVICES is process-global and is read when
# vLLM spawns its EngineCore/worker processes. Boot engines one at a time.
_BOOT_LOCK = threading.Lock()


class VLLMEngine(Engine):
    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        self.cfg, self.tier = cfg, tier
        self.llm = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with _BOOT_LOCK:
            self._start()

    def _start(self) -> None:
        if self.tier.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.tier.gpus))
        from vllm import LLM

        kwargs = dict(
            model=self.cfg.model,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            max_model_len=self.cfg.max_model_len,
            enable_prefix_caching=True,          # TKCC: reuse a fixed request set -> cached prompts
            worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt",
        )
        kwargs.update(self.cfg.extra_args)
        kwargs.update(self.tier.engine_overrides)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if self.tier.num_layers is not None:
            kwargs["hf_overrides"] = {**kwargs.get("hf_overrides", {}),
                                      "num_hidden_layers": self.tier.num_layers}
        t0 = time.time()
        self.llm = LLM(**kwargs)
        print(f"[kiln] {self.tier.name} tier engine up in {time.time() - t0:.0f}s")
        print(f"[kiln] {self.tier.name} info: {self.llm.collective_rpc('kiln_info')[0]}")

    def hot_reload(self, candidate_path: str) -> None:
        secs = self.llm.collective_rpc("kiln_hot_reload", args=(os.path.abspath(candidate_path),),
                                       timeout=self.tier.timeout_s)
        # Cached KV was written by the previous forward; never let a candidate reuse it.
        self.llm.reset_prefix_cache()
        print(f"[kiln] {self.tier.name}: hot reload {max(secs):.1f}s")

    def teardown_candidate(self) -> None:
        self.llm.collective_rpc("kiln_teardown", timeout=self.tier.timeout_s)
        self.llm.reset_prefix_cache()

    def shutdown(self) -> None:
        del self.llm
        self.llm = None

    # ---------------------------------------------------------------- measuring
    def _prompts(self, reqs: list[Request]):
        from vllm.inputs import TokensPrompt
        return [TokensPrompt(prompt_token_ids=r.prompt_ids) for r in reqs]

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Prefill and decode measured separately (TKCC: mixing them adds ~6% noise)."""
        from vllm import SamplingParams

        prompts = self._prompts(requests)
        isl = sum(len(r.prompt_ids) for r in requests)
        warm = SamplingParams(max_tokens=1, temperature=0.0)
        self.llm.generate(prompts, warm, use_tqdm=False)         # warm the prefix cache

        # Prefill: max_tokens=1 on the cached-or-not prompt set.
        self.llm.reset_prefix_cache()
        t0 = time.perf_counter()
        self.llm.generate(prompts, warm, use_tqdm=False)
        prefill_s = time.perf_counter() - t0

        decode_runs = []
        for _ in range(repeats):
            # Prompts are cached now, so wall time is dominated by decode.
            sps = [SamplingParams(max_tokens=r.output_len, ignore_eos=True, temperature=0.0)
                   for r in requests]
            t0 = time.perf_counter()
            outs = self.llm.generate(prompts, sps, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            decode_runs.append(ntok / dt)

        report = self._kernel_report(prompts) if trace else None
        return MeasureResult(
            decode_tps=sum(decode_runs) / len(decode_runs),
            prefill_tps=isl / prefill_s,
            ttft_s=prefill_s / max(len(requests), 1),
            runs_decode_tps=decode_runs,
            kernel_report=report,
        )

    def _kernel_report(self, prompts) -> str | None:
        """Top kernels by CUDA time (fast tier only). Uses vLLM's torch-profiler hooks
        if VLLM_TORCH_PROFILER_DIR is set; otherwise returns None."""
        if not os.environ.get("VLLM_TORCH_PROFILER_DIR"):
            return None
        from vllm import SamplingParams
        self.llm.start_profile()
        self.llm.generate(prompts[:4], SamplingParams(max_tokens=32, ignore_eos=True), use_tqdm=False)
        self.llm.stop_profile()
        return f"trace written to {os.environ['VLLM_TORCH_PROFILER_DIR']}"

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]
