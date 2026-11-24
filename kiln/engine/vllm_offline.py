"""Persistent offline vLLM engine (`vllm.LLM`, one per tier).

Fast tier: `num_layers` set -> model truncated via hf_overrides (fewer layers = faster
start, faster jobs, still real kernels and real data). Slow tier: full model.
"""
from __future__ import annotations

import gc
import os
import threading
import time

from kiln.bench.correctness import collect_probe_logprobs, logprob_value
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import Engine, MeasureResult, bucket_tps_for_requests
from .prebuild import ensure_extension_cache_env
from .replay import mean_ttft
from .vllm_profiler import build_trace_payload, configure_torch_profiler


# Both tiers may live in one process; CUDA_VISIBLE_DEVICES is process-global and is read when
# vLLM spawns its EngineCore/worker processes. Boot engines one at a time.
_BOOT_LOCK = threading.Lock()


class VLLMOfflineEngine(Engine):
    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        self.cfg, self.tier = cfg, tier
        self.llm = None
        self._trace_dir: str | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with _BOOT_LOCK:
            self._start()

    def _start(self) -> None:
        if self.tier.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.tier.gpus))
        os.environ.update(ensure_extension_cache_env(os.environ))
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
        # A candidate can patch child forwards without changing vLLM's compile cache key.
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        from vllm import LLM

        kwargs = dict(
            model=self.cfg.model,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            max_model_len=self.cfg.max_model_len,
            disable_log_stats=False,             # expose per-request first_token_latency
            enable_prefix_caching=True,          # TKCC: reuse a fixed request set -> cached prompts
            worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt",
        )
        kwargs.update(self.cfg.extra_args)
        kwargs.update(self.tier.engine_overrides)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if self.tier.trace_kernels:
            self._trace_dir = configure_torch_profiler(kwargs, self.tier.name)
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
        if self.llm is not None:
            try:
                core = getattr(getattr(self.llm, "llm_engine", None), "engine_core", None)
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
            except Exception:
                pass
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ---------------------------------------------------------------- measuring
    def _prompts(self, reqs: list[Request]):
        from vllm.inputs import TokensPrompt
        return [TokensPrompt(prompt_token_ids=r.prompt_ids) for r in reqs]

    def measure(
        self, requests: list[Request], repeats: int, trace: bool, *, skip_prefill: bool = False
    ) -> MeasureResult:
        """Prefill and decode measured separately (TKCC: mixing them adds ~6% noise)."""
        from vllm import SamplingParams

        prompts = self._prompts(requests)
        prefill_tps = 0.0
        ttft_s = 0.0
        if not skip_prefill:
            isl = sum(len(r.prompt_ids) for r in requests)
            warm = SamplingParams(max_tokens=1, temperature=0.0)
            self.llm.generate(prompts, warm, use_tqdm=False)         # warm the prefix cache

            # Prefill: max_tokens=1 on the cached-or-not prompt set.
            self.llm.reset_prefix_cache()
            t0 = time.perf_counter()
            outputs = self.llm.generate(prompts, warm, use_tqdm=False)
            prefill_s = max(time.perf_counter() - t0, 1e-9)
            prefill_tps = isl / prefill_s
            ttft_s = mean_ttft(getattr(getattr(out, "metrics", None), "first_token_latency", None)
                               for out in outputs)

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

        report, trace_analysis = self._kernel_report(prompts) if trace else (None, None)
        decode_tps = sum(decode_runs) / len(decode_runs) if decode_runs else 0.0
        buckets = bucket_tps_for_requests(requests, decode_tps, prefill_tps)
        return MeasureResult(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            ttft_s=ttft_s,
            runs_decode_tps=decode_runs,
            kernel_report=report,
            trace_analysis=trace_analysis,
            extra=({"bucket_tps": buckets} if buckets else {}),
        )

    def measure_more(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Run extra decode repeats and/or trace without repeating warmup or prefill."""
        return self.measure(requests, repeats, trace, skip_prefill=True)

    def _kernel_report(self, prompts) -> tuple[str | None, dict | None]:
        """Top kernels by CUDA time (fast tier only)."""
        tdir = self._trace_dir
        if not tdir:
            return None, None
        from vllm import SamplingParams
        self.llm.start_profile()
        try:
            self.llm.generate(prompts[:4], SamplingParams(max_tokens=32, ignore_eos=True), use_tqdm=False)
        finally:
            self.llm.stop_profile()
        return build_trace_payload(tdir)

    def capture_snapshots(self, snapshot_dir: str, m_buckets: list[int]) -> dict | None:
        if self.llm is None:
            return None
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.llm.collective_rpc(
            "kiln_enable_snapshot_capture",
            args=(os.path.abspath(snapshot_dir), list(m_buckets)),
            timeout=self.tier.timeout_s,
        )
        try:
            self.llm.generate(
                [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
                SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
                use_tqdm=False,
            )
        finally:
            res = self.llm.collective_rpc("kiln_finish_snapshot_capture", timeout=self.tier.timeout_s)
        return res[0] if res else None

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        if "greedy_outputs" in self.__dict__:
            return self.greedy_outputs(prompts, max_new_tokens), None
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        try:
            sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True, logprobs=1)
        except TypeError:
            sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
        self.llm.reset_prefix_cache()
        out_ids: list[list[int]] = []
        out_lps: list[list[dict[int, float]] | None] = []
        for p in prompts:
            o = self.llm.generate([TokensPrompt(prompt_token_ids=p)], sp, use_tqdm=False)[0]
            comp = o.outputs[0]
            out_ids.append(list(comp.token_ids))
            raw_lps = getattr(comp, "logprobs", None)
            if raw_lps is None:
                out_lps.append(None)
                continue
            steps = []
            if isinstance(raw_lps, list):
                for entry in raw_lps:
                    step = {}
                    if isinstance(entry, dict):
                        for tid, val in entry.items():
                            try:
                                step[int(tid)] = logprob_value(getattr(val, "logprob", val))
                            except (TypeError, ValueError, OverflowError):
                                continue
                    steps.append(step)
            out_lps.append(steps)
        return out_ids, collect_probe_logprobs(out_lps)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.greedy_probe(prompts, max_new_tokens)[0]
