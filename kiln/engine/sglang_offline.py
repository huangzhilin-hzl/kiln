"""Persistent offline SGLang engine (`sglang.Engine`, one per tier).

Fast tier: `num_layers` set -> model truncated via `json_model_override_args`
(`num_hidden_layers`). Slow tier: full model.
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
from pathlib import Path

from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import Engine, MeasureResult
from .sglang_ext import (
    bootstrap_env,
    encode_rpc,
    extract_output_ids,
    install_sglang_hooks,
    merge_json_model_override_args,
    parse_rpc_message,
)
from .vllm_profiler import summarize_trace


# Both tiers may live in one process; CUDA_VISIBLE_DEVICES is process-global and is read when
# SGLang spawns its Scheduler/worker processes. Boot engines one at a time.
_BOOT_LOCK = threading.Lock()


def _parse_update_result(res) -> tuple[bool, str]:
    if isinstance(res, tuple):
        success = bool(res[0])
        message = str(res[1]) if len(res) > 1 and res[1] is not None else ""
        return success, message
    if isinstance(res, dict):
        return bool(res.get("success", False)), str(res.get("message") or "")
    if hasattr(res, "success"):
        return bool(res.success), str(getattr(res, "message", "") or "")
    raise RuntimeError(f"Unexpected SGLang update_weights_from_disk response: {res!r}")


def _completion_tokens(out: dict, fallback: int) -> int:
    meta = out.get("meta_info") or {}
    if meta.get("completion_tokens") is not None:
        return int(meta["completion_tokens"])
    ids = extract_output_ids(out)
    if ids is not None:
        return len(ids)
    return fallback


class SGLangOfflineEngine(Engine):
    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        self.cfg, self.tier = cfg, tier
        self.engine = None
        self._trace_dir: str | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with _BOOT_LOCK:
            self._start()

    def _start(self) -> None:
        if self.tier.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.tier.gpus))
        os.environ.update(bootstrap_env(os.environ))
        install_sglang_hooks()

        kwargs: dict = dict(
            model_path=self.cfg.model,
            tp_size=self.cfg.tensor_parallel_size,
            context_length=self.cfg.max_model_len,
            disable_radix_cache=False,
        )
        if self.cfg.tokenizer:
            kwargs["tokenizer_path"] = self.cfg.tokenizer
        kwargs.update(self.cfg.extra_args)
        kwargs.update(self.tier.engine_overrides)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if self.tier.num_layers is not None:
            kwargs["json_model_override_args"] = merge_json_model_override_args(
                kwargs.get("json_model_override_args"),
                {"num_hidden_layers": self.tier.num_layers},
            )
        elif isinstance(kwargs.get("json_model_override_args"), dict):
            kwargs["json_model_override_args"] = json.dumps(kwargs["json_model_override_args"])

        if self.tier.trace_kernels:
            directory = (
                os.environ.get("SGLANG_TORCH_PROFILER_DIR")
                or Path("runs/traces") / self.tier.name
            )
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._trace_dir = str(directory)
            os.environ["SGLANG_TORCH_PROFILER_DIR"] = self._trace_dir

        import sglang as sgl

        t0 = time.time()
        self.engine = sgl.Engine(**kwargs)
        print(f"[kiln] {self.tier.name} tier sglang engine up in {time.time() - t0:.0f}s")
        print(f"[kiln] {self.tier.name} info: {self._rpc('kiln_info')}")

    def _rpc(self, method: str, *args):
        res = self.engine.update_weights_from_disk(model_path=encode_rpc(method, *args))
        success, message = _parse_update_result(res)
        if not success:
            raise RuntimeError(f"SGLang {method} failed: {message}")
        return parse_rpc_message(message)

    def _flush_cache(self) -> None:
        flush = getattr(self.engine, "flush_cache", None)
        if not callable(flush):
            return
        for _ in range(30):
            try:
                ok = flush()
                if ok is None or ok is True:
                    return
                if isinstance(ok, dict) and ok.get("success", True):
                    return
                if hasattr(ok, "success") and bool(ok.success):
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("SGLang flush_cache kept failing")

    def hot_reload(self, candidate_path: str) -> None:
        secs = self._rpc("kiln_hot_reload", os.path.abspath(candidate_path))
        # Cached KV was written by the previous forward; never let a candidate reuse it.
        self._flush_cache()
        if isinstance(secs, (int, float)):
            print(f"[kiln] {self.tier.name}: hot reload {secs:.1f}s")

    def teardown_candidate(self) -> None:
        self._rpc("kiln_teardown")
        self._flush_cache()

    def shutdown(self) -> None:
        if self.engine is not None:
            try:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
            except Exception:
                pass
            del self.engine
            self.engine = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ---------------------------------------------------------------- measuring
    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Prefill and decode measured separately (TKCC: mixing them adds ~6% noise)."""
        prompts = [r.prompt_ids for r in requests]
        isl = sum(len(r.prompt_ids) for r in requests)
        warm = {"max_new_tokens": 1, "temperature": 0.0}
        self.engine.generate(input_ids=prompts, sampling_params=warm)      # warm radix cache

        self._flush_cache()
        t0 = time.perf_counter()
        self.engine.generate(input_ids=prompts, sampling_params=warm)
        prefill_s = max(time.perf_counter() - t0, 1e-9)

        decode_runs = []
        for _ in range(repeats):
            sps = [
                {"max_new_tokens": r.output_len, "ignore_eos": True, "temperature": 0.0}
                for r in requests
            ]
            t0 = time.perf_counter()
            outs = self.engine.generate(input_ids=prompts, sampling_params=sps)
            dt = max(time.perf_counter() - t0, 1e-9)
            outs_list = [outs] if isinstance(outs, dict) else list(outs)
            ntok = sum(_completion_tokens(o, r.output_len) for o, r in zip(outs_list, requests))
            decode_runs.append(ntok / dt)

        report = self._kernel_report(prompts) if trace else None
        return MeasureResult(
            decode_tps=sum(decode_runs) / len(decode_runs),
            prefill_tps=isl / prefill_s,
            ttft_s=prefill_s / max(len(requests), 1),
            runs_decode_tps=decode_runs,
            kernel_report=report,
        )

    def _kernel_report(self, prompts: list[list[int]]) -> str | None:
        tdir = self._trace_dir
        if not tdir or not prompts:
            return None
        try:
            try:
                self.engine.start_profile(output_dir=tdir)
            except TypeError:
                self.engine.start_profile()
            try:
                self.engine.generate(
                    input_ids=prompts[:4],
                    sampling_params={"max_new_tokens": 32, "ignore_eos": True, "temperature": 0.0},
                )
            finally:
                self.engine.stop_profile()
            return summarize_trace(tdir)
        except Exception:
            return None

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        sp = {"max_new_tokens": max_new_tokens, "temperature": 0.0, "ignore_eos": True}
        self._flush_cache()
        out = []
        for p in prompts:
            res = self.engine.generate(input_ids=p, sampling_params=sp)
            ids = extract_output_ids(res)
            if ids is None:
                res = self.engine.generate(input_ids=p, sampling_params=sp, return_logprob=True)
                ids = extract_output_ids(res)
            if ids is None:
                raise RuntimeError("SGLang did not return output_ids for the correctness probe")
            out.append(ids)
        return out
