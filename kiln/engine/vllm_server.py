"""Persistent `vllm serve` engine, measured with the real AgentX harness (slow tier).

This matches TKCC's protocol: an end-to-end server, with the AgentX load generator, public
traces, seed and scoring left unchanged. Hot reload uses vLLM's dev endpoint
POST /collective_rpc (enabled by VLLM_SERVER_DEV_MODE=1), which calls into the same kiln
worker extension as the offline engine. String args only, so we pass the candidate path.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from kiln.bench.agentx import AgentXConfig, agentx_result_to_extra, run_agentx_profile
from kiln.bench.correctness import collect_probe_logprobs, logprob_value
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import (
    Engine,
    InvalidBenchmark,
    MeasureResult,
    build_server_replay_extra,
    read_prometheus_metrics,
    shutdown_process_group,
)
from .replay import read_completion, time_http_prefill
from .vllm_profiler import build_trace_payload, configure_torch_profiler


def _cli_args(d: dict) -> list[str]:
    import json
    out = []
    for k, v in d.items():
        if v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is False:
            out.append("--no-" + flag[2:])
        elif isinstance(v, (dict, list)):
            out += [flag, json.dumps(v)]
        else:
            out += [flag, str(v)]
    return out


class VLLMServerEngine(Engine):
    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        self.cfg, self.tier = cfg, tier
        self.port = tier.port or 8200
        self.url = f"http://127.0.0.1:{self.port}"
        self.http = httpx.Client(
            trust_env=False,
            timeout=tier.timeout_s,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        self.proc: subprocess.Popen | None = None
        self.agentx = AgentXConfig(**tier.agentx) if tier.agentx is not None else None
        self.served_name = cfg.model
        self._trace_dir: str | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        args = dict(tensor_parallel_size=self.cfg.tensor_parallel_size,
                    max_model_len=self.cfg.max_model_len)
        args.update(self.cfg.extra_args)
        args.update(self.tier.engine_overrides)
        if self.tier.trace_kernels:
            self._trace_dir = configure_torch_profiler(args, self.tier.name)
        args.update(worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt",
                    port=self.port, host="127.0.0.1")
        if self.tier.num_layers is not None:
            args["hf_overrides"] = {**args.get("hf_overrides", {}),
                                    "num_hidden_layers": self.tier.num_layers}
        venv_bin = str(Path(sys.executable).parent)
        path_env = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        vllm_bin = shutil.which("vllm", path=path_env) or "vllm"
        cmd = [vllm_bin, "serve", self.cfg.model, *_cli_args(args)]
        env = {
            **os.environ,
            "PATH": path_env,
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }
        env.pop("VLLM_TORCH_PROFILER_DIR", None)
        if self.tier.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.tier.gpus))
        log = open(Path(f"vllm_serve_{self.tier.name}.log"), "ab")
        print(f"[kiln] {self.tier.name}: {shlex.join(cmd)}")
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 3600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited ({self.proc.returncode}); see vllm_serve_{self.tier.name}.log")
            try:
                if self.http.get(f"{self.url}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("vllm serve did not become healthy in 1h")
        print(f"[kiln] {self.tier.name}: server up in {time.time() - t0:.0f}s, "
              f"info {self._rpc('kiln_info')}")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def shutdown(self) -> None:
        shutdown_process_group(self.proc, timeout=120.0)
        self.proc = None

    # ---------------------------------------------------------------- reload
    def _rpc(self, method: str, *args: str):
        r = self.http.post(f"{self.url}/collective_rpc",
                           json={"method": method, "args": list(args), "timeout": self.tier.timeout_s})
        r.raise_for_status()
        return r.json().get("results") if r.content else None

    def _reset_cache(self) -> None:
        for _ in range(30):     # fails while blocks are still held; retry
            r = self.http.post(f"{self.url}/reset_prefix_cache")
            if r.status_code == 200 and (not r.content or r.json().get("success", True)):
                return
            time.sleep(2)
        raise RuntimeError("reset_prefix_cache kept failing")

    def hot_reload(self, candidate_path: str) -> None:
        self._rpc("kiln_hot_reload", os.path.abspath(candidate_path))
        self._reset_cache()

    def teardown_candidate(self) -> None:
        self._rpc("kiln_teardown")
        self._reset_cache()

    # ---------------------------------------------------------------- measure
    def _complete_one_stats(
        self, prompt_ids: list[int], max_tokens: int, *, stream: bool = True
    ) -> tuple[int, float | None]:
        payload = {
            "model": self.served_name, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": stream,
        }
        if stream:
            payload.update(return_token_ids=True, stream_options={"include_usage": True})
        _, tokens, itl_ms = read_completion(self.http, f"{self.url}/v1/completions", payload, sglang=False)
        return tokens, itl_ms

    def _complete_one(self, prompt_ids: list[int], max_tokens: int) -> int:
        return self._complete_one_stats(prompt_ids, max_tokens, stream=False)[0]

    def _read_spec_metrics(self) -> dict[str, float]:
        return read_prometheus_metrics(
            self.http,
            self.url,
            {
                "spec_accept_length": (
                    "vllm:spec_decode_mean_acceptance_length",
                    "vllm:spec_decode_avg_acceptance_length",
                ),
                "spec_accept_rate": ("vllm:spec_decode_draft_acceptance_rate",),
            },
        )

    def _kernel_report(self, pool: concurrent.futures.ThreadPoolExecutor,
                       requests: list[Request]) -> tuple[str | None, dict | None]:
        if not self._trace_dir or not requests:
            return None, None
        try:
            self.http.post(f"{self.url}/start_profile").raise_for_status()
            try:
                list(pool.map(lambda req: self._complete_one(req.prompt_ids, 32), requests[:4]))
            finally:
                self.http.post(f"{self.url}/stop_profile").raise_for_status()
            return build_trace_payload(self._trace_dir)
        except Exception:
            return None, None

    def capture_snapshots(self, snapshot_dir: str, m_buckets: list[int]) -> dict | None:
        if not self.is_alive():
            return None
        import json as _json
        self._rpc(
            "kiln_enable_snapshot_capture",
            os.path.abspath(snapshot_dir),
            _json.dumps(list(m_buckets)),
        )
        try:
            self._complete_one([1, 2, 3, 4], 2)
        finally:
            res = self._rpc("kiln_finish_snapshot_capture")
        return res[0] if isinstance(res, list) and res else res

    def measure_requests(
        self, requests: list[Request], repeats: int, trace: bool, *, skip_prefill: bool = False
    ) -> MeasureResult:
        """Measure offline `requests` concurrently against `vllm serve` (fast tier)."""
        prefill_tps = 0.0
        ttft_s = 0.0
        workers = min(max(len(requests), 1), 32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            if not skip_prefill:
                isl = sum(len(r.prompt_ids) for r in requests)
                list(pool.map(lambda r: self._complete_one(r.prompt_ids, 1), requests))
                self._reset_cache()
                prefill_s, ttft_s = time_http_prefill(pool, requests, self._complete_one)
                prefill_tps = isl / prefill_s

            decode_runs = []
            itl_samples: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                batch_stats = list(pool.map(lambda r: self._complete_one_stats(r.prompt_ids, r.output_len), requests))
                dt = max(time.perf_counter() - t0, 1e-9)
                ntok = sum(tok for tok, _ in batch_stats)
                decode_runs.append(ntok / dt)
                itl_samples.extend(itl for _, itl in batch_stats if itl is not None)

            report, trace_analysis = self._kernel_report(pool, requests) if trace else (None, None)
        decode_tps = sum(decode_runs) / len(decode_runs) if decode_runs else 0.0
        spec_metrics = self._read_spec_metrics() if repeats > 0 else {}
        extra = build_server_replay_extra(requests, decode_tps, prefill_tps, itl_samples, spec_metrics)
        return MeasureResult(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            ttft_s=ttft_s,
            itl_samples_ms=itl_samples,
            runs_decode_tps=decode_runs,
            kernel_report=report,
            trace_analysis=trace_analysis,
            extra=extra,
        )

    def measure_more(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Run extra decode repeats and/or trace on an already-warmed server."""
        if self.agentx is None:
            return self.measure_requests(requests, repeats, trace, skip_prefill=True)
        return self.measure(requests, repeats, trace)

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Use offline `/v1/completions` replay if `agentx` is None; else run real AgentX harness."""
        if self.agentx is None:
            return self.measure_requests(requests, repeats, trace)
        tp = int(self.tier.engine_overrides.get("tensor_parallel_size", self.cfg.tensor_parallel_size))
        n_gpus = len(self.tier.gpus) or tp
        artifact_root = Path("runs") / "agentx"
        artifact_root.mkdir(parents=True, exist_ok=True)
        runs: list[float] = []
        extra = {"metric": "agentx_total_tput_per_gpu", "repeats": []}
        for _ in range(repeats):
            self._reset_cache()
            artifact_dir = tempfile.mkdtemp(prefix=f"{self.tier.name}_", dir=artifact_root)
            last = run_agentx_profile(
                self.url, self.served_name, self.cfg.tokenizer or self.cfg.model, self.agentx,
                artifact_dir, n_gpus,
                timeout_s=max(self.tier.timeout_s, (self.agentx.benchmark_duration_s or 7200) * 3))
            result = asdict(last)
            for key, value in result.items():
                if isinstance(value, float) and not math.isfinite(value):
                    result[key] = None
            extra["repeats"].append(result)
            require_valid = not self.agentx.unsafe_override
            if (require_valid and last.submission_valid is not True) or not (
                math.isfinite(last.total_tput_per_gpu) and last.total_tput_per_gpu > 0
            ):
                extra["submission_valid"] = False
                raise InvalidBenchmark(f"Invalid AgentX measurement: {last.raw_path}", extra)
            runs.append(last.total_tput_per_gpu)
        extra = agentx_result_to_extra(last, extra["repeats"], self.agentx.unsafe_override)
        return MeasureResult(
            decode_tps=sum(runs) / len(runs),           # score = InferenceX throughput / GPU
            prefill_tps=0.0, ttft_s=0.0, runs_decode_tps=runs,
            extra=extra,
        )

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        if "greedy_outputs" in self.__dict__:
            return self.greedy_outputs(prompts, max_new_tokens), None
        out_ids: list[list[int]] = []
        out_lps: list[list[dict[int, float]] | None] = []
        for p in prompts:
            r = self.http.post(f"{self.url}/v1/completions", json={
                "model": self.served_name, "prompt": p, "max_tokens": max_new_tokens,
                "temperature": 0.0, "ignore_eos": True, "return_token_ids": True, "logprobs": 1,
            })
            r.raise_for_status()
            ch = r.json()["choices"][0]
            ids = ch.get("token_ids")
            if ids is None:
                raise RuntimeError("vLLM did not return token IDs for the correctness probe")
            toks = list(ids)
            out_ids.append(toks)
            lp_obj = ch.get("logprobs") or {}
            tok_lps = lp_obj.get("token_logprobs")
            if tok_lps is None:
                out_lps.append(None)
            elif isinstance(tok_lps, list) and len(tok_lps) == len(toks):
                out_lps.append([{int(tid): logprob_value(lp)} for tid, lp in zip(toks, tok_lps)])
            else:
                out_lps.append([])
        return out_ids, collect_probe_logprobs(out_lps)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.greedy_probe(prompts, max_new_tokens)[0]
