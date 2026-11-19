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

from kiln.bench.agentx import AgentXConfig, run_agentx_profile
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import Engine, InvalidBenchmark, MeasureResult
from .vllm_profiler import configure_torch_profiler, summarize_trace


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

    def shutdown(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
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
    def _complete_one(self, prompt_ids: list[int], max_tokens: int) -> int:
        r = self.http.post(f"{self.url}/v1/completions", json={
            "model": self.served_name, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True,
        })
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage") or {}
        if "completion_tokens" in usage:
            return int(usage["completion_tokens"])
        choices = body.get("choices") or [{}]
        return len(choices[0].get("token_ids") or ()) or max_tokens

    def _kernel_report(self, pool: concurrent.futures.ThreadPoolExecutor,
                       requests: list[Request]) -> str | None:
        if not self._trace_dir or not requests:
            return None
        try:
            self.http.post(f"{self.url}/start_profile").raise_for_status()
            try:
                list(pool.map(lambda req: self._complete_one(req.prompt_ids, 32), requests[:4]))
            finally:
                self.http.post(f"{self.url}/stop_profile").raise_for_status()
            return summarize_trace(self._trace_dir)
        except Exception:
            return None

    def measure_requests(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Measure offline `requests` concurrently against `vllm serve` (fast tier)."""
        isl = sum(len(r.prompt_ids) for r in requests)
        workers = min(max(len(requests), 1), 32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda r: self._complete_one(r.prompt_ids, 1), requests))
            self._reset_cache()
            t0 = time.perf_counter()
            list(pool.map(lambda r: self._complete_one(r.prompt_ids, 1), requests))
            prefill_s = max(time.perf_counter() - t0, 1e-9)

            decode_runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ntok = sum(pool.map(lambda r: self._complete_one(r.prompt_ids, r.output_len), requests))
                dt = max(time.perf_counter() - t0, 1e-9)
                decode_runs.append(ntok / dt)

            report = self._kernel_report(pool, requests) if trace else None
        return MeasureResult(
            decode_tps=sum(decode_runs) / len(decode_runs),
            prefill_tps=isl / prefill_s,
            ttft_s=prefill_s / max(len(requests), 1),
            runs_decode_tps=decode_runs,
            kernel_report=report,
        )

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
                timeout_s=(self.agentx.benchmark_duration_s or 7200) * 3)
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
        extra.update(output_tput=result["output_tput"], itl_p50_ms=result["itl_p50_ms"],
                     itl_p90_ms=result["itl_p90_ms"],
                     submission_valid=(last.submission_valid is True),
                     unsafe_override=self.agentx.unsafe_override,
                     export=last.raw_path)
        itl = result["itl_p90_ms"]
        extra["interactivity_p90"] = 1000.0 / itl if itl and itl > 0 else None
        return MeasureResult(
            decode_tps=sum(runs) / len(runs),           # score = InferenceX throughput / GPU
            prefill_tps=0.0, ttft_s=0.0, runs_decode_tps=runs,
            extra=extra,
        )

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        out = []
        for p in prompts:
            r = self.http.post(f"{self.url}/v1/completions", json={
                "model": self.served_name, "prompt": p, "max_tokens": max_new_tokens,
                "temperature": 0.0, "ignore_eos": True, "return_token_ids": True})
            r.raise_for_status()
            ch = r.json()["choices"][0]
            ids = ch.get("token_ids")
            if ids is None:
                raise RuntimeError("vLLM did not return token IDs for the correctness probe")
            out.append(list(ids))
        return out
