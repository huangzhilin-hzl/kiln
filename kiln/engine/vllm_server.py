"""Persistent `vllm serve` engine, measured with the real AgentX harness (slow tier).

This matches TKCC's protocol: an end-to-end server, with the AgentX load generator, public
traces, seed and scoring left unchanged. Hot reload uses vLLM's dev endpoint
POST /collective_rpc (enabled by VLLM_SERVER_DEV_MODE=1), which calls into the same kiln
worker extension as the offline engine. String args only, so we pass the candidate path.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from kiln.bench.agentx import AgentXConfig, run_agentx_profile
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import Engine, MeasureResult


def _cli_args(d: dict) -> list[str]:
    import json
    out = []
    for k, v in d.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
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
        self.http = httpx.Client(trust_env=False, timeout=tier.timeout_s)
        self.proc: subprocess.Popen | None = None
        self.agentx = AgentXConfig(**(tier.agentx or {}))
        self.served_name = cfg.model
        self._runs = 0

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        args = dict(tensor_parallel_size=self.cfg.tensor_parallel_size,
                    max_model_len=self.cfg.max_model_len)
        args.update(self.cfg.extra_args)
        args.update(self.tier.engine_overrides)
        args.update(worker_extension_cls="kiln.engine.vllm_ext.HotReloadExt",
                    port=self.port, host="127.0.0.1")
        if self.tier.num_layers is not None:
            args["hf_overrides"] = {**args.get("hf_overrides", {}),
                                    "num_hidden_layers": self.tier.num_layers}
        venv_bin = str(Path(sys.executable).resolve().parent)
        path_env = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        vllm_bin = shutil.which("vllm", path=path_env) or "vllm"
        cmd = [vllm_bin, "serve", self.cfg.model, *_cli_args(args)]
        env = {**os.environ, "PATH": path_env, "VLLM_SERVER_DEV_MODE": "1"}
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
    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """`requests` is ignored here: AgentX brings its own traces."""
        tp = int(self.tier.engine_overrides.get("tensor_parallel_size", self.cfg.tensor_parallel_size))
        n_gpus = len(self.tier.gpus) or tp
        runs = []
        last = None
        for _ in range(repeats):
            self._runs += 1
            self._reset_cache()
            last = run_agentx_profile(
                self.url, self.served_name, self.cfg.tokenizer or self.cfg.model, self.agentx,
                Path("runs") / "agentx" / f"{self.tier.name}_{self._runs:05d}", n_gpus,
                timeout_s=(self.agentx.benchmark_duration_s or 7200) * 3)
            runs.append(last.total_tput_per_gpu)
        return MeasureResult(
            decode_tps=sum(runs) / len(runs),           # score = InferenceX throughput / GPU
            prefill_tps=0.0, ttft_s=0.0, runs_decode_tps=runs,
            extra={"metric": "agentx_total_tput_per_gpu", "output_tput": last.output_tput,
                   "itl_p50_ms": last.itl_p50_ms, "itl_p90_ms": last.itl_p90_ms,
                   "interactivity_p90": last.interactivity_p90,
                   "submission_valid": last.submission_valid, "export": last.raw_path},
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
            out.append(list(ids) if ids is not None else [hash(ch["text"])])
        return out
