"""Persistent `sglang.launch_server` HTTP engine (slow tier w/ AgentX, or shared fast+slow).

Matches TKCC's protocol: an end-to-end OpenAI-compatible server measured with the real AgentX
harness (or concurrent `/generate` replay when `agentx` is None / shared via `share_engine`).
Hot reload goes through SGLang's standard `POST /update_weights_from_disk` control-plane route
intercepted by `kiln.engine.sglang_ext`, which broadcasts across all TP/PP/EP/DP ranks.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from kiln.bench.agentx import AgentXConfig, agentx_result_to_extra, run_agentx_profile
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, TierConfig

from .base import (
    Engine,
    InvalidBenchmark,
    MeasureResult,
    bucket_tps_for_requests,
    build_server_replay_extra,
    read_prometheus_metrics,
    shutdown_process_group,
)
from .sglang_ext import (
    bootstrap_env,
    encode_rpc,
    extract_output_ids,
    extract_output_logprobs,
    merge_json_model_override_args,
    parse_rpc_message,
)
from .replay import mean_ttft, read_completion, time_http_prefill
from .vllm_profiler import build_trace_payload


def _sglang_cli_args(d: dict) -> list[str]:
    """Translate a YAML/config dict into `python -m sglang.launch_server` CLI flags.

    SGLang `ServerArgs` uses `action='store_true'` for booleans (omitting `False`),
    `nargs='+'` for list arguments (e.g. `--cuda-graph-bs 1 2 4`), and JSON strings
    for dict arguments (e.g. `--json-model-override-args '{"num_hidden_layers": 4}'`).
    """
    out: list[str] = []
    for k, v in d.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif isinstance(v, dict):
            out += [flag, json.dumps(v)]
        elif isinstance(v, (list, tuple)):
            if v:
                out += [flag, *(str(x) for x in v)]
        else:
            out += [flag, str(v)]
    return out


class SGLangServerEngine(Engine):
    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        self.cfg, self.tier = cfg, tier
        self.port = tier.port or 30000
        self.url = f"http://127.0.0.1:{self.port}"
        self.http = httpx.Client(
            trust_env=False,
            timeout=tier.timeout_s,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        self.proc: subprocess.Popen | None = None
        self._log_file = None
        self.agentx = AgentXConfig(**tier.agentx) if tier.agentx is not None else None
        self.served_name = (
            tier.engine_overrides.get("served_model_name")
            or cfg.extra_args.get("served_model_name")
            or cfg.model
        )
        self._trace_dir: str | None = None
        admin_key = (
            tier.engine_overrides.get("admin_api_key")
            or cfg.extra_args.get("admin_api_key")
            or os.environ.get("SGLANG_ADMIN_API_KEY")
        )
        self._admin_headers = {"Authorization": f"Bearer {admin_key}"} if admin_key else {}

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        args: dict = dict(
            model_path=self.cfg.model,
            tp_size=self.cfg.tensor_parallel_size,
            context_length=self.cfg.max_model_len,
            host="127.0.0.1",
            port=self.port,
        )
        if self.cfg.tokenizer:
            args["tokenizer_path"] = self.cfg.tokenizer
        args.update(self.cfg.extra_args)
        args.update(self.tier.engine_overrides)
        if self.tier.num_layers is not None:
            args["json_model_override_args"] = merge_json_model_override_args(
                args.get("json_model_override_args"),
                {"num_hidden_layers": self.tier.num_layers},
            )
        elif isinstance(args.get("json_model_override_args"), dict):
            args["json_model_override_args"] = json.dumps(args["json_model_override_args"])

        if args.get("served_model_name"):
            self.served_name = str(args["served_model_name"])

        env = bootstrap_env(os.environ)
        if self.tier.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.tier.gpus))
        if self.tier.trace_kernels:
            directory = (
                os.environ.get("SGLANG_TORCH_PROFILER_DIR")
                or Path("runs/traces") / self.tier.name
            )
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._trace_dir = str(directory)
            env["SGLANG_TORCH_PROFILER_DIR"] = self._trace_dir

        cmd = [sys.executable, "-m", "sglang.launch_server", *_sglang_cli_args(args)]
        self._log_file = open(Path(f"sglang_serve_{self.tier.name}.log"), "ab")
        print(f"[kiln] {self.tier.name}: {shlex.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while time.time() - t0 < 3600:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"sglang.launch_server exited ({self.proc.returncode}); "
                    f"see sglang_serve_{self.tier.name}.log"
                )
            try:
                if self.http.get(f"{self.url}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("sglang.launch_server did not become healthy in 1h")
        print(f"[kiln] {self.tier.name}: sglang server up in {time.time() - t0:.0f}s, "
              f"info {self._rpc('kiln_info')}")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def shutdown(self) -> None:
        shutdown_process_group(self.proc, timeout=120.0)
        self.proc = None
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()
        self._log_file = None

    # ---------------------------------------------------------------- reload
    def _rpc(self, method: str, *args: str):
        r = self.http.post(
            f"{self.url}/update_weights_from_disk",
            headers=self._admin_headers,
            json={"model_path": encode_rpc(method, *args)},
        )
        body = {}
        if r.content:
            try:
                body = r.json()
            except ValueError:
                body = {"message": r.text}
        if r.status_code >= 400 or not body.get("success", False):
            msg = body.get("message") or body.get("detail") or r.text or f"HTTP {r.status_code}"
            raise RuntimeError(f"SGLang {method} failed: {msg}")
        return parse_rpc_message(body.get("message"))

    def _reset_cache(self) -> None:
        for _ in range(30):     # fails with 400 while requests are still draining; retry
            r = self.http.post(f"{self.url}/flush_cache", headers=self._admin_headers)
            if r.status_code == 200:
                if not r.content:
                    return
                try:
                    data = r.json()
                    if not isinstance(data, dict) or data.get("success", True):
                        return
                except ValueError:
                    return
            time.sleep(2)
        raise RuntimeError("SGLang /flush_cache kept failing")

    def hot_reload(self, candidate_path: str) -> None:
        args = (
            [os.path.abspath(candidate_path)]
            if self.assessment_phase == "both"
            else [os.path.abspath(candidate_path), self.assessment_phase]
        )
        secs = self._rpc("kiln_hot_reload", *args)
        self._reset_cache()
        if isinstance(secs, (int, float)):
            print(f"[kiln] {self.tier.name}: hot reload {secs:.1f}s")

    def hot_reload_source(
        self, source: str, label: str = "candidate.py", pd_role: str | None = None
    ) -> None:
        args = [source, label] if pd_role is None else [source, label, pd_role]
        secs = self._rpc("kiln_hot_reload_source", *args)
        self._reset_cache()
        if isinstance(secs, (int, float)):
            print(f"[kiln] {self.tier.name}: hot reload (inline) {secs:.1f}s")

    def teardown_candidate(self) -> None:
        self._rpc("kiln_teardown")
        self._reset_cache()

    # ---------------------------------------------------------------- measure
    def _generate_one_stats(self, prompt_ids: list[int], max_tokens: int, *, stream: bool = True) -> dict:
        body, ntok, itl_ms = read_completion(self.http, f"{self.url}/generate", {
            "input_ids": prompt_ids,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True},
            "stream": stream,
        }, sglang=True)
        meta = body.get("meta_info") or {}
        out: dict = {"tokens": ntok, "itl_ms": itl_ms}
        for key in (
            "spec_verify_ct",
            "spec_accept_length",
            "spec_accept_rate",
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
        ):
            val = meta.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                out[key] = float(val)
        return out

    def _generate_one(self, prompt_ids: list[int], max_tokens: int) -> int:
        return int(self._generate_one_stats(prompt_ids, max_tokens, stream=False)["tokens"])

    def _spec_draft_gamma(self) -> int | None:
        merged = {**self.cfg.extra_args, **self.tier.engine_overrides}
        dspark_bs = merged.get("speculative_dspark_block_size")
        if isinstance(dspark_bs, (int, float)) and int(dspark_bs) >= 1:
            return int(dspark_bs)
        n_draft = merged.get("speculative_num_draft_tokens")
        if isinstance(n_draft, (int, float)) and int(n_draft) > 1:
            return int(n_draft) - 1
        return None

    def _read_spec_metrics(self) -> dict[str, float]:
        return read_prometheus_metrics(
            self.http,
            self.url,
            {
                "spec_accept_length": ("sglang:spec_accept_length",),
                "spec_accept_rate": ("sglang:spec_accept_rate",),
            },
        )

    def _aggregate_spec_metrics(self, stats_list: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        verify_stats = [s for s in stats_list if s.get("spec_verify_ct", 0) > 0]
        if verify_stats:
            tot_tokens = sum(int(s["tokens"]) for s in verify_stats)
            tot_verify = sum(int(s["spec_verify_ct"]) for s in verify_stats)
            out["spec_accept_length"] = tot_tokens / tot_verify
        else:
            lens = [float(s["spec_accept_length"]) for s in stats_list if "spec_accept_length" in s]
            if lens:
                out["spec_accept_length"] = sum(lens) / len(lens)

        draft_stats = [
            s
            for s in stats_list
            if s.get("spec_num_proposed_drafts", 0) > 0 and "spec_num_correct_drafts" in s
        ]
        if draft_stats:
            tot_corr = sum(float(s["spec_num_correct_drafts"]) for s in draft_stats)
            tot_prop = sum(float(s["spec_num_proposed_drafts"]) for s in draft_stats)
            out["spec_accept_rate"] = max(0.0, min(1.0, tot_corr / tot_prop))
        elif "spec_accept_length" in out and (gamma := self._spec_draft_gamma()):
            out["spec_accept_rate"] = max(0.0, min(1.0, (out["spec_accept_length"] - 1.0) / gamma))
        else:
            rates = [float(s["spec_accept_rate"]) for s in stats_list if "spec_accept_rate" in s]
            if rates:
                out["spec_accept_rate"] = max(0.0, min(1.0, sum(rates) / len(rates)))

        if "spec_accept_length" not in out or "spec_accept_rate" not in out:
            prom = self._read_spec_metrics()
            for key in ("spec_accept_length", "spec_accept_rate"):
                if key not in out and key in prom:
                    out[key] = prom[key]
        return out

    def _kernel_report(
        self,
        pool: concurrent.futures.ThreadPoolExecutor,
        requests: list[Request],
        max_new_tokens: int = 32,
    ) -> tuple[str | None, dict | None]:
        if not self._trace_dir or not requests:
            return None, None
        try:
            if max_new_tokens == 1:
                # Reproduce the cold prefill used for throughput and TTFT.
                self._reset_cache()
            self.http.post(
                f"{self.url}/start_profile",
                headers=self._admin_headers,
                json={"output_dir": self._trace_dir},
            ).raise_for_status()
            try:
                list(pool.map(lambda req: self._generate_one(req.prompt_ids, max_new_tokens), requests[:4]))
            finally:
                self.http.post(
                    f"{self.url}/stop_profile",
                    headers=self._admin_headers,
                ).raise_for_status()
            return build_trace_payload(self._trace_dir)
        except Exception:
            return None, None

    def capture_snapshots(self, snapshot_dir: str, m_buckets: list[int]) -> dict | None:
        if not self.is_alive():
            return None
        self._rpc("kiln_enable_snapshot_capture", os.path.abspath(snapshot_dir), list(m_buckets))
        try:
            self._generate_one([1, 2, 3, 4], 2)
        finally:
            res = self._rpc("kiln_finish_snapshot_capture")
        return res

    def measure_requests(
        self, requests: list[Request], repeats: int, trace: bool, *, skip_prefill: bool = False
    ) -> MeasureResult:
        """Measure offline `requests` concurrently against `sglang.launch_server` (fast tier)."""
        isl = sum(len(r.prompt_ids) for r in requests)
        prefill_tps = 0.0
        ttft_s = 0.0
        workers = min(max(len(requests), 1), 32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            if self.assessment_phase == "prefill":
                if not skip_prefill:
                    list(pool.map(lambda r: self._generate_one(r.prompt_ids, 1), requests))
                prefill_runs: list[float] = []
                ttft_runs: list[float] = []
                for _ in range(repeats):
                    self._reset_cache()
                    dt, ttft = time_http_prefill(pool, requests, self._generate_one)
                    ttft_runs.append(ttft)
                    prefill_runs.append(isl / dt)
                report, trace_analysis = (
                    self._kernel_report(pool, requests, max_new_tokens=1) if trace else (None, None)
                )
                prefill_tps = (sum(prefill_runs) / len(prefill_runs)) if prefill_runs else 0.0
                buckets = bucket_tps_for_requests(requests, 0.0, prefill_tps)
                return MeasureResult(
                    decode_tps=0.0,
                    prefill_tps=prefill_tps,
                    ttft_s=mean_ttft(ttft_runs),
                    runs_decode_tps=[],
                    runs_prefill_tps=prefill_runs,
                    kernel_report=report,
                    trace_analysis=trace_analysis,
                    extra=({"bucket_tps": buckets} if buckets else {}),
                )

            if not skip_prefill:
                list(pool.map(lambda r: self._generate_one(r.prompt_ids, 1), requests))
                if self.assessment_phase != "decode":
                    self._reset_cache()
                    prefill_s, ttft_s = time_http_prefill(pool, requests, self._generate_one)
                    prefill_tps = isl / prefill_s

            decode_runs = []
            itl_samples: list[float] = []
            decode_stats: list[dict] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                batch_stats = list(pool.map(lambda r: self._generate_one_stats(r.prompt_ids, r.output_len), requests))
                dt = max(time.perf_counter() - t0, 1e-9)
                ntok = sum(int(s["tokens"]) for s in batch_stats)
                decode_runs.append(ntok / dt)
                decode_stats.extend(batch_stats)
                itl_samples.extend(s["itl_ms"] for s in batch_stats if s["itl_ms"] is not None)

            report, trace_analysis = self._kernel_report(pool, requests) if trace else (None, None)
        decode_tps = sum(decode_runs) / len(decode_runs) if decode_runs else 0.0
        spec_metrics = self._aggregate_spec_metrics(decode_stats) if repeats > 0 else {}
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
        """Use offline `/generate` replay if `agentx` is None; else run real AgentX harness."""
        if self.agentx is None:
            return self.measure_requests(requests, repeats, trace)
        tp = int(
            self.tier.engine_overrides.get(
                "tp_size",
                self.cfg.extra_args.get("tp_size", self.cfg.tensor_parallel_size),
            )
        )
        dp = int(self.tier.engine_overrides.get("dp_size", self.cfg.extra_args.get("dp_size", 1)))
        dp_attn = bool(
            self.tier.engine_overrides.get(
                "enable_dp_attention", self.cfg.extra_args.get("enable_dp_attention", False)
            )
        )
        n_gpus = len(self.tier.gpus) or (tp if dp_attn else tp * dp)
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
                timeout_s=max(self.tier.timeout_s, (self.agentx.benchmark_duration_s or 7200) * 3),
            )
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
        self._reset_cache()
        out_ids: list[list[int]] = []
        out_lps: list[list[dict[int, float]]] = []
        all_have_lps = bool(prompts)
        for p in prompts:
            payload = {
                "input_ids": p,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "stream": False,
            }
            r = self.http.post(f"{self.url}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            ids = extract_output_ids(data)
            if ids is None:
                raise RuntimeError("SGLang did not return output_ids for the correctness probe")
            out_ids.append(ids)
            lps = extract_output_logprobs(data)
            if lps is None:
                all_have_lps = False
            else:
                out_lps.append(lps)
        return out_ids, (out_lps if all_have_lps else None)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.greedy_probe(prompts, max_new_tokens)[0]
