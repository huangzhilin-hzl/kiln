"""Composite SGLang Prefill-Decode (`xPyD`) Disaggregated Server Engine.

Architecture:
- Data plane (`greedy_probe`, `/generate` replay, `agentx profile`): targets `router_url`
  (`sglang_router --pd-disaggregation` or an external PD router).
- Control plane (`hot_reload_source`, `teardown_candidate`, `/flush_cache`, `/health`):
  bypasses the router and broadcasts directly in parallel to all `x` Prefill instances
  (`pd_role='prefill'`) and all `y` Decode instances (`pd_role='decode'`).
- Candidate source is shipped inline over HTTP JSON (`kiln_hot_reload_source`) so multi-pod
  Kubernetes deployments require no shared filesystem across Prefill and Decode nodes.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import httpx

from kiln.bench.agentx import AgentXConfig, agentx_result_to_extra, run_agentx_profile
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, PDClusterConfig, PDNodeConfig, RunConfig, TierConfig

from .base import Engine, InvalidBenchmark, MeasureResult, bucket_tps_for_requests, shutdown_process_group
from .sglang_ext import (
    bootstrap_env,
    encode_rpc,
    extract_output_ids,
    extract_output_logprobs,
    merge_json_model_override_args,
    parse_rpc_message,
)
from .sglang_server import _sglang_cli_args
from .vllm_profiler import build_trace_payload


def _looks_like_deepseek_v41(model_name: str, merged_args: dict) -> bool:
    low = model_name.lower()
    if "dsv41" in low or ("deepseek" in low and ("v4.1" in low or "v41" in low)):
        return True
    if str(merged_args.get("tool_call_parser") or "").lower() in ("deepseekv41", "deepseek_v41"):
        return True
    if str(merged_args.get("reasoning_parser") or "").lower() in ("deepseek-v41", "deepseek_v41"):
        return True
    return str(merged_args.get("attention_backend") or "").lower() == "dsv4"


def validate_sglang_pd_args(
    model_name: str,
    transfer_backend: str,
    prefill_args_list: list[dict],
    decode_args_list: list[dict],
) -> None:
    """Validate SGLang PD and DeepSeek-V4.1 disaggregation constraints upfront."""
    if transfer_backend == "fake":
        raise ValueError("SGLang PD prefill server does not support transfer_backend='fake'")

    all_nodes = [("prefill", i, a) for i, a in enumerate(prefill_args_list)] + [
        ("decode", i, a) for i, a in enumerate(decode_args_list)
    ]
    for role, idx, args in all_nodes:
        if args.get("enable_encoder_swa_bounded_replay"):
            raise ValueError(
                f"{role}[{idx}]: --enable-encoder-swa-bounded-replay does not support PD disaggregation"
            )
        if str(args.get("dsv4_attn_backend") or "").lower() == "trtllm":
            raise ValueError(
                f"{role}[{idx}]: --dsv4-attn-backend trtllm does not support PD disaggregation"
            )
        if role == "decode" and args.get("disaggregation_decode_enable_radix_cache"):
            if args.get("speculative_algorithm") is not None:
                raise ValueError(
                    f"decode[{idx}]: disaggregation_decode_enable_radix_cache is incompatible "
                    f"with speculative decoding ({args.get('speculative_algorithm')})"
                )

    is_dsv41 = any(_looks_like_deepseek_v41(model_name, a) for _, _, a in all_nodes)
    any_spec = any(a.get("speculative_algorithm") is not None for _, _, a in all_nodes)
    if is_dsv41 and any_spec:
        if transfer_backend not in ("mooncake", "mooncake_tcp"):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires transfer_backend='mooncake' or 'mooncake_tcp', "
                f"got {transfer_backend!r}"
            )
        spec_blocks: set[int] = set()
        tp_sizes: set[int] = set()
        for role, idx, a in all_nodes:
            alg = str(a.get("speculative_algorithm") or "").upper()
            if alg != "DSPARK":
                raise ValueError(
                    f"DeepSeek-V4.1 DSpark PD requires both prefill and decode servers to enable "
                    f"speculative_algorithm='DSPARK' ({role}[{idx}] has {a.get('speculative_algorithm')!r})"
                )
            spec_blocks.add(int(a.get("speculative_dspark_block_size", 5)))
            tp_sizes.add(int(a.get("tp_size", 1)))
            if int(a.get("dp_size", 1)) != 1 or int(a.get("dcp_size", 1)) != 1:
                raise ValueError(
                    f"DeepSeek-V4.1 DSpark PD requires dp_size=1 and dcp_size=1 ({role}[{idx}])"
                )
            if role == "decode" and (
                a.get("enable_prefill_cp")
                or a.get("enable_dp_attention")
                or int(a.get("attn_cp_size", 1)) != 1
            ):
                raise ValueError(
                    f"decode[{idx}]: DeepSeek-V4.1 DSpark PD forbids CP / DP-attention on decode workers"
                )
            if role == "prefill" and a.get("enable_prefill_cp"):
                cp_strat = str(a.get("cp_strategy") or "interleave")
                if cp_strat != "interleave":
                    raise ValueError(
                        f"prefill[{idx}]: DeepSeek-V4.1 prefill CP only supports cp_strategy='interleave'"
                    )
                if a.get("enable_dsa_cache_layer_split"):
                    raise ValueError(
                        f"prefill[{idx}]: disable enable_dsa_cache_layer_split with DSpark PD prefill CP"
                    )
            elif role == "prefill" and a.get("enable_dp_attention"):
                raise ValueError(
                    f"prefill[{idx}]: DeepSeek-V4.1 DSpark PD forbids enable_dp_attention without prefill CP"
                )
        if len(spec_blocks) > 1:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark PD requires matching speculative_dspark_block_size across P and D, got {sorted(spec_blocks)}"
            )
        if len(tp_sizes) > 1:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark PD requires matching tp_size across P and D, got {sorted(tp_sizes)}"
            )


def build_pd_node_server_args(
    cfg: EngineConfig,
    tier: TierConfig,
    pd: PDClusterConfig,
    node: PDNodeConfig,
    role: Literal["prefill", "decode"],
    default_port: int,
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> dict:
    """Resolve `sglang.launch_server` arguments for one Prefill or Decode node."""
    args: dict = dict(
        model_path=cfg.model,
        tp_size=cfg.tensor_parallel_size,
        context_length=cfg.max_model_len,
        host=host_override or node.host,
        port=port_override or node.port or default_port,
    )
    if cfg.tokenizer:
        args["tokenizer_path"] = cfg.tokenizer
    args.update(cfg.extra_args)
    args.update(tier.engine_overrides)
    args["disaggregation_mode"] = role
    args["disaggregation_transfer_backend"] = pd.transfer_backend
    if pd.ib_device is not None:
        args["disaggregation_ib_device"] = pd.ib_device
    if node.bootstrap_port is not None:
        args["disaggregation_bootstrap_port"] = node.bootstrap_port
    args.update(node.engine_overrides)

    if host_override is not None:
        args["host"] = host_override
    if port_override is not None:
        args["port"] = port_override

    if tier.num_layers is not None:
        args["json_model_override_args"] = merge_json_model_override_args(
            args.get("json_model_override_args"),
            {"num_hidden_layers": tier.num_layers},
        )
    elif isinstance(args.get("json_model_override_args"), dict):
        args["json_model_override_args"] = json.dumps(args["json_model_override_args"])
    return args


def build_pd_node_launch_spec(
    run: RunConfig,
    tier_name: str,
    role: Literal["prefill", "decode"],
    index: int = 0,
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Build `(cmd, full_env, env_overrides)` to launch one SGLang PD node (`kiln pd-node`)."""
    tier = next((t for t in run.tiers if t.name == tier_name), None)
    if tier is None:
        raise KeyError(f"Unknown tier {tier_name!r}; available: {[t.name for t in run.tiers]}")
    if tier.pd is None:
        raise ValueError(f"Tier {tier_name!r} has no 'pd' configuration block")

    pd = tier.pd
    p_args = [
        build_pd_node_server_args(run.engine, tier, pd, n, "prefill", 30100 + i)
        for i, n in enumerate(pd.prefill)
    ]
    d_args = [
        build_pd_node_server_args(run.engine, tier, pd, n, "decode", 30200 + i)
        for i, n in enumerate(pd.decode)
    ]
    validate_sglang_pd_args(run.engine.model, pd.transfer_backend, p_args, d_args)

    nodes = pd.prefill if role == "prefill" else pd.decode
    if index < 0 or index >= len(nodes):
        raise IndexError(f"{role} index {index} out of range (have {len(nodes)})")
    node = nodes[index]
    default_port = (30100 if role == "prefill" else 30200) + index
    args = build_pd_node_server_args(
        run.engine,
        tier,
        pd,
        node,
        role,
        default_port,
        host_override=host_override,
        port_override=port_override,
    )
    cmd = [sys.executable, "-m", "sglang.launch_server", *_sglang_cli_args(args)]

    full_env = bootstrap_env(os.environ)
    overrides: dict[str, str] = {
        "PYTHONPATH": full_env["PYTHONPATH"],
        "KILN_SGLANG_HOOK": full_env["KILN_SGLANG_HOOK"],
        "TORCHINDUCTOR_FX_GRAPH_CACHE": full_env["TORCHINDUCTOR_FX_GRAPH_CACHE"],
        "KILN_PD_ROLE": role,
    }
    full_env["KILN_PD_ROLE"] = role
    if pd.transfer_backend == "mooncake_tcp":
        full_env.setdefault("MC_FORCE_TCP", "1")
        overrides["MC_FORCE_TCP"] = full_env["MC_FORCE_TCP"]
    if node.gpus:
        devs = ",".join(map(str, node.gpus))
        full_env["CUDA_VISIBLE_DEVICES"] = devs
        overrides["CUDA_VISIBLE_DEVICES"] = devs
    for k, v in node.env.items():
        full_env[str(k)] = str(v)
        overrides[str(k)] = str(v)
    return cmd, full_env, overrides


def build_router_cmd(
    pd: PDClusterConfig,
    prefill_urls: list[str],
    decode_urls: list[str],
    host: str,
    port: int,
) -> list[str]:
    """Build CLI argv to launch `sglang_router` in `--pd-disaggregation` mode."""
    if pd.router_command:
        cmd = shlex.split(pd.router_command)
    elif importlib.util.find_spec("sglang_router") is not None:
        cmd = [sys.executable, "-m", "sglang_router.launch_router"]
    else:
        router_bin = shutil.which("sglang-router")
        cmd = [router_bin] if router_bin else [sys.executable, "-m", "sglang_router.launch_router"]

    cmd += ["--pd-disaggregation", "--host", host, "--port", str(port)]
    for node, url in zip(pd.prefill, prefill_urls):
        cmd += ["--prefill", url]
        if node.bootstrap_port is not None:
            cmd.append(str(node.bootstrap_port))
    for url in decode_urls:
        cmd += ["--decode", url]
    if pd.router_policy:
        cmd += ["--policy", pd.router_policy]
    cmd += list(pd.router_extra_args)
    return cmd


class SGLangPDServerEngine(Engine):
    """Persistent SGLang `xPyD` disaggregated cluster engine."""

    def __init__(self, cfg: EngineConfig, tier: TierConfig):
        if tier.pd is None:
            raise ValueError(f"Tier {tier.name!r} requires a 'pd' configuration for SGLangPDServerEngine")
        self.cfg, self.tier = cfg, tier
        self.pd: PDClusterConfig = tier.pd

        self.prefill_args = [
            build_pd_node_server_args(cfg, tier, self.pd, n, "prefill", 30100 + i)
            for i, n in enumerate(self.pd.prefill)
        ]
        self.decode_args = [
            build_pd_node_server_args(cfg, tier, self.pd, n, "decode", 30200 + i)
            for i, n in enumerate(self.pd.decode)
        ]
        validate_sglang_pd_args(cfg.model, self.pd.transfer_backend, self.prefill_args, self.decode_args)

        self.prefill_urls = [
            n.resolved_url(30100 + i) for i, n in enumerate(self.pd.prefill)
        ]
        self.decode_urls = [
            n.resolved_url(30200 + i) for i, n in enumerate(self.pd.decode)
        ]
        self.router_port = self.pd.router_port or tier.port or 30080
        if self.pd.router_url and not self.pd.should_spawn_router():
            self.url = self.pd.router_url.rstrip("/")
        else:
            r_host = "127.0.0.1" if self.pd.router_host in ("0.0.0.0", "::") else self.pd.router_host
            self.url = self.pd.router_url.rstrip("/") if self.pd.router_url else f"http://{r_host}:{self.router_port}"

        self.total_gpus = sum(
            n.resolved_num_gpus(cfg, tier.engine_overrides) for n in self.pd.prefill
        ) + sum(
            n.resolved_num_gpus(cfg, tier.engine_overrides) for n in self.pd.decode
        )

        self.http = httpx.Client(
            trust_env=False,
            timeout=tier.timeout_s,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        )
        self.agentx = AgentXConfig(**tier.agentx) if tier.agentx is not None else None
        self.served_name = (
            tier.engine_overrides.get("served_model_name")
            or cfg.extra_args.get("served_model_name")
            or cfg.model
        )
        admin_key = (
            tier.engine_overrides.get("admin_api_key")
            or cfg.extra_args.get("admin_api_key")
            or os.environ.get("SGLANG_ADMIN_API_KEY")
        )
        self._admin_headers = {"Authorization": f"Bearer {admin_key}"} if admin_key else {}
        self._trace_dir: str | None = None
        self._node_procs: list[tuple[str, subprocess.Popen]] = []
        self._router_proc: subprocess.Popen | None = None
        self._log_files: list = []
        self._started = False

    # ---------------------------------------------------------------- lifecycle
    def _spawn_local_node(
        self, role: Literal["prefill", "decode"], idx: int, node: PDNodeConfig, args: dict
    ) -> None:
        env = bootstrap_env(os.environ)
        env["KILN_PD_ROLE"] = role
        if self.pd.transfer_backend == "mooncake_tcp":
            env.setdefault("MC_FORCE_TCP", "1")
        if node.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, node.gpus))
        if self._trace_dir:
            env["SGLANG_TORCH_PROFILER_DIR"] = self._trace_dir
        for k, v in node.env.items():
            env[str(k)] = str(v)

        cmd = [sys.executable, "-m", "sglang.launch_server", *_sglang_cli_args(args)]
        log_path = Path(f"sglang_pd_{self.tier.name}_{role}_{idx}.log")
        lf = open(log_path, "ab")
        self._log_files.append(lf)
        print(f"[kiln] {self.tier.name} {role}[{idx}]: {shlex.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._node_procs.append((f"{role}[{idx}]", proc))

    def _wait_healthy(self, label: str, url: str, deadline: float) -> None:
        while time.time() < deadline:
            for proc_label, proc in self._node_procs:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"SGLang PD {proc_label} exited ({proc.returncode}) while waiting for {label}"
                    )
            if self._router_proc is not None and self._router_proc.poll() is not None:
                raise RuntimeError(
                    f"sglang_router exited ({self._router_proc.returncode}) while waiting for {label}"
                )
            try:
                if self.http.get(f"{url}/health", timeout=5.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"{label} ({url}) did not become healthy before timeout")

    def start(self) -> None:
        if self.tier.trace_kernels:
            directory = Path(
                os.environ.get("SGLANG_TORCH_PROFILER_DIR") or Path("runs/traces") / self.tier.name
            ).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._trace_dir = str(directory)

        t0 = time.time()
        deadline = t0 + 3600.0
        for idx, (node, args) in enumerate(zip(self.pd.prefill, self.prefill_args)):
            if node.should_spawn():
                self._spawn_local_node("prefill", idx, node, args)
        for idx, (node, args) in enumerate(zip(self.pd.decode, self.decode_args)):
            if node.should_spawn():
                self._spawn_local_node("decode", idx, node, args)

        for idx, url in enumerate(self.prefill_urls):
            self._wait_healthy(f"prefill[{idx}]", url, deadline)
        for idx, url in enumerate(self.decode_urls):
            self._wait_healthy(f"decode[{idx}]", url, deadline)

        # Verify every P and D worker has the kiln SGLang hook active.
        infos = self._broadcast_rpc(
            [(url, "kiln_info", []) for url in (*self.prefill_urls, *self.decode_urls)]
        )

        if self.pd.should_spawn_router():
            rcmd = build_router_cmd(
                self.pd,
                self.prefill_urls,
                self.decode_urls,
                self.pd.router_host,
                self.router_port,
            )
            rlog = open(Path(f"sglang_pd_{self.tier.name}_router.log"), "ab")
            self._log_files.append(rlog)
            print(f"[kiln] {self.tier.name} router: {shlex.join(rcmd)}")
            self._router_proc = subprocess.Popen(
                rcmd,
                env=dict(os.environ),
                stdout=rlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._wait_healthy("router", self.url, deadline)
        self._started = True
        print(
            f"[kiln] {self.tier.name}: PD cluster ({len(self.prefill_urls)}P + {len(self.decode_urls)}D, "
            f"{self.total_gpus} GPUs) up in {time.time() - t0:.0f}s; nodes={infos}"
        )

    def is_alive(self) -> bool:
        if not self._started:
            return False
        if self._router_proc is not None and self._router_proc.poll() is not None:
            return False
        for _, proc in self._node_procs:
            if proc.poll() is not None:
                return False
        # Probe remote/external endpoints quickly
        probe_urls = [self.url]
        for node, url in zip(self.pd.prefill, self.prefill_urls):
            if not node.should_spawn():
                probe_urls.append(url)
        for node, url in zip(self.pd.decode, self.decode_urls):
            if not node.should_spawn():
                probe_urls.append(url)
        for url in probe_urls:
            try:
                if self.http.get(f"{url}/health", timeout=5.0).status_code != 200:
                    return False
            except httpx.HTTPError:
                return False
        return True

    def shutdown(self) -> None:
        self._started = False
        shutdown_process_group(self._router_proc, timeout=60.0)
        self._router_proc = None
        for _, proc in reversed(self._node_procs):
            shutdown_process_group(proc, timeout=120.0)
        self._node_procs.clear()
        for lf in self._log_files:
            try:
                if not lf.closed:
                    lf.close()
            except Exception:
                pass
        self._log_files.clear()

    # ---------------------------------------------------------------- control plane (bypasses router)
    def _rpc_one(self, node_url: str, method: str, *args):
        r = self.http.post(
            f"{node_url}/update_weights_from_disk",
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
            raise RuntimeError(f"SGLang PD {node_url} {method} failed: {msg}")
        return parse_rpc_message(body.get("message"))

    def _broadcast_rpc(self, calls: list[tuple[str, str, list]]) -> list:
        if not calls:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futs = [pool.submit(self._rpc_one, url, method, *args) for url, method, args in calls]
            return [f.result() for f in futs]

    def _flush_one(self, node_url: str) -> None:
        for _ in range(30):
            r = self.http.post(f"{node_url}/flush_cache", headers=self._admin_headers)
            if r.status_code == 200:
                if not r.content:
                    return
                try:
                    data = r.json()
                    if not isinstance(data, dict) or data.get("success", True):
                        return
                except ValueError:
                    return
            time.sleep(1.0)
        raise RuntimeError(f"SGLang PD {node_url} /flush_cache kept failing")

    def _reset_all_caches(self) -> None:
        urls = [*self.prefill_urls, *self.decode_urls]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
            futs = [pool.submit(self._flush_one, u) for u in urls]
            for f in futs:
                f.result()

    def hot_reload_source(self, source: str, label: str = "candidate.py") -> None:
        calls: list[tuple[str, str, list]] = [
            (u, "kiln_hot_reload_source", [source, label, "prefill"]) for u in self.prefill_urls
        ] + [
            (u, "kiln_hot_reload_source", [source, label, "decode"]) for u in self.decode_urls
        ]
        try:
            timings = self._broadcast_rpc(calls)
        except Exception:
            # Roll back all P and D instances so partial reloads never poison the cluster.
            try:
                self._broadcast_rpc(
                    [(u, "kiln_teardown", []) for u in (*self.prefill_urls, *self.decode_urls)]
                )
                self._reset_all_caches()
            except Exception:
                pass
            raise
        self._reset_all_caches()
        nums = [float(x) for x in timings if isinstance(x, (int, float))]
        if nums:
            print(f"[kiln] {self.tier.name}: PD hot reload max {max(nums):.1f}s across {len(calls)} nodes")

    def hot_reload(self, candidate_path: str) -> None:
        p = Path(candidate_path)
        self.hot_reload_source(p.read_text(encoding="utf-8"), label=p.name)

    def teardown_candidate(self) -> None:
        self._broadcast_rpc(
            [(u, "kiln_teardown", []) for u in (*self.prefill_urls, *self.decode_urls)]
        )
        self._reset_all_caches()

    # ---------------------------------------------------------------- data plane (via router)
    def _generate_one(self, prompt_ids: list[int], max_tokens: int) -> int:
        r = self.http.post(
            f"{self.url}/generate",
            json={
                "input_ids": prompt_ids,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "stream": False,
            },
        )
        r.raise_for_status()
        body = r.json()
        meta = body.get("meta_info") or {}
        if meta.get("completion_tokens") is not None:
            return int(meta["completion_tokens"])
        ids = extract_output_ids(body)
        if ids is not None:
            return len(ids)
        return max_tokens

    def _kernel_report(
        self, pool: concurrent.futures.ThreadPoolExecutor, requests: list[Request]
    ) -> tuple[str | None, dict | None]:
        if not self._trace_dir or not requests:
            return None, None
        all_nodes = [*self.prefill_urls, *self.decode_urls]
        try:
            for u in all_nodes:
                self.http.post(
                    f"{u}/start_profile",
                    headers=self._admin_headers,
                    json={"output_dir": self._trace_dir},
                ).raise_for_status()
            try:
                list(pool.map(lambda req: self._generate_one(req.prompt_ids, 32), requests[:4]))
            finally:
                for u in all_nodes:
                    self.http.post(f"{u}/stop_profile", headers=self._admin_headers).raise_for_status()
            return build_trace_payload(self._trace_dir)
        except Exception:
            return None, None

    def measure_requests(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        """Measure concurrent requests end-to-end through the PD router."""
        isl = sum(len(r.prompt_ids) for r in requests)
        workers = min(max(len(requests), 1), 32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda r: self._generate_one(r.prompt_ids, 1), requests))
            self._reset_all_caches()
            t0 = time.perf_counter()
            list(pool.map(lambda r: self._generate_one(r.prompt_ids, 1), requests))
            prefill_s = max(time.perf_counter() - t0, 1e-9)

            decode_runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ntok = sum(pool.map(lambda r: self._generate_one(r.prompt_ids, r.output_len), requests))
                dt = max(time.perf_counter() - t0, 1e-9)
                decode_runs.append(ntok / dt)

            report, trace_analysis = self._kernel_report(pool, requests) if trace else (None, None)
        decode_tps = sum(decode_runs) / len(decode_runs)
        prefill_tps = isl / prefill_s
        return MeasureResult(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            ttft_s=prefill_s / max(len(requests), 1),
            runs_decode_tps=decode_runs,
            kernel_report=report,
            trace_analysis=trace_analysis,
            extra={
                "bucket_tps": bucket_tps_for_requests(requests, decode_tps, prefill_tps),
                "pd_topology": self.topology_summary(),
            },
        )

    def topology_summary(self) -> dict:
        return {
            "prefill_instances": len(self.prefill_urls),
            "decode_instances": len(self.decode_urls),
            "total_gpus": self.total_gpus,
            "transfer_backend": self.pd.transfer_backend,
            "router_url": self.url,
        }

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        if self.agentx is None:
            return self.measure_requests(requests, repeats, trace)
        artifact_root = Path("runs") / "agentx"
        artifact_root.mkdir(parents=True, exist_ok=True)
        runs: list[float] = []
        extra: dict = {"metric": "agentx_total_tput_per_gpu", "repeats": []}
        for _ in range(repeats):
            self._reset_all_caches()
            artifact_dir = tempfile.mkdtemp(prefix=f"{self.tier.name}_pd_", dir=artifact_root)
            last = run_agentx_profile(
                self.url,
                self.served_name,
                self.cfg.tokenizer or self.cfg.model,
                self.agentx,
                artifact_dir,
                self.total_gpus,
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
                extra["pd_topology"] = self.topology_summary()
                raise InvalidBenchmark(f"Invalid AgentX PD measurement: {last.raw_path}", extra)
            runs.append(last.total_tput_per_gpu)
        extra = agentx_result_to_extra(last, extra["repeats"], self.agentx.unsafe_override)
        extra["pd_topology"] = self.topology_summary()
        return MeasureResult(
            decode_tps=sum(runs) / len(runs),
            prefill_tps=0.0,
            ttft_s=0.0,
            runs_decode_tps=runs,
            extra=extra,
        )

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        if "greedy_outputs" in self.__dict__:
            return self.greedy_outputs(prompts, max_new_tokens), None
        self._reset_all_caches()
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
                raise RuntimeError("SGLang PD router did not return output_ids for correctness probe")
            out_ids.append(ids)
            lps = extract_output_logprobs(data)
            if lps is None:
                all_have_lps = False
            else:
                out_lps.append(lps)
        return out_ids, (out_lps if all_have_lps else None)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.greedy_probe(prompts, max_new_tokens)[0]
