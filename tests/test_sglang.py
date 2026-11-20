import asyncio
import dataclasses
import gzip
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kiln.bench.agentx import AgentXResult
from kiln.bench.traffic import Request
from kiln.config import EngineConfig, RunConfig, TierConfig
from kiln.engine import make_engine
from kiln.engine.base import InvalidBenchmark
from kiln.engine.sglang_ext import (
    SGLangHotReloadController,
    bootstrap_env,
    decode_rpc,
    encode_rpc,
    merge_json_model_override_args,
    parse_rpc_message,
    patch_scheduler_cls,
    patch_tokenizer_manager_cls,
)
from kiln.engine.sglang_offline import SGLangOfflineEngine
from kiln.engine.sglang_server import SGLangServerEngine, _sglang_cli_args


def test_engine_factory_and_config_sglang():
    tier = TierConfig(name="fast")
    assert isinstance(make_engine(EngineConfig(kind="sglang_offline"), tier), SGLangOfflineEngine)
    assert isinstance(make_engine(EngineConfig(kind="sglang"), tier), SGLangOfflineEngine)
    assert isinstance(make_engine(EngineConfig(kind="sglang_server"), tier), SGLangServerEngine)

    run = RunConfig.load("configs/llama1b_sglang.yaml")
    assert run.engine.kind == "sglang_offline"
    fast, slow = run.tiers
    assert fast.engine_kind == "sglang_offline" and fast.num_layers == 4
    assert slow.engine_kind == "sglang_server" and slow.port == 30000


def test_sglang_cli_args_and_json_override():
    argv = _sglang_cli_args({
        "model_path": "/models/m",
        "tp_size": 8,
        "enable_torch_compile": True,
        "disable_cuda_graph": False,
        "speculative_algorithm": None,
        "cuda_graph_bs": [1, 2, 4, 8],
        "json_model_override_args": {"num_hidden_layers": 4},
    })
    assert argv == [
        "--model-path", "/models/m",
        "--tp-size", "8",
        "--enable-torch-compile",
        "--cuda-graph-bs", "1", "2", "4", "8",
        "--json-model-override-args", '{"num_hidden_layers": 4}',
    ]
    assert json.loads(merge_json_model_override_args(None, {"num_hidden_layers": 4})) == {"num_hidden_layers": 4}
    assert json.loads(
        merge_json_model_override_args('{"rope_theta": 10000}', {"num_hidden_layers": 4})
    ) == {"rope_theta": 10000, "num_hidden_layers": 4}
    assert json.loads(
        merge_json_model_override_args({"rope_theta": 10000}, {"num_hidden_layers": 2})
    ) == {"rope_theta": 10000, "num_hidden_layers": 2}


def test_sglang_hot_reload_controller_standard_and_speculative(tmp_path: Path, monkeypatch):
    events: list[str] = []

    class FakeMLP:
        def forward(self, x):
            return x + 1

    class FakeModel:
        def __init__(self):
            self.mlp = FakeMLP()

        def named_modules(self):
            yield "", self
            yield "mlp", self.mlp

        def forward(self, x):
            return self.mlp.forward(x)

    class FakeGraphRunner:
        def __init__(self, tag: str):
            self.tag = tag
            self.graphs = {1: object(), 2: object()}

        def destroy(self):
            events.append(f"destroy:{self.tag}")

    pool_calls: list[object] = []
    fake_pool_mod = types.ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    fake_pool_mod.set_global_graph_memory_pool = lambda p: pool_calls.append(p)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.cuda_graph_runner", fake_pool_mod)

    model = FakeModel()
    server_args = SimpleNamespace(
        disable_cuda_graph=False,
        enable_torch_compile=True,
        enable_piecewise_cuda_graph=True,
    )

    class FakeModelRunner:
        def __init__(self):
            self.model = model
            self.server_args = server_args
            self.tp_rank = 0
            self.graph_runner = FakeGraphRunner("decode")
            self.piecewise_cuda_graph_runner = FakeGraphRunner("piecewise")

        def init_device_graphs(self):
            events.append("init_device_graphs")
            self.graph_runner = FakeGraphRunner("decode_new")

        def init_piecewise_cuda_graphs(self):
            events.append("init_piecewise_cuda_graphs")
            self.piecewise_cuda_graph_runner = FakeGraphRunner("piecewise_new")

    class FakeSpecWorker:
        def __init__(self, target_worker):
            self.target_worker = target_worker
            self.cuda_graph_runner = FakeGraphRunner("eagle_draft")

        def init_cuda_graphs(self):
            events.append("spec_init_cuda_graphs")
            self.cuda_graph_runner = FakeGraphRunner("eagle_draft_new")

    target_runner = FakeModelRunner()
    target_worker = SimpleNamespace(model_runner=target_runner)
    spec_worker = FakeSpecWorker(target_worker)
    overlap_client = SimpleNamespace(worker=spec_worker)
    scheduler = SimpleNamespace(
        tp_worker=overlap_client,
        dp_rank=1,
        flush_cache=lambda: events.append("flush_cache"),
    )

    ctrl = SGLangHotReloadController(scheduler)
    info = ctrl.kiln_info()
    assert info["engine"] == "sglang"
    assert info["model_cls"] == "FakeModel"
    assert info["has_speculator"] is True
    assert info["num_graphs"] == 6
    assert info["dp_rank"] == 1

    cand = tmp_path / "cand.py"
    cand.write_text(
        "import types\n"
        "def make_forward(model):\n"
        "    model.mlp.forward = types.MethodType(lambda self, x: x * 10, model.mlp)\n"
        "    return types.MethodType(lambda self, x: self.mlp.forward(x) + 3, model)\n"
    )
    secs = ctrl.kiln_hot_reload(str(cand))
    assert secs >= 0
    assert model.forward(5) == 53
    assert pool_calls == [None]
    # Target graphs must be recaptured before speculative draft graphs.
    assert events.index("init_device_graphs") < events.index("init_piecewise_cuda_graphs")
    assert events.index("init_piecewise_cuda_graphs") < events.index("spec_init_cuda_graphs")

    ctrl.kiln_teardown()
    assert model.forward(5) == 6
    assert "forward" not in model.__dict__
    assert "forward" not in model.mlp.__dict__


@pytest.mark.parametrize("restore_fails", [False, True])
def test_sglang_hot_reload_rollback_and_dirty_state(tmp_path: Path, restore_fails: bool):
    class Model:
        state = "baseline"

        def forward(self):
            return self.state

    model = Model()
    runner = SimpleNamespace(model=model, server_args=SimpleNamespace(disable_cuda_graph=True))
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
    ctrl = SGLangHotReloadController(scheduler)

    cand = tmp_path / "broken.py"
    restore_body = 'raise ValueError("cannot restore")' if restore_fails else 'model.state = "baseline"'
    cand.write_text(
        'def make_forward(model):\n'
        '    model.state = "candidate"\n'
        '    model.forward = lambda: "candidate forward"\n'
        '    raise ValueError("candidate setup failed")\n'
        f'def restore(model):\n    {restore_body}\n'
    )
    if restore_fails:
        with pytest.raises(RuntimeError, match="rollback failed"):
            ctrl.kiln_hot_reload(str(cand))
        with pytest.raises(RuntimeError, match="restart the engine"):
            ctrl.kiln_teardown()
        assert "forward" not in vars(model)
    else:
        with pytest.raises(ValueError, match="candidate setup failed"):
            ctrl.kiln_hot_reload(str(cand))
        ctrl.kiln_teardown()
        assert model.forward() == "baseline"


def test_sglang_scheduler_and_tokenizer_hooks(tmp_path: Path):
    passthrough_calls: list[str] = []

    class FakeModel:
        def forward(self, x):
            return x + 1

    class FakeScheduler:
        def __init__(self):
            runner = SimpleNamespace(model=FakeModel(), server_args=SimpleNamespace(disable_cuda_graph=True))
            self.tp_worker = SimpleNamespace(model_runner=runner)

        def update_weights_from_disk(self, recv_req):
            passthrough_calls.append(recv_req.model_path)
            return SimpleNamespace(success=True, message="weights loaded", num_paused_requests=0)

    patch_scheduler_cls(FakeScheduler)
    sched = FakeScheduler()

    # Non-kiln requests pass straight through.
    out_normal = sched.update_weights_from_disk(SimpleNamespace(model_path="/weights/ckpt"))
    assert out_normal.success and out_normal.message == "weights loaded"
    assert passthrough_calls == ["/weights/ckpt"]

    # Kiln RPC requests dispatch to controller.
    out_info = sched.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_info")))
    assert out_info.success is True
    assert parse_rpc_message(out_info.message)["model_cls"] == "FakeModel"

    # TokenizerManager hook preserves model_path / server_args.model_path across kiln RPCs.
    class FakeTokenizerManager:
        def __init__(self):
            self.model_path = "/models/orig"
            self.server_args = SimpleNamespace(model_path="/models/orig")

        async def update_weights_from_disk(self, obj, request=None):
            # Simulate an upstream version mutating model_path
            self.model_path = obj.model_path
            self.server_args.model_path = obj.model_path
            return True, "ok"

    patch_tokenizer_manager_cls(FakeTokenizerManager)
    tm = FakeTokenizerManager()
    ok, msg = asyncio.run(tm.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_info"))))
    assert ok and msg == "ok"
    assert tm.model_path == "/models/orig"
    assert tm.server_args.model_path == "/models/orig"


def test_sglang_bootstrap_sitecustomize_in_subprocess(tmp_path: Path):
    # Create a fake `sglang.srt.managers.scheduler` package under tmp_path.
    pkg = tmp_path / "sglang" / "srt" / "managers"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "sglang", tmp_path / "sglang" / "srt", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "scheduler.py").write_text(
        "class Scheduler:\n"
        "    def update_weights_from_disk(self, recv_req):\n"
        "        return ('orig', recv_req.model_path)\n"
    )

    env = bootstrap_env(dict(os.environ))
    env["PYTHONPATH"] = f"{env['PYTHONPATH']}{os.pathsep}{tmp_path}"
    res = subprocess.run(
        [
            sys.executable, "-c",
            "from sglang.srt.managers.scheduler import Scheduler; "
            "print(getattr(Scheduler, '_kiln_hooked', False))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "True"

    # With KILN_SGLANG_HOOK disabled, sitecustomize must be a no-op.
    env_off = dict(env)
    env_off["KILN_SGLANG_HOOK"] = "0"
    res_off = subprocess.run(
        [
            sys.executable, "-c",
            "from sglang.srt.managers.scheduler import Scheduler; "
            "print(getattr(Scheduler, '_kiln_hooked', False))",
        ],
        env=env_off,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_off.stdout.strip() == "False"


def test_sglang_offline_engine_full_lifecycle(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_kwargs: list[dict] = []
    rpc_calls: list[dict] = []
    flushes: list[bool] = []
    profile_events: list[str] = []
    state = {"return_direct_ids": True}

    class FakeSGLEngine:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

        def update_weights_from_disk(self, model_path: str, load_format=None):
            rpc = decode_rpc(model_path)
            assert rpc is not None
            rpc_calls.append(rpc)
            if rpc["method"] == "kiln_info":
                return True, json.dumps({"ok": True, "result": {"engine": "sglang", "num_graphs": 4}})
            return True, json.dumps({"ok": True, "result": 0.12})

        def flush_cache(self):
            flushes.append(True)
            return SimpleNamespace(success=True)

        def start_profile(self, output_dir=None):
            profile_events.append(f"start:{output_dir}")
            if output_dir:
                p = Path(output_dir) / "12345.trace.json.gz"
                with gzip.open(p, "wt", encoding="utf-8") as f:
                    json.dump({"traceEvents": [{"cat": "kernel", "name": "sgl_gemm", "dur": 2500.0}]}, f)

        def stop_profile(self):
            profile_events.append("stop")

        def generate(self, input_ids=None, sampling_params=None, return_logprob=False):
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], int):
                max_new = sampling_params["max_new_tokens"]
                toks = [100 + i for i in range(max_new)]
                if state["return_direct_ids"]:
                    return {"text": "ok", "output_ids": toks, "meta_info": {"completion_tokens": max_new}}
                if return_logprob:
                    return {
                        "text": "ok",
                        "meta_info": {"output_token_logprobs": [[-0.1, t, "x"] for t in toks]},
                    }
                return {"text": "ok", "meta_info": {"completion_tokens": max_new}}
            out = []
            for idx, _ in enumerate(input_ids):
                sp = sampling_params[idx] if isinstance(sampling_params, list) else sampling_params
                max_new = sp["max_new_tokens"]
                out.append({
                    "text": "ok",
                    "output_ids": list(range(max_new)),
                    "meta_info": {"completion_tokens": max_new},
                })
            return out

        def shutdown(self):
            profile_events.append("shutdown")

    fake_sglang = types.ModuleType("sglang")
    fake_sglang.Engine = FakeSGLEngine
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)

    cfg = EngineConfig(kind="sglang_offline", model="dummy/model", tensor_parallel_size=2, max_model_len=4096)
    tier = TierConfig(name="fast", num_layers=4, repeats=2, trace_kernels=True)
    eng = SGLangOfflineEngine(cfg, tier)
    eng.start()
    assert init_kwargs[0]["model_path"] == "dummy/model"
    assert init_kwargs[0]["tp_size"] == 2
    assert init_kwargs[0]["context_length"] == 4096
    assert json.loads(init_kwargs[0]["json_model_override_args"]) == {"num_hidden_layers": 4}

    cand = tmp_path / "cand.py"
    cand.write_text("pass\n")
    eng.hot_reload(str(cand))
    assert rpc_calls[-1]["method"] == "kiln_hot_reload"

    assert eng.greedy_outputs([[1, 2, 3]], 4) == [[100, 101, 102, 103]]
    state["return_direct_ids"] = False
    assert eng.greedy_outputs([[1, 2, 3]], 3) == [[100, 101, 102]]

    reqs = [Request(prompt_ids=[1, 2, 3, 4], output_len=8) for _ in range(3)]
    m = eng.measure(reqs, repeats=2, trace=True)
    assert m.decode_tps > 0 and m.prefill_tps > 0 and len(m.runs_decode_tps) == 2
    assert m.kernel_report is not None and "sgl_gemm" in m.kernel_report

    eng.teardown_candidate()
    assert rpc_calls[-1]["method"] == "kiln_teardown"
    assert len(flushes) >= 4
    eng.shutdown()
    assert "shutdown" in profile_events


def test_sglang_server_engine_http_and_agentx(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen_rpc: list[dict] = []
    seen_auth: list[str | None] = []
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/update_weights_from_disk":
            seen_auth.append(request.headers.get("Authorization"))
            body = json.loads(request.content.decode())
            rpc = decode_rpc(body["model_path"])
            assert rpc is not None
            seen_rpc.append(rpc)
            return httpx.Response(200, json={"success": True, "message": json.dumps({"ok": True, "result": 0.2})})
        if request.url.path == "/flush_cache":
            seen_auth.append(request.headers.get("Authorization"))
            return httpx.Response(200, json={"success": True})
        if request.url.path == "/start_profile":
            p = trace_dir / "999.trace.json.gz"
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump({"traceEvents": [{"cat": "kernel", "name": "flashinfer_mla", "dur": 4000.0}]}, f)
            return httpx.Response(200)
        if request.url.path == "/stop_profile":
            return httpx.Response(200)
        if request.url.path == "/generate":
            body = json.loads(request.content.decode())
            max_new = int(body["sampling_params"]["max_new_tokens"])
            return httpx.Response(200, json={
                "text": "ok",
                "output_ids": [10 + i for i in range(max_new)],
                "meta_info": {"completion_tokens": max_new},
            })
        return httpx.Response(404)

    cfg = EngineConfig(
        kind="sglang_server",
        model="dummy/model",
        tensor_parallel_size=8,
        extra_args={"admin_api_key": "secret-admin"},
    )
    tier = TierConfig(name="fast", share_engine="slow", agentx=None, repeats=2, trace_kernels=True)
    eng = SGLangServerEngine(cfg, tier)
    eng.http = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    eng._trace_dir = str(trace_dir)

    eng.hot_reload(str(tmp_path / "c.py"))
    assert seen_rpc[-1]["method"] == "kiln_hot_reload"
    assert "Bearer secret-admin" in seen_auth

    assert eng.greedy_outputs([[1, 2]], 4) == [[10, 11, 12, 13]]

    reqs = [Request(prompt_ids=[1, 2, 3, 4], output_len=8) for _ in range(4)]
    res = eng.measure(reqs, repeats=2, trace=True)
    assert res.decode_tps > 0 and res.prefill_tps > 0 and len(res.runs_decode_tps) == 2
    assert res.kernel_report is not None and "flashinfer_mla" in res.kernel_report

    eng.teardown_candidate()
    assert seen_rpc[-1]["method"] == "kiln_teardown"
    eng.http.close()

    # AgentX path: valid measurement + invalid repeat rejection
    slow_tier = TierConfig(name="slow", agentx={"benchmark_duration_s": 900}, repeats=1)
    slow_eng = SGLangServerEngine(cfg, slow_tier)
    slow_eng.http = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    monkeypatch.setattr(
        "kiln.engine.sglang_server.run_agentx_profile",
        lambda *a, **kw: AgentXResult(4200.0, 100.0, 5.0, 8.0, True, "ok.json"),
    )
    slow_res = slow_eng.measure(reqs, repeats=1, trace=False)
    assert slow_res.decode_tps == 4200.0
    assert slow_res.extra["submission_valid"] is True

    monkeypatch.setattr(
        "kiln.engine.sglang_server.run_agentx_profile",
        lambda *a, **kw: AgentXResult(4200.0, 100.0, 5.0, 8.0, False, "bad.json"),
    )
    with pytest.raises(InvalidBenchmark):
        slow_eng.measure(reqs, repeats=1, trace=False)
    slow_eng.http.close()


def test_smoke_broken_candidate_supports_sglang_logits_output(monkeypatch):
    import numpy as np
    from scripts.smoke_vllm import BROKEN

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = np.ndarray
    fake_torch.full_like = lambda arr, val: np.full_like(arr, val)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    @dataclasses.dataclass
    class FakeLogitsOutput:
        next_token_logits: object

    class FakeSGLangModel:
        def forward(self, x):
            logits = np.zeros((1, 16), dtype=np.float32)
            logits[0, 3] = 5.0
            return FakeLogitsOutput(next_token_logits=logits)

    ns: dict = {}
    exec(BROKEN.replace("REPLACEMENT_TOKEN", "7"), ns)
    model = FakeSGLangModel()
    new_fwd = ns["make_forward"](model)
    assert new_fwd is not None
    model.forward = new_fwd
    out = model.forward(None)
    assert int(np.argmax(out.next_token_logits, axis=-1).item()) == 7
    ns["restore"](model)


def test_sglang_c859d701_weight_updater_and_backend_cleanup(tmp_path: Path, monkeypatch):
    from kiln.engine.sglang_ext import patch_module

    events: list[str] = []

    class FakeModel:
        def forward(self, x):
            return x + 1

    class EagerRunner:
        pass

    class FakeBackend:
        def __init__(self, tag: str):
            self.tag = tag
            self._graphs = {1: object(), 2: object(), 4: object()}
            self._outputs = {1: object()}
            self._capture_inputs = {1: object()}

        def cleanup(self):
            events.append(f"cleanup:{self.tag}")
            self._graphs.clear()
            self._outputs.clear()
            self._capture_inputs.clear()

    class FakeDecodeRunner:
        def __init__(self, tag: str):
            self.backend = FakeBackend(tag)

    pool_calls: list[object] = []
    fake_pool_mod = types.ModuleType("sglang.srt.model_executor.runner_utils.pool")
    fake_pool_mod.set_global_graph_memory_pool = lambda p: pool_calls.append(p)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.runner_utils.pool", fake_pool_mod)

    model = FakeModel()
    eager = EagerRunner()

    class FakeTargetRunner:
        def __init__(self):
            self.model = model
            self.server_args = SimpleNamespace(disable_cuda_graph=False)
            self.tp_rank = 0
            self.eager_runner = eager
            self.prefill_cuda_graph_runner = eager
            self.decode_cuda_graph_runner = FakeDecodeRunner("target_decode")

        def init_prefill_cuda_graph(self):
            events.append("init_prefill_cuda_graph")

        def init_decode_cuda_graph(self):
            events.append("init_decode_cuda_graph")
            self.decode_cuda_graph_runner = FakeDecodeRunner("target_decode_new")

    class FakeDraftRunner:
        def __init__(self):
            self.eager_runner = EagerRunner()
            self.prefill_cuda_graph_runner = self.eager_runner
            self.decode_cuda_graph_runner = FakeDecodeRunner("draft_decode")

    class FakeDraftWorker:
        def __init__(self):
            self.draft_model_runner = FakeDraftRunner()
            self.cuda_graph_runner = FakeDecodeRunner("dspark_draft")

        def init_cuda_graphs(self):
            events.append("draft_init_cuda_graphs")
            self.cuda_graph_runner = FakeDecodeRunner("dspark_draft_new")

    target_runner = FakeTargetRunner()
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=FakeDraftWorker(),
        dp_rank=0,
        flush_cache=lambda: events.append("sched_flush"),
    )

    @dataclasses.dataclass(kw_only=True, slots=True)
    class FakeWeightUpdaterManager:
        scheduler: object
        tp_worker: object
        draft_worker: object
        flush_cache: object

        def update_weights_from_disk(self, recv_req):
            return SimpleNamespace(success=True, message="disk weights", num_paused_requests=0)

    wu_mod = types.ModuleType("sglang.srt.managers.scheduler_components.weight_updater")
    wu_mod.SchedulerWeightUpdaterManager = FakeWeightUpdaterManager
    patch_module(wu_mod)

    mgr = FakeWeightUpdaterManager(
        scheduler=scheduler,
        tp_worker=scheduler.tp_worker,
        draft_worker=scheduler.draft_worker,
        flush_cache=scheduler.flush_cache,
    )
    info_out = mgr.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_info")))
    assert info_out.success is True
    info = parse_rpc_message(info_out.message)
    assert info["has_speculator"] is True
    assert info["num_graphs"] == 9  # 3 target_decode + 3 draft_decode + 3 dspark_draft (EagerRunner skipped)

    cand = tmp_path / "cand.py"
    cand.write_text("import types\ndef make_forward(model):\n    return types.MethodType(lambda self, x: x * 3, model)\n")
    hr_out = mgr.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_hot_reload", str(cand))))
    assert hr_out.success is True
    assert model.forward(4) == 12
    assert target_runner.prefill_cuda_graph_runner is eager  # EagerRunner preserved!
    assert "init_prefill_cuda_graph" not in events
    assert events.index("cleanup:target_decode") < events.index("init_decode_cuda_graph")
    assert events.index("init_decode_cuda_graph") < events.index("draft_init_cuda_graphs")
    assert pool_calls == [None]

    td_out = mgr.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_teardown")))
    assert td_out.success is True
    assert model.forward(4) == 5

    # TokenizerManager _update_model_path_info must skip mutating served_model_name on kiln RPCs
    # and _restore_tm_paths must not write to read-only ServerArgs (c859d701)
    class _ReadOnlyServerArgs:
        def __init__(self, model_path: str):
            object.__setattr__(self, "model_path", model_path)

        def __setattr__(self, name, value):
            raise AttributeError("server_args is read-only")

    class FakeTM:
        def __init__(self):
            self.model_path = "/models/orig"
            self.served_model_name = "/models/orig"
            self.server_args = _ReadOnlyServerArgs("/models/orig")
            self.recorded = []

        def _update_model_path_info(self, model_path, load_format):
            self.served_model_name = model_path
            self.model_path = model_path
            self.recorded.append((model_path, load_format))

        async def update_weights_from_disk(self, obj, request=None):
            self._update_model_path_info(obj.model_path, obj.load_format)
            return True, "ok", 0

    tm_mod = types.ModuleType("sglang.srt.managers.tokenizer_manager")
    tm_mod.TokenizerManager = FakeTM
    patch_module(tm_mod)

    tm = FakeTM()
    ok, msg, paused = asyncio.run(
        tm.update_weights_from_disk(SimpleNamespace(model_path=encode_rpc("kiln_info"), load_format="auto"))
    )
    assert ok and msg == "ok" and paused == 0
    assert tm.served_model_name == "/models/orig"
    assert tm.model_path == "/models/orig"
    assert tm.recorded == []

    # Normal weight update still invokes _update_model_path_info
    asyncio.run(tm.update_weights_from_disk(SimpleNamespace(model_path="/models/new", load_format="auto")))
    assert tm.served_model_name == "/models/new"
    assert tm.recorded == [("/models/new", "auto")]
