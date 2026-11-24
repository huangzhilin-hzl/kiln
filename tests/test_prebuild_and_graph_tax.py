"""Tests for out-of-lock candidate prebuild, lazy baseline teardown, and phase-aware CUDA graph skip."""
from __future__ import annotations

from kiln.patches import compose_candidates, compose_pd_pair, inspect_candidate_source


def test_inspect_candidate_effective_phase_and_only_kiln_patches():
    prefill_src = (
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('fast_mla_prefill', phase='prefill', targets=['mod.mla'])\n"
        "def p1(model):\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    info_p = inspect_candidate_source(prefill_src)
    assert info_p["syntax_ok"] is True
    assert info_p["effective_phase"] == "prefill"
    assert info_p["only_kiln_patches"] is True
    assert info_p["needs_prebuild"] is False

    decode_src = (
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('fast_gemv_decode', phase='decode', targets=['mod.gemv'])\n"
        "def p2(model):\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    info_d = inspect_candidate_source(decode_src)
    assert info_d["effective_phase"] == "decode"
    assert info_d["only_kiln_patches"] is True

    # Mixed prefill + decode patches resolve to effective_phase == 'all'
    mixed_src = prefill_src + "\n" + (
        "@kiln_patch('fast_gemv', phase='decode', targets=['mod.gemv'])\n"
        "def p2(model):\n"
        "    return None\n"
    )
    info_m = inspect_candidate_source(mixed_src)
    assert info_m["effective_phase"] == "all"
    assert info_m["only_kiln_patches"] is True

    # Custom `def make_forward` alongside a prefill @kiln_patch must conservatively report 'all'
    custom_fwd_src = (
        "from kiln.patches import kiln_patch\n"
        "@kiln_patch('fast_mla_prefill', phase='prefill')\n"
        "def p1(model):\n"
        "    return None\n"
        "def make_forward(model):\n"
        "    return lambda x: x\n"
    )
    info_custom = inspect_candidate_source(custom_fwd_src)
    assert info_custom["effective_phase"] == "all"
    assert info_custom["only_kiln_patches"] is False

    # Top-level module attribute mutation conservatively disables only_kiln_patches / prefill inference
    mutating_src = (
        "import math\n"
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "math.custom_attr = 1\n"
        "@kiln_patch('p', phase='prefill')\n"
        "def p1(model):\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    info_mut = inspect_candidate_source(mutating_src)
    assert info_mut["effective_phase"] == "all"
    assert info_mut["only_kiln_patches"] is False

    # Explicit KILN_PHASE declaration overrides for custom `def make_forward`
    explicit_src = (
        "KILN_PHASE = 'prefill'\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    info_exp = inspect_candidate_source(explicit_src)
    assert info_exp["effective_phase"] == "prefill"
    assert info_exp["only_kiln_patches"] is False


def test_inspect_candidate_build_triggers_and_compose_propagation():
    cpp_src = (
        "from torch.utils.cpp_extension import load_inline\n"
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('cpp_prefill', phase='prefill', targets=['mod.a'])\n"
        "def p_a(model):\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    info_cpp = inspect_candidate_source(cpp_src)
    assert info_cpp["needs_prebuild"] is True
    assert "load_inline" in info_cpp["build_triggers"]
    assert info_cpp["effective_phase"] == "prefill"

    hook_src = (
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "def kiln_build():\n"
        "    pass\n"
        "@kiln_patch('hook_prefill', phase='prefill', targets=['mod.b'])\n"
        "def p_b(model):\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    info_hook = inspect_candidate_source(hook_src)
    assert info_hook["needs_prebuild"] is True
    assert "kiln_build" in info_hook["build_triggers"]

    c_cc_src = (
        "import subprocess\n"
        "def _compile():\n"
        "    subprocess.run(['gcc', '-O3', '-shared', '-fPIC', 'a.c', '-o', 'a.so'])\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    info_cc = inspect_candidate_source(c_cc_src)
    assert info_cc["needs_prebuild"] is True
    assert "c_compiler" in info_cc["build_triggers"]

    opt_out_src = "KILN_PREBUILD = False\n" + cpp_src
    assert inspect_candidate_source(opt_out_src)["needs_prebuild"] is False

    # Composing two prefill-only candidates preserves effective_phase='prefill', only_kiln_patches=True, and kiln_build
    merged = compose_candidates([("j1", cpp_src), ("j2", hook_src)])
    info_merged = inspect_candidate_source(merged["source"])
    assert info_merged["effective_phase"] == "prefill"
    assert info_merged["only_kiln_patches"] is True
    assert info_merged["needs_prebuild"] is True
    assert "kiln_build" in info_merged["build_triggers"]

    pd_merged = compose_pd_pair(("jp", hook_src), ("jd", "SPEEDUP = 1.1\ndef make_forward(m): return None\n"))
    info_pd = inspect_candidate_source(pd_merged["source"])
    assert info_pd["needs_prebuild"] is True
    assert "kiln_build" in info_pd["build_triggers"]


def test_prebuild_candidate_syntax_hook_lazy_load_inline_and_timeout(tmp_path, monkeypatch):
    import os
    import threading
    from pathlib import Path

    from kiln.engine.prebuild import ensure_extension_cache_env, prebuild_candidate
    from kiln.engine.sglang_ext import bootstrap_env

    monkeypatch.setenv("KILN_EXT_CACHE_ROOT", str(tmp_path / "ext_cache"))
    env = ensure_extension_cache_env({})
    assert env["TORCH_EXTENSIONS_DIR"] == str((tmp_path / "ext_cache" / "torch_extensions").resolve())
    assert env["KILN_CACHE_DIR"] == str((tmp_path / "ext_cache" / "ext").resolve())
    assert Path(env["TORCH_EXTENSIONS_DIR"]).is_dir()
    assert Path(env["KILN_CACHE_DIR"]).is_dir()

    sgl_env = bootstrap_env({"KILN_EXT_CACHE_ROOT": str(tmp_path / "ext_cache")})
    assert sgl_env["TORCH_EXTENSIONS_DIR"] == env["TORCH_EXTENSIONS_DIR"]
    assert sgl_env["KILN_CACHE_DIR"] == env["KILN_CACHE_DIR"]

    # 1. Plain candidate with no build triggers is skipped in <10ms under mode="auto"
    plain = tmp_path / "plain.py"
    plain.write_text("SPEEDUP = 1.05\ndef make_forward(model):\n    return None\n")
    r_plain = prebuild_candidate(plain, mode="auto")
    assert r_plain.ok is True and r_plain.skipped is True and r_plain.reason == "no_build_triggers"

    # 2. SyntaxError fails fast in-process without spawning a subprocess
    bad_syn = tmp_path / "bad_syn.py"
    bad_syn.write_text("def broken(:\n")
    r_syn = prebuild_candidate(bad_syn, mode="auto")
    assert r_syn.ok is False and r_syn.reason == "syntax_error" and "SyntaxError" in (r_syn.error or "")

    # 3. Explicit kiln_build() hook populates shared KILN_CACHE_DIR in isolated subprocess
    hook_ok = tmp_path / "hook_ok.py"
    hook_ok.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def kiln_build():\n"
        "    out = Path(os.environ['KILN_CACHE_DIR']) / 'built_marker.txt'\n"
        "    out.write_text(os.environ.get('KILN_PD_ROLE', ''))\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    r_hook = prebuild_candidate(hook_ok, mode="auto", pd_role="prefill", base_env=dict(os.environ))
    assert r_hook.ok is True and r_hook.skipped is False and r_hook.reason == "prebuilt"
    assert (Path(env["KILN_CACHE_DIR"]) / "built_marker.txt").read_text() == "prefill"

    # 4. Failing kiln_build() rejects candidate cleanly with compiler error
    hook_fail = tmp_path / "hook_fail.py"
    hook_fail.write_text(
        "def kiln_build():\n"
        "    raise RuntimeError('Error building extension: nvcc fatal : unsupported gpu architecture')\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    r_fail = prebuild_candidate(hook_fail, mode="auto", base_env=dict(os.environ))
    assert r_fail.ok is False and r_fail.reason == "kiln_build_failed"
    assert "nvcc fatal" in (r_fail.error or "")

    # 5. Lazy load_inline inside make_forward(model) warms cache AND catches build errors
    fake_torch_dir = tmp_path / "fake_pkgs"
    (fake_torch_dir / "torch" / "utils").mkdir(parents=True)
    (fake_torch_dir / "torch" / "__init__.py").write_text("")
    (fake_torch_dir / "torch" / "utils" / "__init__.py").write_text("")
    (fake_torch_dir / "torch" / "utils" / "cpp_extension.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def load_inline(name, cpp_sources='', cuda_sources='', **kw):\n"
        "    if 'SYNTAX_ERR_CUDA' in cuda_sources:\n"
        "        raise RuntimeError(f'Error building extension {name!r}: identifier undefined')\n"
        "    p = Path(os.environ['TORCH_EXTENSIONS_DIR']) / f'{name}.so'\n"
        "    p.write_text('compiled')\n"
        "    return str(p)\n"
        "def load(name, sources, **kw):\n"
        "    return load_inline(name)\n"
    )
    custom_env = {
        **os.environ,
        "PYTHONPATH": f"{fake_torch_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    lazy_ok = tmp_path / "lazy_ok.py"
    lazy_ok.write_text(
        "from torch.utils.cpp_extension import load_inline\n"
        "def make_forward(model):\n"
        "    load_inline(name='fast_k', cuda_sources='__global__ void k(){}')\n"
        "    return model.nonexistent_real_model_attr.layers[0]\n"  # dummy model raises AttributeError after build!
    )
    r_lazy_ok = prebuild_candidate(lazy_ok, mode="auto", base_env=custom_env)
    assert r_lazy_ok.ok is True and r_lazy_ok.extensions_built == 1
    assert (Path(env["TORCH_EXTENSIONS_DIR"]) / "fast_k.so").read_text() == "compiled"

    lazy_bad = tmp_path / "lazy_bad.py"
    lazy_bad.write_text(
        "from torch.utils.cpp_extension import load_inline\n"
        "def make_forward(model):\n"
        "    load_inline(name='broken_k', cuda_sources='SYNTAX_ERR_CUDA')\n"
        "    return model.layers[0]\n"
    )
    r_lazy_bad = prebuild_candidate(lazy_bad, mode="auto", base_env=custom_env)
    assert r_lazy_bad.ok is False and r_lazy_bad.reason == "compile_error"
    assert "Error building extension" in (r_lazy_bad.error or "")

    # 6. Timeout / cancellation terminates hung build subprocess group cleanly
    hung = tmp_path / "hung.py"
    hung.write_text(
        "import time\n"
        "def kiln_build():\n"
        "    time.sleep(30)\n"
        "def make_forward(model):\n"
        "    return None\n"
    )
    r_to = prebuild_candidate(hung, mode="auto", timeout_s=0.25, base_env=dict(os.environ))
    assert r_to.ok is False and r_to.reason == "timeout"

    cancel_ev = threading.Event()
    threading.Timer(0.15, cancel_ev.set).start()
    r_cancel = prebuild_candidate(hung, mode="auto", timeout_s=10.0, cancel_event=cancel_ev, base_env=dict(os.environ))
    assert r_cancel.ok is False and r_cancel.reason == "cancelled"


def test_vllm_lazy_teardown_and_phase_aware_skip_recapture(tmp_path):
    from types import SimpleNamespace
    from kiln.config import EngineConfig, TierConfig
    from kiln.engine.vllm_ext import HotReloadExt
    from kiln.engine.vllm_offline import VLLMOfflineEngine

    class FakeModel:
        def __init__(self):
            self.tag = "baseline"

        def named_modules(self):
            yield "", self

        def forward(self, mode="decode"):
            return f"{self.tag}:{mode}"

    events: list[str] = []
    model = FakeModel()
    ext = HotReloadExt()
    ext.model_config = SimpleNamespace(enforce_eager=False)
    ext.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode="CUDAGraphMode.FULL_DECODE_ONLY"),
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=False),
    )
    ext.model_runner = SimpleNamespace(
        get_model=lambda: model,
        speculator=None,
        model_state=None,
        cudagraph_manager=SimpleNamespace(graphs={1: "g1", 2: "g2"}),
    )

    def fake_release():
        events.append("release")
        ext.model_runner.cudagraph_manager.graphs.clear()
        ext._kiln_graphs_released = True

    def fake_rebuild():
        if not ext._kiln_graphs_released:
            fake_release()
        events.append(f"capture:{model.forward('decode')}")
        ext.model_runner.cudagraph_manager.graphs = {1: model.forward("decode")}

    ext._kiln_release_graphs = fake_release
    ext._kiln_rebuild = fake_rebuild

    # 1. Prefill-only @kiln_patch on decode-only-graph vLLM skips graph release & recapture completely!
    cand_prefill = tmp_path / "cand_prefill.py"
    cand_prefill.write_text(
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('prefill_opt', phase='prefill')\n"
        "def patch_p(model):\n"
        "    orig = model.forward\n"
        "    model.forward = lambda mode='decode': 'fast_prefill' if mode == 'prefill' else orig(mode)\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    ext.kiln_hot_reload(str(cand_prefill))
    assert model.forward("prefill") == "fast_prefill"
    assert model.forward("decode") == "baseline:decode"
    assert events == []
    assert ext.kiln_info()["baseline_graphs_valid"] is True
    assert ext.kiln_info()["last_skipped_graph_rebuild"] is True

    td_lazy_p = ext.kiln_teardown_lazy()
    assert td_lazy_p["baseline_graphs_valid"] is True
    assert events == []
    assert model.forward("prefill") == "baseline:prefill"

    # 2. General / decode candidates: back-to-back jobs capture 1x each instead of 2x!
    cand_d1 = tmp_path / "cand_d1.py"
    cand_d1.write_text("def make_forward(model):\n    return lambda mode='decode': f'd1:{mode}'\n")
    cand_d2 = tmp_path / "cand_d2.py"
    cand_d2.write_text("def make_forward(model):\n    return lambda mode='decode': f'd2:{mode}'\n")

    ext.kiln_hot_reload(str(cand_d1))
    assert events == ["release", "capture:d1:decode"]
    assert ext.kiln_info()["baseline_graphs_valid"] is False

    td1 = ext.kiln_teardown_lazy()
    assert td1["baseline_graphs_valid"] is False
    # Released d1's graphs immediately, ZERO baseline recapture!
    assert events == ["release", "capture:d1:decode", "release"]
    assert model.forward("decode") == "baseline:decode"

    ext.kiln_hot_reload(str(cand_d2))
    # Directly captured d2 without redundant release or baseline capture!
    assert events == ["release", "capture:d1:decode", "release", "capture:d2:decode"]
    td2 = ext.kiln_teardown_lazy()
    assert td2["baseline_graphs_valid"] is False
    assert events == ["release", "capture:d1:decode", "release", "capture:d2:decode", "release"]

    # 3. Prefill-only candidate arriving while baseline graphs are invalid first restores baseline graphs once,
    #    then leaves baseline_graphs_valid=True for subsequent prefill candidates!
    ext.kiln_hot_reload(str(cand_prefill))
    assert events == [
        "release", "capture:d1:decode", "release",
        "capture:d2:decode", "release",
        "capture:baseline:decode",
    ]
    assert ext.kiln_info()["baseline_graphs_valid"] is True
    assert ext.kiln_teardown_lazy()["baseline_graphs_valid"] is True
    assert len(events) == 6  # still 6: no release or capture on teardown!

    # 4. If torch.compile is enabled (mode=3), prefill candidate must NOT skip rebuild
    ext.vllm_config.compilation_config.mode = 3
    ext.kiln_hot_reload(str(cand_prefill))
    assert events[-2:] == ["release", "capture:baseline:decode"]
    assert ext.kiln_info()["baseline_graphs_valid"] is False
    ext.kiln_teardown_lazy()

    # 5. Host VLLMOfflineEngine auto-guards baseline greedy_probe / ensure_baseline
    rpc_log: list[str] = []

    class FakeLLM:
        def collective_rpc(self, method, args=(), timeout=None):
            rpc_log.append(method)
            if method == "kiln_hot_reload":
                return [float(ext.kiln_hot_reload(*args))]
            if method == "kiln_teardown_lazy":
                return [ext.kiln_teardown_lazy()]
            if method == "kiln_ensure_baseline":
                return [ext.kiln_ensure_baseline()]
            if method == "kiln_teardown":
                return [ext.kiln_teardown()]
            raise AssertionError(method)

        def reset_prefix_cache(self):
            rpc_log.append("reset_cache")

    host_eng = VLLMOfflineEngine(EngineConfig(kind="vllm_offline"), TierConfig(name="fast"))
    host_eng.llm = FakeLLM()
    host_eng.greedy_outputs = lambda prompts, n: [[7] * n for _ in prompts]
    host_eng.hot_reload(str(cand_d1))
    host_eng.teardown_candidate_lazy()
    assert host_eng.needs_baseline_rebuild() is True
    assert "kiln_ensure_baseline" not in rpc_log

    # Baseline greedy_probe automatically triggers ensure_baseline once, then stays clean!
    assert host_eng.greedy_probe([[1, 2]], 2) == ([[7, 7]], None)
    assert "kiln_ensure_baseline" in rpc_log
    assert host_eng.needs_baseline_rebuild() is False
    prev_len = len(rpc_log)
    host_eng.ensure_baseline()
    assert len(rpc_log) == prev_len  # 0 RPCs when already clean!


def test_sglang_lazy_teardown_topology_retention_and_pd_role_skip(tmp_path):
    from types import SimpleNamespace
    from kiln.engine.sglang_ext import SGLangHotReloadController

    class EagerRunner:
        pass

    class FakeGraphRunner:
        def __init__(self, label: str):
            self.label = label
            self.graphs = {1: object(), 4: object()}

        def destroy(self):
            events.append(f"destroy:{self.label}")

    class FakeModel:
        def __init__(self):
            self.mode_val = "base"

        def named_modules(self):
            yield "", self

        def forward(self, phase="decode"):
            return f"{self.mode_val}:{phase}"

    events: list[str] = []
    eager = EagerRunner()
    model = FakeModel()
    server_args = SimpleNamespace(
        disable_cuda_graph=False,
        enable_torch_compile=False,
        enable_piecewise_cuda_graph=False,
    )

    class FakeTargetRunner:
        def __init__(self):
            self.model = model
            self.server_args = server_args
            self.tp_rank = 0
            self.eager_runner = eager
            self.prefill_cuda_graph_runner = eager
            self.decode_cuda_graph_runner = FakeGraphRunner("boot_decode")

        def init_decode_cuda_graph(self):
            tag = model.forward("decode")
            events.append(f"capture_decode:{tag}")
            self.decode_cuda_graph_runner = FakeGraphRunner(tag)

    runner = FakeTargetRunner()
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner), dp_rank=0)
    ctrl = SGLangHotReloadController(scheduler)

    cand_prefill = tmp_path / "sgl_prefill.py"
    cand_prefill.write_text(
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('p_only', phase='prefill')\n"
        "def patch_p(model):\n"
        "    orig = model.forward\n"
        "    model.forward = lambda phase='decode': 'opt_prefill' if phase == 'prefill' else orig(phase)\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )
    cand_decode = tmp_path / "sgl_decode.py"
    cand_decode.write_text(
        "from kiln.patches import kiln_patch, make_composite_forward\n"
        "@kiln_patch('d_only', phase='decode')\n"
        "def patch_d(model):\n"
        "    orig = model.forward\n"
        "    model.forward = lambda phase='decode': 'opt_decode' if phase == 'decode' else orig(phase)\n"
        "    return None\n"
        "make_forward, restore = make_composite_forward(globals())\n"
    )

    # 1. Prefill-only candidate on decode-only SGLang skips decode CUDA graph destroy & capture!
    ctrl.kiln_hot_reload(str(cand_prefill))
    assert events == []
    assert model.forward("prefill") == "opt_prefill"
    assert runner.decode_cuda_graph_runner.label == "boot_decode"
    assert ctrl.kiln_teardown_lazy()["baseline_graphs_valid"] is True
    assert events == []

    # 2. Back-to-back decode candidates retain _target_graph_attrs across lazy release!
    ctrl.kiln_hot_reload(str(cand_decode))
    assert events == ["destroy:boot_decode", "capture_decode:opt_decode"]
    assert ctrl.kiln_teardown_lazy()["baseline_graphs_valid"] is False
    assert events == ["destroy:boot_decode", "capture_decode:opt_decode", "destroy:opt_decode"]
    assert runner.decode_cuda_graph_runner is None  # released during lazy teardown

    # Second decode candidate still knows to recapture decode_cuda_graph_runner!
    ctrl.kiln_hot_reload(str(cand_decode))
    assert events == [
        "destroy:boot_decode", "capture_decode:opt_decode", "destroy:opt_decode",
        "capture_decode:opt_decode",
    ]
    assert runner.decode_cuda_graph_runner is not None
    ctrl.kiln_teardown_lazy()

    # ensure_baseline restores baseline decode graph
    ens = ctrl.kiln_ensure_baseline()
    assert ens["rebuilt"] is True and ens["baseline_graphs_valid"] is True
    assert runner.decode_cuda_graph_runner.label == "base:decode"

    # 3. Asymmetric PD role skip: even with enable_torch_compile=True, a decode-only patch
    #    on pd_role='prefill' (and a prefill-only patch on pd_role='decode') skips rebuild!
    server_args.enable_torch_compile = True
    events.clear()
    ctrl.kiln_hot_reload(str(cand_decode), pd_role="prefill")
    assert events == []
    assert ctrl.kiln_info()["last_skipped_graph_rebuild"] is True
    assert ctrl.kiln_teardown_lazy()["baseline_graphs_valid"] is True

    ctrl.kiln_hot_reload(str(cand_prefill), pd_role="decode")
    assert events == []
    assert ctrl.kiln_info()["last_skipped_graph_rebuild"] is True
    assert ctrl.kiln_teardown_lazy()["baseline_graphs_valid"] is True
