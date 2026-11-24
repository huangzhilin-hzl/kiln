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
