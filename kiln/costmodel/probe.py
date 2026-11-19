"""Empirical hardware & roofline probing (`kiln probe`, TKCC §4.3 Level 2).

Measures actual GPU / node characteristics before persistent inference engines boot:
  1. Topology & static specs (GPU SKU, SM count, L2 & HBM capacity, clocks, NVLink clique, host RAM)
  2. Memory hierarchy bandwidth (L2 hot, HBM copy & triad across sizes, pinned PCIe H2D/D2H)
  3. Cold-L2 Dense GEMM roofline across M={1..4096} for BF16, FP8 (e4m3), and FP4/MXFP4 (if supported)
  4. Grouped MoE GEMM throughput & roofline efficiency across token batch sizes
  5. Kernel launch & CUDA-graph replay overhead + host-sync penalty
  6. Multi-GPU NCCL `all_reduce` and `all_to_all_single` latency (µs) & busbw (GB/s) across TP/EP sizes

Produces a JSON profile (`HardwareProfile`) that:
  - Calibrates `ChipSpec` + `EmpiricalCurves` in `kiln costmodel --profile <path>`
  - Generates a concise Roofline reference (`HARDWARE.md` + prompt block) for `kiln agents`
"""
from __future__ import annotations

import inspect
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kiln.costmodel.model import CHIPS, GB, ChipSpec, EmpiricalCurves

MIB = 1024 * 1024
GIB = 1024 * MIB

# Reference datasheet peaks (unidirectional link_bw matches `nccl-tests` busbw convention).
# Order matters: more specific SKU substrings come first.
_SPEC_DB: list[tuple[str, dict]] = [
    ("H20-3E", {"name": "H20-3e", "flops": 148e12,  "fp8_flops": 296e12,  "fp4_flops": None,     "hbm_bw": 4.0e12,   "link_bw": 450 * GB, "l2_mb": 60,  "fp4_tc": False}),
    ("H200",   {"name": "H200",   "flops": 989e12,  "fp8_flops": 1979e12, "fp4_flops": None,     "hbm_bw": 4.8e12,   "link_bw": 450 * GB, "l2_mb": 50,  "fp4_tc": False}),
    ("H20",    {"name": "H20",    "flops": 148e12,  "fp8_flops": 296e12,  "fp4_flops": None,     "hbm_bw": 4.0e12,   "link_bw": 450 * GB, "l2_mb": 60,  "fp4_tc": False}),
    ("H100",   {"name": "H100",   "flops": 989e12,  "fp8_flops": 1979e12, "fp4_flops": None,     "hbm_bw": 3.35e12,  "link_bw": 450 * GB, "l2_mb": 50,  "fp4_tc": False}),
    ("GB300",  {"name": "GB300",  "flops": 2500e12, "fp8_flops": 5000e12, "fp4_flops": 15000e12, "hbm_bw": 8.0e12,   "link_bw": 900 * GB, "l2_mb": 126, "fp4_tc": True}),
    ("B300",   {"name": "B300",   "flops": 2250e12, "fp8_flops": 4500e12, "fp4_flops": 13500e12, "hbm_bw": 8.0e12,   "link_bw": 900 * GB, "l2_mb": 126, "fp4_tc": True}),
    ("GB200",  {"name": "GB200",  "flops": 2500e12, "fp8_flops": 5000e12, "fp4_flops": 10000e12, "hbm_bw": 8.0e12,   "link_bw": 900 * GB, "l2_mb": 126, "fp4_tc": True}),
    ("B200",   {"name": "B200",   "flops": 2250e12, "fp8_flops": 4500e12, "fp4_flops": 9000e12,  "hbm_bw": 8.0e12,   "link_bw": 900 * GB, "l2_mb": 126, "fp4_tc": True}),
    ("A100",   {"name": "A100",   "flops": 312e12,  "fp8_flops": None,    "fp4_flops": None,     "hbm_bw": 2.0e12,   "link_bw": 300 * GB, "l2_mb": 40,  "fp4_tc": False}),
    ("L40S",   {"name": "L40S",   "flops": 362e12,  "fp8_flops": 733e12,  "fp4_flops": None,     "hbm_bw": 0.864e12, "link_bw": 32 * GB,  "l2_mb": 96,  "fp4_tc": False}),
    ("4090",   {"name": "RTX4090","flops": 165e12,  "fp8_flops": 330e12,  "fp4_flops": None,     "hbm_bw": 1.008e12, "link_bw": 32 * GB,  "l2_mb": 72,  "fp4_tc": False}),
    ("MI355X", {"name": "MI355X", "flops": 2500e12, "fp8_flops": 5000e12, "fp4_flops": 10000e12, "hbm_bw": 8.0e12,   "link_bw": 538 * GB, "l2_mb": 256, "fp4_tc": True}),
    ("MI325X", {"name": "MI325X", "flops": 1307e12, "fp8_flops": 2614e12, "fp4_flops": None,     "hbm_bw": 6.0e12,   "link_bw": 448 * GB, "l2_mb": 256, "fp4_tc": False}),
    ("MI300X", {"name": "MI300X", "flops": 1307e12, "fp8_flops": 2614e12, "fp4_flops": None,     "hbm_bw": 5.3e12,   "link_bw": 448 * GB, "l2_mb": 256, "fp4_tc": False}),
]

M_BUCKETS = [1, 8, 32, 128, 512, 4096]
MOE_TOKENS = [1, 8, 32, 128, 512, 2048]
COMM_BYTES = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024,
              1 * MIB, 4 * MIB, 16 * MIB, 64 * MIB, 256 * MIB]


def _fmt_size(b: int) -> str:
    if b >= GIB and b % GIB == 0:
        return f"{b // GIB} GiB"
    if b >= MIB and b % MIB == 0:
        return f"{b // MIB} MiB"
    if b >= 1024 and b % 1024 == 0:
        return f"{b // 1024} KiB"
    return f"{b} B"


def _row_bytes(r: dict) -> int:
    if "bytes" in r:
        return int(r["bytes"])
    if "size_mib" in r:
        return int(round(float(r["size_mib"]) * MIB))
    num_s, _, unit = str(r.get("size", "0")).strip().partition(" ")
    mult = {"B": 1, "KIB": 1024, "KB": 1000, "MIB": MIB, "MB": 10**6, "GIB": GIB, "GB": 10**9}.get(unit.upper(), 1)
    return int(round(float(num_s) * mult))


def _topo_hbm_bytes(t: dict) -> float:
    return float(t["hbm_gb"]) * GB if "hbm_gb" in t else float(t.get("hbm_bytes", 80 * GB))


def _topo_free_bytes(t: dict) -> int:
    return int(float(t["free_hbm_gb"]) * GB) if "free_hbm_gb" in t else int(t.get("free_hbm_bytes", 0))


def _topo_l2_bytes(t: dict) -> int:
    return int(float(t["l2_cache_mib"]) * MIB) if "l2_cache_mib" in t else int(t.get("l2_cache_bytes", 50 * MIB))


def _match_spec(device_name: str) -> dict | None:
    up = device_name.upper()
    for key, spec in _SPEC_DB:
        if key in up:
            return spec
    return None


def _host_info(include_hostname: bool = False) -> dict:
    cpus = os.cpu_count() or 1
    ram_total, ram_avail = 0, 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        text = meminfo.read_text(errors="ignore")
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                ram_total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                ram_avail = int(line.split()[1]) * 1024
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
            ram_total = int(out)
        except Exception:
            pass
    info: dict = {
        "os": platform.system(),
        "host_cpus": cpus,
        "host_ram_gb": round(ram_total / GB, 1) if ram_total else None,
        "host_ram_avail_gb": round(ram_avail / GB, 1) if ram_avail else None,
    }
    if include_hostname:
        info["hostname"] = platform.node()
    return info


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


# --------------------------------------------------------------------------- Profile model
@dataclass
class HardwareProfile:
    schema_version: int = 1
    timestamp: str = ""
    is_mock: bool = False
    suite: str = "deep"
    topology: dict = field(default_factory=dict)
    bandwidth: dict = field(default_factory=dict)
    gemm: dict = field(default_factory=dict)
    moe: dict = field(default_factory=dict)
    comm: dict = field(default_factory=dict)
    overhead: dict = field(default_factory=dict)
    calibrated_chip: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "is_mock": self.is_mock,
            "suite": self.suite,
            "topology": self.topology,
            "bandwidth": self.bandwidth,
            "gemm": self.gemm,
            "moe": self.moe,
            "comm": self.comm,
            "overhead": self.overhead,
            "calibrated_chip": self.calibrated_chip,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=out.parent, delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
            try:
                json.dump(self.to_dict(), f, indent=2, allow_nan=False)
                f.write("\n")
                f.close()
                tmp.replace(out)
            finally:
                tmp.unlink(missing_ok=True)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            timestamp=str(data.get("timestamp", "")),
            is_mock=bool(data.get("is_mock", False)),
            suite=str(data.get("suite", "deep")),
            topology=dict(data.get("topology") or {}),
            bandwidth=dict(data.get("bandwidth") or {}),
            gemm=dict(data.get("gemm") or {}),
            moe=dict(data.get("moe") or {}),
            comm=dict(data.get("comm") or {}),
            overhead=dict(data.get("overhead") or {}),
            calibrated_chip=dict(data.get("calibrated_chip") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HardwareProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_chip_spec(self) -> ChipSpec:
        """Build a calibrated `ChipSpec` (with `EmpiricalCurves`) for `kiln costmodel`."""
        cc = self.calibrated_chip
        dtypes = self.gemm.get("dtypes", {})

        def _curve(dtype_key: str) -> list[tuple[int, float]]:
            info = dtypes.get(dtype_key) or {}
            if not info.get("supported"):
                return []
            return sorted((int(r["m"]), float(r["tflops"])) for r in info.get("by_m", []))

        moe_pts: list[tuple[int, float]] = []
        if self.moe.get("supported"):
            moe_pts = sorted(
                (int(r["tokens"]), float(r["roofline_eff"]))
                for r in self.moe.get("by_tokens", [])
            )

        ar_by_k: dict[int, list[tuple[int, float]]] = {}
        a2a_by_k: dict[int, list[tuple[int, float]]] = {}
        for k_str, g in (self.comm.get("groups") or {}).items():
            k = int(k_str)
            if g.get("all_reduce"):
                ar_by_k[k] = sorted((_row_bytes(r), float(r["lat_us"]) * 1e-6) for r in g["all_reduce"])
            if g.get("all_to_all"):
                a2a_by_k[k] = sorted((_row_bytes(r), float(r["lat_us"]) * 1e-6) for r in g["all_to_all"])

        fp4_info = dtypes.get("fp4") or {}
        emp = EmpiricalCurves(
            bf16_tflops_by_m=_curve("bf16"),
            fp8_tflops_by_m=_curve("fp8_e4m3"),
            fp4_tflops_by_m=_curve("fp4"),
            fp4_native_tc=bool(fp4_info.get("native_tc", False)),
            moe_roofline_eff_by_tokens=moe_pts,
            allreduce_by_k=ar_by_k,
            alltoall_by_k=a2a_by_k,
            cudagraph_per_kernel_s=float(self.overhead.get("cudagraph_per_kernel_us", 0.0)) * 1e-6,
        )
        spec_map = cc.get("spec_tflops") if isinstance(cc.get("spec_tflops"), dict) else {}
        peak_map = cc.get("peak_tflops") if isinstance(cc.get("peak_tflops"), dict) else {}
        eff_map = cc.get("flops_eff") if isinstance(cc.get("flops_eff"), dict) else {}

        if spec_map.get("bf16") is not None:
            flops = float(spec_map["bf16"]) * 1e12
        elif "tflops" in cc:
            flops = float(cc["tflops"]) * 1e12
        else:
            flops = float(cc["flops"])

        flops_eff = float(eff_map["bf16"]) if eff_map.get("bf16") is not None else float(cc["flops_eff"])
        if peak_map.get("fp8") and peak_map.get("bf16"):
            fp8_scale = float(peak_map["fp8"]) / float(peak_map["bf16"])
        else:
            fp8_scale = float(cc.get("fp8_scale", 2.0))

        hbm_bw = float(cc["hbm_bw_gbps"]) * 1e9 if "hbm_bw_gbps" in cc else float(cc["hbm_bw"])
        hbm_bytes = float(cc["hbm_gb"]) * GB if "hbm_gb" in cc else float(cc["hbm_bytes"])
        link_bw = float(cc["link_bw_gbps"]) * 1e9 if "link_bw_gbps" in cc else float(cc["link_bw"])
        link_lat = float(cc["link_latency_us"]) * 1e-6 if "link_latency_us" in cc else float(cc["link_latency_s"])
        return ChipSpec(
            name=str(cc.get("name", self.topology.get("device_name", "probed-gpu"))),
            flops=flops,
            hbm_bw=hbm_bw,
            hbm_bytes=hbm_bytes,
            link_bw=link_bw,
            link_latency=link_lat,
            flops_eff=flops_eff,
            bw_eff=float(cc["bw_eff"]),
            link_eff=float(cc["link_eff"]),
            max_chips=int(cc["max_chips"]),
            fp8_scale=fp8_scale,
            empirical=emp,
        )

    def to_markdown(self) -> str:
        """Render a human- and agent-friendly Hardware & Roofline reference."""
        t = self.topology
        bw = self.bandwidth
        cc = self.calibrated_chip
        dtypes = self.gemm.get("dtypes", {})
        lines: list[str] = []

        tag = " (MOCK — synthetic fallback, NOT measured hardware)" if self.is_mock else " (measured)"
        lines.append(f"# Hardware & Roofline Profile{tag}")
        lines.append("")
        if self.is_mock:
            lines.append("> **Note:** Generated in CPU mock mode. Re-run `kiln probe` on the target GPU node before `kiln serve` for empirical calibration.")
            lines.append("")

        # 1. Topology
        ngpu = t.get("num_gpus", 1)
        hbm_gb = _topo_hbm_bytes(t) / GB
        l2_mb = _topo_l2_bytes(t) / MIB
        lines.append("## 1. Node & GPU Topology")
        lines.append(
            f"- **GPU:** {ngpu}× `{t.get('device_name', 'unknown')}` "
            f"(`{t.get('arch', '?')}`, {t.get('sm_count', '?')} SMs, "
            f"{l2_mb:.1f} MiB L2, {hbm_gb:.1f} GB HBM/GPU)"
        )
        lines.append(
            f"- **Interconnect:** `{t.get('interconnect', 'unknown')}` "
            f"(high-speed clique = {t.get('nvlink_clique_size', 1)} GPUs)"
        )
        if t.get("sm_clock_mhz"):
            lines.append(
                f"- **Clocks / Power:** SM {t['sm_clock_mhz']} MHz (max {t.get('sm_max_clock_mhz', '?')} MHz), "
                f"power {t.get('power_draw_w', '?')} / {t.get('power_limit_w', '?')} W"
            )
        if t.get("host_cpus"):
            ram = f", {t['host_ram_gb']} GB RAM" if t.get("host_ram_gb") else ""
            host_prefix = f"`{t['hostname']}` " if t.get("hostname") else ""
            lines.append(f"- **Host:** {host_prefix}{t['host_cpus']} CPUs{ram}")
        for w in t.get("warnings", ()):
            lines.append(f"- **Warning:** {w}")
        lines.append("")

        # 2. Memory & Roofline Ceilings
        hbm_ach = bw.get("hbm_peak_achieved_gbps", 0.0)
        lines.append("## 2. Memory Bandwidth & Roofline Ridge Points")
        lines.append(
            f"- **HBM Streaming:** `{hbm_ach:,.1f} GB/s` achieved "
            f"(copy `{bw.get('hbm_copy_gbps', 0):,.1f} GB/s`, triad `{bw.get('hbm_triad_gbps', 0):,.1f} GB/s`, "
            f"`bw_eff = {cc.get('bw_eff', 0):.2f}`)"
        )
        lines.append(
            f"- **L2 Resident Copy:** `{bw.get('l2_gbps', 0):,.1f} GB/s`  |  "
            f"**Pinned Host↔GPU (PCIe/C2C):** H2D `{bw.get('h2d_gbps', 0):,.1f} GB/s`, "
            f"D2H `{bw.get('d2h_gbps', 0):,.1f} GB/s`"
        )
        lines.append("")
        lines.append("| Precision | Mode | Peak Achieved (TFLOP/s) | Ridge Point (FLOP/Byte) | GEMM Compute-Bound Threshold |")
        lines.append("|---|---|---:|---:|---|")
        bytes_per_w = {"bf16": 2.0, "fp8_e4m3": 1.0, "fp4": 0.5625}
        for key, label in (("bf16", "BF16"), ("fp8_e4m3", "FP8 (e4m3)"), ("fp4", "FP4 / MXFP4")):
            info = dtypes.get(key) or {}
            if not info.get("supported"):
                reason = info.get("reason", "not supported")
                lines.append(f"| {label} | — | — | — | _{reason}_ |")
                continue
            peak = float(info["peak_tflops"])
            ridge = float(info["ridge_flop_per_byte"])
            m_thresh = max(1, int(round(ridge * bytes_per_w[key] / 2.0)))
            mode = info.get("mode", "tensor_core")
            lines.append(f"| {label} | `{mode}` | {peak:,.1f} | {ridge:,.1f} | `M >= ~{m_thresh}` tokens |")
        lines.append("")

        # 3. Cold-L2 Dense GEMM by M
        nk = self.gemm.get("shape_nk", [0, 0])
        lines.append(f"## 3. Cold-L2 Dense GEMM Scaling (`N={nk[0]}, K={nk[1]}`)")
        supported_keys = [k for k in ("bf16", "fp8_e4m3", "fp4") if (dtypes.get(k) or {}).get("supported")]
        if supported_keys:
            hdr = "| M (tokens) | " + " | ".join(f"{k} TFLOP/s (GB/s)" for k in supported_keys) + " |"
            sep = "|---:|" + "|".join("---:" for _ in supported_keys) + "|"
            lines.append(hdr)
            lines.append(sep)
            by_m_maps = {k: {int(r["m"]): r for r in dtypes[k]["by_m"]} for k in supported_keys}
            all_ms = sorted({m for mp in by_m_maps.values() for m in mp})
            for m in all_ms:
                cells = []
                for k in supported_keys:
                    r = by_m_maps[k].get(m)
                    cells.append(f"{r['tflops']:,.1f} ({r['gbps']:,.0f})" if r else "—")
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")

        # 4. Grouped MoE
        if self.moe.get("supported"):
            lines.append(
                f"## 4. Grouped MoE GEMM (`{self.moe.get('backend')}`, "
                f"E={self.moe.get('num_experts')}, top_k={self.moe.get('top_k')}, "
                f"hidden={self.moe.get('hidden')}, inter={self.moe.get('inter')})"
            )
            lines.append("| Batch Tokens | Avg Tokens/Expert | Touched Experts | Achieved TFLOP/s | Streamed GB/s | Roofline Eff |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for r in self.moe.get("by_tokens", ()):
                lines.append(
                    f"| {r['tokens']} | {r['avg_m_per_expert']:.2f} | {r['touched_experts']:.1f} | "
                    f"{r['tflops']:,.1f} | {r['gbps']:,.0f} | {100.0 * r['roofline_eff']:.1f}% |"
                )
            lines.append("")

        # 5. Multi-GPU Collectives
        groups = self.comm.get("groups") or {}
        if groups:
            lines.append(f"## 5. Multi-GPU Collectives (`{self.comm.get('backend', 'nccl')}`)")
            for k_str in sorted(groups, key=int):
                g = groups[k_str]
                lines.append(
                    f"### Group size `k={k_str}` "
                    f"(floor latency ≈ `{g.get('link_latency_us', 0):.1f} µs`, "
                    f"peak busbw ≈ `{g.get('peak_busbw_gbps', 0):,.1f} GB/s`)"
                )
                lines.append("| Message Size | AllReduce Lat (µs) | AllReduce BusBW (GB/s) | AllToAll Lat (µs) | AllToAll BusBW (GB/s) |")
                lines.append("|---:|---:|---:|---:|---:|")
                ar_map = {_row_bytes(r): r for r in g.get("all_reduce", ())}
                a2a_map = {_row_bytes(r): r for r in g.get("all_to_all", ())}
                for b in sorted(set(ar_map) | set(a2a_map)):
                    label = _fmt_size(b)
                    ar = ar_map.get(b)
                    a2a = a2a_map.get(b)
                    ar_c = f"{ar['lat_us']:,.1f} | {ar['busbw_gbps']:,.1f}" if ar else "— | —"
                    a2a_c = f"{a2a['lat_us']:,.1f} | {a2a['busbw_gbps']:,.1f}" if a2a else "— | —"
                    lines.append(f"| {label} | {ar_c} | {a2a_c} |")
                lines.append("")
        elif self.comm.get("error"):
            lines.append(f"## 5. Multi-GPU Collectives\n- Skipped / failed: `{self.comm['error']}`\n")

        # 6. Launch & Graph Overhead
        if self.overhead:
            ov = self.overhead
            lines.append("## 6. Kernel Launch & Synchronization Overhead")
            lines.append(
                f"- **Eager CPU launch:** `{ov.get('eager_launch_us', 0):.2f} µs/op` "
                f"(GPU back-to-back `{ov.get('eager_gpu_step_us', 0):.2f} µs/op`)"
            )
            lines.append(
                f"- **CUDA Graph replay:** `{ov.get('cudagraph_fixed_launch_us', 0):.2f} µs` fixed launch + "
                f"`{ov.get('cudagraph_per_kernel_us', 0):.2f} µs/kernel` node transition"
            )
            lines.append(
                f"- **Host `.item()` / sync round-trip:** `{ov.get('host_sync_us', 0):.1f} µs` "
                "(keep hot path 100% CUDA-graph safe; zero host syncs!)"
            )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_prompt_summary(self) -> str:
        """Compact ~10-line Roofline block appended to `PROMPT.md` for coding agents."""
        t = self.topology
        bw = self.bandwidth
        dtypes = self.gemm.get("dtypes", {})
        ov = self.overhead
        tag = " [MOCK]" if self.is_mock else ""
        parts = [f"\n## Hardware & Roofline Reference{tag}"]
        parts.append(
            f"- GPU: {t.get('num_gpus', 1)}× {t.get('device_name', '?')} ({t.get('arch', '?')}, "
            f"{t.get('sm_count', '?')} SMs, {_topo_l2_bytes(t) / MIB:.0f} MiB L2, "
            f"HBM achieved `{bw.get('hbm_peak_achieved_gbps', 0):,.0f} GB/s`, "
            f"L2 `{bw.get('l2_gbps', 0):,.0f} GB/s`, PCIe H2D `{bw.get('h2d_gbps', 0):,.0f} GB/s`)."
        )
        roof_bits = []
        for k, name in (("bf16", "BF16"), ("fp8_e4m3", "FP8"), ("fp4", "FP4")):
            d = dtypes.get(k) or {}
            if d.get("supported"):
                roof_bits.append(f"{name} `{d['peak_tflops']:,.0f} TFLOP/s` (ridge `{d['ridge_flop_per_byte']:.0f} FLOP/B`)")
        if roof_bits:
            parts.append("- Compute ceilings: " + "; ".join(roof_bits) + ".")
        if self.moe.get("supported") and self.moe.get("by_tokens"):
            rows = self.moe["by_tokens"]
            lo, hi = rows[0], rows[-1]
            parts.append(
                f"- Grouped MoE (`{self.moe.get('backend')}`): roofline eff "
                f"`{100 * lo['roofline_eff']:.0f}%` @ {lo['tokens']} tok -> "
                f"`{100 * hi['roofline_eff']:.0f}%` @ {hi['tokens']} tok."
            )
        groups = self.comm.get("groups") or {}
        if groups:
            k_max = str(max(map(int, groups)))
            g = groups[k_max]
            parts.append(
                f"- Collectives (k={k_max} `{self.comm.get('backend', 'nccl')}`): "
                f"floor latency `{g.get('link_latency_us', 0):.1f} µs`, "
                f"peak busbw `{g.get('peak_busbw_gbps', 0):,.0f} GB/s`."
            )
        if ov:
            parts.append(
                f"- Launch overhead: eager `{ov.get('eager_launch_us', 0):.2f} µs/op`, "
                f"CUDA Graph `{ov.get('cudagraph_per_kernel_us', 0):.2f} µs/kernel` "
                f"(+{ov.get('cudagraph_fixed_launch_us', 0):.1f} µs launch), "
                f"host sync `{ov.get('host_sync_us', 0):.1f} µs`."
            )
        parts.append("- Full M-bucket GEMM, MoE, and collective curves are in `HARDWARE.md` and `hw_profile.json` in your workspace.")
        return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- Calibration builder
def _build_calibrated_chip(topology: dict, bandwidth: dict, gemm: dict, comm: dict) -> dict:
    dev_name = topology.get("device_name", "unknown")
    spec = _match_spec(dev_name)
    dtypes = gemm.get("dtypes", {})
    bf16_tflops = float((dtypes.get("bf16") or {}).get("peak_tflops", 0.0)) or None
    fp8_info = dtypes.get("fp8_e4m3") or {}
    fp8_tflops = float(fp8_info["peak_tflops"]) if fp8_info.get("supported") else None
    fp4_info = dtypes.get("fp4") or {}
    fp4_tflops = float(fp4_info["peak_tflops"]) if fp4_info.get("supported") else None
    hbm_ach = float(bandwidth.get("hbm_peak_achieved_gbps", 0.0)) * 1e9

    if spec:
        bf16_spec = float(spec["flops"]) / 1e12
        fp8_spec = float(spec["fp8_flops"]) / 1e12 if spec.get("fp8_flops") else None
        fp4_spec = float(spec["fp4_flops"]) / 1e12 if spec.get("fp4_flops") else None
        hbm_spec = float(spec["hbm_bw"])
        bw_eff = min(1.0, hbm_ach / hbm_spec) if hbm_ach > 0 else 0.8
        link_spec = float(spec["link_bw"])
        short_name = spec["name"]
    else:
        bf16_spec = bf16_tflops if bf16_tflops else 100.0
        fp8_spec = fp8_tflops
        fp4_spec = fp4_tflops if fp4_info.get("native_tc") else None
        hbm_spec = hbm_ach if hbm_ach > 0 else 1.0e12
        bw_eff = 1.0
        link_spec = 300 * GB
        short_name = dev_name

    def _eff(ach: float | None, sp: float | None) -> float | None:
        if ach is None or sp is None or sp <= 0:
            return None
        return round(min(1.0, ach / sp), 4)

    groups = comm.get("groups") or {}
    link_lat_s = 5e-6
    link_eff = 0.7
    peak_busbw_gbps = None
    if groups:
        k_max = str(max(map(int, groups)))
        g = groups[k_max]
        link_lat_s = max(0.5e-6, float(g.get("link_latency_us", 5.0)) * 1e-6)
        peak_busbw = float(g.get("peak_busbw_gbps", 0.0)) * 1e9
        if peak_busbw > 0:
            peak_busbw_gbps = round(peak_busbw / 1e9, 2)
            if spec:
                link_eff = min(1.0, peak_busbw / link_spec)
            else:
                link_spec, link_eff = peak_busbw, 1.0

    max_chips = max(int(topology.get("nvlink_clique_size", 1)), int(topology.get("num_gpus", 1)))
    return {
        "name": short_name,
        "peak_tflops": {
            "bf16": round(bf16_tflops, 2) if bf16_tflops is not None else None,
            "fp8": round(fp8_tflops, 2) if fp8_tflops is not None else None,
            "fp4": round(fp4_tflops, 2) if fp4_tflops is not None else None,
        },
        "spec_tflops": {
            "bf16": round(bf16_spec, 2),
            "fp8": round(fp8_spec, 2) if fp8_spec is not None else None,
            "fp4": round(fp4_spec, 2) if fp4_spec is not None else None,
        },
        "flops_eff": {
            "bf16": _eff(bf16_tflops, bf16_spec) or 0.6,
            "fp8": _eff(fp8_tflops, fp8_spec),
            "fp4": _eff(fp4_tflops, fp4_spec) if fp4_info.get("native_tc") else None,
        },
        "hbm_bw_gbps": round(hbm_spec / 1e9, 2),
        "hbm_achieved_gbps": round(hbm_ach / 1e9, 2) if hbm_ach > 0 else None,
        "bw_eff": round(bw_eff, 4),
        "hbm_gb": round(_topo_hbm_bytes(topology) / GB, 2),
        "link_bw_gbps": round(link_spec / 1e9, 2),
        "link_bidir_bw_gbps": round(2.0 * link_spec / 1e9, 2),
        "link_achieved_gbps": peak_busbw_gbps,
        "link_eff": round(link_eff, 4),
        "link_latency_us": round(link_lat_s * 1e6, 2),
        "max_chips": max_chips,
    }


# --------------------------------------------------------------------------- CUDA benchmarks
def _cuda_bench(fn, warmup: int = 5, rep: int = 15, min_total_ms: float = 20.0) -> float:
    """Return median per-call GPU execution time in seconds via CUDA Events."""
    import torch

    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()

    s0 = torch.cuda.Event(enable_timing=True)
    e0 = torch.cuda.Event(enable_timing=True)
    s0.record()
    fn(warmup)
    e0.record()
    torch.cuda.synchronize()
    one_ms = max(float(s0.elapsed_time(e0)), 1e-4)
    inner = max(1, min(40, int(1.0 / one_ms)))
    n_samples = max(5, min(rep, int(max(min_total_ms, 5.0) / (one_ms * inner)) + 1))

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(n_samples)
    ]
    step = warmup + 1
    for st, en in events:
        st.record()
        for _ in range(inner):
            fn(step)
            step += 1
        en.record()
    torch.cuda.synchronize()
    samples = [(float(st.elapsed_time(en)) / 1000.0) / inner for st, en in events]
    return statistics.median(samples)


def _parse_nvlink_topology(num_gpus: int) -> tuple[str, int]:
    """Inspect `nvidia-smi topo -m` (or peer-access matrix) for high-speed clique size."""
    try:
        # Nodes with dozens of RDMA NICs can take 5-8s to render `topo -m` and emit ANSI escapes.
        raw = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=15, stderr=subprocess.DEVNULL)
        out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
        nv_peers = 0
        nv_tag = ""
        has_xgmi = "XGMI" in out
        for line in out.splitlines():
            if line.startswith("GPU0") and (len(line) == 4 or line[4].isspace()):
                tokens = line.split()[1 : num_gpus + 1]
                nv_toks = [tok for tok in tokens if tok.startswith(("NV", "XGMI"))]
                nv_peers = 1 + len(nv_toks)
                if nv_toks:
                    nv_tag = nv_toks[0]
                break
        if nv_peers > 1:
            kind = "XGMI" if has_xgmi else (f"NVLink/NVSwitch ({nv_tag})" if nv_tag else "NVLink/NVSwitch")
            return kind, nv_peers
    except Exception:
        pass

    try:
        import torch
        peers = 1 + sum(1 for j in range(1, num_gpus) if torch.cuda.can_device_access_peer(0, j))
        return ("P2P/Link" if peers > 1 else ("Single-GPU" if num_gpus == 1 else "PCIe/SYS")), peers
    except Exception:
        return "unknown", max(1, num_gpus)


def _probe_topology_cuda() -> dict:
    import torch

    num_gpus = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    l2_b = int(getattr(props, "L2_cache_size", 0) or (50 * MIB))
    arch = getattr(props, "gcnArchName", "") if getattr(torch.version, "hip", None) else f"sm_{props.major}{props.minor}"

    # Brief compute pulse so clock queries reflect boosted P0 state instead of idle P8.
    x = torch.randn((2048, 2048), device="cuda:0", dtype=torch.bfloat16)
    torch.mm(x, x)
    torch.cuda.synchronize()
    del x

    warnings: list[str] = []
    sm_cur, sm_max, pwr_draw, pwr_lim = None, None, None, None
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=clocks.current.sm,clocks.max.sm,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3, stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        parts = [p.strip() for p in q.split(",")]
        sm_cur = int(float(parts[0]))
        sm_max = int(float(parts[1]))
        pwr_draw = round(float(parts[2]), 1)
        pwr_lim = round(float(parts[3]), 1)
        if sm_max > 0 and sm_cur < 0.7 * sm_max:
            warnings.append(f"SM clock ({sm_cur} MHz) is well below max boost ({sm_max} MHz); check clock lock / thermal throttling.")
    except Exception:
        pass

    if free_b < 0.5 * total_b:
        warnings.append(
            f"Only {free_b / GB:.1f} / {total_b / GB:.1f} GB HBM is free; another process may be occupying the GPU."
        )

    interconnect, clique = _parse_nvlink_topology(num_gpus)
    info = {
        "device_name": props.name,
        "arch": arch,
        "major": int(props.major),
        "minor": int(props.minor),
        "sm_count": int(props.multi_processor_count),
        "num_gpus": int(num_gpus),
        "hbm_gb": round(total_b / GB, 2),
        "free_hbm_gb": round(free_b / GB, 2),
        "l2_cache_mib": round(l2_b / MIB, 1),
        "interconnect": interconnect,
        "nvlink_clique_size": int(clique),
        "sm_clock_mhz": sm_cur,
        "sm_max_clock_mhz": sm_max,
        "power_draw_w": pwr_draw,
        "power_limit_w": pwr_lim,
        "warnings": warnings,
        **_host_info(),
    }
    return info


def _probe_bandwidth_cuda(l2_bytes: int, free_bytes: int) -> dict:
    import torch

    dev = torch.device("cuda:0")
    # 1. L2-resident triad inside a CUDA Graph (3 buffers <= ~38% of L2 avoids partition spill & launch gap)
    l2_each = max(1 * MIB, min(l2_bytes // 8, 16 * MIB))
    n_l2 = l2_each // 4
    a_l2 = torch.ones(n_l2, device=dev, dtype=torch.float32)
    b_l2 = torch.ones(n_l2, device=dev, dtype=torch.float32)
    c_l2 = torch.empty_like(a_l2)
    try:
        s_l2 = torch.cuda.Stream(device=dev)
        s_l2.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s_l2):
            for _ in range(5):
                torch.add(a_l2, b_l2, out=c_l2)
        torch.cuda.synchronize()
        g_l2 = torch.cuda.CUDAGraph()
        k_l2 = 20
        with torch.cuda.graph(g_l2, stream=s_l2):
            for _ in range(k_l2):
                torch.add(a_l2, b_l2, out=c_l2)
        torch.cuda.synchronize()
        dt_l2 = _cuda_bench(lambda _: g_l2.replay(), warmup=10, rep=15, min_total_ms=15.0) / k_l2
        l2_gbps = (3.0 * l2_each) / dt_l2 / 1e9
        del g_l2
    except Exception:
        dt_l2 = _cuda_bench(lambda _: torch.add(a_l2, b_l2, out=c_l2))
        l2_gbps = (3.0 * l2_each) / dt_l2 / 1e9
    del a_l2, b_l2, c_l2

    # 2. HBM streaming copy & triad across buffer sizes well beyond L2
    max_buf = max(64 * MIB, min(1024 * MIB, int(free_bytes // 6)))
    sizes = [s for s in (64 * MIB, 256 * MIB, 512 * MIB, 1024 * MIB) if s <= max_buf] or [max_buf]
    curve = []
    for sz in sizes:
        n = sz // 4  # float32
        a = torch.ones(n, device=dev, dtype=torch.float32)
        b = torch.ones(n, device=dev, dtype=torch.float32)
        c = torch.empty_like(a)
        dt_copy = _cuda_bench(lambda _: c.copy_(a), min_total_ms=15.0)
        dt_triad = _cuda_bench(lambda _: torch.add(a, b, out=c), min_total_ms=15.0)
        curve.append({
            "size_mib": round(sz / MIB, 1),
            "copy_gbps": round((2.0 * sz) / dt_copy / 1e9, 2),
            "triad_gbps": round((3.0 * sz) / dt_triad / 1e9, 2),
        })
        del a, b, c
    torch.cuda.empty_cache()

    copy_peak = max(r["copy_gbps"] for r in curve)
    triad_peak = max(r["triad_gbps"] for r in curve)
    hbm_peak = max(copy_peak, triad_peak)

    # 3. Pinned Host <-> Device PCIe / C2C
    pcie_sz = min(256 * MIB, max(16 * MIB, int(free_bytes // 10)))
    h2d_gbps, d2h_gbps = 0.0, 0.0
    try:
        host_buf = torch.ones(pcie_sz // 2, dtype=torch.bfloat16, pin_memory=True)
        dev_buf = torch.empty(pcie_sz // 2, dtype=torch.bfloat16, device=dev)
        dt_h2d = _cuda_bench(lambda _: dev_buf.copy_(host_buf, non_blocking=True), warmup=3, rep=8, min_total_ms=15.0)
        dt_d2h = _cuda_bench(lambda _: host_buf.copy_(dev_buf, non_blocking=True), warmup=3, rep=8, min_total_ms=15.0)
        h2d_gbps = pcie_sz / dt_h2d / 1e9
        d2h_gbps = pcie_sz / dt_d2h / 1e9
        del host_buf, dev_buf
    except Exception:
        pass
    torch.cuda.empty_cache()

    return {
        "hbm_copy_gbps": round(copy_peak, 2),
        "hbm_triad_gbps": round(triad_peak, 2),
        "hbm_peak_achieved_gbps": round(hbm_peak, 2),
        "l2_gbps": round(l2_gbps, 2),
        "h2d_gbps": round(h2d_gbps, 2),
        "d2h_gbps": round(d2h_gbps, 2),
        "hbm_curve": curve,
    }


def _ring_count(weight_bytes: int, l2_bytes: int, free_bytes: int) -> int:
    target = min(max(256 * MIB, 3 * l2_bytes), max(weight_bytes, int(free_bytes // 8)))
    return max(1, min(16, int(target // max(weight_bytes, 1))))


def _probe_fp4_gemm_cuda(N: int, K: int, l2_bytes: int, free_bytes: int, hbm_gbps: float, major: int) -> dict:
    """Probe FP4 / MXFP4 GEMM via PyTorch cuBLASLt NVFP4, vLLM CUTLASS FP4, or Triton `dot_scaled`."""
    import torch

    dev = torch.device("cuda:0")
    w_bytes = int(N * K * 0.5625)  # 4-bit + block scale

    # Path 0: Zero-JIT PyTorch cuBLASLt native NVFP4 (`float4_e2m1fn_x2` + 1x16 `float8_e4m3fn` scales)
    if major >= 10 and hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e4m3fn"):
        try:
            def _nvfp4_scales(rows: int) -> torch.Tensor:
                rp = ((rows + 127) // 128) * 128
                cp = ((K // 16 + 3) // 4) * 4
                return torch.ones((rp * cp,), device=dev, dtype=torch.float8_e4m3fn)

            n_ring = _ring_count(w_bytes, l2_bytes, free_bytes)
            b_ring = [
                torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t()
                for _ in range(n_ring)
            ]
            sb_ring = [_nvfp4_scales(N) for _ in range(n_ring)]
            # Smoke-test at M=128
            a128 = torch.randint(0, 255, (128, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            torch._scaled_mm(a128, b_ring[0], scale_a=_nvfp4_scales(128), scale_b=sb_ring[0], out_dtype=torch.bfloat16)

            rows = []
            for m in M_BUCKETS:
                m_eff = m
                a, sa = None, None
                for align in (1, 16, 128):
                    cand_m = ((m + align - 1) // align) * align
                    a_try = torch.randint(0, 255, (cand_m, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
                    sa_try = _nvfp4_scales(cand_m)
                    try:
                        torch._scaled_mm(a_try, b_ring[0], scale_a=sa_try, scale_b=sb_ring[0], out_dtype=torch.bfloat16)
                        m_eff, a, sa = cand_m, a_try, sa_try
                        break
                    except RuntimeError:
                        continue
                assert a is not None and sa is not None
                dt = _cuda_bench(
                    lambda i: torch._scaled_mm(a, b_ring[i % n_ring], scale_a=sa, scale_b=sb_ring[i % n_ring], out_dtype=torch.bfloat16)
                )
                flops = 2.0 * m * N * K
                moved = w_bytes + m * K * 0.5625 + m * N * 2.0
                rec = {"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)}
                if m_eff > m:
                    rec["padded_m"] = m_eff
                rows.append(rec)
            del b_ring, sb_ring
            peak = max(r["tflops"] for r in rows)
            return {
                "supported": True,
                "native_tc": True,
                "mode": "nvfp4_cublaslt_tc",
                "peak_tflops": peak,
                "ridge_flop_per_byte": round(peak * 1e3 / max(hbm_gbps, 1e-6), 1),
                "by_m": rows,
            }
        except Exception:
            torch.cuda.empty_cache()

    # Path 1: Native Blackwell NVFP4 Tensor Core via vLLM `_custom_ops`
    try:
        from vllm import _custom_ops as ops
        if hasattr(ops, "scaled_fp4_quant") and hasattr(ops, "cutlass_scaled_fp4_mm") and major >= 10:
            n_ring = _ring_count(w_bytes, l2_bytes, free_bytes)
            b_src = torch.randn((N, K), device=dev, dtype=torch.bfloat16)
            gs = torch.ones((1,), device=dev, dtype=torch.float32)
            wq, w_scale = ops.scaled_fp4_quant(b_src, gs)
            ring = [(wq.clone(), w_scale.clone()) for _ in range(n_ring)]
            alpha = torch.ones((1,), device=dev, dtype=torch.float32)
            rows = []
            for m in M_BUCKETS:
                m_eff = max(16, ((m + 15) // 16) * 16)
                a_src = torch.randn((m_eff, K), device=dev, dtype=torch.bfloat16)
                aq, a_scale = ops.scaled_fp4_quant(a_src, gs)
                dt = _cuda_bench(
                    lambda i: ops.cutlass_scaled_fp4_mm(aq, ring[i % n_ring][0], a_scale, ring[i % n_ring][1], alpha, torch.bfloat16)
                )
                flops = 2.0 * m * N * K
                moved = w_bytes + m * K * 0.5625 + m * N * 2
                rows.append({"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)})
            peak = max(r["tflops"] for r in rows)
            return {
                "supported": True,
                "native_tc": True,
                "mode": "nvfp4_cutlass_tc",
                "peak_tflops": peak,
                "ridge_flop_per_byte": round(peak * 1e3 / max(hbm_gbps, 1e-6), 1),
                "by_m": rows,
            }
    except Exception:
        pass

    # Path 2: Triton `tl.dot_scaled` MXFP4 weights (e2m1) x FP8 activations (e4m3)
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _mxfp4_fp8_gemm(
            a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr,
            M, N_dim: tl.constexpr, K_dim: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                offs_kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
                offs_ks = (k0 // 32) + tl.arange(0, BLOCK_K // 32)
                a = tl.load(a_ptr + offs_m[:, None] * K_dim + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
                a_sc = tl.load(as_ptr + offs_m[:, None] * (K_dim // 32) + offs_ks[None, :], mask=offs_m[:, None] < M, other=127)
                b = tl.load(b_ptr + offs_kp[:, None] * N_dim + offs_n[None, :])
                b_sc = tl.load(bs_ptr + offs_n[:, None] * (K_dim // 32) + offs_ks[None, :])
                acc = tl.dot_scaled(a, a_sc, "e4m3", b, b_sc, "e2m1", acc)
            c = acc.to(tl.bfloat16)
            tl.store(c_ptr + offs_m[:, None] * N_dim + offs_n[None, :], c, mask=offs_m[:, None] < M)

        n_ring = _ring_count(w_bytes, l2_bytes, free_bytes)
        b_ring = [torch.randint(0, 255, (K // 2, N), device=dev, dtype=torch.uint8) for _ in range(n_ring)]
        bs_ring = [torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8) for _ in range(n_ring)]
        rows = []
        for m in M_BUCKETS:
            bm = 64 if m <= 64 else 128  # SM90 WGMMA / SM100 MMA require M tile >= 64
            m_pad = ((m + bm - 1) // bm) * bm
            a = torch.ones((m_pad, K), device=dev, dtype=torch.float8_e4m3fn)
            a_sc = torch.full((m_pad, K // 32), 127, device=dev, dtype=torch.uint8)
            out = torch.empty((m_pad, N), device=dev, dtype=torch.bfloat16)
            grid = (m_pad // bm, N // 128)
            dt = _cuda_bench(
                lambda i: _mxfp4_fp8_gemm[grid](
                    a, a_sc, b_ring[i % n_ring], bs_ring[i % n_ring], out,
                    m_pad, N, K, BLOCK_M=bm, BLOCK_N=128, BLOCK_K=128,
                )
            )
            flops = 2.0 * m * N * K
            moved = w_bytes + m * K * 1.0 + m * N * 2.0
            rows.append({"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)})
        peak = max(r["tflops"] for r in rows)
        return {
            "supported": True,
            "native_tc": major >= 10,
            "mode": "mxfp4_w4a8_triton_tc" if major >= 10 else "mxfp4_w4a8_triton",
            "peak_tflops": peak,
            "ridge_flop_per_byte": round(peak * 1e3 / max(hbm_gbps, 1e-6), 1),
            "by_m": rows,
        }
    except Exception as exc:
        reason = f"no native FP4 TC (sm_{major}x) / kernel unavailable ({type(exc).__name__})"
        return {"supported": False, "native_tc": False, "reason": reason, "by_m": []}


def _probe_gemm_cuda(l2_bytes: int, free_bytes: int, hbm_gbps: float, major: int, deep: bool) -> dict:
    import torch

    dev = torch.device("cuda:0")
    N, K = (8192, 4096) if free_bytes >= 4 * GIB else (4096, 2048)
    dtypes: dict[str, dict] = {}

    # 1. BF16 Dense GEMM
    w_bf16 = N * K * 2
    n_ring = _ring_count(w_bf16, l2_bytes, free_bytes)
    b_ring = [torch.randn((K, N), device=dev, dtype=torch.bfloat16) for _ in range(n_ring)]
    bf16_rows = []
    for m in M_BUCKETS:
        a = torch.randn((m, K), device=dev, dtype=torch.bfloat16)
        dt = _cuda_bench(lambda i: torch.mm(a, b_ring[i % n_ring]))
        flops = 2.0 * m * N * K
        moved = w_bf16 + m * (K + N) * 2.0
        bf16_rows.append({"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)})
    del b_ring
    torch.cuda.empty_cache()
    bf16_peak = max(r["tflops"] for r in bf16_rows)
    dtypes["bf16"] = {
        "supported": True,
        "mode": "tensor_core",
        "peak_tflops": bf16_peak,
        "ridge_flop_per_byte": round(bf16_peak * 1e3 / max(hbm_gbps, 1e-6), 1),
        "by_m": bf16_rows,
    }

    # 2. FP8 (e4m3) Dense GEMM
    fp8_candidates = [getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e4m3fnuz", None)]
    fp8_done = False
    for fp8_dt in [d for d in fp8_candidates if d is not None]:
        try:
            w_fp8 = N * K
            n_ring = _ring_count(w_fp8, l2_bytes, free_bytes)
            # Column-major weight view `(K, N)` required by cuBLASLt `_scaled_mm`.
            b_ring = [torch.randn((N, K), device=dev, dtype=torch.bfloat16).to(fp8_dt).t() for _ in range(n_ring)]
            s1 = torch.ones((1,), device=dev, dtype=torch.float32)
            # Smoke-test at M=16
            a16 = torch.randn((16, K), device=dev, dtype=torch.bfloat16).to(fp8_dt)
            torch._scaled_mm(a16, b_ring[0], scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16)

            fp8_rows = []
            for m in M_BUCKETS:
                a = torch.randn((m, K), device=dev, dtype=torch.bfloat16).to(fp8_dt)
                padded_m = None
                try:
                    torch._scaled_mm(a, b_ring[0], scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16)
                except RuntimeError:
                    padded_m = max(16, ((m + 15) // 16) * 16)
                    a = torch.randn((padded_m, K), device=dev, dtype=torch.bfloat16).to(fp8_dt)
                dt = _cuda_bench(
                    lambda i: torch._scaled_mm(a, b_ring[i % n_ring], scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16)
                )
                flops = 2.0 * m * N * K
                moved = w_fp8 + m * K * 1.0 + m * N * 2.0
                rec = {"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)}
                if padded_m is not None:
                    rec["padded_m"] = padded_m
                fp8_rows.append(rec)
            del b_ring
            torch.cuda.empty_cache()
            fp8_peak = max(r["tflops"] for r in fp8_rows)
            dtypes["fp8_e4m3"] = {
                "supported": True,
                "mode": str(fp8_dt).replace("torch.", ""),
                "peak_tflops": fp8_peak,
                "ridge_flop_per_byte": round(fp8_peak * 1e3 / max(hbm_gbps, 1e-6), 1),
                "by_m": fp8_rows,
            }
            fp8_done = True
            break
        except Exception:
            torch.cuda.empty_cache()
    if not fp8_done:
        dtypes["fp8_e4m3"] = {"supported": False, "reason": "torch._scaled_mm FP8 unsupported on this device", "by_m": []}

    # 3. FP4 / MXFP4 (deep suite)
    if deep:
        dtypes["fp4"] = _probe_fp4_gemm_cuda(N, K, l2_bytes, free_bytes, hbm_gbps, major)
        torch.cuda.empty_cache()

    return {"shape_nk": [N, K], "dtypes": dtypes}


def _probe_moe_cuda(free_bytes: int, bf16_peak_tflops: float, hbm_gbps: float) -> dict:
    """Benchmark Grouped MoE GEMM across batch token counts."""
    import torch

    dev = torch.device("cuda:0")
    E, top_k, hidden, inter = (32, 8, 4096, 2048) if free_bytes >= 6 * GIB else (16, 4, 2048, 1024)
    F_ceil = max(1e9, bf16_peak_tflops * 1e12)
    BW_ceil = max(1e9, hbm_gbps * 1e9)

    def _finish(backend: str, rows: list[dict]) -> dict:
        torch.cuda.empty_cache()
        return {
            "supported": True,
            "backend": backend,
            "num_experts": E,
            "top_k": top_k,
            "hidden": hidden,
            "inter": inter,
            "by_tokens": rows,
        }

    # Option 1: vLLM's production `fused_experts` kernel
    try:
        from vllm.model_executor.layers.fused_moe import fused_experts

        w1 = torch.randn((E, 2 * inter, hidden), device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn((E, hidden, inter), device=dev, dtype=torch.bfloat16) * 0.02
        rows = []
        for t in MOE_TOKENS:
            x = torch.randn((t, hidden), device=dev, dtype=torch.bfloat16)
            ids = (torch.arange(t * top_k, device=dev, dtype=torch.int32) % E).view(t, top_k)
            wts = torch.full((t, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)
            dt = _cuda_bench(lambda _: fused_experts(x, w1, w2, wts, ids, inplace=False))
            routed = t * top_k
            touched = E * (1.0 - ((E - 1) / E) ** routed)
            flops = 2.0 * routed * (3.0 * hidden * inter)
            moved = touched * (3.0 * hidden * inter) * 2.0 + routed * hidden * 4.0
            t_ideal = max(flops / F_ceil, moved / BW_ceil)
            rows.append({
                "tokens": t,
                "avg_m_per_expert": round(routed / E, 2),
                "touched_experts": round(touched, 2),
                "tflops": round(flops / dt / 1e12, 2),
                "gbps": round(moved / dt / 1e9, 1),
                "roofline_eff": round(min(1.2, t_ideal / dt), 4),
            })
        del w1, w2
        return _finish("vllm_fused_experts_bf16", rows)
    except Exception:
        torch.cuda.empty_cache()

    # Option 2: PyTorch 2.6+ native `torch._grouped_mm`
    if hasattr(torch, "_grouped_mm"):
        try:
            w = torch.randn((E, 2 * inter, hidden), device=dev, dtype=torch.bfloat16).transpose(1, 2)
            rows = []
            for t in MOE_TOKENS:
                routed = t * top_k
                # Round per-expert counts cleanly while preserving realistic small-group sizes.
                base = routed // E
                rem = routed % E
                counts = [base + (1 if e < rem else 0) for e in range(E)]
                offs = torch.tensor(
                    [sum(counts[: e + 1]) for e in range(E)], device=dev, dtype=torch.int32
                )
                x = torch.randn((routed, hidden), device=dev, dtype=torch.bfloat16)
                dt = _cuda_bench(lambda _: torch._grouped_mm(x, w, offs=offs))
                touched = E * (1.0 - ((E - 1) / E) ** routed)
                flops = 2.0 * routed * hidden * (2 * inter)
                moved = touched * hidden * (2 * inter) * 2.0 + routed * (hidden + 2 * inter) * 2.0
                t_ideal = max(flops / F_ceil, moved / BW_ceil)
                rows.append({
                    "tokens": t,
                    "avg_m_per_expert": round(routed / E, 2),
                    "touched_experts": round(touched, 2),
                    "tflops": round(flops / dt / 1e12, 2),
                    "gbps": round(moved / dt / 1e9, 1),
                    "roofline_eff": round(min(1.2, t_ideal / dt), 4),
                })
            del w
            return _finish("torch_grouped_mm_bf16", rows)
        except Exception:
            torch.cuda.empty_cache()

    # Option 3: Padded BMM fallback
    try:
        w = torch.randn((E, hidden, 2 * inter), device=dev, dtype=torch.bfloat16)
        rows = []
        for t in MOE_TOKENS:
            routed = t * top_k
            m_pad = max(1, -(-routed // E))
            x = torch.randn((E, m_pad, hidden), device=dev, dtype=torch.bfloat16)
            dt = _cuda_bench(lambda _: torch.bmm(x, w))
            touched = E * (1.0 - ((E - 1) / E) ** routed)
            flops = 2.0 * routed * hidden * (2 * inter)
            moved = touched * hidden * (2 * inter) * 2.0
            t_ideal = max(flops / F_ceil, moved / BW_ceil)
            rows.append({
                "tokens": t,
                "avg_m_per_expert": round(routed / E, 2),
                "touched_experts": round(touched, 2),
                "tflops": round(flops / dt / 1e12, 2),
                "gbps": round(moved / dt / 1e9, 1),
                "roofline_eff": round(min(1.2, t_ideal / dt), 4),
            })
        del w
        return _finish("torch_bmm_padded_bf16", rows)
    except Exception as exc:
        return {"supported": False, "reason": str(exc), "by_tokens": []}


def _probe_overhead_cuda() -> dict:
    """Measure eager CPU launch cost, CUDA Graph replay floor, and `.item()` sync penalty."""
    import torch

    dev = torch.device("cuda:0")
    x = torch.ones(16, device=dev, dtype=torch.float32)
    y = torch.zeros(16, device=dev, dtype=torch.float32)
    for _ in range(50):
        y.add_(x)
    torch.cuda.synchronize()

    # 1. Eager CPU dispatch latency (enqueue 200 tiny kernels without sync inside loop)
    k_ops = 200
    cpu_samples = []
    for _ in range(25):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k_ops):
            y.add_(x)
        t_enq = time.perf_counter() - t0
        torch.cuda.synchronize()
        cpu_samples.append(t_enq / k_ops * 1e6)
    eager_launch_us = statistics.median(cpu_samples)

    # 2. Back-to-back GPU execution step in eager mode
    eager_gpu_step_us = _cuda_bench(lambda _: y.add_(x), warmup=20, rep=20, min_total_ms=10.0) * 1e6

    # 3. CUDA Graph fixed launch vs per-node transition ((t_100 - t_1) / 99)
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        for _ in range(10):
            y.add_(x)
    torch.cuda.synchronize()

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, stream=s):
        y.add_(x)

    g100 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g100, stream=s):
        for _ in range(100):
            y.add_(x)
    torch.cuda.synchronize()

    t1_us = _cuda_bench(lambda _: g1.replay(), warmup=20, rep=20, min_total_ms=10.0) * 1e6
    t100_us = _cuda_bench(lambda _: g100.replay(), warmup=20, rep=20, min_total_ms=10.0) * 1e6
    per_node_us = max(0.05, (t100_us - t1_us) / 99.0)
    fixed_launch_us = max(0.1, t1_us - per_node_us)

    # 4. Host `.item()` device->host scalar sync round-trip
    sync_samples = []
    for _ in range(80):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = y[0].item()
        sync_samples.append((time.perf_counter() - t0) * 1e6)
    host_sync_us = statistics.median(sync_samples)

    return {
        "eager_launch_us": round(eager_launch_us, 3),
        "eager_gpu_step_us": round(eager_gpu_step_us, 3),
        "cudagraph_fixed_launch_us": round(fixed_launch_us, 3),
        "cudagraph_per_kernel_us": round(per_node_us, 3),
        "host_sync_us": round(host_sync_us, 2),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _nccl_worker(rank: int, world_size: int, port: int, sizes: list[int], out_path: str) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    init_kwargs: dict = dict(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_kwargs["device_id"] = dev
    dist.init_process_group(**init_kwargs)

    try:
        ks = [2 ** i for i in range(1, world_size.bit_length()) if 2 ** i <= world_size]
        if world_size not in ks:
            ks.append(world_size)
        # All ranks in WORLD must call `new_group` in identical order.
        subgroups = {k: dist.new_group(ranks=list(range(k))) for k in ks}

        groups_out: dict[str, dict] = {}
        for k in ks:
            grp = subgroups[k]
            ar_rows, a2a_rows = [], []
            if rank < k:
                for b in sizes:
                    # Ensure divisibility by (2 bytes * k) for all_to_all_single
                    elems = max(k, (b // 2 // k) * k)
                    actual_b = elems * 2
                    buf = torch.ones(elems, device=dev, dtype=torch.bfloat16)
                    out_buf = torch.empty_like(buf)

                    # Warmup + median-of-samples AllReduce (rejects single-sample OS scheduling spikes)
                    for _ in range(5):
                        dist.all_reduce(buf, group=grp)
                    dist.barrier(group=grp)
                    torch.cuda.synchronize(dev)
                    inner = 5 if actual_b <= 4 * MIB else 2
                    ar_evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                              for _ in range(7)]
                    for st, en in ar_evs:
                        st.record()
                        for _ in range(inner):
                            dist.all_reduce(buf, group=grp)
                        en.record()
                    torch.cuda.synchronize(dev)
                    dt_ar = statistics.median((float(st.elapsed_time(en)) / 1000.0) / inner for st, en in ar_evs)
                    alg_ar = actual_b / dt_ar / 1e9
                    bus_ar = alg_ar * 2.0 * (k - 1) / k
                    ar_rows.append({
                        "size": _fmt_size(actual_b),
                        "lat_us": round(dt_ar * 1e6, 2),
                        "algbw_gbps": round(alg_ar, 2),
                        "busbw_gbps": round(bus_ar, 2),
                    })

                    # Warmup + median-of-samples AllToAll
                    for _ in range(5):
                        dist.all_to_all_single(out_buf, buf, group=grp)
                    dist.barrier(group=grp)
                    torch.cuda.synchronize(dev)
                    a2a_evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                               for _ in range(7)]
                    for st, en in a2a_evs:
                        st.record()
                        for _ in range(inner):
                            dist.all_to_all_single(out_buf, buf, group=grp)
                        en.record()
                    torch.cuda.synchronize(dev)
                    dt_a2a = statistics.median((float(st.elapsed_time(en)) / 1000.0) / inner for st, en in a2a_evs)
                    alg_a2a = actual_b / dt_a2a / 1e9
                    bus_a2a = alg_a2a * (k - 1) / k
                    a2a_rows.append({
                        "size": _fmt_size(actual_b),
                        "lat_us": round(dt_a2a * 1e6, 2),
                        "algbw_gbps": round(alg_a2a, 2),
                        "busbw_gbps": round(bus_a2a, 2),
                    })
                    del buf, out_buf

            dist.barrier()
            if rank == 0:
                lat_floor = min(ar_rows[0]["lat_us"] / 2.0, a2a_rows[0]["lat_us"])
                peak_bus = max(
                    max(r["busbw_gbps"] for r in ar_rows),
                    max(r["busbw_gbps"] for r in a2a_rows),
                )
                groups_out[str(k)] = {
                    "link_latency_us": round(lat_floor, 2),
                    "peak_busbw_gbps": round(peak_bus, 2),
                    "all_reduce": ar_rows,
                    "all_to_all": a2a_rows,
                }

        if rank == 0:
            Path(out_path).write_text(json.dumps({"backend": "nccl", "groups": groups_out}), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _probe_comm_cuda(num_gpus: int, free_bytes: int, timeout_s: float = 120.0) -> dict:
    if num_gpus < 2:
        return {"backend": "none", "groups": {}, "note": "single GPU; collectives skipped"}

    import torch.multiprocessing as mp

    max_b = max(4 * MIB, min(256 * MIB, int(free_bytes // 16)))
    sizes = [b for b in COMM_BYTES if b <= max_b]
    port = _free_port()
    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out_path = f.name
    # Intra-node NVLink/P2P probing does not use IB/SHARP; disabling RDMA net plugins avoids
    # HPC-X `nccl_rdma_sharp_plugin` / UCX segfaults inside containers without IB verbs mapped.
    saved_env: dict[str, str | None] = {}
    if os.environ.get("KILN_PROBE_KEEP_NET") != "1":
        for k_env, v_env in (("NCCL_IB_DISABLE", "1"), ("NCCL_NET_PLUGIN", "none"), ("NCCL_SHARP_DISABLE", "1")):
            if k_env not in os.environ:
                saved_env[k_env] = None
                os.environ[k_env] = v_env
    try:
        procs = []
        for r in range(num_gpus):
            p = ctx.Process(target=_nccl_worker, args=(r, num_gpus, port, sizes, out_path))
            p.start()
            procs.append(p)
    finally:
        for k_env, old in saved_env.items():
            if old is None:
                os.environ.pop(k_env, None)
            else:
                os.environ[k_env] = old
    try:
        deadline = time.monotonic() + timeout_s
        for p in procs:
            rem = max(0.1, deadline - time.monotonic())
            p.join(timeout=rem)
        alive = [p for p in procs if p.is_alive()]
        for p in alive:
            p.terminate()
            p.join(timeout=2.0)
        if alive:
            return {"backend": "nccl", "groups": {}, "error": f"NCCL probe timed out after {timeout_s:.0f}s"}
        raw = Path(out_path).read_text(encoding="utf-8").strip()
        if raw:
            return json.loads(raw)
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            return {"backend": "nccl", "groups": {}, "error": f"NCCL worker exited with codes {bad}"}
        return {"backend": "nccl", "groups": {}, "error": "NCCL worker produced no output"}
    except Exception as exc:
        return {"backend": "nccl", "groups": {}, "error": str(exc)}
    finally:
        Path(out_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- Mock fallback
def _smooth_roof(flops: float, moved_bytes: float, peak_flops: float, bw_bytes_s: float) -> float:
    """Smooth Roofline time `((flops / F)^p + (bytes / BW)^p)^(1/p)` with p=1.6."""
    t_c = flops / max(peak_flops, 1.0)
    t_m = moved_bytes / max(bw_bytes_s, 1.0)
    p = 1.6
    return (t_c ** p + t_m ** p) ** (1.0 / p)


def _probe_mock(chip_name: str, suite: str, no_comm: bool = False) -> HardwareProfile:
    lookup = {k.upper(): v for k, v in CHIPS.items()}
    base = lookup.get(chip_name.upper()) or CHIPS["H20-3e"]
    spec = _match_spec(base.name) or {"l2_mb": 60, "fp4_tc": base.name in ("B200", "MI355X")}
    l2_bytes = int(spec["l2_mb"]) * MIB
    num_gpus = int(base.max_chips)

    topology = {
        "device_name": f"{base.name} (Mock)",
        "arch": "sm_100" if spec["fp4_tc"] else "sm_90",
        "major": 10 if spec["fp4_tc"] else 9,
        "minor": 0,
        "sm_count": 148 if spec["fp4_tc"] else 78,
        "num_gpus": num_gpus,
        "hbm_gb": round(base.hbm_bytes / GB, 2),
        "free_hbm_gb": round(0.98 * base.hbm_bytes / GB, 2),
        "l2_cache_mib": round(l2_bytes / MIB, 1),
        "interconnect": "NVLink/NVSwitch",
        "nvlink_clique_size": num_gpus,
        "sm_clock_mhz": 1980,
        "sm_max_clock_mhz": 1980,
        "power_draw_w": 350.0,
        "power_limit_w": 700.0,
        "warnings": [],
        **_host_info(),
    }

    hbm_ach = base.hbm_bw * base.bw_eff
    hbm_gbps = hbm_ach / 1e9
    bw_curve = [
        {"size_mib": round(sz / MIB, 1), "copy_gbps": round(hbm_gbps * frac, 2), "triad_gbps": round(hbm_gbps * (frac - 0.02), 2)}
        for sz, frac in ((64 * MIB, 0.91), (256 * MIB, 0.97), (512 * MIB, 0.995), (1024 * MIB, 1.00))
    ]
    bandwidth = {
        "hbm_copy_gbps": round(hbm_gbps, 2),
        "hbm_triad_gbps": round(hbm_gbps * 0.98, 2),
        "hbm_peak_achieved_gbps": round(hbm_gbps, 2),
        "l2_gbps": round(hbm_gbps * 2.35, 2),
        "h2d_gbps": 54.0,
        "d2h_gbps": 53.2,
        "hbm_curve": bw_curve,
    }

    N, K = 8192, 4096
    f_bf16 = base.flops * base.flops_eff
    f_fp8 = f_bf16 * 1.92  # realistic achieved FP8/BF16 speedup (~1.9x vs 2.0x peak)

    def _synth_gemm(peak_f: float, w_bpe: float, act_bpe: float) -> list[dict]:
        rows = []
        for m in M_BUCKETS:
            flops = 2.0 * m * N * K
            moved = N * K * w_bpe + m * K * act_bpe + m * N * 2.0
            dt = _smooth_roof(flops, moved, peak_f, hbm_ach)
            rows.append({"m": m, "tflops": round(flops / dt / 1e12, 2), "gbps": round(moved / dt / 1e9, 1)})
        return rows

    bf16_rows = _synth_gemm(f_bf16, 2.0, 2.0)
    fp8_rows = _synth_gemm(f_fp8, 1.0, 1.0)
    dtypes: dict[str, dict] = {
        "bf16": {
            "supported": True,
            "mode": "tensor_core",
            "peak_tflops": max(r["tflops"] for r in bf16_rows),
            "ridge_flop_per_byte": round((f_bf16 / hbm_ach), 1),
            "by_m": bf16_rows,
        },
        "fp8_e4m3": {
            "supported": True,
            "mode": "float8_e4m3fn",
            "peak_tflops": max(r["tflops"] for r in fp8_rows),
            "ridge_flop_per_byte": round((f_fp8 / hbm_ach), 1),
            "by_m": fp8_rows,
        },
    }
    if suite == "deep":
        f_fp4 = f_bf16 * (3.6 if spec["fp4_tc"] else 1.75)
        fp4_rows = _synth_gemm(f_fp4, 0.5625, 0.5625 if spec["fp4_tc"] else 1.0)
        dtypes["fp4"] = {
            "supported": True,
            "native_tc": bool(spec["fp4_tc"]),
            "mode": "nvfp4_cutlass_tc" if spec["fp4_tc"] else "mxfp4_w4a8_triton",
            "peak_tflops": max(r["tflops"] for r in fp4_rows),
            "ridge_flop_per_byte": round((f_fp4 / hbm_ach), 1),
            "by_m": fp4_rows,
        }

    gemm = {"shape_nk": [N, K], "dtypes": dtypes}

    moe: dict = {"supported": False, "by_tokens": []}
    if suite == "deep":
        E, top_k, hidden, inter = 32, 8, 4096, 2048
        moe_rows = []
        for t in MOE_TOKENS:
            routed = t * top_k
            touched = E * (1.0 - ((E - 1) / E) ** routed)
            flops = 2.0 * routed * (3.0 * hidden * inter)
            moved = touched * (3.0 * hidden * inter) * 2.0 + routed * hidden * 4.0
            t_ideal = max(flops / f_bf16, moved / hbm_ach)
            # Tile-padding & routing overhead diminishes as avg_m_per_expert grows
            avg_m = routed / E
            eff = 0.48 + 0.42 * (1.0 - math.exp(-avg_m / 24.0))
            dt = t_ideal / eff
            moe_rows.append({
                "tokens": t,
                "avg_m_per_expert": round(avg_m, 2),
                "touched_experts": round(touched, 2),
                "tflops": round(flops / dt / 1e12, 2),
                "gbps": round(moved / dt / 1e9, 1),
                "roofline_eff": round(eff, 4),
            })
        moe = {
            "supported": True,
            "backend": "mock_fused_experts_bf16",
            "num_experts": E,
            "top_k": top_k,
            "hidden": hidden,
            "inter": inter,
            "by_tokens": moe_rows,
        }

    comm: dict = {"backend": "none", "groups": {}}
    if suite in ("standard", "deep") and not no_comm and num_gpus >= 2:
        eff_link = base.link_bw * base.link_eff
        lat0 = base.link_latency
        groups = {}
        for k in [2 ** i for i in range(1, num_gpus.bit_length()) if 2 ** i <= num_gpus]:
            ar_rows, a2a_rows = [], []
            for b in COMM_BYTES:
                # Protocol ramp: mid-sized payloads achieve lower busbw than 256 MiB ring saturation
                ramp = 0.35 + 0.65 * (b / (b + 2.0 * MIB))
                dt_ar = 2.0 * lat0 + (2.0 * (k - 1) / k * b) / (eff_link * ramp)
                alg_ar = b / dt_ar / 1e9
                ar_rows.append({
                    "size": _fmt_size(b),
                    "lat_us": round(dt_ar * 1e6, 2),
                    "algbw_gbps": round(alg_ar, 2),
                    "busbw_gbps": round(alg_ar * 2.0 * (k - 1) / k, 2),
                })
                dt_a2a = lat0 * (1.0 + 0.08 * k) + ((k - 1) / k * b) / (eff_link * ramp * 0.92)
                alg_a2a = b / dt_a2a / 1e9
                a2a_rows.append({
                    "size": _fmt_size(b),
                    "lat_us": round(dt_a2a * 1e6, 2),
                    "algbw_gbps": round(alg_a2a, 2),
                    "busbw_gbps": round(alg_a2a * (k - 1) / k, 2),
                })
            groups[str(k)] = {
                "link_latency_us": round(min(ar_rows[0]["lat_us"] / 2.0, a2a_rows[0]["lat_us"]), 2),
                "peak_busbw_gbps": round(max(r["busbw_gbps"] for r in ar_rows), 2),
                "all_reduce": ar_rows,
                "all_to_all": a2a_rows,
            }
        comm = {"backend": "mock_nccl", "groups": groups}

    overhead: dict = {}
    if suite == "deep":
        overhead = {
            "eager_launch_us": 3.4,
            "eager_gpu_step_us": 1.15,
            "cudagraph_fixed_launch_us": 2.8,
            "cudagraph_per_kernel_us": 0.42,
            "host_sync_us": 8.6,
        }

    calibrated = _build_calibrated_chip(topology, bandwidth, gemm, comm)
    return HardwareProfile(
        schema_version=1,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        is_mock=True,
        suite=suite,
        topology=topology,
        bandwidth=bandwidth,
        gemm=gemm,
        moe=moe,
        comm=comm,
        overhead=overhead,
        calibrated_chip=calibrated,
    )


# --------------------------------------------------------------------------- Entrypoint
def probe_hardware(
    suite: str = "deep",
    gpus: list[int] | None = None,
    no_comm: bool = False,
    mock_chip: str | None = None,
    require_gpu: bool = False,
    progress: bool = True,
) -> HardwareProfile:
    """Run hardware & roofline microbenchmarks (or CPU mock fallback) and return `HardwareProfile`."""
    if suite not in ("quick", "standard", "deep"):
        raise ValueError(f"unknown suite {suite!r}; choose from 'quick', 'standard', 'deep'")

    def _log(msg: str) -> None:
        if progress:
            print(f"[kiln probe] {msg}", file=sys.stderr, flush=True)

    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    if mock_chip is not None:
        _log(f"generating mock '{suite}' profile for chip={mock_chip}")
        return _probe_mock(mock_chip, suite=suite, no_comm=no_comm)

    if not _has_cuda():
        if require_gpu:
            raise RuntimeError("No CUDA/ROCm GPU detected (`torch.cuda.is_available()` is False)")
        fallback = "H20-3e"
        _log(f"no CUDA/ROCm GPU detected; using CPU mock fallback ({fallback}). Pass --require-gpu to fail.")
        return _probe_mock(fallback, suite=suite, no_comm=no_comm)

    t0 = time.perf_counter()
    _log("[1/5] probing GPU & node topology...")
    topo = _probe_topology_cuda()
    l2_b = _topo_l2_bytes(topo)
    free_b = _topo_free_bytes(topo)

    _log("[2/5] measuring L2, HBM, and PCIe bandwidth...")
    bw = _probe_bandwidth_cuda(l2_b, free_b)

    _log(f"[3/5] measuring cold-L2 dense GEMM roofline (suite={suite})...")
    gemm = _probe_gemm_cuda(
        l2_b,
        free_b,
        bw["hbm_peak_achieved_gbps"],
        topo["major"],
        deep=(suite == "deep"),
    )

    moe: dict = {"supported": False, "by_tokens": []}
    overhead: dict = {}
    if suite == "deep":
        _log("[4/5] measuring grouped MoE GEMM & launch/CUDA-graph overhead...")
        bf16_peak = float(gemm["dtypes"]["bf16"]["peak_tflops"])
        moe = _probe_moe_cuda(free_b, bf16_peak, bw["hbm_peak_achieved_gbps"])
        overhead = _probe_overhead_cuda()
    else:
        _log("[4/5] skipping MoE & launch overhead (use --suite deep to enable)...")

    comm: dict = {"backend": "none", "groups": {}}
    if suite in ("standard", "deep") and not no_comm and topo["num_gpus"] >= 2:
        _log(f"[5/5] measuring multi-GPU NCCL collectives across {topo['num_gpus']} GPUs...")
        comm = _probe_comm_cuda(topo["num_gpus"], free_b)
    else:
        _log("[5/5] skipping multi-GPU collectives...")

    calibrated = _build_calibrated_chip(topo, bw, gemm, comm)
    _log(f"done in {time.perf_counter() - t0:.1f}s")
    return HardwareProfile(
        schema_version=1,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        is_mock=False,
        suite=suite,
        topology=topo,
        bandwidth=bw,
        gemm=gemm,
        moe=moe,
        comm=comm,
        overhead=overhead,
        calibrated_chip=calibrated,
    )
