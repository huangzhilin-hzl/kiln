"""Hardware counter & compiler diagnostics (`ptxas` register/spill & `ncu` roofline/bank conflicts)."""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

_PTXAS_ENTRY_RE = re.compile(r"Compiling entry function\s+['\"]([^'\"]+)['\"]")
_PTXAS_REGS_RE = re.compile(r"Used\s+(\d+)\s+registers?", re.IGNORECASE)
_PTXAS_SMEM_RE = re.compile(r"(\d+)\s+bytes\s+(?:smem|shared\s+memory)", re.IGNORECASE)
_PTXAS_STACK_RE = re.compile(r"(\d+)\s+bytes\s+(?:cumulative\s+)?stack\s+frame", re.IGNORECASE)
_PTXAS_SPILL_ST_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+stores", re.IGNORECASE)
_PTXAS_SPILL_LD_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+loads", re.IGNORECASE)

_PTX_ENTRY_HDR_RE = re.compile(r"(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$]+)\s*\(")
_PTX_REG_DECL_RE = re.compile(r"\.reg\s+\.[a-z0-9]+\s+%[a-zA-Z_]+<(\d+)>")
_PTX_SHARED_DECL_RE = re.compile(r"\.shared\s+\.[a-z0-9]+(?:\s*\[\s*(\d+)\s*\])?")
_PTX_LOCAL_DECL_RE = re.compile(r"\.local\s+\.[a-z0-9]+(?:\s*\[\s*(\d+)\s*\])?")


def _estimate_occupancy_pct(
    registers: int,
    threads_per_block: int = 256,
    sm_regs_limit: int = 65536,
    max_threads_per_sm: int = 2048,
) -> float:
    if registers <= 0:
        return 100.0
    regs_per_block = max(1, registers * threads_per_block)
    blocks_by_regs = sm_regs_limit // regs_per_block
    max_blocks_by_threads = max(1, max_threads_per_sm // threads_per_block)
    active_threads = min(max_threads_per_sm, blocks_by_regs * threads_per_block, max_blocks_by_threads * threads_per_block)
    return round(min(100.0, 100.0 * active_threads / max(1, max_threads_per_sm)), 1)


def parse_ptx_or_ptxas_text(
    text: str,
    sm_regs_limit: int = 65536,
    max_threads_per_sm: int = 2048,
) -> list[dict[str, Any]]:
    """Parse either `ptxas -v` diagnostics or raw `.ptx` text for register counts and spills."""
    entries: list[dict[str, Any]] = []
    cur_kernel: str | None = None
    pending_stack = 0
    pending_spill_st = 0
    pending_spill_ld = 0

    for line in text.splitlines():
        m_ent = _PTXAS_ENTRY_RE.search(line)
        if m_ent:
            cur_kernel = m_ent.group(1)
            pending_stack = 0
            pending_spill_st = 0
            pending_spill_ld = 0
            continue
        if m_stk := _PTXAS_STACK_RE.search(line):
            pending_stack = max(pending_stack, int(m_stk.group(1)))
        if m_st := _PTXAS_SPILL_ST_RE.search(line):
            pending_spill_st = max(pending_spill_st, int(m_st.group(1)))
        if m_ld := _PTXAS_SPILL_LD_RE.search(line):
            pending_spill_ld = max(pending_spill_ld, int(m_ld.group(1)))

        m_reg = _PTXAS_REGS_RE.search(line)
        if m_reg:
            regs = int(m_reg.group(1))
            smem = int(m.group(1)) if (m := _PTXAS_SMEM_RE.search(line)) else 0
            stack = pending_stack
            spill_st = pending_spill_st
            spill_ld = pending_spill_ld
            has_spills = (spill_st > 0) or (spill_ld > 0) or (stack > 0)
            warnings: list[str] = []
            if has_spills:
                warnings.append(
                    f"register spill detected ({spill_st}B stores, {spill_ld}B loads, {stack}B stack)"
                )
            if regs >= 128:
                warnings.append(f"high register count ({regs} regs/thread)")
            entries.append({
                "kernel": cur_kernel or f"kernel_{len(entries)}",
                "registers": regs,
                "smem_bytes": smem,
                "stack_frame_bytes": stack,
                "spill_stores_bytes": spill_st,
                "spill_loads_bytes": spill_ld,
                "has_spills": has_spills,
                "est_occupancy_pct": _estimate_occupancy_pct(regs, sm_regs_limit=sm_regs_limit, max_threads_per_sm=max_threads_per_sm),
                "warnings": warnings,
            })
            cur_kernel = None
            pending_stack = 0
            pending_spill_st = 0
            pending_spill_ld = 0

    if entries:
        return entries

    # Fallback: scan raw PTX `.entry` blocks
    matches = list(_PTX_ENTRY_HDR_RE.finditer(text))
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        kname = m.group(1)
        regs = sum(int(x) for x in _PTX_REG_DECL_RE.findall(body))
        smem = sum(int(x or 4) for x in _PTX_SHARED_DECL_RE.findall(body))
        local_bytes = sum(int(x or 4) for x in _PTX_LOCAL_DECL_RE.findall(body))
        has_spills = local_bytes > 0
        warnings = []
        if has_spills:
            warnings.append(f"PTX .local allocation detected ({local_bytes} bytes)")
        if regs >= 128:
            warnings.append(f"high register declaration ({regs} virtual regs)")
        entries.append({
            "kernel": kname,
            "registers": regs,
            "smem_bytes": smem,
            "stack_frame_bytes": local_bytes,
            "spill_stores_bytes": local_bytes,
            "spill_loads_bytes": 0,
            "has_spills": has_spills,
            "est_occupancy_pct": _estimate_occupancy_pct(regs, sm_regs_limit=sm_regs_limit, max_threads_per_sm=max_threads_per_sm),
            "warnings": warnings,
        })
    return entries


def _detect_ptx_target(ptx_text: str, arch: str | None = None) -> str:
    if arch:
        return arch
    m = re.search(r"\.target\s+(sm_\d+[a-z]?)", ptx_text)
    return m.group(1) if m else "sm_90"


def inspect_ptxas_and_spills(
    cache_dir: Path | None = None,
    ptx_texts: Sequence[str] = (),
    run_ptxas_bin: bool = True,
    arch: str | None = None,
) -> dict[str, Any]:
    """Inspect cached `.ptx` files and inline PTX strings for registers and local-memory spills."""
    ptx_sources: list[tuple[str, str]] = []
    if cache_dir is not None and Path(cache_dir).is_dir():
        for p in sorted(Path(cache_dir).rglob("*.ptx")):
            try:
                ptx_sources.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    for idx, txt in enumerate(ptx_texts):
        if txt and txt.strip():
            ptx_sources.append((f"<inline_{idx}>", txt))

    if not ptx_sources:
        return {
            "available": False,
            "ptx_files": 0,
            "any_spills": False,
            "max_registers": 0,
            "kernels": [],
            "warnings": [],
        }

    ptxas_bin = shutil.which("ptxas") if run_ptxas_bin else None
    all_kernels: list[dict[str, Any]] = []
    used_ptxas = False

    for src_label, txt in ptx_sources:
        parsed: list[dict[str, Any]] = []
        if ptxas_bin and (".version" in txt or ".target" in txt):
            with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False, encoding="utf-8") as f:
                tmp_ptx = Path(f.name)
                f.write(txt)
            tmp_cubin = tmp_ptx.with_suffix(".cubin")
            try:
                proc = subprocess.run(
                    [ptxas_bin, "-v", f"-arch={_detect_ptx_target(txt, arch)}", "-o", str(tmp_cubin), str(tmp_ptx)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                diag = (proc.stderr or "") + "\n" + (proc.stdout or "")
                parsed = parse_ptx_or_ptxas_text(diag)
                if parsed:
                    used_ptxas = True
            except Exception:
                parsed = []
            finally:
                tmp_ptx.unlink(missing_ok=True)
                tmp_cubin.unlink(missing_ok=True)
        if not parsed:
            parsed = parse_ptx_or_ptxas_text(txt)
        for row in parsed:
            row.setdefault("source", src_label)
            all_kernels.append(row)

    any_spills = any(k.get("has_spills") for k in all_kernels)
    max_regs = max((int(k.get("registers", 0)) for k in all_kernels), default=0)
    warnings: list[str] = []
    for k in all_kernels:
        for w in k.get("warnings", ()):
            warnings.append(f"{k['kernel']}: {w}")

    return {
        "available": True,
        "used_ptxas_bin": used_ptxas,
        "ptx_files": len(ptx_sources),
        "any_spills": any_spills,
        "max_registers": max_regs,
        "kernels": all_kernels,
        "warnings": warnings,
    }


def _parse_num(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


_NCU_METRICS = (
    "dram__bytes.sum.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
)


def parse_ncu_csv(csv_text: str) -> dict[str, Any]:
    """Parse Nsight Compute `--csv` output into HBM, TensorCore, occupancy, and bank-conflict stats."""
    lines = [
        ln for ln in csv_text.splitlines()
        if ln.strip() and not ln.startswith("==PROF==") and not ln.startswith("==WARNING==")
    ]
    if not lines:
        return {"available": False, "reason": "empty ncu output"}

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    fieldnames = reader.fieldnames or []
    if not fieldnames:
        return {"available": False, "reason": "missing CSV header"}

    metrics_agg: dict[str, list[float]] = {}
    kernel_names: list[str] = []

    is_long = "Metric Name" in fieldnames and "Metric Value" in fieldnames
    for row in reader:
        kname = (row.get("Kernel Name") or "").strip()
        if kname and kname not in kernel_names and not kname.startswith("("):
            kernel_names.append(kname)
        if is_long:
            mname = (row.get("Metric Name") or "").strip()
            val = _parse_num(row.get("Metric Value"))
            if not mname or val is None:
                continue
            unit = (row.get("Metric Unit") or "").strip().lower()
            if mname == "dram__bytes.sum.per_second":
                if "gbyte" in unit or "gb" in unit:
                    pass
                elif "mbyte" in unit or "mb" in unit:
                    val /= 1000.0
                elif val > 1e6:
                    val /= 1e9
            metrics_agg.setdefault(mname, []).append(val)
        else:
            for col, raw_val in row.items():
                if not col or col in ("ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream"):
                    continue
                val = _parse_num(raw_val)
                if val is None:
                    continue
                if "dram__bytes.sum.per_second" in col and val > 1e6:
                    val /= 1e9
                metrics_agg.setdefault(col, []).append(val)

    def _mean(key: str) -> float | None:
        vals = metrics_agg.get(key)
        if not vals:
            return None
        return round(sum(vals) / len(vals), 3)

    def _sum(key: str) -> float | None:
        vals = metrics_agg.get(key)
        if not vals:
            return None
        return round(sum(vals), 1)

    bc_ld = _sum("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    bc_st = _sum("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    bc_total = None
    if bc_ld is not None or bc_st is not None:
        bc_total = round((bc_ld or 0.0) + (bc_st or 0.0), 1)

    swizzle_hint = None
    if bc_total is not None and bc_total > 0:
        swizzle_hint = (
            f"Detected {bc_total:,.0f} shared-memory bank conflicts (ld={bc_ld or 0:.0f}, st={bc_st or 0:.0f}); "
            "apply XOR/CuTe Swizzle layout or pad SMEM stride."
        )

    return {
        "available": True,
        "kernels": kernel_names,
        "hbm_gbps": _mean("dram__bytes.sum.per_second"),
        "dram_bw_pct": _mean("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "tensor_core_pct": _mean("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
        "achieved_occupancy_pct": _mean("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "l2_hit_rate_pct": _mean("lts__t_sector_hit_rate.pct"),
        "shared_bank_conflicts_ld": bc_ld,
        "shared_bank_conflicts_st": bc_st,
        "shared_bank_conflicts_total": bc_total,
        "swizzle_hint": swizzle_hint,
    }


def collect_ncu_counters(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Run Nsight Compute (`ncu`) around `cmd` and return parsed hardware counter stats."""
    ncu_bin = shutil.which("ncu", path=env.get("PATH"))
    if ncu_bin is None:
        return {"available": False, "reason": "ncu binary not found on PATH"}
    full_cmd = [
        ncu_bin,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(_NCU_METRICS),
        *cmd,
    ]
    try:
        proc = subprocess.run(
            full_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"ncu timed out after {timeout_s:.0f}s"}
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "ERR_NVGPUCTRPERM" in combined or "permission" in combined.lower():
        return {"available": False, "reason": "ncu permission denied (ERR_NVGPUCTRPERM)"}
    if proc.returncode != 0 and "Kernel Name" not in combined:
        return {"available": False, "reason": combined.strip()[-400:] or f"ncu exited rc={proc.returncode}"}
    return parse_ncu_csv(proc.stdout or combined)
