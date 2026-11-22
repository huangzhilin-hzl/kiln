"""Lock-free micro-kernel sandbox, cold-boot tensor snapshots, and compiler/hardware counters."""

from .counters import collect_ncu_counters, inspect_ptxas_and_spills, parse_ncu_csv, parse_ptx_or_ptxas_text
from .sandbox import MicroRunner, execute_candidate_file
from .snapshots import (
    DEFAULT_M_BUCKETS,
    SnapshotCollector,
    classify_submodule_tag,
    load_snapshot_payload,
    load_snapshots,
    nearest_m_bucket,
    write_mock_snapshots,
)

__all__ = [
    "DEFAULT_M_BUCKETS",
    "MicroRunner",
    "SnapshotCollector",
    "classify_submodule_tag",
    "collect_ncu_counters",
    "execute_candidate_file",
    "inspect_ptxas_and_spills",
    "load_snapshot_payload",
    "load_snapshots",
    "nearest_m_bucket",
    "parse_ncu_csv",
    "parse_ptx_or_ptxas_text",
    "write_mock_snapshots",
]
