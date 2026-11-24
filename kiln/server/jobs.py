"""Per-tier job queue with a persistent engine (TKCC §4.2).

One worker thread per tier owns one persistent engine. Each job is handled as:
    hot_reload -> correctness check -> measure -> teardown
If the engine crashes, it restarts automatically and the job is marked as crashed.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import queue
import statistics
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from kiln.bench.agentx import compare_agentx_paired
from kiln.bench.correctness import evaluate_correctness, validate_probe_logprobs
from kiln.bench.traffic import Request
from kiln.config import RunConfig, TierConfig
from kiln.engine import Engine, make_engine
from kiln.engine.base import (
    InvalidBenchmark,
    MeasureResult,
    bucket_tps_for_requests,
    evaluating_phase,
    replay_itl_metrics,
    shutdown_process_group,
)
from kiln.engine.prebuild import PrebuildResult, prebuild_candidate
from kiln.engine.vllm_profiler import (
    TraceAnalysis,
    compute_roofline_attribution,
    diff_traces,
    format_trace_report,
)
from kiln.patches import (
    compose_candidates,
    compose_pd_pair,
    detect_candidate_conflicts,
    inspect_candidate_source,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Job:
    id: str
    agent: str
    tier: str
    note: str
    candidate_path: str
    status: str = "queued"          # queued | running | done | incorrect | invalid | crashed | invalidated
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    speedup: float | None = None
    decode_tps: float | None = None
    prefill_tps: float | None = None
    variance_pct: float | None = None
    token_match: float | None = None
    max_logit_abs_diff: float | None = None
    kernel_report: str | None = None
    extra: dict = field(default_factory=dict)
    error: str | None = None
    source_sha256: str | None = None
    invalidated_reason: str | None = None
    baseline_tps: float | None = None
    engine_boot_id: str | None = None
    parent_id: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    force_trace: bool = False
    worker_id: str | None = None
    attempt_id: str | None = None
    lease_expires_at: float | None = None


REMOTE_LEASE_S = 90.0
REMOTE_HEARTBEAT_S = 10.0
TERMINAL_STATUSES = {"done", "incorrect", "invalid", "crashed", "invalidated", "cancelled"}


class QueueLimitError(ValueError):
    """Raised when an agent exceeds `TierConfig.max_queued_per_agent`."""


class DuplicateJobError(ValueError):
    """Raised when an identical active candidate is already queued/running on the tier."""


_FATAL_KEYWORDS = (
    "illegal memory", "cudaerror", "nccl", "device-side assert",
    "timed out", "timeout", "exited", "connection", "broken pipe", "eof",
)


def _run_with_timeout(fn, timeout_s: float, label: str):
    """Run `fn()` in a daemon thread and raise TimeoutError if it exceeds `timeout_s` (> 0)."""
    if not timeout_s or timeout_s <= 0:
        return fn()
    outcome: dict = {}

    def _target() -> None:
        try:
            outcome["val"] = fn()
        except BaseException as exc:  # propagate worker exceptions cleanly
            outcome["exc"] = exc

    t = threading.Thread(target=_target, name=f"watchdog-{label}", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    if "exc" in outcome:
        raise outcome["exc"]
    return outcome.get("val")


class SharedEngineAdapter(Engine):
    """Proxy engine that serializes onto an owning tier's persistent Engine."""

    def __init__(self, owner: "TierWorker", tier: TierConfig):
        self.owner = owner
        self.tier = tier

    def start(self) -> None:
        if self.owner.engine is None:
            self.owner._ensure_own_engine()

    def hot_reload(self, candidate_path: str) -> None:
        with evaluating_phase(self.tier.phase):
            self.owner.engine.hot_reload(candidate_path)

    def teardown_candidate(self) -> None:
        self.owner.engine.teardown_candidate()

    def teardown_candidate_lazy(self) -> None:
        fn = getattr(self.owner.engine, "teardown_candidate_lazy", self.owner.engine.teardown_candidate)
        fn()

    def ensure_baseline(self) -> None:
        fn = getattr(self.owner.engine, "ensure_baseline", None)
        if callable(fn):
            fn()

    def needs_baseline_rebuild(self) -> bool:
        fn = getattr(self.owner.engine, "needs_baseline_rebuild", None)
        return bool(fn()) if callable(fn) else False

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        with evaluating_phase(self.tier.phase):
            eng = self.owner.engine
            if self.tier.agentx is None and hasattr(eng, "measure_requests"):
                return eng.measure_requests(requests, repeats, trace)
            return eng.measure(requests, repeats, trace)

    def measure_more(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        with evaluating_phase(self.tier.phase):
            eng = self.owner.engine
            if self.tier.agentx is None and hasattr(eng, "measure_requests"):
                return eng.measure_requests(requests, repeats, trace, skip_prefill=True)
            return eng.measure_more(requests, repeats, trace)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.owner.engine.greedy_outputs(prompts, max_new_tokens)

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        return self.owner.engine.greedy_probe(prompts, max_new_tokens)

    def is_alive(self) -> bool:
        return self.owner.engine.is_alive() if self.owner.engine is not None else False

    def shutdown(self) -> None:
        self.owner._shutdown_own_engine()


class TierWorker:
    def __init__(self, run: RunConfig, tier: TierConfig, requests: list[Request], jobs_dir: Path,
                 local: bool = True, on_job_finished: Callable[[Job], None] | None = None,
                 screen_threshold_fn: Callable[[], float] | None = None):
        self.run, self.tier, self.jobs_dir, self.local = run, tier, jobs_dir, local
        self.on_job_finished = on_job_finished
        self.screen_threshold_fn = screen_threshold_fn
        self.requests = requests[: tier.max_requests] if tier.max_requests else requests
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.stop_requested = threading.Event()
        self.engine: Engine | None = None
        self.owner: TierWorker | None = None
        self.borrowers: list[TierWorker] = []
        self.engine_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._prebuild_lock = threading.Lock()
        self._prebuild_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._prebuild_futures: dict[str, concurrent.futures.Future[PrebuildResult]] = {}
        self._prebuild_cancels: dict[str, threading.Event] = {}
        self._prebuild_procs: dict[str, subprocess.Popen | None] = {}
        self._last_job_finished_mono: float = time.monotonic()
        self.baseline_tps: float | None = None
        self.baseline_decode_tps: float | None = None
        self.baseline_prefill_tps: float | None = None
        self.baseline_ttft_s: float | None = None
        self.baseline_bucket_tps: dict[str, float] = {}
        self.baseline_trace: TraceAnalysis | None = None
        self.hw_profile = None
        self.baseline_extra: dict = {}
        self.baseline_tokens: list[list[int]] | None = None
        self.baseline_logprobs: list[list[dict[int, float]]] | None = None
        self.boot_id: str | None = None
        self.last_canary: dict | None = None
        self._jobs_since_canary: int = 0
        self.durations: list[float] = []
        self.crashes = 0
        self.boot_error: str | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"tier-{tier.name}", daemon=True)

    def _ensure_hw_profile(self) -> None:
        if self.hw_profile is not None:
            return
        try:
            hw_path = self.run.resolve_hw_profile()
            if hw_path is not None:
                from kiln.costmodel.probe import HardwareProfile

                self.hw_profile = HardwareProfile.load(hw_path)
        except Exception as exc:
            print(f"[kiln] {self.tier.name}: failed loading hw_profile ({exc})")

    # ------------------------------------------------------------------ engine
    def _ensure_own_engine(self) -> None:
        if self.engine is None:
            eng = make_engine(self.run.engine, self.tier)
            try:
                eng.start()
            except Exception:
                try:
                    eng.shutdown()
                except Exception:
                    pass
                raise
            self.engine = eng

    def _shutdown_own_engine(self) -> None:
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            self.engine = None

    @property
    def score_metric_name(self) -> str:
        if self.tier.agentx is not None:
            return "agentx_total_tput_per_gpu"
        return self.tier.resolved_metric

    def _primary_score(self, m: MeasureResult) -> float:
        if self.tier.agentx is not None or (m.extra or {}).get("metric") == "agentx_total_tput_per_gpu":
            return float(m.decode_tps)
        metric = self.tier.resolved_metric
        if metric == "prefill_tps":
            return float(m.prefill_tps)
        if metric == "ttft":
            return (1.0 / float(m.ttft_s)) if (m.ttft_s and m.ttft_s > 0) else 0.0
        return float(m.decode_tps)

    def _boot(self) -> None:
        with self.engine_lock:
            if self.stop_requested.is_set():
                return
            self.ready.clear()
            self._ensure_hw_profile()
            if self.owner is None:
                self._ensure_own_engine()
                self.boot_id = uuid.uuid4().hex[:8]
            else:
                self.owner._ensure_own_engine()
                self.engine = SharedEngineAdapter(self.owner, self.tier)
                if not self.owner.boot_id:
                    self.owner.boot_id = uuid.uuid4().hex[:8]
                self.boot_id = self.owner.boot_id
            if not self.tier.lazy_baseline and (self.baseline_tokens is None or self.baseline_tps is None):
                self._measure_baseline()
            snap_dir = self.jobs_dir.parent / "micro_snapshots"
            if (
                self.run.micro.enabled
                and self.run.micro.capture_on_boot
                and self.owner is None
                and not (snap_dir / "manifest.json").exists()
            ):
                try:
                    self.engine.capture_snapshots(str(snap_dir), list(self.run.micro.m_buckets))
                except Exception as exc:
                    print(f"[kiln] {self.tier.name}: cold-boot snapshot capture skipped ({exc})")
            self.boot_error = None
            self.ready.set()

    def _measure_baseline(self) -> None:
        self._ensure_baseline_ready()
        cc = self.run.correctness
        probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
        tokens, logprobs = self.engine.greedy_probe(probe, cc.max_new_tokens)
        if not probe or len(tokens) != len(probe) or not all(tokens):
            raise RuntimeError("Baseline returned incomplete correctness outputs")
        error = validate_probe_logprobs(tokens, logprobs)
        if error:
            raise RuntimeError(f"Invalid baseline logprobs: {error}")
        base = self.engine.measure(
            self.requests,
            self.tier.repeats,
            trace=(self.tier.trace_kernels and self.tier.trace_policy != "never"),
        )
        primary = self._primary_score(base)
        if not math.isfinite(primary) or primary <= 0:
            raise RuntimeError(f"Baseline {self.score_metric_name} must be finite and positive")
        self.baseline_tokens = tokens
        self.baseline_logprobs = logprobs
        self.baseline_tps = primary
        self.baseline_decode_tps = base.decode_tps
        self.baseline_prefill_tps = base.prefill_tps
        self.baseline_ttft_s = base.ttft_s
        self.baseline_extra = dict(base.extra or {})
        self.baseline_bucket_tps = {
            str(k): float(v)
            for k, v in (self.baseline_extra.get("bucket_tps") or {}).items()
            if isinstance(v, (int, float)) and v > 0
        }
        if base.trace_analysis:
            self.baseline_trace = TraceAnalysis.from_dict(base.trace_analysis)
        print(
            f"[kiln] {self.tier.name} ({self.tier.phase}/{self.score_metric_name}): "
            f"baseline {primary:.2f} (variance {base.variance_pct:.2f}%)"
        )

    def _restart_engine(self) -> None:
        with self.engine_lock:
            if self.owner is None:
                self._shutdown_own_engine()
                self._boot()
            else:
                self.ready.clear()
                owner_was_ready = self.owner.ready.is_set()
                self.owner.ready.clear()
                self.owner._shutdown_own_engine()
                self.owner._ensure_own_engine()
                self.owner.boot_id = uuid.uuid4().hex[:8]
                self.boot_id = self.owner.boot_id
                if owner_was_ready:
                    self.owner.ready.set()
                self.ready.set()

    def _recover_or_restart(self, err: Exception) -> None:
        self.crashes += 1
        msg = str(err).lower()
        if (
            not any(k in msg for k in _FATAL_KEYWORDS)
            and self.engine is not None
            and self.engine.is_alive()
            and self.baseline_tokens
        ):
            try:
                wd = float(self.tier.timeout_s or 0)
                td = min(wd, 60.0) if wd > 0 else 0.0
                _run_with_timeout(self.engine.teardown_candidate, td, f"{self.tier.name} recover-teardown")
                n_check = min(4, self.run.correctness.max_new_tokens)
                got = _run_with_timeout(
                    lambda: self.engine.greedy_outputs([self.requests[0].prompt_ids], n_check),
                    td,
                    f"{self.tier.name} recover-probe",
                )
                if self.baseline_tokens and got == [self.baseline_tokens[0][:n_check]] and len(got[0]) == n_check:
                    print(f"[kiln] {self.tier.name}: recovered warm engine after non-fatal candidate error")
                    return
            except Exception:
                pass
        self._restart_engine()

    # ------------------------------------------------------------------ prebuild & idle baseline
    @property
    def _effective_engine_kind(self) -> str:
        return str(self.tier.engine_kind or self.run.engine.kind)

    @property
    def _root_worker(self) -> "TierWorker":
        return self.owner if self.owner is not None else self

    def _has_queued_jobs(self) -> bool:
        with self.q.mutex:
            return any(j is not None and j.status == "queued" for j in self.q.queue)

    def _any_shared_queued(self) -> bool:
        root = self._root_worker
        if root._has_queued_jobs():
            return True
        return any(b._has_queued_jobs() for b in root.borrowers)

    def _engine_needs_baseline_rebuild(self) -> bool:
        eng = self.engine
        if eng is None:
            return False
        fn = getattr(eng, "needs_baseline_rebuild", None)
        return bool(fn()) if callable(fn) else False

    def _ensure_baseline_ready(self) -> None:
        eng = self.engine
        if eng is None:
            return
        fn = getattr(eng, "ensure_baseline", None)
        if not callable(fn) or not self._engine_needs_baseline_rebuild():
            return
        timeout_s = float(self.tier.timeout_s or 0)
        _run_with_timeout(fn, timeout_s, f"{self.tier.name} ensure_baseline")

    def _idle_rebuild_timeout(self) -> float | None:
        delay = self.tier.idle_baseline_rebuild_s
        if delay is None or not self.ready.is_set() or not self._engine_needs_baseline_rebuild():
            return None
        if self._any_shared_queued():
            return None
        elapsed = time.monotonic() - self._root_worker._last_job_finished_mono
        return max(0.01, float(delay) - elapsed)

    def _maybe_rebuild_idle_baseline(self) -> bool:
        delay = self.tier.idle_baseline_rebuild_s
        if delay is None or self.stop_requested.is_set() or not self._engine_needs_baseline_rebuild():
            return False
        with self.engine_lock:
            if self.stop_requested.is_set() or self._any_shared_queued():
                return False
            if not self._engine_needs_baseline_rebuild():
                return False
            if (time.monotonic() - self._root_worker._last_job_finished_mono) < float(delay):
                return False
            try:
                self._ensure_baseline_ready()
                return True
            except Exception as exc:
                print(f"[kiln] {self.tier.name}: idle baseline rebuild failed ({exc}); restarting")
                self.ready.clear()
                try:
                    self._restart_engine()
                except Exception as restart_exc:
                    self.boot_error = str(restart_exc)
                return False

    def _run_prebuild_now(self, job: Job, cancel_ev: threading.Event | None = None) -> PrebuildResult:
        cpath = Path(job.candidate_path)
        if not cpath.is_file():
            return PrebuildResult(ok=True, skipped=True, reason="missing_file")

        def _track_proc(proc: subprocess.Popen | None) -> None:
            with self._prebuild_lock:
                if proc is None:
                    self._prebuild_procs.pop(job.id, None)
                else:
                    self._prebuild_procs[job.id] = proc

        gpus = self.tier.gpus or (self.owner.tier.gpus if self.owner is not None else [])
        return prebuild_candidate(
            cpath,
            mode=self.tier.prebuild,
            timeout_s=float(self.tier.prebuild_timeout_s),
            pd_role=self.tier.phase,
            engine_kind=self._effective_engine_kind,
            gpus=gpus,
            cancel_event=cancel_ev,
            on_proc=_track_proc,
        )

    def _start_prebuild(self, job: Job) -> None:
        if not self.local or self.tier.prebuild == "off":
            return
        cpath = Path(job.candidate_path)
        if not cpath.is_file():
            return
        try:
            src = cpath.read_text(encoding="utf-8")
        except OSError:
            return
        info = inspect_candidate_source(src)
        # Fast-path candidates (valid syntax, no native build triggers under mode="auto") need no pool thread.
        if info.get("syntax_ok") and self.tier.prebuild == "auto" and not info.get("needs_prebuild"):
            return
        with self._prebuild_lock:
            if self.stop_requested.is_set() or job.id in self._prebuild_futures:
                return
            if self._prebuild_pool is None:
                self._prebuild_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix=f"prebuild-{self.tier.name}"
                )
            cancel_ev = threading.Event()
            self._prebuild_cancels[job.id] = cancel_ev
            self._prebuild_futures[job.id] = self._prebuild_pool.submit(
                self._run_prebuild_now, job, cancel_ev
            )

    def _await_prebuild(self, job: Job) -> PrebuildResult:
        with self._prebuild_lock:
            fut = self._prebuild_futures.get(job.id)
        if fut is not None:
            try:
                return fut.result()
            finally:
                with self._prebuild_lock:
                    self._prebuild_futures.pop(job.id, None)
                    self._prebuild_cancels.pop(job.id, None)
                    self._prebuild_procs.pop(job.id, None)
        return self._run_prebuild_now(job)

    def cancel_prebuild(self, jid: str) -> None:
        with self._prebuild_lock:
            ev = self._prebuild_cancels.pop(jid, None)
            fut = self._prebuild_futures.pop(jid, None)
            proc = self._prebuild_procs.pop(jid, None)
        if ev is not None:
            ev.set()
        if fut is not None:
            fut.cancel()
        if proc is not None:
            shutdown_process_group(proc, timeout=3.0)

    def shutdown_prebuild(self) -> None:
        with self._prebuild_lock:
            jids = list({*self._prebuild_cancels, *self._prebuild_futures, *self._prebuild_procs})
            pool = self._prebuild_pool
            self._prebuild_pool = None
        for jid in jids:
            self.cancel_prebuild(jid)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _fail_prebuild_job(self, job: Job, pre: PrebuildResult) -> None:
        now = time.time()
        started = job.started_at or now
        extra = dict(job.extra or {})
        extra["prebuild"] = pre.to_dict()
        result = replace(
            job,
            status="crashed",
            started_at=started,
            finished_at=now,
            error=pre.error or f"Candidate prebuild failed ({pre.reason})",
            extra=extra,
        )
        with self._stats_lock:
            self.crashes += 1
            self.durations.append(max(0.0, now - started))
        self._save(result)
        job.__dict__.update(result.__dict__)
        if self.on_job_finished is not None:
            self.on_job_finished(job)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        self.wait_until_ready()
        while not self.stop_requested.is_set():
            try:
                idle_wait = self._idle_rebuild_timeout()
                job = self.q.get(timeout=idle_wait)
            except queue.Empty:
                self._maybe_rebuild_idle_baseline()
                continue
            if job is None:
                return
            if self.stop_requested.is_set() or job.status != "queued":
                continue
            pre = self._await_prebuild(job)
            if self.stop_requested.is_set() or job.status != "queued":
                continue
            if not pre.ok:
                self._fail_prebuild_job(job, pre)
                continue
            with self.engine_lock:
                if self.stop_requested.is_set():
                    return
                if job.status != "queued":
                    continue
                if self.engine is not None and not self.engine.is_alive():
                    print(f"[kiln] {self.tier.name}: engine exited unexpectedly; restarting")
                    self.ready.clear()
                    try:
                        self._restart_engine()
                    except Exception as exc:
                        self.boot_error = str(exc)
                self.wait_until_ready()
                if not self.stop_requested.is_set():
                    self.execute_job(job, _prebuild_result=pre)

    def wait_until_ready(self) -> None:
        while not self.ready.is_set() and not self.stop_requested.is_set():
            try:
                self._boot()
            except Exception as exc:
                if self.boot_error != str(exc):
                    print(f"[kiln] {self.tier.name}: boot failed ({exc}); retrying")
                self.boot_error = str(exc)
                self.stop_requested.wait(5)

    def execute_job(self, job: Job, *, _prebuild_result: PrebuildResult | None = None) -> None:
        if job.status == "cancelled":
            return
        pre = _prebuild_result if _prebuild_result is not None else self._await_prebuild(job)
        if job.status == "cancelled":
            return
        if not pre.ok:
            if pre.reason == "cancelled":
                return
            self._fail_prebuild_job(job, pre)
            return

        with self.engine_lock:
            if job.status == "cancelled":
                return
            job.status, job.started_at = "running", time.time()
            if not pre.skipped:
                job.extra = {**(job.extra or {}), "prebuild": pre.to_dict()}
            self._save(job)
            result = replace(job)
            try:
                self._run_job(result)
            except Exception as e:  # crash / IMA / hang surfaced as exception
                result.status, result.error = "crashed", f"{e}\n{traceback.format_exc()[-2000:]}"
                try:
                    self._recover_or_restart(e)
                except Exception as exc:
                    self.ready.clear()
                    self.boot_error = str(exc)
                    result.error += f"\nEngine recovery failed: {exc}"
            finally:
                result.finished_at = time.time()
                self._root_worker._last_job_finished_mono = time.monotonic()
                self._save(result)
                job.__dict__.update(result.__dict__)
                with self._stats_lock:
                    self.durations.append(result.finished_at - result.started_at)
                if (
                    result.status == "done"
                    and self.tier.canary_every_n_jobs is not None
                ):
                    self._jobs_since_canary += 1
                    if self._jobs_since_canary >= self.tier.canary_every_n_jobs:
                        try:
                            self._run_canary()
                        except Exception as exc:
                            print(f"[kiln] {self.tier.name}: canary calibration skipped ({exc})")
                if self.on_job_finished is not None:
                    self.on_job_finished(job)

    def _run_canary(self) -> dict | None:
        if self.engine is None or not self.baseline_tps or self.baseline_tps <= 0:
            return None
        self._ensure_baseline_ready()
        timeout_s = float(self.tier.timeout_s or 0)
        ax_dur = ((self.tier.agentx or {}).get("benchmark_duration_s") or 0)
        measure_timeout = (
            max(timeout_s, float(ax_dur) * 3 + 300.0)
            if (self.tier.agentx is not None and timeout_s > 0)
            else timeout_s
        )
        m = _run_with_timeout(
            lambda: self.engine.measure(self.requests, repeats=1, trace=False),
            measure_timeout,
            f"{self.tier.name} canary",
        )
        tps = float(self._primary_score(m))
        if not math.isfinite(tps) or tps <= 0:
            raise RuntimeError("Canary throughput must be finite and positive")
        drift_pct = 100.0 * (tps - self.baseline_tps) / self.baseline_tps
        rec = {
            "ts": time.time(),
            "tier": self.tier.name,
            "boot_id": self.boot_id,
            "baseline_tps": round(self.baseline_tps, 4),
            "measured_tps": round(tps, 4),
            "drift_pct": round(drift_pct, 3),
        }
        self.last_canary = rec
        self._jobs_since_canary = 0
        cpath = self.jobs_dir.parent / f"canary_{self.tier.name}.jsonl"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "a", encoding="utf-8") as cf:
            cf.write(json.dumps(rec) + "\n")
        if abs(drift_pct) > 3.0:
            print(
                f"[kiln] WARNING {self.tier.name}: baseline canary drift {drift_pct:+.2f}% "
                f"({tps:.1f} vs {self.baseline_tps:.1f} tok/s)"
            )
        return rec

    def _effective_screen_threshold(self, threshold_override: float | None = None) -> float:
        if isinstance(threshold_override, (int, float)) and math.isfinite(threshold_override) and threshold_override > 0:
            return float(threshold_override)
        if self.screen_threshold_fn is not None:
            val = float(self.screen_threshold_fn())
            if math.isfinite(val) and val > 0:
                return val
        return float(self.tier.screen_min_speedup)

    @staticmethod
    def _merge_stage_results(
        m1: MeasureResult, m2: MeasureResult, requests: list[Request]
    ) -> MeasureResult:
        runs = list(m1.runs_decode_tps) + list(m2.runs_decode_tps)
        runs_prefill = list(m1.runs_prefill_tps) + list(m2.runs_prefill_tps)
        decode_tps = (sum(runs) / len(runs)) if runs else float(m1.decode_tps)
        prefill_tps = (sum(runs_prefill) / len(runs_prefill)) if runs_prefill else float(m1.prefill_tps)
        extra = dict(m1.extra or {})
        for k, v in (m2.extra or {}).items():
            if k == "bucket_tps":
                continue
            extra[k] = v
        if isinstance(extra.get("bucket_tps"), dict):
            bt = dict(extra["bucket_tps"])
            if m1.decode_tps > 0 and decode_tps > 0:
                ratio = decode_tps / float(m1.decode_tps)
                for bkey in ("decode_small", "decode_mid"):
                    if isinstance(bt.get(bkey), (int, float)):
                        bt[bkey] = float(bt[bkey]) * ratio
            if prefill_tps > 0:
                bt["prefill_large"] = prefill_tps
            extra["bucket_tps"] = bt
        elif decode_tps > 0 or prefill_tps > 0:
            bt = bucket_tps_for_requests(requests, decode_tps, prefill_tps)
            if bt:
                extra["bucket_tps"] = bt
        itl_samples = m1.itl_samples_ms + m2.itl_samples_ms
        if itl_samples:
            extra.update(replay_itl_metrics(itl_samples))
        ttft_s = m1.ttft_s
        if m2.runs_prefill_tps:
            n1, n2 = len(m1.runs_prefill_tps), len(m2.runs_prefill_tps)
            ttft_s = ((m1.ttft_s * n1 + m2.ttft_s * n2) / (n1 + n2)
                      if (n1 == 0 or m1.ttft_s > 0) and m2.ttft_s > 0 else 0.0)
        return MeasureResult(
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            ttft_s=ttft_s,
            itl_samples_ms=itl_samples,
            runs_decode_tps=runs,
            runs_prefill_tps=runs_prefill,
            kernel_report=m2.kernel_report if m2.kernel_report is not None else m1.kernel_report,
            trace_analysis=m2.trace_analysis if m2.trace_analysis is not None else m1.trace_analysis,
            extra=extra,
        )

    def _run_job(self, job: Job) -> None:
        eng, cc = self.engine, self.run.correctness
        base_job_extra = dict(job.extra or {})
        threshold_override = base_job_extra.pop("_screen_threshold", None)
        job.extra = dict(base_job_extra)
        if self.owner is not None and self.owner.boot_id:
            self.boot_id = self.owner.boot_id
        if self.baseline_tokens is None or self.baseline_tps is None:
            self._measure_baseline()
        job.baseline_tps = self.baseline_tps
        job.engine_boot_id = self.boot_id
        reload_timeout = float(self.tier.timeout_s or 0)
        ax_dur = ((self.tier.agentx or {}).get("benchmark_duration_s") or 0)
        measure_timeout = (
            max(reload_timeout, float(ax_dur) * 3 + 300.0)
            if (self.tier.agentx is not None and reload_timeout > 0)
            else reload_timeout
        )
        teardown_timeout = min(reload_timeout, 60.0) if reload_timeout > 0 else 0.0

        _run_with_timeout(
            lambda: eng.hot_reload(job.candidate_path),
            reload_timeout,
            f"{self.tier.name} hot_reload",
        )
        try:
            probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
            cand_tokens, cand_logprobs = _run_with_timeout(
                lambda: eng.greedy_probe(probe, cc.max_new_tokens),
                reload_timeout,
                f"{self.tier.name} greedy_probe",
            )
            corr = evaluate_correctness(
                self.baseline_tokens, cand_tokens, cc, self.baseline_logprobs, cand_logprobs
            )
            job.token_match = corr.token_match
            job.max_logit_abs_diff = corr.max_logit_abs_diff
            if not corr.ok:
                job.status, job.error = "incorrect", corr.detail
                return
            screening_meta: dict | None = None
            try:
                if (
                    self.tier.agentx is None
                    and self.tier.screen_repeats is not None
                    and self.tier.screen_repeats < self.tier.repeats
                ):
                    m1 = _run_with_timeout(
                        lambda: eng.measure(self.requests, self.tier.screen_repeats, trace=False),
                        measure_timeout,
                        f"{self.tier.name} screen",
                    )
                    s1_score = self._primary_score(m1)
                    if not math.isfinite(s1_score) or s1_score <= 0:
                        job.status, job.error = (
                            "invalid",
                            f"Candidate {self.score_metric_name} must be finite and positive",
                        )
                        return
                    s1_speedup = s1_score / self.baseline_tps
                    threshold = self._effective_screen_threshold(threshold_override)
                    promoted = bool(job.force_trace or s1_speedup >= threshold)
                    want_trace = bool(
                        job.force_trace
                        or (self.tier.trace_kernels and self.tier.trace_policy != "never" and promoted)
                    )
                    if promoted:
                        rem = self.tier.repeats - self.tier.screen_repeats
                        m2 = _run_with_timeout(
                            lambda: eng.measure_more(self.requests, rem, trace=want_trace),
                            measure_timeout,
                            f"{self.tier.name} measure",
                        )
                        m = self._merge_stage_results(m1, m2, self.requests)
                    else:
                        m = m1
                    screening_meta = {
                        "stage": 2 if promoted else 1,
                        "promoted": promoted,
                        "stage1_speedup": round(s1_speedup, 4),
                        "threshold": round(threshold, 4),
                        "screen_repeats": self.tier.screen_repeats,
                        "total_repeats": len(m.runs_decode_tps or m.runs_prefill_tps) or self.tier.screen_repeats,
                        "traced": want_trace if promoted else False,
                        "forced_trace": bool(job.force_trace),
                    }
                elif (
                    self.tier.agentx is None
                    and self.tier.trace_kernels
                    and self.tier.trace_policy == "on_improve"
                    and not job.force_trace
                ):
                    m1 = _run_with_timeout(
                        lambda: eng.measure(self.requests, self.tier.repeats, trace=False),
                        measure_timeout,
                        f"{self.tier.name} measure",
                    )
                    s1_score = self._primary_score(m1)
                    if not math.isfinite(s1_score) or s1_score <= 0:
                        job.status, job.error = (
                            "invalid",
                            f"Candidate {self.score_metric_name} must be finite and positive",
                        )
                        return
                    s1_speedup = s1_score / self.baseline_tps
                    threshold = self._effective_screen_threshold(threshold_override)
                    promoted = s1_speedup >= threshold
                    if promoted:
                        m2 = _run_with_timeout(
                            lambda: eng.measure_more(self.requests, 0, trace=True),
                            measure_timeout,
                            f"{self.tier.name} trace",
                        )
                        m = self._merge_stage_results(m1, m2, self.requests)
                    else:
                        m = m1
                    screening_meta = {
                        "stage": 2 if promoted else 1,
                        "promoted": promoted,
                        "stage1_speedup": round(s1_speedup, 4),
                        "threshold": round(threshold, 4),
                        "screen_repeats": self.tier.repeats,
                        "total_repeats": len(m.runs_decode_tps or m.runs_prefill_tps) or self.tier.repeats,
                        "traced": promoted,
                        "forced_trace": False,
                    }
                else:
                    want_trace = bool(
                        job.force_trace or (self.tier.trace_kernels and self.tier.trace_policy != "never")
                    )
                    m = _run_with_timeout(
                        lambda: eng.measure(self.requests, self.tier.repeats, trace=want_trace),
                        measure_timeout,
                        f"{self.tier.name} measure",
                    )
            except InvalidBenchmark as exc:
                job.status, job.error, job.extra = "invalid", str(exc), {**base_job_extra, **(exc.extra or {})}
                return
            cand_score = self._primary_score(m)
            if not math.isfinite(cand_score) or cand_score <= 0:
                job.status, job.error = (
                    "invalid",
                    f"Candidate {self.score_metric_name} must be finite and positive",
                )
                return
            extra = {**base_job_extra, **(m.extra or {})}
            extra.setdefault("phase", self.tier.phase)
            extra.setdefault("score_metric", self.score_metric_name)
            if m.ttft_s and m.ttft_s > 0:
                extra.setdefault("ttft_s", round(float(m.ttft_s), 6))
            if screening_meta is not None:
                extra["screening"] = screening_meta
            if self.baseline_extra.get("export") and extra.get("export"):
                paired = compare_agentx_paired(self.baseline_extra["export"], extra["export"])
                if paired is not None:
                    extra["paired"] = paired

            bucket_tps = extra.get("bucket_tps")
            if isinstance(bucket_tps, dict):
                bucket_speedups: dict[str, float] = {}
                for bkey, bval in bucket_tps.items():
                    if not isinstance(bval, (int, float)) or bval <= 0:
                        continue
                    base_b = self.baseline_bucket_tps.get(bkey)
                    if base_b is None and bkey == "prefill_large":
                        base_b = self.baseline_prefill_tps
                    elif base_b is None and bkey in ("decode_small", "decode_mid"):
                        base_b = self.baseline_decode_tps or self.baseline_tps
                    if base_b and base_b > 0:
                        bucket_speedups[str(bkey)] = round(float(bval) / float(base_b), 4)
                if bucket_speedups:
                    extra["bucket_speedups"] = bucket_speedups

            cand_trace = TraceAnalysis.from_dict(m.trace_analysis) if m.trace_analysis else None
            if cand_trace is not None:
                extra["trace_diff"] = diff_traces(cand_trace, self.baseline_trace)
            rf = compute_roofline_attribution(
                cand_trace,
                self.hw_profile,
                model_name=self.run.engine.model,
                tp_size=self.run.engine.tensor_parallel_size,
                decode_tps=m.decode_tps,
                prefill_tps=m.prefill_tps,
                batch_size=len(self.requests),
            )
            if rf is not None:
                extra["roofline"] = rf

            job.decode_tps, job.prefill_tps = m.decode_tps, m.prefill_tps
            job.variance_pct = m.variance_pct
            job.kernel_report = (
                format_trace_report(cand_trace, self.baseline_trace, rf)
                if cand_trace is not None
                else m.kernel_report
            )
            job.extra = extra
            job.speedup = cand_score / self.baseline_tps
        finally:
            teardown_fn = (
                getattr(eng, "teardown_candidate_lazy", eng.teardown_candidate)
                if self.tier.lazy_teardown
                else eng.teardown_candidate
            )
            _run_with_timeout(teardown_fn, teardown_timeout, f"{self.tier.name} teardown")
        job.status = "done"

    # ------------------------------------------------------------------ api
    def submit(self, job: Job) -> None:
        self._save(job)
        self._start_prebuild(job)
        self.q.put(job)

    def stats(self) -> dict:
        if self.tier.share_engine:
            mode = f"shared:{self.tier.share_engine}" if self.local else f"remote:shared:{self.tier.share_engine}"
        else:
            mode = "local" if self.local else "remote"
        with self.q.mutex:
            q_depth = sum(1 for j in self.q.queue if j is not None and j.status == "queued")
        return {
            "tier": self.tier.name,
            "mode": mode,
            "phase": self.tier.phase,
            "metric": self.score_metric_name,
            "ready": self.ready.is_set(),
            "boot_error": self.boot_error,
            "queue_depth": q_depth,
            "median_job_s": statistics.median(self.durations) if self.durations else None,
            "baseline_decode_tps": self.baseline_tps,
            "boot_id": self.boot_id,
            "last_canary": self.last_canary,
            "crashes": self.crashes,
            "layers": self.tier.num_layers or "all",
        }

    def _save(self, job: Job) -> None:
        d = self.jobs_dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=d, delete=False) as f:
            temp = Path(f.name)
            try:
                json.dump(asdict(job), f, indent=2, allow_nan=False)
                f.close()
                temp.replace(d / "result.json")
            finally:
                temp.unlink(missing_ok=True)


def _wire_shared_tiers(
    workers: dict[str, TierWorker],
    tiers: list[TierConfig],
    *,
    enforce_same_locality: bool = False,
) -> None:
    """Attach borrower TierWorkers to their owner engine & lock."""
    for t in tiers:
        if t.name not in workers or not t.share_engine:
            continue
        if t.share_engine == t.name or t.share_engine not in workers:
            raise ValueError(f"invalid share_engine {t.share_engine!r} on tier {t.name!r}")
        owner = workers[t.share_engine]
        if owner.tier.share_engine:
            raise ValueError(f"chained share_engine not supported: {t.name} -> {t.share_engine}")
        borrower = workers[t.name]
        if enforce_same_locality and borrower.local != owner.local:
            raise ValueError(
                f"tier {t.name} (local={borrower.local}) cannot share engine "
                f"{t.share_engine} (local={owner.local})"
            )
        borrower.owner = owner
        borrower.engine_lock = owner.engine_lock
        if borrower not in owner.borrowers:
            owner.borrowers.append(borrower)
        if t.trace_kernels and not owner.tier.trace_kernels and (not enforce_same_locality or borrower.local):
            owner.tier = owner.tier.model_copy(update={"trace_kernels": True})


class JobManager:
    def __init__(self, run: RunConfig, requests: list[Request], local_tiers: set[str] | None = None):
        self.run = run
        self.requests = requests
        self.root = Path(run.workdir) / run.name
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        active = {t.name for t in run.tiers} if local_tiers is None else set(local_tiers)
        self.tiers = {
            t.name: TierWorker(
                run, t, requests, self.jobs_dir,
                local=(t.name in active),
                on_job_finished=self._on_job_finished,
                screen_threshold_fn=(lambda name=t.name: self._screen_threshold(name)),
            )
            for t in run.tiers
        }
        _wire_shared_tiers(self.tiers, run.tiers, enforce_same_locality=True)
        self.lock = threading.Condition()
        self.remote_workers: dict[str, dict[str, float]] = {
            name: {} for name, worker in self.tiers.items() if not worker.local
        }
        job_fields = {f.name for f in fields(Job)}
        for path in self.jobs_dir.glob("*/result.json"):
            raw = json.loads(path.read_text())
            job = Job(**{k: v for k, v in raw.items() if k in job_fields})
            cand_path = path.with_name("candidate.py")
            job.candidate_path = str(cand_path)
            if job.source_sha256 is None and cand_path.exists():
                job.source_sha256 = _sha256_text(cand_path.read_text())
            self.jobs[job.id] = job
        for job in sorted(self.jobs.values(), key=lambda job: (job.submitted_at, job.id)):
            worker = self.tiers.get(job.tier)
            if worker is None:
                continue
            if job.status == "running" and (
                worker.local or not job.lease_expires_at or job.lease_expires_at <= time.time()
            ):
                self._requeue(worker, job)
            elif job.status == "queued":
                worker._start_prebuild(job)
                worker.q.put(job)
            elif job.finished_at and job.started_at:
                worker.durations.append(job.finished_at - job.started_at)

    def start(self) -> None:
        for w in self.tiers.values():
            if w.local:
                w.thread.start()

    def shutdown(self) -> None:
        """Finish active jobs, leave queued work on disk, and release each local engine once."""
        local_workers = [w for w in self.tiers.values() if w.local]
        for worker in local_workers:
            worker.stop_requested.set()
            worker.shutdown_prebuild()
            worker.q.put(None)
        for worker in local_workers:
            if worker.thread.is_alive():
                worker.thread.join()
        for worker in local_workers:
            worker.ready.clear()
            if worker.owner is None:
                worker._shutdown_own_engine()
            else:
                worker.engine = None

    def _screen_threshold_locked(self, tier: str) -> float:
        cfg = self.tiers[tier].tier
        thresh = float(cfg.screen_min_speedup)
        if cfg.screen_top_k is not None:
            done = [
                j for j in self._valid_done_jobs_locked(tier)
                if j.speedup is not None and math.isfinite(j.speedup)
            ]
            if len(done) >= cfg.screen_top_k:
                done.sort(key=lambda j: float(j.speedup), reverse=True)
                kth = float(done[cfg.screen_top_k - 1].speedup)
                thresh = max(thresh, kth)
        return thresh

    def _screen_threshold(self, tier: str) -> float:
        with self.lock:
            return self._screen_threshold_locked(tier)

    def submit(
        self,
        agent: str,
        tier: str,
        source: str,
        note: str = "",
        parent_id: str | None = None,
        parent_ids: list[str] | None = None,
        tags: list[str] | None = None,
        force_trace: bool = False,
        supersede: bool = False,
    ) -> Job:
        if tier not in self.tiers:
            raise KeyError(f"unknown tier {tier}; have {list(self.tiers)}")
        src_sha = _sha256_text(source)
        inspected = inspect_candidate_source(source)

        merged_parents: list[str] = []
        for pid in ([parent_id] if parent_id else []) + list(parent_ids or ()) + list(inspected.get("parents") or ()):
            p_clean = str(pid).strip() if pid else ""
            if p_clean and p_clean not in merged_parents:
                merged_parents.append(p_clean)
        resolved_parent_id = merged_parents[0] if merged_parents else None

        merged_tags: list[str] = []
        for t in list(tags or ()) + list(inspected.get("tags") or ()):
            t_clean = str(t).strip() if t else ""
            if t_clean and t_clean not in merged_tags:
                merged_tags.append(t_clean)

        tier_cfg = self.tiers[tier].tier
        with self.lock:
            if tier_cfg.deduplicate_queued:
                dup = next(
                    (
                        j for j in self.jobs.values()
                        if j.tier == tier and j.source_sha256 == src_sha and j.status in ("queued", "running")
                    ),
                    None,
                )
                if dup is not None:
                    raise DuplicateJobError(
                        f"Duplicate active candidate already {dup.status} as {dup.id} on {tier}"
                    )
            agent_queued = sorted(
                (j for j in self.jobs.values() if j.tier == tier and j.agent == agent and j.status == "queued"),
                key=lambda j: (j.submitted_at, j.id),
            )
            running_cnt = sum(
                1 for j in self.jobs.values() if j.tier == tier and j.agent == agent and j.status == "running"
            )
            jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
            superseded_ids: list[str] = []
            if supersede or tier_cfg.supersede_queued:
                if tier_cfg.max_inflight_per_agent is not None and running_cnt >= tier_cfg.max_inflight_per_agent:
                    raise QueueLimitError(
                        f"Agent {agent!r} already has {running_cnt} running jobs on {tier} "
                        f"(max_inflight {tier_cfg.max_inflight_per_agent})"
                    )
                if supersede or (
                    tier_cfg.max_queued_per_agent is None and tier_cfg.max_inflight_per_agent is None
                ):
                    keep_queued = 0
                else:
                    limits: list[int] = []
                    if tier_cfg.max_queued_per_agent is not None:
                        limits.append(tier_cfg.max_queued_per_agent - 1)
                    if tier_cfg.max_inflight_per_agent is not None:
                        limits.append(tier_cfg.max_inflight_per_agent - running_cnt - 1)
                    keep_queued = max(0, min(limits))
                to_cancel = agent_queued[: max(0, len(agent_queued) - keep_queued)]
                now = time.time()
                for old in to_cancel:
                    old.status = "cancelled"
                    old.finished_at = now
                    old.invalidated_reason = f"Superseded by {jid}"
                    self.tiers[tier].cancel_prebuild(old.id)
                    self.tiers[tier]._save(old)
                    superseded_ids.append(old.id)
            else:
                if tier_cfg.max_queued_per_agent is not None and len(agent_queued) >= tier_cfg.max_queued_per_agent:
                    raise QueueLimitError(
                        f"Agent {agent!r} already has {len(agent_queued)} queued jobs on {tier} "
                        f"(max {tier_cfg.max_queued_per_agent})"
                    )
                if (
                    tier_cfg.max_inflight_per_agent is not None
                    and running_cnt + len(agent_queued) >= tier_cfg.max_inflight_per_agent
                ):
                    raise QueueLimitError(
                        f"Agent {agent!r} already has {running_cnt + len(agent_queued)} in-flight jobs on {tier} "
                        f"(max {tier_cfg.max_inflight_per_agent})"
                    )
            d = self.jobs_dir / jid
            d.mkdir(parents=True)
            path = d / "candidate.py"
            path.write_text(source)
            job = Job(
                id=jid, agent=agent, tier=tier, note=note,
                candidate_path=str(path), source_sha256=src_sha,
                parent_id=resolved_parent_id,
                parent_ids=merged_parents,
                tags=merged_tags,
                force_trace=force_trace,
                extra=({"superseded": superseded_ids} if superseded_ids else {}),
            )
            self.tiers[tier].submit(job)
            self.jobs[jid] = job
            self.lock.notify_all()
        return job

    def get_job_source(self, jid: str) -> dict:
        with self.lock:
            job = self.jobs[jid]
            source = Path(job.candidate_path).read_text()
            return {
                "id": job.id,
                "tier": job.tier,
                "agent": job.agent,
                "source_sha256": job.source_sha256,
                "parent_id": job.parent_id,
                "parent_ids": list(job.parent_ids),
                "tags": list(job.tags),
                "source": source,
            }

    def cancel_job(self, jid: str) -> dict:
        with self.lock:
            job = self.jobs[jid]
            if job.status != "queued":
                raise ValueError(f"Cannot cancel {job.status} job {jid}; only queued jobs can be cancelled")
            job.status = "cancelled"
            job.finished_at = time.time()
            job.invalidated_reason = "Cancelled"
            worker = self.tiers.get(job.tier)
            if worker is not None:
                worker.cancel_prebuild(job.id)
                worker._save(job)
            self.lock.notify_all()
            return {"id": job.id, "status": job.status}

    def _invalidate_matching_sha256_locked(
        self, src_sha: str, exclude_id: str, reason: str
    ) -> list[str]:
        invalidated: list[str] = []
        for other in self.jobs.values():
            if other.id == exclude_id or other.source_sha256 != src_sha or other.status != "done":
                continue
            other.status = "invalidated"
            other.invalidated_reason = reason
            worker = self.tiers.get(other.tier)
            if worker is not None:
                worker._save(other)
            invalidated.append(other.id)
        return invalidated

    def _maybe_cascade_invalidate_locked(self, job: Job) -> list[str]:
        if (
            job.status == "incorrect"
            and job.token_match is not None
            and job.token_match < 0.5
            and job.source_sha256
        ):
            reason = (
                f"Auto-invalidated: severe mismatch on {job.tier} "
                f"({job.id}, token_match={job.token_match:.4f})"
            )
            return self._invalidate_matching_sha256_locked(job.source_sha256, job.id, reason)
        return []

    def _on_job_finished(self, job: Job) -> None:
        with self.lock:
            self._maybe_cascade_invalidate_locked(job)

    def invalidate_job(self, jid: str, reason: str = "", cascade_sha256: bool = True) -> list[str]:
        with self.lock:
            job = self.jobs[jid]
            if job.status in ("queued", "running"):
                raise ValueError(f"Cannot invalidate {job.status} job {jid}")
            changed: list[str] = []
            why = reason.strip() or "Withdrawn"
            if job.status != "invalidated":
                job.status = "invalidated"
                job.invalidated_reason = why
                worker = self.tiers.get(job.tier)
                if worker is not None:
                    worker._save(job)
                changed.append(jid)
            if cascade_sha256 and job.source_sha256:
                changed.extend(
                    self._invalidate_matching_sha256_locked(
                        job.source_sha256,
                        jid,
                        reason.strip() or f"Cascaded from withdrawn job {jid}",
                    )
                )
            return changed

    # ------------------------------------------------------------------ remote node workers
    def _requeue(self, worker: TierWorker, job: Job) -> None:
        job.status = "queued"
        job.started_at = job.finished_at = None
        job.worker_id = job.attempt_id = job.lease_expires_at = None
        worker.submit(job)

    def _expire_remote(self, tier: str) -> None:
        worker = self.tiers[tier]
        now = time.time()
        workers = self.remote_workers[tier]
        for worker_id, expires_at in list(workers.items()):
            if expires_at <= now:
                del workers[worker_id]
        if not workers:
            worker.ready.clear()
        for job in self.jobs.values():
            if (job.tier == tier and job.status == "running" and
                    (job.lease_expires_at is None or job.lease_expires_at <= now)):
                self._requeue(worker, job)
                self.lock.notify_all()

    def status(self) -> dict:
        with self.lock:
            for tier in self.remote_workers:
                self._expire_remote(tier)
            return {"run": self.run.name, "tiers": [w.stats() for w in self.tiers.values()],
                    "jobs": len(self.jobs)}

    def get_job(self, jid: str) -> dict:
        with self.lock:
            job = self.jobs[jid]
            if job.tier in self.remote_workers:
                self._expire_remote(job.tier)
            data = asdict(job)
            for key in ("worker_id", "attempt_id", "lease_expires_at"):
                data.pop(key)
            return data

    def heartbeat_remote(self, tier: str, worker_id: str, ready: bool,
                         baseline_tps: float | None, crashes: int,
                         attempt_id: str | None = None,
                         boot_id: str | None = None,
                         last_canary: dict | None = None) -> None:
        worker = self.tiers[tier]
        with self.lock:
            self._expire_remote(tier)
            expires_at = time.time() + REMOTE_LEASE_S
            if ready:
                self.remote_workers[tier][worker_id] = expires_at
                worker.ready.set()
                worker.baseline_tps = baseline_tps
                worker.crashes = crashes
                if boot_id is not None:
                    worker.boot_id = boot_id
                if last_canary is not None:
                    worker.last_canary = last_canary
            else:
                self.remote_workers[tier].pop(worker_id, None)
                if not self.remote_workers[tier]:
                    worker.ready.clear()
            if attempt_id:
                for job in self.jobs.values():
                    if (job.tier == tier and job.status == "running" and
                            job.worker_id == worker_id and job.attempt_id == attempt_id):
                        job.lease_expires_at = expires_at
                        worker._save(job)
                        break

    def claim_remote(self, tier: str, worker_id: str, wait_s: float = 2.0) -> dict | None:
        worker = self.tiers[tier]
        deadline = time.monotonic() + wait_s
        with self.lock:
            while True:
                self._expire_remote(tier)
                job = next((job for job in self.jobs.values()
                            if job.tier == tier and job.status == "running" and
                            job.worker_id == worker_id), None)
                if job is not None:
                    break
                try:
                    job = worker.q.get_nowait()
                except queue.Empty:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self.lock.wait(timeout=remaining)
                else:
                    if job.status != "queued":
                        continue
                    job.status, job.started_at = "running", time.time()
                    job.worker_id = worker_id
                    job.attempt_id = uuid.uuid4().hex
                    break
            job.lease_expires_at = time.time() + REMOTE_LEASE_S
            worker._save(job)
            return {
                "id": job.id, "agent": job.agent, "note": job.note,
                "attempt_id": job.attempt_id,
                "force_trace": job.force_trace,
                "screen_threshold": self._screen_threshold_locked(tier),
                "source": Path(job.candidate_path).read_text(),
            }

    def complete_remote(self, tier: str, data: dict) -> None:
        worker = self.tiers[tier]
        with self.lock:
            self._expire_remote(tier)
            job = self.jobs[data["id"]]
            if job.tier != tier or job.worker_id != data["worker_id"] or job.attempt_id != data["attempt_id"]:
                raise ValueError("Job claim is stale or belongs to another worker")
            if job.status in TERMINAL_STATUSES:
                return
            if job.status != "running" or data["status"] not in TERMINAL_STATUSES:
                raise ValueError("Invalid job status transition")
            if data["status"] == "done" and (
                data.get("speedup") is None or not math.isfinite(data["speedup"]) or data["speedup"] <= 0
            ):
                raise ValueError("Completed jobs require positive finite speedup")

            changes = {key: data[key] for key in (
                "status", "speedup", "decode_tps", "prefill_tps", "variance_pct",
                "token_match", "max_logit_abs_diff", "baseline_tps", "engine_boot_id",
                "kernel_report", "extra", "error",
            ) if key in data}
            result = replace(job, finished_at=time.time(), lease_expires_at=None, **changes)
            worker._save(result)
            job.__dict__.update(result.__dict__)
            worker.durations.append(job.finished_at - job.started_at)
            if isinstance(data.get("baseline_tps"), (int, float)) and data["baseline_tps"] > 0:
                worker.baseline_tps = float(data["baseline_tps"])
            if data.get("crashes") is not None:
                worker.crashes = data["crashes"]
            self._maybe_cascade_invalidate_locked(job)

    def promote_job(
        self, jid: str, target_tier: str = "slow", agent: str | None = None, note: str = ""
    ) -> Job:
        with self.lock:
            orig = self.jobs[jid]
            if target_tier not in self.tiers:
                raise KeyError(f"unknown tier {target_tier}; have {list(self.tiers)}")
            source = Path(orig.candidate_path).read_text()
            who = agent or orig.agent
            desc = note.strip() or f"Promote {orig.tier} job {orig.id}: {orig.note}".strip()
            tags = list(orig.tags)
        return self.submit(who, target_tier, source, desc, parent_id=orig.id, parent_ids=[orig.id], tags=tags)

    _LEADERBOARD_SORTS = {
        "speedup",
        "decode_tps",
        "prefill_tps",
        "ttft_s",
        "decode_small",
        "decode_mid",
        "prefill_large",
        "output_tput",
        "active_decode_tps",
        "itl_p50_ms",
        "itl_p90_ms",
        "interactivity_p90",
        "spec_accept_length",
        "spec_accept_rate",
        "paired_speedup",
    }

    @staticmethod
    def _leaderboard_metric(j: Job, key: str) -> float | None:
        if key == "speedup":
            return j.speedup
        if key == "decode_tps":
            return j.decode_tps
        if key == "prefill_tps":
            return j.prefill_tps
        if key in ("decode_small", "decode_mid", "prefill_large"):
            bs = j.extra.get("bucket_speedups") if isinstance(j.extra, dict) else None
            if isinstance(bs, dict) and isinstance(bs.get(key), (int, float)):
                return float(bs[key])
            return None
        if key == "paired_speedup":
            paired = j.extra.get("paired") if isinstance(j.extra, dict) else None
            return paired.get("decode_duration_speedup") if isinstance(paired, dict) else None
        val = j.extra.get(key) if isinstance(j.extra, dict) else None
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _job_metrics(j: Job) -> dict:
        row: dict = {
            "speedup": round(j.speedup, 4) if j.speedup is not None else None,
            "decode_tps": round(j.decode_tps, 2) if j.decode_tps is not None else None,
            "prefill_tps": round(j.prefill_tps, 2) if j.prefill_tps is not None else None,
            "token_match": round(j.token_match, 4) if j.token_match is not None else None,
            "max_logit_abs_diff": (
                round(j.max_logit_abs_diff, 4) if j.max_logit_abs_diff is not None else None
            ),
        }
        ex = j.extra if isinstance(j.extra, dict) else {}
        for key, digits in (
            ("output_tput", 2),
            ("active_decode_tps", 2),
            ("itl_p50_ms", 3),
            ("itl_p90_ms", 3),
            ("interactivity_p90", 2),
            ("effective_concurrency", 3),
            ("spec_accept_length", 3),
            ("spec_accept_rate", 4),
        ):
            val = ex.get(key)
            if isinstance(val, (int, float)):
                row[key] = round(float(val), digits)
        for key in ("request_count", "submission_valid"):
            if key in ex and ex[key] is not None:
                row[key] = ex[key]
        paired = ex.get("paired")
        if isinstance(paired, dict) and isinstance(paired.get("decode_duration_speedup"), (int, float)):
            row["paired_decode_speedup"] = round(float(paired["decode_duration_speedup"]), 4)
        bs = ex.get("bucket_speedups")
        if isinstance(bs, dict) and bs:
            row["bucket_speedups"] = {str(k): round(float(v), 4) for k, v in bs.items() if isinstance(v, (int, float))}
        rf = ex.get("roofline")
        if isinstance(rf, dict) and rf.get("bottleneck_flags"):
            row["roofline_flags"] = list(rf["bottleneck_flags"])
        scr = ex.get("screening")
        if isinstance(scr, dict) and scr.get("stage") is not None:
            row["screen_stage"] = scr["stage"]
        return row

    @classmethod
    def _leaderboard_row(cls, j: Job) -> dict:
        metrics = cls._job_metrics(j)
        return {
            "id": j.id,
            "agent": j.agent,
            "variance_pct": round(j.variance_pct or 0, 2),
            "parent_id": j.parent_id,
            "parent_ids": list(j.parent_ids),
            "tags": list(j.tags),
            "note": j.note,
            **metrics,
        }

    def list_jobs(
        self,
        tier: str | None = None,
        status: str | None = None,
        agent: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        with self.lock:
            for rtier in (self.remote_workers if tier is None else ([tier] if tier in self.remote_workers else [])):
                self._expire_remote(rtier)
            rows = [
                j for j in self.jobs.values()
                if (tier is None or j.tier == tier)
                and (status is None or j.status == status)
                and (agent is None or j.agent == agent)
            ]
            rows.sort(key=lambda j: (j.submitted_at, j.id), reverse=True)
            out: list[dict] = []
            for j in rows[: max(1, limit)]:
                out.append({
                    "id": j.id,
                    "tier": j.tier,
                    "agent": j.agent,
                    "status": j.status,
                    "parent_id": j.parent_id,
                    "parent_ids": list(j.parent_ids),
                    "tags": list(j.tags),
                    **self._job_metrics(j),
                    "baseline_tps": round(j.baseline_tps, 2) if j.baseline_tps is not None else None,
                    "engine_boot_id": j.engine_boot_id,
                    "submitted_at": j.submitted_at,
                    "finished_at": j.finished_at,
                    "note": j.note,
                    "error": j.error,
                })
            return out

    def _valid_done_jobs_locked(self, tier: str) -> list[Job]:
        return [
            j for j in self.jobs.values()
            if j.tier == tier and j.status == "done"
            and (j.extra.get("metric") != "agentx_total_tput_per_gpu" or
                 ((j.extra.get("submission_valid") is True or j.extra.get("unsafe_override") is True)
                  and j.extra.get("repeats")))
        ]

    def leaderboard(self, tier: str = "slow", k: int = 10, sort_by: str = "speedup") -> list[dict]:
        if sort_by not in self._LEADERBOARD_SORTS:
            raise ValueError(f"unsupported sort_by {sort_by!r}; choose from {sorted(self._LEADERBOARD_SORTS)}")
        asc = sort_by in ("itl_p50_ms", "itl_p90_ms", "ttft_s")
        with self.lock:
            done = self._valid_done_jobs_locked(tier)

            def _sort_key(j: Job) -> tuple[int, float]:
                val = self._leaderboard_metric(j, sort_by)
                if val is None or not math.isfinite(val):
                    return (1, 0.0)
                return (0, float(val) if asc else -float(val))

            done.sort(key=_sort_key)
            return [self._leaderboard_row(j) for j in done[:k]]

    @staticmethod
    def _parents_of(j: Job) -> list[str]:
        out = list(j.parent_ids)
        if j.parent_id and j.parent_id not in out:
            out.insert(0, j.parent_id)
        return out

    def get_lineage(self, jid: str) -> dict:
        """Return ancestor chain, direct children, and lineage depth for `jid`."""
        with self.lock:
            root = self.jobs[jid]
            ancestors: list[dict] = []
            visited: set[str] = {jid}
            frontier: list[tuple[str, int]] = [(pid, 1) for pid in self._parents_of(root)]
            max_depth = 0
            while frontier:
                pid, dist = frontier.pop(0)
                if pid in visited:
                    continue
                visited.add(pid)
                max_depth = max(max_depth, dist)
                pjob = self.jobs.get(pid)
                if pjob is None:
                    ancestors.append({"id": pid, "missing": True, "distance": dist})
                    continue
                ancestors.append({
                    **self._leaderboard_row(pjob),
                    "tier": pjob.tier,
                    "status": pjob.status,
                    "distance": dist,
                })
                for gp in self._parents_of(pjob):
                    if gp not in visited:
                        frontier.append((gp, dist + 1))

            children = [
                {
                    **self._leaderboard_row(other),
                    "tier": other.tier,
                    "status": other.status,
                }
                for other in sorted(self.jobs.values(), key=lambda x: (x.submitted_at, x.id))
                if other.id != jid and jid in self._parents_of(other)
            ]
            return {
                "job": {
                    **self._leaderboard_row(root),
                    "tier": root.tier,
                    "status": root.status,
                },
                "depth": max_depth,
                "ancestors": ancestors,
                "children": children,
            }

    def archive(self, tier: str = "fast", k: int = 3) -> dict:
        """Build ShinkaEvolve-style multi-island archive & conflict-free orthogonal merge suggestions."""
        k = max(1, int(k))
        regime_labels = {
            "decode_small": "M<=48",
            "decode_mid": "M=65..256",
            "prefill_large": "M>512",
        }
        with self.lock:
            done = self._valid_done_jobs_locked(tier)

            def _top_by(pool: list[Job], metric_key: str) -> list[Job]:
                scored: list[tuple[float, Job]] = []
                for j in pool:
                    val = self._leaderboard_metric(j, metric_key)
                    if val is not None and math.isfinite(val):
                        scored.append((float(val), j))
                scored.sort(key=lambda item: (item[0], item[1].submitted_at), reverse=True)
                return [j for _, j in scored[:k]]

            overall_jobs = _top_by(done, "speedup")
            overall_rows = [self._leaderboard_row(j) for j in overall_jobs]

            regimes: dict[str, dict] = {}
            island_leaders: dict[str, list[str]] = {}
            job_by_id: dict[str, Job] = {j.id: j for j in done}

            if overall_jobs:
                island_leaders.setdefault(overall_jobs[0].id, []).append("overall")

            for rkey, rlabel in regime_labels.items():
                top_r = _top_by(done, rkey)
                if top_r:
                    island_leaders.setdefault(top_r[0].id, []).append(f"regime:{rkey}")
                regimes[rkey] = {
                    "label": rlabel,
                    "top": [self._leaderboard_row(j) for j in top_r],
                }

            all_tags = sorted({t for j in done for t in j.tags})
            submodules: dict[str, list[dict]] = {}
            for tag in all_tags:
                pool = [j for j in done if tag in j.tags]
                top_t = _top_by(pool, "speedup")
                if top_t:
                    island_leaders.setdefault(top_t[0].id, []).append(f"submodule:{tag}")
                submodules[tag] = [self._leaderboard_row(j) for j in top_t]

            stepping_stones: list[dict] = []
            stone_jobs: list[Job] = []
            for jid, islands in island_leaders.items():
                j = job_by_id[jid]
                stone_jobs.append(j)
                stepping_stones.append({
                    **self._leaderboard_row(j),
                    "islands": islands,
                })
            stepping_stones.sort(key=lambda r: (len(r["islands"]), r.get("speedup") or 0.0), reverse=True)

            # Inspect stepping stones (and overall leaders) for conflict-free orthogonal merge pairs.
            inspect_pool: list[Job] = []
            seen_pool: set[str] = set()
            for j in [*stone_jobs, *overall_jobs]:
                if j.id not in seen_pool:
                    seen_pool.add(j.id)
                    inspect_pool.append(j)
                if len(inspect_pool) >= 12:
                    break

            inspected_map: dict[str, dict] = {}
            for j in inspect_pool:
                try:
                    src = Path(j.candidate_path).read_text()
                    info = inspect_candidate_source(src)
                    if info.get("syntax_ok"):
                        inspected_map[j.id] = info
                except OSError:
                    continue

            suggestions: list[dict] = []
            for i in range(len(inspect_pool)):
                ja = inspect_pool[i]
                if ja.id not in inspected_map:
                    continue
                for idx_b in range(i + 1, len(inspect_pool)):
                    jb = inspect_pool[idx_b]
                    if jb.id not in inspected_map:
                        continue
                    if ja.id in self._parents_of(jb) or jb.id in self._parents_of(ja):
                        continue
                    if ja.source_sha256 and ja.source_sha256 == jb.source_sha256:
                        continue
                    conflicts = detect_candidate_conflicts([
                        (ja.id, inspected_map[ja.id]),
                        (jb.id, inspected_map[jb.id]),
                    ])
                    if conflicts:
                        continue
                    islands_a = set(island_leaders.get(ja.id, ()))
                    islands_b = set(island_leaders.get(jb.id, ()))
                    tags_a = set(ja.tags)
                    tags_b = set(jb.tags)
                    # Prefer pairs covering distinct submodules or distinct regimes
                    distinct_tags = (tags_a != tags_b) and bool(tags_a or tags_b)
                    distinct_islands = bool(islands_a - islands_b) and bool(islands_b - islands_a)
                    if not (distinct_tags or distinct_islands):
                        continue
                    sp_a = float(ja.speedup or 1.0)
                    sp_b = float(jb.speedup or 1.0)
                    # Estimate upside using best metric across regimes if one specializes in a regime
                    best_a = max([sp_a, *[float(v) for v in ((ja.extra or {}).get("bucket_speedups") or {}).values()]])
                    best_b = max([sp_b, *[float(v) for v in ((jb.extra or {}).get("bucket_speedups") or {}).values()]])
                    suggestions.append({
                        "job_ids": [ja.id, jb.id],
                        "tags_a": list(ja.tags),
                        "tags_b": list(jb.tags),
                        "islands_a": sorted(islands_a),
                        "islands_b": sorted(islands_b),
                        "est_combined_speedup": round(sp_a * sp_b, 4),
                        "est_regime_upside": round(best_a * best_b, 4),
                    })

            suggestions.sort(
                key=lambda s: (s["est_regime_upside"], s["est_combined_speedup"]),
                reverse=True,
            )

            pd_pairs: list[dict] = []
            req_worker = self.tiers.get(tier)
            prefill_tiers = (
                [tier]
                if (req_worker is not None and req_worker.tier.phase == "prefill")
                else [t for t, w in self.tiers.items() if w.tier.phase == "prefill"]
            )
            decode_tiers = (
                [tier]
                if (req_worker is not None and req_worker.tier.phase == "decode")
                else [t for t, w in self.tiers.items() if w.tier.phase == "decode"]
            )
            if prefill_tiers and decode_tiers:
                p_done = [j for pt in prefill_tiers for j in self._valid_done_jobs_locked(pt)]
                d_done = [j for dt in decode_tiers for j in self._valid_done_jobs_locked(dt)]
                top_p = _top_by(p_done, "speedup")
                top_d = _top_by(d_done, "speedup")
                for jp in top_p[:2]:
                    for jd in top_d[:2]:
                        pd_pairs.append({
                            "prefill_job_id": jp.id,
                            "prefill_tier": jp.tier,
                            "prefill_speedup": round(float(jp.speedup or 1.0), 4),
                            "decode_job_id": jd.id,
                            "decode_tier": jd.tier,
                            "decode_speedup": round(float(jd.speedup or 1.0), 4),
                        })

            return {
                "tier": tier,
                "total_done": len(done),
                "overall": overall_rows,
                "regimes": regimes,
                "submodules": submodules,
                "stepping_stones": stepping_stones,
                "merge_suggestions": suggestions[:k],
                "pd_pair_suggestions": pd_pairs[:k],
            }

    def compose_jobs(self, job_ids: list[str], allow_target_overlap: bool = False) -> dict:
        """Synthesize a merged orthogonal candidate module from existing `job_ids`."""
        with self.lock:
            items: list[tuple[str, str]] = []
            for jid in job_ids:
                job = self.jobs[jid]
                items.append((job.id, Path(job.candidate_path).read_text()))
        return compose_candidates(items, allow_target_overlap=allow_target_overlap)

    def compose_pd_jobs(self, prefill_job_id: str, decode_job_id: str) -> dict:
        """Synthesize a PD role-dispatched bundle from a prefill job and a decode job."""
        with self.lock:
            p_job = self.jobs[prefill_job_id]
            d_job = self.jobs[decode_job_id]
            p_item = (p_job.id, Path(p_job.candidate_path).read_text())
            d_item = (d_job.id, Path(d_job.candidate_path).read_text())
        return compose_pd_pair(p_item, d_item)


def run_remote_worker(
    run: RunConfig,
    tier_name: str,
    server_url: str,
    *,
    extra_tiers: list[str] | None = None,
    include_shared: bool = True,
) -> None:
    """Run one persistent tier engine (and any tiers sharing it) on a GPU worker node."""
    import httpx
    from kiln.bench.traffic import load_traffic
    from kiln.server.auth import auth_headers

    tier_by_name = {t.name: t for t in run.tiers}
    names: list[str] = []

    def _add_name(name: str) -> None:
        if name not in names:
            names.append(name)

    for name in [tier_name, *(extra_tiers or [])]:
        if name not in tier_by_name:
            raise KeyError(f"unknown tier {name}; have {[t.name for t in run.tiers]}")
        _add_name(name)
    if include_shared:
        for name in list(names):
            owner_name = tier_by_name[name].share_engine
            if owner_name:
                _add_name(owner_name)
        for t in run.tiers:
            if t.share_engine and t.share_engine in names:
                _add_name(t.name)

    client = httpx.Client(trust_env=False, timeout=30.0, headers=auth_headers())
    base_url = server_url.rstrip("/")
    traffic_exists = Path(run.traffic_file).exists()
    cached_requests: list[Request] | None = None
    workers: dict[str, TierWorker] = {}
    for name in names:
        t = tier_by_name[name]
        if traffic_exists:
            if cached_requests is None:
                tok_name = None if (t.engine_kind or run.engine.kind) == "mock" else (run.engine.tokenizer or run.engine.model)
                cached_requests = load_traffic(run.traffic_file, model_name=tok_name)
            tier_requests = cached_requests
        else:
            r = client.get(f"{base_url}/workers/{t.name}/requests")
            r.raise_for_status()
            tier_requests = [Request(**row) for row in r.json()]
        jobs_dir = Path(run.workdir) / run.name / f"worker_{t.name}" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        workers[t.name] = TierWorker(run, t, tier_requests, jobs_dir, local=False)

    _wire_shared_tiers(workers, [tier_by_name[n] for n in names], enforce_same_locality=False)
    boot_order = [workers[n] for n in names if workers[n].owner is None] + [
        workers[n] for n in names if workers[n].owner is not None
    ]
    poll_tiers = [tier_by_name[n] for n in names if tier_by_name[n].share_engine] + [
        tier_by_name[n] for n in names if not tier_by_name[n].share_engine
    ]

    worker_id = uuid.uuid4().hex
    active_lock = threading.Lock()
    active_tier: str | None = None
    attempt_id: str | None = None
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            for t_name, w in workers.items():
                with active_lock:
                    cur_attempt = attempt_id if active_tier == t_name else None
                try:
                    client.post(f"{base_url}/workers/{t_name}/heartbeat", timeout=5.0, json={
                        "worker_id": worker_id, "attempt_id": cur_attempt,
                        "ready": w.ready.is_set(), "baseline_tps": w.baseline_tps,
                        "crashes": w.crashes,
                        "boot_id": w.boot_id, "last_canary": w.last_canary,
                    }).raise_for_status()
                except httpx.HTTPError as exc:
                    print(f"[kiln] {t_name}: heartbeat failed ({exc})")
            stop.wait(REMOTE_HEARTBEAT_S)

    heartbeat_thread = threading.Thread(target=heartbeat, name=f"heartbeat-{'-'.join(names)}", daemon=True)
    heartbeat_thread.start()
    try:
        while True:
            for w in boot_order:
                w.wait_until_ready()
            claimed = None
            claimed_worker: TierWorker | None = None
            for idx, t in enumerate(poll_tiers):
                poll_wait = 2.0 if idx == len(poll_tiers) - 1 else 0.0
                try:
                    r = client.post(
                        f"{base_url}/workers/{t.name}/claim",
                        params={"worker_id": worker_id, "wait_s": poll_wait},
                    )
                    r.raise_for_status()
                    claimed = r.json().get("job")
                    if claimed:
                        claimed_worker = workers[t.name]
                        break
                except httpx.HTTPError as e:
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                        raise
                    print(f"[kiln] {t.name}: server poll error ({e}); retrying in 5s")
                    time.sleep(5)
                    claimed = None
                    break
            if not claimed or claimed_worker is None:
                for w in boot_order:
                    if w.owner is None:
                        w._maybe_rebuild_idle_baseline()
                continue
            t_name = claimed_worker.tier.name
            with active_lock:
                active_tier = t_name
                attempt_id = claimed["attempt_id"]
            try:
                jid = claimed["id"]
                cand_dir = claimed_worker.jobs_dir / jid
                cand_dir.mkdir(parents=True, exist_ok=True)
                cand_path = cand_dir / "candidate.py"
                cand_path.write_text(claimed["source"])
                job_extra: dict = {}
                st_val = claimed.get("screen_threshold")
                if isinstance(st_val, (int, float)) and math.isfinite(st_val) and st_val > 0:
                    job_extra["_screen_threshold"] = float(st_val)
                job = Job(
                    id=jid,
                    agent=claimed["agent"],
                    tier=t_name,
                    note=claimed.get("note", ""),
                    candidate_path=str(cand_path),
                    force_trace=bool(claimed.get("force_trace", False)),
                    extra=job_extra,
                )
                claimed_worker.execute_job(job)
                payload = asdict(job)
                payload.update(worker_id=worker_id, attempt_id=attempt_id, crashes=claimed_worker.crashes)
                while True:
                    try:
                        r = client.post(f"{base_url}/workers/{t_name}/complete", json=payload)
                        if r.status_code == 409:
                            print(f"[kiln] {t_name}: claim expired for {jid}; coordinator will retry it")
                            break
                        r.raise_for_status()
                        break
                    except httpx.HTTPError as exc:
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                            raise
                        print(f"[kiln] {t_name}: result delivery failed for {jid} ({exc}); retrying")
                        time.sleep(5)
            finally:
                with active_lock:
                    active_tier = None
                    attempt_id = None
    finally:
        stop.set()
        heartbeat_thread.join()
        client.close()
        for w in workers.values():
            w.shutdown_prebuild()
            if w.owner is None and w.engine:
                w.engine.shutdown()
