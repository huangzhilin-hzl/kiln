"""Per-tier job queue with a persistent engine (TKCC §4.2).

One worker thread per tier owns one persistent engine. Each job is handled as:
    hot_reload -> correctness check -> measure -> teardown
If the engine crashes, it restarts automatically and the job is marked as crashed.
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import statistics
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from kiln.bench.correctness import evaluate_correctness
from kiln.bench.traffic import Request
from kiln.config import RunConfig, TierConfig
from kiln.engine import Engine, make_engine
from kiln.engine.base import InvalidBenchmark, MeasureResult


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
    worker_id: str | None = None
    attempt_id: str | None = None
    lease_expires_at: float | None = None


REMOTE_LEASE_S = 90.0
REMOTE_HEARTBEAT_S = 10.0
TERMINAL_STATUSES = {"done", "incorrect", "invalid", "crashed", "invalidated"}


_FATAL_KEYWORDS = (
    "illegal memory", "cudaerror", "nccl", "device-side assert",
    "timed out", "timeout", "exited", "connection", "broken pipe", "eof",
)


class SharedEngineAdapter(Engine):
    """Proxy engine that serializes onto an owning tier's persistent Engine."""

    def __init__(self, owner: "TierWorker", tier: TierConfig):
        self.owner = owner
        self.tier = tier

    def start(self) -> None:
        if self.owner.engine is None:
            self.owner._ensure_own_engine()

    def hot_reload(self, candidate_path: str) -> None:
        self.owner.engine.hot_reload(candidate_path)

    def teardown_candidate(self) -> None:
        self.owner.engine.teardown_candidate()

    def measure(self, requests: list[Request], repeats: int, trace: bool) -> MeasureResult:
        eng = self.owner.engine
        if self.tier.agentx is None and hasattr(eng, "measure_requests"):
            return eng.measure_requests(requests, repeats, trace)
        return eng.measure(requests, repeats, trace)

    def greedy_outputs(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return self.owner.engine.greedy_outputs(prompts, max_new_tokens)

    def greedy_probe(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> tuple[list[list[int]], list[list[dict[int, float]]] | None]:
        return self.owner.engine.greedy_probe(prompts, max_new_tokens)

    def shutdown(self) -> None:
        self.owner._shutdown_own_engine()


class TierWorker:
    def __init__(self, run: RunConfig, tier: TierConfig, requests: list[Request], jobs_dir: Path,
                 local: bool = True, on_job_finished: Callable[[Job], None] | None = None):
        self.run, self.tier, self.jobs_dir, self.local = run, tier, jobs_dir, local
        self.on_job_finished = on_job_finished
        self.requests = requests[: tier.max_requests] if tier.max_requests else requests
        self.q: queue.Queue[Job] = queue.Queue()
        self.engine: Engine | None = None
        self.owner: TierWorker | None = None
        self.engine_lock = threading.RLock()
        self.baseline_tps: float | None = None
        self.baseline_tokens: list[list[int]] | None = None
        self.baseline_logprobs: list[list[dict[int, float]]] | None = None
        self.durations: list[float] = []
        self.crashes = 0
        self.boot_error: str | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"tier-{tier.name}", daemon=True)

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

    def _boot(self) -> None:
        with self.engine_lock:
            self.ready.clear()
            if self.owner is None:
                self._ensure_own_engine()
            else:
                self.owner._ensure_own_engine()
                self.engine = SharedEngineAdapter(self.owner, self.tier)
            if self.baseline_tokens is None or self.baseline_tps is None:
                cc = self.run.correctness
                probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
                tokens, logprobs = self.engine.greedy_probe(probe, cc.max_new_tokens)
                if not probe or len(tokens) != len(probe) or not all(tokens):
                    raise RuntimeError("Baseline returned incomplete correctness outputs")
                base = self.engine.measure(self.requests, self.tier.repeats, trace=False)
                if not math.isfinite(base.decode_tps) or base.decode_tps <= 0:
                    raise RuntimeError("Baseline throughput must be finite and positive")
                self.baseline_tokens = tokens
                self.baseline_logprobs = logprobs
                self.baseline_tps = base.decode_tps
                print(f"[kiln] {self.tier.name}: baseline decode {base.decode_tps:.1f} tok/s "
                      f"(variance {base.variance_pct:.2f}%)")
            self.boot_error = None
            self.ready.set()

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
                if owner_was_ready:
                    self.owner.ready.set()
                self.ready.set()

    def _recover_or_restart(self, err: Exception) -> None:
        self.crashes += 1
        msg = str(err).lower()
        if not any(k in msg for k in _FATAL_KEYWORDS) and self.engine and self.baseline_tokens:
            try:
                self.engine.teardown_candidate()
                n_check = min(4, self.run.correctness.max_new_tokens)
                got = self.engine.greedy_outputs([self.requests[0].prompt_ids], n_check)
                if got == [self.baseline_tokens[0][:n_check]] and len(got[0]) == n_check:
                    print(f"[kiln] {self.tier.name}: recovered warm engine after non-fatal candidate error")
                    return
            except Exception:
                pass
        self._restart_engine()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        self.wait_until_ready()
        while True:
            job = self.q.get()
            with self.engine_lock:
                self.wait_until_ready()
                self.execute_job(job)

    def wait_until_ready(self) -> None:
        while not self.ready.is_set():
            try:
                self._boot()
            except Exception as exc:
                if self.boot_error != str(exc):
                    print(f"[kiln] {self.tier.name}: boot failed ({exc}); retrying")
                self.boot_error = str(exc)
                time.sleep(5)

    def execute_job(self, job: Job) -> None:
        with self.engine_lock:
            job.status, job.started_at = "running", time.time()
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
                self._save(result)
                job.__dict__.update(result.__dict__)
                self.durations.append(result.finished_at - result.started_at)
                if self.on_job_finished is not None:
                    self.on_job_finished(job)

    def _run_job(self, job: Job) -> None:
        eng, cc = self.engine, self.run.correctness
        eng.hot_reload(job.candidate_path)
        try:
            probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
            cand_tokens, cand_logprobs = eng.greedy_probe(probe, cc.max_new_tokens)
            corr = evaluate_correctness(
                self.baseline_tokens, cand_tokens, cc, self.baseline_logprobs, cand_logprobs
            )
            job.token_match = corr.token_match
            job.max_logit_abs_diff = corr.max_logit_abs_diff
            if not corr.ok:
                job.status, job.error = "incorrect", corr.detail
                return
            try:
                m = eng.measure(self.requests, self.tier.repeats, trace=self.tier.trace_kernels)
            except InvalidBenchmark as exc:
                job.status, job.error, job.extra = "invalid", str(exc), exc.extra
                return
            if not math.isfinite(m.decode_tps) or m.decode_tps <= 0:
                job.status, job.error = "invalid", "Candidate throughput must be finite and positive"
                return
            job.decode_tps, job.prefill_tps = m.decode_tps, m.prefill_tps
            job.variance_pct, job.kernel_report = m.variance_pct, m.kernel_report
            job.extra = m.extra
            job.speedup = m.decode_tps / self.baseline_tps
        finally:
            eng.teardown_candidate()
        job.status = "done"

    # ------------------------------------------------------------------ api
    def submit(self, job: Job) -> None:
        self._save(job)
        self.q.put(job)

    def stats(self) -> dict:
        mode = f"shared:{self.tier.share_engine}" if self.tier.share_engine else ("local" if self.local else "remote")
        return {
            "tier": self.tier.name,
            "mode": mode,
            "ready": self.ready.is_set(),
            "boot_error": self.boot_error,
            "queue_depth": self.q.qsize(),
            "median_job_s": statistics.median(self.durations) if self.durations else None,
            "baseline_decode_tps": self.baseline_tps,
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
            )
            for t in run.tiers
        }
        for t in run.tiers:
            if not t.share_engine:
                continue
            if t.share_engine == t.name or t.share_engine not in self.tiers:
                raise ValueError(f"invalid share_engine {t.share_engine!r} on tier {t.name!r}")
            owner = self.tiers[t.share_engine]
            if owner.tier.share_engine:
                raise ValueError(f"chained share_engine not supported: {t.name} -> {t.share_engine}")
            if not owner.local:
                raise ValueError(f"tier {t.name} cannot share non-local engine {t.share_engine}")
            borrower = self.tiers[t.name]
            borrower.owner = owner
            borrower.engine_lock = owner.engine_lock
            if borrower.local and t.trace_kernels and not owner.tier.trace_kernels:
                owner.tier = owner.tier.model_copy(update={"trace_kernels": True})
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
                worker.q.put(job)
            elif job.finished_at and job.started_at:
                worker.durations.append(job.finished_at - job.started_at)

    def start(self) -> None:
        for w in self.tiers.values():
            if w.local:
                w.thread.start()

    def submit(self, agent: str, tier: str, source: str, note: str = "") -> Job:
        if tier not in self.tiers:
            raise KeyError(f"unknown tier {tier}; have {list(self.tiers)}")
        jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = self.jobs_dir / jid
        d.mkdir(parents=True)
        path = d / "candidate.py"
        path.write_text(source)
        job = Job(
            id=jid, agent=agent, tier=tier, note=note,
            candidate_path=str(path), source_sha256=_sha256_text(source),
        )
        with self.lock:
            self.tiers[tier].submit(job)
            self.jobs[jid] = job
            self.lock.notify_all()
        return job

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
                         attempt_id: str | None = None) -> None:
        worker = self.tiers[tier]
        with self.lock:
            self._expire_remote(tier)
            expires_at = time.time() + REMOTE_LEASE_S
            if ready:
                self.remote_workers[tier][worker_id] = expires_at
                worker.ready.set()
                worker.baseline_tps = baseline_tps
                worker.crashes = crashes
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
                    job.status, job.started_at = "running", time.time()
                    job.worker_id = worker_id
                    job.attempt_id = uuid.uuid4().hex
                    break
            job.lease_expires_at = time.time() + REMOTE_LEASE_S
            worker._save(job)
            return {
                "id": job.id, "agent": job.agent, "note": job.note,
                "attempt_id": job.attempt_id,
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
                "token_match", "max_logit_abs_diff", "kernel_report", "extra", "error",
            ) if key in data}
            result = replace(job, finished_at=time.time(), lease_expires_at=None, **changes)
            worker._save(result)
            job.__dict__.update(result.__dict__)
            worker.durations.append(job.finished_at - job.started_at)
            if data.get("crashes") is not None:
                worker.crashes = data["crashes"]
            self._maybe_cascade_invalidate_locked(job)

    def leaderboard(self, tier: str = "slow", k: int = 10) -> list[dict]:
        with self.lock:
            done = [j for j in self.jobs.values()
                    if j.tier == tier and j.status == "done"
                    and (j.extra.get("metric") != "agentx_total_tput_per_gpu" or
                         ((j.extra.get("submission_valid") is True or j.extra.get("unsafe_override") is True)
                          and j.extra.get("repeats")))]
            done.sort(key=lambda j: j.speedup or 0, reverse=True)
            return [{"id": j.id, "agent": j.agent, "speedup": round(j.speedup, 4),
                     "variance_pct": round(j.variance_pct or 0, 2), "note": j.note} for j in done[:k]]


def run_remote_worker(run: RunConfig, tier_name: str, server_url: str) -> None:
    """Run one persistent tier engine on a second GPU node, pulling jobs from `server_url`."""
    import httpx
    from kiln.bench.traffic import load_traffic
    from kiln.server.auth import auth_headers

    tier = next((t for t in run.tiers if t.name == tier_name), None)
    if tier is None:
        raise KeyError(f"unknown tier {tier_name}; have {[t.name for t in run.tiers]}")

    client = httpx.Client(trust_env=False, timeout=30.0, headers=auth_headers())
    url = f"{server_url.rstrip('/')}/workers/{tier_name}"

    if Path(run.traffic_file).exists():
        tok_name = None if (tier.engine_kind or run.engine.kind) == "mock" else (run.engine.tokenizer or run.engine.model)
        requests = load_traffic(run.traffic_file, model_name=tok_name)
    else:
        r = client.get(f"{url}/requests")
        r.raise_for_status()
        requests = [Request(**row) for row in r.json()]

    jobs_dir = Path(run.workdir) / run.name / f"worker_{tier_name}" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    worker = TierWorker(run, tier, requests, jobs_dir, local=False)

    worker_id = uuid.uuid4().hex
    attempt_id = None
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            try:
                client.post(f"{url}/heartbeat", timeout=5.0, json={
                    "worker_id": worker_id, "attempt_id": attempt_id,
                    "ready": worker.ready.is_set(), "baseline_tps": worker.baseline_tps,
                    "crashes": worker.crashes,
                }).raise_for_status()
            except httpx.HTTPError as exc:
                print(f"[kiln] {tier_name}: heartbeat failed ({exc})")
            stop.wait(REMOTE_HEARTBEAT_S)

    heartbeat_thread = threading.Thread(target=heartbeat, name=f"heartbeat-{tier_name}", daemon=True)
    heartbeat_thread.start()
    try:
        while True:
            worker.wait_until_ready()
            try:
                r = client.post(f"{url}/claim", params={"worker_id": worker_id, "wait_s": 2.0})
                r.raise_for_status()
                claimed = r.json().get("job")
            except httpx.HTTPError as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise
                print(f"[kiln] {tier_name}: server poll error ({e}); retrying in 5s")
                time.sleep(5)
                continue
            if not claimed:
                continue
            attempt_id = claimed["attempt_id"]
            jid = claimed["id"]
            cand_dir = jobs_dir / jid
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand_path = cand_dir / "candidate.py"
            cand_path.write_text(claimed["source"])
            job = Job(id=jid, agent=claimed["agent"], tier=tier_name,
                      note=claimed.get("note", ""), candidate_path=str(cand_path))
            worker.execute_job(job)
            payload = asdict(job)
            payload.update(worker_id=worker_id, attempt_id=attempt_id, crashes=worker.crashes)
            while True:
                try:
                    r = client.post(f"{url}/complete", json=payload)
                    if r.status_code == 409:
                        print(f"[kiln] {tier_name}: claim expired for {jid}; coordinator will retry it")
                        break
                    r.raise_for_status()
                    break
                except httpx.HTTPError as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                        raise
                    print(f"[kiln] {tier_name}: result delivery failed for {jid} ({exc}); retrying")
                    time.sleep(5)
            attempt_id = None
    finally:
        stop.set()
        heartbeat_thread.join()
        client.close()
        if worker.engine:
            worker.engine.shutdown()
