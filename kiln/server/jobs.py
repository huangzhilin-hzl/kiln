"""Per-tier job queue with a persistent engine (TKCC §4.2).

One worker thread per tier owns one persistent engine. Each job is handled as:
    hot_reload -> correctness check -> measure -> teardown
If the engine crashes, it restarts automatically and the job is marked as crashed.
"""
from __future__ import annotations

import json
import queue
import statistics
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kiln.bench.correctness import compare_tokens
from kiln.bench.traffic import Request
from kiln.config import RunConfig, TierConfig
from kiln.engine import Engine, make_engine


@dataclass
class Job:
    id: str
    agent: str
    tier: str
    note: str
    candidate_path: str
    status: str = "queued"          # queued | running | done | incorrect | crashed
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    speedup: float | None = None
    decode_tps: float | None = None
    prefill_tps: float | None = None
    variance_pct: float | None = None
    token_match: float | None = None
    kernel_report: str | None = None
    extra: dict = field(default_factory=dict)
    error: str | None = None


class TierWorker:
    def __init__(self, run: RunConfig, tier: TierConfig, requests: list[Request], jobs_dir: Path):
        self.run, self.tier, self.requests, self.jobs_dir = run, tier, requests, jobs_dir
        self.q: queue.Queue[Job] = queue.Queue()
        self.engine: Engine | None = None
        self.baseline_tps: float | None = None
        self.baseline_tokens: list[list[int]] | None = None
        self.durations: list[float] = []
        self.crashes = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"tier-{tier.name}", daemon=True)

    # ------------------------------------------------------------------ engine
    def _boot(self) -> None:
        self.engine = make_engine(self.run.engine, self.tier)
        self.engine.start()
        cc = self.run.correctness
        probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
        self.baseline_tokens = self.engine.greedy_outputs(probe, cc.max_new_tokens)
        base = self.engine.measure(self.requests, self.tier.repeats, trace=False)
        self.baseline_tps = base.decode_tps
        print(f"[kiln] {self.tier.name}: baseline decode {base.decode_tps:.1f} tok/s "
              f"(variance {base.variance_pct:.2f}%)")
        self.ready.set()

    def _restart(self) -> None:
        self.crashes += 1
        try:
            if self.engine:
                self.engine.shutdown()
        except Exception:
            pass
        self._boot()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        self._boot()
        while True:
            job = self.q.get()
            job.status, job.started_at = "running", time.time()
            self._save(job)
            try:
                self._run_job(job)
            except Exception as e:  # crash / IMA / hang surfaced as exception
                job.status, job.error = "crashed", f"{e}\n{traceback.format_exc()[-2000:]}"
                self._save(job)
                self._restart()
            finally:
                job.finished_at = time.time()
                self.durations.append(job.finished_at - job.started_at)
                self._save(job)

    def _run_job(self, job: Job) -> None:
        eng, cc = self.engine, self.run.correctness
        eng.hot_reload(job.candidate_path)
        try:
            probe = [r.prompt_ids for r in self.requests[: cc.num_prompts]]
            cand = eng.greedy_outputs(probe, cc.max_new_tokens)
            corr = compare_tokens(self.baseline_tokens, cand, cc)
            job.token_match = corr.token_match
            if not corr.ok:
                job.status, job.error = "incorrect", corr.detail
                return
            m = eng.measure(self.requests, self.tier.repeats, trace=self.tier.trace_kernels)
            job.decode_tps, job.prefill_tps = m.decode_tps, m.prefill_tps
            job.variance_pct, job.kernel_report = m.variance_pct, m.kernel_report
            job.extra = m.extra
            job.speedup = m.decode_tps / self.baseline_tps
            job.status = "done"
        finally:
            eng.teardown_candidate()

    # ------------------------------------------------------------------ api
    def submit(self, job: Job) -> None:
        self._save(job)
        self.q.put(job)

    def stats(self) -> dict:
        return {
            "tier": self.tier.name,
            "ready": self.ready.is_set(),
            "queue_depth": self.q.qsize(),
            "median_job_s": statistics.median(self.durations) if self.durations else None,
            "baseline_decode_tps": self.baseline_tps,
            "crashes": self.crashes,
            "layers": self.tier.num_layers or "all",
        }

    def _save(self, job: Job) -> None:
        d = self.jobs_dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(asdict(job), indent=2))


class JobManager:
    def __init__(self, run: RunConfig, requests: list[Request]):
        self.run = run
        self.root = Path(run.workdir) / run.name
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.tiers = {t.name: TierWorker(run, t, requests, self.jobs_dir) for t in run.tiers}
        self.lock = threading.Lock()

    def start(self) -> None:
        for w in self.tiers.values():
            w.thread.start()

    def submit(self, agent: str, tier: str, source: str, note: str = "") -> Job:
        if tier not in self.tiers:
            raise KeyError(f"unknown tier {tier}; have {list(self.tiers)}")
        jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = self.jobs_dir / jid
        d.mkdir(parents=True)
        path = d / "candidate.py"
        path.write_text(source)
        job = Job(id=jid, agent=agent, tier=tier, note=note, candidate_path=str(path))
        with self.lock:
            self.jobs[jid] = job
        self.tiers[tier].submit(job)
        return job

    def leaderboard(self, tier: str = "slow", k: int = 10) -> list[dict]:
        done = [j for j in self.jobs.values() if j.tier == tier and j.status == "done"]
        done.sort(key=lambda j: j.speedup or 0, reverse=True)
        return [{"id": j.id, "agent": j.agent, "speedup": round(j.speedup, 4),
                 "variance_pct": round(j.variance_pct or 0, 2), "note": j.note} for j in done[:k]]
