"""HTTP API used by agents (via the `kiln` CLI or curl)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from hmac import compare_digest
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, FiniteFloat

from kiln.bench.traffic import load_traffic
from kiln.config import RunConfig
from kiln.micro.sandbox import MicroRunner
from kiln.patches import PatchConflictError

from .board import Board
from .auth import auth_headers
from .jobs import DuplicateJobError, JobManager, QueueLimitError


class SubmitReq(BaseModel):
    agent: str
    tier: str = "fast"
    source: str
    note: str = ""
    parent_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    force_trace: bool = False
    supersede: bool = False


class ComposeReq(BaseModel):
    job_ids: list[str] = Field(default_factory=list)
    allow_target_overlap: bool = False
    prefill_job_id: str | None = None
    decode_job_id: str | None = None


class ComposePDReq(BaseModel):
    prefill_job_id: str = Field(min_length=1)
    decode_job_id: str = Field(min_length=1)


class MicroReq(BaseModel):
    agent: str
    source: str
    note: str = ""
    snapshot_tag: str | None = None
    enable_ncu: bool | None = None
    enable_ptxas: bool | None = None
    mode: Literal["numeric", "timing"] = "numeric"
    timeout_s: FiniteFloat | None = Field(default=None, gt=0)


class PostReq(BaseModel):
    agent: str
    text: str
    kind: str = "note"


class HeartbeatReq(BaseModel):
    worker_id: str = Field(min_length=1)
    attempt_id: str | None = None
    ready: bool
    baseline_tps: FiniteFloat | None = None
    crashes: int = 0
    boot_id: str | None = None
    last_canary: dict | None = None


class CompleteReq(BaseModel):
    id: str
    worker_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    status: Literal["done", "incorrect", "invalid", "crashed"]
    speedup: FiniteFloat | None = None
    decode_tps: FiniteFloat | None = None
    prefill_tps: FiniteFloat | None = None
    variance_pct: FiniteFloat | None = None
    token_match: FiniteFloat | None = None
    max_logit_abs_diff: FiniteFloat | None = None
    baseline_tps: FiniteFloat | None = None
    engine_boot_id: str | None = None
    kernel_report: str | None = None
    extra: dict = Field(default_factory=dict)
    error: str | None = None
    crashes: int | None = None


class InvalidateReq(BaseModel):
    reason: str = ""
    cascade: bool = True
    cascade_ast: list[str] = Field(default_factory=list)
    cancel_queued: bool = True


class CertifyReq(BaseModel):
    agent: str
    patch: str | None = None
    func: str | None = None
    scope: str = ""
    note: str = ""


class PromoteReq(BaseModel):
    target_tier: str = "slow"
    agent: str | None = None
    note: str = ""


def create_app(run: RunConfig, local_tiers: set[str] | None = None) -> FastAPI:
    if run.hw_profile:
        run.resolve_hw_profile()
    tok_name = None if run.engine.kind == "mock" else (run.engine.tokenizer or run.engine.model)
    requests = load_traffic(run.traffic_file, model_name=tok_name)
    mgr = JobManager(run, requests, local_tiers=local_tiers)
    board = Board(mgr.root / "board.jsonl")
    micro = MicroRunner(run, mgr.root)
    expected_auth = auth_headers().get("Authorization")

    def authenticate(authorization: Annotated[str | None, Header()] = None):
        if expected_auth and not compare_digest((authorization or "").encode(), expected_auth.encode()):
            raise HTTPException(401, "A valid bearer token is required",
                                headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            mgr.start()
            yield
        finally:
            mgr.shutdown()

    app = FastAPI(title="kiln", lifespan=lifespan, dependencies=[Depends(authenticate)])

    def require_remote(tier: str) -> None:
        if tier not in mgr.tiers:
            raise HTTPException(404, f"unknown tier {tier}")
        if mgr.tiers[tier].local:
            raise HTTPException(409, f"tier {tier} runs locally")

    @app.get("/status")
    def status():
        return mgr.status()

    @app.get("/hardware")
    def hardware():
        path = run.resolve_hw_profile()
        if path is None:
            return {"available": False, "profile": None}
        from kiln.costmodel.probe import HardwareProfile
        return {"available": True, "path": str(path), "profile": HardwareProfile.load(path).to_dict()}

    @app.get("/jobs")
    def list_jobs(
        tier: str | None = None,
        status: str | None = None,
        agent: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ):
        return mgr.list_jobs(tier=tier, status=status, agent=agent, limit=limit)

    @app.post("/jobs")
    def submit(req: SubmitReq):
        try:
            job = mgr.submit(
                req.agent,
                req.tier,
                req.source,
                req.note,
                parent_id=req.parent_id,
                parent_ids=req.parent_ids,
                tags=req.tags,
                force_trace=req.force_trace,
                supersede=req.supersede,
            )
        except KeyError as e:
            raise HTTPException(400, str(e))
        except QueueLimitError as e:
            raise HTTPException(429, str(e))
        except DuplicateJobError as e:
            raise HTTPException(409, str(e))
        return {
            "id": job.id,
            "tier": job.tier,
            "parent_id": job.parent_id,
            "parent_ids": job.parent_ids,
            "tags": job.tags,
            "force_trace": job.force_trace,
            "superseded": list((job.extra or {}).get("superseded") or []),
        }

    @app.get("/jobs/{jid}")
    def get_job(jid: str):
        try:
            return mgr.get_job(jid)
        except KeyError:
            raise HTTPException(404, "no such job")

    @app.get("/jobs/{jid}/lineage")
    def get_job_lineage(jid: str):
        try:
            return mgr.get_lineage(jid)
        except KeyError:
            raise HTTPException(404, "no such job")

    @app.get("/jobs/{jid}/source")
    def get_job_source(jid: str):
        try:
            return mgr.get_job_source(jid)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "no such job or source")

    @app.get("/archive")
    def archive(tier: str = "fast", k: Annotated[int, Query(ge=1, le=20)] = 3):
        return mgr.archive(tier=tier, k=k)

    @app.post("/ensembles/compose")
    def compose_ensemble(req: ComposeReq):
        try:
            if req.prefill_job_id or req.decode_job_id:
                if not (req.prefill_job_id and req.decode_job_id) or req.job_ids:
                    raise HTTPException(
                        400,
                        "Provide both prefill_job_id and decode_job_id (without job_ids), or job_ids",
                    )
                return mgr.compose_pd_jobs(req.prefill_job_id, req.decode_job_id)
            return mgr.compose_jobs(req.job_ids, allow_target_overlap=req.allow_target_overlap)
        except KeyError as e:
            raise HTTPException(404, f"no such job: {e}")
        except PatchConflictError as e:
            raise HTTPException(409, str(e))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @app.post("/ensembles/compose_pd")
    def compose_pd_ensemble(req: ComposePDReq):
        try:
            return mgr.compose_pd_jobs(req.prefill_job_id, req.decode_job_id)
        except KeyError as e:
            raise HTTPException(404, f"no such job: {e}")
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @app.get("/micro/snapshots")
    def micro_snapshots(tag: str | None = None):
        return micro.list_snapshots(tag=tag)

    @app.post("/micro")
    def run_micro(req: MicroReq):
        try:
            return micro.run_candidate(
                req.agent,
                req.source,
                note=req.note,
                snapshot_tag=req.snapshot_tag,
                enable_ncu=req.enable_ncu,
                enable_ptxas=req.enable_ptxas,
                mode=req.mode,
                timeout_s=req.timeout_s,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/micro")
    def list_micro(limit: Annotated[int, Query(ge=1, le=200)] = 30, agent: str | None = None):
        return micro.list_results(limit=limit, agent=agent)

    @app.get("/micro/{mid}")
    def get_micro(mid: str):
        try:
            return micro.get_result(mid)
        except KeyError:
            raise HTTPException(404, "no such micro run")

    @app.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        try:
            return mgr.cancel_job(jid)
        except KeyError:
            raise HTTPException(404, "no such job")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/jobs/{jid}/invalidate")
    def invalidate_job(jid: str, req: InvalidateReq):
        try:
            ids = mgr.invalidate_job(
                jid,
                reason=req.reason,
                cascade_sha256=req.cascade,
                cascade_ast=req.cascade_ast,
                cancel_queued=req.cancel_queued,
            )
        except KeyError:
            raise HTTPException(404, "no such job")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"invalidated": ids}

    @app.post("/jobs/{jid}/certify")
    def certify_job(jid: str, req: CertifyReq):
        try:
            return mgr.certify_job(
                jid,
                agent=req.agent,
                patch=req.patch,
                func=req.func,
                scope=req.scope,
                note=req.note,
            )
        except KeyError:
            raise HTTPException(404, "no such job")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/jobs/{jid}/promote")
    def promote_job(jid: str, req: PromoteReq):
        try:
            promoted = mgr.promote_job(jid, target_tier=req.target_tier, agent=req.agent, note=req.note)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return {"id": promoted.id, "tier": promoted.tier, "promoted_from": jid}

    @app.get("/leaderboard")
    def leaderboard(tier: str = "slow", k: int = 10, sort_by: str = "speedup"):
        try:
            return mgr.leaderboard(tier, k, sort_by=sort_by)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/board")
    def post(req: PostReq):
        return board.post(req.agent, req.text, req.kind)

    @app.get("/board")
    def read(since: float = 0.0, limit: int = 200):
        return board.read(since, limit)

    # ------------------------------------------------------ multi-node worker RPCs
    @app.get("/workers/{tier}/requests")
    def worker_requests(tier: str):
        require_remote(tier)
        return [asdict(r) for r in mgr.requests]

    @app.post("/workers/{tier}/heartbeat")
    def worker_heartbeat(tier: str, req: HeartbeatReq):
        require_remote(tier)
        mgr.heartbeat_remote(tier, **req.model_dump())
        return {"ok": True}

    @app.post("/workers/{tier}/claim")
    def worker_claim(tier: str, worker_id: Annotated[str, Query(min_length=1)],
                     wait_s: Annotated[float, Query(ge=0, le=10)] = 2.0):
        require_remote(tier)
        return {"job": mgr.claim_remote(tier, worker_id, wait_s=wait_s)}

    @app.post("/workers/{tier}/complete")
    def worker_complete(tier: str, data: CompleteReq):
        require_remote(tier)
        try:
            mgr.complete_remote(tier, data.model_dump(exclude_unset=True))
        except (KeyError, ValueError):
            raise HTTPException(409, "Unknown job, stale claim, or invalid completion")
        return {"ok": True}

    app.state.mgr, app.state.board, app.state.micro = mgr, board, micro
    return app
