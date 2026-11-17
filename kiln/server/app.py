"""HTTP API used by agents (via the `kiln` CLI or curl)."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kiln.bench.traffic import load_traffic
from kiln.config import RunConfig

from .board import Board
from .jobs import JobManager


class SubmitReq(BaseModel):
    agent: str
    tier: str = "fast"
    source: str
    note: str = ""


class PostReq(BaseModel):
    agent: str
    text: str
    kind: str = "note"


def create_app(run: RunConfig) -> FastAPI:
    requests = load_traffic(run.traffic_file)
    mgr = JobManager(run, requests)
    board = Board(mgr.root / "board.jsonl")
    app = FastAPI(title="kiln")

    @app.on_event("startup")
    def _start():
        mgr.start()

    @app.get("/status")
    def status():
        return {"run": run.name, "tiers": [w.stats() for w in mgr.tiers.values()],
                "jobs": len(mgr.jobs)}

    @app.post("/jobs")
    def submit(req: SubmitReq):
        try:
            job = mgr.submit(req.agent, req.tier, req.source, req.note)
        except KeyError as e:
            raise HTTPException(400, str(e))
        return {"id": job.id, "tier": job.tier}

    @app.get("/jobs/{jid}")
    def get_job(jid: str):
        job = mgr.jobs.get(jid)
        if not job:
            raise HTTPException(404, "no such job")
        return asdict(job)

    @app.get("/leaderboard")
    def leaderboard(tier: str = "slow", k: int = 10):
        return mgr.leaderboard(tier, k)

    @app.post("/board")
    def post(req: PostReq):
        return board.post(req.agent, req.text, req.kind)

    @app.get("/board")
    def read(since: float = 0.0, limit: int = 200):
        return board.read(since, limit)

    app.state.mgr, app.state.board = mgr, board
    return app
