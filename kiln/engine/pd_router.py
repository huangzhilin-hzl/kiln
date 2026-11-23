"""Built-in async HTTP Prefill-Decode (`xPyD`) router fallback for SGLang PD.

When `sglang_router` is installed without its native Rust extension (`sglang_router.router`
is missing, leaving only `MiniLB`), upstream `MiniLB` has two critical limitations for
`kiln` `xPyD` evaluation:
1. `MiniLB.select_pair()` cycles `prefill_urls` sequentially while indexing
   `prefill_bootstrap_ports` randomly, mismatching `(prefill_url, bootstrap_port)` on
   multi-prefill (`x >= 2`) setups with distinct bootstrap ports.
2. `MiniLB.generate()` concatenates `meta_info["input_token_logprobs"]` unconditionally
   when `return_logprob=True`, raising `TypeError` when `input_token_logprobs` is `None`.

This lightweight FastAPI + `httpx.AsyncClient` router accepts the same CLI flags as
`sglang_router.launch_router --pd-disaggregation` and serves as an automatic fallback
whenever the native Rust router is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import itertools
import random
import sys
import threading
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _maybe_wrap_ipv6(address: str | None) -> str:
    if not address:
        return "127.0.0.1"
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def _generate_bootstrap_room() -> int:
    return random.randint(0, 2**63 - 1)


def _get_request_batch_size(request_data: dict) -> int | None:
    text = request_data.get("text")
    if text is not None:
        return None if isinstance(text, str) else len(text)
    input_ids = request_data.get("input_ids")
    if input_ids is not None:
        if not input_ids or isinstance(input_ids[0], int):
            return None
        return len(input_ids)
    return None


def _parse_prefill_entries(raw: list[list[str]] | None) -> list[tuple[str, int | None]]:
    if not raw:
        return []
    out: list[tuple[str, int | None]] = []
    for item in raw:
        if not item:
            continue
        url = item[0].rstrip("/")
        bport: int | None = None
        if len(item) >= 2 and item[1].lower() != "none":
            bport = int(item[1])
        out.append((url, bport))
    return out


def _parse_decode_entries(raw: list[list[str]] | None) -> list[str]:
    if not raw:
        return []
    return [item[0].rstrip("/") for item in raw if item]


@dataclass
class PDRouterState:
    prefill_nodes: list[tuple[str, int | None]]
    decode_urls: list[str]
    policy: str = "round_robin"
    timeout_s: float = 1800.0
    client: httpx.AsyncClient | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _p_counter: itertools.count = field(default_factory=itertools.count)
    _d_counter: itertools.count = field(default_factory=itertools.count)
    prefill_counts: dict[str, int] = field(default_factory=dict)
    decode_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for u, _ in self.prefill_nodes:
            self.prefill_counts.setdefault(u, 0)
        for u in self.decode_urls:
            self.decode_counts.setdefault(u, 0)

    def select_pair(self) -> tuple[str, int | None, str]:
        if not self.prefill_nodes or not self.decode_urls:
            raise RuntimeError("PDRouter requires at least one prefill and one decode backend")
        with self._lock:
            if self.policy == "random":
                p_url, bport = random.choice(self.prefill_nodes)
                d_url = random.choice(self.decode_urls)
            else:
                p_idx = next(self._p_counter) % len(self.prefill_nodes)
                d_idx = next(self._d_counter) % len(self.decode_urls)
                p_url, bport = self.prefill_nodes[p_idx]
                d_url = self.decode_urls[d_idx]
            self.prefill_counts[p_url] = self.prefill_counts.get(p_url, 0) + 1
            self.decode_counts[d_url] = self.decode_counts.get(d_url, 0) + 1
            return p_url, bport, d_url

    def inject_bootstrap(
        self, request_data: dict, prefill_url: str, bootstrap_port: int | None, *, batch_aware: bool
    ) -> dict:
        parsed = urllib.parse.urlparse(prefill_url)
        hostname = _maybe_wrap_ipv6(parsed.hostname)
        modified = dict(request_data)
        batch_size = _get_request_batch_size(modified) if batch_aware else None
        if batch_size is not None:
            modified["bootstrap_host"] = [hostname] * batch_size
            modified["bootstrap_port"] = [bootstrap_port] * batch_size
            modified["bootstrap_room"] = [_generate_bootstrap_room() for _ in range(batch_size)]
        else:
            modified["bootstrap_host"] = hostname
            modified["bootstrap_port"] = bootstrap_port
            modified["bootstrap_room"] = _generate_bootstrap_room()
        return modified


class _DecodeStreamResponse(StreamingResponse):
    def __init__(self, upstream: httpx.Response):
        self.upstream = upstream
        super().__init__(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.upstream.aclose()


async def _stream_pair(client: httpx.AsyncClient, prefill_url: str, decode_url: str, body: dict):
    async def start_backend(url: str, payload: dict, stream: bool) -> httpx.Response:
        request = client.build_request("POST", url, json=payload)
        response = await client.send(request, stream=stream)
        if response.is_success:
            return response
        try:
            await response.aread()
            response.raise_for_status()
        finally:
            await response.aclose()

    # Ask prefill for a completed HTTP response so generation failures are visible
    # before committing the decode stream. Both halves still start concurrently.
    prefill_body = {**body, "stream": False}
    prefill_body.pop("stream_options", None)
    prefill_task = asyncio.create_task(start_backend(prefill_url, prefill_body, False))
    decode_task = asyncio.create_task(start_backend(decode_url, body, True))
    streaming = False
    try:
        _, decode = await asyncio.gather(prefill_task, decode_task)
        response = _DecodeStreamResponse(decode)
        streaming = True
        return response
    except httpx.HTTPStatusError as exc:
        upstream = exc.response
        headers = {key: upstream.headers[key] for key in ("content-type", "retry-after") if key in upstream.headers}
        return Response(upstream.content, status_code=upstream.status_code, headers=headers)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="PD backend timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="PD backend request failed") from exc
    finally:
        if not streaming:
            prefill_task.cancel()
            decode_task.cancel()
            _, decode = await asyncio.gather(prefill_task, decode_task, return_exceptions=True)
            if isinstance(decode, httpx.Response):
                await decode.aclose()


def create_pd_router_app(
    prefill_nodes: list[tuple[str, int | None]],
    decode_urls: list[str],
    *,
    policy: str = "round_robin",
    timeout_s: float = 1800.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    state = PDRouterState(
        prefill_nodes=list(prefill_nodes),
        decode_urls=list(decode_urls),
        policy=policy,
        timeout_s=timeout_s,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.client = httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
            transport=transport,
        )
        try:
            yield
        finally:
            await state.client.aclose()
            state.client = None

    app = FastAPI(lifespan=lifespan)
    app.state.pd = state

    def _client() -> httpx.AsyncClient:
        if state.client is None:
            state.client = httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(timeout_s),
                limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
                transport=transport,
            )
        return state.client

    async def _dispatch(request_data: dict, endpoint: str, *, batch_aware: bool):
        p_url, bport, d_url = state.select_pair()
        modified = state.inject_bootstrap(request_data, p_url, bport, batch_aware=batch_aware)
        client = _client()

        if modified.get("stream", False):
            return await _stream_pair(client, f"{p_url}/{endpoint}", f"{d_url}/{endpoint}", modified)

        p_resp, d_resp = await asyncio.gather(
            client.post(f"{p_url}/{endpoint}", json=modified),
            client.post(f"{d_url}/{endpoint}", json=modified),
        )
        if d_resp.status_code >= 400:
            return Response(
                content=d_resp.content,
                status_code=d_resp.status_code,
                media_type=d_resp.headers.get("content-type", "application/json"),
            )
        if p_resp.status_code >= 400:
            return Response(
                content=p_resp.content,
                status_code=p_resp.status_code,
                media_type=p_resp.headers.get("content-type", "application/json"),
            )

        ret_json = d_resp.json()
        if modified.get("return_logprob"):
            try:
                p_json = p_resp.json()
            except ValueError:
                p_json = None
            p_meta = p_json.get("meta_info") if isinstance(p_json, dict) else None
            d_meta = ret_json.get("meta_info") if isinstance(ret_json, dict) else None
            if isinstance(p_meta, dict) and isinstance(d_meta, dict):
                p_in = p_meta.get("input_token_logprobs")
                d_in = d_meta.get("input_token_logprobs")
                if isinstance(p_in, list) and isinstance(d_in, list):
                    d_meta["input_token_logprobs"] = p_in + d_in
        return JSONResponse(content=ret_json, status_code=d_resp.status_code)

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.get("/router_stats")
    async def router_stats() -> dict:
        return {
            "policy": state.policy,
            "prefill_counts": dict(state.prefill_counts),
            "decode_counts": dict(state.decode_counts),
        }

    @app.get("/health_generate")
    async def health_generate() -> Response:
        client = _client()
        urls = [u for u, _ in state.prefill_nodes] + list(state.decode_urls)
        resps = await asyncio.gather(*(client.get(f"{u}/health_generate") for u in urls))
        for r in resps:
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
        return Response(status_code=200)

    @app.post("/flush_cache")
    async def flush_cache() -> Response:
        client = _client()
        urls = [u for u, _ in state.prefill_nodes] + list(state.decode_urls)
        await asyncio.gather(*(client.post(f"{u}/flush_cache") for u in urls))
        return Response(status_code=200)

    @app.get("/get_model_info")
    async def get_model_info():
        r = await _client().get(f"{state.prefill_nodes[0][0]}/get_model_info")
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    @app.get("/get_server_info")
    async def get_server_info():
        client = _client()
        p_resps = await asyncio.gather(*(client.get(f"{u}/get_server_info") for u, _ in state.prefill_nodes))
        d_resps = await asyncio.gather(*(client.get(f"{u}/get_server_info") for u in state.decode_urls))
        p_infos = [r.json() for r in p_resps]
        d_infos = [r.json() for r in d_resps]
        internal_states = []
        for info in d_infos:
            if isinstance(info, dict) and "internal_states" in info:
                internal_states.extend(info["internal_states"])
        return {
            "internal_states": internal_states or [{"last_gen_throughput": 0.0, "avg_spec_accept_length": None}],
            "prefill": p_infos,
            "decode": d_infos,
        }

    @app.get("/v1/models")
    async def v1_models():
        r = await _client().get(f"{state.prefill_nodes[0][0]}/v1/models")
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    @app.post("/generate")
    async def generate(req: Request):
        return await _dispatch(await req.json(), "generate", batch_aware=True)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        return await _dispatch(await req.json(), "v1/chat/completions", batch_aware=False)

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await _dispatch(await req.json(), "v1/completions", batch_aware=False)

    return app


def parse_pd_router_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiln fallback HTTP PD router")
    p.add_argument("--pd-disaggregation", action="store_true", default=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=30080)
    p.add_argument("--prefill", nargs="+", action="append", default=[])
    p.add_argument("--decode", nargs="+", action="append", default=[])
    p.add_argument("--policy", type=str, default="round_robin")
    p.add_argument("--request-timeout-secs", type=float, default=1800.0)
    p.add_argument("--log-level", type=str, default="warning")
    ns, _unknown = p.parse_known_args(argv)
    return ns


def main(argv: list[str] | None = None) -> None:
    ns = parse_pd_router_cli(sys.argv[1:] if argv is None else argv)
    prefill_nodes = _parse_prefill_entries(ns.prefill)
    decode_urls = _parse_decode_entries(ns.decode)
    if not prefill_nodes or not decode_urls:
        raise SystemExit("kiln.engine.pd_router requires at least one --prefill and one --decode")
    app = create_pd_router_app(
        prefill_nodes,
        decode_urls,
        policy=ns.policy,
        timeout_s=ns.request_timeout_secs,
    )
    uvicorn.run(app, host=ns.host, port=ns.port, log_level=ns.log_level)


if __name__ == "__main__":
    main()
