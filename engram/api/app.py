"""FastAPI application (§11).

Responsibilities:
  - REST surface per §11.1.
  - Middleware stack: body-size limits, rate limiting, request-ID tagging.
  - Lifespan: boot the durable ingest worker, consolidation worker, and
    reconciliation worker. Graceful shutdown joins all three.
  - Liveness (`/livez`) vs readiness (`/readyz`) split so orchestrators
    can distinguish "process is up" from "dependencies are reachable".
  - Streaming `/api/v1/query` when `stream=true` in the request body.

Boot order:
  1. Logging is configured as the very first thing.
  2. App state is built lazily on first request or explicitly in lifespan.
  3. Middleware stack is installed (outermost → innermost): request-id,
     body-size, rate limit.
  4. Lifespan starts workers.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from engram import metrics as metrics_mod
from engram.admin import routes as admin_ui_routes
from engram.api import schemas
from engram.api.auth import AuthDep
from engram.api.body_limit import BodySizeLimitMiddleware
from engram.api.rate_limit import RateLimitMiddleware
from engram.api.request_id import RequestIdMiddleware
from engram.api.routes import (
    admin as admin_route,
)
from engram.api.routes import (
    bulk_ingest as bulk_ingest_route,
)
from engram.api.routes import (
    chat as chat_route,
)
from engram.api.routes import (
    consolidation as consolidation_route,
)
from engram.api.routes import (
    events as events_route,
)
from engram.api.routes import (
    kg as kg_route,
)
from engram.api.routes import (
    memories as memories_route,
)
from engram.api.routes import (
    sessions as sessions_route,
)
from engram.config import get_config
from engram.consolidation.reconciliation import ReconciliationContext
from engram.consolidation.reconciliation import start_background as start_reconciliation
from engram.consolidation.worker import ConsolidationContext
from engram.consolidation.worker import start_background as start_consolidation
from engram.deps import (
    AppState,
    get_state,
    make_orchestrator_context,
    reset_state,
)
from engram.ingest.durable_worker import DurableIngestWorker
from engram.ingest.durable_worker import start_background as start_durable_ingest
from engram.logging_setup import configure_logging
from engram.resilience import breaker_snapshot
from engram.retrieval.orchestrator import run_query
from engram.tracing import configure_tracing
from engram.uri import pair_id as pair_id_fn

log = logging.getLogger(__name__)

configure_logging()

# Handles to joinable background threads + stop events.
_cons_handle: tuple[threading.Thread, threading.Event] | None = None
_recon_handle: tuple[threading.Thread, threading.Event] | None = None
_ingest_worker: DurableIngestWorker | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start workers on boot, signal them to stop on shutdown."""
    global _cons_handle, _recon_handle, _ingest_worker
    try:
        state = get_state()
    except Exception:
        log.exception("failed to build app state during lifespan startup")
        yield
        return

    redis_url = (
        state.cfg.session_cache.redis_url if state.cfg.session_cache.backend == "redis" else None
    )

    # Production Temporal deployments keep all durable work out of API
    # processes. The PostgreSQL polling workers remain available when Temporal
    # is intentionally disabled.
    legacy_workers = not state.cfg.temporal.enabled
    legacy_fs = state.fs
    if legacy_workers and legacy_fs is None:
        raise RuntimeError("legacy background workers require the filesystem store")

    if legacy_workers and state.cfg.consolidation.poll_interval_seconds > 0:
        cons_ctx = ConsolidationContext(
            cfg=state.cfg,
            control_plane=state.control_plane,
            fs=legacy_fs,
            neo4j=state.neo4j,
            core=state.core,
            embed=state.embed,
            overview_cache=state.overview_cache,
            memory_repository=state.memory_repository,
        )
        _cons_handle = start_consolidation(cons_ctx, redis_url=redis_url)
        log.info("consolidation worker started (leased=%s)", bool(redis_url))

    if legacy_workers and state.cfg.event_ledger.reconciliation_interval_seconds > 0:
        rec_ctx = ReconciliationContext(
            cfg=state.cfg,
            control_plane=state.control_plane,
            neo4j=state.neo4j,
        )
        _recon_handle = start_reconciliation(rec_ctx, redis_url=redis_url)
        log.info("reconciliation worker started (leased=%s)", bool(redis_url))

    if legacy_workers:
        assert legacy_fs is not None
        _ingest_worker = start_durable_ingest(
            cfg=state.cfg,
            control_plane=state.control_plane,
            fs=legacy_fs,
            neo4j=state.neo4j,
            core=state.core,
            embed=state.embed,
            max_concurrent=state.cfg.consolidation.max_concurrent_tasks,
            poll_interval_seconds=1.0,
        )
        log.info("legacy durable ingest worker started")
    else:
        log.info("Temporal orchestration enabled; API background workers disabled")

    try:
        yield
    finally:
        if _ingest_worker is not None:
            _ingest_worker.stop(timeout_s=10.0)
        for handle in (_cons_handle, _recon_handle):
            if handle is not None:
                thread, stop = handle
                stop.set()
                thread.join(timeout=5.0)
        reset_state()
        log.info("engram shutdown complete")


app = FastAPI(title="Engram", version="0.1.0", lifespan=_lifespan)
configure_tracing(app)


def _install_middleware() -> None:
    """Install middleware in inner-to-outer order (Starlette runs LIFO)."""
    try:
        cfg = get_config()
    except Exception:
        # Tests that mount the app without real config still get request IDs + body limits.
        app.add_middleware(BodySizeLimitMiddleware)
        app.add_middleware(RequestIdMiddleware)
        return

    app.add_middleware(
        RateLimitMiddleware,
        query_per_min=cfg.api.rate_limit_query_per_minute,
        ingest_per_min=cfg.api.rate_limit_ingest_per_minute,
        redis_url=(cfg.session_cache.redis_url if cfg.session_cache.backend == "redis" else None),
    )
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(RequestIdMiddleware)


_install_middleware()

app.include_router(sessions_route.router)
app.include_router(memories_route.router)
app.include_router(events_route.router)
app.include_router(bulk_ingest_route.router)
app.include_router(kg_route.router)
app.include_router(consolidation_route.router)
app.include_router(admin_route.router)
app.include_router(chat_route.router)
app.include_router(admin_ui_routes.admin_router)


# ----------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> PlainTextResponse:
    try:
        state = get_state()
        metrics_mod.consolidation_queue_depth.set(state.control_plane.queue_depth())
        kg_counts = _kg_counts_safe(state)
        if kg_counts:
            metrics_mod.kg_node_count.set(kg_counts["nodes"])
            metrics_mod.kg_edge_count.set(kg_counts["edges"])
    except Exception:
        log.debug("metrics pre-render update failed", exc_info=True)
    body, content_type = metrics_mod.render_latest()
    return PlainTextResponse(content=body, media_type=content_type)


def _kg_counts_safe(state: AppState) -> dict[str, int] | None:
    try:
        rows = state.neo4j.run_template(
            "MATCH (n:Node) WITH count(n) AS nodes "
            "OPTIONAL MATCH ()-[r]->() "
            "RETURN nodes, count(r) AS edges",
            {},
            timeout_s=3,
        )
    except Exception:
        return None
    if not rows:
        return None
    return {"nodes": int(rows[0].get("nodes", 0)), "edges": int(rows[0].get("edges", 0))}


# ----------------------------------------------------------------------
# Health — liveness vs readiness split
# ----------------------------------------------------------------------


@app.get("/livez", include_in_schema=False)
def livez() -> PlainTextResponse:
    """Liveness probe — returns 200 as long as the process is alive.

    Kubernetes liveness probes should hit this. Failure means the pod
    is wedged and should be restarted.
    """
    return PlainTextResponse("ok", status_code=200)


@app.get("/readyz", include_in_schema=False)
def readyz() -> JSONResponse:
    """Readiness probe for dependencies required to serve canonical memory.

    PostgreSQL is authoritative in V2. Neo4j and Redis are reported as
    capabilities, but their failure must not make exact canonical reads
    unavailable or prevent durable ingestion from being accepted.
    """
    try:
        state = get_state()
    except Exception as err:
        return JSONResponse({"status": "not_ready", "reason": str(err)[:500]}, status_code=503)
    components = _component_health(state)
    required = (
        ("postgres", "configuration") if state.cfg.canonical_memory.enabled else tuple(components)
    )
    ready = all(components.get(name, False) for name in required)
    return JSONResponse(
        {
            "status": "ready" if ready else "not_ready",
            "components": components,
            "required_components": list(required),
            "degraded_capabilities": [
                name
                for name, available in components.items()
                if not available and name not in required
            ],
            "breakers": breaker_snapshot(),
        },
        status_code=200 if ready else 503,
    )


@app.get("/api/v1/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    """Operator-facing health — never raises; always 200 with component status."""
    components: dict[str, bool] = {}
    try:
        state = get_state()
    except Exception:
        log.exception("failed to build app state")
        return schemas.HealthResponse(status="unavailable", components={"startup": False})
    components.update(_component_health(state))
    overall = "healthy" if all(components.values()) else "degraded"
    return schemas.HealthResponse(status=overall, components=components)


def _component_health(state: AppState) -> dict[str, bool]:
    components = {
        "postgres": _control_plane_ready(state),
        "configuration": True,
        "neo4j_projection": _safe_ping(state.neo4j),
        "redis_cache": _safe_ping(state.session_cache),
    }
    if not state.cfg.canonical_memory.enabled:
        components["filesystem"] = bool(state.fs and state.fs.data_dir.exists())
    return components


def _safe_ping(component: object) -> bool:
    try:
        return bool(component.ping())  # type: ignore[attr-defined]
    except Exception:
        return False


def _control_plane_ready(state: AppState) -> bool:
    """Check the PostgreSQL control plane with a cheap query."""
    try:
        return state.control_plane.get_conn().execute("SELECT 1").fetchone() is not None
    except Exception:
        return False


@app.get("/api/v1/config", response_model=schemas.ConfigResponse, dependencies=[AuthDep])
def get_config_endpoint() -> schemas.ConfigResponse:
    state = get_state()
    cfg = state.cfg
    return schemas.ConfigResponse(
        retrieval=cfg.retrieval.model_dump(),
        core_model={
            "provider": cfg.core_model.provider,
            "model_path": cfg.core_model.model_path,
        },
        frontier_llm={
            "provider": cfg.frontier_llm.provider,
            "model_path": cfg.frontier_llm.model_path,
        },
    )


# ----------------------------------------------------------------------
# Ingest (§11.3)
# ----------------------------------------------------------------------


@app.post("/api/v1/ingest", response_model=schemas.IngestResponse, dependencies=[AuthDep])
def ingest(req: schemas.IngestRequest) -> JSONResponse:
    """Durable ingest: synchronously commit to the configured control plane.

    Production creates the PostgreSQL event and Temporal dispatch outbox in
    one transaction. Non-Temporal deployments use the PostgreSQL durable poller.
    """
    from engram.tenancy import current_tenant_id

    state = get_state()
    tid = current_tenant_id()
    _check_backpressure(state, tenant_id=tid)
    pair = req.effective_pair()
    user_idx = pair.user.turn_idx or 0
    asst_idx = pair.assistant.turn_idx or (user_idx + 1)
    pid = pair_id_fn(req.session_id or "stateless", user_idx, asst_idx)
    event_id, _ = state.control_plane.record_event(
        pair_id=pid,
        session_id=req.session_id,
        source=req.source,
        event_type="INGEST",
        payload=req.model_dump(),
        tenant_id=tid,
    )
    response = schemas.IngestResponse(
        event_id=event_id,
        pair_id=pid,
        status="RECEIVED",
        projection_status="PENDING",
    )
    return JSONResponse(content=response.model_dump(), status_code=status.HTTP_202_ACCEPTED)


@app.post(
    "/api/v1/ingest/batch",
    response_model=list[schemas.IngestResponse],
    dependencies=[AuthDep],
)
def ingest_batch(batch: list[schemas.IngestRequest]) -> JSONResponse:
    from engram.tenancy import current_tenant_id

    if len(batch) > 100:
        raise HTTPException(status_code=400, detail="max batch size is 100")
    state = get_state()
    tid = current_tenant_id()
    _check_backpressure(state, tenant_id=tid)
    results: list[dict[str, Any]] = []
    for req in batch:
        pair = req.effective_pair()
        user_idx = pair.user.turn_idx or 0
        asst_idx = pair.assistant.turn_idx or (user_idx + 1)
        pid = pair_id_fn(req.session_id or "stateless", user_idx, asst_idx)
        event_id, _ = state.control_plane.record_event(
            pair_id=pid,
            session_id=req.session_id,
            source=req.source,
            event_type="INGEST",
            payload=req.model_dump(),
            tenant_id=tid,
        )
        results.append(
            {
                "event_id": event_id,
                "pair_id": pid,
                "status": "RECEIVED",
                "projection_status": "PENDING",
            }
        )
    return JSONResponse(content=results, status_code=status.HTTP_202_ACCEPTED)


def _check_backpressure(state: AppState, *, tenant_id: str | None = None) -> None:
    max_backlog = state.cfg.consolidation.max_backlog
    depth = state.control_plane.queue_depth(tenant_id=tenant_id)
    if depth > max_backlog:
        raise HTTPException(
            status_code=503,
            detail=f"consolidation queue saturated ({depth} > {max_backlog})",
            headers={"Retry-After": "30"},
        )


# ----------------------------------------------------------------------
# Query (§11.2) — supports streaming via SSE when stream=true
# ----------------------------------------------------------------------


@app.post("/api/v1/query", dependencies=[AuthDep])
def query(req: schemas.QueryRequest):
    state = get_state()
    ctx = make_orchestrator_context(state)
    started = time.perf_counter()
    try:
        result = run_query(
            ctx,
            session_id=req.session_id,
            query=req.query,
            session_context=req.session_context,
            max_depth=req.max_depth,
            max_reentries=req.max_reentries,
            answer_style=req.answer_style,
            answer_instructions=req.answer_instructions,
        )
    except Exception as err:
        log.exception("query failed")
        raise HTTPException(status_code=500, detail=f"query failed: {err}") from err

    md = result.retrieval_metadata
    metrics_mod.query_latency.labels(phase="total").observe(time.perf_counter() - started)
    for phase, ms in md.latency_ms.items():
        metrics_mod.query_latency.labels(phase=phase).observe(ms / 1000.0)
    metrics_mod.query_depth_predicted_vs_reached.labels(
        predicted=md.predicted_depth or "unknown",
        reached=md.cascade_depth_reached,
    ).inc()
    metrics_mod.reentries_per_query.observe(md.reentries)
    if md.l0_decision:
        metrics_mod.l0_gate_decisions.labels(
            decision=md.l0_decision,
            reason=(md.l0_reason or "").split(":", 1)[0],
        ).inc()
    metrics_mod.answerability.labels(state=result.answerability).inc()
    for rejection in md.missing_evidence:
        reason = rejection.split(":", 1)[0][:80] or "unknown"
        metrics_mod.canonical_verification_rejections.labels(reason=reason).inc()

    if not req.stream:
        return schemas.QueryResponse(
            answer=result.answer,
            session_id=req.session_id,
            retrieval_metadata=md.to_dict(),
            answerability=result.answerability,
            sources=[item.model_dump(mode="json") for item in result.verified_evidence],
            conflicts=list(md.conflicts),
            verification_rejections=list(md.missing_evidence),
            route_metadata={
                "selected": md.retrieval_route,
                "attempted": list(md.routes_attempted),
            },
            projection_metadata={"canonical_revision": md.canonical_revision},
        )

    # Streaming: emit the buffered answer as a single SSE event and close.
    # For providers that support native streaming we could open a second
    # pass via ctx.frontier.stream_answer here; the cascade re-entry logic
    # already ran, so the answer is stable.
    return StreamingResponse(
        _sse_query_stream(ctx, result, req),
        media_type="text/event-stream",
    )


def _sse_query_stream(ctx, result, req):
    """SSE generator for /api/v1/query with stream=true.

    Yields:
      event: metadata           — retrieval metadata as JSON
      event: delta              — one per streamed answer chunk
      event: done               — final marker
    """
    import json as _json

    md = result.retrieval_metadata.to_dict()
    yield f"event: metadata\ndata: {_json.dumps(md)}\n\n"

    # Stream the buffered answer so the response matches the verdict already produced.
    try:
        answer = result.answer or ""
        chunk = 200
        for i in range(0, len(answer), chunk):
            yield f"event: delta\ndata: {_json.dumps({'text': answer[i : i + chunk]})}\n\n"
    except Exception as err:
        log.exception("streaming failed")
        yield f"event: error\ndata: {_json.dumps({'error': str(err)})}\n\n"
    yield "event: done\ndata: {}\n\n"
