"""Thin Engram API client for the benchmark harness.

Wraps the things the harness needs from a running Engram instance and,
critically, solves the two correctness issues that would otherwise make the
baseline lie:

* Issue C (cross-conversation contamination) -> `create_tenant()` gives each
  conversation its own isolated memory space. Engram's multi-tenancy does the
  actual isolation; we just create one tenant per conversation and send that
  conversation's traffic under its key.

* Issue A (async ingest race) -> `wait_for_drain()`. /ingest returns 202
  before any memory exists. In the PostgreSQL-canonical architecture a message
  is only findable after two background steps:

    1. the canonical write -- its `events` row leaves RECEIVED
       (COMPLETE, GATED_SKIP or FAILED), and
    2. the Neo4j copy -- its PROJECTION rows in `workflow_dispatches` leave
       PENDING/DISPATCHING/STARTED. Semantic retrieval searches Neo4j, so a
       message written to PostgreSQL but not yet copied is still invisible.

  Both are read straight from PostgreSQL, the source of truth. The old
  consolidation-queue signal is NOT used: canonical ingest never fills that
  queue (overviews are built on demand at query time), so it would report
  "drained" before anything was stored -- exactly how a phantom 0% happened
  on the old architecture.

`wait_for_drain()` also returns terminal counts and the most common failure
reasons, so the caller can refuse to score a conversation whose ingest did
not actually succeed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg

# Event statuses that mean "ingest still owes us work for this message".
# GATED_STORE / INDEXED only occur in the legacy pipeline; harmless here.
_PENDING_EVENT_STATUSES = ("RECEIVED", "PROCESSING", "GATED_STORE", "INDEXED")
# Neo4j copy jobs that have not finished yet, and those that gave up.
_PENDING_DISPATCH_STATUSES = ("PENDING", "DISPATCHING", "STARTED")
_FAILED_DISPATCH_STATUSES = ("FAILED", "DEAD")


class EngramError(RuntimeError):
    """Raised when Engram returns an unexpected HTTP status."""


class DrainTimeout(RuntimeError):
    """Raised when ingest does not finish within the allotted time."""


@dataclass
class DrainConfig:
    """Tunables for the drain wait."""

    # The smoke test measured ~60 s per message for the write plus ~20 s for
    # the Neo4j copy, 4 messages in parallel, so a ~340-pair conversation
    # legitimately takes well over an hour.
    max_wait_s: float = 10800.0  # hard ceiling for one conversation's ingest
    poll_interval_s: float = 5.0  # how often to poll
    stall_timeout_s: float = 600.0  # abort only if nothing moves for this long
    # Tolerate a few permanently-wedged messages once this share is finished.
    straggler_ok_ratio: float = 0.97
    straggler_grace_s: float = 180.0
    # Consecutive PostgreSQL read failures tolerated before giving up.
    max_read_errors: int = 5


@dataclass
class DrainResult:
    """Outcome of a drain wait — enough for the caller to judge run health."""

    waited_s: float
    events: dict[str, int]  # events by status for this tenant
    projections: dict[str, int]  # Neo4j copy jobs by status for this tenant
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.events.values())

    @property
    def failed(self) -> int:
        return self.events.get("FAILED", 0)

    @property
    def pending(self) -> int:
        return sum(self.events.get(s, 0) for s in _PENDING_EVENT_STATUSES)

    @property
    def stored(self) -> int:
        """Events that produced memory (COMPLETE); GATED_SKIP stored nothing."""
        return self.events.get("COMPLETE", 0)

    @property
    def projection_pending(self) -> int:
        return sum(self.projections.get(s, 0) for s in _PENDING_DISPATCH_STATUSES)

    @property
    def projection_failed(self) -> int:
        return sum(self.projections.get(s, 0) for s in _FAILED_DISPATCH_STATUSES)

    def healthy(
        self,
        *,
        min_stored_ratio: float = 0.5,
        max_failed_ratio: float = 0.05,
        max_pending_ratio: float = 0.03,
        max_projection_failed_ratio: float = 0.05,
    ) -> bool:
        """True when ingest actually produced findable memory for most messages.

        A run that fails this should not be scored: the questions would be
        answered against a memory that was never built. Small tolerances apply —
        a couple of wedged or failed messages out of hundreds does not
        meaningfully change what is in memory.
        """
        if self.total == 0:
            return False
        if (self.failed / self.total) > max_failed_ratio:
            return False
        if (self.pending / self.total) > max_pending_ratio:
            return False
        projection_total = sum(self.projections.values())
        if projection_total and (self.projection_failed / projection_total) > (
            max_projection_failed_ratio
        ):
            return False
        return (self.stored / self.total) >= min_stored_ratio

    def summary(self) -> str:
        events = ", ".join(f"{k}={v}" for k, v in sorted(self.events.items()))
        copies = ", ".join(f"{k}={v}" for k, v in sorted(self.projections.items()))
        return f"events: {events or '(none)'} | neo4j copies: {copies or '(none)'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "waited_s": round(self.waited_s, 1),
            "events": self.events,
            "projections": self.projections,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class EngramClient:
    """One client, bound to a single tenant's API key.

    Create the tenant + key first with `EngramClient.create_tenant(...)`, which
    returns a client already bound to that tenant.
    """

    base_url: str
    api_key: str
    tenant_id: str = "_default"
    timeout_s: float = 60.0  # ingest / status / health (fast)
    query_timeout_s: float = 300.0  # /query fires several LLM calls; needs headroom
    database_url: str | None = None  # PostgreSQL DSN; ENGRAM_DATABASE_URL when None
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EngramClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- tenant setup (Issue C) -------------------------------------------

    @classmethod
    def create_tenant(
        cls,
        *,
        base_url: str,
        admin_key: str,
        tenant_id: str,
        display_name: str = "",
        timeout_s: float = 60.0,
        query_timeout_s: float = 300.0,
        database_url: str | None = None,
        reuse_if_exists: bool = True,
    ) -> EngramClient:
        """Create an isolated tenant and return a client bound to its key.

        POST /api/v1/admin/tenants returns the fresh tenant's api_key directly,
        so no separate mint call is needed. Requires the admin key.

        If the tenant already exists (409) and `reuse_if_exists` is set, mint a
        fresh key for it instead of failing -- this makes re-runs idempotent.
        """
        admin_http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=timeout_s,
        )
        try:
            resp = admin_http.post(
                "/api/v1/admin/tenants",
                json={"tenant_id": tenant_id, "display_name": display_name},
            )
            if resp.status_code == 409 and reuse_if_exists:
                key_resp = admin_http.post(f"/api/v1/admin/tenants/{tenant_id}/keys")
                if key_resp.status_code != 200:
                    raise EngramError(
                        f"mint-key for existing tenant {tenant_id} failed: "
                        f"{key_resp.status_code} {key_resp.text}"
                    )
                api_key = key_resp.json()["api_key"]
            elif resp.status_code in (200, 201):
                api_key = resp.json()["api_key"]
            else:
                raise EngramError(
                    f"create-tenant {tenant_id} failed: {resp.status_code} {resp.text}"
                )
        finally:
            admin_http.close()

        return cls(
            base_url=base_url,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout_s=timeout_s,
            query_timeout_s=query_timeout_s,
            database_url=database_url,
        )

    # -- ingest ------------------------------------------------------------

    def ingest_pair(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        user_timestamp: str | None = None,
        assistant_timestamp: str | None = None,
        user_turn_idx: int | None = None,
        assistant_turn_idx: int | None = None,
        source: str = "locomo",
    ) -> dict[str, Any]:
        """POST one user/assistant turn pair. Returns the 202 body (event_id...).

        Content is capped at Engram's per-turn limit; longer turns raise 422 at
        the API, which we surface rather than silently truncate.
        """
        turn_pair: dict[str, Any] = {
            "user": _turn(user_content, user_timestamp, user_turn_idx),
            "assistant": _turn(assistant_content, assistant_timestamp, assistant_turn_idx),
        }
        body = {"session_id": session_id, "turn_pair": turn_pair, "source": source}
        resp = self._http.post("/api/v1/ingest", json=body)
        if resp.status_code != 202:
            raise EngramError(f"ingest failed: {resp.status_code} {resp.text}")
        return resp.json()

    def event_status(self, event_id: str) -> dict[str, Any]:
        """GET one ingest event: {event_id, status, retry_count, projection_status}.

        `status` is the canonical write (e.g. COMPLETE, GATED_SKIP, FAILED);
        `projection_status` is the Neo4j copy (PENDING, PROJECTING, INDEXED,
        FAILED). A gated-skip event has no projection, so it stays PENDING.
        """
        resp = self._http.get(f"/api/v1/events/{event_id}")
        if resp.status_code != 200:
            raise EngramError(f"event status failed: {resp.status_code} {resp.text}")
        return resp.json()

    # -- drain (Issue A) ---------------------------------------------------

    def _dsn(self) -> str:
        dsn = self.database_url or os.environ.get("ENGRAM_DATABASE_URL")
        if not dsn:
            raise EngramError("ENGRAM_DATABASE_URL is required to watch ingest progress")
        return dsn

    def ingest_status(self) -> tuple[dict[str, int], dict[str, int]] | None:
        """(events by status, Neo4j copy jobs by status) for this tenant.

        Read straight from PostgreSQL. Returns None if the read fails this
        tick, so a brief connection hiccup does not abort a long wait.
        """
        try:
            with psycopg.connect(self._dsn(), connect_timeout=10) as conn:
                events = conn.execute(
                    "SELECT status, count(*) FROM events WHERE tenant_id = %s GROUP BY status",
                    (self.tenant_id,),
                ).fetchall()
                projections = conn.execute(
                    "SELECT status, count(*) FROM workflow_dispatches "
                    "WHERE tenant_id = %s AND workflow_type = 'PROJECTION' GROUP BY status",
                    (self.tenant_id,),
                ).fetchall()
        except psycopg.Error:
            return None
        return (
            {str(s): int(n) for s, n in events},
            {str(s): int(n) for s, n in projections},
        )

    def failure_reasons(self, limit: int = 5) -> dict[str, int]:
        """Most common error messages of this tenant's FAILED events."""
        try:
            with psycopg.connect(self._dsn(), connect_timeout=10) as conn:
                rows = conn.execute(
                    "SELECT left(coalesce(error_message, '(no message)'), 160), count(*) "
                    "FROM events WHERE tenant_id = %s AND status = 'FAILED' "
                    "GROUP BY 1 ORDER BY 2 DESC LIMIT %s",
                    (self.tenant_id, limit),
                ).fetchall()
        except psycopg.Error:
            return {}
        return {str(reason): int(n) for reason, n in rows}

    def wait_for_drain(
        self,
        cfg: DrainConfig | None = None,
        *,
        on_progress: Callable[[float, dict[str, int], dict[str, int]], None] | None = None,
        progress_every_s: float = 60.0,
    ) -> DrainResult:
        """Block until every message is written AND copied to Neo4j.

        Raises DrainTimeout on the overall `max_wait_s`, when the counts are
        completely frozen for `stall_timeout_s` (e.g. the worker or Temporal
        is down), or when PostgreSQL cannot be read at all. A handful of
        permanently-stuck stragglers is tolerated once `straggler_ok_ratio` of
        messages are finished and nothing has moved for `straggler_grace_s`.
        """
        cfg = cfg or DrainConfig()
        start = time.monotonic()
        last_snapshot: tuple | None = None
        last_change = start
        last_report = start
        read_errors = 0
        events: dict[str, int] = {}
        projections: dict[str, int] = {}

        while True:
            now = time.monotonic()
            elapsed = now - start

            status = self.ingest_status()
            if status is None:
                read_errors += 1
                if read_errors > cfg.max_read_errors:
                    raise DrainTimeout(
                        f"tenant {self.tenant_id}: cannot read ingest progress from "
                        f"PostgreSQL ({read_errors} failed reads). Is the stack up?"
                    )
                time.sleep(cfg.poll_interval_s)
                continue
            read_errors = 0
            events, projections = status

            pending = sum(events.get(s, 0) for s in _PENDING_EVENT_STATUSES)
            pending += sum(projections.get(s, 0) for s in _PENDING_DISPATCH_STATUSES)
            total = sum(events.values()) or 1
            finished_ratio = 1 - sum(events.get(s, 0) for s in _PENDING_EVENT_STATUSES) / total

            # Progress = ANY movement in either breakdown, not just a drop in the
            # pending total: RECEIVED -> COMPLETE also creates new copy jobs, so
            # the total alone can stand still while real work is happening.
            snapshot = (tuple(sorted(events.items())), tuple(sorted(projections.items())))
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                last_change = now

            if on_progress is not None and (now - last_report) >= progress_every_s:
                last_report = now
                on_progress(elapsed, events, projections)

            if pending == 0:
                break

            frozen_for = now - last_change
            if finished_ratio >= cfg.straggler_ok_ratio and frozen_for >= cfg.straggler_grace_s:
                # Nearly everything landed; a stuck message or two is not worth
                # abandoning an otherwise complete ingest.
                break
            if elapsed > cfg.max_wait_s:
                raise DrainTimeout(
                    f"tenant {self.tenant_id}: ingest not finished within "
                    f"{cfg.max_wait_s:.0f}s; events={events} neo4j copies={projections}"
                )
            if frozen_for > cfg.stall_timeout_s:
                raise DrainTimeout(
                    f"tenant {self.tenant_id}: ingest STALLED — nothing changed for "
                    f"{cfg.stall_timeout_s:.0f}s. Are the worker, dispatcher and "
                    f"Temporal running? events={events} neo4j copies={projections}"
                )
            time.sleep(cfg.poll_interval_s)

        return DrainResult(
            waited_s=time.monotonic() - start,
            events=events,
            projections=projections,
            failure_reasons=self.failure_reasons() if events.get("FAILED") else {},
        )

    # -- query -------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        max_depth: str | None = None,
        session_context: str | None = None,
    ) -> dict[str, Any]:
        """POST a question. Returns {answer, retrieval_metadata, answerability}.

        The full `retrieval_metadata` is preserved by the caller -- it is what
        turns a bare score into failure analysis (route, answer mode, stop
        reason, evidence counts, latency).
        """
        body: dict[str, Any] = {"query": question}
        if max_depth is not None:
            body["max_depth"] = max_depth
        if session_context is not None:
            body["session_context"] = session_context
        # A transport error (incl. timeout) is wrapped as EngramError so callers
        # catch it uniformly and one slow query cannot abort a whole run.
        try:
            resp = self._http.post("/api/v1/query", json=body, timeout=self.query_timeout_s)
        except httpx.HTTPError as err:
            raise EngramError(f"query request failed: {err}") from err
        if resp.status_code != 200:
            raise EngramError(f"query failed: {resp.status_code} {resp.text}")
        return resp.json()

    # -- health ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = self._http.get("/api/v1/health")
        if resp.status_code != 200:
            raise EngramError(f"health failed: {resp.status_code} {resp.text}")
        return resp.json()


def _turn(content: str, timestamp: str | None, turn_idx: int | None) -> dict[str, Any]:
    turn: dict[str, Any] = {"content": content}
    if timestamp is not None:
        turn["timestamp"] = timestamp
    if turn_idx is not None:
        turn["turn_idx"] = turn_idx
    return turn


if __name__ == "__main__":
    # Tiny connectivity check. Needs a running Engram + ENGRAM_API_KEY and
    # ENGRAM_DATABASE_URL (loaded from .env.local when run from the repo root).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from envfile import load_env_file

    load_env_file()
    base = os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8001")
    key = os.environ.get("ENGRAM_API_KEY")
    if not key:
        raise SystemExit("set ENGRAM_API_KEY to run the connectivity check")

    with EngramClient(base_url=base, api_key=key) as client:
        print("health:", client.health())
        print("default-tenant ingest status:", client.ingest_status())
