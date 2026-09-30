"""Thin Engram API client for the benchmark harness.

Wraps the four things the harness needs from a running Engram instance and,
critically, solves the two correctness issues that would otherwise make the
baseline lie:

* Issue C (cross-conversation contamination) -> `create_tenant()` gives each
  conversation its own isolated memory space. Engram's multi-tenancy does the
  actual isolation; we just create one tenant per conversation and send that
  conversation's traffic under its key.

* Issue A (async ingest race) -> `wait_for_drain()`. Engram's /ingest returns
  202 immediately and builds the memory in a background worker. Querying before
  that finishes scores ~0% and looks like a retrieval failure when it is really
  a timing bug. We block until the background work is done.

Drain detection uses ONLY the public consolidation-status endpoint
(GET /api/v1/consolidation/status), per the chosen approach. That endpoint
reports the consolidation queue, which fills at the LAST step of ingest, so a
naive "queue == 0" check suffers a "premature zero": right after POSTing, the
worker has not started, the queue is still empty, and we would wrongly conclude
we are done. `wait_for_drain()` guards against this with a rise-then-settle
wait (see its docstring).

Residual limitation of the consolidation-only signal: if some ingest events
never enqueue a consolidation task (e.g. a gated-skip or a failure), they leave
no trace in this queue. The rise-then-settle wait plus a minimum floor covers
the common case; if a full baseline ever looks suspiciously low, tighten this
by also checking the event ledger's RECEIVED/PROCESSING counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class EngramError(RuntimeError):
    """Raised when Engram returns an unexpected HTTP status."""


class DrainTimeoutError(RuntimeError):
    """Raised when ingest does not finish within the allotted time."""


@dataclass
class DrainConfig:
    """Tunables for the rise-then-settle drain wait."""

    max_wait_s: float = 600.0  # hard ceiling for one conversation's ingest
    poll_interval_s: float = 2.0  # how often to poll the status endpoint
    settle_s: float = 8.0  # queue must stay empty this long after activity
    activity_grace_s: float = 45.0  # if no activity is ever seen, give up waiting
    #                                 for a rise after this long and treat as drained


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

    def consolidation_status(self) -> dict[str, Any]:
        resp = self._http.get("/api/v1/consolidation/status")
        if resp.status_code != 200:
            raise EngramError(f"status failed: {resp.status_code} {resp.text}")
        return resp.json()

    def _is_busy(self, status: dict[str, Any]) -> bool:
        """True while the tenant still has consolidation work outstanding."""
        if int(status.get("queue_depth", 0)) > 0:
            return True
        by_status = status.get("by_status") or {}
        return any(int(by_status.get(s, 0)) > 0 for s in ("PENDING", "PROCESSING"))

    def wait_for_drain(self, cfg: DrainConfig | None = None) -> dict[str, float]:
        """Block until background ingest/consolidation for this tenant is done.

        Rise-then-settle to defeat the "premature zero" (see module docstring):

          1. Poll until we observe the queue go BUSY at least once -- proof the
             worker actually picked up the just-ingested events. If we never see
             activity within `activity_grace_s`, assume the batch drained faster
             than our poll interval (or produced no consolidation work) and
             proceed.
          2. Once activity has been seen, wait for the queue to read empty
             continuously for `settle_s` -- a momentary dip to zero between two
             tasks does not count as drained.

        All bounded by `max_wait_s`. Returns timing telemetry for logging.
        """
        cfg = cfg or DrainConfig()
        start = time.monotonic()
        seen_activity = False
        idle_since: float | None = None

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed > cfg.max_wait_s:
                raise DrainTimeoutError(
                    f"tenant {self.tenant_id}: ingest did not drain within "
                    f"{cfg.max_wait_s:.0f}s (seen_activity={seen_activity})"
                )

            busy = self._is_busy(self.consolidation_status())

            if busy:
                seen_activity = True
                idle_since = None
            else:
                if not seen_activity:
                    # Possibly a premature zero: worker hasn't started. Only give
                    # up waiting for a rise after the grace period.
                    if elapsed >= cfg.activity_grace_s:
                        return {
                            "waited_s": elapsed,
                            "saw_activity": 0.0,
                        }
                else:
                    # Activity happened and the queue is now empty -> settle.
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= cfg.settle_s:
                        return {
                            "waited_s": time.monotonic() - start,
                            "saw_activity": 1.0,
                        }

            time.sleep(cfg.poll_interval_s)

    # -- query -------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        max_depth: str | None = None,
        max_reentries: int | None = None,
        session_context: str | None = None,
    ) -> dict[str, Any]:
        """POST a question. Returns {answer, retrieval_metadata, ...}.

        The full `retrieval_metadata` is preserved by the caller -- it is what
        turns a bare score into failure analysis (l0_decision, cascade depth,
        nodes retrieved, latency).
        """
        body: dict[str, Any] = {"query": question}
        if max_depth is not None:
            body["max_depth"] = max_depth
        if max_reentries is not None:
            body["max_reentries"] = max_reentries
        if session_context is not None:
            body["session_context"] = session_context
        # Longer per-request timeout than the client default: a query drives the
        # full cascade + several LLM calls. A transport error (incl. timeout) is
        # wrapped as EngramError so callers catch it uniformly and one slow query
        # cannot abort a whole benchmark run.
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
    # Tiny connectivity smoke test. Needs a running Engram + env vars:
    #   ENGRAM_BASE_URL (default http://127.0.0.1:8000)
    #   ENGRAM_API_KEY  (default local tenant key)
    import os

    base = os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8000")
    key = os.environ.get("ENGRAM_API_KEY")
    if not key:
        raise SystemExit("set ENGRAM_API_KEY to run the smoke test")

    with EngramClient(base_url=base, api_key=key) as client:
        print("health:", client.health())
        print("consolidation status:", client.consolidation_status())
