"""Accuracy-first adaptive retrieval orchestration.

Retrieval levels are capabilities selected by the L0 planner; they are not a
mandatory waterfall.  PostgreSQL (through ``MemoryRepository``) is the only
source allowed into factual model context.  Neo4j/vector results are
discovery candidates and must be hydrated by the repository before they are
verified and rendered.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engram import tokens as tok_mod
from engram import tracing
from engram.config import EngramConfig
from engram.models.core import CoreModelError, CoreModelProvider
from engram.models.embeddings import EmbeddingService
from engram.models.frontier import FrontierLLMProvider, FrontierVerdict
from engram.models.request_context import model_request_session
from engram.predicate_registry import normalize_predicate
from engram.resilience import CircuitOpenError
from engram.retrieval.context import ContextBuilder
from engram.retrieval.domain import (
    AnswerabilityState,
    DiscoveryCandidate,
    EvidenceAssessment,
    MemoryRepository,
    RetrievalCandidate,
    RetrievalIntent,
    RetrievalPlan,
    RetrievalRoute,
    TemporalKind,
    VerifiedEvidence,
)
from engram.retrieval.evidence import CanonicalVerifier, EvidenceGate
from engram.retrieval.planner import AdaptivePlanner
from engram.tenancy import current_tenant_id

log = logging.getLogger(__name__)


_DEPTH_ORDER = ["SESSION", "L0", "L1", "L2", "L3", "L4"]


class CanonicalRepositoryUnavailableError(RuntimeError):
    """Canonical verification failed because PostgreSQL could not be read."""


class AmbiguousEntityReferenceError(ValueError):
    """An exact alias maps to multiple active canonical identities."""

    def __init__(self, alias: str, candidate_ids: Sequence[str]) -> None:
        self.alias = alias
        self.candidate_ids = tuple(candidate_ids)
        super().__init__(f"entity alias {alias!r} is ambiguous")


def _depth_rank(depth: str) -> int:
    try:
        return _DEPTH_ORDER.index(str(depth).upper())
    except ValueError:
        return len(_DEPTH_ORDER) - 1


def _notify_step(
    on_step: Callable[[dict[str, Any]], None] | None,
    md: RetrievalMetadata,
    step: str,
) -> None:
    if on_step is not None:
        on_step({"step": step, **md.to_dict()})


@dataclass
class RetrievalMetadata:
    """Observable retrieval state, retaining fields used by existing clients."""

    cascade_depth_reached: str = "L0"
    levels_visited: list[str] = field(default_factory=list)
    predicted_depth: str | None = None
    nodes_retrieved: int = 0
    total_context_tokens: int = 0
    reentries: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    l0_decision: str | None = None
    l0_reason: str | None = None
    answerability_state: str = AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    retrieval_route: str | None = None
    routes_attempted: list[str] = field(default_factory=list)
    candidates_discovered: int = 0
    verified_evidence: int = 0
    missing_evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    canonical_revision: int | None = None
    temporal_scope: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    answer_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cascade_depth_reached": self.cascade_depth_reached,
            "levels_visited": list(self.levels_visited),
            "predicted_depth": self.predicted_depth,
            "nodes_retrieved": self.nodes_retrieved,
            "total_context_tokens": self.total_context_tokens,
            "reentries": self.reentries,
            "latency_ms": dict(self.latency_ms),
            "l0_decision": self.l0_decision,
            "l0_reason": self.l0_reason,
            "answerability_state": self.answerability_state,
            "retrieval_route": self.retrieval_route,
            "routes_attempted": list(self.routes_attempted),
            "candidates_discovered": self.candidates_discovered,
            "verified_evidence": self.verified_evidence,
            "missing_evidence": list(self.missing_evidence),
            "conflicts": list(self.conflicts),
            "canonical_revision": self.canonical_revision,
            "temporal_scope": dict(self.temporal_scope),
            "stop_reason": self.stop_reason,
            "answer_mode": self.answer_mode,
        }


@dataclass
class QueryResult:
    answer: str
    retrieval_metadata: RetrievalMetadata
    answerability_state: str = AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    verified_evidence: list[VerifiedEvidence] = field(default_factory=list)

    @property
    def answerability(self) -> str:
        return self.answerability_state


@dataclass
class OrchestratorContext:
    """Runtime dependencies for retrieval.

    ``fs`` remains an optional compatibility constructor field for callers
    still assembling the broader application state.  Adaptive retrieval never
    reads it.  ``memory_repository`` is canonical; ``memory`` is accepted as a
    short injection alias while the repository package is being migrated.
    """

    cfg: EngramConfig
    fs: Any | None = None
    neo4j: Any = None
    core: CoreModelProvider | None = None
    frontier: FrontierLLMProvider | None = None
    embed: EmbeddingService | None = None
    l0_classifier: Any = None
    memory_repository: MemoryRepository | Any | None = None
    memory: MemoryRepository | Any | None = None
    planner: AdaptivePlanner | None = None
    verifier: CanonicalVerifier | None = None
    evidence_gate: EvidenceGate | None = None
    context_builder: ContextBuilder | None = None

    def repository(self) -> MemoryRepository | Any | None:
        return self.memory_repository or self.memory


def run_query(
    ctx: OrchestratorContext,
    *,
    session_id: str | None,
    query: str,
    session_context: str | None = None,
    max_depth: str | None = None,
    max_reentries: int | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> QueryResult:
    """Plan, retrieve, verify, and answer one query."""

    del max_reentries  # Adaptive evidence routing owns escalation and stopping.
    scoped_session_id = session_id or f"query-{uuid.uuid4().hex}"
    with model_request_session(scoped_session_id):
        return _run_query(
            ctx,
            session_id=session_id,
            query=query,
            session_context=session_context,
            max_depth=max_depth,
            on_step=on_step,
        )


def _run_query(
    ctx: OrchestratorContext,
    *,
    session_id: str | None,
    query: str,
    session_context: str | None,
    max_depth: str | None,
    on_step: Callable[[dict[str, Any]], None] | None,
) -> QueryResult:
    del session_id
    md = RetrievalMetadata()
    max_depth = str(max_depth or ctx.cfg.retrieval.max_depth).upper()
    tenant_id = current_tenant_id()
    planner = ctx.planner or AdaptivePlanner(max_candidates=ctx.cfg.retrieval.max_l1_vector_results)
    evidence_gate = ctx.evidence_gate or EvidenceGate()

    started = time.perf_counter()
    with tracing.span("query.l0_adaptive_plan"):
        plan = planner.plan(query, session_context=session_context, core=ctx.core)
    md.latency_ms["l0_plan"] = (time.perf_counter() - started) * 1000
    md.levels_visited.append("L0")
    md.cascade_depth_reached = "L0"
    md.predicted_depth = plan.primary_route.value
    md.retrieval_route = plan.primary_route.value
    md.l0_decision = "ROUTE"
    md.l0_reason = plan.reason
    md.temporal_scope = plan.temporal_scope.model_dump(mode="json")
    _notify_step(on_step, md, "l0_plan")

    repository = ctx.repository()
    if repository is None and not ctx.cfg.canonical_memory.enabled:
        return _run_v1_projection_compat(
            ctx,
            md,
            plan,
            query=query,
            session_context=session_context,
            tenant_id=tenant_id,
            on_step=on_step,
        )
    attempted: list[RetrievalRoute] = []
    verified: list[VerifiedEvidence] = []
    assessment = evidence_gate.assess(plan, [], attempted_routes=attempted)

    if repository is None:
        md.stop_reason = "canonical-memory-repository-unavailable"
        return _finish_query(
            ctx,
            md,
            plan,
            assessment,
            session_context=session_context,
            query=query,
            on_step=on_step,
        )

    allowed_cap = _depth_rank(max_depth)
    for route in plan.routes:
        if _depth_rank(route.value) > allowed_cap:
            continue
        attempted.append(route)
        md.routes_attempted.append(route.value)
        if route.value not in md.levels_visited:
            md.levels_visited.append(route.value)
        md.cascade_depth_reached = route.value
        started = time.perf_counter()
        with tracing.span(f"query.{route.value.lower()}_retrieve"):
            try:
                route_rows, discovered = _retrieve_route(
                    ctx,
                    repository,
                    plan,
                    route,
                    tenant_id=tenant_id,
                )
            except AmbiguousEntityReferenceError as err:
                md.stop_reason = "ambiguous-entity-reference"
                assessment = EvidenceAssessment(
                    state=AnswerabilityState.INSUFFICIENT_EVIDENCE,
                    verified_evidence=[],
                    missing_evidence=[f"disambiguation:{err.alias}"],
                    conflicts=[],
                    can_escalate=False,
                    reason=str(err),
                )
                md.latency_ms[f"{route.value.lower()}_retrieve"] = (
                    time.perf_counter() - started
                ) * 1000
                return _finish_query(
                    ctx,
                    md,
                    plan,
                    assessment,
                    session_context=session_context,
                    query=query,
                    on_step=on_step,
                )
            except CanonicalRepositoryUnavailableError as err:
                log.error("canonical memory retrieval unavailable: %s", err)
                md.stop_reason = "canonical-memory-repository-unavailable"
                assessment = evidence_gate.assess(plan, [], attempted_routes=attempted)
                md.latency_ms[f"{route.value.lower()}_retrieve"] = (
                    time.perf_counter() - started
                ) * 1000
                return _finish_query(
                    ctx,
                    md,
                    plan,
                    assessment,
                    session_context=session_context,
                    query=query,
                    on_step=on_step,
                )
        md.latency_ms[f"{route.value.lower()}_retrieve"] = (time.perf_counter() - started) * 1000
        md.candidates_discovered += discovered
        verified.extend(route_rows)
        verified = _dedupe_verified(verified)
        verified = _rank_and_limit_semantic_evidence(verified, plan)
        md.nodes_retrieved = len(verified)
        md.verified_evidence = len(verified)
        md.canonical_revision = _latest_revision(verified)
        _notify_step(on_step, md, f"{route.value.lower()}_retrieve")

        assessment = evidence_gate.assess(plan, verified, attempted_routes=attempted)
        _record_assessment(md, assessment)
        _notify_step(on_step, md, "evidence_gate")
        if assessment.state is AnswerabilityState.ANSWERABLE:
            md.stop_reason = f"evidence-sufficient-at-{route.value}"
            break
        if not assessment.can_escalate:
            md.stop_reason = "evidence-gate-terminal"
            break

    _record_assessment(md, assessment)
    if md.stop_reason is None:
        md.stop_reason = "adaptive-routes-exhausted"
    return _finish_query(
        ctx,
        md,
        plan,
        assessment,
        session_context=session_context,
        query=query,
        on_step=on_step,
    )


def _retrieve_route(
    ctx: OrchestratorContext,
    repository: MemoryRepository | Any,
    plan: RetrievalPlan,
    route: RetrievalRoute,
    *,
    tenant_id: str,
) -> tuple[list[VerifiedEvidence], int]:
    """Retrieve one capability and enforce PG hydration for discovery."""

    if route is RetrievalRoute.L2:
        # L2 uses Neo4j only to discover a small, relevance-ranked set of
        # stable IDs. PostgreSQL hydration below remains mandatory before any
        # candidate can become evidence. If the replaceable projection is
        # unavailable, canonical PostgreSQL neighbor retrieval is the safe
        # (but broader) degraded fallback.
        raw_rows = _neo_discover(ctx, plan, tenant_id=tenant_id)
        if not raw_rows:
            raw_rows, _ = _repository_retrieve(repository, plan, route, tenant_id=tenant_id)
        discovered_count = len(raw_rows)
        canonical_neighbors = [
            RetrievalCandidate.from_row(row)
            for row in raw_rows
            if bool(_field(row, "canonical"))
            and str(_field(row, "retrieval_level") or "").startswith(
                "L2_canonical_neighbor"
            )
        ]
        if canonical_neighbors and len(canonical_neighbors) == discovered_count:
            verified = (ctx.verifier or CanonicalVerifier()).verify(
                canonical_neighbors,
                plan=plan,
                tenant_id=tenant_id,
                require_stable_id=True,
            )
            return verified, discovered_count
        candidates = [
            DiscoveryCandidate.from_row(row, retrieval_level="L2_discovery") for row in raw_rows
        ]
        candidates = [row for row in candidates if row.memory_id or row.claim_id]
        if not candidates:
            return [], discovered_count
        hydrated = _repository_hydrate(repository, candidates, plan, tenant_id=tenant_id)
        verified = (ctx.verifier or CanonicalVerifier()).verify(
            hydrated,
            plan=plan,
            tenant_id=tenant_id,
            require_stable_id=True,
            discovery=True,
        )
        return verified, discovered_count

    raw_rows, used_repository = _repository_retrieve(repository, plan, route, tenant_id=tenant_id)
    if not used_repository:
        return [], 0
    candidates = [
        RetrievalCandidate.from_row(
            row,
            retrieval_level=f"{route.value}_canonical",
            tenant_id=tenant_id,
            canonical=True,
        )
        for row in raw_rows
    ]
    if route is RetrievalRoute.L3:
        candidates = [row for row in candidates if _overview_is_fresh(row)]
    verified = (ctx.verifier or CanonicalVerifier()).verify(
        candidates,
        plan=plan,
        tenant_id=tenant_id,
        require_stable_id=True,
    )
    return verified, len(candidates)


def _repository_retrieve(
    repository: MemoryRepository | Any,
    plan: RetrievalPlan,
    route: RetrievalRoute,
    *,
    tenant_id: str,
) -> tuple[list[Any], bool]:
    # Prefer the explicit canonical repository adapter before probing generic
    # method names.  In particular, ``MemoryRepository.get_history`` requires
    # a resolved UUID; invoking it as a generic search method would turn a
    # valid L4 request into a false repository-unavailable result.
    builtin = _retrieve_typed_repository(repository, plan, route, tenant_id=tenant_id)
    if builtin is not None:
        return builtin, True
    names = {
        RetrievalRoute.L1: (
            "retrieve_l1",
            "search_current_claims" if plan.intent.value == "current_fact" else "search_recent",
            "search_current",
            "retrieve_current",
            "retrieve",
        ),
        RetrievalRoute.L2: (
            "retrieve_l2",
            "search_semantic",
            "semantic_search",
            "search_relationships",
            "relationship_search",
            "retrieve",
        ),
        RetrievalRoute.L3: (
            "retrieve_l3",
            "overview",
            "fetch_overview",
            "retrieve",
        ),
        RetrievalRoute.L4: (
            "retrieve_l4",
            "get_history",
            "history",
            "search_history",
            "retrieve",
        ),
    }[route]
    for name in names:
        method = getattr(repository, name, None)
        if not callable(method):
            continue
        kwargs = {
            "tenant_id": tenant_id,
            "plan": plan,
            "route": route,
            "query": plan.query,
            "limit": plan.max_candidates,
            "entity_hints": plan.entity_hints,
            "predicate": plan.predicate_hint,
            "predicate_hint": plan.predicate_hint,
            "temporal_scope": plan.temporal_scope,
            "as_of": plan.temporal_scope.point,
        }
        try:
            rows = _invoke(method, kwargs)
        except (CoreModelError, CircuitOpenError, Exception) as err:
            raise CanonicalRepositoryUnavailableError(
                f"{route.value} repository read failed"
            ) from err
        return _coerce_rows(rows), True
    return [], False


def _repository_hydrate(
    repository: MemoryRepository | Any,
    candidates: Sequence[RetrievalCandidate],
    plan: RetrievalPlan,
    *,
    tenant_id: str,
) -> list[Any]:
    """Hydrate discovery IDs through canonical storage or reject them all."""

    names = (
        "hydrate",
        "hydrate_candidates",
        "verify_candidates",
        "hydrate_discovery",
        "get_canonical",
        "get_by_ids",
    )
    kwargs = {
        "tenant_id": tenant_id,
        "candidates": candidates,
        "candidate_rows": candidates,
        "memory_ids": [row.memory_id for row in candidates if row.memory_id],
        "claim_ids": [row.claim_id for row in candidates if row.claim_id],
        "plan": plan,
    }
    for name in names:
        method = getattr(repository, name, None)
        if not callable(method):
            continue
        try:
            rows = _invoke(method, kwargs)
        except (CoreModelError, CircuitOpenError, Exception) as err:
            raise CanonicalRepositoryUnavailableError("candidate hydration failed") from err
        return [
            RetrievalCandidate.from_row(
                row,
                retrieval_level="L2_verified",
                canonical=True,
            )
            for row in _coerce_rows(rows)
        ]
    builtin = _hydrate_typed_repository(repository, candidates, plan, tenant_id=tenant_id)
    if builtin is not None:
        return builtin
    # A discovery result without a canonical hydration method is not usable.
    log.warning(
        "repository has no canonical hydration method; dropping %d candidates", len(candidates)
    )
    return []


def _retrieve_typed_repository(
    repository: Any,
    plan: RetrievalPlan,
    route: RetrievalRoute,
    *,
    tenant_id: str,
) -> list[Any] | None:
    """Bridge the canonical repository's typed read methods to retrieval rows."""

    if not callable(getattr(repository, "resolve_aliases", None)):
        return None
    subject_ids = _resolve_subject_ids(repository, plan, tenant_id=tenant_id)
    if not subject_ids:
        if route is RetrievalRoute.L1 and plan.intent.value == "recent_context":
            recent_method = getattr(repository, "get_recent_states", None)
            if not callable(recent_method):
                return []
            try:
                states = _coerce_rows(
                    _invoke(
                        recent_method,
                        {
                            "since": plan.temporal_scope.since,
                            "until": plan.temporal_scope.until,
                            "limit": plan.max_candidates,
                            "tenant_id": tenant_id,
                        },
                    )
                )
            except Exception as err:
                raise CanonicalRepositoryUnavailableError(
                    "recent canonical memory retrieval failed"
                ) from err
            recent_rows: list[RetrievalCandidate] = []
            for state in states:
                node = _field(state, "node")
                version = _field(state, "current_version")
                if node is None or version is None:
                    continue
                evidence = _coerce_rows(_field(state, "evidence") or [])
                recent_rows.append(
                    RetrievalCandidate.from_row(
                        {
                            "memory_id": str(_field(node, "id")),
                            "tenant_id": tenant_id,
                            "body": _field(version, "body"),
                            "abstract": _field(version, "abstract"),
                            "status": _field(node, "status") or "ACTIVE",
                            "asserted_at": _field(version, "asserted_at"),
                            "valid_from": _field(version, "valid_from"),
                            "valid_until": _field(version, "valid_until"),
                            "canonical_revision": _field(node, "revision"),
                            "evidence_ids": [
                                str(_field(item, "id"))
                                for item in evidence
                                if _field(item, "id") is not None
                            ],
                            "provenance": [_row_to_mapping(item) for item in evidence],
                            "canonical": True,
                        },
                        retrieval_level="L1_recent_canonical",
                        canonical=True,
                    )
                )
            return recent_rows
        return []
    if route is RetrievalRoute.L2 and plan.intent.value == "semantic":
        neighbor_method = getattr(repository, "get_neighbor_states", None)
        if callable(neighbor_method):
            neighbor_rows: list[RetrievalCandidate] = []
            context_anchor = plan.entity_hints[0] if plan.entity_hints else None
            for subject_id in subject_ids:
                try:
                    states = _coerce_rows(
                        _invoke(
                            neighbor_method,
                            {
                                "memory_id": subject_id,
                                "before": 0,
                                "after": 2,
                                "limit": plan.max_candidates,
                                "tenant_id": tenant_id,
                            },
                        )
                    )
                except Exception as err:
                    raise CanonicalRepositoryUnavailableError(
                        "canonical neighbor retrieval failed"
                    ) from err
                for state in states:
                    node = _field(state, "node")
                    version = _field(state, "current_version")
                    if node is None or version is None:
                        continue
                    evidence = _coerce_rows(_field(state, "evidence") or [])
                    neighbor_rows.append(
                        RetrievalCandidate.from_row(
                            {
                                "memory_id": str(_field(node, "id")),
                                "tenant_id": tenant_id,
                                "body": _field(version, "body"),
                                "abstract": _field(version, "abstract"),
                                "status": _field(node, "status") or "ACTIVE",
                                "asserted_at": _field(version, "asserted_at"),
                                "canonical_revision": _field(node, "revision"),
                                "evidence_ids": [
                                    str(_field(item, "id"))
                                    for item in evidence
                                    if _field(item, "id") is not None
                                ],
                                "provenance": [_row_to_mapping(item) for item in evidence],
                                "context_anchor": context_anchor,
                                "canonical": True,
                            },
                            retrieval_level="L2_canonical_neighbor",
                            canonical=True,
                        )
                    )
            if neighbor_rows:
                return neighbor_rows
    if route is RetrievalRoute.L3:
        get_overview = getattr(repository, "get_overview", None)
        if not callable(get_overview):
            return []
        rows: list[Any] = []
        for subject_id in subject_ids:
            try:
                overview = _invoke(
                    get_overview,
                    {
                        "scope_id": subject_id,
                        "tenant_id": tenant_id,
                        "require_current": True,
                    },
                )
            except Exception as err:
                raise CanonicalRepositoryUnavailableError("overview retrieval failed") from err
            if overview is not None:
                rows.append(overview)
            else:
                enqueue = getattr(repository, "enqueue_overview_regeneration", None)
                if callable(enqueue):
                    try:
                        _invoke(
                            enqueue,
                            {"scope_id": subject_id, "tenant_id": tenant_id},
                        )
                    except Exception as err:
                        log.warning("could not enqueue stale overview regeneration: %s", err)
        return rows

    profile_lookup = route is RetrievalRoute.L1 and plan.intent is RetrievalIntent.OVERVIEW
    if profile_lookup:
        method_name = "get_current_related_claims"
    elif route is RetrievalRoute.L4:
        method_name = "get_claim_history"
    elif plan.temporal_scope.kind in {TemporalKind.AS_OF, TemporalKind.BEFORE}:
        method_name = "get_claims_as_of"
    else:
        method_name = "get_current_claims"
    claims_method = getattr(repository, method_name, None)
    if not callable(claims_method):
        return []
    rows = []
    for subject_id in subject_ids:
        kwargs: dict[str, Any] = {
            "predicate": plan.predicate_hint,
            "tenant_id": tenant_id,
        }
        if profile_lookup:
            kwargs["entity_id"] = subject_id
        else:
            kwargs["subject_id"] = subject_id
        if method_name == "get_claims_as_of" or route is RetrievalRoute.L1:
            kwargs["at"] = plan.temporal_scope.point
        try:
            claims = _coerce_rows(_invoke(claims_method, kwargs))
        except Exception as err:
            raise CanonicalRepositoryUnavailableError("claim retrieval failed") from err
        for claim in claims:
            candidate = RetrievalCandidate.from_row(
                claim,
                retrieval_level=f"{route.value}_canonical",
                tenant_id=tenant_id,
                canonical=True,
            )
            candidate = _attach_typed_state(repository, candidate, plan=plan, tenant_id=tenant_id)
            rows.append(_attach_typed_evidence(repository, candidate, tenant_id=tenant_id))
        if route is RetrievalRoute.L4:
            versions_method = getattr(repository, "get_versions", None)
            state_method = getattr(repository, "get_current_state", None)
            if not callable(versions_method) or not callable(state_method):
                continue
            try:
                versions = _coerce_rows(
                    _invoke(
                        versions_method,
                        {"memory_id": subject_id, "tenant_id": tenant_id},
                    )
                )
                state = _invoke(
                    state_method,
                    {
                        "memory_id": subject_id,
                        "tenant_id": tenant_id,
                        "include_historical_claims": True,
                        "require_current_overview": False,
                    },
                )
            except Exception as err:
                raise CanonicalRepositoryUnavailableError(
                    "version history retrieval failed"
                ) from err
            node = _field(state, "node")
            for version in versions:
                candidate = RetrievalCandidate.from_row(
                    {
                        "memory_id": str(subject_id),
                        "version_id": str(_field(version, "id")),
                        "tenant_id": tenant_id,
                        "body": _field(version, "body"),
                        "abstract": _field(version, "abstract"),
                        "status": _field(node, "status") or "ACTIVE",
                        "asserted_at": _field(version, "asserted_at"),
                        "valid_from": _field(version, "valid_from"),
                        "valid_until": _field(version, "valid_until"),
                        "canonical_revision": _field(node, "revision"),
                        "provenance": _field(version, "provenance"),
                        "canonical": True,
                    },
                    retrieval_level="L4_canonical_version",
                    canonical=True,
                )
                rows.append(_attach_typed_evidence(repository, candidate, tenant_id=tenant_id))
    return rows


def _resolve_subject_ids(repository: Any, plan: RetrievalPlan, *, tenant_id: str) -> list[Any]:
    resolver = getattr(repository, "resolve_aliases", None)
    if not callable(resolver):
        return []
    out: list[Any] = []
    seen: set[str] = set()
    for hint in plan.entity_hints:
        direct_id = _coerce_uuid(hint)
        if direct_id is not None:
            key = str(direct_id)
            if key not in seen:
                seen.add(key)
                out.append(direct_id)
            continue
        try:
            aliases = _invoke(
                resolver,
                {
                    "alias": hint,
                    "tenant_id": tenant_id,
                    "include_historical": plan.temporal_scope.kind
                    in {
                        TemporalKind.HISTORY,
                        TemporalKind.ANY,
                        TemporalKind.AS_OF,
                        TemporalKind.BEFORE,
                    },
                },
            )
        except Exception as err:
            raise CanonicalRepositoryUnavailableError(
                f"alias resolution failed for {hint!r}"
            ) from err
        alias_rows = _coerce_rows(aliases)
        candidate_ids = {
            str(entity_id)
            for entity_id in (_field(alias, "entity_id") for alias in alias_rows)
            if entity_id is not None
        }
        if len(candidate_ids) > 1:
            raise AmbiguousEntityReferenceError(hint, sorted(candidate_ids))
        for alias in alias_rows:
            entity_id = _field(alias, "entity_id")
            if entity_id is None:
                continue
            key = str(entity_id)
            if key not in seen:
                seen.add(key)
                out.append(entity_id)
    return out


def _attach_typed_evidence(
    repository: Any, candidate: RetrievalCandidate, *, tenant_id: str
) -> Any:
    """Attach evidence IDs/provenance to a typed claim without guessing text."""

    get_evidence = getattr(repository, "get_evidence", None)
    if not callable(get_evidence) or not candidate.memory_id:
        return candidate

    try:
        rows = _coerce_rows(
            _invoke(
                get_evidence,
                {
                    "memory_id": _coerce_uuid(candidate.memory_id),
                    "claim_id": _coerce_uuid(candidate.claim_id),
                    "tenant_id": tenant_id,
                },
            )
        )
    except Exception as err:
        raise CanonicalRepositoryUnavailableError("evidence retrieval failed") from err
    evidence_ids = [
        str(identifier) for identifier in (_field(row, "id") for row in rows) if identifier
    ]
    data = candidate.model_dump()
    data["evidence_ids"] = evidence_ids
    if rows:
        data["provenance"] = [_row_to_mapping(row) for row in rows]
    return RetrievalCandidate.model_validate(data)


def _attach_typed_state(
    repository: Any,
    candidate: RetrievalCandidate,
    *,
    plan: RetrievalPlan,
    tenant_id: str,
) -> RetrievalCandidate:
    """Add canonical node revision/body when the repository exposes state."""

    state_getter = getattr(repository, "get_current_state", None)
    memory_id = _coerce_uuid(candidate.memory_id)
    if not callable(state_getter) or memory_id is None:
        return candidate
    try:
        state = _invoke(
            state_getter,
            {
                "memory_id": memory_id,
                "tenant_id": tenant_id,
                "include_historical_claims": plan.temporal_scope.kind.value in {"history", "any"},
                "require_current_overview": False,
            },
        )
    except Exception as err:
        raise CanonicalRepositoryUnavailableError("memory state retrieval failed") from err
    if state is None:
        return candidate
    node = _field(state, "node")
    version = _field(state, "current_version")
    updates: dict[str, Any] = {
        "canonical_revision": _field(node, "revision"),
        "status": _field(node, "status") or candidate.status,
        "source_uri": _field(node, "canonical_uri"),
        "subject_name": _field(node, "canonical_name"),
    }
    # Structured claims are already the narrow canonical evidence. Attaching
    # the entity's latest narrative body to every claim duplicates context
    # and can reintroduce a previously generated conversational answer. Whole
    # memory/version candidates still receive their canonical body below.
    if not candidate.predicate:
        updates["body"] = _field(version, "body")
        updates["abstract"] = _field(version, "abstract")
    object_id = _coerce_uuid(candidate.object_entity_id)
    get_memory = getattr(repository, "get_memory", None)
    if object_id is not None and callable(get_memory):
        try:
            object_node = _invoke(
                get_memory,
                {"memory_id": object_id, "tenant_id": tenant_id},
            )
        except Exception as err:
            raise CanonicalRepositoryUnavailableError(
                "claim object hydration failed"
            ) from err
        updates["object_entity_name"] = _field(object_node, "canonical_name")
    return _replace_candidate(
        candidate,
        **{key: value for key, value in updates.items() if value is not None},
    )


def _hydrate_typed_repository(
    repository: Any,
    candidates: Sequence[RetrievalCandidate],
    plan: RetrievalPlan,
    *,
    tenant_id: str,
) -> list[Any] | None:
    """Hydrate graph IDs through ``get_current_state`` and canonical claims."""

    state_getter = getattr(repository, "get_current_state", None)
    if not callable(state_getter):
        return None
    output: list[Any] = []
    include_history = plan.temporal_scope.kind.value in {"history", "any"}
    for candidate in candidates:
        memory_id = _coerce_uuid(candidate.memory_id)
        if memory_id is None and candidate.claim_id:
            claim_getter = getattr(repository, "get_claim", None)
            if callable(claim_getter):
                try:
                    claim = _invoke(
                        claim_getter,
                        {
                            "claim_id": _coerce_uuid(candidate.claim_id),
                            "tenant_id": tenant_id,
                        },
                    )
                except Exception as err:
                    raise CanonicalRepositoryUnavailableError(
                        "canonical claim hydration failed"
                    ) from err
                memory_id = _coerce_uuid(_field(claim, "subject_id"))
        if memory_id is None:
            continue
        try:
            state = _invoke(
                state_getter,
                {
                    "memory_id": memory_id,
                    "tenant_id": tenant_id,
                    "include_historical_claims": include_history,
                    "require_current_overview": False,
                },
            )
        except Exception as err:
            raise CanonicalRepositoryUnavailableError("canonical state hydration failed") from err
        if state is None:
            continue
        node = _field(state, "node")
        revision = _field(node, "revision")
        claims = _coerce_rows(_field(state, "claims") or [])
        if candidate.claim_id:
            claims = [row for row in claims if str(_field(row, "id")) == candidate.claim_id]
        if plan.predicate_hint:
            claims = [
                row
                for row in claims
                if _predicate_equal(_field(row, "predicate"), plan.predicate_hint)
            ]
        if not claims:
            # A graph candidate can identify a memory without a claim.  The
            # current version is still canonical evidence for overview-like
            # semantic queries, but only when it has a stable memory ID.
            version = _field(state, "current_version")
            if version is None:
                continue
            data = RetrievalCandidate.from_row(
                {
                    "memory_id": str(memory_id),
                    "tenant_id": tenant_id,
                    "text": _field(version, "body"),
                    "abstract": _field(version, "abstract"),
                    "status": _field(node, "status") or "ACTIVE",
                    "canonical_revision": revision,
                    "projected_revision": candidate.projected_revision,
                    "retrieval_score": candidate.retrieval_score,
                    "context_anchor": (candidate.model_extra or {}).get("context_anchor"),
                    "canonical": True,
                },
                retrieval_level="L2_verified",
                canonical=True,
            )
            output.append(data)
            continue
        for claim in claims:
            data = RetrievalCandidate.from_row(
                claim,
                retrieval_level="L2_verified",
                tenant_id=tenant_id,
                canonical=True,
            )
            data = _replace_candidate(
                data,
                canonical_revision=revision,
                projected_revision=candidate.projected_revision,
                retrieval_score=candidate.retrieval_score,
            )
            output.append(_attach_typed_evidence(repository, data, tenant_id=tenant_id))
    return output


def _replace_candidate(candidate: RetrievalCandidate, **updates: Any) -> RetrievalCandidate:
    data = candidate.model_dump()
    data.update(updates)
    return RetrievalCandidate.model_validate(data)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _row_to_mapping(row: Any) -> Any:
    if isinstance(row, Mapping):
        return dict(row)
    return {
        key: _field(row, key)
        for key in (
            "id",
            "source_event_id",
            "source_session_id",
            "extractor",
            "confidence",
            "source_span",
        )
        if _field(row, key) is not None
    }


def _coerce_uuid(value: Any) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _predicate_equal(actual: Any, expected: str) -> bool:
    return str(actual or "").casefold().replace(" ", "_").replace("-", "_") == expected.casefold()


def _neo_discover(ctx: OrchestratorContext, plan: RetrievalPlan, *, tenant_id: str) -> list[Any]:
    """Use Neo4j only as L2 discovery; no row is trusted here."""

    neo = ctx.neo4j
    embed = ctx.embed
    if neo is None or embed is None:
        return []
    search = getattr(neo, "vector_search", None)
    if not callable(search):
        return []
    try:
        embedding = embed.embed(plan.query)
        kwargs = {
            "k": plan.max_candidates,
            "dormant_floor": ctx.cfg.decay.dormant_floor,
            "tenant_id": tenant_id,
        }
        return _coerce_rows(search(embedding, **kwargs))
    except (CoreModelError, CircuitOpenError, Exception) as err:
        log.warning("L2 discovery failed: %s", err)
        return []


def _run_v1_projection_compat(
    ctx: OrchestratorContext,
    md: RetrievalMetadata,
    plan: RetrievalPlan,
    *,
    query: str,
    session_context: str | None,
    tenant_id: str,
    on_step: Callable[[dict[str, Any]], None] | None,
) -> QueryResult:
    """Keep explicitly configured V1 instances queryable during cutover.

    Production V2 never enters this branch. It exists only for deployments
    that deliberately keep ``canonical_memory.enabled=false`` and therefore
    still have Neo4j-only legacy data. No filesystem content is read here.
    """

    started = time.perf_counter()
    candidates = _neo_discover(ctx, plan, tenant_id=tenant_id)
    md.latency_ms["legacy_projection_discovery"] = (time.perf_counter() - started) * 1000
    md.levels_visited.append("L1")
    md.cascade_depth_reached = "L1"
    md.retrieval_route = "V1_NEO4J_COMPAT"
    md.routes_attempted.append("L1")
    md.candidates_discovered = len(candidates)
    md.nodes_retrieved = len(candidates)
    _notify_step(on_step, md, "legacy_projection_discovery")

    evidence_lines: list[str] = []
    for row in candidates:
        content = next(
            (
                str(value).strip()
                for value in (
                    _field(row, "l0_abstract"),
                    _field(row, "abstract"),
                    _field(row, "body"),
                    _field(row, "overview"),
                )
                if value is not None and str(value).strip()
            ),
            "",
        )
        if not content:
            continue
        source = str(_field(row, "source_uri") or _field(row, "id") or "legacy-projection")
        evidence_lines.append(f"{content} (source: {source})")

    msc_parts = [part for part in (session_context, *evidence_lines) if part]
    msc = "\n".join(msc_parts)
    md.total_context_tokens = tok_mod.count_tokens(msc)
    state = (
        AnswerabilityState.ANSWERABLE
        if evidence_lines
        else AnswerabilityState.INSUFFICIENT_EVIDENCE
    )
    md.answerability_state = state.value

    frontier = ctx.frontier
    if frontier is None:
        answer = evidence_lines[0] if evidence_lines else "No relevant memory was retrieved."
    else:
        try:
            verdict = frontier.answer(
                system_prompt=(
                    "Answer only from the supplied legacy projection context. "
                    "This compatibility route is disabled in canonical V2."
                ),
                msc=msc,
                user_query=query,
                allow_need_more=False,
            )
            answer = verdict.answer or (
                evidence_lines[0] if evidence_lines else "No relevant memory was retrieved."
            )
        except (CoreModelError, CircuitOpenError, Exception) as err:
            log.error("legacy frontier call failed: %s", err)
            answer = "The answer service is temporarily unavailable."
            md.stop_reason = "frontier-provider-unavailable"
    return QueryResult(answer, md, state.value, [])


def _invoke(method: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call repository methods with only parameters they declare."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return method(**kwargs)
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return method(**filtered)


def _coerce_rows(rows: Any) -> list[Any]:
    if rows is None:
        return []
    if isinstance(rows, Mapping):
        return [rows]
    if isinstance(rows, (str, bytes)):
        return []
    if isinstance(rows, Iterable):
        return list(rows)
    return [rows]


def _overview_is_fresh(row: RetrievalCandidate) -> bool:
    extra = row.model_extra or {}
    input_revision = extra.get("input_revision")
    canonical_revision = row.canonical_revision or extra.get("current_revision")
    if input_revision is None or canonical_revision is None:
        # Repositories that do not expose revision metadata cannot prove a
        # derived overview is current.
        return False
    try:
        return int(input_revision) == int(canonical_revision)
    except (TypeError, ValueError):
        return False


def _dedupe_verified(rows: Sequence[VerifiedEvidence]) -> list[VerifiedEvidence]:
    out: list[VerifiedEvidence] = []
    positions: dict[str, int] = {}
    for row in rows:
        version_id = (row.model_extra or {}).get("version_id")
        if row.predicate:
            identity = ":".join(
                (
                    str(row.subject_id or row.memory_id or ""),
                    str(row.predicate).casefold(),
                    str(row.object_entity_id or row.object_value).casefold(),
                    str(row.valid_from or ""),
                    str(row.valid_until or ""),
                    str(row.status),
                )
            )
        else:
            identity = str(version_id or row.memory_id or row.source_uri or row.identity)
        existing_position = positions.get(identity)
        if existing_position is not None:
            existing = out[existing_position]
            data = existing.model_dump()
            data["evidence_ids"] = list(dict.fromkeys([*existing.evidence_ids, *row.evidence_ids]))
            if row.provenance is not None:
                provenance = existing.provenance
                if provenance is None:
                    data["provenance"] = row.provenance
                elif isinstance(provenance, list):
                    additions = (
                        row.provenance if isinstance(row.provenance, list) else [row.provenance]
                    )
                    data["provenance"] = [*provenance, *additions]
            out[existing_position] = VerifiedEvidence.model_validate(data)
            continue
        positions[identity] = len(out)
        out.append(row)
    return out


_SEMANTIC_RANK_STOP_WORDS = {
    "about",
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "please",
    "that",
    "the",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _rank_and_limit_semantic_evidence(
    rows: Sequence[VerifiedEvidence],
    plan: RetrievalPlan,
) -> list[VerifiedEvidence]:
    """Rank and globally bound semantic evidence after PG verification.

    Existing vector scores remain the primary ordering signal. Canonical
    fallback rows have no vector score, so deterministic query coverage ranks
    them before the cap is applied. Claim confidence is deliberately not
    mixed into retrieval relevance.
    """

    if plan.intent is RetrievalIntent.OVERVIEW:
        return _rank_profile_evidence(rows, plan)
    if plan.intent.value not in {"semantic", "general"}:
        return list(rows)

    ranked: list[tuple[tuple[float, float, float, int], VerifiedEvidence]] = []
    for position, row in enumerate(rows):
        lexical_score = _semantic_lexical_score(plan.query, row)
        vector_score = row.retrieval_score
        score = (
            1.0 if vector_score is not None else 0.0,
            float(vector_score or lexical_score),
            lexical_score,
            -position,
        )
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked[: plan.max_candidates]]


_PROFILE_PREDICATE_PRIORITY = {
    "HAS_MANAGER": 100,
    "MANAGES": 100,
    "HAS_ASSISTANT_MANAGER": 95,
    "HAS_ROLE": 95,
    "WORKS_AT": 90,
    "MEMBER_OF": 85,
    "WORKING_ON": 80,
    "LOCATED_IN": 75,
    "APPROVES": 55,
    "REQUESTS": 45,
}


def _rank_profile_evidence(
    rows: Sequence[VerifiedEvidence],
    plan: RetrievalPlan,
) -> list[VerifiedEvidence]:
    """Keep entity profiles compact and centered on identity-bearing facts."""

    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            _PROFILE_PREDICATE_PRIORITY.get(str(item[1].predicate or "").upper(), 20),
            item[1].claim_confidence or 0.0,
            _semantic_lexical_score(plan.query, item[1]),
            -item[0],
        ),
        reverse=True,
    )
    selected: list[VerifiedEvidence] = []
    seen: set[str] = set()
    limit = min(plan.max_candidates, 5)
    for _, row in ranked:
        identity = _profile_fact_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


_PROFILE_VALUE_STOP_WORDS = {"a", "an", "fictional", "for", "of", "story", "the"}


def _profile_fact_identity(row: VerifiedEvidence) -> str:
    """Collapse wording variants without merging genuinely different facts."""

    predicate = str(row.predicate or "").upper()
    subject = _profile_value_key(row.subject_name or row.subject_id)
    object_value = _profile_value_key(
        row.object_entity_name or row.object_entity_id or row.object_value
    )
    if predicate == "HAS_MANAGER":
        return f"MANAGER|{object_value}|{subject}"
    if predicate == "MANAGES":
        return f"MANAGER|{subject}|{object_value}"
    return f"{predicate}|{subject}|{object_value}"


def _profile_value_key(value: Any) -> str:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold().replace("_", " "))
        if token not in _PROFILE_VALUE_STOP_WORDS
    }
    return " ".join(sorted(tokens))


def _semantic_lexical_score(query: str, row: VerifiedEvidence) -> float:
    query_terms = {
        _semantic_term_root(token)
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", query.casefold())
        if len(token) > 2 and token not in _SEMANTIC_RANK_STOP_WORDS
    }
    if not query_terms:
        return 0.0
    searchable = " ".join(
        str(value)
        for value in (
            row.content,
            row.subject_name,
            row.predicate,
            row.object_entity_name,
            row.object_value,
        )
        if value is not None
    ).casefold()
    searchable_terms = {
        _semantic_term_root(token)
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", searchable)
    }
    return len(query_terms & searchable_terms) / len(query_terms)


def _semantic_term_root(term: str) -> str:
    """Normalize common English inflections without a model dependency."""

    if term.endswith("ies") and len(term) > 4:
        return f"{term[:-3]}y"
    if term.endswith("ing") and len(term) > 5:
        return term[:-3]
    if term.endswith("ed") and len(term) > 4:
        return term[:-2]
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def _latest_revision(rows: Sequence[VerifiedEvidence]) -> int | None:
    revisions = [row.canonical_revision for row in rows if row.canonical_revision is not None]
    return max(revisions) if revisions else None


def _record_assessment(md: RetrievalMetadata, assessment: EvidenceAssessment) -> None:
    md.answerability_state = assessment.state.value
    md.verified_evidence = len(assessment.verified_evidence)
    md.nodes_retrieved = len(assessment.verified_evidence)
    md.missing_evidence = list(assessment.missing_evidence)
    md.conflicts = list(assessment.conflicts)


def _finish_query(
    ctx: OrchestratorContext,
    md: RetrievalMetadata,
    plan: RetrievalPlan,
    assessment: EvidenceAssessment,
    *,
    session_context: str | None,
    query: str,
    on_step: Callable[[dict[str, Any]], None] | None,
) -> QueryResult:
    _record_assessment(md, assessment)

    # Exact current facts do not need a generative answer when PostgreSQL has
    # already supplied one unambiguous, evidence-backed claim.  Rendering the
    # small supported predicate set directly both improves factual precision
    # and avoids spending answer-model tokens on routine lookups.
    deterministic_answer = _deterministic_current_answer(plan, assessment)
    if deterministic_answer is None:
        deterministic_answer = _deterministic_profile_answer(plan, assessment)
    if deterministic_answer is not None:
        md.total_context_tokens = 0
        md.answer_mode = "deterministic-canonical"
        _notify_step(on_step, md, "deterministic_answer")
        return QueryResult(
            deterministic_answer,
            md,
            assessment.state.value,
            assessment.verified_evidence,
        )

    # Terminal evidence outcomes are deterministic product responses.  Do not
    # assemble model context that will never be sent to a provider.
    if assessment.state is AnswerabilityState.INSUFFICIENT_EVIDENCE:
        if md.stop_reason == "canonical-memory-repository-unavailable":
            answer = (
                "Canonical memory retrieval is temporarily unavailable; "
                "I don't have verified memory evidence to answer that."
            )
        elif md.stop_reason == "ambiguous-entity-reference":
            answer = (
                "I found multiple canonical entities matching that name; "
                "please provide more identifying context."
            )
        else:
            answer = "I don't have enough verified memory evidence to answer that."
        md.total_context_tokens = 0
        md.answer_mode = "evidence-terminal"
        md.stop_reason = md.stop_reason or "insufficient-canonical-evidence"
        _notify_step(on_step, md, "evidence_terminal")
        return QueryResult(answer, md, assessment.state.value, assessment.verified_evidence)
    if assessment.state is AnswerabilityState.CONFLICTING_EVIDENCE:
        md.total_context_tokens = 0
        md.answer_mode = "evidence-terminal"
        answer = (
            "I found conflicting verified memory evidence, so I cannot select one current answer."
        )
        md.stop_reason = md.stop_reason or "conflicting-canonical-evidence"
        _notify_step(on_step, md, "evidence_terminal")
        return QueryResult(answer, md, assessment.state.value, assessment.verified_evidence)

    started = time.perf_counter()
    context_budget = (
        ctx.cfg.retrieval.full_doc_budget_tokens
        if RetrievalRoute.L4 in plan.routes and RetrievalRoute.L4.value in md.routes_attempted
        else ctx.cfg.retrieval.overview_budget_tokens
    )
    builder = ctx.context_builder or ContextBuilder(
        context_window=context_budget,
        session_share=0.15,
        evidence_share=0.70,
    )
    msc = builder.build(
        plan,
        assessment,
        session_context=session_context,
        user_query=query,
        context_window=context_budget,
    )
    md.total_context_tokens = tok_mod.count_tokens(msc)
    md.latency_ms["msc_assembly"] = (time.perf_counter() - started) * 1000
    _notify_step(on_step, md, "msc_assembly")

    frontier = ctx.frontier
    if frontier is None:
        md.stop_reason = md.stop_reason or "frontier-provider-unavailable"
        return QueryResult(
            _fallback_answer(assessment),
            md,
            assessment.state.value,
            assessment.verified_evidence,
        )

    started = time.perf_counter()
    try:
        verdict: FrontierVerdict = frontier.answer(
            system_prompt=(
                "You are Engram's final answer generator. Use only the verified "
                "canonical evidence in the context. Preserve the stated "
                "answerability state, temporal scope, and uncertainty. "
                "Do not infer missing facts from similarity scores. Answer the "
                "user's exact question in the first sentence, then add only "
                "useful supporting facts. For an entity profile, prioritize "
                "identity, role, organization or project, and important current "
                "relationships. Preserve relationship direction exactly. Do not "
                "describe how many memories, events, or chapters mention a fact "
                "unless the user asks for provenance or history. Do not append a "
                "generic no-more-evidence disclaimer when the retrieval state is "
                "ANSWERABLE. Avoid internal retrieval terminology and repeated "
                "facts. In entity profiles, render machine-style scalar values "
                "such as project_orion as readable names and state qualifiers "
                "such as fictional only once. Prefer two to four concise sentences."
            ),
            msc=msc,
            user_query=query,
            allow_need_more=False,
        )
        answer = verdict.answer or _fallback_answer(assessment)
    except (CoreModelError, CircuitOpenError, Exception) as err:
        log.error("frontier call failed: %s", err)
        answer = (
            "The answer generator is temporarily unavailable. "
            "Verified retrieval completed but no answer could be produced."
        )
    md.latency_ms["frontier_answer_0"] = (time.perf_counter() - started) * 1000
    md.reentries = 0
    _notify_step(on_step, md, "frontier_answer")
    return QueryResult(answer, md, assessment.state.value, assessment.verified_evidence)


_EXACT_PREDICATE_TEMPLATES: dict[str, str] = {
    "WORKS_AT": "{subject} works at {objects}.",
    "HAS_MANAGER": "{subject}'s manager is {objects}.",
    "HAS_ASSISTANT_MANAGER": "{subject}'s assistant manager is {objects}.",
    "HAS_ROLE": "{subject}'s current role is {objects}.",
    "MEMBER_OF": "{subject} is a member of {objects}.",
    "LOCATED_IN": "{subject} is located in {objects}.",
    "HAS_BIRTH_DATE": "{subject}'s birth date is {objects}.",
    "ATTENDS_SCHOOL": "{subject} attended {objects}.",
}


def _deterministic_current_answer(
    plan: RetrievalPlan,
    assessment: EvidenceAssessment,
) -> str | None:
    """Render verified exact/current claims without invoking a language model."""

    if (
        assessment.state is not AnswerabilityState.ANSWERABLE
        or plan.intent.value != "current_fact"
        or plan.temporal_scope.kind is not TemporalKind.CURRENT
        or plan.requires_evidence
        or not plan.predicate_hint
        # A fixed sentence about a guessed label answers a different question.
        or plan.predicate_hint_guessed
    ):
        return None
    predicate = normalize_predicate(plan.predicate_hint).canonical_predicate
    template = _EXACT_PREDICATE_TEMPLATES.get(predicate)
    rows = [
        row
        for row in assessment.verified_evidence
        if normalize_predicate(row.predicate or "").canonical_predicate == predicate
    ]
    if template is None or not rows or len(rows) > 5:
        return None
    subject_names = {str(row.subject_name).strip() for row in rows if row.subject_name}
    if len(subject_names) != 1:
        return None
    values_by_identity: dict[str, str] = {}
    for row in rows:
        raw_value = row.object_entity_name
        if not raw_value and row.object_entity_id is None:
            raw_value = _answer_scalar(row.object_value)
        value = str(raw_value or "").strip()
        if not value:
            return None
        identity = value.casefold()
        existing = values_by_identity.get(identity)
        if existing is None or _answer_value_quality(value) > _answer_value_quality(existing):
            values_by_identity[identity] = value
    values = list(values_by_identity.values())
    if not values:
        return None
    return template.format(
        subject=next(iter(subject_names)),
        objects=_join_answer_values(values),
    )


def _deterministic_profile_answer(
    plan: RetrievalPlan,
    assessment: EvidenceAssessment,
) -> str | None:
    """Render an exact entity profile from current canonical relationships.

    Identity questions such as ``Who is Rahul?`` do not need a generative
    model once PostgreSQL has supplied a compact, verified claim set.  Keeping
    this path deterministic prevents wording drift, avoids leaking storage
    spellings such as ``project_orion``, and spends no frontier-model tokens.
    """

    if (
        assessment.state is not AnswerabilityState.ANSWERABLE
        or plan.intent is not RetrievalIntent.OVERVIEW
        or plan.primary_route is not RetrievalRoute.L1
        or plan.temporal_scope.kind is not TemporalKind.CURRENT
        or len(plan.entity_hints) != 1
    ):
        return None

    rows = [row for row in assessment.verified_evidence if row.predicate]
    if not rows:
        return None
    target = _profile_target_name(plan.entity_hints[0], rows)
    if not target:
        return None

    clauses: list[tuple[str, bool]] = []
    for row in rows:
        predicate = normalize_predicate(row.predicate or "").canonical_predicate
        subject = str(row.subject_name or "").strip()
        object_raw = row.object_entity_name
        if not object_raw and row.object_entity_id is None:
            object_raw = _answer_scalar(row.object_value)
        object_value, fictional = _clean_profile_answer_value(object_raw)
        if not object_value:
            continue

        target_is_subject = _same_profile_entity(subject, target)
        target_is_object = _same_profile_entity(row.object_entity_name, target)
        clause: str | None = None
        if predicate == "MANAGES":
            if target_is_subject:
                clause = f"is {object_value}'s manager"
            elif target_is_object and subject:
                clause = f"is managed by {subject}"
        elif predicate == "HAS_MANAGER":
            if target_is_subject:
                clause = f"reports to {object_value}"
            elif target_is_object and subject:
                clause = f"is {subject}'s manager"
        elif predicate == "HAS_ASSISTANT_MANAGER":
            if target_is_subject:
                clause = f"has {object_value} as assistant manager"
            elif target_is_object and subject:
                clause = f"is {subject}'s assistant manager"
        elif not target_is_subject:
            continue
        elif predicate == "HAS_ROLE":
            clause = f"serves as {object_value}"
        elif predicate == "WORKS_AT":
            clause = f"works at {object_value}"
        elif predicate == "MEMBER_OF":
            clause = f"is a member of {object_value}"
        elif predicate == "WORKING_ON":
            clause = f"works on {object_value}"
        elif predicate == "LOCATED_IN":
            clause = f"is located in {object_value}"
        elif predicate == "APPROVES":
            clause = f"approves {_profile_approval_object(object_value)}"
        elif predicate == "REQUESTS":
            clause = f"requested {object_value}"
        if clause:
            clauses.append((clause, fictional))

    clauses = _dedupe_profile_clauses(clauses)[:3]
    if not clauses:
        return None
    prefix = "Within the uploaded story, " if any(flag for _, flag in clauses) else ""
    first = clauses[:2]
    answer = f"{prefix}{target} {_join_answer_values([clause for clause, _ in first])}."
    if len(clauses) == 3:
        answer += f" {target} {_profile_also_clause(clauses[2][0])}."
    return answer


def _profile_target_name(
    hint: str,
    rows: Sequence[VerifiedEvidence],
) -> str:
    for row in rows:
        for value in (row.subject_name, row.object_entity_name):
            if value and _same_profile_entity(value, hint):
                return str(value).strip()
    value, _ = _clean_profile_answer_value(hint)
    return value


def _same_profile_entity(value: Any, target: str) -> bool:
    return _profile_value_key(value) == _profile_value_key(target)


def _clean_profile_answer_value(value: Any) -> tuple[str, bool]:
    text = str(value or "").strip()
    fictional = bool(re.search(r"\bfictional\b", text, flags=re.IGNORECASE))
    text = re.sub(r"\bfictional\b\s*", "", text, flags=re.IGNORECASE)
    snake_case = bool(text) and bool(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", text))
    text = re.sub(r"[_\s]+", " ", text).strip()
    if snake_case:
        text = text.title()
    return text, fictional


def _dedupe_profile_clauses(
    clauses: Sequence[tuple[str, bool]],
) -> list[tuple[str, bool]]:
    output: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for clause, fictional in clauses:
        identity = _profile_value_key(clause)
        if identity in seen:
            continue
        seen.add(identity)
        output.append((clause, fictional))
    return output


def _profile_also_clause(clause: str) -> str:
    if clause.startswith("is "):
        return f"is also {clause[3:]}"
    if clause.startswith("has "):
        return f"also has {clause[4:]}"
    return f"also {clause}"


def _profile_approval_object(value: str) -> str:
    first_word = value.split(maxsplit=1)[0].casefold()
    if first_word in {"a", "an", "the", "this", "that"}:
        return value
    if re.search(r"\b(checklist|design|document|plan|proposal|report)\b", value, re.I):
        return f"the {value}"
    return value


def _answer_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        nested = value.get("value")
        return str(nested) if nested is not None else ""
    if isinstance(value, (str, int, float)):
        return str(value)
    return ""


def _join_answer_values(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _answer_value_quality(value: str) -> tuple[int, int]:
    letters = [character for character in value if character.isalpha()]
    uppercase = sum(1 for character in letters if character.isupper())
    title_words = sum(1 for word in value.split() if word[:1].isupper())
    return uppercase, title_words


def _fallback_answer(assessment: EvidenceAssessment) -> str:
    if assessment.state is AnswerabilityState.PARTIALLY_ANSWERABLE:
        return "I can answer only part of that from the verified memory evidence available."
    return "No verified answer was produced."


__all__ = [
    "OrchestratorContext",
    "QueryResult",
    "RetrievalMetadata",
    "run_query",
]
