"""Validated domain objects used by adaptive retrieval.

The retrieval pipeline intentionally keeps its domain contract independent of
any concrete database adapter.  PostgreSQL repositories can return mappings,
ORM objects, or these models; the orchestration boundary normalises them here.
In particular, a vector/graph result is a *candidate* until it has been
hydrated by :class:`MemoryRepository` and marked canonical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnswerabilityState(StrEnum):
    """Evidence state exposed before final answer generation."""

    ANSWERABLE = "ANSWERABLE"
    PARTIALLY_ANSWERABLE = "PARTIALLY_ANSWERABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class RetrievalIntent(StrEnum):
    """Small, stable intent vocabulary for the L0 adaptive planner."""

    CURRENT_FACT = "current_fact"
    RECENT_CONTEXT = "recent_context"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    OVERVIEW = "overview"
    HISTORY = "history"
    EVIDENCE = "evidence"
    GENERAL = "general"


class RetrievalRoute(StrEnum):
    """Retrieval capabilities, not an ordered waterfall."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class TemporalKind(StrEnum):
    CURRENT = "current"
    AS_OF = "as_of"
    RECENT = "recent"
    BEFORE = "before"
    HISTORY = "history"
    ANY = "any"


class TemporalScope(BaseModel):
    """Validated temporal intent passed to canonical repository queries.

    ``as_of``/``since``/``until`` are deliberately optional because the
    repository can apply its transaction-time clock for ``current`` queries.
    The planner, rather than the final LLM, owns this interpretation.
    """

    model_config = ConfigDict(extra="forbid")

    kind: TemporalKind = TemporalKind.CURRENT
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    days: int | None = Field(default=None, ge=1, le=3650)

    @model_validator(mode="after")
    def _validate_bounds(self) -> TemporalScope:
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("temporal since must be earlier than until")
        if self.kind is TemporalKind.AS_OF and self.as_of is None:
            raise ValueError("as_of temporal scope requires as_of")
        if self.kind is TemporalKind.RECENT and self.since is None and self.days is None:
            raise ValueError("recent temporal scope requires since or days")
        if self.kind is TemporalKind.BEFORE and self.until is None and self.as_of is None:
            raise ValueError("before temporal scope requires until or as_of")
        return self

    @property
    def point(self) -> datetime | None:
        """Return the point-in-time bound for current/as-of/before queries."""

        return self.as_of or (self.until if self.kind is TemporalKind.BEFORE else None)


class RetrievalPlan(BaseModel):
    """Structured, validated output of L0 adaptive routing."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8_000)
    intent: RetrievalIntent = RetrievalIntent.GENERAL
    primary_route: RetrievalRoute = RetrievalRoute.L2
    routes: list[RetrievalRoute] = Field(default_factory=list, max_length=4)
    entity_hints: list[str] = Field(default_factory=list, max_length=16)
    predicate_hint: str | None = Field(default=None, max_length=128)
    # True when predicate_hint is only guessed from a keyword in a question
    # that asks for something more ("How does John say his team handles tough
    # opponents?"): the label orders evidence but never ends the search alone.
    predicate_hint_guessed: bool = False
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    requires_evidence: bool = False
    requires_conflict_check: bool = False
    max_candidates: int = Field(default=30, ge=1, le=200)
    allow_escalation: bool = True
    reason: str = Field(default="adaptive route", max_length=500)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("entity_hints")
    @classmethod
    def _normalise_hints(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = str(value).strip()
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
        return out

    @model_validator(mode="after")
    def _normalise_routes(self) -> RetrievalPlan:
        ordered: list[RetrievalRoute] = []
        for route in [self.primary_route, *self.routes]:
            if route not in ordered:
                ordered.append(route)
        self.routes = ordered[:4]
        return self

    @property
    def route(self) -> RetrievalRoute:
        """Compatibility alias used by callers that call it a route."""

        return self.primary_route

    @property
    def primary_level(self) -> str:
        """Compatibility alias for metadata and older planner consumers."""

        return self.primary_route.value


class RetrievalCandidate(BaseModel):
    """A discovery or repository row before canonical verification."""

    model_config = ConfigDict(extra="allow")

    memory_id: str | None = None
    claim_id: str | None = None
    subject_id: str | None = None
    subject_name: str | None = None
    object_entity_id: str | None = None
    object_entity_name: str | None = None
    object_value: Any = None
    object_type: str | None = None
    predicate: str | None = None
    source_uri: str | None = None
    tenant_id: str | None = None
    text: str | None = None
    body: str | None = None
    abstract: str | None = None
    overview: str | None = None
    status: str = "ACTIVE"
    retrieval_level: str | None = None
    retrieval_score: float | None = None
    claim_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    asserted_at: datetime | None = None
    canonical_revision: int | None = Field(default=None, ge=0)
    projected_revision: int | None = Field(default=None, ge=0)
    canonical: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: Any = None
    conflict_group: str | None = None
    conflict: bool = False

    @field_validator("status")
    @classmethod
    def _normalise_status(cls, value: str) -> str:
        return str(value or "ACTIVE").upper()

    @property
    def content(self) -> str:
        """Best available content without reading a physical backing store."""

        return str(self.text or self.body or self.abstract or self.overview or "")

    @property
    def identity(self) -> str | None:
        return self.claim_id or self.memory_id or self.source_uri

    @classmethod
    def from_row(
        cls,
        row: RetrievalCandidate | Mapping[str, Any] | Any,
        *,
        retrieval_level: str | None = None,
        tenant_id: str | None = None,
        canonical: bool | None = None,
    ) -> RetrievalCandidate:
        """Normalise a repository/Neo4j mapping while preserving score fields.

        ``score`` is deliberately mapped to ``retrieval_score`` only.  A
        generic ``confidence`` value is accepted as claim confidence for
        compatibility with current rows, but callers should prefer the
        explicit ``claim_confidence`` field.
        """

        if isinstance(row, cls):
            data = row.model_dump()
        elif isinstance(row, Mapping):
            data = dict(row)
        elif is_dataclass(row):
            data = asdict(cast(Any, row))
        else:
            data = {
                key: getattr(row, key)
                for key in (
                    "id",
                    "memory_id",
                    "claim_id",
                    "entity_id",
                    "scope_id",
                    "subject_id",
                    "subject_name",
                    "object_entity_id",
                    "object_entity_name",
                    "object_value",
                    "object_type",
                    "predicate",
                    "source_uri",
                    "canonical_uri",
                    "tenant_id",
                    "text",
                    "body",
                    "abstract",
                    "overview",
                    "status",
                    "retrieval_level",
                    "retrieval_score",
                    "claim_confidence",
                    "confidence",
                    "valid_from",
                    "valid_until",
                    "asserted_at",
                    "canonical_revision",
                    "revision",
                    "input_revision",
                    "current_revision",
                    "projected_revision",
                    "canonical",
                    "evidence_ids",
                    "provenance",
                    "content",
                    "conflict_group",
                    "conflict",
                )
                if hasattr(row, key)
            }
        class_name = type(row).__name__.casefold()
        if data.get("memory_id") is None:
            if "overview" in class_name and data.get("scope_id") is not None:
                data["memory_id"] = data.get("scope_id")
            elif "node" in class_name and data.get("id") is not None:
                data["memory_id"] = data.get("id")
            elif "version" in class_name and data.get("memory_id") is not None:
                data["memory_id"] = data.get("memory_id")
            elif "claim" in class_name and data.get("subject_id") is not None:
                data["memory_id"] = data.get("subject_id")
            elif data.get("entity_id") is not None:
                data["memory_id"] = data.get("entity_id")
            elif data.get("subject_id") is not None and data.get("predicate") is not None:
                data["memory_id"] = data.get("subject_id")
        if data.get("claim_id") is None and "claim" in class_name and data.get("id") is not None:
            data["claim_id"] = data.get("id")
        if (
            data.get("claim_id") is None
            and data.get("id") is not None
            and data.get("subject_id") is not None
            and data.get("predicate") is not None
        ):
            data["claim_id"] = data.get("id")
        if data.get("source_uri") is None and data.get("canonical_uri") is not None:
            data["source_uri"] = data.get("canonical_uri")
        if data.get("overview") is None and "overview" in class_name:
            data["overview"] = data.get("content")
        if data.get("canonical_revision") is None:
            data["canonical_revision"] = data.get("revision") or data.get("input_revision")
        if (
            data.get("evidence_ids") is None
            and "evidence" in class_name
            and data.get("id") is not None
        ):
            data["evidence_ids"] = [str(data.get("id"))]
        if data.get("retrieval_score") is None and data.get("score") is not None:
            data["retrieval_score"] = data.get("score")
        if data.get("claim_confidence") is None and data.get("confidence") is not None:
            data["claim_confidence"] = data.get("confidence")
        if data.get("memory_id") is None and data.get("entity_id") is not None:
            data["memory_id"] = data.get("entity_id")
        if data.get("text") is None:
            data["text"] = data.get("content")
        for key in (
            "memory_id",
            "claim_id",
            "subject_id",
            "object_entity_id",
            "source_uri",
            "tenant_id",
        ):
            if data.get(key) is not None:
                data[key] = str(data[key])
        if data.get("evidence_ids") is not None:
            data["evidence_ids"] = [str(value) for value in data["evidence_ids"]]
        if retrieval_level is not None:
            data["retrieval_level"] = retrieval_level
        if tenant_id is not None and data.get("tenant_id") is None:
            data["tenant_id"] = tenant_id
        if canonical is not None:
            data["canonical"] = canonical
        return cls.model_validate(data)


class DiscoveryCandidate(RetrievalCandidate):
    """Neo4j/vector discovery output that has not yet been verified."""

    canonical: bool = False


class VerifiedEvidence(RetrievalCandidate):
    """Candidate proven against canonical PostgreSQL state."""

    canonical: bool = True
    verification_source: str = "postgresql"


class EvidenceAssessment(BaseModel):
    """Evidence-gate decision made before final model context construction."""

    model_config = ConfigDict(extra="forbid")

    state: AnswerabilityState
    verified_evidence: list[VerifiedEvidence] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    can_escalate: bool = False
    next_route: RetrievalRoute | None = None
    reason: str = Field(default="", max_length=500)

    @property
    def answerability_state(self) -> AnswerabilityState:
        return self.state


class MemoryRepository(Protocol):
    """Structural protocol consumed by adaptive retrieval.

    Implementations may expose the single ``retrieve``/``hydrate`` pair or
    route-specific methods (``search_current``, ``get_history``, etc.).  The
    orchestrator intentionally detects both shapes so the canonical PG
    repository can evolve without coupling retrieval to SQL implementation
    details.
    """

    def retrieve(
        self,
        *,
        tenant_id: str,
        plan: RetrievalPlan,
        route: RetrievalRoute,
        limit: int,
    ) -> Sequence[Mapping[str, Any] | RetrievalCandidate]: ...

    def hydrate(
        self,
        *,
        tenant_id: str,
        candidates: Sequence[RetrievalCandidate],
        plan: RetrievalPlan,
    ) -> Sequence[Mapping[str, Any] | RetrievalCandidate]: ...
