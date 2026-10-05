"""Canonical verification and evidence sufficiency for adaptive retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from engram.predicate_registry import normalize_predicate
from engram.retrieval.domain import (
    AnswerabilityState,
    EvidenceAssessment,
    RetrievalCandidate,
    RetrievalIntent,
    RetrievalPlan,
    RetrievalRoute,
    TemporalKind,
    VerifiedEvidence,
)

_ACTIVE_STATUSES = {"ACTIVE", "CURRENT"}
_TERMINAL_STATUSES = {"DELETED", "INVALID"}


class CanonicalVerifier:
    """Validate hydrated rows before they can become model context."""

    def verify(
        self,
        candidates: Iterable[RetrievalCandidate | dict[str, Any] | Any],
        *,
        plan: RetrievalPlan,
        tenant_id: str,
        require_stable_id: bool = False,
        discovery: bool = False,
    ) -> list[VerifiedEvidence]:
        verified: list[VerifiedEvidence] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for row in candidates:
            candidate = RetrievalCandidate.from_row(row)
            if not candidate.canonical:
                continue
            if candidate.tenant_id != tenant_id:
                continue
            if require_stable_id and not (candidate.memory_id or candidate.claim_id):
                # A source_uri-only graph hit is legacy/discovery data, not a
                # canonical identity.  It must not bypass PostgreSQL hydration.
                continue
            if candidate.status in _TERMINAL_STATUSES:
                continue
            if not _status_allowed(candidate.status, plan):
                continue
            if discovery and _projection_is_stale(candidate):
                continue
            if not _temporal_match(candidate, plan):
                continue
            identity = (
                candidate.claim_id or candidate.memory_id or candidate.source_uri or "",
                candidate.predicate,
                _value_key(candidate),
            )
            if identity in seen:
                continue
            seen.add(identity)
            data = candidate.model_dump()
            data["canonical"] = True
            # Canonical repositories may encode an unresolved policy conflict
            # in lifecycle status rather than a separate boolean.  Preserve
            # that signal so the evidence gate cannot present it as a normal
            # current fact.
            data["conflict"] = candidate.conflict or candidate.status == "CONFLICTING"
            data["verification_source"] = "postgresql"
            verified.append(VerifiedEvidence.model_validate(data))
        verified.sort(
            key=lambda row: (
                row.retrieval_score is not None,
                row.retrieval_score or 0.0,
                row.claim_confidence or 0.0,
            ),
            reverse=True,
        )
        return verified


class EvidenceGate:
    """Decide whether verified canonical evidence is sufficient to answer."""

    def __init__(self, *, minimum_confidence: float = 0.0) -> None:
        self.minimum_confidence = max(0.0, min(minimum_confidence, 1.0))

    def assess(
        self,
        plan: RetrievalPlan,
        evidence: Sequence[VerifiedEvidence | RetrievalCandidate | dict[str, Any]],
        *,
        attempted_routes: Sequence[RetrievalRoute | str] = (),
    ) -> EvidenceAssessment:
        verified = [
            row
            if isinstance(row, VerifiedEvidence)
            else VerifiedEvidence.model_validate(
                RetrievalCandidate.from_row(row, canonical=True).model_dump()
            )
            for row in evidence
        ]
        next_route = _next_route(plan, attempted_routes)
        if plan.predicate_hint:
            exact = [
                row for row in verified if _predicate_matches(row.predicate, plan.predicate_hint)
            ]
            if exact or (plan.allow_escalation and next_route is not None):
                # While a route that may hold the exact fact remains, an exact
                # predicate query must not become answerable from an unrelated
                # claim or a whole-memory body.
                verified = exact
            else:
                # Every route was tried and no fact carries the expected label.
                # The label is only a guess from the question's wording (or the
                # model planner), so answer from the found memories like an
                # ordinary semantic query instead of refusing. Conflict checks
                # concern the asked label, which none of these rows carries.
                plan = plan.model_copy(
                    update={"predicate_hint": None, "requires_conflict_check": False}
                )
                verified = [row for row in verified if _query_covered(plan.query, row)]
        elif plan.intent in {RetrievalIntent.SEMANTIC, RetrievalIntent.GENERAL}:
            # Vector similarity is discovery, not evidence of query coverage.
            # Require a lexical bridge in hydrated canonical text before an
            # unrelated semantic hit can reach the answer model.
            verified = [row for row in verified if _query_covered(plan.query, row)]
        missing: list[str] = []
        conflicts = self._conflicts(plan, verified)
        if conflicts:
            state = AnswerabilityState.CONFLICTING_EVIDENCE
            reason = "multiple active canonical claims conflict"
        elif not verified:
            state = AnswerabilityState.INSUFFICIENT_EVIDENCE
            missing.append(_missing_label(plan))
            reason = "no candidate survived canonical verification"
        else:
            low_confidence = [
                row
                for row in verified
                if row.claim_confidence is not None
                and row.claim_confidence < self.minimum_confidence
            ]
            if low_confidence:
                verified = [row for row in verified if row not in low_confidence]
                missing.append("claim confidence")

            if plan.requires_evidence:
                without_provenance = [
                    row for row in verified if not row.evidence_ids and row.provenance is None
                ]
                if without_provenance:
                    missing.append("provenance/evidence")

            if missing:
                state = AnswerabilityState.PARTIALLY_ANSWERABLE
                reason = "canonical evidence covers only part of the request"
            else:
                state = AnswerabilityState.ANSWERABLE
                reason = "verified canonical evidence is sufficient"

        can_escalate = bool(plan.allow_escalation and next_route is not None)
        if state is AnswerabilityState.ANSWERABLE:
            can_escalate = False
            next_route = None
        elif not can_escalate:
            reason = f"{reason}; no approved escalation route remains"
        return EvidenceAssessment(
            state=state,
            verified_evidence=verified,
            missing_evidence=_unique(missing),
            conflicts=_unique(conflicts),
            can_escalate=can_escalate,
            next_route=next_route,
            reason=reason,
        )

    def _conflicts(self, plan: RetrievalPlan, evidence: Sequence[VerifiedEvidence]) -> list[str]:
        if not plan.requires_conflict_check or plan.temporal_scope.kind is TemporalKind.HISTORY:
            return [row.identity or "unknown-claim" for row in evidence if row.conflict]
        explicit = [
            row.conflict_group or row.identity or "unknown-claim"
            for row in evidence
            if row.conflict
        ]
        groups: dict[tuple[str, str], list[VerifiedEvidence]] = defaultdict(list)
        for row in evidence:
            predicate = (row.predicate or "").casefold()
            if not predicate:
                continue
            subject = row.subject_id or row.memory_id or row.source_uri or "unknown-subject"
            groups[(subject, predicate)].append(row)
        inferred: list[str] = []
        for (subject, predicate), rows in groups.items():
            if len(rows) < 2:
                continue
            values = {_value_key(row) for row in rows}
            if len(values) < 2:
                continue
            if _intervals_overlap(rows):
                inferred.append(f"{subject}:{predicate}")
        return _unique([*explicit, *inferred])


def _status_allowed(status: str, plan: RetrievalPlan) -> bool:
    if plan.temporal_scope.kind in {
        TemporalKind.HISTORY,
        TemporalKind.ANY,
        TemporalKind.AS_OF,
        TemporalKind.BEFORE,
    }:
        # Retired and superseded state is valid historical evidence. Deleted
        # or invalid state remains unusable at every retrieval depth.
        return status not in _TERMINAL_STATUSES
    return status in _ACTIVE_STATUSES


def _projection_is_stale(candidate: RetrievalCandidate) -> bool:
    return bool(
        candidate.projected_revision is not None
        and candidate.canonical_revision is not None
        and candidate.projected_revision < candidate.canonical_revision
    )


def _temporal_match(candidate: RetrievalCandidate, plan: RetrievalPlan) -> bool:
    scope = plan.temporal_scope
    valid_from = _aware(candidate.valid_from)
    valid_until = _aware(candidate.valid_until)
    if scope.kind in {TemporalKind.CURRENT, TemporalKind.AS_OF, TemporalKind.BEFORE}:
        point = _aware(scope.point)
        if point is None:
            # Current canonical rows have already been status-filtered.  A row
            # with no validity interval is valid for current point-in-time use.
            return True
        asserted = _aware(candidate.asserted_at)
        # Knowledge cannot be used for an as-of/before answer prior to the
        # time it was asserted, even when the extractor supplied no explicit
        # validity interval.
        if asserted is not None and asserted > point:
            return False
        return not (
            (valid_from is not None and valid_from > point)
            or (valid_until is not None and valid_until <= point)
        )
    if scope.kind is TemporalKind.RECENT:
        since = _aware(scope.since)
        until = _aware(scope.until)
        asserted = _aware(candidate.asserted_at)
        if since is None and scope.days is not None:
            now = datetime.now(timezone.utc)
            since = now - timedelta(days=scope.days)
        if valid_from is None and valid_until is None:
            return asserted is None or since is None or asserted >= since
        return not (
            (since is not None and valid_until is not None and valid_until <= since)
            or (until is not None and valid_from is not None and valid_from >= until)
        )
    return True


def _intervals_overlap(rows: Sequence[VerifiedEvidence]) -> bool:
    for idx, left in enumerate(rows):
        for right in rows[idx + 1 :]:
            if _interval_overlap(left, right):
                return True
    return False


def _interval_overlap(left: VerifiedEvidence, right: VerifiedEvidence) -> bool:
    # Open intervals are considered overlapping unless explicit bounds prove
    # they are disjoint.  This is conservative for current-state answers.
    left_start, left_end = _aware(left.valid_from), _aware(left.valid_until)
    right_start, right_end = _aware(right.valid_from), _aware(right.valid_until)
    if left_end is not None and right_start is not None and left_end <= right_start:
        return False
    return not (right_end is not None and left_start is not None and right_end <= left_start)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _predicate_matches(actual: str | None, expected: str) -> bool:
    if not actual:
        return False
    return (
        normalize_predicate(actual).canonical_predicate
        == normalize_predicate(expected).canonical_predicate
    )


_QUERY_STOP_WORDS = {
    "about",
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "give",
    "have",
    "how",
    "is",
    "me",
    "my",
    "of",
    "please",
    # Broad domain words are not a sufficient lexical bridge for semantic
    # verification.  Requiring a specific term prevents an unrelated
    # Project/record from being treated as evidence for another project.
    "project",
    "tell",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _query_covered(query: str, row: VerifiedEvidence) -> bool:
    context_anchor = str((row.model_extra or {}).get("context_anchor") or "").strip()
    if context_anchor and context_anchor.casefold() in query.casefold():
        # PostgreSQL session-neighbor retrieval establishes a deterministic
        # canonical context bridge even when the answer event does not repeat
        # the exact anchor string.
        return True
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", query.casefold())
        if len(token) > 2 and token not in _QUERY_STOP_WORDS
    }
    if not query_tokens:
        return False
    searchable = " ".join(
        str(value)
        for value in (
            row.content,
            row.predicate,
            row.object_value,
            row.object_entity_id,
            row.subject_id,
        )
        if value is not None
    ).casefold()
    return any(token in searchable for token in query_tokens)


def _value_key(row: RetrievalCandidate) -> str:
    if row.object_entity_id:
        return f"entity:{row.object_entity_id}"
    if row.object_value is not None:
        return f"value:{row.object_value!r}"
    return row.content.casefold().strip()


def _missing_label(plan: RetrievalPlan) -> str:
    if plan.predicate_hint:
        return f"predicate:{plan.predicate_hint}"
    if plan.entity_hints:
        return f"entity:{plan.entity_hints[0]}"
    return "canonical memory evidence"


def _next_route(
    plan: RetrievalPlan, attempted: Sequence[RetrievalRoute | str]
) -> RetrievalRoute | None:
    attempted_values = {
        str(route.value if isinstance(route, RetrievalRoute) else route) for route in attempted
    }
    for route in plan.routes:
        if route.value not in attempted_values:
            return route
    return None


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = ["CanonicalVerifier", "EvidenceGate"]
