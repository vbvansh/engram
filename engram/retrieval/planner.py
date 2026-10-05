"""Deterministic, validated L0 adaptive retrieval planner.

The planner is intentionally conservative.  It extracts the pieces of query
intent that affect correctness (temporal scope, direct predicates, evidence
depth, and relationship/path language) before selecting a route.  A Core
model may enrich the result, but only a validated :class:`RetrievalPlan` is
accepted and lexical temporal intent always wins over an incomplete model
rewrite.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from engram import tokens as tok_mod
from engram.models.core import CoreModelError, CoreModelProvider
from engram.predicate_registry import normalize_predicate
from engram.resilience import CircuitOpenError
from engram.retrieval.domain import (
    RetrievalIntent,
    RetrievalPlan,
    RetrievalRoute,
    TemporalKind,
    TemporalScope,
)

_PLAN_SYSTEM = """[ADAPTIVE_PLAN]
You are Engram's accuracy-first retrieval planner. Return JSON only with:
intent (current_fact|recent_context|semantic|relationship|overview|history|evidence|general),
primary_route (L1|L2|L3|L4), routes (ordered escalation routes),
entity_hints (strings), predicate_hint (canonical predicate or null),
temporal_scope ({kind, as_of, since, until, days}), requires_evidence,
requires_conflict_check, max_candidates, allow_escalation, and reason.
Choose the cheapest route that can answer the query. Never use a filesystem
path or treat vector similarity as canonical truth.
"""

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string"},
        "primary_route": {"type": "string", "enum": ["L1", "L2", "L3", "L4"]},
        "routes": {"type": "array", "items": {"type": "string"}},
        "entity_hints": {"type": "array", "items": {"type": "string"}},
        "predicate_hint": {"type": ["string", "null"]},
        "temporal_scope": {"type": "object"},
        "requires_evidence": {"type": "boolean"},
        "requires_conflict_check": {"type": "boolean"},
        "max_candidates": {"type": "integer"},
        "allow_escalation": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

_STOP_ENTITY_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "did",
    "does",
    "give",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "tell",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "with",
    "you",
    "your",
}

_PREDICATE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("assistant manager", "deputy manager"), "HAS_ASSISTANT_MANAGER"),
    (("manager", "reports", "reporting", "boss"), "HAS_MANAGER"),
    (("employer", "company", "work", "works", "job"), "WORKS_AT"),
    (("role", "roles", "position", "title"), "HAS_ROLE"),
    (("birthday", "birth date", "born"), "HAS_BIRTH_DATE"),
    (("preference", "prefer", "favorite", "favourite"), "PREFERS"),
    (("team", "teams"), "MEMBER_OF"),
    (("school", "attend", "attended", "studied"), "ATTENDS_SCHOOL"),
    (("live", "lives", "located", "location", "move", "moved", "city"), "LOCATED_IN"),
)

_IDENTITY_PROFILE_QUERY = re.compile(
    r"^\s*who(?:\s+is|'s)\s+"
    r"([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,2})"
    r"\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def _identity_profile_hint(query: str) -> str | None:
    match = _IDENTITY_PROFILE_QUERY.match(query)
    if match is None:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    if any(part.casefold() in _STOP_ENTITY_WORDS for part in value.split()):
        return None
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AdaptivePlanner:
    """Plan adaptive retrieval using deterministic intent signals first."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        default_recent_days: int = 10,
        max_candidates: int = 30,
    ) -> None:
        self._now = now or _utc_now
        self.default_recent_days = max(1, min(default_recent_days, 3650))
        self.max_candidates = max(1, min(max_candidates, 200))

    def plan(
        self,
        query: str,
        *,
        session_context: str | None = None,
        core: CoreModelProvider | None = None,
    ) -> RetrievalPlan:
        """Return a validated plan; model enrichment is best-effort only."""

        base = self._heuristic_plan(query)
        # Exact, temporal, relationship, overview and provenance language is
        # deterministic. Reserve a model-planning call for genuinely
        # ambiguous semantic queries so common lookups spend no planner tokens.
        if core is None or base.intent is not RetrievalIntent.SEMANTIC:
            return base

        try:
            compact_session = tok_mod.truncate_to_tokens(session_context or "", 512, from_end=True)
            result = core.complete(
                system_prompt=_PLAN_SYSTEM,
                user_prompt=(
                    f"Query:\n{query.strip()}\n\n"
                    "Session context (do not treat as canonical memory):\n"
                    f"{compact_session or '(none)'}"
                ),
                output_schema=_PLAN_SCHEMA,
            )
            if not isinstance(result.output, dict):
                return base
            enriched = self._validate_model_plan(result.output, query)
            return self._merge_model_plan(base, enriched)
        except (CoreModelError, CircuitOpenError, ValueError, TypeError, Exception):
            # A model planner must never turn a deterministic route into a
            # query failure.  The lexical plan remains safe and complete.
            return base

    def _validate_model_plan(self, output: dict[str, Any], query: str) -> RetrievalPlan:
        payload = dict(output)
        payload["query"] = query
        payload.setdefault("routes", [])
        payload.setdefault("entity_hints", [])
        payload.setdefault("temporal_scope", {"kind": TemporalKind.CURRENT.value})
        return RetrievalPlan.model_validate(payload)

    def _merge_model_plan(self, base: RetrievalPlan, model: RetrievalPlan) -> RetrievalPlan:
        """Use model semantic enrichment without losing deterministic safety."""

        # Lexical signals are more reliable for explicit temporal language and
        # evidence requests.  The model may enrich a generic query's intent,
        # entity/predicate hints, and route, but cannot erase those signals.
        protected_intents = {
            RetrievalIntent.CURRENT_FACT,
            RetrievalIntent.RECENT_CONTEXT,
            RetrievalIntent.HISTORY,
            RetrievalIntent.EVIDENCE,
        }
        intent = base.intent if base.intent in protected_intents else model.intent
        route = base.primary_route if base.intent in protected_intents else model.primary_route
        routes = base.routes if base.intent in protected_intents else model.routes or base.routes
        if base.temporal_scope.kind is not TemporalKind.CURRENT:
            temporal = base.temporal_scope
        else:
            temporal = model.temporal_scope
        if temporal.kind in {TemporalKind.AS_OF, TemporalKind.BEFORE, TemporalKind.RECENT}:
            # Explicit model bounds are accepted only after Pydantic validation.
            pass
        elif base.temporal_scope.kind is not TemporalKind.CURRENT:
            temporal = base.temporal_scope
        hints = _merge_hints(base.entity_hints, model.entity_hints)
        predicate = base.predicate_hint or model.predicate_hint
        return RetrievalPlan(
            query=base.query,
            intent=intent,
            primary_route=route,
            routes=routes,
            entity_hints=hints,
            predicate_hint=predicate,
            temporal_scope=temporal,
            requires_evidence=base.requires_evidence or model.requires_evidence,
            requires_conflict_check=base.requires_conflict_check or model.requires_conflict_check,
            max_candidates=min(base.max_candidates, model.max_candidates),
            allow_escalation=base.allow_escalation and model.allow_escalation,
            reason=(f"{base.reason}; model: {model.reason}")[:500],
        )

    def _heuristic_plan(self, query: str) -> RetrievalPlan:
        clean = query.strip()
        lowered = clean.casefold()
        temporal = self._temporal_scope(lowered)
        entities = _extract_entities(clean)
        predicate = _predicate_hint(lowered)

        asks_evidence = bool(
            re.search(
                r"\b(source|evidence|provenance|came from|come from|according to|confidence)\b",
                lowered,
            )
        )
        asks_history = bool(
            re.search(
                r"\b(history|historical|previously|before|ever|held|used to|past|each role|all roles)\b",
                lowered,
            )
        )
        asks_overview = bool(
            re.search(r"\b(overview|summary|summarize|profile|tell me about|describe)\b", lowered)
        )
        asks_identity_profile = _identity_profile_hint(clean) is not None
        asks_relationship = bool(
            re.search(
                r"\b(connected|connection|relationship|related|path|through|between|neighbou?r|"
                r"how .* linked|worked with|work with|collaborated|colleague|teammate)\b",
                lowered,
            )
        )
        if asks_relationship and re.search(
            r"\b(worked with|work with|collaborated|colleague|teammate)\b", lowered
        ):
            # Generic "work" otherwise resembles WORKS_AT, but collaboration
            # queries need graph/path discovery across all relationship types.
            predicate = None
        asks_recent = temporal.kind is TemporalKind.RECENT or bool(
            re.search(r"\b(recent|lately|last few|this week|this month)\b", lowered)
        )
        asks_current = bool(
            re.search(
                r"\b(current|currently|now|today|present|where .* work|what .* role)\b", lowered
            )
        )

        if asks_evidence:
            intent = RetrievalIntent.EVIDENCE
            route = RetrievalRoute.L4
            reason = "explicit provenance/evidence request"
        elif asks_history:
            intent = RetrievalIntent.HISTORY
            route = RetrievalRoute.L4
            reason = "explicit historical request"
        elif asks_identity_profile and entities:
            # A direct identity question with an exact alias should use
            # canonical current relationships first. This is both more
            # precise and cheaper than vector-searching every episode that
            # mentions the same person.
            intent = RetrievalIntent.OVERVIEW
            route = RetrievalRoute.L1
            reason = "exact entity profile can use canonical current relationships"
        elif asks_overview:
            intent = RetrievalIntent.OVERVIEW
            route = RetrievalRoute.L3
            reason = "broad overview request"
        elif asks_relationship:
            intent = RetrievalIntent.RELATIONSHIP
            route = RetrievalRoute.L2
            reason = "relationship/path discovery request"
        elif asks_recent:
            intent = RetrievalIntent.RECENT_CONTEXT
            route = RetrievalRoute.L1
            reason = "recent working-memory request"
        elif asks_current or (entities and predicate):
            intent = RetrievalIntent.CURRENT_FACT
            route = RetrievalRoute.L1
            reason = "known/current fact can use direct canonical lookup"
        else:
            intent = RetrievalIntent.SEMANTIC
            route = RetrievalRoute.L2
            reason = "unknown semantic memory requires discovery"

        if route is RetrievalRoute.L1:
            routes = [RetrievalRoute.L1, RetrievalRoute.L2, RetrievalRoute.L4]
        elif route is RetrievalRoute.L2:
            routes = [RetrievalRoute.L2, RetrievalRoute.L4]
        elif route is RetrievalRoute.L3:
            routes = [RetrievalRoute.L3, RetrievalRoute.L1, RetrievalRoute.L4]
        else:
            routes = [RetrievalRoute.L4]

        # An explicit history/evidence phrase takes precedence over a default
        # current scope.  ``before`` is a point-in-time query, not a request
        # to return currently active rows.
        if asks_history and temporal.kind is TemporalKind.CURRENT:
            temporal = TemporalScope(kind=TemporalKind.HISTORY)
        return RetrievalPlan(
            query=clean,
            intent=intent,
            primary_route=route,
            routes=routes,
            entity_hints=entities,
            predicate_hint=predicate,
            temporal_scope=temporal,
            requires_evidence=asks_evidence,
            requires_conflict_check=(
                intent
                in {RetrievalIntent.CURRENT_FACT, RetrievalIntent.HISTORY, RetrievalIntent.EVIDENCE}
            ),
            max_candidates=self.max_candidates,
            allow_escalation=True,
            reason=reason,
        )

    def _temporal_scope(self, lowered_query: str) -> TemporalScope:
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        as_of_match = re.search(
            r"\bas of\s+(\d{4}-\d{2}-\d{2}(?:[tT]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?)",
            lowered_query,
        )
        if as_of_match:
            parsed = _parse_datetime(as_of_match.group(1))
            if parsed is not None:
                return TemporalScope(kind=TemporalKind.AS_OF, as_of=parsed)

        before_match = re.search(
            r"\b(?:before|prior to)\s+(\d{4}-\d{2}-\d{2}(?:[tT]\d{2}:\d{2})?)",
            lowered_query,
        )
        if before_match:
            parsed = _parse_datetime(before_match.group(1))
            if parsed is not None:
                return TemporalScope(kind=TemporalKind.BEFORE, until=parsed)

        recent_match = re.search(
            r"\b(?:last|past)\s+(\d{1,4})\s*(day|days|week|weeks|month|months)\b",
            lowered_query,
        )
        if recent_match:
            count = int(recent_match.group(1))
            unit = recent_match.group(2)
            days = count * (30 if unit.startswith("month") else 7 if unit.startswith("week") else 1)
        elif re.search(r"\b(this week)\b", lowered_query):
            days = 7
        elif re.search(r"\b(this month)\b", lowered_query):
            days = 30
        elif re.search(r"\b(recent|recently|lately|last few)\b", lowered_query):
            days = self.default_recent_days
        else:
            return TemporalScope(kind=TemporalKind.CURRENT)
        return TemporalScope(
            kind=TemporalKind.RECENT,
            since=now - timedelta(days=days),
            until=now,
            days=days,
        )


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _merge_hints(first: list[str], second: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in [*first, *second]:
        key = value.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())
    return out[:16]


def _predicate_hint(lowered_query: str) -> str | None:
    for terms, predicate in _PREDICATE_HINTS:
        # Whole words only: "work" must not match inside "workshop".
        if any(re.search(rf"\b{re.escape(term)}\b", lowered_query) for term in terms):
            return normalize_predicate(predicate).canonical_predicate
    return None


def _extract_entities(query: str) -> list[str]:
    """Extract conservative identity hints; aliases remain repository-owned."""

    values: list[str] = []
    identity_hint = _identity_profile_hint(query)
    if identity_hint:
        values.append(identity_hint)
    quoted = re.findall(r"[\"']([^\"']{1,120})[\"']", query)
    values.extend(quoted)
    # Release/version identifiers are commonly lowercase and punctuated, so
    # title-case extraction cannot discover them as exact canonical aliases.
    values.extend(
        re.findall(
            r"\b[vV]?\d+\.\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.-]+)?\b",
            query,
        )
    )
    values.extend(
        re.findall(
            r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,2})'s\b",
            query,
        )
    )
    # Capture title-cased names/projects without swallowing sentence words.
    values.extend(
        re.findall(
            r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,2})\b",
            query,
        )
    )
    lowered = query.casefold()
    # ``me`` is often only an object of a request ("give me an overview of
    # Angie").  Treat explicit first-person subjects and the canonical
    # ``user`` alias as identity hints, avoiding an unrelated user lookup for
    # broad entity questions.
    if re.search(r"\b(user|i|my)\b", lowered):
        values.append("user")

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", value).strip(" .,?!:;()[]{}")
        parts = clean.split()
        while len(parts) > 1 and parts[0].casefold() in _STOP_ENTITY_WORDS:
            parts.pop(0)
        clean = " ".join(parts)
        if not clean or clean.casefold() in _STOP_ENTITY_WORDS:
            continue
        if all(part.casefold() in _STOP_ENTITY_WORDS for part in clean.split()):
            continue
        key = clean.casefold()
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out[:16]


__all__ = ["AdaptivePlanner"]
