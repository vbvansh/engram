"""Focused tests for canonical adaptive retrieval boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from engram.config import EngramConfig
from engram.models.core import CompletionResult, CoreModelProvider
from engram.models.frontier import FrontierLLMProvider, FrontierVerdict
from engram.retrieval.domain import (
    AnswerabilityState,
    RetrievalCandidate,
    RetrievalPlan,
    RetrievalRoute,
    TemporalKind,
    TemporalScope,
    VerifiedEvidence,
)
from engram.retrieval.evidence import CanonicalVerifier, EvidenceGate
from engram.retrieval.orchestrator import OrchestratorContext, run_query
from engram.retrieval.planner import AdaptivePlanner
from tests.integration.providers import DeterministicCoreProvider, DeterministicEmbeddingService


class RecordingFrontier(FrontierLLMProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer(
        self,
        *,
        system_prompt: str,
        msc: str,
        user_query: str,
        allow_need_more: bool = True,
    ) -> FrontierVerdict:
        del system_prompt, user_query, allow_need_more
        self.calls.append(msc)
        return FrontierVerdict(verdict="ANSWER", answer="verified answer")


class CountingCore(CoreModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **_kwargs: Any) -> CompletionResult:
        self.calls += 1
        return CompletionResult(output={}, raw_text="{}")


def _cfg() -> EngramConfig:
    return EngramConfig.model_validate(
        {
            "api": {"api_key": "test-key"},
            "core_model": {"provider": "openai_responses", "api_key": "x"},
            "frontier_llm": {"provider": "openai_responses", "api_key": "x"},
            "event_ledger": {"dsn": "postgresql://test:test/test"},
            "session_cache": {"backend": "memory"},
            "knowledge_graph": {"backend": "memory", "writer_password": "x"},
        }
    )


def _canonical_claim(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tenant_id": "_default",
        "memory_id": "memory-angie",
        "claim_id": "claim-role",
        "subject_id": "entity-angie",
        "predicate": "has_role",
        "object_value": "Director",
        "object_type": "string",
        "status": "ACTIVE",
        "claim_confidence": 0.95,
        "canonical_revision": 4,
        "text": "Angie is a Director.",
        "evidence_ids": ["evidence-1"],
        "canonical": True,
    }
    row.update(overrides)
    return row


class DirectRepository:
    def __init__(self) -> None:
        self.calls: list[RetrievalRoute] = []

    def retrieve(self, *, route: RetrievalRoute, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        self.calls.append(route)
        if route is RetrievalRoute.L1:
            return [_canonical_claim()]
        return []


class UnavailableRepository:
    def retrieve(self, **_kwargs: Any) -> list[dict[str, Any]]:
        raise ConnectionError("PostgreSQL is unavailable")


class DiscoveryRepository:
    def __init__(self) -> None:
        self.hydrate_calls = 0

    def hydrate(
        self, *, candidates: list[RetrievalCandidate], **kwargs: Any
    ) -> list[dict[str, Any]]:
        del kwargs
        self.hydrate_calls += 1
        assert candidates[0].memory_id == "memory-angie"
        return [_canonical_claim()]


class StaleProjectionRepository:
    def hydrate(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [_canonical_claim(projected_revision=2, canonical_revision=4)]


class TypedHistoryRepository:
    entity_id = UUID("11111111-1111-1111-1111-111111111111")
    version_id = UUID("22222222-2222-2222-2222-222222222222")

    def resolve_aliases(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(entity_id=self.entity_id)]

    def get_claim_history(self, **_kwargs: Any) -> list[Any]:
        return []

    def get_versions(self, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                id=self.version_id,
                body="Angie's historical profile version.",
                abstract="Historical Angie profile.",
                asserted_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                valid_from=None,
                valid_until=None,
                provenance={"source": "test"},
            )
        ]

    def get_current_state(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            node=SimpleNamespace(revision=2, status="ACTIVE"),
            current_version=None,
        )

    def get_evidence(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(id="evidence-version")]

    def get_history(self, **_kwargs: Any) -> Any:
        raise AssertionError("generic get_history must not run before UUID resolution")


class RecentRepository:
    def resolve_aliases(self, **_kwargs: Any) -> list[Any]:
        return []

    def get_recent_states(self, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                node=SimpleNamespace(
                    id=UUID("33333333-3333-3333-3333-333333333333"),
                    revision=1,
                    status="ACTIVE",
                ),
                current_version=SimpleNamespace(
                    body="A recent verified episode.",
                    abstract="Recent episode.",
                    asserted_at=datetime.now(timezone.utc),
                    valid_from=None,
                    valid_until=None,
                ),
                evidence=(SimpleNamespace(id="evidence-recent"),),
            )
        ]


class AmbiguousAliasRepository:
    def resolve_aliases(self, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(entity_id=UUID("44444444-4444-4444-4444-444444444444")),
            SimpleNamespace(entity_id=UUID("55555555-5555-5555-5555-555555555555")),
        ]


class AsOfRepository:
    entity_id = UUID("66666666-6666-6666-6666-666666666666")
    claim_id = UUID("77777777-7777-7777-7777-777777777777")

    def __init__(self) -> None:
        self.as_of_calls = 0
        self.include_historical = False

    def resolve_aliases(self, *, include_historical: bool = False, **_kwargs: Any) -> list[Any]:
        self.include_historical = include_historical
        return [SimpleNamespace(entity_id=self.entity_id)]

    def get_current_claims(self, **_kwargs: Any) -> list[Any]:
        raise AssertionError("as-of retrieval must not read current-only claims")

    def get_claims_as_of(self, **_kwargs: Any) -> list[Any]:
        self.as_of_calls += 1
        return [
            SimpleNamespace(
                id=self.claim_id,
                tenant_id="_default",
                subject_id=self.entity_id,
                predicate="HAS_ROLE",
                object_value="Engineer",
                object_entity_id=None,
                object_type="ROLE",
                confidence=0.95,
                status="SUPERSEDED",
                asserted_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                valid_until=datetime(2022, 1, 1, tzinfo=timezone.utc),
            )
        ]

    def get_current_state(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            node=SimpleNamespace(revision=2, status="ACTIVE", canonical_uri="mem://test"),
            current_version=None,
        )

    def get_evidence(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(id="evidence-as-of")]


class TypedExactRepository:
    subject_id = UUID("88888888-8888-8888-8888-888888888888")
    object_id = UUID("99999999-9999-9999-9999-999999999999")
    claim_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def resolve_aliases(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(entity_id=self.subject_id)]

    def get_current_claims(self, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                id=self.claim_id,
                tenant_id="_default",
                subject_id=self.subject_id,
                predicate="WORKS_AT",
                object_value=None,
                object_entity_id=self.object_id,
                object_type="ENTITY",
                confidence=0.95,
                status="ACTIVE",
                asserted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_from=None,
                valid_until=None,
            )
        ]

    def get_current_state(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            node=SimpleNamespace(
                revision=2,
                status="ACTIVE",
                canonical_uri="mem://memory/88888888-8888-8888-8888-888888888888",
                canonical_name="Atul Singh",
            ),
            current_version=SimpleNamespace(
                body="Atul Singh works at 63moons.",
                abstract="Atul works at 63moons.",
            ),
        )

    def get_memory(self, memory_id: UUID, **_kwargs: Any) -> Any:
        assert memory_id == self.object_id
        return SimpleNamespace(canonical_name="63moons")

    def get_evidence(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(id="evidence-exact")]


class TypedNeighborRepository:
    anchor_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    episode_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")

    def resolve_aliases(self, **_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(entity_id=self.anchor_id)]

    def get_neighbor_states(self, **_kwargs: Any) -> list[Any]:
        return [self.get_current_state(memory_id=self.episode_id)]

    def get_current_state(self, *, memory_id: UUID, **_kwargs: Any) -> Any:
        assert memory_id == self.episode_id
        return SimpleNamespace(
            node=SimpleNamespace(
                id=self.episode_id,
                revision=1,
                status="ACTIVE",
                canonical_name="Release answer episode",
            ),
            current_version=SimpleNamespace(
                body="The release contains validation fixes and improved checks.",
                abstract="Release changes.",
                asserted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            claims=(),
            evidence=(SimpleNamespace(id="evidence-neighbor"),),
        )


class NeoDiscovery:
    def vector_search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"memory_id": "memory-angie", "claim_id": "claim-role", "score": 0.99}]


class UriOnlyNeo:
    def vector_search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"source_uri": "mem://legacy/angie.md", "score": 0.99}]


def _context(repo: Any, frontier: RecordingFrontier, *, neo: Any = None) -> OrchestratorContext:
    return OrchestratorContext(
        cfg=_cfg(),
        fs=None,
        neo4j=neo,
        core=DeterministicCoreProvider(),
        frontier=frontier,
        embed=DeterministicEmbeddingService(),
        memory_repository=repo,
    )


def test_current_fact_routes_directly_to_l1_and_stops() -> None:
    repo = DirectRepository()
    frontier = RecordingFrontier()
    result = run_query(
        _context(repo, frontier),
        session_id=None,
        query="What is Angie's current role?",
    )

    assert repo.calls == [RetrievalRoute.L1]
    assert result.answerability_state == AnswerabilityState.ANSWERABLE.value
    assert result.retrieval_metadata.levels_visited == ["L0", "L1"]
    assert result.retrieval_metadata.routes_attempted == ["L1"]
    assert "claim_confidence: 0.950" in frontier.calls[0]
    assert "retrieval_score" not in frontier.calls[0]


def test_postgres_unavailable_never_falls_back_to_unverified_projection() -> None:
    frontier = RecordingFrontier()
    result = run_query(
        _context(UnavailableRepository(), frontier, neo=NeoDiscovery()),
        session_id=None,
        query="What is Angie's current role?",
    )

    assert result.answerability_state == AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    assert result.retrieval_metadata.stop_reason == "canonical-memory-repository-unavailable"
    assert result.retrieval_metadata.total_context_tokens == 0
    assert result.retrieval_metadata.answer_mode == "evidence-terminal"
    assert "temporarily unavailable" in result.answer.lower()
    assert frontier.calls == []


def test_neo4j_candidate_requires_repository_hydration() -> None:
    repo = DiscoveryRepository()
    frontier = RecordingFrontier()
    result = run_query(
        _context(repo, frontier, neo=NeoDiscovery()),
        session_id=None,
        query="Who is connected to Angie?",
    )

    assert repo.hydrate_calls == 1
    assert result.answerability_state == AnswerabilityState.ANSWERABLE.value
    assert result.retrieval_metadata.candidates_discovered == 1
    assert len(result.verified_evidence) == 1
    assert frontier.calls


def test_stale_neo4j_candidate_is_rejected_after_postgres_hydration() -> None:
    frontier = RecordingFrontier()
    result = run_query(
        _context(StaleProjectionRepository(), frontier, neo=NeoDiscovery()),
        session_id=None,
        query="Who is connected to Angie?",
    )

    assert result.answerability_state == AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    assert frontier.calls == []


def test_l4_uses_resolved_postgres_versions_instead_of_generic_history_probe() -> None:
    frontier = RecordingFrontier()
    result = run_query(
        _context(TypedHistoryRepository(), frontier),
        session_id=None,
        query="Tell me Angie's history.",
    )

    assert result.retrieval_metadata.routes_attempted == ["L4"]
    assert "historical profile version" in frontier.calls[0]


def test_recent_query_reads_postgres_episodes_without_semantic_search() -> None:
    frontier = RecordingFrontier()
    result = run_query(
        _context(RecentRepository(), frontier, neo=NeoDiscovery()),
        session_id=None,
        query="What happened recently?",
    )

    assert result.retrieval_metadata.routes_attempted == ["L1"]
    assert result.retrieval_metadata.candidates_discovered == 1
    assert "recent verified episode" in frontier.calls[0].lower()


def test_ambiguous_alias_requires_disambiguation_instead_of_arbitrary_entity() -> None:
    frontier = RecordingFrontier()
    result = run_query(
        _context(AmbiguousAliasRepository(), frontier),
        session_id=None,
        query="What is Alex Smith's current role?",
    )

    assert result.answerability_state == AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    assert result.retrieval_metadata.stop_reason == "ambiguous-entity-reference"
    assert "identifying context" in result.answer
    assert frontier.calls == []


def test_uri_only_discovery_is_insufficient_and_frontier_is_not_called() -> None:
    repo = DiscoveryRepository()
    frontier = RecordingFrontier()
    result = run_query(
        _context(repo, frontier, neo=UriOnlyNeo()),
        session_id=None,
        query="Who is connected to Angie?",
    )

    # There is no stable memory/claim ID to hydrate; the repository is never
    # allowed to turn a legacy URI hit into factual context.
    assert repo.hydrate_calls == 0
    assert result.answerability_state == AnswerabilityState.INSUFFICIENT_EVIDENCE.value
    assert frontier.calls == []


def test_temporal_verifier_rejects_claim_outside_as_of_point() -> None:
    plan = RetrievalPlan(
        query="role as of 2024-01-01",
        intent="current_fact",
        primary_route="L1",
        routes=["L1"],
        predicate_hint="has_role",
        temporal_scope=TemporalScope(
            kind=TemporalKind.AS_OF,
            as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    candidate = RetrievalCandidate.from_row(
        _canonical_claim(valid_from=datetime(2024, 2, 1, tzinfo=timezone.utc)),
        canonical=True,
    )
    verified = CanonicalVerifier().verify([candidate], plan=plan, tenant_id="_default")
    assert verified == []


def test_temporal_verifier_rejects_claim_asserted_after_as_of_point() -> None:
    plan = RetrievalPlan(
        query="role as of 2024-01-01",
        intent="current_fact",
        primary_route="L1",
        routes=["L1"],
        predicate_hint="HAS_ROLE",
        temporal_scope=TemporalScope(
            kind=TemporalKind.AS_OF,
            as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    candidate = RetrievalCandidate.from_row(
        _canonical_claim(
            valid_from=None,
            valid_until=None,
            asserted_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        ),
        canonical=True,
    )

    assert CanonicalVerifier().verify([candidate], plan=plan, tenant_id="_default") == []


def test_as_of_query_uses_historical_claim_read() -> None:
    repository = AsOfRepository()
    frontier = RecordingFrontier()

    result = run_query(
        _context(repository, frontier),
        session_id=None,
        query="What was Angie Jones's role as of 2021-01-01?",
    )

    assert repository.as_of_calls == 1
    assert repository.include_historical is True
    assert result.answerability_state == AnswerabilityState.ANSWERABLE.value
    assert "object_value: Engineer" in frontier.calls[0]


def test_exact_query_skips_model_planner_to_save_tokens() -> None:
    core = CountingCore()

    plan = AdaptivePlanner().plan("Who is Atul Singh's manager?", core=core)

    assert core.calls == 0
    assert plan.primary_route is RetrievalRoute.L1
    assert plan.predicate_hint == "HAS_MANAGER"
    assert "Atul Singh" in plan.entity_hints


def test_exact_current_claim_is_rendered_without_frontier_tokens() -> None:
    frontier = RecordingFrontier()

    result = run_query(
        _context(TypedExactRepository(), frontier),
        session_id=None,
        query="Where does Atul Singh work?",
    )

    assert result.answer == "Atul Singh works at 63moons."
    assert result.retrieval_metadata.answer_mode == "deterministic-canonical"
    assert result.retrieval_metadata.total_context_tokens == 0
    assert frontier.calls == []


def test_assistant_manager_is_not_collapsed_into_manager() -> None:
    plan = AdaptivePlanner().plan("Who is Atul Singh's assistant manager?")

    assert plan.predicate_hint == "HAS_ASSISTANT_MANAGER"


def test_school_query_routes_to_exact_canonical_claims() -> None:
    plan = AdaptivePlanner().plan("What school did Atul Singh attend?")

    assert plan.primary_route is RetrievalRoute.L1
    assert plan.predicate_hint == "ATTENDS_SCHOOL"


def test_version_query_uses_verified_postgres_neighbor_context() -> None:
    frontier = RecordingFrontier()
    context = _context(TypedNeighborRepository(), frontier)
    context.core = None

    result = run_query(
        context,
        session_id=None,
        query="What changed in v0.1.0?",
    )

    assert result.answerability_state == AnswerabilityState.ANSWERABLE.value
    assert result.retrieval_metadata.routes_attempted == ["L2"]
    assert "validation fixes and improved checks" in frontier.calls[0]


def test_worked_with_is_relationship_discovery_not_employer_lookup() -> None:
    plan = AdaptivePlanner().plan("Who worked with Atul Singh on Project Orion?")

    assert plan.primary_route is RetrievalRoute.L2
    assert plan.predicate_hint is None


def test_unrelated_semantic_body_is_not_answerable() -> None:
    plan = RetrievalPlan(
        query="Tell me about Muspar",
        intent="semantic",
        primary_route="L2",
        routes=["L2"],
    )
    unrelated = VerifiedEvidence.model_validate(
        RetrievalCandidate.from_row(
            _canonical_claim(
                predicate=None,
                claim_id=None,
                text="Atul Singh works on Project Orion.",
            ),
            canonical=True,
        ).model_dump()
    )

    assessment = EvidenceGate().assess(
        plan,
        [unrelated],
        attempted_routes=[RetrievalRoute.L2],
    )

    assert assessment.state is AnswerabilityState.INSUFFICIENT_EVIDENCE
    assert assessment.verified_evidence == []


def test_keyword_inside_a_longer_word_sets_no_expected_label() -> None:
    # "work" inside "workshop" is not a question about an employer.
    plan = AdaptivePlanner().plan("When did Melanie go to the pottery workshop?")

    assert plan.predicate_hint is None


def _expected_label_plan(query: str) -> RetrievalPlan:
    return RetrievalPlan(
        query=query,
        intent="current_fact",
        primary_route="L1",
        routes=["L1", "L2", "L4"],
        predicate_hint="PREFERS",
        requires_conflict_check=True,
    )


def _verified(**overrides: Any) -> VerifiedEvidence:
    return VerifiedEvidence.model_validate(
        RetrievalCandidate.from_row(_canonical_claim(**overrides), canonical=True).model_dump()
    )


_ALL_ROUTES = [RetrievalRoute.L1, RetrievalRoute.L2, RetrievalRoute.L4]


def test_missing_expected_label_still_escalates_while_routes_remain() -> None:
    found = _verified(predicate="HAS_FAVORITE_COLOR", object_value="chartreuse",
                      text="Zorblax's favourite colour is chartreuse.")

    assessment = EvidenceGate().assess(
        _expected_label_plan("What is Zorblax's favourite colour?"),
        [found],
        attempted_routes=[RetrievalRoute.L1],
    )

    assert assessment.state is AnswerabilityState.INSUFFICIENT_EVIDENCE
    assert assessment.can_escalate


def test_found_memory_is_kept_when_no_fact_has_the_expected_label() -> None:
    # After the last route, a memory that matches the question must reach the
    # answer model instead of being discarded only because its label differs.
    found = _verified(predicate="HAS_FAVORITE_COLOR", object_value="chartreuse",
                      text="Zorblax's favourite colour is chartreuse.")

    assessment = EvidenceGate().assess(
        _expected_label_plan("What is Zorblax's favourite colour?"),
        [found],
        attempted_routes=_ALL_ROUTES,
    )

    assert assessment.state is AnswerabilityState.ANSWERABLE
    assert [row.object_value for row in assessment.verified_evidence] == ["chartreuse"]


def test_fallback_memories_are_not_checked_for_label_conflicts() -> None:
    # With no fact for the asked label, two values of another label are just
    # found memories (Melanie plays two instruments), not a contradiction.
    rows = [
        _verified(claim_id="c1", subject_id="entity-melanie", predicate="PLAYS",
                  object_value="clarinet", text="Melanie plays the clarinet."),
        _verified(claim_id="c2", subject_id="entity-melanie", predicate="PLAYS",
                  object_value="violin", text="Melanie plays the violin."),
    ]

    assessment = EvidenceGate().assess(
        _expected_label_plan("Which instruments are Melanie's favourite to play?"),
        rows,
        attempted_routes=_ALL_ROUTES,
    )

    assert assessment.state is AnswerabilityState.ANSWERABLE
    assert len(assessment.verified_evidence) == 2


def test_exact_label_facts_still_win_over_other_memories() -> None:
    exact = _verified(claim_id="c1", predicate="PREFERS", object_value="chartreuse",
                      text="Zorblax prefers chartreuse.")
    other = _verified(claim_id="c2", predicate="HAS_FAVORITE_FOOD", object_value="soup",
                      text="Zorblax's favourite food is soup.")

    assessment = EvidenceGate().assess(
        _expected_label_plan("What is Zorblax's favourite colour?"),
        [exact, other],
        attempted_routes=_ALL_ROUTES,
    )

    assert assessment.state is AnswerabilityState.ANSWERABLE
    assert [row.object_value for row in assessment.verified_evidence] == ["chartreuse"]


def test_conflicting_active_values_are_not_answerable() -> None:
    plan = RetrievalPlan(
        query="What is the current role?",
        intent="current_fact",
        primary_route="L1",
        routes=["L1"],
        predicate_hint="has_role",
        requires_conflict_check=True,
    )
    rows = [
        RetrievalCandidate.from_row(
            _canonical_claim(claim_id="c1", object_value="Director"), canonical=True
        ),
        RetrievalCandidate.from_row(
            _canonical_claim(claim_id="c2", object_value="Engineer"), canonical=True
        ),
    ]
    evidence = CanonicalVerifier().verify(rows, plan=plan, tenant_id="_default")
    assessment = EvidenceGate().assess(plan, evidence, attempted_routes=[RetrievalRoute.L1])
    assert assessment.state is AnswerabilityState.CONFLICTING_EVIDENCE
    assert not assessment.can_escalate
