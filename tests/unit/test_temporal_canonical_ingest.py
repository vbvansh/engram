"""Acceptance checks for the bounded canonical conversational ingest path."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from engram.ingest.worker import _call_extract, _spoken_on, _turn_pair
from engram.predicate_registry import normalize_extracted_triplets
from engram.storage.memory_repository import MemoryRepository
from engram.temporal import activities
from engram.temporal.activities import (
    canonical_extract_activity,
    canonical_gate_activity,
    commit_canonical_activity,
)
from engram.temporal.workflows import CanonicalIngestWorkflow
from tests.postgres_support import PostgresTestStore


def _repository() -> tuple[MemoryRepository, PostgresTestStore]:
    if not os.environ.get("ENGRAM_TEST_DATABASE_URL"):
        pytest.skip("set ENGRAM_TEST_DATABASE_URL to run PostgreSQL canonical tests")
    store = PostgresTestStore()
    repository = MemoryRepository(store)
    repository.ensure_schema()
    return repository, store


def _event(store: PostgresTestStore, label: str) -> str:
    event_id, created = store.record_event(
        pair_id=f"canonical-conversation-{label}-{uuid.uuid4()}",
        session_id=f"session-{label}",
        source="test",
        event_type="INGEST",
        payload={
            "turn_pair": {
                "user": {"content": f"Tell me about {label}."},
                "assistant": {"content": f"{label} is a useful canonical test case."},
            }
        },
        tenant_id="tenant-canonical",
    )
    assert created
    return event_id


def test_angie_reuses_exact_entity_and_appends_scalar_role_history():
    repository, store = _repository()
    first_event = _event(store, "angie-first")
    first = repository.commit_conversational_event(
        event_id=first_event,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction={
            "resolved_text": "Angie joined as an engineer.",
            "l0_abstract": "Angie joined as an engineer.",
            "triplets": [
                {
                    "subject": "Angie Jones",
                    "relation": "role",
                    "object": "Engineer",
                    "valid_from": "2025-01-01",
                }
            ],
        },
    )
    entity_id = first["entity_ids"][0]

    second_event = _event(store, "angie-second")
    second = repository.commit_conversational_event(
        event_id=second_event,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction={
            "resolved_text": "Angie became a director.",
            "l0_abstract": "Angie became a director.",
            "triplets": [
                {
                    "subject_id": entity_id,
                    "relation": "became",
                    "object": "Director",
                    "valid_from": "2026-01-01",
                }
            ],
        },
    )

    assert second["entity_ids"] == [entity_id]
    node_id = uuid.UUID(entity_id)
    claims = repository.get_claim_history(node_id, tenant_id="tenant-canonical")
    assert [claim.object_value for claim in claims if claim.predicate == "HAS_ROLE"] == [
        "Director",
        "Engineer",
    ]
    assert all(claim.object_entity_id is None for claim in claims)
    assert all(claim.object_type == "ROLE" for claim in claims)
    versions = repository.get_versions(node_id, tenant_id="tenant-canonical")
    assert len(versions) == 2
    assert len(repository.get_evidence(node_id, tenant_id="tenant-canonical")) >= 2
    assert repository.get_recent_states(tenant_id="tenant-canonical")


def test_caleb_replay_does_not_duplicate_rows_or_projection_dispatches():
    repository, store = _repository()
    event_id = _event(store, "caleb-replay")
    extraction = {
        "resolved_text": "Caleb works at Acme.",
        "l0_abstract": "Caleb works at Acme.",
        "triplets": [
            {
                "subject": "Caleb",
                "relation": "works_for",
                "object": "Acme",
                "valid_from": "2026-01-01",
            }
        ],
    }
    first = repository.commit_conversational_event(
        event_id=event_id,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction=extraction,
    )
    replay = repository.commit_conversational_event(
        event_id=event_id,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction=extraction,
    )

    assert replay["replayed"] is True
    assert replay["episode_id"] == first["episode_id"]
    assert replay["entity_ids"] == first["entity_ids"]
    assert replay["claim_ids"] == first["claim_ids"]
    counts = (
        store.get_conn()
        .execute(
            "SELECT "
            "(SELECT count(*) FROM memory_nodes WHERE tenant_id = ?) AS nodes, "
            "(SELECT count(*) FROM memory_versions WHERE tenant_id = ?) AS versions, "
            "(SELECT count(*) FROM memory_claims WHERE tenant_id = ?) AS claims, "
            "(SELECT count(*) FROM memory_evidence WHERE tenant_id = ?) AS evidence, "
            "(SELECT count(*) FROM workflow_dispatches WHERE tenant_id = ? "
            "AND workflow_type = 'PROJECTION') AS projections",
            ("tenant-canonical",) * 5,
        )
        .fetchone()
    )
    assert counts["nodes"] == 3  # episode, Caleb, Acme
    assert counts["versions"] == 3  # one version per node for one event
    assert counts["claims"] == 1
    assert counts["evidence"] == 4  # two entity + one claim + one episode
    assert counts["projections"] >= 4
    event = store.get_event(event_id, tenant_id="tenant-canonical")
    assert event is not None and event["status"] == "COMPLETE"


def test_session_local_name_variants_reuse_identity_and_logical_claim():
    repository, store = _repository()
    atul_ids: set[str] = set()
    claim_ids: set[str] = set()
    variants = ("Atul", "atul_singh", "Atul Singh")
    for index, name in enumerate(variants):
        event_id = _event(store, f"atul-variant-{index}")
        # Keep all records in one conversation session, as bulk JSONL does.
        store.get_conn().execute(
            "UPDATE events SET session_id = ? WHERE tenant_id = ? AND event_id = ?",
            ("atul-story", "tenant-canonical", event_id),
        )
        result = repository.commit_conversational_event(
            event_id=event_id,
            tenant_id="tenant-canonical",
            gate={"store": True},
            extraction={
                "resolved_text": f"{name} works at 63moons.",
                "l0_abstract": "Atul works at 63moons.",
                "triplets": [
                    {
                        "subject": name,
                        "relation": "works_for",
                        "object": "63moons",
                    }
                ],
            },
        )
        states = [
            repository.get_memory(uuid.UUID(item), tenant_id="tenant-canonical")
            for item in result["entity_ids"]
        ]
        atul_ids.update(str(item.id) for item in states if item and "atul" in item.canonical_name.lower())
        claim_ids.update(result["claim_ids"])

    rows = store.get_conn().execute(
        "SELECT id, canonical_name FROM memory_nodes WHERE tenant_id = ? "
        "AND memory_type = 'ENTITY' ORDER BY canonical_name",
        ("tenant-canonical",),
    ).fetchall()
    assert [row["canonical_name"] for row in rows] == ["63moons", "Atul Singh"]
    assert len(atul_ids) == 1
    assert len(claim_ids) == 1
    claim_count = store.get_conn().execute(
        "SELECT count(*) AS count FROM memory_claims WHERE tenant_id = ?",
        ("tenant-canonical",),
    ).fetchone()
    evidence_count = store.get_conn().execute(
        "SELECT count(*) AS count FROM memory_evidence WHERE tenant_id = ? "
        "AND claim_id IS NOT NULL",
        ("tenant-canonical",),
    ).fetchone()
    assert claim_count["count"] == 1
    assert evidence_count["count"] == 3


def test_session_action_phrase_does_not_replace_person_display_name():
    repository, store = _repository()
    session_id = "atul-story"
    for index, (subject, predicate, object_value) in enumerate(
        (
            ("Atul Singh", "works_for", "63moons"),
            ("Atul", "works_for", "63moons"),
            ("atul_UAT_rerun", "reruns", "UAT_test"),
        )
    ):
        event_id = _event(store, f"atul-distinct-{index}")
        store.get_conn().execute(
            "UPDATE events SET session_id = ? WHERE tenant_id = ? AND event_id = ?",
            (session_id, "tenant-canonical", event_id),
        )
        repository.commit_conversational_event(
            event_id=event_id,
            tenant_id="tenant-canonical",
            gate={"store": True},
            extraction={
                "resolved_text": f"{subject} {predicate} {object_value}.",
                "l0_abstract": "Distinct identity regression case.",
                "triplets": [
                    {
                        "subject": subject,
                        "relation": predicate,
                        "object": object_value,
                    }
                ],
            },
        )

    rows = store.get_conn().execute(
        "SELECT canonical_name FROM memory_nodes WHERE tenant_id = ? "
        "AND memory_type = 'ENTITY' ORDER BY canonical_name",
        ("tenant-canonical",),
    ).fetchall()
    names = [row["canonical_name"] for row in rows]
    assert "Atul Singh" in names
    assert "atul UAT rerun" not in names


def test_neighbor_states_follow_exact_anchor_evidence_in_session_order():
    repository, store = _repository()
    session_id = "release-story"
    records = (
        (
            "v0.1.0 is a release tag.",
            [{"subject": "v0.1.0", "relation": "is_a", "object": "release_tag"}],
        ),
        ("Rahul asks what changed in v0.1.0.", []),
        ("The release contains validation fixes and improved checks.", []),
    )
    for index, (body, triplets) in enumerate(records):
        event_id = _event(store, f"release-neighbor-{index}")
        store.get_conn().execute(
            "UPDATE events SET session_id = ? WHERE tenant_id = ? AND event_id = ?",
            (session_id, "tenant-canonical", event_id),
        )
        repository.commit_conversational_event(
            event_id=event_id,
            tenant_id="tenant-canonical",
            gate={"store": True},
            extraction={
                "resolved_text": body,
                "l0_abstract": body,
                "triplets": triplets,
            },
        )

    aliases = repository.resolve_aliases("v0.1.0", tenant_id="tenant-canonical")
    assert len(aliases) == 1
    neighbors = repository.get_neighbor_states(
        aliases[0].entity_id,
        before=0,
        after=2,
        tenant_id="tenant-canonical",
    )
    bodies = [snapshot.current_version.body for snapshot in neighbors if snapshot.current_version]
    assert bodies == [body for body, _triplets in records]


def test_muspar_unknown_predicate_stays_scalar_and_never_manufactures_object_entity():
    repository, store = _repository()
    event_id = _event(store, "muspar")
    result = repository.commit_conversational_event(
        event_id=event_id,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction={
            "resolved_text": "Muspar prefers a compact editor.",
            "l0_abstract": "Muspar prefers a compact editor.",
            "triplets": [
                {
                    "subject": "Muspar",
                    "relation": "favorite_editor",
                    "object": "Compact Editor",
                }
            ],
        },
    )
    muspar_id = uuid.UUID(result["entity_ids"][0])
    claims = repository.get_claim_history(muspar_id, tenant_id="tenant-canonical")
    claim = next(item for item in claims if item.predicate == "FAVORITE_EDITOR")
    assert claim.object_type == "STRING"
    assert claim.object_entity_id is None
    entities = (
        store.get_conn()
        .execute(
            "SELECT count(*) AS c FROM memory_nodes WHERE tenant_id = ? AND memory_type = 'ENTITY'",
            ("tenant-canonical",),
        )
        .fetchone()
    )
    assert entities["c"] == 1


def test_canonical_workflow_accepts_only_opaque_routing_and_is_registered_by_name():
    import inspect

    assert list(inspect.signature(CanonicalIngestWorkflow.run).parameters) == [
        "self",
        "event_id",
        "tenant_id",
    ]
    assert CanonicalIngestWorkflow.__temporal_workflow_definition.name == "engram.canonical_ingest"
    assert canonical_gate_activity.__temporal_activity_definition.name == "engram.canonical_gate"
    assert (
        canonical_extract_activity.__temporal_activity_definition.name == "engram.canonical_extract"
    )
    assert (
        commit_canonical_activity.__temporal_activity_definition.name == "engram.commit_canonical"
    )


def test_extract_activity_persists_typed_artifact_before_commit(monkeypatch):
    repository, store = _repository()
    event_id = _event(store, "typed-boundary")
    repository.record_ingest_artifact(
        event_id=event_id,
        artifact_type="EXTRACTION",
        artifact_key="gate-v1",
        payload={"store": True},
        content=b'{"store": true}',
        media_type="application/json",
        tenant_id="tenant-canonical",
    )
    state = SimpleNamespace(
        cfg=object(),
        core=object(),
        control_plane=store,
        memory_repository=repository,
    )
    monkeypatch.setattr(activities, "get_state", lambda: state)
    monkeypatch.setattr(
        activities,
        "_call_extract",
        lambda *_args, **_kwargs: {
            "resolved_text": "Angie was an engineer and director.",
            "l0_abstract": "Angie's roles.",
            "triplets": [
                {
                    "subject": "Angie Jones",
                    "relation": "job title",
                    "object": "Engineer and Director",
                    "valid_from": "2026-09-01",
                }
            ],
        },
    )

    assert canonical_extract_activity(event_id, "tenant-canonical") == "READY"
    typed = repository.get_ingest_artifact(
        event_id,
        "EXTRACTION",
        "typed-v1",
        tenant_id="tenant-canonical",
    )
    assert typed is not None
    assert typed.payload["normalization_stage"] == "typed-v1"
    triplets = typed.payload["triplets"]
    assert isinstance(triplets, list)
    assert [item["object_value"] for item in triplets] == ["Engineer", "Director"]
    assert {item["object_type"] for item in triplets} == {"ROLE"}


def test_force_store_gate_is_deterministic_and_skips_model_call(monkeypatch):
    repository, store = _repository()
    event_id, created = store.record_event(
        pair_id=f"force-store-{uuid.uuid4()}",
        session_id="session-force-store",
        source="bulk_jsonl",
        event_type="INGEST",
        payload={
            "force_store": True,
            "turn_pair": {
                "user": {"content": "A fictional reference fact."},
                "assistant": {"content": "The reference fact is retained."},
            },
        },
        tenant_id="tenant-canonical",
    )
    assert created
    state = SimpleNamespace(
        cfg=object(),
        core=object(),
        control_plane=store,
        memory_repository=repository,
    )
    monkeypatch.setattr(activities, "get_state", lambda: state)
    monkeypatch.setattr(
        activities,
        "_call_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("force_store must bypass the model gate")
        ),
    )

    result = canonical_gate_activity(event_id, "tenant-canonical")

    assert result["store"] is True
    artifact = repository.get_ingest_artifact(
        event_id,
        "EXTRACTION",
        "gate-v1",
        tenant_id="tenant-canonical",
    )
    assert artifact is not None
    assert artifact.payload["policy"] == "force-store-v1"


def test_empty_profile_extraction_uses_reversible_recovery(monkeypatch):
    repository, store = _repository()
    event_id, created = store.record_event(
        pair_id=f"profile-recovery-{uuid.uuid4()}",
        session_id="session-profile-recovery",
        source="bulk_jsonl",
        event_type="INGEST",
        payload={
            "turn_pair": {
                "user": {
                    "content": "Story event 72: Atul creates a personal profile entry "
                    "for testing memory retrieval. This happens during chapter 2."
                },
                "assistant": {
                    "content": "The profile entry stores the real reference facts: "
                    "Atul Singh, 63moons, Rahul, Noel, and GSS. This is part of chapter 2."
                },
            },
        },
        tenant_id="tenant-canonical",
    )
    assert created
    repository.record_ingest_artifact(
        event_id=event_id,
        artifact_type="EXTRACTION",
        artifact_key="gate-v1",
        payload={"store": True},
        content=b'{"store": true}',
        media_type="application/json",
        tenant_id="tenant-canonical",
    )
    recovery_calls: list[dict[str, object]] = []

    class RecoveryCore:
        def complete(self, **kwargs):
            recovery_calls.append(kwargs)
            return SimpleNamespace(
                output={
                    "resolved_text": "Atul creates a memory test record.",
                    "l0_abstract": "A memory test record.",
                    "triplets": [
                        {
                            "subject": "Atul",
                            "relation": "creates",
                            "object": "memory test record",
                        }
                    ],
                }
            )

    state = SimpleNamespace(
        cfg=object(),
        core=RecoveryCore(),
        control_plane=store,
        memory_repository=repository,
    )
    calls: list[dict[str, str]] = []

    def fake_extract(_context, turn_pair, *, session_context, spoken_on=None):
        del session_context, spoken_on
        calls.append(turn_pair)
        if "personal profile entry" in "\n".join(turn_pair.values()).lower():
            raise activities.CoreModelError("could not parse JSON from model output: ''")
        raise AssertionError("recovery must use the compact core prompt")

    monkeypatch.setattr(activities, "get_state", lambda: state)
    monkeypatch.setattr(activities, "_call_extract", fake_extract)

    assert canonical_extract_activity(event_id, "tenant-canonical") == "READY"
    assert len(calls) == 1
    assert len(recovery_calls) == 1
    recovery_prompt = str(recovery_calls[0]["user_prompt"]).lower()
    assert "user: atul creates a memory test record." in recovery_prompt
    assert "chapter" not in recovery_prompt
    artifact = repository.get_ingest_artifact(
        event_id,
        "EXTRACTION",
        "typed-v1",
        tenant_id="tenant-canonical",
    )
    assert artifact is not None
    assert artifact.payload["recovery_policy"] == "neutral-profile-term-v1"
    assert artifact.payload["triplets"][0]["object_value"] == "personal profile entry"


def test_profile_retry_skips_known_empty_primary_prompt(monkeypatch):
    repository, store = _repository()
    event_id, created = store.record_event(
        pair_id=f"profile-retry-{uuid.uuid4()}",
        session_id="session-profile-retry",
        source="bulk_jsonl",
        event_type="INGEST",
        payload={
            "turn_pair": {
                "user": {"content": "Atul creates a personal profile entry."},
                "assistant": {"content": "The profile entry references Rahul."},
            },
        },
        tenant_id="tenant-canonical",
    )
    assert created
    repository.record_ingest_artifact(
        event_id=event_id,
        artifact_type="EXTRACTION",
        artifact_key="gate-v1",
        payload={"store": True},
        content=b'{"store": true}',
        media_type="application/json",
        tenant_id="tenant-canonical",
    )
    recovery_calls: list[dict[str, object]] = []

    class RecoveryCore:
        def complete(self, **kwargs):
            recovery_calls.append(kwargs)
            return SimpleNamespace(
                output={
                    "resolved_text": "Atul creates a memory test record.",
                    "l0_abstract": "A memory test record.",
                    "triplets": [],
                }
            )

    state = SimpleNamespace(
        cfg=object(),
        core=RecoveryCore(),
        control_plane=store,
        memory_repository=repository,
    )
    calls: list[dict[str, str]] = []

    def fake_extract(_context, turn_pair, *, session_context, spoken_on=None):
        del _context, turn_pair, session_context, spoken_on
        raise AssertionError("a retried profile event must skip the known-empty primary prompt")

    monkeypatch.setattr(activities, "get_state", lambda: state)
    monkeypatch.setattr(activities, "_call_extract", fake_extract)
    monkeypatch.setattr(activities, "_activity_attempt", lambda: 2)

    assert canonical_extract_activity(event_id, "tenant-canonical") == "READY"
    assert calls == []
    assert len(recovery_calls) == 1


def test_profile_repeat_reuses_exact_session_extraction_without_model(monkeypatch):
    repository, store = _repository()
    source_event, created = store.record_event(
        pair_id=f"profile-source-{uuid.uuid4()}",
        session_id="session-profile-repeat",
        source="bulk_jsonl",
        event_type="INGEST",
        payload={
            "turn_pair": {
                "user": {"content": "Story event 22: Atul creates a personal profile entry."},
                "assistant": {
                    "content": "The profile entry stores the real reference facts: "
                    "Atul Singh, 63moons, Rahul, Noel, and GSS."
                },
            },
        },
        tenant_id="tenant-canonical",
    )
    assert created
    repository.record_ingest_artifact(
        event_id=source_event,
        artifact_type="EXTRACTION",
        artifact_key="extract-v1",
        payload={
            "resolved_text": "Atul creates a personal profile entry.",
            "l0_abstract": "Atul's profile references known facts.",
            "triplets": [
                {
                    "subject": "Atul Singh",
                    "relation": "creates",
                    "object": "personal profile entry",
                }
            ],
        },
        content=b"{}",
        media_type="application/json",
        tenant_id="tenant-canonical",
    )
    event_id, created = store.record_event(
        pair_id=f"profile-repeat-{uuid.uuid4()}",
        session_id="session-profile-repeat",
        source="bulk_jsonl",
        event_type="INGEST",
        payload={
            "turn_pair": {
                "user": {
                    "content": "Story event 72: Atul creates a personal profile entry. "
                    "This happens during chapter 2."
                },
                "assistant": {
                    "content": "The profile entry stores the real reference facts: "
                    "Atul Singh, 63moons, Rahul, Noel, and GSS. This is part of chapter 2."
                },
            },
        },
        tenant_id="tenant-canonical",
    )
    assert created
    repository.record_ingest_artifact(
        event_id=event_id,
        artifact_type="EXTRACTION",
        artifact_key="gate-v1",
        payload={"store": True},
        content=b'{"store": true}',
        media_type="application/json",
        tenant_id="tenant-canonical",
    )
    state = SimpleNamespace(
        cfg=object(),
        core=object(),
        control_plane=store,
        memory_repository=repository,
    )
    monkeypatch.setattr(activities, "get_state", lambda: state)
    monkeypatch.setattr(
        activities,
        "_call_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact session repeats must not call the model")
        ),
    )

    assert canonical_extract_activity(event_id, "tenant-canonical") == "READY"
    artifact = repository.get_ingest_artifact(
        event_id,
        "EXTRACTION",
        "typed-v1",
        tenant_id="tenant-canonical",
    )
    assert artifact is not None
    assert artifact.payload["recovery_policy"] == "same-session-profile-signature-v1"
    assert artifact.payload["reused_from_event_id"] == source_event
    assert "chapter 2" in artifact.payload["resolved_text"]


class _PromptRecordingCore:
    def __init__(self, output):
        self.output = output
        self.prompts: list[str] = []

    def complete(self, **kwargs):
        self.prompts.append(kwargs["system_prompt"])
        return SimpleNamespace(output=self.output, raw_text="")


def _support_group_payload(timestamp: str | None) -> dict:
    user = {"content": "Caroline: I went to an LGBTQ support group yesterday."}
    if timestamp is not None:
        user["timestamp"] = timestamp
    return {"turn_pair": {"user": user, "assistant": {"content": "Melanie: That's great!"}}}


def test_extractor_is_told_the_day_the_turn_was_said():
    core = _PromptRecordingCore({"resolved_text": "x", "l0_abstract": "x", "triplets": []})
    payload = _support_group_payload("2023-05-08T13:56:00")

    _call_extract(
        SimpleNamespace(core=core),
        _turn_pair(payload),
        session_context=None,
        spoken_on=_spoken_on(payload),
    )

    assert "This turn was said on Monday, 8 May 2023." in core.prompts[0]
    assert "Turn pair (said on Monday, 8 May 2023):" in core.prompts[0]


def test_without_a_usable_timestamp_relative_times_stay_as_written():
    core = _PromptRecordingCore({"resolved_text": "x", "l0_abstract": "x", "triplets": []})
    payload = _support_group_payload("last Tuesday")

    assert _spoken_on(payload) is None
    assert _spoken_on(_support_group_payload(None)) is None
    _call_extract(SimpleNamespace(core=core), _turn_pair(payload), session_context=None)

    assert "This turn was said on an unknown date." in core.prompts[0]


def test_unusable_model_dates_are_dropped_instead_of_failing_the_message():
    core = _PromptRecordingCore(
        {
            "resolved_text": "x",
            "l0_abstract": "x",
            "triplets": [
                {"subject": "Caroline", "relation": "visited", "object": "support group",
                 "valid_from": "2023-05-07"},
                {"subject": "Melanie", "relation": "plans", "object": "camping",
                 "valid_from": "June 2023"},
                {"subject": "Melanie", "relation": "ran", "object": "charity race",
                 "valid_from": "2023-05-21", "valid_until": "2023-05-01"},
            ],
        }
    )

    out = _call_extract(
        SimpleNamespace(core=core), {"user": "u", "assistant": "a"}, session_context=None
    )

    first, vague, backwards = out["triplets"]
    assert first["valid_from"] == "2023-05-07"
    assert "valid_from" not in vague
    assert backwards["valid_from"] == "2023-05-21" and "valid_until" not in backwards
    # The typed-claim stage, which rejects unparseable dates, accepts what is left.
    assert len(normalize_extracted_triplets(out["triplets"])) == 3


def _commit_support_group(repository, store, timestamp: str | None) -> list:
    event_id, created = store.record_event(
        pair_id=f"canonical-said-on-{uuid.uuid4()}",
        session_id="session-said-on",
        source="test",
        event_type="INGEST",
        payload=_support_group_payload(timestamp),
        tenant_id="tenant-canonical",
    )
    assert created
    result = repository.commit_conversational_event(
        event_id=event_id,
        tenant_id="tenant-canonical",
        gate={"store": True},
        extraction={
            "resolved_text": "Caroline went to an LGBTQ support group yesterday (7 May 2023).",
            "l0_abstract": "Caroline went to an LGBTQ support group on 7 May 2023.",
            "triplets": [
                {"subject": "Caroline", "relation": "visited", "object": "LGBTQ support group",
                 "valid_from": "2023-05-07"}
            ],
        },
    )
    node_id = uuid.UUID(result["entity_ids"][0])
    claims = repository.get_claim_history(node_id, tenant_id="tenant-canonical")
    versions = repository.get_versions(node_id, tenant_id="tenant-canonical")
    return [*claims, *versions]


def test_facts_are_stamped_with_the_day_they_were_said():
    repository, store = _repository()

    stamped = _commit_support_group(repository, store, "2023-05-08T13:56:00")

    said = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    assert stamped and all(item.asserted_at == said for item in stamped)
    assert stamped[0].valid_from == datetime(2023, 5, 7, tzinfo=timezone.utc)


def test_without_a_turn_timestamp_facts_keep_the_storage_time():
    repository, store = _repository()

    stamped = _commit_support_group(repository, store, None)

    now = datetime.now(timezone.utc)
    assert stamped and all(abs(item.asserted_at - now) < timedelta(minutes=10) for item in stamped)
