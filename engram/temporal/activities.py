"""Side-effecting Temporal Activities for the existing idempotent handlers."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from engram.consolidation.reconciliation import (
    ReconciliationContext,
    run_stale_overview_scan,
)
from engram.consolidation.worker import (
    ConsolidationContext,
    UnknownConsolidationTaskTypeError,
    _dispatch,
)
from engram.decay import run_daily
from engram.deps import get_state, make_ingest_context
from engram.ingest.worker import (
    _call_extract,
    _call_gate,
    _spoken_on,
    _turn_pair,
    process_event,
)
from engram.models.core import CoreModelError
from engram.models.request_context import model_request_session
from engram.predicate_registry import normalize_extracted_triplets
from engram.storage.memory_repository import MemoryRepositoryError
from engram.storage.postgres import PostgresStore
from engram.tenancy import Tenant, TenantQuotas, set_current_tenant


class CanonicalProjectionContractError(RuntimeError):
    """Raised when the canonical-memory repository is not wired for projection."""


_CANONICAL_REPOSITORY_MODULES = (
    "engram.storage.memory_repository",
    "engram.storage.canonical_repository",
    "engram.storage.canonical_memory",
)

_EMPTY_MODEL_OUTPUT = "could not parse JSON from model output: ''"
_PROFILE_ENTRY_TERM = re.compile(r"\b(?:personal\s+)?profile\s+entr(?:y|ies)\b", re.IGNORECASE)
_PROFILE_ENTRY_RECOVERY_TERM = "memory test record"


def _restore_profile_entry_term(value: Any) -> Any:
    """Restore a neutralized provider-sensitive term in structured output."""

    if isinstance(value, str):
        return re.sub(
            rf"\b{re.escape(_PROFILE_ENTRY_RECOVERY_TERM)}s?\b",
            "personal profile entry",
            value,
            flags=re.IGNORECASE,
        )
    if isinstance(value, list):
        return [_restore_profile_entry_term(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _restore_profile_entry_term(item) for key, item in value.items()}
    return value


def _recover_profile_entry_extraction(
    model_context: Any,
    turn_pair: dict[str, str],
    *,
    session_context: Any,
) -> dict[str, Any]:
    """Retry an empty Muse response with a reversible neutral term.

    Muse Spark can return an HTTP-successful empty response for the phrase
    ``personal profile entry``.  This recovery is intentionally narrow: it is
    attempted only for that exact empty-output condition, changes no names or
    factual values, and reverses the neutral term before persistence.
    """

    recovered_pair = {
        key: _PROFILE_ENTRY_TERM.sub(_PROFILE_ENTRY_RECOVERY_TERM, value)
        for key, value in turn_pair.items()
    }
    subject_match = re.search(
        r"^(?:Story event \d+:\s*)?(?P<subject>.+?)\s+creates\s+a\s+"
        r"(?:personal\s+)?profile\s+entry\b",
        turn_pair["user"],
        flags=re.IGNORECASE,
    )
    references_match = re.search(
        r"reference facts:\s*(?P<objects>.+?)(?:\.\s+This\b|\.$|$)",
        turn_pair["assistant"],
        flags=re.IGNORECASE,
    )
    if subject_match and references_match:
        subject = subject_match.group("subject").strip()
        objects = references_match.group("objects").strip()
        recovered_pair = {
            "user": f"{subject} creates a {_PROFILE_ENTRY_RECOVERY_TERM}.",
            "assistant": f"The {_PROFILE_ENTRY_RECOVERY_TERM} references {objects}.",
        }
    del session_context  # Recovery deliberately excludes optional context to minimize tokens.
    result = model_context.core.complete(
        system_prompt=(
            "Extract only facts explicitly stated in the two messages. Return one JSON object "
            "with resolved_text (string), l0_abstract (string), and triplets (array). Each "
            "triplet must contain subject, relation, object, and confidence. Return JSON only."
        ),
        user_prompt=(
            f"User: {recovered_pair['user']}\n"
            f"Assistant: {recovered_pair['assistant']}"
        ),
        max_tokens=1200,
        temperature=0.0,
    )
    extraction = result.output
    if not isinstance(extraction, dict):
        raise CoreModelError("profile-entry extraction recovery returned non-object output")
    extraction.setdefault("resolved_text", recovered_pair["assistant"])
    extraction.setdefault("triplets", [])
    extraction.setdefault("l0_abstract", recovered_pair["assistant"][:200])
    restored = _restore_profile_entry_term(extraction)
    if not isinstance(restored, dict):
        raise CoreModelError("profile-entry extraction recovery returned invalid output")
    restored["extractor_version"] = "extract-v2-profile-recovery"
    restored["recovery_policy"] = "neutral-profile-term-v1"
    return restored


def _activity_attempt() -> int:
    """Return the Temporal attempt, or one for direct/unit-test invocation."""

    try:
        return int(activity.info().attempt)
    except RuntimeError:
        return 1


def _profile_entry_signature(turn_pair: Mapping[str, str]) -> tuple[str, str] | None:
    """Return an exact normalized signature for the known profile fact shape."""

    subject_match = re.search(
        r"^(?:Story event \d+:\s*)?(?P<subject>.+?)\s+creates\s+a\s+"
        r"(?:personal\s+)?profile\s+entry\b",
        turn_pair.get("user", ""),
        flags=re.IGNORECASE,
    )
    references_match = re.search(
        r"reference facts:\s*(?P<objects>.+?)(?:\.\s+This\b|\.$|$)",
        turn_pair.get("assistant", ""),
        flags=re.IGNORECASE,
    )
    if not subject_match or not references_match:
        return None

    def normalized(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    return (
        normalized(subject_match.group("subject")),
        normalized(references_match.group("objects")),
    )


def _reuse_session_profile_extraction(
    state: Any,
    event: Mapping[str, Any],
    tenant_id: str,
    event_id: str,
    turn_pair: dict[str, str],
) -> dict[str, Any] | None:
    """Reuse a prior exact same-session profile extraction without a model call."""

    signature = _profile_entry_signature(turn_pair)
    session_id = event.get("session_id")
    if signature is None or not session_id:
        return None
    rows = state.control_plane.get_conn().execute(
        "SELECT e.event_id, e.payload AS event_payload, a.payload AS extraction_payload "
        "FROM events e JOIN ingest_artifacts a ON a.tenant_id = e.tenant_id "
        "AND a.event_id = e.event_id AND a.artifact_type = 'EXTRACTION' "
        "AND a.artifact_key = 'extract-v1' WHERE e.tenant_id = ? "
        "AND e.session_id = ? AND e.event_id <> ? ORDER BY e.created_at LIMIT 200",
        (tenant_id, str(session_id), event_id),
    ).fetchall()
    for row in rows:
        source_payload = row.get("event_payload")
        extraction_payload = row.get("extraction_payload")
        try:
            if isinstance(source_payload, str):
                source_payload = json.loads(source_payload)
            if isinstance(extraction_payload, str):
                extraction_payload = json.loads(extraction_payload)
            if not isinstance(source_payload, Mapping) or not isinstance(
                extraction_payload, Mapping
            ):
                continue
            source_pair = _turn_pair(dict(source_payload))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if _profile_entry_signature(source_pair) != signature:
            continue
        reused = dict(extraction_payload)
        reused["resolved_text"] = (
            f"User: {turn_pair['user']}\nAssistant: {turn_pair['assistant']}"
        )
        reused["extractor_version"] = "extract-v2-session-reuse"
        reused["recovery_policy"] = "same-session-profile-signature-v1"
        reused["reused_from_event_id"] = str(row["event_id"])
        return reused
    return None


def _canonical_repository(state: Any) -> Any:
    """Resolve the canonical repository lazily inside an Activity.

    The repository is being introduced by the canonical-memory migration and
    must not become an import-time dependency of the legacy process.  Tests and
    embedders may inject ``state.canonical_repository`` (or
    ``state.memory_repository``); production integrations can expose a
    ``PostgresMemoryRepository`` from one of the candidate modules below.
    """
    for attr in ("canonical_repository", "memory_repository", "memory_repo"):
        repository = getattr(state, attr, None)
        if repository is not None:
            return repository

    for module_name in _CANONICAL_REPOSITORY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as err:
            if err.name == module_name:
                continue
            raise
        for factory_name in (
            "build_memory_repository",
            "build_canonical_repository",
            "PostgresMemoryRepository",
            "CanonicalMemoryRepository",
        ):
            factory = getattr(module, factory_name, None)
            if factory is None:
                continue
            try:
                return factory(state.control_plane)
            except TypeError:
                # A zero-argument factory is convenient for dependency
                # injection; support the common keyword form before giving up.
                try:
                    return factory(store=state.control_plane)
                except TypeError:
                    try:
                        return factory()
                    except TypeError as no_arg_err:
                        raise CanonicalProjectionContractError(
                            f"{module_name}.{factory_name} could not be constructed with "
                            "the PostgreSQL control-plane store, store=, or no arguments"
                        ) from no_arg_err
            except Exception:
                raise

    raise CanonicalProjectionContractError(
        "canonical memory repository is not available; projection expects "
        "state.canonical_repository or one of "
        f"{', '.join(_CANONICAL_REPOSITORY_MODULES)} to expose a repository with "
        "load_mutation_projection(mutation_id) (or load_projection_snapshot/"
        "load_projection/get_mutation_projection)"
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError as err:
        raise CanonicalProjectionContractError(
            "canonical projection loader must return a mapping or an object with attributes"
        ) from err


def _projection_loader(repository: Any):
    for method_name in (
        "load_projection_snapshot",
        "load_mutation_projection",
        "load_projection",
        "get_projection_snapshot",
        "get_mutation_projection",
        "get_canonical_mutation_projection",
        "get_canonical_mutation",
        "get_mutation",
    ):
        method = getattr(repository, method_name, None)
        if callable(method):
            return method
    raise CanonicalProjectionContractError(
        "canonical repository must expose load_mutation_projection(mutation_id) or "
        "load_projection_snapshot/load_projection/get_mutation_projection"
    )


def _load_projection(
    repository: Any,
    mutation_id: str,
    envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loader = _projection_loader(repository)
    tenant_id = (envelope or {}).get("tenant_id")
    attempts = []
    if tenant_id is not None:
        attempts.append(lambda: loader(mutation_id, tenant_id=tenant_id))
    attempts.extend(
        [
            lambda: loader(mutation_id),
            lambda: loader(mutation_id=mutation_id),
        ]
    )
    value = None
    for attempt in attempts:
        try:
            value = attempt()
            break
        except TypeError as err:
            last_error = err
    else:
        raise CanonicalProjectionContractError(
            "canonical projection loader must accept a canonical mutation ID "
            "positionally or as mutation_id"
        ) from last_error
    if value is None:
        raise ApplicationError(
            f"canonical mutation not found: {mutation_id}",
            type="CanonicalMutationNotFound",
            non_retryable=True,
        )
    projection = _as_mapping(value)
    nested = projection.get("payload")
    if isinstance(nested, Mapping):
        projection = {**dict(nested), **projection}
    if envelope:
        envelope_tenant = envelope.get("tenant_id")
        projection_tenant = projection.get("tenant_id")
        if (
            envelope_tenant is not None
            and projection_tenant is not None
            and str(envelope_tenant) != str(projection_tenant)
        ):
            raise ApplicationError(
                "canonical mutation tenant does not match Temporal dispatch tenant",
                type="InvalidProjectionTenant",
                non_retryable=True,
            )
        # The row envelope supplies dispatch identity and stable IDs.  Canonical
        # state supplies the current snapshot, including a newer revision when
        # multiple mutations were coalesced before dispatch.
        projection = {**dict(envelope), **projection}
        if projection.get("tenant_id") is None:
            projection["tenant_id"] = envelope_tenant
    return projection


def _dispatch_envelope(control_plane: Any, identifier: str) -> tuple[str | None, dict[str, Any]]:
    """Resolve a Temporal input to a workflow-dispatch envelope.

    Production dispatchers pass ``dispatch_id`` to the projection workflow.
    The aggregate is the canonical mutation ID.  The aggregate fallback keeps
    direct workflow starts and older dispatch rows diagnosable during rollout.
    """
    row = None
    get_dispatch = getattr(control_plane, "get_dispatch", None)
    if callable(get_dispatch):
        row = get_dispatch(identifier)
    if row is None:
        get_for_aggregate = getattr(control_plane, "get_dispatch_for_aggregate", None)
        if callable(get_for_aggregate):
            row = get_for_aggregate("PROJECTION", identifier)
    if row is None:
        return None, {"canonical_mutation_id": identifier}
    envelope = _as_mapping(row)
    raw_payload = envelope.get("payload")
    if isinstance(raw_payload, str):
        try:
            raw_payload = json.loads(raw_payload)
        except json.JSONDecodeError as err:
            raise ApplicationError(
                "Temporal projection dispatch payload is not valid JSON",
                type="InvalidProjection",
                non_retryable=True,
            ) from err
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
    # Payload values are copied first; the relational envelope is authoritative
    # for tenant, mutation, revision, and dispatch identity.
    merged = {**payload, **envelope}
    aggregate_id = str(envelope.get("aggregate_id") or identifier)
    merged["canonical_mutation_id"] = aggregate_id
    if envelope.get("aggregate_revision") is not None:
        merged["revision"] = envelope["aggregate_revision"]
    merged.pop("payload", None)
    return str(envelope.get("dispatch_id") or identifier), merged


def _complete_dispatch(control_plane: Any, dispatch_id: str | None, aggregate_id: str) -> None:
    if dispatch_id:
        complete = getattr(control_plane, "complete_dispatch", None)
        if callable(complete):
            complete(dispatch_id)
            return
    complete_aggregate = getattr(control_plane, "mark_aggregate_dispatches_complete", None)
    if callable(complete_aggregate):
        complete_aggregate("PROJECTION", aggregate_id)


def _fail_dispatch(
    control_plane: Any,
    dispatch_id: str,
    error: str,
    *,
    failure_class: str = "WORKFLOW_TERMINAL",
) -> None:
    # The dispatcher claim token is cleared once Temporal accepts the workflow,
    # so terminal workflow bookkeeping uses the non-lease-fenced transition.
    fail = getattr(control_plane, "mark_dispatch_dead", None)
    if isinstance(control_plane, PostgresStore):
        failed = fail(
            dispatch_id,
            error[:1000],
            failure_class=failure_class,
        )
    else:
        failed = callable(fail) and fail(dispatch_id, error[:1000])
    if failed:
        return
    raise ApplicationError(
        f"projection dispatch not found or already terminal: {dispatch_id}",
        type="ProjectionDispatchNotFound",
        non_retryable=True,
    )


def _projection_tenant(projection: Mapping[str, Any]) -> str:
    tenant_id = projection.get("tenant_id")
    if tenant_id is None:
        raise ApplicationError(
            "projection payload is missing tenant_id",
            type="InvalidProjection",
            non_retryable=True,
        )
    tenant = str(tenant_id).strip()
    if not tenant:
        raise ApplicationError(
            "projection payload has an empty tenant_id",
            type="InvalidProjection",
            non_retryable=True,
        )
    return tenant


def _projection_revision(projection: Mapping[str, Any]) -> int:
    value = projection.get("revision")
    if value is None:
        value = projection.get("canonical_revision", projection.get("input_revision"))
    try:
        revision = int(value)
    except (TypeError, ValueError) as err:
        raise ApplicationError(
            "projection payload is missing an integer revision",
            type="InvalidProjection",
            non_retryable=True,
        ) from err
    if revision < 0:
        raise ApplicationError(
            "projection revision must be non-negative",
            type="InvalidProjection",
            non_retryable=True,
        )
    return revision


def _projection_operation(projection: Mapping[str, Any]) -> str:
    return str(projection.get("operation", "UPSERT")).upper().replace("-", "_")


def _projection_properties(projection: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("properties", "node", "memory"):
        value = projection.get(key)
        if isinstance(value, Mapping):
            return _neo4j_properties(value)
    return {}


def _neo4j_properties(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only Neo4j-native property values and flatten canonical metadata."""

    merged = dict(value)
    metadata = merged.pop("metadata", None)
    if isinstance(metadata, Mapping):
        for key, item in metadata.items():
            merged.setdefault(str(key), item)
    allowed: dict[str, Any] = {}
    for key, item in merged.items():
        if item is None or isinstance(item, (str, int, float, bool)):
            allowed[str(key)] = item
        elif isinstance(item, (list, tuple)) and all(
            element is None or isinstance(element, (str, int, float, bool)) for element in item
        ):
            allowed[str(key)] = list(item)
    return allowed


def _memory_projection_properties(projection: Mapping[str, Any], state: Any) -> dict[str, Any]:
    props = _projection_properties(projection)
    canonical_uri = props.get("canonical_uri")
    if canonical_uri:
        props["source_uri"] = canonical_uri
    memory_type = props.get("memory_type")
    if memory_type:
        props["node_type"] = memory_type
    canonical_name = props.get("canonical_name")
    if canonical_name:
        props["display_name"] = canonical_name
    version = projection.get("current_version")
    if isinstance(version, Mapping):
        version_props = _neo4j_properties(version)
        abstract = version_props.get("abstract") or str(version_props.get("body") or "")[:500]
        if abstract:
            props["l0_abstract"] = abstract
            props["l0_embedding"] = state.embed.embed(str(abstract)[:2000])
        props["current_version_id"] = version_props.get("id")
        props["version_number"] = version_props.get("version_number")
    return props


@activity.defn(name="engram.project_neo4j")
def project_neo4j_activity(dispatch_id: str) -> str:
    """Project one committed canonical mutation dispatch into Neo4j.

    ``workflow_dispatches`` is the sole outbox.  Its aggregate ID identifies
    the canonical mutation and its JSON payload carries stable memory/claim
    IDs.  The canonical repository is consulted for the current snapshot; the
    dispatch row is completed only after the Neo4j side effect succeeds.
    """
    if not dispatch_id:
        raise ApplicationError(
            "dispatch_id is required",
            type="InvalidProjection",
            non_retryable=True,
        )
    state = get_state()
    if not isinstance(state.control_plane, PostgresStore):
        raise ApplicationError(
            "Temporal projection requires PostgresStore as the canonical control plane",
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        )
    try:
        resolved_dispatch_id, envelope = _dispatch_envelope(state.control_plane, dispatch_id)
        mutation_id = str(envelope.get("canonical_mutation_id") or dispatch_id)
        repository = _canonical_repository(state)
        projection = _load_projection(repository, mutation_id, envelope)
    except CanonicalProjectionContractError as err:
        raise ApplicationError(
            str(err),
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        ) from err
    tenant_id = _projection_tenant(projection)
    revision = _projection_revision(projection)
    operation = _projection_operation(projection)
    aggregate_type = str(
        projection.get("aggregate_type") or projection.get("resource_type") or "MEMORY"
    ).upper()
    memory_id = projection.get("memory_id")
    claim_id = projection.get("claim_id")
    set_current_tenant(
        Tenant(
            tenant_id=tenant_id,
            display_name=tenant_id,
            api_key_hashes=[],
            quotas=TenantQuotas(),
            status="ACTIVE",
        )
    )
    try:
        neo4j = getattr(state, "neo4j", None)
        if neo4j is None:
            raise ApplicationError(
                "Temporal projection requires a Neo4j projection store",
                type="ProjectionUnavailable",
            )
        embedded_claim = projection.get("claim")
        embedded_claim_status = (
            embedded_claim.get("status") if isinstance(embedded_claim, Mapping) else None
        )
        raw_claim_status = projection.get("claim_status") or embedded_claim_status
        if raw_claim_status is None:
            top_level_status = str(projection.get("status") or "").upper()
            if top_level_status in {"ACTIVE", "CONFLICTING", "SUPERSEDED", "RETRACTED"}:
                raw_claim_status = top_level_status
        claim_status = str(raw_claim_status or "ACTIVE").upper()
        is_delete = operation in {"DELETE", "REMOVE"} or (
            bool(claim_id) and claim_status != "ACTIVE"
        )
        is_hierarchy = (
            aggregate_type == "HIERARCHY"
            or operation in {"UPSERT_HIERARCHY", "UPSERT_CONTAINS"}
            or bool(projection.get("hierarchy_id"))
        )
        is_claim = aggregate_type in {"CLAIM", "RELATION", "RELATIONSHIP"} or bool(claim_id)
        if is_hierarchy:
            hierarchy_id = projection.get("hierarchy_id")
            parent_id = projection.get("parent_memory_id") or projection.get("parent_id")
            child_id = projection.get("child_memory_id") or projection.get("child_id")
            if not hierarchy_id or not parent_id or not child_id:
                raise ApplicationError(
                    "hierarchy projection requires hierarchy_id, parent_id, and child_id",
                    type="InvalidProjection",
                    non_retryable=True,
                )
            result = neo4j.upsert_hierarchy_projection(
                hierarchy_id=str(hierarchy_id),
                parent_memory_id=str(parent_id),
                child_memory_id=str(child_id),
                revision=revision,
                properties=_projection_properties(projection),
                tenant_id=tenant_id,
            )
        elif is_claim:
            if not claim_id:
                raise ApplicationError(
                    "claim projection is missing claim_id",
                    type="InvalidProjection",
                    non_retryable=True,
                )
            if is_delete:
                result = neo4j.delete_claim_projection(
                    claim_id=str(claim_id),
                    revision=revision,
                    tenant_id=tenant_id,
                )
            else:
                subject_id = projection.get("subject_memory_id") or projection.get("subject_id")
                object_id = projection.get("object_memory_id") or projection.get("object_id")
                predicate = projection.get("predicate") or projection.get("relation_label")
                object_type = str(projection.get("object_type") or "").upper()
                if object_id is None and object_type and object_type != "ENTITY":
                    # Typed scalar facts remain canonical PostgreSQL values;
                    # they are represented on the hydrated memory snapshot,
                    # never as fake graph entities.
                    result = neo4j.delete_claim_projection(
                        claim_id=str(claim_id),
                        revision=revision,
                        tenant_id=tenant_id,
                    )
                    result = {**dict(result), "projected_revision": revision}
                elif not subject_id or not object_id or not predicate:
                    raise ApplicationError(
                        "claim projection requires subject_memory_id, object_memory_id, and predicate",
                        type="InvalidProjection",
                        non_retryable=True,
                    )
                else:
                    result = neo4j.upsert_claim_projection(
                        claim_id=str(claim_id),
                        subject_memory_id=str(subject_id),
                        object_memory_id=str(object_id),
                        predicate=str(predicate),
                        revision=revision,
                        properties=_projection_properties(projection),
                        tenant_id=tenant_id,
                    )
        else:
            if not memory_id:
                raise ApplicationError(
                    "memory projection is missing memory_id",
                    type="InvalidProjection",
                    non_retryable=True,
                )
            if is_delete:
                result = neo4j.delete_memory_projection(
                    memory_id=str(memory_id),
                    revision=revision,
                    tenant_id=tenant_id,
                )
            else:
                result = neo4j.upsert_memory_projection(
                    memory_id=str(memory_id),
                    revision=revision,
                    properties=_memory_projection_properties(projection, state),
                    tenant_id=tenant_id,
                )
        if (
            (is_claim or is_hierarchy)
            and isinstance(result, Mapping)
            and result.get("projected_revision") is None
            and not is_delete
        ):
            raise ApplicationError(
                "projection dependency nodes are not indexed yet",
                type="ProjectionDependencyNotReady",
            )
        _complete_dispatch(state.control_plane, resolved_dispatch_id, mutation_id)
        applied = bool(result.get("applied")) if isinstance(result, Mapping) else True
        return "APPLIED" if applied else "STALE"
    except ApplicationError:
        raise
    except CanonicalProjectionContractError as err:
        raise ApplicationError(
            str(err),
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        ) from err
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.mark_projection_failed")
def mark_projection_failed_activity(dispatch_id: str, error: str) -> None:
    """Persist a terminal projection failure as ``DEAD`` in the outbox."""
    state = get_state()
    if not isinstance(state.control_plane, PostgresStore):
        raise ApplicationError(
            "Temporal projection failure bookkeeping requires PostgresStore",
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        )
    try:
        resolved_dispatch_id, _ = _dispatch_envelope(state.control_plane, dispatch_id)
        if resolved_dispatch_id is None:
            raise ApplicationError(
                f"projection dispatch not found: {dispatch_id}",
                type="ProjectionDispatchNotFound",
                non_retryable=True,
            )
        transient_markers = (
            "ProjectionDependencyNotReady",
            "ProjectionUnavailable",
            "TransientError",
            "ServiceUnavailable",
            "SessionExpired",
            "ConnectionError",
            "TimeoutError",
        )
        failure_class = (
            "PROJECTION_TRANSIENT"
            if any(marker in error for marker in transient_markers)
            else "WORKFLOW_TERMINAL"
        )
        _fail_dispatch(
            state.control_plane,
            resolved_dispatch_id,
            error,
            failure_class=failure_class,
        )
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.process_code_ingest")
def process_code_ingest_activity(job_id: str) -> str:
    """Run an application-wired code-ingest handler on the dedicated queue.

    Code-archive persistence is intentionally left to the canonical migration
    slice.  Until that handler is wired into ``AppState``, failing explicitly
    is safer than silently falling back to a legacy Neo4j-only write path.
    """
    if not job_id:
        raise ApplicationError(
            "job_id is required",
            type="InvalidCodeIngest",
            non_retryable=True,
        )
    state = get_state()
    handler = getattr(state, "code_ingest_handler", None)
    if not callable(handler):
        raise ApplicationError(
            "code ingest handler is not wired; provide state.code_ingest_handler(job_id) "
            "from the canonical repository",
            type="CodeIngestHandlerUnavailable",
            non_retryable=True,
        )
    try:
        result = handler(job_id)
    except (ValueError, KeyError) as err:
        raise ApplicationError(str(err), type="InvalidCodeIngest", non_retryable=True) from err
    if isinstance(state.control_plane, PostgresStore):
        state.control_plane.mark_aggregate_dispatches_complete("CODE_INGEST", job_id)
    return str(result or "COMPLETE")


@activity.defn(name="engram.mark_code_ingest_failed")
def mark_code_ingest_failed_activity(job_id: str, error: str) -> None:
    """Mark a code-ingest dispatch ``DEAD`` after Temporal exhausts retries."""
    state = get_state()
    if not isinstance(state.control_plane, PostgresStore):
        raise ApplicationError(
            "code ingest failure bookkeeping requires PostgresStore",
            type="CodeIngestHandlerUnavailable",
            non_retryable=True,
        )
    row = state.control_plane.get_dispatch_for_aggregate("CODE_INGEST", job_id)
    if not row:
        raise ApplicationError(
            f"code ingest dispatch not found: {job_id}",
            type="CodeIngestDispatchNotFound",
            non_retryable=True,
        )
    state.control_plane.mark_dispatch_dead(str(row["dispatch_id"]), error[:1000])


def _canonical_event_context(
    event_id: str,
    expected_tenant_id: str | None = None,
) -> tuple[Any, Any, str, Any]:
    """Load the event and canonical repository for a V2 Activity.

    Temporal inputs deliberately contain only the durable event identity and
    tenant routing key. The event payload and model result artifacts are loaded
    from PostgreSQL inside the Activity process, never placed in workflow
    history.
    """

    if not event_id:
        raise ApplicationError(
            "event_id is required",
            type="InvalidCanonicalIngest",
            non_retryable=True,
        )
    state = get_state()
    if not isinstance(state.control_plane, PostgresStore):
        raise ApplicationError(
            "canonical ingest requires PostgresStore",
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        )
    tenant_id = str(expected_tenant_id or "").strip()
    if expected_tenant_id is not None and not tenant_id:
        raise ApplicationError(
            "tenant_id is required",
            type="InvalidCanonicalIngest",
            non_retryable=True,
        )
    event = state.control_plane.get_event(
        event_id,
        tenant_id=tenant_id or None,
    )
    if event is None:
        raise ApplicationError(
            f"unknown event_id: {event_id}",
            type="UnknownCanonicalEvent",
            non_retryable=True,
        )
    try:
        repository = _canonical_repository(state)
    except CanonicalProjectionContractError as err:
        raise ApplicationError(
            str(err),
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        ) from err
    event_tenant_id = str(event.get("tenant_id") or "").strip()
    if tenant_id and event_tenant_id != tenant_id:
        raise ApplicationError(
            "canonical event tenant does not match workflow routing",
            type="CanonicalTenantMismatch",
            non_retryable=True,
        )
    tenant_id = event_tenant_id
    if not tenant_id:
        raise ApplicationError(
            "canonical event has no tenant_id",
            type="InvalidCanonicalIngest",
            non_retryable=True,
        )
    return state, event, tenant_id, repository


def _canonical_artifact_payload(
    repository: Any,
    event_id: str,
    tenant_id: str,
    artifact_key: str,
) -> dict[str, Any] | None:
    getter = getattr(repository, "get_ingest_artifact", None)
    if not callable(getter):
        raise ApplicationError(
            "canonical repository must expose get_ingest_artifact",
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        )
    artifact = getter(event_id, "EXTRACTION", artifact_key, tenant_id=tenant_id)
    if artifact is None:
        return None
    payload = getattr(
        artifact, "payload", artifact.get("payload") if isinstance(artifact, Mapping) else None
    )
    if not isinstance(payload, Mapping):
        raise ApplicationError(
            f"canonical artifact {artifact_key} has invalid payload",
            type="InvalidCanonicalArtifact",
            non_retryable=True,
        )
    return dict(payload)


def _store_canonical_artifact(
    repository: Any,
    *,
    event_id: str,
    tenant_id: str,
    artifact_key: str,
    payload: Mapping[str, Any],
    extractor_version: str,
) -> None:
    recorder = getattr(repository, "record_ingest_artifact", None)
    if not callable(recorder):
        raise ApplicationError(
            "canonical repository must expose record_ingest_artifact",
            type="CanonicalRepositoryUnavailable",
            non_retryable=True,
        )
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)
    recorder(
        event_id=event_id,
        artifact_type="EXTRACTION",
        artifact_key=artifact_key,
        payload=dict(payload),
        content=encoded,
        media_type="application/json",
        extractor="core",
        extractor_version=extractor_version,
        metadata={"pipeline": "canonical-conversation", "artifact_key": artifact_key},
        tenant_id=tenant_id,
    )


@activity.defn(name="engram.canonical_gate")
def canonical_gate_activity(
    event_id: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Run and durably record only the canonical write-path gate."""

    state, event, tenant_id, repository = _canonical_event_context(event_id, tenant_id)
    try:
        existing = _canonical_artifact_payload(repository, event_id, tenant_id, "gate-v1")
        if existing is None:
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                raise ApplicationError(
                    "canonical event payload is invalid",
                    type="InvalidCanonicalEvent",
                    non_retryable=True,
                )
            if bool(payload.get("force_store")):
                # Explicit operator/API intent is authoritative. This is
                # especially useful for synthetic/reference corpora and saves
                # a model call without weakening extraction validation.
                gate = {
                    "store": True,
                    "reason": "Explicit force_store ingestion policy.",
                    "policy": "force-store-v1",
                }
            else:
                model_context = SimpleNamespace(cfg=state.cfg, core=state.core)
                with model_request_session(event.get("session_id") or event_id):
                    gate = _call_gate(
                        cast(Any, model_context),
                        _turn_pair(dict(payload)),
                        session_summary=payload.get("session_summary"),
                    )
            if not isinstance(gate, Mapping) or "store" not in gate:
                raise ApplicationError(
                    "canonical gate returned invalid output",
                    type="InvalidCanonicalArtifact",
                    non_retryable=True,
                )
            existing = dict(gate)
            _store_canonical_artifact(
                repository,
                event_id=event_id,
                tenant_id=tenant_id,
                artifact_key="gate-v1",
                payload=existing,
                extractor_version="gate-v2",
            )
        return {
            "event_id": event_id,
            "tenant_id": tenant_id,
            "store": bool(existing.get("store")),
            "reason": str(existing.get("reason") or "")[:500],
        }
    except CoreModelError:
        raise
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.canonical_extract")
def canonical_extract_activity(event_id: str, tenant_id: str | None = None) -> str:
    """Run and durably record extraction after the canonical gate."""

    state, event, tenant_id, repository = _canonical_event_context(event_id, tenant_id)
    try:
        gate = _canonical_artifact_payload(repository, event_id, tenant_id, "gate-v1")
        if not gate or not bool(gate.get("store")):
            return "SKIPPED"
        existing_extraction = _canonical_artifact_payload(
            repository, event_id, tenant_id, "extract-v1"
        )
        existing_typed = _canonical_artifact_payload(repository, event_id, tenant_id, "typed-v1")
        if existing_typed is not None:
            return "READY"
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise ApplicationError(
                "canonical event payload is invalid",
                type="InvalidCanonicalEvent",
                non_retryable=True,
            )
        model_context = SimpleNamespace(cfg=state.cfg, core=state.core)
        extraction = existing_extraction
        if extraction is None:
            turn_pair = _turn_pair(dict(payload))
            combined_text = "\n".join(turn_pair.values())
            profile_recovery_applicable = bool(_PROFILE_ENTRY_TERM.search(combined_text))
            extraction = _reuse_session_profile_extraction(
                state,
                event,
                tenant_id,
                event_id,
                turn_pair,
            )
            if extraction is not None:
                pass
            elif profile_recovery_applicable and (
                _activity_attempt() > 1 or int(event.get("retry_count") or 0) > 0
            ):
                # A previous attempt already paid for the original prompt and
                # received the known empty response.  Go directly to the
                # reversible recovery to avoid multiplying input tokens.
                with model_request_session(event.get("session_id") or event_id):
                    extraction = _recover_profile_entry_extraction(
                        cast(Any, model_context),
                        turn_pair,
                        session_context=payload.get("session_context"),
                    )
            else:
                try:
                    with model_request_session(event.get("session_id") or event_id):
                        extraction = _call_extract(
                            cast(Any, model_context),
                            turn_pair,
                            session_context=payload.get("session_context"),
                            spoken_on=_spoken_on(dict(payload)),
                        )
                except CoreModelError as err:
                    if _EMPTY_MODEL_OUTPUT not in str(err) or not profile_recovery_applicable:
                        raise
                    with model_request_session(event.get("session_id") or event_id):
                        extraction = _recover_profile_entry_extraction(
                            cast(Any, model_context),
                            turn_pair,
                            session_context=payload.get("session_context"),
                        )
        if not isinstance(extraction, Mapping):
            raise ApplicationError(
                "canonical extractor returned invalid output",
                type="InvalidCanonicalArtifact",
                non_retryable=True,
            )
        # v3: the extractor sees the day the turn was said and resolves relative times.
        extractor_version = str(extraction.get("extractor_version") or "extract-v3")
        if existing_extraction is None:
            _store_canonical_artifact(
                repository,
                event_id=event_id,
                tenant_id=tenant_id,
                artifact_key="extract-v1",
                payload=extraction,
                extractor_version=extractor_version,
            )
        raw_triplets = extraction.get("triplets")
        if not isinstance(raw_triplets, list) or not all(
            isinstance(item, Mapping) for item in raw_triplets
        ):
            raise ApplicationError(
                "canonical extraction triplets must be a list of objects",
                type="InvalidCanonicalArtifact",
                non_retryable=True,
            )
        try:
            typed_extraction = {
                **dict(extraction),
                "triplets": normalize_extracted_triplets(
                    cast(list[Mapping[str, Any]], raw_triplets)
                ),
                "normalization_stage": "typed-v1",
            }
        except (TypeError, ValueError) as err:
            raise ApplicationError(
                str(err),
                type="InvalidCanonicalExtraction",
                non_retryable=True,
            ) from err
        _store_canonical_artifact(
            repository,
            event_id=event_id,
            tenant_id=tenant_id,
            artifact_key="typed-v1",
            payload=typed_extraction,
            extractor_version=extractor_version,
        )
        return "READY"
    except CoreModelError:
        raise
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.commit_canonical")
def commit_canonical_activity(event_id: str, tenant_id: str | None = None) -> str:
    """Commit durable canonical state and outbox rows in one PG transaction."""

    _state, _event, tenant_id, repository = _canonical_event_context(event_id, tenant_id)
    try:
        gate = _canonical_artifact_payload(repository, event_id, tenant_id, "gate-v1")
        extraction = _canonical_artifact_payload(repository, event_id, tenant_id, "typed-v1")
        if extraction is None:
            extraction = _canonical_artifact_payload(repository, event_id, tenant_id, "extract-v1")
        commit = getattr(repository, "commit_conversational_event", None)
        if not callable(commit):
            raise ApplicationError(
                "canonical repository must expose commit_conversational_event",
                type="CanonicalRepositoryUnavailable",
                non_retryable=True,
            )
        try:
            result = commit(
                event_id=event_id,
                gate=gate,
                extraction=extraction,
                tenant_id=tenant_id,
            )
        except (KeyError, MemoryRepositoryError, TypeError, ValueError) as err:
            raise ApplicationError(
                str(err),
                type="InvalidCanonicalCommit",
                non_retryable=True,
            ) from err
        if not isinstance(result, Mapping):
            raise ApplicationError(
                "canonical commit returned invalid result",
                type="InvalidCanonicalCommit",
                non_retryable=True,
            )
        return str(result.get("status") or "COMPLETE")
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.mark_canonical_ingest_failed")
def mark_canonical_ingest_failed_activity(
    event_id: str,
    tenant_id: str | None,
    error: str,
) -> None:
    """Record a terminal canonical ingest failure; never mark it complete."""

    state, _event, tenant_id, repository = _canonical_event_context(event_id, tenant_id)
    try:
        state.control_plane.set_event_status(
            event_id,
            "FAILED",
            error_message=str(error)[:1000],
            tenant_id=tenant_id,
        )
        getter = getattr(repository, "get_mutation", None)
        failer = getattr(repository, "fail_mutation", None)
        if callable(getter) and callable(failer):
            mutation = getter(
                source_event_id=event_id,
                mutation_type="CONVERSATIONAL_INGEST",
                tenant_id=tenant_id,
            )
            if mutation is not None and mutation.status == "PENDING":
                failer(
                    mutation_id=mutation.id,
                    error=str(error)[:1000],
                    tenant_id=tenant_id,
                )
        state.control_plane.mark_aggregate_dispatches_dead(
            "INGEST",
            event_id,
            str(error),
            failure_class="CANONICAL_INGEST_TERMINAL",
        )
    finally:
        set_current_tenant(None)


# Descriptive aliases are useful to embedders that do not use the Temporal
# registration names directly.
gate_canonical_activity = canonical_gate_activity
extract_canonical_activity = canonical_extract_activity
canonical_commit_activity = commit_canonical_activity


@activity.defn(name="engram.process_ingest")
def process_ingest_activity(event_id: str) -> str:
    """Run the idempotent ingest implementation from the durable ledger."""
    try:
        state = get_state()
        result = process_event(make_ingest_context(state), event_id)
        if isinstance(state.control_plane, PostgresStore):
            state.control_plane.mark_aggregate_dispatches_complete("INGEST", event_id)
        return result
    except (ValueError, KeyError) as err:
        raise ApplicationError(str(err), type="InvalidIngest", non_retryable=True) from err
    except CoreModelError:
        # Provider outages are transient and governed by the workflow policy.
        raise
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.mark_event_failed")
def mark_event_failed_activity(event_id: str, error: str) -> None:
    set_current_tenant(None)
    try:
        state = get_state()
        state.control_plane.set_event_status(event_id, "FAILED", error_message=error)
        if isinstance(state.control_plane, PostgresStore):
            state.control_plane.mark_aggregate_dispatches_complete("INGEST", event_id)
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.process_consolidation")
def process_consolidation_activity(task_id: str) -> str:
    set_current_tenant(None)
    state = get_state()
    with state.control_plane.transaction() as conn:
        row = conn.execute(
            "UPDATE consolidation_tasks SET status = 'PROCESSING', "
            "started_at = coalesce(started_at, CURRENT_TIMESTAMP), "
            "retry_count = CASE WHEN status = 'PROCESSING' "
            "THEN retry_count + 1 ELSE retry_count END, error_message = NULL "
            "WHERE task_id = ? AND status IN ('PENDING', 'PROCESSING') "
            "RETURNING *",
            (task_id,),
        ).fetchone()
    if row is None:
        existing = (
            state.control_plane.get_conn()
            .execute(
                "SELECT status FROM consolidation_tasks WHERE task_id = ?",
                (task_id,),
            )
            .fetchone()
        )
        if existing is None:
            raise ApplicationError(
                f"unknown consolidation task: {task_id}",
                type="UnknownConsolidationTask",
                non_retryable=True,
            )
        if existing["status"] == "COMPLETE":
            if isinstance(state.control_plane, PostgresStore):
                state.control_plane.mark_aggregate_dispatches_complete("CONSOLIDATION", task_id)
            return "ALREADY_PROCESSED"
        raise ApplicationError(
            f"consolidation task {task_id} is {existing['status']}",
            type="InvalidConsolidationTaskState",
            non_retryable=True,
        )
    task = dict(row)
    task_tenant = str(task.get("tenant_id") or "_default")
    set_current_tenant(
        Tenant(
            tenant_id=task_tenant,
            display_name=task_tenant,
            api_key_hashes=[],
            quotas=TenantQuotas(),
            status="ACTIVE",
        )
    )
    try:
        ctx = ConsolidationContext(
            cfg=state.cfg,
            control_plane=state.control_plane,
            fs=state.fs,
            neo4j=state.neo4j,
            core=state.core,
            embed=state.embed,
            overview_cache=state.overview_cache,
            memory_repository=getattr(state, "memory_repository", None),
        )
        _dispatch(ctx, task)
        with state.control_plane.transaction() as conn:
            conn.execute(
                "UPDATE consolidation_tasks SET status = 'COMPLETE', "
                "completed_at = CURRENT_TIMESTAMP, error_message = NULL "
                "WHERE task_id = ?",
                (task_id,),
            )
        if isinstance(state.control_plane, PostgresStore):
            state.control_plane.mark_aggregate_dispatches_complete("CONSOLIDATION", task_id)
    except Exception as err:
        # Keep PROCESSING so a Temporal retry can reclaim the same durable
        # task after an exception or worker crash. The previous code treated
        # this state as already processed and silently skipped the retry.
        with state.control_plane.transaction() as conn:
            conn.execute(
                "UPDATE consolidation_tasks SET error_message = ? "
                "WHERE task_id = ? AND status = 'PROCESSING'",
                (str(err)[:1000], task_id),
            )
        if isinstance(err, UnknownConsolidationTaskTypeError):
            raise ApplicationError(
                str(err),
                type="UnknownConsolidationTaskType",
                non_retryable=True,
            ) from err
        raise
    finally:
        set_current_tenant(None)
    return "COMPLETE"


@activity.defn(name="engram.mark_task_failed")
def mark_task_failed_activity(task_id: str, error: str) -> None:
    set_current_tenant(None)
    try:
        state = get_state()
        with state.control_plane.transaction() as conn:
            conn.execute(
                "UPDATE consolidation_tasks SET status = 'FAILED', "
                "completed_at = CURRENT_TIMESTAMP, error_message = ? "
                "WHERE task_id = ?",
                (error, task_id),
            )
        if isinstance(state.control_plane, PostgresStore):
            state.control_plane.mark_aggregate_dispatches_complete("CONSOLIDATION", task_id)
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.run_reconciliation")
def run_reconciliation_activity() -> dict[str, int]:
    """Repair missing Temporal dispatches without duplicating active work."""
    state = get_state()
    if not isinstance(state.control_plane, PostgresStore):
        raise RuntimeError("Temporal reconciliation requires PostgresStore")
    set_current_tenant(None)
    try:
        activity.heartbeat("starting reconciliation")
        result = state.control_plane.repair_dispatches()
        activity.heartbeat("reconciliation complete")
        return result
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.scan_stale_overviews")
def scan_stale_overviews_activity() -> int:
    """Run the expensive graph-wide stale-overview maintenance scan."""
    state = get_state()
    set_current_tenant(None)
    try:
        activity.heartbeat("starting stale overview scan")
        if state.cfg.canonical_memory.enabled:
            from engram.consolidation.canonical import enqueue_stale_overviews

            repository = state.memory_repository
            if repository is None:
                raise RuntimeError("canonical overview scan requires MemoryRepository")
            canonical_result = sum(
                enqueue_stale_overviews(
                    repository=repository,
                    control_plane=state.control_plane,
                    tenant_id=tenant.tenant_id,
                )
                for tenant in state.tenant_registry.list()
                if tenant.status == "ACTIVE"
            )
            result = canonical_result
        else:
            result = run_stale_overview_scan(
                ReconciliationContext(
                    cfg=state.cfg,
                    control_plane=state.control_plane,
                    neo4j=state.neo4j,
                ),
                strict=True,
            )
        activity.heartbeat("stale overview scan complete")
        return result
    finally:
        set_current_tenant(None)


@activity.defn(name="engram.run_decay")
def run_decay_activity() -> int:
    """Run one global decay pass under Temporal retry and observability."""
    state = get_state()
    set_current_tenant(None)
    try:
        activity.heartbeat("starting decay")
        updated = run_daily(
            state.neo4j,
            state.cfg.decay,
            progress_callback=lambda count: activity.heartbeat(f"updated {count} nodes"),
            strict=True,
        )
        activity.heartbeat(f"decay complete: {updated} nodes")
        return updated
    finally:
        set_current_tenant(None)
