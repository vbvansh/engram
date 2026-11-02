"""Canonical PostgreSQL memory repository.

This module is deliberately independent from the legacy filesystem and from
the Neo4j projection.  It provides typed domain records, deterministic
normalisation helpers, and transactional operations for the first canonical
memory slice.  Callers pass the existing PostgreSQL control-plane store (the
store supplies the repository's transaction and placeholder adapter).

The repository is intentionally usable before ingest/retrieval integration:
    all writes are explicit, tenant-scoped, and can emit Temporal dispatch rows,
while no operation reads or writes a filesystem path.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import unicodedata
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field as PydanticField

from engram.predicate_registry import normalize_claim_value
from engram.storage.canonical_schema import schema_statements
from engram.tenancy import DEFAULT_TENANT_ID

JsonValue = Any
ObjectType = Literal[
    "ENTITY",
    "STRING",
    "TEXT",
    "ROLE",
    "NUMBER",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "DATE",
    "DATETIME",
    "JSON",
    "LIST",
]

_MEMORY_TYPES = frozenset(
    {
        "ENTITY",
        "EPISODE",
        "SESSION_SUMMARY",
        "COLLECTION",
        "PROJECT",
        "FILE",
        "CLASS",
        "FUNCTION",
        "METHOD",
        "EXTERNAL_MODULE",
        "EVENT",
        "FACT",
        "DOCUMENT",
        "DIRECTORY",
        "PROFILE",
        "PREFERENCE",
    }
)
_NODE_STATUSES = frozenset(
    {"ACTIVE", "HISTORICAL", "RETIRED", "MERGED", "LOW_CONFIDENCE", "CONFLICTING", "DELETED"}
)
_CLAIM_STATUSES = frozenset(
    {"ACTIVE", "HISTORICAL", "SUPERSEDED", "LOW_CONFIDENCE", "CONFLICTING", "RETRACTED"}
)
_OBJECT_TYPES = frozenset(
    {
        "ENTITY",
        "STRING",
        "TEXT",
        "ROLE",
        "NUMBER",
        "INTEGER",
        "FLOAT",
        "BOOLEAN",
        "DATE",
        "DATETIME",
        "JSON",
        "LIST",
    }
)
_MAX_ZIP_MEMBERS = 1000
_MAX_ZIP_MEMBER_BYTES = 10 * 1024 * 1024
_MAX_ZIP_EXPANDED_BYTES = 50 * 1024 * 1024
_MUTATION_STATUSES = frozenset({"PENDING", "APPLIED", "FAILED"})
_PROJECTION_STATUSES = frozenset({"PENDING", "PROCESSING", "PROCESSED", "FAILED"})
_MUTATION_NAMESPACE = uuid.UUID("7f8f285b-9586-4c17-bf38-04cf4e2c6aa2")
_CLAIM_NAMESPACE = uuid.UUID("9b0cc6af-7bd7-4bbd-89ac-8305c3e0df95")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:")


class MemoryRepositoryError(RuntimeError):
    """Base error raised for canonical repository failures."""


class MemoryNotFoundError(MemoryRepositoryError, LookupError):
    """Raised when a tenant-scoped memory cannot be found."""


class InvalidClaimError(MemoryRepositoryError, ValueError):
    """Raised when a typed claim violates the canonical object contract."""


class RepositoryStore(Protocol):
    """Minimal interface implemented by :class:`PostgresStore`."""

    def get_conn(self) -> Any: ...

    def transaction(self) -> Any: ...


class _BoundTransactionStore:
    """Store facade that keeps repository operations inside one outer tx."""

    def __init__(self, conn: Any, *, projection_task_queue: str = "engram-projection") -> None:
        self.conn = conn
        self.projection_task_queue = projection_task_queue

    def get_conn(self) -> Any:
        return self.conn

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        yield self.conn


@dataclass(frozen=True, slots=True)
class MemoryNode:
    id: uuid.UUID
    tenant_id: str
    memory_type: str
    canonical_name: str | None
    canonical_uri: str
    status: str
    current_version_id: uuid.UUID | None
    revision: int
    origin_event_id: str | None
    metadata: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    id: uuid.UUID
    tenant_id: str
    memory_id: uuid.UUID
    version_number: int
    body: str
    abstract: str | None
    asserted_at: datetime
    valid_from: datetime | None
    valid_until: datetime | None
    source_event_id: str | None
    provenance: dict[str, JsonValue]
    metadata: dict[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TypedClaimCandidate:
    """Typed claim candidate accepted by ``MemoryRepository.add_claim``.

    Exactly one of ``object_entity_id`` and ``object_value`` must be present.
    The object type is explicit so dates, numbers, roles, and booleans are not
    accidentally turned into graph entities.
    """

    subject_id: uuid.UUID
    predicate: str
    object_type: ObjectType
    object_entity_id: uuid.UUID | None = None
    object_value: JsonValue = None
    confidence: float | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    asserted_at: datetime | None = None
    source_event_id: str | None = None
    source_version_id: uuid.UUID | None = None
    source_triplet_index: int | None = None
    normalized_object_hash: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.object_type not in _OBJECT_TYPES:
            raise InvalidClaimError(f"unsupported object_type: {self.object_type!r}")
        has_entity = self.object_entity_id is not None
        has_value = self.object_value is not None
        if has_entity == has_value:
            raise InvalidClaimError("a claim must have exactly one entity or scalar object")
        if self.object_type == "ENTITY" and not has_entity:
            raise InvalidClaimError("ENTITY claims require object_entity_id")
        if self.object_type != "ENTITY" and has_entity:
            raise InvalidClaimError(f"{self.object_type} claims cannot reference an entity")
        if self.confidence is not None and not 0 <= float(self.confidence) <= 1:
            raise InvalidClaimError("claim confidence must be between 0 and 1")
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise InvalidClaimError("valid_until must be after valid_from")


# Compatibility name used by the initial repository slice.
TypedClaim = TypedClaimCandidate


@dataclass(frozen=True, slots=True)
class MemoryClaim:
    id: uuid.UUID
    tenant_id: str
    subject_id: uuid.UUID
    predicate: str
    object_entity_id: uuid.UUID | None
    object_value: JsonValue
    object_type: str
    status: str
    confidence: float | None
    valid_from: datetime | None
    valid_until: datetime | None
    asserted_at: datetime
    source_event_id: str | None
    source_version_id: uuid.UUID | None
    source_triplet_index: int | None
    supersedes_claim_id: uuid.UUID | None
    superseded_by_claim_id: uuid.UUID | None
    normalized_object_hash: str | None
    metadata: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    id: uuid.UUID
    tenant_id: str
    memory_id: uuid.UUID
    claim_id: uuid.UUID | None
    source_event_id: str | None
    source_session_id: str | None
    extractor: str | None
    extractor_version: str | None
    confidence: float | None
    source_span: JsonValue
    source_text_hash: str | None
    metadata: dict[str, JsonValue]
    idempotency_key: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EntityAlias:
    id: uuid.UUID
    tenant_id: str
    entity_id: uuid.UUID
    alias: str
    normalized_alias: str
    confidence: float | None
    source_event_id: str | None
    metadata: dict[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryHierarchy:
    id: uuid.UUID
    tenant_id: str
    parent_id: uuid.UUID
    child_id: uuid.UUID
    position: int | None
    metadata: dict[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryOverview:
    id: uuid.UUID
    tenant_id: str
    scope_id: uuid.UUID
    content: str
    input_revision: int
    model_metadata: dict[str, JsonValue]
    generated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryUriAlias:
    id: uuid.UUID
    tenant_id: str
    memory_id: uuid.UUID
    uri: str
    alias_type: str
    metadata: dict[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IngestArtifact:
    id: uuid.UUID
    tenant_id: str
    event_id: str
    artifact_type: str
    artifact_key: str
    payload: dict[str, JsonValue]
    content: bytes
    content_hash: str
    media_type: str
    expires_at: datetime | None
    extractor: str | None
    extractor_version: str | None
    source_span: JsonValue
    metadata: dict[str, JsonValue]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class CodeProjectCommit:
    """Result of committing a ZIP-backed code project to canonical PG state."""

    project: MemoryNode
    nodes: tuple[MemoryNode, ...]
    artifact: IngestArtifact

    @property
    def project_uri(self) -> str:
        return self.project.canonical_uri


@dataclass(frozen=True, slots=True)
class MemoryPage:
    """Keyset-paginated tenant memory listing."""

    nodes: tuple[MemoryNode, ...]
    next_cursor: str | None

    @property
    def items(self) -> tuple[MemoryNode, ...]:
        return self.nodes

    def __iter__(self) -> Iterator[Any]:
        yield self.nodes
        yield self.next_cursor


@dataclass(frozen=True, slots=True)
class CanonicalMutation:
    id: uuid.UUID
    tenant_id: str
    source_event_id: str
    mutation_type: str
    status: str
    revision: int
    result_memory_id: uuid.UUID | None
    payload: dict[str, JsonValue]
    result: dict[str, JsonValue]
    error_message: str | None
    created_at: datetime
    applied_at: datetime | None
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.source_event_id.strip():
            raise ValueError("canonical mutation requires tenant and source event identities")
        if self.status not in _MUTATION_STATUSES:
            raise ValueError(f"unsupported canonical mutation status: {self.status}")
        if self.revision < 1:
            raise ValueError("canonical mutation revision must be positive")

    @property
    def mutation_key(self) -> str:
        """Compatibility alias for callers using the early repository API."""

        return self.source_event_id

    @property
    def event_id(self) -> str:
        """Compatibility alias for the source event identity."""

        return self.source_event_id

    @property
    def operation(self) -> str:
        """Compatibility alias; mutation type is the canonical operation name."""

        return self.mutation_type


@dataclass(frozen=True, slots=True)
class WorkflowDispatch:
    id: str
    tenant_id: str
    aggregate_type: str
    aggregate_id: str
    operation: str
    revision: int
    payload: dict[str, JsonValue]
    status: str
    attempt_count: int
    available_at: datetime
    created_at: datetime
    processed_at: datetime | None
    last_error: str | None

    @property
    def dispatch_id(self) -> str:
        return self.id

    @property
    def workflow_type(self) -> str:
        return self.aggregate_type

    @property
    def aggregate_revision(self) -> int:
        return self.revision


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Canonical read model assembled from PostgreSQL only."""

    node: MemoryNode
    current_version: MemoryVersion | None
    claims: tuple[MemoryClaim, ...] = ()
    aliases: tuple[EntityAlias, ...] = ()
    evidence: tuple[MemoryEvidence, ...] = ()
    parents: tuple[MemoryHierarchy, ...] = ()
    children: tuple[MemoryHierarchy, ...] = ()
    overview: MemoryOverview | None = None


# Compatibility name used by existing API and repository callers.
MemoryState = MemorySnapshot


class ProjectionSnapshot(BaseModel):
    """Validated projection envelope hydrated from canonical PostgreSQL."""

    model_config = ConfigDict(extra="allow")

    tenant_id: str = PydanticField(min_length=1)
    operation: str = PydanticField(min_length=1)
    aggregate_type: str = "MEMORY"
    revision: int = PydanticField(ge=1)
    canonical_mutation_id: str | None = None
    memory_id: str | None = None
    claim_id: str | None = None
    hierarchy_id: str | None = None

    @field_validator("tenant_id", "operation", "aggregate_type")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("projection routing fields must not be blank")
        return value


class CanonicalMutationCallback(Protocol):
    def __call__(self, conn: Any, mutation: CanonicalMutation) -> Any: ...


def normalize_alias(value: str) -> str:
    """Normalize an entity alias without using slugs or filesystem semantics."""

    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    # Extractors commonly emit identifier-shaped names (``atul_singh``) for
    # human-readable entities.  An underscore is a token separator here, not
    # part of canonical identity.  Hyphens remain significant because names
    # such as ``Orion-DEV`` and ``Orion-UAT`` are distinct domain identifiers.
    return " ".join(re.sub(r"_+", " ", normalized).split())


def normalize_predicate(value: str) -> str:
    """Return the deterministic canonical predicate spelling.

    The mapping is intentionally small and explicit for the first slice.  A
    domain vocabulary can extend it without changing storage identity.
    """

    raw = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    legacy_aliases = {
        "has_role": "HAS_ROLE",
        "holds_role": "HAS_ROLE",
        "current_role": "HAS_ROLE",
        "role": "HAS_ROLE",
        "works_at": "WORKS_AT",
        "works_for": "WORKS_AT",
        "employed_by": "WORKS_AT",
        "lives_in": "LOCATED_IN",
        "located_in": "LOCATED_IN",
        "resides_in": "LOCATED_IN",
        "born_on": "HAS_BIRTH_DATE",
        "birthday": "HAS_BIRTH_DATE",
    }
    if raw in legacy_aliases:
        return legacy_aliases[raw]
    try:
        from engram.predicate_registry import normalize_predicate as registry_normalize

        return str(registry_normalize(str(value)).canonical_predicate)
    except (ImportError, AttributeError, TypeError, ValueError):
        # Keep this module usable in minimal migration tooling where the
        # optional policy registry is not installed yet.
        pass
    aliases = {
        "has_role": "HAS_ROLE",
        "holds_role": "HAS_ROLE",
        "current_role": "HAS_ROLE",
        "role": "HAS_ROLE",
        "works_at": "WORKS_AT",
        "works_for": "WORKS_AT",
        "employed_by": "WORKS_AT",
        "lives_in": "LIVES_IN",
        "located_in": "LIVES_IN",
        "resides_in": "LIVES_IN",
        "born_on": "BORN_ON",
        "birthday": "BORN_ON",
    }
    return aliases.get(raw, raw.upper())


def infer_object_type(value: JsonValue, *, hint: str | None = None) -> ObjectType:
    """Infer a safe scalar type; explicit entity IDs always bypass this helper."""

    if hint:
        normalized_hint = str(hint).strip().upper()
        if normalized_hint in _OBJECT_TYPES:
            return cast(ObjectType, normalized_hint)
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int) and not isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, datetime):
        return "DATETIME"
    if isinstance(value, date):
        return "DATE"
    if isinstance(value, list):
        return "LIST"
    if isinstance(value, Mapping):
        return "JSON"
    if isinstance(value, str):
        if _ISO_DATE_RE.fullmatch(value.strip()):
            return "DATE"
        if _ISO_DATETIME_RE.match(value.strip()):
            return "DATETIME"
    return "STRING"


def typed_claim(
    *,
    subject_id: uuid.UUID,
    predicate: str,
    object_value: JsonValue = None,
    object_entity_id: uuid.UUID | None = None,
    object_type: str | None = None,
    **kwargs: Any,
) -> TypedClaim:
    """Build and validate a typed claim candidate from extraction output."""

    if object_entity_id is not None:
        resolved_type: ObjectType = "ENTITY"
    else:
        resolved_type = cast(ObjectType, infer_object_type(object_value, hint=object_type))
    return TypedClaim(
        subject_id=subject_id,
        predicate=normalize_predicate(predicate),
        object_type=resolved_type,
        object_entity_id=object_entity_id,
        object_value=object_value,
        **kwargs,
    )


def _json(value: JsonValue, *, default: JsonValue) -> str:
    if value is None:
        value = default
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object(value: Any) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_any(value: Any) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, int, float, bool, str)):
        return list(value) if isinstance(value, tuple) else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _record_to_json(value: Any) -> JsonValue:
    """Convert typed repository records to JSON-safe projection payloads."""

    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _record_to_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _record_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_to_json(item) for item in value]
    return value


def _uuid(value: Any) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _datetime(value: Any, *, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _turn_said_at(*turns: Any) -> datetime | None:
    """The first parseable client timestamp of the given turns, if any."""
    for turn in turns:
        raw = turn.get("timestamp") if isinstance(turn, Mapping) else None
        if not raw:
            continue
        try:
            return _datetime(str(raw).strip())
        except ValueError:
            continue
    return None


def _required_datetime(value: Any) -> datetime:
    parsed = _datetime(value, default=datetime.now(timezone.utc))
    assert parsed is not None
    return parsed


def _hash_object(object_type: str, object_entity_id: uuid.UUID | None, value: Any) -> str:
    if object_entity_id is not None:
        canonical = {"object_entity_id": str(object_entity_id), "object_type": object_type}
    else:
        canonical = {"object_type": object_type, "object_value": _json_any(value)}
    return hashlib.sha256(_json(canonical, default={}).encode("utf-8")).hexdigest()


def _deterministic_uuid(namespace: uuid.UUID, *parts: Any) -> uuid.UUID:
    return uuid.uuid5(namespace, "|".join(str(part) for part in parts))


def _tenant_id(value: str | None, default: str) -> str:
    tenant = str(value or default).strip()
    if not tenant:
        raise ValueError("tenant_id must not be empty")
    return tenant


def _validate_uri(uri: str) -> str:
    normalized = str(uri).strip()
    if not normalized:
        raise ValueError("canonical_uri must not be empty")
    if not normalized.startswith("mem://"):
        raise ValueError("canonical_uri must use the mem:// scheme")
    if normalized.endswith(".md"):
        raise ValueError("canonical_uri must not encode a physical .md file")
    return normalized.rstrip("/")


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _node_from_row(row: Any) -> MemoryNode:
    item = _row_dict(row)
    return MemoryNode(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        memory_type=str(item["memory_type"]),
        canonical_name=item.get("canonical_name"),
        canonical_uri=str(item["canonical_uri"]),
        status=str(item["status"]),
        current_version_id=_uuid(item.get("current_version_id")),
        revision=int(item["revision"]),
        origin_event_id=item.get("origin_event_id"),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
        updated_at=_required_datetime(item.get("updated_at")),
    )


def _version_from_row(row: Any) -> MemoryVersion:
    item = _row_dict(row)
    return MemoryVersion(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        memory_id=cast(uuid.UUID, _uuid(item["memory_id"])),
        version_number=int(item["version_number"]),
        body=str(item.get("body") or ""),
        abstract=item.get("abstract"),
        asserted_at=_required_datetime(item.get("asserted_at")),
        valid_from=_datetime(item.get("valid_from")),
        valid_until=_datetime(item.get("valid_until")),
        source_event_id=item.get("source_event_id"),
        provenance=_json_object(item.get("provenance")),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
    )


def _claim_from_row(row: Any) -> MemoryClaim:
    item = _row_dict(row)
    return MemoryClaim(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        subject_id=cast(uuid.UUID, _uuid(item["subject_id"])),
        predicate=str(item["predicate"]),
        object_entity_id=_uuid(item.get("object_entity_id")),
        object_value=_json_any(item.get("object_value")),
        object_type=str(item["object_type"]),
        status=str(item["status"]),
        confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
        valid_from=_datetime(item.get("valid_from")),
        valid_until=_datetime(item.get("valid_until")),
        asserted_at=_required_datetime(item.get("asserted_at")),
        source_event_id=item.get("source_event_id"),
        source_version_id=_uuid(item.get("source_version_id")),
        source_triplet_index=(
            int(item["source_triplet_index"])
            if item.get("source_triplet_index") is not None
            else None
        ),
        supersedes_claim_id=_uuid(item.get("supersedes_claim_id")),
        superseded_by_claim_id=_uuid(item.get("superseded_by_claim_id")),
        normalized_object_hash=item.get("normalized_object_hash"),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
        updated_at=_required_datetime(item.get("updated_at")),
    )


def _evidence_from_row(row: Any) -> MemoryEvidence:
    item = _row_dict(row)
    return MemoryEvidence(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        memory_id=cast(uuid.UUID, _uuid(item["memory_id"])),
        claim_id=_uuid(item.get("claim_id")),
        source_event_id=item.get("source_event_id"),
        source_session_id=item.get("source_session_id"),
        extractor=item.get("extractor"),
        extractor_version=item.get("extractor_version"),
        confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
        source_span=_json_any(item.get("source_span")),
        source_text_hash=item.get("source_text_hash"),
        metadata=_json_object(item.get("metadata")),
        idempotency_key=item.get("idempotency_key"),
        created_at=_required_datetime(item.get("created_at")),
    )


def _alias_from_row(row: Any) -> EntityAlias:
    item = _row_dict(row)
    return EntityAlias(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        entity_id=cast(uuid.UUID, _uuid(item["entity_id"])),
        alias=str(item["alias"]),
        normalized_alias=str(item["normalized_alias"]),
        confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
        source_event_id=item.get("source_event_id"),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
    )


def _hierarchy_from_row(row: Any) -> MemoryHierarchy:
    item = _row_dict(row)
    return MemoryHierarchy(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        parent_id=cast(uuid.UUID, _uuid(item["parent_id"])),
        child_id=cast(uuid.UUID, _uuid(item["child_id"])),
        position=int(item["position"]) if item.get("position") is not None else None,
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
    )


def _overview_from_row(row: Any) -> MemoryOverview:
    item = _row_dict(row)
    return MemoryOverview(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        scope_id=cast(uuid.UUID, _uuid(item["scope_id"])),
        content=str(item.get("content") or ""),
        input_revision=int(item["input_revision"]),
        model_metadata=_json_object(item.get("model_metadata")),
        generated_at=_required_datetime(item.get("generated_at")),
    )


def _uri_alias_from_row(row: Any) -> MemoryUriAlias:
    item = _row_dict(row)
    return MemoryUriAlias(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        memory_id=cast(uuid.UUID, _uuid(item["memory_id"])),
        uri=str(item["uri"]),
        alias_type=str(item["alias_type"]),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
    )


def _artifact_from_row(row: Any) -> IngestArtifact:
    item = _row_dict(row)
    raw_content = item.get("content")
    if isinstance(raw_content, memoryview):
        raw_content = raw_content.tobytes()
    if isinstance(raw_content, bytearray):
        raw_content = bytes(raw_content)
    if isinstance(raw_content, str):
        # This branch only supports rows written by an early development
        # schema; production canonical rows are BYTEA and are returned as
        # bytes by psycopg.
        raw_content = raw_content.encode("utf-8")
    return IngestArtifact(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        event_id=str(item["event_id"]),
        artifact_type=str(item["artifact_type"]),
        artifact_key=str(item["artifact_key"]),
        payload=_json_object(item.get("payload")),
        content=bytes(raw_content or b""),
        content_hash=str(item.get("content_hash") or ""),
        media_type=str(item.get("media_type") or "application/zip"),
        expires_at=_datetime(item.get("expires_at")),
        extractor=item.get("extractor"),
        extractor_version=item.get("extractor_version"),
        source_span=_json_any(item.get("source_span")),
        metadata=_json_object(item.get("metadata")),
        created_at=_required_datetime(item.get("created_at")),
    )


def _mutation_from_row(row: Any) -> CanonicalMutation:
    item = _row_dict(row)
    return CanonicalMutation(
        id=cast(uuid.UUID, _uuid(item["id"])),
        tenant_id=str(item["tenant_id"]),
        source_event_id=str(item["source_event_id"]),
        mutation_type=str(item["mutation_type"]),
        status=str(item["status"]),
        revision=int(item["revision"]),
        result_memory_id=_uuid(item.get("result_memory_id")),
        payload=_json_object(item.get("payload")),
        result=_json_object(item.get("result")),
        error_message=item.get("error_message"),
        created_at=_required_datetime(item.get("created_at")),
        applied_at=_datetime(item.get("applied_at")),
        updated_at=_required_datetime(item.get("updated_at")),
    )


def _dispatch_from_row(row: Any) -> WorkflowDispatch:
    item = _row_dict(row)
    payload = _json_object(item.get("payload"))
    aggregate_type = str(item.get("workflow_type") or payload.get("aggregate_type") or "PROJECTION")
    operation = str(payload.get("operation") or item.get("operation") or "UPSERT")
    revision_value = item.get("aggregate_revision")
    if revision_value is None:
        revision_value = payload.get("revision") or 1
    return WorkflowDispatch(
        id=str(item.get("dispatch_id") or item.get("id")),
        tenant_id=str(item["tenant_id"]),
        aggregate_type=aggregate_type,
        aggregate_id=str(item["aggregate_id"]),
        operation=operation,
        revision=int(revision_value),
        payload=payload,
        status=str(item["status"]),
        attempt_count=int(item.get("attempts") or item.get("failure_count") or 0),
        available_at=_required_datetime(item.get("available_at")),
        created_at=_required_datetime(item.get("created_at")),
        processed_at=_datetime(item.get("completed_at") or item.get("processed_at")),
        last_error=item.get("last_error"),
    )


class MemoryRepository:
    """Tenant-scoped PostgreSQL repository for canonical memory state."""

    def __init__(
        self, store: RepositoryStore, *, default_tenant_id: str = DEFAULT_TENANT_ID
    ) -> None:
        if not hasattr(store, "get_conn") or not hasattr(store, "transaction"):
            raise TypeError("MemoryRepository requires a PostgreSQL-compatible store")
        self.store = store
        self.default_tenant_id = default_tenant_id
        self.projection_task_queue = str(
            getattr(store, "projection_task_queue", "engram-projection")
        )

    # ------------------------------------------------------------------
    # Schema/bootstrap and transaction helpers
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the canonical tables for local bootstrap/tests.

        Production should apply the forward-only Alembic revision instead;
        this helper mirrors the existing ``PostgresStore`` bootstrap pattern.
        """

        with self.store.transaction() as conn:
            for statement in schema_statements():
                conn.execute(statement)

    initialize_schema = ensure_schema

    @contextmanager
    def transaction(self, *, tenant_id: str | None = None) -> Iterator[Any]:
        """Expose a tenant-contextual PostgreSQL transaction for compositions."""

        with self.store.transaction() as conn:
            if tenant_id is not None:
                self._set_tenant_context(conn, self._tenant(tenant_id), local=True)
            yield conn

    @staticmethod
    def _set_tenant_context(conn: Any, tenant_id: str, *, local: bool) -> None:
        """Set the RLS tenant variable without interpolating tenant input."""

        scope = "true" if local else "false"
        conn.execute(f"SELECT set_config('app.tenant_id', ?, {scope})", (tenant_id,)).fetchone()

    @contextmanager
    def _transaction(self, tenant_id: str) -> Iterator[Any]:
        with self.store.transaction() as conn:
            self._set_tenant_context(conn, tenant_id, local=True)
            yield conn

    def _tenant_connection(self, tenant_id: str) -> Any:
        """Prepare an autocommit read connection for canonical RLS checks."""

        conn = self.store.get_conn()
        self._set_tenant_context(conn, tenant_id, local=False)
        return conn

    def _tenant(self, tenant_id: str | None) -> str:
        return _tenant_id(tenant_id, self.default_tenant_id)

    @staticmethod
    def _get_node(
        conn: Any, tenant_id: str, memory_id: uuid.UUID, *, lock: bool = False
    ) -> MemoryNode | None:
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            "SELECT * FROM memory_nodes WHERE tenant_id = ? AND id = ?" + suffix,
            (tenant_id, memory_id),
        ).fetchone()
        return _node_from_row(row) if row is not None else None

    def _require_node(
        self, conn: Any, tenant_id: str, memory_id: uuid.UUID, *, lock: bool = False
    ) -> MemoryNode:
        node = self._get_node(conn, tenant_id, memory_id, lock=lock)
        if node is None:
            raise MemoryNotFoundError(f"memory {memory_id} not found for tenant {tenant_id}")
        return node

    @staticmethod
    def _next_canonical_revision(conn: Any, tenant_id: str) -> int:
        # Serialize revisions per tenant.  No PostgreSQL extension is needed;
        # hashtextextended is built into PostgreSQL 16.
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))", (tenant_id,)
        ).fetchone()
        row = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) + 1 AS revision "
            "FROM canonical_mutations WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchone()
        return int(row["revision"] if row else 1)

    @staticmethod
    def _begin_mutation_in_tx(
        conn: Any,
        *,
        tenant_id: str,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        # Compatibility names retained at the boundary while the persisted
        # identity is unambiguously (tenant, source_event_id, mutation_type).
        mutation_key: str | None = None,
        operation: str | None = None,
        event_id: str | None = None,
        payload: Mapping[str, JsonValue] | None,
    ) -> tuple[CanonicalMutation, bool]:
        source = str(source_event_id or event_id or mutation_key or "").strip()
        if not source:
            raise ValueError("source_event_id must not be empty")
        kind = str(mutation_type or operation or "").strip().upper()
        if not kind:
            raise ValueError("mutation_type must not be empty")
        stored_payload = dict(payload or {})
        if mutation_key is not None and str(mutation_key).strip() != source:
            stored_payload.setdefault("mutation_key", str(mutation_key).strip())
        existing = conn.execute(
            "SELECT * FROM canonical_mutations "
            "WHERE tenant_id = ? AND source_event_id = ? AND mutation_type = ? FOR UPDATE",
            (tenant_id, source, kind),
        ).fetchone()
        if existing is not None:
            return _mutation_from_row(existing), False
        revision = MemoryRepository._next_canonical_revision(conn, tenant_id)
        mutation_id = uuid.uuid4()
        row = conn.execute(
            "INSERT INTO canonical_mutations "
            "(id, tenant_id, source_event_id, mutation_type, revision, payload) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING *",
            (
                mutation_id,
                tenant_id,
                source,
                kind,
                revision,
                _json(stored_payload, default={}),
            ),
        ).fetchone()
        if row is None:
            raise MemoryRepositoryError("canonical mutation insert returned no row")
        return _mutation_from_row(row), True

    @staticmethod
    def _complete_mutation_in_tx(
        conn: Any,
        mutation: CanonicalMutation,
        *,
        result_memory_id: uuid.UUID | None = None,
        result: Mapping[str, JsonValue] | None = None,
    ) -> CanonicalMutation:
        row = conn.execute(
            "UPDATE canonical_mutations SET status = 'APPLIED', result_memory_id = ?, "
            "result = ?, applied_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP, "
            "error_message = NULL WHERE tenant_id = ? AND id = ? RETURNING *",
            (
                result_memory_id,
                _json(dict(result or {}), default={}),
                mutation.tenant_id,
                mutation.id,
            ),
        ).fetchone()
        if row is None:
            raise MemoryRepositoryError(f"canonical mutation disappeared: {mutation.id}")
        return _mutation_from_row(row)

    @staticmethod
    def _fail_mutation_in_tx(conn: Any, mutation: CanonicalMutation, error: str) -> None:
        conn.execute(
            "UPDATE canonical_mutations SET status = 'FAILED', error_message = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ?",
            (str(error)[:2000], mutation.tenant_id, mutation.id),
        )

    def begin_mutation(
        self,
        *,
        mutation_key: str | None = None,
        operation: str | None = None,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        tenant_id: str | None = None,
        event_id: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> CanonicalMutation:
        """Create or return a tenant-scoped idempotency record."""

        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            mutation, _ = self._begin_mutation_in_tx(
                conn,
                tenant_id=tenant,
                source_event_id=source_event_id,
                mutation_type=mutation_type,
                mutation_key=mutation_key,
                operation=operation,
                event_id=event_id,
                payload=payload,
            )
            return mutation

    def get_mutation(
        self,
        identifier: str | uuid.UUID | None = None,
        *,
        mutation_id: str | uuid.UUID | None = None,
        mutation_key: str | None = None,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        tenant_id: str | None = None,
    ) -> CanonicalMutation | None:
        tenant = self._tenant(tenant_id)
        lookup = mutation_id or identifier
        conn = self._tenant_connection(tenant)
        row = None
        try_id = None
        if lookup is not None:
            try:
                try_id = uuid.UUID(str(lookup))
            except (TypeError, ValueError, AttributeError):
                try_id = None
        if try_id is not None:
            row = conn.execute(
                "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND id = ?",
                (tenant, try_id),
            ).fetchone()
        if row is None:
            source = (
                source_event_id or mutation_key or (str(lookup) if lookup is not None else None)
            )
            if source is None:
                raise TypeError("mutation identifier is required")
            sql = "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND source_event_id = ?"
            params: list[Any] = [tenant, source]
            if mutation_type is not None:
                sql += " AND mutation_type = ?"
                params.append(str(mutation_type).upper())
            sql += " ORDER BY revision DESC LIMIT 1"
            row = conn.execute(sql, tuple(params)).fetchone()
            if row is None:
                compatibility_sql = (
                    "SELECT * FROM canonical_mutations WHERE tenant_id = ? "
                    "AND payload->>'mutation_key' = ?"
                )
                compatibility_params: list[Any] = [tenant, source]
                if mutation_type is not None:
                    compatibility_sql += " AND mutation_type = ?"
                    compatibility_params.append(str(mutation_type).upper())
                compatibility_sql += " ORDER BY revision DESC LIMIT 1"
                row = conn.execute(compatibility_sql, tuple(compatibility_params)).fetchone()
        return _mutation_from_row(row) if row is not None else None

    def load_mutation_projection(
        self, mutation_id: str | uuid.UUID, *, tenant_id: str | None = None
    ) -> dict[str, JsonValue] | None:
        """Return the compact canonical envelope consumed by projection Activities."""

        mutation = self.get_mutation(mutation_id, tenant_id=tenant_id)
        if mutation is None:
            return None
        projection: dict[str, JsonValue] = {
            **mutation.payload,
            **mutation.result,
            "canonical_mutation_id": str(mutation.id),
            "tenant_id": mutation.tenant_id,
            "source_event_id": mutation.source_event_id,
            "mutation_type": mutation.mutation_type,
            "operation": mutation.payload.get("operation") or mutation.mutation_type,
            "revision": mutation.payload.get("revision") or mutation.revision,
            # Mutation execution state and projected resource lifecycle are
            # different domains. Using the generic ``status`` key here caused
            # an APPLIED mutation to be mistaken for a non-ACTIVE claim and
            # silently routed through the relationship-delete path.
            "mutation_status": mutation.status,
        }
        if mutation.result_memory_id is not None:
            projection.setdefault("memory_id", str(mutation.result_memory_id))
        return projection

    # Names used by the Temporal adapter during the canonical cutover.
    get_mutation_projection = load_mutation_projection
    get_canonical_mutation_projection = load_mutation_projection

    @staticmethod
    def _select_mutation_for_update(
        conn: Any,
        *,
        tenant_id: str,
        mutation_id: str | uuid.UUID | None,
        source_event_id: str | None,
        mutation_type: str | None,
    ) -> Any:
        """Select one mutation by stable ID or by its composite idempotency key."""

        if mutation_id is not None:
            try:
                parsed_id = uuid.UUID(str(mutation_id))
            except (TypeError, ValueError, AttributeError):
                parsed_id = None
            if parsed_id is not None:
                return conn.execute(
                    "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND id = ? FOR UPDATE",
                    (tenant_id, parsed_id),
                ).fetchone()
        if source_event_id is None:
            return None
        sql = "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND source_event_id = ?"
        params: list[Any] = [tenant_id, source_event_id]
        if mutation_type is not None:
            sql += " AND mutation_type = ?"
            params.append(str(mutation_type).upper())
        sql += " ORDER BY revision DESC LIMIT 1 FOR UPDATE"
        row = conn.execute(sql, tuple(params)).fetchone()
        if row is not None:
            return row
        compatibility_sql = (
            "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND payload->>'mutation_key' = ?"
        )
        compatibility_params: list[Any] = [tenant_id, source_event_id]
        if mutation_type is not None:
            compatibility_sql += " AND mutation_type = ?"
            compatibility_params.append(str(mutation_type).upper())
        compatibility_sql += " ORDER BY revision DESC LIMIT 1 FOR UPDATE"
        return conn.execute(compatibility_sql, tuple(compatibility_params)).fetchone()

    def complete_mutation(
        self,
        mutation_key: str | None = None,
        *,
        mutation_id: str | uuid.UUID | None = None,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        tenant_id: str | None = None,
        result_memory_id: uuid.UUID | None = None,
        result: Mapping[str, JsonValue] | None = None,
    ) -> CanonicalMutation:
        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            row = self._select_mutation_for_update(
                conn,
                tenant_id=tenant,
                mutation_id=mutation_id,
                source_event_id=source_event_id or mutation_key,
                mutation_type=mutation_type,
            )
            if row is None:
                raise MemoryRepositoryError(
                    f"unknown mutation: {mutation_id or source_event_id or mutation_key}"
                )
            return self._complete_mutation_in_tx(
                conn,
                _mutation_from_row(row),
                result_memory_id=result_memory_id,
                result=result,
            )

    def fail_mutation(
        self,
        mutation_key: str | None = None,
        error: str = "",
        *,
        mutation_id: str | uuid.UUID | None = None,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        tenant_id: str | None = None,
    ) -> CanonicalMutation:
        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            row = self._select_mutation_for_update(
                conn,
                tenant_id=tenant,
                mutation_id=mutation_id,
                source_event_id=source_event_id or mutation_key,
                mutation_type=mutation_type,
            )
            if row is None:
                raise MemoryRepositoryError(
                    f"unknown mutation: {mutation_id or source_event_id or mutation_key}"
                )
            mutation = _mutation_from_row(row)
            self._fail_mutation_in_tx(conn, mutation, error)
            refreshed = conn.execute(
                "SELECT * FROM canonical_mutations WHERE tenant_id = ? AND id = ?",
                (tenant, mutation.id),
            ).fetchone()
            return _mutation_from_row(refreshed)

    def apply_mutation(
        self,
        *,
        mutation_key: str | None = None,
        operation: str | None = None,
        callback: CanonicalMutationCallback,
        tenant_id: str | None = None,
        source_event_id: str | None = None,
        mutation_type: str | None = None,
        event_id: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> Any:
        """Run a caller-supplied canonical mutation in one transaction.

        The callback receives the transaction connection and mutation record.
        An already-applied mutation returns its recorded result when possible;
        domain-specific convenience methods below provide typed replay paths.
        """

        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            mutation, created = self._begin_mutation_in_tx(
                conn,
                tenant_id=tenant,
                source_event_id=source_event_id,
                mutation_type=mutation_type,
                mutation_key=mutation_key,
                operation=operation,
                event_id=event_id,
                payload=payload,
            )
            if not created and mutation.status == "APPLIED":
                return mutation.result
            result = callback(conn, mutation)
            result_memory_id = getattr(result, "id", None)
            result_payload = result if isinstance(result, Mapping) else {"value": _json_any(result)}
            self._complete_mutation_in_tx(
                conn,
                mutation,
                result_memory_id=result_memory_id
                if isinstance(result_memory_id, uuid.UUID)
                else None,
                result=result_payload,
            )
            return result

    def commit_conversational_event(
        self,
        *,
        event_id: str,
        gate: Mapping[str, JsonValue] | None = None,
        extraction: Mapping[str, JsonValue] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, JsonValue]:
        """Commit one gated conversation into canonical PostgreSQL state.

        This is the bounded V2 conversational write boundary.  The caller is
        expected to have run the model gate and extractor in separate
        Temporal Activities; this method only consumes their durable results.
        Every canonical row, conflict transition, evidence record, and
        projection dispatch is written through one outer PostgreSQL
        transaction.  Neo4j and the filesystem are deliberately absent from
        this method.

        ``event_id`` is the existing event-ledger identity and is the
        idempotency key for the complete conversation.  Child projection
        mutations use deterministic aggregate/revision identities, so a retry
        after a worker crash cannot create a second memory, claim, or
        projection dispatch.
        """

        event_key = str(event_id).strip()
        if not event_key:
            raise ValueError("event_id must not be empty")
        tenant = self._tenant(tenant_id)

        def artifact_payload(conn: Any, artifact_key: str) -> dict[str, JsonValue] | None:
            row = conn.execute(
                "SELECT payload FROM ingest_artifacts WHERE tenant_id = ? AND event_id = ? "
                "AND artifact_type = 'EXTRACTION' AND artifact_key = ?",
                (tenant, event_key, artifact_key),
            ).fetchone()
            if row is None:
                return None
            payload = _json_object(row.get("payload"))
            return payload or None

        def mapping_value(item: Mapping[str, JsonValue], *keys: str) -> JsonValue:
            for key in keys:
                if key in item and item[key] is not None:
                    return item[key]
            return None

        def ref_parts(value: JsonValue) -> tuple[str | None, str | None]:
            if isinstance(value, Mapping):
                reference = mapping_value(
                    value,
                    "id",
                    "memory_id",
                    "entity_id",
                    "uuid",
                )
                name = mapping_value(value, "name", "canonical_name", "label", "value")
                return (
                    str(reference).strip() if reference is not None else None,
                    str(name).strip() if name is not None else None,
                )
            if value is None:
                return None, None
            # A plain extractor string is an entity name/alias, not a UUID.
            # Stable identity is accepted only from an explicit *_id field or
            # a structured object carrying an ID.
            return None, str(value).strip()

        def datetime_value(value: JsonValue) -> datetime | None:
            if value is None:
                return None
            return _datetime(value)

        def complete_ingest_dispatch(conn: Any) -> None:
            # The INGEST dispatch is the workflow trigger, while projection
            # dispatches are created below.  Once this commit owns the event
            # terminal state, close that trigger in the same transaction so a
            # dispatcher crash cannot start an already-applied ingest again.
            conn.execute(
                "UPDATE workflow_dispatches SET status = 'COMPLETE', "
                "completed_at = CURRENT_TIMESTAMP, claim_token = NULL, "
                "lease_owner = NULL, claimed_until = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? "
                "AND workflow_type = 'INGEST' AND aggregate_id = ? "
                "AND status IN ('PENDING', 'DISPATCHING', 'STARTED')",
                (tenant, event_key),
            )

        with self._transaction(tenant) as conn:
            event_row = conn.execute(
                "SELECT * FROM events WHERE tenant_id = ? AND event_id = ? FOR UPDATE",
                (tenant, event_key),
            ).fetchone()
            if event_row is None:
                raise MemoryNotFoundError(f"event {event_key} not found for tenant {tenant}")
            event_payload = _json_object(event_row.get("payload"))
            gate_payload = dict(gate or artifact_payload(conn, "gate-v1") or {})
            if "store" not in gate_payload:
                raise ValueError("canonical gate result is missing store")

            mutation, created = self._begin_mutation_in_tx(
                conn,
                tenant_id=tenant,
                source_event_id=event_key,
                mutation_type="CONVERSATIONAL_INGEST",
                payload={
                    "event_id": event_key,
                    "session_id": event_row.get("session_id"),
                    "gate_artifact_key": "gate-v1",
                    "extraction_artifact_key": "extract-v1",
                },
            )
            if not created and mutation.status == "APPLIED":
                replay_result = dict(mutation.result)
                replay_result.setdefault("status", "COMPLETE")
                conn.execute(
                    "UPDATE events SET status = ?, processed_at = COALESCE(processed_at, CURRENT_TIMESTAMP), "
                    "error_message = NULL WHERE tenant_id = ? AND event_id = ?",
                    (str(replay_result["status"]), tenant, event_key),
                )
                complete_ingest_dispatch(conn)
                replay_result.setdefault("event_id", event_key)
                replay_result["replayed"] = True
                replay_result["mutation_id"] = str(mutation.id)
                return replay_result
            if not bool(gate_payload.get("store")):
                skip_result: dict[str, JsonValue] = {
                    "event_id": event_key,
                    "status": "GATED_SKIP",
                    "reason": str(gate_payload.get("reason") or "gate declined"),
                }
                self._complete_mutation_in_tx(conn, mutation, result=skip_result)
                conn.execute(
                    "UPDATE events SET status = 'GATED_SKIP', error_message = ?, "
                    "processed_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND event_id = ?",
                    (str(skip_result["reason"])[:1000], tenant, event_key),
                )
                complete_ingest_dispatch(conn)
                skip_result["mutation_id"] = str(mutation.id)
                skip_result["replayed"] = False
                return skip_result

            extraction_payload = dict(extraction or artifact_payload(conn, "extract-v1") or {})
            raw_triplets = extraction_payload.get("triplets")
            if not isinstance(raw_triplets, list):
                raise ValueError("canonical extraction result must contain a triplets list")
            triplets = [dict(item) for item in raw_triplets if isinstance(item, Mapping)]
            if len(triplets) != len(raw_triplets):
                raise ValueError("canonical extraction triplets must be objects")
            turn_pair = event_payload.get("turn_pair")
            if not isinstance(turn_pair, Mapping):
                turn_pair = {}
            user_turn = turn_pair.get("user")
            assistant_turn = turn_pair.get("assistant")
            user_text = (
                str(user_turn.get("content") or "")
                if isinstance(user_turn, Mapping)
                else str(user_turn or "")
            )
            assistant_text = (
                str(assistant_turn.get("content") or "")
                if isinstance(assistant_turn, Mapping)
                else str(assistant_turn or "")
            )
            resolved_text = str(extraction_payload.get("resolved_text") or "").strip()
            body = resolved_text or f"User: {user_text}\nAssistant: {assistant_text}".strip()
            if not body:
                raise ValueError("canonical extraction produced an empty conversation body")
            abstract = str(extraction_payload.get("l0_abstract") or body[:200])
            extractor_version = str(extraction_payload.get("extractor_version") or "core-v2")
            # A fact is asserted when it was said, not when it was stored; the
            # storage time is the fallback for clients that send no timestamp.
            asserted_at = _turn_said_at(user_turn, assistant_turn) or _datetime(
                event_row.get("created_at"), default=datetime.now(timezone.utc)
            )

            bound = MemoryRepository(
                _BoundTransactionStore(conn, projection_task_queue=self.projection_task_queue),
                default_tenant_id=self.default_tenant_id,
            )

            def contextual_name_candidates(clean_name: str) -> list[MemoryNode]:
                """Resolve a unique short/full-name expansion inside one session.

                Exact aliases remain the primary identity rule.  This fallback
                only relates ``Atul`` and ``Atul Singh`` when exactly one
                session-local entity has the complementary alias form.  The
                session boundary and ambiguity check prevent tenant-wide
                first-name guesses from silently joining unrelated people.
                """

                session_id = event_row.get("session_id")
                normalized = normalize_alias(clean_name)
                parts = normalized.split()
                if not session_id or not parts or any(
                    not part.replace("-", "").isalnum() for part in parts
                ):
                    return []
                if len(parts) == 1:
                    alias_clause = "a.normalized_alias LIKE ?"
                    alias_value = f"{parts[0]} %"
                else:
                    alias_clause = "a.normalized_alias = ?"
                    alias_value = parts[0]
                rows = conn.execute(
                    "SELECT DISTINCT n.* FROM memory_nodes n "
                    "JOIN entity_aliases a ON a.tenant_id = n.tenant_id "
                    "AND a.entity_id = n.id "
                    "WHERE n.tenant_id = ? AND n.memory_type = 'ENTITY' "
                    "AND n.status = 'ACTIVE' AND "
                    + alias_clause
                    + " AND (n.origin_event_id = ? OR EXISTS ("
                    "SELECT 1 FROM memory_evidence e "
                    "WHERE e.tenant_id = n.tenant_id AND e.memory_id = n.id "
                    "AND e.source_session_id = ?)) "
                    "ORDER BY n.canonical_name NULLS LAST, n.id LIMIT 3",
                    (tenant, alias_value, event_key, str(session_id)),
                ).fetchall()
                return [_node_from_row(row) for row in rows]

            def resolve_entity(
                value: JsonValue,
                *,
                explicit_id: JsonValue = None,
                role: str,
            ) -> MemoryNode:
                ref, name = ref_parts(value)
                exact_ref = explicit_id if explicit_id is not None else ref
                exact_id = _uuid(exact_ref)
                if exact_id is not None:
                    node = bound._require_node(conn, tenant, exact_id)
                    if node.memory_type != "ENTITY":
                        raise InvalidClaimError(
                            f"{role} reference {exact_id} is not an ENTITY memory"
                        )
                else:
                    raw_name = str(name or ref or "").strip()
                    clean_name = re.sub(r"_+", " ", raw_name).strip()
                    if not clean_name:
                        raise InvalidClaimError(f"{role} entity name is empty")
                    candidates = bound.resolve_entity_candidates(
                        clean_name, tenant_id=tenant, limit=3
                    )
                    if len(candidates) > 1:
                        raise MemoryRepositoryError(
                            f"ambiguous {role} entity reference: {clean_name}"
                        )
                    if candidates:
                        node = candidates[0]
                    else:
                        contextual = contextual_name_candidates(clean_name)
                        if len(contextual) > 1:
                            raise MemoryRepositoryError(
                                f"ambiguous session-local {role} entity reference: {clean_name}"
                            )
                        if contextual:
                            node = contextual[0]
                        else:
                            identity = _deterministic_uuid(
                                _MUTATION_NAMESPACE,
                                tenant,
                                "conversation-entity",
                                normalize_alias(clean_name),
                            )
                            node = bound.create_memory(
                                memory_type="ENTITY",
                                canonical_uri=f"mem://memory/{identity}",
                                canonical_name=clean_name,
                                memory_id=identity,
                                origin_event_id=event_key,
                                metadata={"source": "canonical-conversation"},
                                tenant_id=tenant,
                                emit_projection=False,
                            )
                display_name = re.sub(r"_+", " ", str(name or "")).strip()
                current_name = str(node.canonical_name or "")

                display_normalized = normalize_alias(display_name)
                current_normalized = normalize_alias(current_name)
                display_tokens = display_normalized.split()
                current_tokens = current_normalized.split()
                improves_spelling = (
                    display_normalized == current_normalized
                    and sum(1 for token in display_name.split() if token[:1].isupper())
                    > sum(1 for token in current_name.split() if token[:1].isupper())
                )
                expands_short_name = (
                    len(current_tokens) == 1
                    and len(display_tokens) == 2
                    and current_tokens[0] == display_tokens[0]
                )
                if display_name and (improves_spelling or expands_short_name):
                    upgraded = conn.execute(
                        "UPDATE memory_nodes SET canonical_name = ?, "
                        "revision = revision + 1, updated_at = CURRENT_TIMESTAMP "
                        "WHERE tenant_id = ? AND id = ? RETURNING *",
                        (display_name, tenant, node.id),
                    ).fetchone()
                    if upgraded is not None:
                        node = _node_from_row(upgraded)
                if name:
                    bound.add_alias(
                        entity_id=node.id,
                        alias=name,
                        source_event_id=event_key,
                        metadata={"source": "canonical-conversation", "role": role},
                        tenant_id=tenant,
                    )
                return node

            episode_identity = _deterministic_uuid(
                _MUTATION_NAMESPACE, tenant, event_key, "conversation-episode"
            )
            episode = bound.create_memory(
                memory_type="EPISODE",
                canonical_uri=f"mem://memory/{episode_identity}",
                canonical_name=f"Conversation {event_key}",
                memory_id=episode_identity,
                origin_event_id=event_key,
                metadata={
                    "source": "canonical-conversation",
                    "session_id": event_row.get("session_id"),
                },
                tenant_id=tenant,
                emit_projection=False,
            )
            version = bound.append_version(
                episode.id,
                body=body,
                abstract=abstract,
                asserted_at=asserted_at,
                source_event_id=event_key,
                provenance={
                    "extractor": "core",
                    "extractor_version": extractor_version,
                    "event_id": event_key,
                },
                metadata={"source": "canonical-conversation"},
                tenant_id=tenant,
                emit_projection=False,
            )
            source_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            bound.record_evidence(
                memory_id=episode.id,
                source_event_id=event_key,
                source_session_id=event_row.get("session_id"),
                extractor="core",
                extractor_version=extractor_version,
                source_span={"kind": "conversation", "event_id": event_key},
                source_text_hash=source_hash,
                metadata={"stage": "episode"},
                idempotency_key=f"{event_key}:episode",
                tenant_id=tenant,
            )

            affected_nodes: dict[uuid.UUID, MemoryNode] = {episode.id: episode}
            claims: list[MemoryClaim] = []
            entity_ids: set[uuid.UUID] = set()
            for index, triplet in enumerate(triplets):
                raw_predicate = mapping_value(
                    triplet, "raw_predicate", "predicate", "relation", "property"
                )
                if raw_predicate is None or not str(raw_predicate).strip():
                    raise InvalidClaimError(f"triplet {index} is missing predicate/relation")
                subject_value = mapping_value(triplet, "subject", "subject_name", "subject_entity")
                subject_ref = mapping_value(
                    triplet, "subject_id", "subject_memory_id", "subject_entity_id"
                )
                subject = resolve_entity(
                    subject_value,
                    explicit_id=subject_ref,
                    role=f"triplet {index} subject",
                )
                entity_ids.add(subject.id)
                affected_nodes[subject.id] = subject

                object_value = mapping_value(
                    triplet,
                    "object_value",
                    "object",
                    "value",
                    "object_name",
                )
                object_ref = mapping_value(
                    triplet, "object_entity_id", "object_memory_id", "object_id"
                )
                if object_value is None and object_ref is None:
                    raise InvalidClaimError(f"triplet {index} is missing object")
                object_for_policy = object_value
                if isinstance(object_for_policy, Mapping):
                    object_for_policy = mapping_value(
                        object_for_policy, "value", "name", "label", "text", "id"
                    )
                if object_for_policy is None:
                    object_for_policy = object_ref
                is_pretyped = bool(triplet.get("typed"))
                if is_pretyped:
                    canonical_predicate = str(triplet.get("predicate") or "").strip()
                    canonical_object_type = str(triplet.get("object_type") or "").upper()
                    if not canonical_predicate or canonical_object_type not in _OBJECT_TYPES:
                        raise InvalidClaimError(
                            f"triplet {index} has an invalid persisted typing decision"
                        )
                    policy_version = str(triplet.get("predicate_policy_version") or "").strip()
                    cardinality = str(triplet.get("cardinality") or "").upper()
                    conflict_policy = str(triplet.get("conflict_policy") or "").upper()
                    if not policy_version or cardinality not in {"SINGLE", "MULTIPLE"}:
                        raise InvalidClaimError(
                            f"triplet {index} is missing persisted predicate policy metadata"
                        )
                else:
                    # Direct repository callers remain supported, but the
                    # production Temporal path persists this decision in the
                    # typed-v1 artifact before opening the canonical commit.
                    normalized = normalize_claim_value(str(raw_predicate), object_for_policy)
                    canonical_predicate = normalized.predicate
                    canonical_object_type = normalized.object_type.value
                    policy_version = normalized.policy_version
                    cardinality = normalized.policy.cardinality.value
                    conflict_policy = normalized.policy.conflict_policy.value
                object_entity: MemoryNode | None = None
                if canonical_object_type == "ENTITY":
                    object_entity = resolve_entity(
                        object_value if object_value is not None else object_ref,
                        explicit_id=object_ref,
                        role=f"triplet {index} object",
                    )
                    entity_ids.add(object_entity.id)
                    affected_nodes[object_entity.id] = object_entity
                    normalized_object_value = None
                else:
                    normalized_object_value = object_for_policy if is_pretyped else normalized.value

                valid_from_raw = mapping_value(
                    triplet,
                    "valid_from",
                    "effective_from",
                    "effective_date",
                    "start_date",
                )
                valid_until_raw = mapping_value(
                    triplet, "valid_until", "effective_until", "end_date"
                )
                valid_from = datetime_value(valid_from_raw)
                valid_until = datetime_value(valid_until_raw)
                confidence_value = triplet.get("confidence")
                confidence = float(confidence_value) if confidence_value is not None else None
                claim_id = _deterministic_uuid(
                    _CLAIM_NAMESPACE, tenant, event_key, "triplet", index
                )
                claim = bound.add_claim(
                    subject_id=subject.id,
                    predicate=canonical_predicate,
                    object_entity_id=object_entity.id if object_entity is not None else None,
                    object_value=normalized_object_value,
                    object_type=cast(ObjectType, canonical_object_type),
                    confidence=confidence,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    asserted_at=asserted_at,
                    source_event_id=event_key,
                    source_version_id=version.id,
                    source_triplet_index=index,
                    normalized_object_hash=None,
                    metadata={
                        "raw_predicate": str(raw_predicate),
                        "predicate_policy_version": policy_version,
                        "source": "canonical-conversation",
                    },
                    claim_id=claim_id,
                    tenant_id=tenant,
                    emit_projection=False,
                )
                active = bound.get_current_claims(
                    subject.id,
                    predicate=canonical_predicate,
                    at=valid_from,
                    tenant_id=tenant,
                )
                conflicting = [
                    item
                    for item in active
                    if item.id != claim.id
                    and item.normalized_object_hash != claim.normalized_object_hash
                ]
                if conflicting and cardinality == "SINGLE":
                    for previous in conflicting:
                        claim = bound.supersede_claim(
                            previous.id,
                            claim.id,
                            tenant_id=tenant,
                            replacement_status="ACTIVE",
                            emit_projection=False,
                        )
                elif conflicting and conflict_policy == "FLAG":
                    conflict_row = conn.execute(
                        "UPDATE memory_claims SET status = 'CONFLICTING', "
                        "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ? RETURNING *",
                        (tenant, claim.id),
                    ).fetchone()
                    if conflict_row is not None:
                        claim = _claim_from_row(conflict_row)
                        bound._bump_node_revision(conn, tenant, subject.id)
                claims.append(claim)
                bound.record_evidence(
                    memory_id=subject.id,
                    claim_id=claim.id,
                    source_event_id=event_key,
                    source_session_id=event_row.get("session_id"),
                    extractor="core",
                    extractor_version=extractor_version,
                    confidence=confidence,
                    source_span=triplet.get("source_span") or {"triplet_index": index},
                    source_text_hash=source_hash,
                    metadata={
                        "predicate_policy_version": policy_version,
                        "raw_predicate": str(raw_predicate),
                    },
                    idempotency_key=f"{event_key}:claim:{index}",
                    tenant_id=tenant,
                )

            # Keep a durable, idempotent version/evidence trail for every
            # entity touched by this conversation as well as for the episode.
            # Entity identity is stable across re-ingest; only the immutable
            # version and evidence rows advance.  This is intentionally done
            # before dispatch creation so every projection observes the final
            # node revision from this transaction.
            #
            # Conversation episodes are the source/document anchor in the
            # semantic graph.  Persist an explicit hierarchy edge for every
            # entity mentioned by the event so the graph can be explored from
            # the episode instead of leaving the canonical nodes disconnected.
            hierarchy_ids: list[str] = []
            dispatch_ids: list[str] = []
            for position, entity_id in enumerate(sorted(entity_ids, key=str)):
                edge = bound.add_hierarchy(
                    parent_id=episode.id,
                    child_id=entity_id,
                    position=position,
                    metadata={
                        "source": "canonical-conversation",
                        "relation": "MENTIONS",
                        "source_event_id": event_key,
                    },
                    tenant_id=tenant,
                    emit_projection=True,
                )
                hierarchy_ids.append(str(edge.id))
                hierarchy_dispatch = conn.execute(
                    "SELECT dispatch_id FROM workflow_dispatches "
                    "WHERE tenant_id = ? AND workflow_type = 'PROJECTION' "
                    "AND payload->>'hierarchy_id' = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (tenant, str(edge.id)),
                ).fetchone()
                if hierarchy_dispatch is not None:
                    dispatch_ids.append(str(hierarchy_dispatch["dispatch_id"]))

            entity_version_ids: list[str] = []
            entity_body = resolved_text or body
            for entity_id in sorted(entity_ids, key=str):
                entity = bound._require_node(conn, tenant, entity_id)
                entity_version = bound.append_version(
                    entity.id,
                    body=entity_body,
                    abstract=abstract,
                    asserted_at=asserted_at,
                    source_event_id=event_key,
                    provenance={
                        "extractor": "core",
                        "extractor_version": extractor_version,
                        "event_id": event_key,
                        "memory_type": "ENTITY",
                    },
                    metadata={
                        "source": "canonical-conversation",
                        "canonical_name": entity.canonical_name,
                    },
                    version_id=_deterministic_uuid(
                        _MUTATION_NAMESPACE, tenant, event_key, "entity-version", entity.id
                    ),
                    tenant_id=tenant,
                    emit_projection=False,
                )
                entity_version_ids.append(str(entity_version.id))
                bound.record_evidence(
                    memory_id=entity.id,
                    source_event_id=event_key,
                    source_session_id=event_row.get("session_id"),
                    extractor="core",
                    extractor_version=extractor_version,
                    source_span={
                        "kind": "entity",
                        "memory_id": str(entity.id),
                    },
                    source_text_hash=source_hash,
                    metadata={"stage": "entity"},
                    idempotency_key=f"{event_key}:entity:{entity.id}",
                    tenant_id=tenant,
                )

            for node_id in sorted(affected_nodes, key=str):
                node = bound._require_node(conn, tenant, node_id)
                dispatch = bound._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(node.id),
                    operation="UPSERT_NODE",
                    revision=node.revision,
                    payload={
                        "memory_id": str(node.id),
                        "aggregate_type": "MEMORY",
                        "operation": "UPSERT_NODE",
                        "revision": node.revision,
                    },
                )
                dispatch_ids.append(dispatch.dispatch_id)
            for claim in claims:
                if claim.object_entity_id is None:
                    continue
                dispatch = bound._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="CLAIM",
                    aggregate_id=str(claim.id),
                    operation="UPSERT_CLAIM",
                    revision=bound._require_node(conn, tenant, claim.subject_id).revision,
                    payload={
                        "claim_id": str(claim.id),
                        "subject_memory_id": str(claim.subject_id),
                        "object_memory_id": str(claim.object_entity_id),
                        "predicate": claim.predicate,
                        "aggregate_type": "CLAIM",
                        "operation": "UPSERT_CLAIM",
                        "revision": bound._require_node(conn, tenant, claim.subject_id).revision,
                    },
                )
                dispatch_ids.append(dispatch.dispatch_id)

            commit_result: dict[str, JsonValue] = {
                "event_id": event_key,
                "status": "COMPLETE",
                "episode_id": str(episode.id),
                "version_ids": [str(version.id), *entity_version_ids],
                "entity_ids": [str(item) for item in sorted(entity_ids, key=str)],
                "claim_ids": [str(item.id) for item in claims],
                "hierarchy_ids": hierarchy_ids,
                "dispatch_ids": dispatch_ids,
                "replayed": False,
            }
            self._complete_mutation_in_tx(
                conn,
                mutation,
                result_memory_id=episode.id,
                result=commit_result,
            )
            conn.execute(
                "UPDATE events SET status = 'COMPLETE', error_message = NULL, "
                "processed_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND event_id = ?",
                (tenant, event_key),
            )
            complete_ingest_dispatch(conn)
            commit_result["mutation_id"] = str(mutation.id)
            return commit_result

    # Short aliases make the activity boundary easy to discover while keeping
    # the descriptive method as the canonical API.
    commit_conversation = commit_conversational_event
    commit_canonical_event = commit_conversational_event

    # ------------------------------------------------------------------
    # Memory nodes and immutable versions
    # ------------------------------------------------------------------

    def create_memory(
        self,
        *,
        memory_type: str,
        canonical_uri: str | None = None,
        canonical_name: str | None = None,
        status: str = "ACTIVE",
        memory_id: uuid.UUID | None = None,
        origin_event_id: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
        mutation_operation: str = "CREATE_MEMORY",
        emit_projection: bool = True,
    ) -> MemoryNode:
        tenant = self._tenant(tenant_id)
        memory_kind = str(memory_type).upper()
        if memory_kind not in _MEMORY_TYPES:
            raise ValueError(f"unsupported memory_type: {memory_type!r}")
        node_status = str(status).upper()
        if node_status not in _NODE_STATUSES:
            raise ValueError(f"unsupported memory status: {status!r}")
        identity_key = mutation_key or origin_event_id
        if identity_key:
            chosen_id = memory_id or _deterministic_uuid(
                _MUTATION_NAMESPACE, tenant, identity_key, "memory"
            )
        else:
            chosen_id = memory_id or uuid.uuid4()
        uri = _validate_uri(canonical_uri or f"mem://memory/{chosen_id}")

        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    mutation_key=mutation_key,
                    operation=mutation_operation,
                    event_id=origin_event_id,
                    payload={
                        "memory_type": memory_kind,
                        "canonical_uri": uri,
                        "canonical_name": canonical_name,
                    },
                )
                if not created and mutation.status == "APPLIED" and mutation.result_memory_id:
                    existing = self._require_node(conn, tenant, mutation.result_memory_id)
                    return existing
            row = conn.execute(
                "INSERT INTO memory_nodes "
                "(id, tenant_id, memory_type, canonical_name, canonical_uri, status, "
                "origin_event_id, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (tenant_id, canonical_uri) DO NOTHING RETURNING *",
                (
                    chosen_id,
                    tenant,
                    memory_kind,
                    canonical_name,
                    uri,
                    node_status,
                    origin_event_id,
                    _json(dict(metadata or {}), default={}),
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM memory_nodes WHERE tenant_id = ? AND canonical_uri = ? FOR UPDATE",
                    (tenant, uri),
                ).fetchone()
                if row is None:
                    raise MemoryRepositoryError(
                        f"memory insert conflict without existing row: {uri}"
                    )
                existing = _node_from_row(row)
                if existing.id != chosen_id:
                    raise MemoryRepositoryError(
                        f"canonical URI already belongs to another memory: {tenant}:{uri}"
                    )
            node = _node_from_row(row)
            self._add_uri_alias_in_tx(
                conn,
                tenant_id=tenant,
                memory_id=node.id,
                uri=uri,
                alias_type="CANONICAL",
                metadata={"source": "canonical-create"},
            )
            if emit_projection:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(node.id),
                    operation="UPSERT_NODE",
                    revision=node.revision,
                    payload={"memory_id": str(node.id), "memory_type": node.memory_type},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=node.id,
                    result={"memory_id": str(node.id)},
                )
            return node

    def get_memory(
        self, memory_id: uuid.UUID, *, tenant_id: str | None = None
    ) -> MemoryNode | None:
        tenant = self._tenant(tenant_id)
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_nodes WHERE tenant_id = ? AND id = ?",
                (tenant, memory_id),
            )
            .fetchone()
        )
        return _node_from_row(row) if row is not None else None

    @staticmethod
    def _encode_memory_cursor(node: MemoryNode) -> str:
        value = json.dumps(
            {"updated_at": node.updated_at.isoformat(), "id": str(node.id)},
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_memory_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
        try:
            padded = str(cursor) + "=" * (-len(str(cursor)) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
            updated_at = _datetime(payload["updated_at"])
            memory_id = uuid.UUID(str(payload["id"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
            raise ValueError("invalid memory cursor") from err
        if updated_at is None:
            raise ValueError("invalid memory cursor timestamp")
        return updated_at, memory_id

    def list_memories(
        self,
        tenant_id: str,
        prefix: str | None = None,
        status: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> MemoryPage:
        """List canonical memories using tenant-scoped keyset pagination."""

        tenant = self._tenant(tenant_id)
        if limit <= 0 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        clauses = ["n.tenant_id = ?"]
        params: list[Any] = [tenant]
        if prefix:
            pattern = str(prefix).rstrip("%") + "%"
            # URI aliases are labels for the same stable memory identity. A
            # project-list request using ``mem://projects/...`` must still
            # return the canonical ``mem://memory/{uuid}`` node.
            clauses.append(
                "(n.canonical_uri LIKE ? OR EXISTS (SELECT 1 FROM memory_uri_aliases a "
                "WHERE a.tenant_id = n.tenant_id AND a.memory_id = n.id AND a.uri LIKE ?))"
            )
            params.extend([pattern, pattern])
        if status is not None:
            normalized_status = str(status).upper()
            if normalized_status not in _NODE_STATUSES:
                raise ValueError(f"unsupported memory status: {status!r}")
            clauses.append("n.status = ?")
            params.append(normalized_status)
        if cursor:
            updated_at, memory_id = self._decode_memory_cursor(cursor)
            clauses.append("(n.updated_at, n.id) < (?, ?)")
            params.extend([updated_at, memory_id])
        params.append(limit)
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.* FROM memory_nodes n WHERE "
                + " AND ".join(clauses)
                + " ORDER BY n.updated_at DESC, n.id DESC LIMIT ?",
                tuple(params),
            )
            .fetchall()
        )
        nodes = tuple(_node_from_row(row) for row in rows)
        next_cursor = self._encode_memory_cursor(nodes[-1]) if len(nodes) == limit else None
        return MemoryPage(nodes=nodes, next_cursor=next_cursor)

    def resolve_memory_ref(
        self,
        reference: str | uuid.UUID,
        *,
        tenant_id: str | None = None,
        include_historical: bool = False,
    ) -> MemoryNode | None:
        """Resolve a stable UUID, canonical/legacy URI, alias, or exact name.

        Ambiguous names and aliases intentionally return ``None`` rather than
        selecting an arbitrary identity.  Callers that need to disambiguate
        can use :meth:`resolve_entity_candidates` and apply their own policy.
        """

        tenant = self._tenant(tenant_id)
        if isinstance(reference, uuid.UUID):
            node = self.get_memory(reference, tenant_id=tenant)
            if node is None or (not include_historical and node.status != "ACTIVE"):
                return None
            return node
        text = str(reference).strip()
        if not text:
            return None
        try:
            parsed = uuid.UUID(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            return self.resolve_memory_ref(
                parsed, tenant_id=tenant, include_historical=include_historical
            )
        if text.startswith("mem://"):
            node = self.resolve_uri(text, tenant_id=tenant)
            if node is None or (not include_historical and node.status != "ACTIVE"):
                return None
            return node
        candidates = self.resolve_entity_candidates(
            text, tenant_id=tenant, include_historical=include_historical, limit=2
        )
        return candidates[0] if len(candidates) == 1 else None

    def resolve_entity_candidates(
        self,
        alias: str,
        *,
        tenant_id: str | None = None,
        include_historical: bool = False,
        limit: int = 20,
    ) -> list[MemoryNode]:
        """Return all tenant-local identities matching an alias/name.

        The query is intentionally deterministic and does not collapse an
        ambiguous alias.  Every result remains tenant-scoped and lifecycle
        filtered before the caller performs any model-assisted resolution.
        """

        if limit <= 0:
            return []
        tenant = self._tenant(tenant_id)
        normalized = normalize_alias(alias)
        status_clause = "" if include_historical else " AND n.status = 'ACTIVE'"
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.* FROM memory_nodes n "
                "WHERE n.tenant_id = ? AND n.memory_type = 'ENTITY'"
                + status_clause
                + " AND (EXISTS (SELECT 1 FROM entity_aliases a "
                "WHERE a.tenant_id = n.tenant_id AND a.entity_id = n.id "
                "AND a.normalized_alias = ?) OR lower(regexp_replace("
                "btrim(COALESCE(n.canonical_name, '')), '\\s+', ' ', 'g')) = ?) "
                "ORDER BY n.canonical_name NULLS LAST, n.id LIMIT ?",
                (tenant, normalized, normalized, limit),
            )
            .fetchall()
        )
        return [_node_from_row(row) for row in rows]

    def create_or_reuse_entity(
        self,
        name: str,
        *,
        aliases: Sequence[str] = (),
        canonical_uri: str | None = None,
        origin_event_id: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
    ) -> MemoryNode:
        """Resolve one exact entity or create a stable canonical identity.

        An ambiguous existing alias is a hard error.  This prevents entity
        linking from silently attaching a claim to whichever row happened to
        sort first.
        """

        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("entity name must not be empty")
        tenant = self._tenant(tenant_id)
        candidates = self.resolve_entity_candidates(clean_name, tenant_id=tenant, limit=3)
        if len(candidates) > 1:
            raise MemoryRepositoryError(f"ambiguous entity reference: {clean_name}")
        if candidates:
            entity = candidates[0]
        else:
            entity = self.create_memory(
                memory_type="ENTITY",
                canonical_uri=canonical_uri,
                canonical_name=clean_name,
                origin_event_id=origin_event_id,
                metadata=metadata,
                tenant_id=tenant,
                mutation_key=mutation_key,
            )
        all_aliases = [clean_name, *[str(value) for value in aliases]]
        for alias in dict.fromkeys(value.strip() for value in all_aliases if value.strip()):
            self.add_alias(
                entity_id=entity.id,
                alias=alias,
                source_event_id=origin_event_id,
                tenant_id=tenant,
            )
        return entity

    def append_version(
        self,
        memory_id: uuid.UUID,
        *,
        body: str,
        abstract: str | None = None,
        asserted_at: datetime | None = None,
        valid_from: datetime | date | None = None,
        valid_until: datetime | date | None = None,
        source_event_id: str | None = None,
        provenance: Mapping[str, JsonValue] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        version_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
        emit_projection: bool = True,
    ) -> MemoryVersion:
        tenant = self._tenant(tenant_id)
        from_value = _datetime(valid_from)
        until_value = _datetime(valid_until)
        if from_value and until_value and until_value <= from_value:
            raise ValueError("valid_until must be after valid_from")
        chosen_id = version_id or (
            _deterministic_uuid(_MUTATION_NAMESPACE, tenant, mutation_key, "version")
            if mutation_key
            else uuid.uuid4()
        )
        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    mutation_key=mutation_key,
                    operation="APPEND_VERSION",
                    event_id=source_event_id,
                    payload={"memory_id": str(memory_id)},
                )
                if not created and mutation.status == "APPLIED":
                    result_id = mutation.result.get("version_id")
                    if result_id:
                        row = conn.execute(
                            "SELECT * FROM memory_versions WHERE tenant_id = ? AND id = ?",
                            (tenant, uuid.UUID(str(result_id))),
                        ).fetchone()
                        if row is not None:
                            return _version_from_row(row)
            self._require_node(conn, tenant, memory_id, lock=True)
            if source_event_id:
                existing = conn.execute(
                    "SELECT * FROM memory_versions WHERE tenant_id = ? AND memory_id = ? "
                    "AND source_event_id = ?",
                    (tenant, memory_id, source_event_id),
                ).fetchone()
                if existing is not None:
                    version = _version_from_row(existing)
                    if mutation is not None:
                        self._complete_mutation_in_tx(
                            conn,
                            mutation,
                            result_memory_id=memory_id,
                            result={"version_id": str(version.id), "memory_id": str(memory_id)},
                        )
                    return version
            next_row = conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 AS version_number "
                "FROM memory_versions WHERE tenant_id = ? AND memory_id = ?",
                (tenant, memory_id),
            ).fetchone()
            version_number = int(next_row["version_number"] if next_row else 1)
            row = conn.execute(
                "INSERT INTO memory_versions "
                "(id, tenant_id, memory_id, version_number, body, abstract, asserted_at, "
                "valid_from, valid_until, source_event_id, provenance, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (
                    chosen_id,
                    tenant,
                    memory_id,
                    version_number,
                    str(body),
                    abstract,
                    asserted_at or datetime.now(timezone.utc),
                    from_value,
                    until_value,
                    source_event_id,
                    _json(dict(provenance or {}), default={}),
                    _json(dict(metadata or {}), default={}),
                ),
            ).fetchone()
            if row is None:
                raise MemoryRepositoryError("version insert returned no row")
            version = _version_from_row(row)
            updated = conn.execute(
                "UPDATE memory_nodes SET current_version_id = ?, revision = revision + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ? RETURNING *",
                (version.id, tenant, memory_id),
            ).fetchone()
            if updated is None:
                raise MemoryNotFoundError(f"memory {memory_id} disappeared while appending version")
            if emit_projection:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(memory_id),
                    operation="UPSERT_NODE",
                    revision=int(updated["revision"]),
                    payload={"memory_id": str(memory_id), "version_id": str(version.id)},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=memory_id,
                    result={"memory_id": str(memory_id), "version_id": str(version.id)},
                )
            return version

    def get_versions(
        self, memory_id: uuid.UUID, *, tenant_id: str | None = None
    ) -> list[MemoryVersion]:
        tenant = self._tenant(tenant_id)
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_versions WHERE tenant_id = ? AND memory_id = ? "
                "ORDER BY version_number DESC",
                (tenant, memory_id),
            )
            .fetchall()
        )
        return [_version_from_row(row) for row in rows]

    get_version_history = get_versions

    # ------------------------------------------------------------------
    # Claims, evidence, aliases, hierarchy, and overviews
    # ------------------------------------------------------------------

    def add_claim(
        self,
        claim: TypedClaim | None = None,
        *,
        tenant_id: str | None = None,
        subject_id: uuid.UUID | None = None,
        predicate: str | None = None,
        object_entity_id: uuid.UUID | None = None,
        object_value: JsonValue = None,
        object_type: str | None = None,
        confidence: float | None = None,
        valid_from: datetime | date | None = None,
        valid_until: datetime | date | None = None,
        asserted_at: datetime | None = None,
        source_event_id: str | None = None,
        source_version_id: uuid.UUID | None = None,
        source_triplet_index: int | None = None,
        supersedes_claim_id: uuid.UUID | None = None,
        normalized_object_hash: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        claim_id: uuid.UUID | None = None,
        mutation_key: str | None = None,
        emit_projection: bool = True,
    ) -> MemoryClaim:
        tenant = self._tenant(tenant_id)
        if claim is not None:
            if any(
                value is not None
                for value in (
                    subject_id,
                    predicate,
                    object_entity_id,
                    source_event_id,
                    source_version_id,
                    claim_id,
                )
            ):
                raise TypeError("pass either claim or keyword claim fields, not both")
            subject_id = claim.subject_id
            predicate = claim.predicate
            object_entity_id = claim.object_entity_id
            object_value = claim.object_value
            object_type = claim.object_type
            confidence = claim.confidence
            valid_from = claim.valid_from
            valid_until = claim.valid_until
            asserted_at = claim.asserted_at
            source_event_id = claim.source_event_id
            source_version_id = claim.source_version_id
            source_triplet_index = claim.source_triplet_index
            normalized_object_hash = claim.normalized_object_hash
            metadata = claim.metadata
        if subject_id is None or predicate is None:
            raise TypeError("subject_id and predicate are required")
        normalized_predicate = normalize_predicate(predicate)
        resolved_type: ObjectType
        if object_entity_id is not None:
            resolved_type = "ENTITY"
        else:
            resolved_type = cast(ObjectType, infer_object_type(object_value, hint=object_type))
        candidate = TypedClaim(
            subject_id=subject_id,
            predicate=normalized_predicate,
            object_type=resolved_type,
            object_entity_id=object_entity_id,
            object_value=object_value,
            confidence=confidence,
            valid_from=_datetime(valid_from),
            valid_until=_datetime(valid_until),
            asserted_at=asserted_at,
            source_event_id=source_event_id,
            source_version_id=source_version_id,
            source_triplet_index=source_triplet_index,
            normalized_object_hash=normalized_object_hash,
            metadata=dict(metadata or {}),
        )
        object_hash = candidate.normalized_object_hash or _hash_object(
            candidate.object_type, candidate.object_entity_id, candidate.object_value
        )
        chosen_id = claim_id or (
            _deterministic_uuid(
                _CLAIM_NAMESPACE,
                tenant,
                source_event_id or mutation_key or uuid.uuid4(),
                source_triplet_index if source_triplet_index is not None else object_hash,
                subject_id,
                normalized_predicate,
            )
            if (source_event_id or mutation_key)
            else uuid.uuid4()
        )
        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    # A mutation key is the caller's idempotency identity.
                    # Prefer it over the provenance event when both are
                    # supplied; code-graph ingestion emits one mutation key
                    # per edge, while all edges share one source event.
                    source_event_id=mutation_key,
                    operation="ADD_CLAIM",
                    event_id=source_event_id,
                    payload={
                        "subject_id": str(subject_id),
                        "predicate": normalized_predicate,
                        "object_type": candidate.object_type,
                    },
                )
                if not created and mutation.status == "APPLIED":
                    result_id = mutation.result.get("claim_id")
                    if result_id:
                        existing = conn.execute(
                            "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                            (tenant, uuid.UUID(str(result_id))),
                        ).fetchone()
                        if existing is not None:
                            return _claim_from_row(existing)
            self._require_node(conn, tenant, subject_id)
            if candidate.object_entity_id is not None:
                self._require_node(conn, tenant, candidate.object_entity_id)
            if candidate.source_version_id is not None:
                self._require_version(conn, tenant, candidate.source_version_id)
            if source_event_id is not None and source_triplet_index is not None:
                existing = conn.execute(
                    "SELECT * FROM memory_claims WHERE tenant_id = ? AND source_event_id = ? "
                    "AND source_triplet_index = ?",
                    (tenant, source_event_id, source_triplet_index),
                ).fetchone()
                if existing is not None:
                    result = _claim_from_row(existing)
                    if mutation is not None:
                        self._complete_mutation_in_tx(
                            conn,
                            mutation,
                            result_memory_id=result.subject_id,
                            result={
                                "claim_id": str(result.id),
                                "memory_id": str(result.subject_id),
                            },
                        )
                    return result
            row = conn.execute(
                "INSERT INTO memory_claims "
                "(id, tenant_id, subject_id, predicate, object_entity_id, object_value, object_type, "
                "status, confidence, valid_from, valid_until, asserted_at, source_event_id, "
                "source_version_id, source_triplet_index, supersedes_claim_id, normalized_object_hash, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING RETURNING *",
                (
                    chosen_id,
                    tenant,
                    subject_id,
                    normalized_predicate,
                    candidate.object_entity_id,
                    _json(candidate.object_value, default=None)
                    if candidate.object_entity_id is None
                    else None,
                    candidate.object_type,
                    candidate.confidence,
                    candidate.valid_from,
                    candidate.valid_until,
                    candidate.asserted_at or datetime.now(timezone.utc),
                    source_event_id,
                    source_version_id,
                    source_triplet_index,
                    supersedes_claim_id,
                    object_hash,
                    _json(dict(candidate.metadata), default={}),
                ),
            ).fetchone()
            inserted = row is not None
            if row is None:
                row = conn.execute(
                    "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                    (tenant, chosen_id),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM memory_claims WHERE tenant_id = ? AND subject_id = ? "
                    "AND predicate = ? AND normalized_object_hash = ? "
                    "AND valid_from IS NOT DISTINCT FROM ? "
                    "AND valid_until IS NOT DISTINCT FROM ? "
                    "ORDER BY CASE status WHEN 'ACTIVE' THEN 0 ELSE 1 END, created_at LIMIT 1",
                    (
                        tenant,
                        subject_id,
                        normalized_predicate,
                        object_hash,
                        candidate.valid_from,
                        candidate.valid_until,
                    ),
                ).fetchone()
            if row is None:
                raise MemoryRepositoryError("claim insert returned no row")
            result = _claim_from_row(row)
            updated = (
                self._bump_node_revision(conn, tenant, subject_id)
                if inserted
                else self._require_node(conn, tenant, subject_id).revision
            )
            if emit_projection and inserted:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(subject_id),
                    operation="UPSERT_NODE",
                    revision=updated,
                    payload={"memory_id": str(subject_id), "claim_id": str(result.id)},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="CLAIM",
                    aggregate_id=str(result.id),
                    operation="UPSERT_CLAIM",
                    revision=updated,
                    payload={"memory_id": str(subject_id), "claim_id": str(result.id)},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=subject_id,
                    result={"claim_id": str(result.id), "memory_id": str(subject_id)},
                )
            return result

    def apply_claims(
        self,
        subject_id: uuid.UUID,
        claims: Sequence[TypedClaim | Mapping[str, Any]],
        *,
        source_event_id: str | None = None,
        source_version_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
    ) -> list[MemoryClaim]:
        """Apply typed claims atomically and enforce registered cardinality.

        Predicate normalization and cardinality policy are applied before
        supersession.  A ``SINGLE`` policy closes different active values for
        the same subject/predicate; ``COEXIST`` preserves them; ``FLAG`` keeps
        the new assertion but marks it ``CONFLICTING`` when another value is
        active.  Unknown predicates use the registry's conservative policy.
        """

        tenant = self._tenant(tenant_id)
        if not claims:
            return []
        prepared: list[TypedClaim] = []
        for index, raw in enumerate(claims):
            if isinstance(raw, TypedClaim):
                candidate = raw
            elif isinstance(raw, Mapping):
                raw_subject = _uuid(raw.get("subject_id")) or subject_id
                entity_id = _uuid(raw.get("object_entity_id"))
                candidate = typed_claim(
                    subject_id=raw_subject,
                    predicate=str(raw.get("predicate") or ""),
                    object_entity_id=entity_id,
                    object_value=raw.get("object_value"),
                    object_type=raw.get("object_type"),
                    confidence=raw.get("confidence"),
                    valid_from=_datetime(raw.get("valid_from")),
                    valid_until=_datetime(raw.get("valid_until")),
                    asserted_at=_datetime(raw.get("asserted_at")),
                    source_event_id=raw.get("source_event_id"),
                    source_version_id=_uuid(raw.get("source_version_id")),
                    source_triplet_index=raw.get("source_triplet_index"),
                    normalized_object_hash=raw.get("normalized_object_hash"),
                    metadata=_json_object(raw.get("metadata")),
                )
            else:
                raise TypeError("claims must contain TypedClaim or mapping values")
            if candidate.subject_id != subject_id:
                raise InvalidClaimError(
                    "apply_claims candidates must all reference the supplied subject_id"
                )
            updates: dict[str, Any] = {}
            if source_event_id is not None and candidate.source_event_id is None:
                updates["source_event_id"] = source_event_id
            if source_version_id is not None and candidate.source_version_id is None:
                updates["source_version_id"] = source_version_id
            if (
                source_event_id is not None or candidate.source_event_id is not None
            ) and candidate.source_triplet_index is None:
                updates["source_triplet_index"] = index
            if updates:
                candidate = replace(candidate, **updates)
            prepared.append(candidate)

        # Bind every low-level operation to one transaction.  This lets the
        # existing convenience methods remain independently usable while a
        # batch either commits in full or rolls back in full.
        with self._transaction(tenant) as conn:
            bound = MemoryRepository(
                _BoundTransactionStore(conn, projection_task_queue=self.projection_task_queue),
                default_tenant_id=self.default_tenant_id,
            )
            results: list[MemoryClaim] = []
            for candidate in prepared:
                result = bound.add_claim(candidate, tenant_id=tenant)
                policy = None
                try:
                    from engram.predicate_registry import normalize_predicate as registry_normalize

                    policy = registry_normalize(normalize_predicate(candidate.predicate))
                except (ImportError, AttributeError, TypeError, ValueError):
                    pass
                active = bound.get_current_claims(
                    candidate.subject_id,
                    predicate=candidate.predicate,
                    at=candidate.valid_from,
                    tenant_id=tenant,
                )
                conflicting = [
                    item
                    for item in active
                    if item.id != result.id
                    and item.normalized_object_hash != result.normalized_object_hash
                ]
                policy_name = getattr(getattr(policy, "cardinality", None), "value", "")
                conflict_name = getattr(getattr(policy, "conflict_policy", None), "value", "")
                if conflicting and policy_name == "SINGLE":
                    for old in conflicting:
                        result = bound.supersede_claim(old.id, result.id, tenant_id=tenant)
                elif conflicting and conflict_name == "FLAG":
                    row = conn.execute(
                        "UPDATE memory_claims SET status = 'CONFLICTING', "
                        "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ? RETURNING *",
                        (tenant, result.id),
                    ).fetchone()
                    if row is not None:
                        result = _claim_from_row(row)
                        revision = bound._bump_node_revision(conn, tenant, result.subject_id)
                        bound._enqueue_dispatch_in_tx(
                            conn,
                            tenant_id=tenant,
                            aggregate_type="CLAIM",
                            aggregate_id=str(result.id),
                            operation="CONFLICTING_CLAIM",
                            revision=revision,
                            payload={
                                "memory_id": str(result.subject_id),
                                "claim_id": str(result.id),
                            },
                        )
                results.append(result)
            return results

    def _require_version(self, conn: Any, tenant_id: str, version_id: uuid.UUID) -> MemoryVersion:
        row = conn.execute(
            "SELECT * FROM memory_versions WHERE tenant_id = ? AND id = ?",
            (tenant_id, version_id),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"version {version_id} not found for tenant {tenant_id}")
        return _version_from_row(row)

    @staticmethod
    def _bump_node_revision(conn: Any, tenant_id: str, memory_id: uuid.UUID) -> int:
        row = conn.execute(
            "UPDATE memory_nodes SET revision = revision + 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE tenant_id = ? AND id = ? RETURNING revision",
            (tenant_id, memory_id),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"memory {memory_id} not found for tenant {tenant_id}")
        return int(row["revision"])

    def get_current_claims(
        self,
        subject_id: uuid.UUID,
        *,
        predicate: str | None = None,
        at: datetime | date | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryClaim]:
        tenant = self._tenant(tenant_id)
        effective_at = _datetime(at, default=datetime.now(timezone.utc))
        predicate_clause = "" if predicate is None else " AND predicate = ?"
        params: tuple[Any, ...] = (
            (tenant, subject_id)
            if predicate is None
            else (tenant, subject_id, normalize_predicate(predicate))
        )
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND subject_id = ? "
                "AND status = 'ACTIVE'"
                + predicate_clause
                + " AND (valid_from IS NULL OR valid_from <= ?)"
                + " AND (valid_until IS NULL OR valid_until > ?) "
                "ORDER BY valid_from DESC NULLS LAST, asserted_at DESC, created_at DESC",
                (*params, effective_at, effective_at),
            )
            .fetchall()
        )
        return [_claim_from_row(row) for row in rows]

    def get_current_related_claims(
        self,
        entity_id: uuid.UUID,
        *,
        predicate: str | None = None,
        at: datetime | date | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryClaim]:
        """Return active claims where an entity is subject or object.

        Entity-profile questions need both directions. For example, the
        canonical relationship ``Atul --HAS_MANAGER--> Rahul`` is essential
        evidence for "Who is Rahul?" even though Rahul is the object rather
        than the claim subject.
        """

        tenant = self._tenant(tenant_id)
        effective_at = _datetime(at, default=datetime.now(timezone.utc))
        predicate_clause = "" if predicate is None else " AND predicate = ?"
        params: tuple[Any, ...] = (
            (tenant, entity_id, entity_id)
            if predicate is None
            else (tenant, entity_id, entity_id, normalize_predicate(predicate))
        )
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? "
                "AND (subject_id = ? OR object_entity_id = ?) "
                "AND status = 'ACTIVE'"
                + predicate_clause
                + " AND (valid_from IS NULL OR valid_from <= ?)"
                + " AND (valid_until IS NULL OR valid_until > ?) "
                "ORDER BY asserted_at DESC, created_at DESC",
                (*params, effective_at, effective_at),
            )
            .fetchall()
        )
        return [_claim_from_row(row) for row in rows]

    def get_claim(
        self,
        claim_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
    ) -> MemoryClaim | None:
        """Load one tenant-scoped claim for canonical discovery hydration."""

        tenant = self._tenant(tenant_id)
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                (tenant, claim_id),
            )
            .fetchone()
        )
        return _claim_from_row(row) if row is not None else None

    def get_claims_as_of(
        self,
        subject_id: uuid.UUID,
        at: datetime | date,
        *,
        predicate: str | None = None,
        tenant_id: str | None = None,
        include_retracted: bool = False,
    ) -> list[MemoryClaim]:
        """Return claims valid at a point in time, including superseded history."""

        tenant = self._tenant(tenant_id)
        effective_at = _datetime(at)
        assert effective_at is not None
        predicate_clause = "" if predicate is None else " AND predicate = ?"
        statuses = (
            "('ACTIVE', 'HISTORICAL', 'SUPERSEDED', 'LOW_CONFIDENCE', 'CONFLICTING', 'RETRACTED')"
            if include_retracted
            else "('ACTIVE', 'HISTORICAL', 'SUPERSEDED', 'LOW_CONFIDENCE', 'CONFLICTING')"
        )
        params: tuple[Any, ...] = (
            (tenant, subject_id)
            if predicate is None
            else (tenant, subject_id, normalize_predicate(predicate))
        )
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND subject_id = ? "
                "AND status IN "
                + statuses
                + predicate_clause
                + " AND (valid_from IS NULL OR valid_from <= ?)"
                + " AND (valid_until IS NULL OR valid_until > ?) "
                "ORDER BY COALESCE(valid_from, asserted_at) DESC, asserted_at DESC, created_at DESC",
                (*params, effective_at, effective_at),
            )
            .fetchall()
        )
        return [_claim_from_row(row) for row in rows]

    def get_claim_history(
        self,
        subject_id: uuid.UUID,
        *,
        predicate: str | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryClaim]:
        tenant = self._tenant(tenant_id)
        predicate_clause = "" if predicate is None else " AND predicate = ?"
        params: tuple[Any, ...] = (
            (tenant, subject_id)
            if predicate is None
            else (tenant, subject_id, normalize_predicate(predicate))
        )
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND subject_id = ?"
                + predicate_clause
                + " ORDER BY asserted_at DESC, created_at DESC",
                params,
            )
            .fetchall()
        )
        return [_claim_from_row(row) for row in rows]

    get_historical_claims = get_claim_history

    def supersede_claim(
        self,
        claim_id: uuid.UUID,
        replacement_claim_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
        replacement_status: str = "ACTIVE",
        emit_projection: bool = True,
    ) -> MemoryClaim:
        tenant = self._tenant(tenant_id)
        if replacement_status not in _CLAIM_STATUSES:
            raise ValueError(f"unsupported claim status: {replacement_status!r}")
        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    mutation_key=mutation_key,
                    operation="SUPERSEDE_CLAIM",
                    event_id=None,
                    payload={
                        "claim_id": str(claim_id),
                        "replacement_claim_id": str(replacement_claim_id),
                    },
                )
                if not created and mutation.status == "APPLIED":
                    row = conn.execute(
                        "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                        (tenant, replacement_claim_id),
                    ).fetchone()
                    if row is not None:
                        return _claim_from_row(row)
            old = conn.execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ? FOR UPDATE",
                (tenant, claim_id),
            ).fetchone()
            new = conn.execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                (tenant, replacement_claim_id),
            ).fetchone()
            if old is None or new is None:
                raise MemoryNotFoundError("claim supersession requires both claims in one tenant")
            if old["subject_id"] != new["subject_id"]:
                raise InvalidClaimError("superseded claims must have the same subject")
            close_at = _datetime(new.get("valid_from") or new.get("asserted_at"))
            if close_at is None:
                close_at = datetime.now(timezone.utc)
            old_from = _datetime(old.get("valid_from"))
            if old_from is not None and close_at <= old_from:
                # Preserve the interval check even when a replacement carries
                # an older asserted/valid-from timestamp.
                close_at = old_from + timedelta(microseconds=1)
            conn.execute(
                "UPDATE memory_claims SET status = 'HISTORICAL', superseded_by_claim_id = ?, "
                "valid_until = COALESCE(valid_until, ?), updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = ? AND id = ?",
                (replacement_claim_id, close_at, tenant, claim_id),
            )
            conn.execute(
                "UPDATE memory_claims SET supersedes_claim_id = ?, status = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ?",
                (claim_id, replacement_status, tenant, replacement_claim_id),
            )
            revision = self._bump_node_revision(conn, tenant, uuid.UUID(str(old["subject_id"])))
            if emit_projection:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="CLAIM",
                    aggregate_id=str(replacement_claim_id),
                    operation="SUPERSEDE_CLAIM",
                    revision=revision,
                    payload={"claim_id": str(replacement_claim_id), "supersedes": str(claim_id)},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
            result_row = conn.execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND id = ?",
                (tenant, replacement_claim_id),
            ).fetchone()
            result = _claim_from_row(result_row)
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=result.subject_id,
                    result={"claim_id": str(result.id)},
                )
            return result

    def record_evidence(
        self,
        *,
        memory_id: uuid.UUID,
        claim_id: uuid.UUID | None = None,
        source_event_id: str | None = None,
        source_session_id: str | None = None,
        extractor: str | None = None,
        extractor_version: str | None = None,
        confidence: float | None = None,
        source_span: JsonValue = None,
        source_text_hash: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
        evidence_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
    ) -> MemoryEvidence:
        tenant = self._tenant(tenant_id)
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("evidence confidence must be between 0 and 1")
        chosen_id = evidence_id or uuid.uuid4()
        with self._transaction(tenant) as conn:
            self._require_node(conn, tenant, memory_id)
            if claim_id is not None:
                row = conn.execute(
                    "SELECT id, subject_id FROM memory_claims WHERE tenant_id = ? AND id = ?",
                    (tenant, claim_id),
                ).fetchone()
                if row is None:
                    raise MemoryNotFoundError(f"claim {claim_id} not found for tenant {tenant}")
                if row["subject_id"] != memory_id:
                    raise InvalidClaimError("evidence memory_id must match the claim subject_id")
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM memory_evidence WHERE tenant_id = ? AND idempotency_key = ?",
                    (tenant, idempotency_key),
                ).fetchone()
                if existing is not None:
                    return _evidence_from_row(existing)
            row = conn.execute(
                "INSERT INTO memory_evidence "
                "(id, tenant_id, memory_id, claim_id, source_event_id, source_session_id, extractor, "
                "extractor_version, confidence, source_span, source_text_hash, metadata, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (tenant_id, id) DO NOTHING RETURNING *",
                (
                    chosen_id,
                    tenant,
                    memory_id,
                    claim_id,
                    source_event_id,
                    source_session_id,
                    extractor,
                    extractor_version,
                    confidence,
                    _json(source_span, default=None) if source_span is not None else None,
                    source_text_hash,
                    _json(dict(metadata or {}), default={}),
                    idempotency_key,
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM memory_evidence WHERE tenant_id = ? AND id = ?",
                    (tenant, chosen_id),
                ).fetchone()
            if row is None:
                raise MemoryRepositoryError("evidence insert returned no row")
            return _evidence_from_row(row)

    def get_evidence(
        self,
        memory_id: uuid.UUID,
        *,
        claim_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryEvidence]:
        tenant = self._tenant(tenant_id)
        claim_clause = "" if claim_id is None else " AND claim_id = ?"
        params: tuple[Any, ...] = (
            (tenant, memory_id) if claim_id is None else (tenant, memory_id, claim_id)
        )
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_evidence WHERE tenant_id = ? AND memory_id = ?"
                + claim_clause
                + " ORDER BY created_at DESC",
                params,
            )
            .fetchall()
        )
        return [_evidence_from_row(row) for row in rows]

    def add_alias(
        self,
        *,
        entity_id: uuid.UUID,
        alias: str,
        confidence: float | None = None,
        source_event_id: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        alias_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
    ) -> EntityAlias:
        tenant = self._tenant(tenant_id)
        normalized = normalize_alias(alias)
        if not normalized:
            raise ValueError("alias must not be empty")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("alias confidence must be between 0 and 1")
        with self._transaction(tenant) as conn:
            node = self._require_node(conn, tenant, entity_id)
            if node.memory_type != "ENTITY":
                raise ValueError("entity_aliases may only reference ENTITY memories")
            row = conn.execute(
                "SELECT * FROM entity_aliases WHERE tenant_id = ? AND entity_id = ? "
                "AND normalized_alias = ?",
                (tenant, entity_id, normalized),
            ).fetchone()
            if row is not None:
                return _alias_from_row(row)
            row = conn.execute(
                "INSERT INTO entity_aliases "
                "(id, tenant_id, entity_id, alias, normalized_alias, confidence, source_event_id, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (
                    alias_id or uuid.uuid4(),
                    tenant,
                    entity_id,
                    str(alias).strip(),
                    normalized,
                    confidence,
                    source_event_id,
                    _json(dict(metadata or {}), default={}),
                ),
            ).fetchone()
            if row is None:
                raise MemoryRepositoryError("alias insert returned no row")
            return _alias_from_row(row)

    add_entity_alias = add_alias

    def resolve_aliases(
        self,
        alias: str,
        *,
        tenant_id: str | None = None,
        include_historical: bool = False,
    ) -> list[EntityAlias]:
        tenant = self._tenant(tenant_id)
        status_clause = "" if include_historical else " AND n.status = 'ACTIVE'"
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT a.* FROM entity_aliases a "
                "JOIN memory_nodes n ON n.tenant_id = a.tenant_id AND n.id = a.entity_id "
                "WHERE a.tenant_id = ? AND a.normalized_alias = ?"
                + status_clause
                + " ORDER BY a.confidence DESC NULLS LAST, a.created_at ASC, a.id ASC",
                (tenant, normalize_alias(alias)),
            )
            .fetchall()
        )
        return [_alias_from_row(row) for row in rows]

    def resolve_alias(
        self,
        alias: str,
        *,
        tenant_id: str | None = None,
        include_historical: bool = False,
    ) -> EntityAlias | None:
        """Resolve an alias only when it identifies one entity unambiguously."""

        aliases = self.resolve_aliases(
            alias, tenant_id=tenant_id, include_historical=include_historical
        )
        return aliases[0] if len(aliases) == 1 else None

    def _add_uri_alias_in_tx(
        self,
        conn: Any,
        *,
        tenant_id: str,
        memory_id: uuid.UUID,
        uri: str,
        alias_type: str,
        metadata: Mapping[str, JsonValue] | None,
    ) -> MemoryUriAlias:
        alias_kind = str(alias_type).upper()
        allowed = {"LEGACY_FILE", "CANONICAL", "EXTERNAL", "REDIRECT", "OTHER"}
        if alias_kind not in allowed:
            raise ValueError(f"unsupported URI alias type: {alias_type!r}")
        row = conn.execute(
            "SELECT * FROM memory_uri_aliases WHERE tenant_id = ? AND uri = ?",
            (tenant_id, uri),
        ).fetchone()
        if row is not None:
            existing = _uri_alias_from_row(row)
            if existing.memory_id != memory_id:
                raise MemoryRepositoryError(
                    f"URI alias already belongs to {existing.memory_id}: {uri}"
                )
            return existing
        row = conn.execute(
            "INSERT INTO memory_uri_aliases "
            "(id, tenant_id, memory_id, uri, alias_type, metadata) VALUES (?, ?, ?, ?, ?, ?) "
            "RETURNING *",
            (
                uuid.uuid4(),
                tenant_id,
                memory_id,
                uri,
                alias_kind,
                _json(dict(metadata or {}), default={}),
            ),
        ).fetchone()
        if row is None:
            raise MemoryRepositoryError("URI alias insert returned no row")
        return _uri_alias_from_row(row)

    def add_uri_alias(
        self,
        *,
        memory_id: uuid.UUID,
        uri: str,
        alias_type: str = "LEGACY_FILE",
        metadata: Mapping[str, JsonValue] | None = None,
        tenant_id: str | None = None,
    ) -> MemoryUriAlias:
        tenant = self._tenant(tenant_id)
        normalized_uri = str(uri).strip()
        if not normalized_uri:
            raise ValueError("URI alias must not be empty")
        with self._transaction(tenant) as conn:
            self._require_node(conn, tenant, memory_id)
            return self._add_uri_alias_in_tx(
                conn,
                tenant_id=tenant,
                memory_id=memory_id,
                uri=normalized_uri,
                alias_type=alias_type,
                metadata=metadata,
            )

    def resolve_uri(self, uri: str, *, tenant_id: str | None = None) -> MemoryNode | None:
        tenant = self._tenant(tenant_id)
        normalized_uri = str(uri).strip()
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM memory_nodes WHERE tenant_id = ? AND canonical_uri = ?",
                (tenant, normalized_uri),
            )
            .fetchone()
        )
        if row is not None:
            return _node_from_row(row)
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.* FROM memory_uri_aliases a "
                "JOIN memory_nodes n ON n.tenant_id = a.tenant_id AND n.id = a.memory_id "
                "WHERE a.tenant_id = ? AND a.uri = ?",
                (tenant, normalized_uri),
            )
            .fetchone()
        )
        return _node_from_row(row) if row is not None else None

    def add_hierarchy(
        self,
        *,
        parent_id: uuid.UUID,
        child_id: uuid.UUID,
        position: int | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        hierarchy_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
        emit_projection: bool = True,
    ) -> MemoryHierarchy:
        tenant = self._tenant(tenant_id)
        if parent_id == child_id:
            raise ValueError("memory hierarchy cannot contain a self edge")
        with self._transaction(tenant) as conn:
            # Serialize hierarchy mutations per tenant so two concurrent
            # transactions cannot each pass the cycle check and jointly form
            # a cycle. Hash collisions only reduce concurrency; they do not
            # weaken correctness.
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"memory-hierarchy:{tenant}",),
            )
            self._require_node(conn, tenant, parent_id)
            self._require_node(conn, tenant, child_id)
            existing = conn.execute(
                "SELECT * FROM memory_hierarchy WHERE tenant_id = ? AND parent_id = ? AND child_id = ?",
                (tenant, parent_id, child_id),
            ).fetchone()
            if existing is not None:
                return _hierarchy_from_row(existing)
            cycle = conn.execute(
                "WITH RECURSIVE descendants(id) AS ("
                "SELECT CAST(? AS UUID) "
                "UNION "
                "SELECT h.child_id FROM memory_hierarchy h "
                "JOIN descendants d ON d.id = h.parent_id "
                "WHERE h.tenant_id = ?) "
                "SELECT 1 FROM descendants WHERE id = ? LIMIT 1",
                (child_id, tenant, parent_id),
            ).fetchone()
            if cycle is not None:
                raise ValueError("memory hierarchy edge would create a cycle")
            row = conn.execute(
                "INSERT INTO memory_hierarchy "
                "(id, tenant_id, parent_id, child_id, position, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?) RETURNING *",
                (
                    hierarchy_id or uuid.uuid4(),
                    tenant,
                    parent_id,
                    child_id,
                    position,
                    _json(dict(metadata or {}), default={}),
                ),
            ).fetchone()
            if row is None:
                raise MemoryRepositoryError("hierarchy insert returned no row")
            edge = _hierarchy_from_row(row)
            revision = self._bump_node_revision(conn, tenant, parent_id)
            if emit_projection:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(parent_id),
                    operation="UPSERT_HIERARCHY",
                    revision=revision,
                    payload={
                        "hierarchy_id": str(edge.id),
                        "parent_id": str(parent_id),
                        "child_id": str(child_id),
                        "aggregate_type": "HIERARCHY",
                    },
                )
            return edge

    add_child = add_hierarchy

    def get_children(
        self, parent_id: uuid.UUID, *, tenant_id: str | None = None
    ) -> list[MemoryNode]:
        tenant = self._tenant(tenant_id)
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.* FROM memory_hierarchy h "
                "JOIN memory_nodes n ON n.tenant_id = h.tenant_id AND n.id = h.child_id "
                "WHERE h.tenant_id = ? AND h.parent_id = ? "
                "ORDER BY h.position NULLS LAST, n.canonical_name NULLS LAST, n.id",
                (tenant, parent_id),
            )
            .fetchall()
        )
        return [_node_from_row(row) for row in rows]

    def get_parents(self, child_id: uuid.UUID, *, tenant_id: str | None = None) -> list[MemoryNode]:
        tenant = self._tenant(tenant_id)
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.* FROM memory_hierarchy h "
                "JOIN memory_nodes n ON n.tenant_id = h.tenant_id AND n.id = h.parent_id "
                "WHERE h.tenant_id = ? AND h.child_id = ? "
                "ORDER BY h.position NULLS LAST, n.canonical_name NULLS LAST, n.id",
                (tenant, child_id),
            )
            .fetchall()
        )
        return [_node_from_row(row) for row in rows]

    def save_overview(
        self,
        *,
        scope_id: uuid.UUID,
        content: str,
        input_revision: int,
        model_metadata: Mapping[str, JsonValue] | None = None,
        overview_id: uuid.UUID | None = None,
        generated_at: datetime | None = None,
        tenant_id: str | None = None,
    ) -> MemoryOverview:
        tenant = self._tenant(tenant_id)
        if input_revision <= 0:
            raise ValueError("input_revision must be positive")
        with self._transaction(tenant) as conn:
            self._require_node(conn, tenant, scope_id)
            row = conn.execute(
                "INSERT INTO memory_overviews "
                "(id, tenant_id, scope_id, content, input_revision, model_metadata, generated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (tenant_id, scope_id, input_revision) DO UPDATE SET "
                "content = EXCLUDED.content, model_metadata = EXCLUDED.model_metadata, "
                "generated_at = EXCLUDED.generated_at RETURNING *",
                (
                    overview_id or uuid.uuid4(),
                    tenant,
                    scope_id,
                    str(content),
                    input_revision,
                    _json(dict(model_metadata or {}), default={}),
                    generated_at or datetime.now(timezone.utc),
                ),
            ).fetchone()
            if row is None:
                raise MemoryRepositoryError("overview upsert returned no row")
            return _overview_from_row(row)

    def get_overview(
        self,
        scope_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
        require_current: bool = False,
    ) -> MemoryOverview | None:
        tenant = self._tenant(tenant_id)
        current_clause = " AND o.input_revision = n.revision" if require_current else ""
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT o.* FROM memory_overviews o "
                "JOIN memory_nodes n ON n.tenant_id = o.tenant_id AND n.id = o.scope_id "
                "WHERE o.tenant_id = ? AND o.scope_id = ?"
                + current_clause
                + " ORDER BY o.input_revision DESC LIMIT 1",
                (tenant, scope_id),
            )
            .fetchone()
        )
        return _overview_from_row(row) if row is not None else None

    def enqueue_overview_regeneration(
        self,
        scope_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
    ) -> str | None:
        """Durably enqueue regeneration for a missing or stale overview.

        Retrieval calls this only after a current-revision overview lookup
        misses.  The consolidation task uniqueness constraint makes repeated
        reads idempotent, while the control-plane outbox ensures Temporal sees
        the work after the task transaction commits.
        """

        tenant = self._tenant(tenant_id)
        connection = self._tenant_connection(tenant)
        self._require_node(connection, tenant, scope_id)
        enqueue = getattr(self.store, "enqueue_task", None)
        if not callable(enqueue):
            raise MemoryRepositoryError("control-plane store cannot enqueue overview regeneration")
        result = enqueue(
            node_id=str(scope_id),
            task_type="CONSOLIDATE_OVERVIEW",
            priority=6,
            tenant_id=tenant,
        )
        return str(result) if result is not None else None

    # ------------------------------------------------------------------
    # Ingest artifacts and the shared Temporal workflow dispatch outbox
    # ------------------------------------------------------------------

    def record_ingest_artifact(
        self,
        *,
        event_id: str,
        artifact_type: str = "UPLOAD_ZIP",
        artifact_key: str = "upload.zip",
        payload: Mapping[str, JsonValue] | None = None,
        content: bytes | bytearray | memoryview | str | None = None,
        artifact_bytes: bytes | bytearray | memoryview | None = None,
        content_hash: str | None = None,
        media_type: str = "application/zip",
        expires_at: datetime | None = None,
        extractor: str | None = None,
        extractor_version: str | None = None,
        source_span: JsonValue = None,
        metadata: Mapping[str, JsonValue] | None = None,
        artifact_id: uuid.UUID | None = None,
        tenant_id: str | None = None,
    ) -> IngestArtifact:
        tenant = self._tenant(tenant_id)
        kind = str(artifact_type).upper()
        allowed = {
            "UPLOAD_ZIP",
            "ZIP",
            "RAW_EVENT",
            "SOURCE_TEXT",
            "EXTRACTION",
            "TYPED_CLAIMS",
            "NORMALIZED_CLAIMS",
            "OTHER",
        }
        if kind not in allowed:
            raise ValueError(f"unsupported ingest artifact type: {artifact_type!r}")
        key = str(artifact_key).strip()
        if not key:
            raise ValueError("artifact_key must not be empty")
        if content is not None and artifact_bytes is not None:
            raise TypeError("pass content or artifact_bytes, not both")
        raw_content: bytes
        supplied_content = artifact_bytes if artifact_bytes is not None else content
        if isinstance(supplied_content, memoryview):
            raw_content = supplied_content.tobytes()
        elif isinstance(supplied_content, bytearray):
            raw_content = bytes(supplied_content)
        elif isinstance(supplied_content, bytes):
            raw_content = supplied_content
        elif isinstance(supplied_content, str):
            # Keep a narrow compatibility path for callers that used the
            # provisional text field; storage is always BYTEA.
            raw_content = supplied_content.encode("utf-8")
        elif supplied_content is None and kind not in {"UPLOAD_ZIP", "ZIP"}:
            # Gate/extraction metadata may be represented by ``payload`` and
            # need not pretend to be an uploaded archive.  The upload types
            # below remain strict so a missing ZIP can never be acknowledged
            # as a durable code-ingest artifact.
            raw_content = b""
        else:
            raise ValueError("uploaded ingest artifact content must be bytes")
        if not raw_content and kind in {"UPLOAD_ZIP", "ZIP"}:
            raise ValueError("uploaded ingest artifact content must be non-empty bytes")
        digest = content_hash or hashlib.sha256(raw_content).hexdigest()
        if digest != hashlib.sha256(raw_content).hexdigest():
            raise ValueError("content_hash does not match uploaded artifact bytes")
        media = str(media_type).strip()
        if not media:
            raise ValueError("media_type must not be empty")
        with self._transaction(tenant) as conn:
            existing = conn.execute(
                "SELECT * FROM ingest_artifacts WHERE tenant_id = ? AND event_id = ? "
                "AND artifact_type = ? AND artifact_key = ?",
                (tenant, event_id, kind, key),
            ).fetchone()
            if existing is not None:
                return _artifact_from_row(existing)
            row = conn.execute(
                "INSERT INTO ingest_artifacts "
                "(id, tenant_id, event_id, artifact_type, artifact_key, payload, content, content_hash, "
                "media_type, expires_at, extractor, extractor_version, source_span, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "RETURNING *",
                (
                    artifact_id or uuid.uuid4(),
                    tenant,
                    event_id,
                    kind,
                    key,
                    _json(dict(payload or {}), default={}),
                    raw_content,
                    digest,
                    media,
                    expires_at,
                    extractor,
                    extractor_version,
                    _json(source_span, default=None) if source_span is not None else None,
                    _json(dict(metadata or {}), default={}),
                ),
            ).fetchone()
            if row is None:
                raise MemoryRepositoryError("ingest artifact insert returned no row")
            return _artifact_from_row(row)

    def get_ingest_artifact(
        self,
        event_id: str,
        artifact_type: str,
        artifact_key: str,
        *,
        tenant_id: str | None = None,
    ) -> IngestArtifact | None:
        tenant = self._tenant(tenant_id)
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM ingest_artifacts WHERE tenant_id = ? AND event_id = ? "
                "AND artifact_type = ? AND artifact_key = ?",
                (tenant, event_id, str(artifact_type).upper(), artifact_key),
            )
            .fetchone()
        )
        return _artifact_from_row(row) if row is not None else None

    def list_ingest_artifacts(
        self, event_id: str, *, tenant_id: str | None = None
    ) -> list[IngestArtifact]:
        tenant = self._tenant(tenant_id)
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM ingest_artifacts WHERE tenant_id = ? AND event_id = ? "
                "ORDER BY created_at, id",
                (tenant, event_id),
            )
            .fetchall()
        )
        return [_artifact_from_row(row) for row in rows]

    def commit_code_project(
        self,
        *,
        event_id: str,
        project_name: str,
        archive_bytes: bytes | bytearray | memoryview | None = None,
        zip_bytes: bytes | bytearray | memoryview | None = None,
        project_uri: str | None = None,
        filename: str = "project.zip",
        media_type: str = "application/zip",
        expires_at: datetime | None = None,
        files: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
    ) -> CodeProjectCommit:
        """Persist a ZIP upload and its optional code hierarchy in canonical PG.

        ZIP parsing, when descriptors are not supplied, happens in memory only;
        no temporary or generated file is opened.  Archive bytes remain the
        durable ingest artifact while code nodes/versions are independently
        rebuildable from canonical rows.
        """

        if archive_bytes is not None and zip_bytes is not None:
            raise TypeError("pass archive_bytes or zip_bytes, not both")
        raw_archive = archive_bytes if archive_bytes is not None else zip_bytes
        if raw_archive is None:
            raise ValueError("archive_bytes is required for a code project")
        if isinstance(raw_archive, memoryview):
            raw = raw_archive.tobytes()
        elif isinstance(raw_archive, bytearray):
            raw = bytes(raw_archive)
        elif isinstance(raw_archive, bytes):
            raw = raw_archive
        else:
            raise TypeError("archive_bytes must be bytes-like")
        if not raw:
            raise ValueError("archive_bytes must not be empty")
        clean_name = str(project_name).strip()
        if not clean_name:
            raise ValueError("project_name must not be empty")
        tenant = self._tenant(tenant_id)
        if project_uri is None:
            slug = re.sub(r"[^a-z0-9]+", "-", normalize_alias(clean_name)).strip("-")
            project_uri = f"mem://projects/{slug or uuid.uuid4().hex}"
        project_uri = str(project_uri).strip()
        if not project_uri.startswith("mem://") or not project_uri:
            raise ValueError("project_uri must use the mem:// scheme")
        project_identity = _deterministic_uuid(
            _MUTATION_NAMESPACE, tenant, event_id, "code-project", project_uri
        )
        canonical_project_uri = f"mem://memory/{project_identity}"
        existing = self.resolve_uri(project_uri, tenant_id=tenant)
        if existing is None:
            existing = self.resolve_memory_ref(
                project_identity, tenant_id=tenant, include_historical=True
            )
        if existing is not None:
            if existing.memory_type != "PROJECT":
                raise MemoryRepositoryError(
                    f"project URI belongs to {existing.memory_type}: {project_uri}"
                )
            project = existing
        else:
            project = self.create_memory(
                memory_type="PROJECT",
                canonical_uri=canonical_project_uri,
                canonical_name=clean_name,
                memory_id=project_identity,
                origin_event_id=event_id,
                tenant_id=tenant,
                mutation_key=mutation_key,
                metadata={"source": "code-upload", "project_uri": project_uri},
            )
        if project_uri != project.canonical_uri:
            self.add_uri_alias(
                memory_id=project.id,
                uri=project_uri,
                alias_type="EXTERNAL",
                metadata={"source": "code-upload", "project_name": clean_name},
                tenant_id=tenant,
            )
        artifact = self.record_ingest_artifact(
            event_id=event_id,
            artifact_type="UPLOAD_ZIP",
            artifact_key=filename,
            content=raw,
            media_type=media_type,
            expires_at=expires_at,
            tenant_id=tenant,
            metadata={"project_id": str(project.id), "project_uri": project_uri},
        )

        descriptors: list[Mapping[str, Any]] = []
        if files is not None:
            if isinstance(files, Mapping):
                descriptors = [{"path": str(path), "body": body} for path, body in files.items()]
            else:
                descriptors = [dict(item) for item in files]
        else:
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    members = [info for info in archive.infolist() if not info.is_dir()]
                    if len(members) > _MAX_ZIP_MEMBERS:
                        raise ValueError("ZIP archive contains too many files")
                    if sum(info.file_size for info in members) > _MAX_ZIP_EXPANDED_BYTES:
                        raise ValueError("ZIP expanded content exceeds 50 MiB")
                    for info in members:
                        if info.is_dir():
                            continue
                        safe_path = info.filename.replace("\\", "/").lstrip("/")
                        if not safe_path or any(part == ".." for part in safe_path.split("/")):
                            continue
                        if info.flag_bits & 0x1:
                            raise ValueError("encrypted ZIP members are not supported")
                        if info.file_size > _MAX_ZIP_MEMBER_BYTES:
                            raise ValueError("ZIP member exceeds 10 MiB")
                        body_bytes = archive.read(info)
                        try:
                            body: Any = body_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            body = None
                        descriptors.append({"path": safe_path, "body": body})
            except (zipfile.BadZipFile, OSError):
                # The artifact is still durable and can be quarantined by a
                # later ingest validator; canonical upload commit is not lost.
                descriptors = []

        committed_nodes: list[MemoryNode] = [project]
        for index, descriptor in enumerate(descriptors):
            path = str(descriptor.get("path") or descriptor.get("name") or "").strip()
            if not path or path.startswith("/") or any(part == ".." for part in path.split("/")):
                raise ValueError(f"invalid code path at index {index}")
            node_kind = str(
                descriptor.get("memory_type") or descriptor.get("node_type") or "FILE"
            ).upper()
            if node_kind not in {"FILE", "CLASS", "FUNCTION", "METHOD", "EXTERNAL_MODULE"}:
                raise ValueError(f"unsupported code node type: {node_kind!r}")
            node_id = _deterministic_uuid(_MUTATION_NAMESPACE, tenant, project.id, "code", path)
            node = self.resolve_memory_ref(node_id, tenant_id=tenant, include_historical=True)
            if node is None:
                node = self.create_memory(
                    memory_type=node_kind,
                    canonical_uri=f"mem://memory/{node_id}",
                    canonical_name=str(descriptor.get("canonical_name") or path),
                    memory_id=node_id,
                    origin_event_id=event_id,
                    metadata={"project_uri": project_uri, "path": path},
                    tenant_id=tenant,
                )
            self.add_uri_alias(
                memory_id=node.id,
                uri=f"{project_uri}/files/{path}",
                alias_type="EXTERNAL",
                metadata={"source": "code-upload", "path": path},
                tenant_id=tenant,
            )
            body_value = descriptor.get("body", descriptor.get("content"))
            if body_value is not None:
                body = (
                    body_value.decode("utf-8", errors="replace")
                    if isinstance(body_value, bytes)
                    else str(body_value)
                )
                if body:
                    self.append_version(
                        node.id,
                        body=body,
                        source_event_id=event_id,
                        provenance={"artifact_id": str(artifact.id), "path": path},
                        tenant_id=tenant,
                    )
            self.add_hierarchy(
                parent_id=project.id,
                child_id=node.id,
                position=index,
                metadata={"path": path},
                tenant_id=tenant,
            )
            committed_nodes.append(node)
        return CodeProjectCommit(project=project, nodes=tuple(committed_nodes), artifact=artifact)

    def _enqueue_dispatch_in_tx(
        self,
        conn: Any,
        *,
        tenant_id: str,
        aggregate_type: str,
        aggregate_id: str,
        operation: str,
        revision: int,
        payload: Mapping[str, JsonValue] | None,
        available_at: datetime | None = None,
        canonical_mutation_id: str | uuid.UUID | None = None,
    ) -> WorkflowDispatch:
        """Insert a projection dispatch in the same transaction as canonical state.

        ``workflow_dispatches`` is the repository's sole outbox.  The
        dispatch aggregate is always the canonical mutation UUID so Temporal
        can reload the committed canonical snapshot instead of trusting a
        stale serialized graph payload.
        """

        envelope = dict(payload or {})
        envelope.setdefault("aggregate_type", str(aggregate_type))
        envelope.setdefault("aggregate_id", str(aggregate_id))
        envelope.setdefault("operation", str(operation))
        envelope.setdefault("revision", int(revision))
        mutation_id = str(canonical_mutation_id) if canonical_mutation_id is not None else None
        if mutation_id is None:
            source_event_id = str(
                envelope.get("source_event_id")
                or f"projection:{aggregate_type}:{aggregate_id}:{operation}:{revision}"
            )
            mutation, _ = MemoryRepository._begin_mutation_in_tx(
                conn,
                tenant_id=tenant_id,
                source_event_id=source_event_id,
                mutation_type=f"PROJECTION_{str(operation).upper()}",
                payload=envelope,
            )
            mutation_id = str(mutation.id)
            envelope["canonical_mutation_id"] = mutation_id
            if mutation.status != "APPLIED":
                MemoryRepository._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=(
                        _uuid(aggregate_id)
                        if str(aggregate_type).upper() in {"MEMORY", "NODE"}
                        else None
                    ),
                    result=envelope,
                )
        else:
            envelope.setdefault("canonical_mutation_id", mutation_id)
            # Keep the compact dispatch envelope on the canonical mutation so
            # a Temporal Activity can reload operation/aggregate metadata from
            # PostgreSQL after the workflow is started.
            conn.execute(
                "UPDATE canonical_mutations SET payload = payload || CAST(? AS JSONB), "
                "result = result || CAST(? AS JSONB), updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = ? AND id = ?",
                (_json(envelope, default={}), _json(envelope, default={}), tenant_id, mutation_id),
            )
        dispatch_id = f"dsp-{uuid.uuid4().hex[:16]}"
        row = conn.execute(
            "INSERT INTO workflow_dispatches "
            "(dispatch_id, workflow_type, aggregate_id, generation, tenant_id, task_queue, "
            "workflow_id, payload, aggregate_revision, available_at) "
            "VALUES (?, 'PROJECTION', ?, 1, ?, ?, ?, ?, ?, "
            "COALESCE(?, CURRENT_TIMESTAMP)) "
            "ON CONFLICT (workflow_type, aggregate_id, generation) DO NOTHING "
            "RETURNING *",
            (
                dispatch_id,
                mutation_id,
                tenant_id,
                self.projection_task_queue,
                f"projection:{mutation_id}",
                _json(envelope, default={}),
                int(revision),
                available_at,
            ),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM workflow_dispatches WHERE tenant_id = ? "
                "AND workflow_type = 'PROJECTION' AND aggregate_id = ? AND generation = 1",
                (tenant_id, mutation_id),
            ).fetchone()
        if row is None:
            raise MemoryRepositoryError("workflow projection dispatch insert returned no row")
        return _dispatch_from_row(row)

    def enqueue_projection(
        self,
        *,
        aggregate_type: str,
        aggregate_id: str | uuid.UUID,
        operation: str,
        revision: int,
        payload: Mapping[str, JsonValue] | None = None,
        available_at: datetime | None = None,
        canonical_mutation_id: str | uuid.UUID | None = None,
        tenant_id: str | None = None,
    ) -> WorkflowDispatch:
        tenant = self._tenant(tenant_id)
        if revision <= 0:
            raise ValueError("projection revision must be positive")
        with self._transaction(tenant) as conn:
            return self._enqueue_dispatch_in_tx(
                conn,
                tenant_id=tenant,
                aggregate_type=str(aggregate_type),
                aggregate_id=str(aggregate_id),
                operation=str(operation),
                revision=revision,
                payload=payload,
                available_at=available_at,
                canonical_mutation_id=canonical_mutation_id,
            )

    def get_projection_dispatch(
        self, outbox_id: str, *, tenant_id: str | None = None
    ) -> WorkflowDispatch | None:
        tenant = self._tenant(tenant_id)
        row = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT * FROM workflow_dispatches WHERE tenant_id = ? AND dispatch_id = ?",
                (tenant, str(outbox_id)),
            )
            .fetchone()
        )
        return _dispatch_from_row(row) if row is not None else None

    def claim_projection_dispatches(
        self, *, limit: int = 32, tenant_id: str | None = None, lease_seconds: int = 300
    ) -> list[WorkflowDispatch]:
        if limit <= 0:
            return []
        tenant = self._tenant(tenant_id)
        tenant_clause = " AND tenant_id = ?"
        owner = f"canonical-repository-{uuid.uuid4().hex}"
        token = uuid.uuid4().hex
        params: tuple[Any, ...] = (tenant, limit, token, owner, max(1, int(lease_seconds)))
        with self._transaction(tenant) as conn:
            rows = conn.execute(
                "WITH picked AS (SELECT dispatch_id FROM workflow_dispatches "
                "WHERE workflow_type = 'PROJECTION' AND status = 'PENDING' "
                "AND available_at <= CURRENT_TIMESTAMP"
                + tenant_clause
                + " ORDER BY available_at, created_at FOR UPDATE SKIP LOCKED LIMIT ?) "
                "UPDATE workflow_dispatches d SET status = 'DISPATCHING', attempts = attempts + 1, "
                "claim_token = ?, lease_owner = ?, "
                "claimed_until = CURRENT_TIMESTAMP + (? * INTERVAL '1 second'), "
                "updated_at = CURRENT_TIMESTAMP WHERE d.dispatch_id IN "
                "(SELECT dispatch_id FROM picked) RETURNING d.*",
                params,
            ).fetchall()
        return [_dispatch_from_row(row) for row in rows]

    def mark_dispatch_complete(
        self, outbox_id: str, *, tenant_id: str | None = None
    ) -> WorkflowDispatch:
        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            row = conn.execute(
                "UPDATE workflow_dispatches SET status = 'COMPLETE', completed_at = CURRENT_TIMESTAMP, "
                "last_error = NULL, claim_token = NULL, lease_owner = NULL, claimed_until = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND dispatch_id = ? "
                "AND status IN ('PENDING', 'DISPATCHING', 'STARTED') RETURNING *",
                (tenant, str(outbox_id)),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(
                    f"projection dispatch {outbox_id} not found or already complete"
                )
            return _dispatch_from_row(row)

    def fail_dispatch(
        self,
        outbox_id: str,
        error: str,
        *,
        tenant_id: str | None = None,
        retry_after_seconds: int = 30,
    ) -> WorkflowDispatch:
        tenant = self._tenant(tenant_id)
        delay = max(0, int(retry_after_seconds))
        with self._transaction(tenant) as conn:
            row = conn.execute(
                "UPDATE workflow_dispatches SET status = 'PENDING', last_error = ?, "
                "failure_count = failure_count + 1, "
                "available_at = CURRENT_TIMESTAMP + (? * INTERVAL '1 second'), "
                "claim_token = NULL, lease_owner = NULL, claimed_until = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND dispatch_id = ? "
                "AND status IN ('DISPATCHING', 'STARTED', 'PENDING') RETURNING *",
                (str(error)[:2000], delay, tenant, str(outbox_id)),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"projection dispatch {outbox_id} not found")
            return _dispatch_from_row(row)

    # ------------------------------------------------------------------
    # Canonical read model and lifecycle
    # ------------------------------------------------------------------

    def get_current_state(
        self,
        memory_id: uuid.UUID | None = None,
        *,
        uri: str | None = None,
        tenant_id: str | None = None,
        include_historical_claims: bool = False,
        require_current_overview: bool = True,
    ) -> MemoryState | None:
        tenant = self._tenant(tenant_id)
        conn = self._tenant_connection(tenant)
        if memory_id is None and uri is None:
            raise TypeError("memory_id or uri is required")
        if memory_id is not None:
            row = conn.execute(
                "SELECT * FROM memory_nodes WHERE tenant_id = ? AND id = ?",
                (tenant, memory_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT n.* FROM memory_uri_aliases a "
                "JOIN memory_nodes n ON n.tenant_id = a.tenant_id AND n.id = a.memory_id "
                "WHERE a.tenant_id = ? AND a.uri = ?",
                (tenant, uri),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM memory_nodes WHERE tenant_id = ? AND canonical_uri = ?",
                    (tenant, uri),
                ).fetchone()
        if row is None:
            return None
        node = _node_from_row(row)
        version_row = None
        if node.current_version_id is not None:
            version_row = conn.execute(
                "SELECT * FROM memory_versions WHERE tenant_id = ? AND id = ?",
                (tenant, node.current_version_id),
            ).fetchone()
        status_clause = "" if include_historical_claims else " AND status = 'ACTIVE'"
        claims = [
            _claim_from_row(item)
            for item in conn.execute(
                "SELECT * FROM memory_claims WHERE tenant_id = ? AND subject_id = ?"
                + status_clause
                + " ORDER BY asserted_at DESC, created_at DESC",
                (tenant, node.id),
            ).fetchall()
        ]
        aliases = [
            _alias_from_row(item)
            for item in conn.execute(
                "SELECT * FROM entity_aliases WHERE tenant_id = ? AND entity_id = ? "
                "ORDER BY confidence DESC NULLS LAST, created_at ASC",
                (tenant, node.id),
            ).fetchall()
        ]
        evidence = [
            _evidence_from_row(item)
            for item in conn.execute(
                "SELECT * FROM memory_evidence WHERE tenant_id = ? AND memory_id = ? "
                "ORDER BY created_at DESC",
                (tenant, node.id),
            ).fetchall()
        ]
        parents = [
            _hierarchy_from_row(item)
            for item in conn.execute(
                "SELECT * FROM memory_hierarchy WHERE tenant_id = ? AND child_id = ? "
                "ORDER BY position NULLS LAST, created_at",
                (tenant, node.id),
            ).fetchall()
        ]
        children = [
            _hierarchy_from_row(item)
            for item in conn.execute(
                "SELECT * FROM memory_hierarchy WHERE tenant_id = ? AND parent_id = ? "
                "ORDER BY position NULLS LAST, created_at",
                (tenant, node.id),
            ).fetchall()
        ]
        overview_clause = " AND o.input_revision = n.revision" if require_current_overview else ""
        overview_row = conn.execute(
            "SELECT o.* FROM memory_overviews o "
            "JOIN memory_nodes n ON n.tenant_id = o.tenant_id AND n.id = o.scope_id "
            "WHERE o.tenant_id = ? AND o.scope_id = ?"
            + overview_clause
            + " ORDER BY o.input_revision DESC LIMIT 1",
            (tenant, node.id),
        ).fetchone()
        return MemoryState(
            node=node,
            current_version=_version_from_row(version_row) if version_row is not None else None,
            claims=tuple(claims),
            aliases=tuple(aliases),
            evidence=tuple(evidence),
            parents=tuple(parents),
            children=tuple(children),
            overview=_overview_from_row(overview_row) if overview_row is not None else None,
        )

    get_state = get_current_state

    def get_recent_states(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 30,
        tenant_id: str | None = None,
    ) -> list[MemorySnapshot]:
        """Return recent canonical episodes/session summaries in time order."""

        tenant = self._tenant(tenant_id)
        bounded_limit = max(1, min(int(limit), 200))
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "SELECT n.id FROM memory_nodes n "
                "JOIN memory_versions v ON v.tenant_id = n.tenant_id "
                "AND v.id = n.current_version_id "
                "WHERE n.tenant_id = ? AND n.status = 'ACTIVE' "
                "AND n.memory_type IN ('EPISODE', 'SESSION_SUMMARY') "
                "AND (CAST(? AS TIMESTAMPTZ) IS NULL "
                "OR COALESCE(v.valid_from, v.asserted_at, n.updated_at) >= "
                "CAST(? AS TIMESTAMPTZ)) "
                "AND (CAST(? AS TIMESTAMPTZ) IS NULL "
                "OR COALESCE(v.valid_from, v.asserted_at, n.updated_at) < "
                "CAST(? AS TIMESTAMPTZ)) "
                "ORDER BY COALESCE(v.valid_from, v.asserted_at, n.updated_at) DESC "
                "LIMIT ?",
                (tenant, since, since, until, until, bounded_limit),
            )
            .fetchall()
        )
        snapshots: list[MemorySnapshot] = []
        for row in rows:
            snapshot = self.get_current_state(
                cast(uuid.UUID, _uuid(row["id"])),
                tenant_id=tenant,
                require_current_overview=False,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def get_neighbor_states(
        self,
        memory_id: uuid.UUID,
        *,
        before: int = 1,
        after: int = 2,
        limit: int = 30,
        tenant_id: str | None = None,
    ) -> list[MemorySnapshot]:
        """Return canonical episode context around evidence for one memory.

        This is a deterministic PostgreSQL retrieval capability for narrative
        follow-ups.  It is especially useful when a question about a stable
        anchor (for example a release tag) is answered by the next event in
        the same session rather than by a claim on the anchor itself.
        """

        tenant = self._tenant(tenant_id)
        bounded_before = max(0, min(int(before), 10))
        bounded_after = max(0, min(int(after), 10))
        bounded_limit = max(1, min(int(limit), 200))
        rows = (
            self._tenant_connection(tenant)
            .execute(
                "WITH anchor_events AS ("
                "SELECT origin_event_id AS event_id FROM memory_nodes "
                "WHERE tenant_id = ? AND id = ? AND origin_event_id IS NOT NULL "
                "UNION SELECT source_event_id FROM memory_evidence "
                "WHERE tenant_id = ? AND memory_id = ? "
                "UNION SELECT source_event_id FROM memory_claims "
                "WHERE tenant_id = ? AND source_event_id IS NOT NULL "
                "AND (subject_id = ? OR object_entity_id = ?)"
                "), ranked AS ("
                "SELECT event_id, session_id, created_at, "
                "row_number() OVER (PARTITION BY session_id ORDER BY created_at, event_id) AS seq "
                "FROM events WHERE tenant_id = ? AND session_id IS NOT NULL"
                "), neighbor_events AS ("
                "SELECT DISTINCT neighbor.event_id, neighbor.created_at "
                "FROM ranked anchor JOIN anchor_events source ON source.event_id = anchor.event_id "
                "JOIN ranked neighbor ON neighbor.session_id = anchor.session_id "
                "AND neighbor.seq BETWEEN anchor.seq - ? AND anchor.seq + ?"
                ") SELECT n.id FROM neighbor_events e JOIN memory_nodes n "
                "ON n.tenant_id = ? AND n.origin_event_id = e.event_id "
                "AND n.memory_type = 'EPISODE' AND n.status = 'ACTIVE' "
                "ORDER BY e.created_at, n.id LIMIT ?",
                (
                    tenant,
                    memory_id,
                    tenant,
                    memory_id,
                    tenant,
                    memory_id,
                    memory_id,
                    tenant,
                    bounded_before,
                    bounded_after,
                    tenant,
                    bounded_limit,
                ),
            )
            .fetchall()
        )
        snapshots: list[MemorySnapshot] = []
        for row in rows:
            snapshot = self.get_current_state(
                cast(uuid.UUID, _uuid(row["id"])),
                tenant_id=tenant,
                require_current_overview=False,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def get_history(
        self,
        memory_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
        include_evidence: bool = True,
    ) -> dict[str, Any] | None:
        """Hydrate immutable versions, historical claims, and provenance."""

        state = self.get_current_state(
            memory_id,
            tenant_id=tenant_id,
            include_historical_claims=True,
            require_current_overview=False,
        )
        if state is None:
            return None
        history: dict[str, Any] = {
            "memory": state.node,
            "current_version": state.current_version,
            "versions": tuple(self.get_versions(memory_id, tenant_id=state.node.tenant_id)),
            "claims": state.claims,
            "aliases": state.aliases,
            "parents": state.parents,
            "children": state.children,
            "overview": state.overview,
        }
        if include_evidence:
            history["evidence"] = state.evidence
        return history

    def load_projection_snapshot(
        self,
        mutation_id: str | uuid.UUID,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load a fresh canonical snapshot for one Temporal projection mutation."""

        projection = self.load_mutation_projection(mutation_id, tenant_id=tenant_id)
        if projection is None:
            return None
        raw_memory_id = projection.get("memory_id") or projection.get("result_memory_id")
        memory_id = _uuid(raw_memory_id)
        if memory_id is None:
            return ProjectionSnapshot.model_validate(projection).model_dump(mode="python")
        state = self.get_current_state(
            memory_id,
            tenant_id=tenant_id or str(projection.get("tenant_id") or self.default_tenant_id),
            include_historical_claims=True,
            require_current_overview=False,
        )
        if state is None:
            return ProjectionSnapshot.model_validate(projection).model_dump(mode="python")
        current_version = state.current_version
        node_properties: dict[str, JsonValue] = {
            "memory_id": str(state.node.id),
            "tenant_id": state.node.tenant_id,
            "canonical_name": state.node.canonical_name,
            "canonical_uri": state.node.canonical_uri,
            "memory_type": state.node.memory_type,
            # ``type`` is retained as a compact projection-facing alias for
            # consumers that do not use the storage field name.
            "type": state.node.memory_type,
            "status": state.node.status,
            "revision": state.node.revision,
            "current_version_id": (
                str(state.node.current_version_id)
                if state.node.current_version_id is not None
                else None
            ),
            "metadata": state.node.metadata,
        }
        if current_version is not None:
            node_properties.update(
                {
                    "body": current_version.body,
                    "abstract": current_version.abstract,
                    "current_body": current_version.body,
                    "current_abstract": current_version.abstract,
                    "version_number": current_version.version_number,
                    "asserted_at": current_version.asserted_at,
                    "valid_from": current_version.valid_from,
                    "valid_until": current_version.valid_until,
                    "version_metadata": current_version.metadata,
                    "provenance": current_version.provenance,
                }
            )
        snapshot = dict(projection)
        snapshot.update(
            {
                "memory_id": str(state.node.id),
                "canonical_revision": state.node.revision,
                "revision": state.node.revision,
                "node": _record_to_json(state.node),
                "memory": _record_to_json(state.node),
                "current_version": _record_to_json(current_version),
                "node_properties": _record_to_json(node_properties),
                "properties": _record_to_json(node_properties),
                "claims": _record_to_json(state.claims),
                "aliases": _record_to_json(state.aliases),
                "evidence": _record_to_json(state.evidence),
                "parents": _record_to_json(state.parents),
                "children": _record_to_json(state.children),
                "overview": _record_to_json(state.overview),
            }
        )

        # Claim dispatches carry only stable identifiers in the outbox.  The
        # projection worker must hydrate the matching canonical claim before
        # deciding whether it can create a graph relationship.  Keep the
        # subject/node snapshot available for both relationship and scalar
        # claims, while flattening the relationship fields expected by the
        # Neo4j adapter.
        raw_claim_id = projection.get("claim_id")
        claim_id = _uuid(raw_claim_id)
        if claim_id is not None:
            claim = next((item for item in state.claims if item.id == claim_id), None)
            if claim is not None:
                claim_properties: dict[str, JsonValue] = {
                    "claim_id": str(claim.id),
                    "tenant_id": claim.tenant_id,
                    "subject_memory_id": str(claim.subject_id),
                    "predicate": claim.predicate,
                    "object_type": claim.object_type,
                    "status": claim.status,
                    "confidence": claim.confidence,
                    "valid_from": claim.valid_from,
                    "valid_until": claim.valid_until,
                    "asserted_at": claim.asserted_at,
                    "source_event_id": claim.source_event_id,
                    "source_version_id": (
                        str(claim.source_version_id)
                        if claim.source_version_id is not None
                        else None
                    ),
                    "metadata": claim.metadata,
                }
                flattened: dict[str, JsonValue] = {
                    "claim": _record_to_json(claim),
                    "claim_id": str(claim.id),
                    "subject_memory_id": str(claim.subject_id),
                    "predicate": claim.predicate,
                    "status": claim.status,
                    "claim_status": claim.status,
                    "claim_confidence": claim.confidence,
                    "valid_from": claim.valid_from,
                    "valid_until": claim.valid_until,
                    "asserted_at": claim.asserted_at,
                }
                if claim.object_entity_id is not None:
                    object_id = str(claim.object_entity_id)
                    flattened.update(
                        {
                            "object_memory_id": object_id,
                            "object_id": object_id,
                            "object_type": "ENTITY",
                        }
                    )
                    claim_properties["object_memory_id"] = object_id
                    claim_properties["object_id"] = object_id
                else:
                    # Scalar claims are canonical data but are not Neo4j
                    # relationships.  Keep their value in the node/claim
                    # properties and deliberately remove the top-level
                    # claim_id so the existing relationship adapter cannot
                    # mistake a scalar for an entity-to-entity edge.
                    flattened.update(
                        {
                            "object_type": claim.object_type,
                            "object_value": claim.object_value,
                            "is_scalar_claim": True,
                            "scalar_claim_id": str(claim.id),
                        }
                    )
                    claim_properties.update(
                        {
                            "object_value": claim.object_value,
                            "is_scalar_claim": True,
                        }
                    )
                    flattened["claim_id"] = None
                    flattened["aggregate_type"] = "MEMORY"
                    flattened["operation"] = "UPSERT_NODE"
                flattened["properties"] = _record_to_json(claim_properties)
                snapshot.update(flattened)
        return ProjectionSnapshot.model_validate(snapshot).model_dump(mode="python")

    def retire_memory(
        self,
        memory_id: uuid.UUID,
        *,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
        emit_projection: bool = True,
    ) -> MemoryNode:
        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    mutation_key=mutation_key,
                    operation="RETIRE_MEMORY",
                    event_id=None,
                    payload={"memory_id": str(memory_id)},
                )
                if not created and mutation.status == "APPLIED":
                    return self._require_node(conn, tenant, memory_id)
            node = self._require_node(conn, tenant, memory_id, lock=True)
            if node.status != "HISTORICAL":
                row = conn.execute(
                    "UPDATE memory_nodes SET status = 'HISTORICAL', revision = revision + 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ? RETURNING *",
                    (tenant, memory_id),
                ).fetchone()
                conn.execute(
                    "UPDATE memory_claims SET status = 'HISTORICAL', updated_at = CURRENT_TIMESTAMP "
                    "WHERE tenant_id = ? AND status = 'ACTIVE' "
                    "AND (subject_id = ? OR object_entity_id = ?)",
                    (tenant, memory_id, memory_id),
                )
            else:
                row = conn.execute(
                    "SELECT * FROM memory_nodes WHERE tenant_id = ? AND id = ?",
                    (tenant, memory_id),
                ).fetchone()
            result = _node_from_row(row)
            if emit_projection:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(memory_id),
                    operation="RETIRE_NODE",
                    revision=result.revision,
                    payload={"memory_id": str(memory_id), "status": result.status},
                    canonical_mutation_id=mutation.id if mutation is not None else None,
                )
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=memory_id,
                    result={"memory_id": str(memory_id), "status": result.status},
                )
            return result

    def unmerge_entity(
        self,
        merged_id: uuid.UUID,
        splits: Sequence[Mapping[str, Any] | MemoryNode],
        *,
        tenant_id: str | None = None,
        mutation_key: str | None = None,
    ) -> list[MemoryNode]:
        """Create canonical split entities and retire a merged entity atomically.

        Model-driven split selection remains an ingest/consolidation concern;
        this operation accepts already validated split descriptors and handles
        only durable canonical state.
        """

        tenant = self._tenant(tenant_id)
        with self._transaction(tenant) as conn:
            mutation: CanonicalMutation | None = None
            if mutation_key:
                mutation, created = self._begin_mutation_in_tx(
                    conn,
                    tenant_id=tenant,
                    mutation_key=mutation_key,
                    operation="UNMERGE_ENTITY",
                    event_id=None,
                    payload={"merged_id": str(merged_id), "split_count": len(splits)},
                )
                if not created and mutation.status == "APPLIED":
                    ids = mutation.result.get("split_ids") or []
                    return [self._require_node(conn, tenant, uuid.UUID(str(item))) for item in ids]
            merged = self._require_node(conn, tenant, merged_id, lock=True)
            if merged.memory_type != "ENTITY":
                raise ValueError("unmerge_entity requires an ENTITY memory")
            result_nodes: list[MemoryNode] = []
            for index, descriptor in enumerate(splits):
                if isinstance(descriptor, MemoryNode):
                    name = descriptor.canonical_name or str(descriptor.id)
                    split_id = descriptor.id
                    metadata = descriptor.metadata
                    origin = descriptor.origin_event_id
                else:
                    name = str(
                        descriptor.get("canonical_name") or descriptor.get("name") or ""
                    ).strip()
                    if not name:
                        raise ValueError(f"split {index} has no canonical name")
                    split_id = _uuid(descriptor.get("id")) or (
                        _deterministic_uuid(
                            _MUTATION_NAMESPACE, tenant, mutation_key, "split", index
                        )
                        if mutation_key
                        else uuid.uuid4()
                    )
                    metadata = _json_object(descriptor.get("metadata"))
                    origin = descriptor.get("origin_event_id")
                split_uri = str(
                    descriptor.canonical_uri
                    if isinstance(descriptor, MemoryNode)
                    else descriptor.get("canonical_uri") or f"mem://memory/{split_id}"
                )
                split_uri = _validate_uri(split_uri)
                row = conn.execute(
                    "INSERT INTO memory_nodes "
                    "(id, tenant_id, memory_type, canonical_name, canonical_uri, status, "
                    "origin_event_id, metadata) VALUES (?, ?, 'ENTITY', ?, ?, 'ACTIVE', ?, ?) "
                    "ON CONFLICT (tenant_id, canonical_uri) DO NOTHING RETURNING *",
                    (split_id, tenant, name, split_uri, origin, _json(metadata, default={})),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        "SELECT * FROM memory_nodes WHERE tenant_id = ? AND canonical_uri = ?",
                        (tenant, split_uri),
                    ).fetchone()
                if row is None:
                    raise MemoryRepositoryError(f"unable to create unmerge split: {split_uri}")
                split_node = _node_from_row(row)
                self._add_uri_alias_in_tx(
                    conn,
                    tenant_id=tenant,
                    memory_id=split_node.id,
                    uri=split_uri,
                    alias_type="CANONICAL",
                    metadata={"source": "unmerge"},
                )
                aliases = (
                    []
                    if isinstance(descriptor, MemoryNode)
                    else descriptor.get("aliases") or [name]
                )
                for alias in aliases:
                    self._add_alias_in_tx(
                        conn,
                        tenant_id=tenant,
                        entity_id=split_node.id,
                        alias=str(alias),
                        confidence=None,
                        source_event_id=None,
                        metadata={"source": "unmerge"},
                    )
                result_nodes.append(split_node)
            conn.execute(
                "UPDATE memory_nodes SET status = 'HISTORICAL', revision = revision + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ?",
                (tenant, merged_id),
            )
            conn.execute(
                "UPDATE memory_claims SET status = 'HISTORICAL', updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = ? AND status = 'ACTIVE' "
                "AND (subject_id = ? OR object_entity_id = ?)",
                (tenant, merged_id, merged_id),
            )
            merged_after = self._require_node(conn, tenant, merged_id)
            self._enqueue_dispatch_in_tx(
                conn,
                tenant_id=tenant,
                aggregate_type="MEMORY",
                aggregate_id=str(merged_id),
                operation="UNMERGE_NODE",
                revision=merged_after.revision,
                payload={
                    "memory_id": str(merged_id),
                    "split_ids": [str(item.id) for item in result_nodes],
                },
                canonical_mutation_id=mutation.id if mutation is not None else None,
            )
            for split_node in result_nodes:
                self._enqueue_dispatch_in_tx(
                    conn,
                    tenant_id=tenant,
                    aggregate_type="MEMORY",
                    aggregate_id=str(split_node.id),
                    operation="UPSERT_NODE",
                    revision=split_node.revision,
                    payload={"memory_id": str(split_node.id), "superseded_from": str(merged_id)},
                )
            if mutation is not None:
                self._complete_mutation_in_tx(
                    conn,
                    mutation,
                    result_memory_id=merged_id,
                    result={
                        "merged_id": str(merged_id),
                        "split_ids": [str(item.id) for item in result_nodes],
                    },
                )
            return result_nodes

    def _add_alias_in_tx(
        self,
        conn: Any,
        *,
        tenant_id: str,
        entity_id: uuid.UUID,
        alias: str,
        confidence: float | None,
        source_event_id: str | None,
        metadata: Mapping[str, JsonValue] | None,
    ) -> EntityAlias:
        normalized = normalize_alias(alias)
        row = conn.execute(
            "INSERT INTO entity_aliases "
            "(id, tenant_id, entity_id, alias, normalized_alias, confidence, source_event_id, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (tenant_id, entity_id, normalized_alias) DO UPDATE "
            "SET alias = entity_aliases.alias RETURNING *",
            (
                uuid.uuid4(),
                tenant_id,
                entity_id,
                str(alias).strip(),
                normalized,
                confidence,
                source_event_id,
                _json(dict(metadata or {}), default={}),
            ),
        ).fetchone()
        if row is None:
            raise MemoryRepositoryError("alias insert returned no row")
        return _alias_from_row(row)


__all__ = [
    "CanonicalMutation",
    "CodeProjectCommit",
    "EntityAlias",
    "IngestArtifact",
    "InvalidClaimError",
    "MemoryClaim",
    "MemoryEvidence",
    "MemoryHierarchy",
    "MemoryNode",
    "MemoryNotFoundError",
    "MemoryOverview",
    "MemoryPage",
    "MemoryRepository",
    "MemoryRepositoryError",
    "MemorySnapshot",
    "MemoryState",
    "MemoryUriAlias",
    "MemoryVersion",
    "ProjectionSnapshot",
    "TypedClaim",
    "TypedClaimCandidate",
    "WorkflowDispatch",
    "infer_object_type",
    "normalize_alias",
    "normalize_predicate",
    "typed_claim",
]
