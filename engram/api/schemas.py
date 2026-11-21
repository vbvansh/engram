"""Request/response schemas for the REST API (§11).

All string fields carry explicit length caps. Combined with
`engram.api.body_limit.BodySizeLimitMiddleware` this gives defence in
depth against oversized payloads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --- Per-field caps (characters, not tokens) -----------------------------
MAX_CONTENT_LENGTH = 32_000  # per turn
MAX_QUERY_LENGTH = 8_000  # single user query
MAX_SESSION_CONTEXT_LENGTH = 64_000
MAX_SESSION_SUMMARY_LENGTH = 16_000
MAX_ID_LENGTH = 128


class TurnContent(BaseModel):
    content: str = Field(..., max_length=MAX_CONTENT_LENGTH)
    timestamp: str | None = Field(default=None, max_length=64)
    turn_idx: int | None = Field(default=None, ge=0, le=1_000_000)
    # Optional tool-call / tool-result payloads (§5.2 turn groups).
    tool_calls: list[dict[str, Any]] | None = Field(default=None, max_length=20)
    tool_results: list[dict[str, Any]] | None = Field(default=None, max_length=20)


class TurnPair(BaseModel):
    user: TurnContent
    assistant: TurnContent


class TurnGroup(BaseModel):
    """Extended ingest unit (§5.2) covering tool-using assistant flows.

    The conventional shape is:
        user → assistant(tool_call) → tool_result → assistant(final_response)

    The outer API collapses this into `user_turn + assistant_turn` (the first
    user turn and the final assistant turn) so the downstream gate/extract
    pipeline still operates on a pair, with the tool steps preserved in the
    event payload for provenance.
    """

    user: TurnContent
    assistant: TurnContent
    intermediate: list[TurnContent] | None = None  # tool-call, tool-result, etc.


class IngestRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    turn_pair: TurnPair | None = None
    turn_group: TurnGroup | None = None
    source: str = Field(default="client", max_length=32)
    session_summary: str | None = Field(default=None, max_length=MAX_SESSION_SUMMARY_LENGTH)
    session_context: str | None = Field(default=None, max_length=MAX_SESSION_CONTEXT_LENGTH)
    force_store: bool = False

    def effective_pair(self) -> TurnPair:
        if self.turn_pair is not None:
            return self.turn_pair
        if self.turn_group is not None:
            return TurnPair(user=self.turn_group.user, assistant=self.turn_group.assistant)
        raise ValueError("IngestRequest requires either turn_pair or turn_group")

    @field_validator("session_id", "source")
    @classmethod
    def _safe_ascii(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Session IDs and source tags should be safe for logs, filenames, metric labels.
        if any(c.isspace() for c in v):
            raise ValueError("must not contain whitespace")
        return v


class IngestResponse(BaseModel):
    event_id: str
    pair_id: str
    status: str
    projection_status: Literal["PENDING", "PROJECTING", "INDEXED", "FAILED"] = "PENDING"


class QueryRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    session_context: str | None = Field(default=None, max_length=MAX_SESSION_CONTEXT_LENGTH)
    max_depth: str | None = Field(default=None, pattern=r"^L[0-4]$|^SESSION$")
    max_reentries: int | None = Field(default=None, ge=0, le=5)
    stream: bool = False
    # "short" answers with a short phrase instead of sentences (benchmark QA
    # formats such as LoCoMo's). Memory search is the same for both styles.
    answer_style: Literal["default", "short"] = "default"
    # Extra wording shown only to the answer step, never to memory search,
    # e.g. a benchmark's answer options or "answer with an approximate date".
    answer_instructions: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)


class QueryResponse(BaseModel):
    answer: str
    session_id: str | None = None
    retrieval_metadata: dict[str, Any]
    answerability: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    verification_rejections: list[str] = Field(default_factory=list)
    route_metadata: dict[str, Any] = Field(default_factory=dict)
    projection_metadata: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=MAX_CONTENT_LENGTH)


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionMessage] = Field(..., min_length=1)
    session_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    stream: bool = False
    session_context: str | None = Field(default=None, max_length=MAX_SESSION_CONTEXT_LENGTH)
    max_depth: str | None = Field(default=None, pattern=r"^L[0-4]$|^SESSION$")
    max_reentries: int | None = Field(default=None, ge=0, le=5)

    @field_validator("messages")
    @classmethod
    def _messages_rules(cls, v: list[ChatCompletionMessage]) -> list[ChatCompletionMessage]:
        if not any(m.role == "user" for m in v):
            raise ValueError("messages must contain at least one user message")
        if v[-1].role != "user":
            raise ValueError("last message must be from user")
        return v

    @field_validator("session_id")
    @classmethod
    def _safe_ascii(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if any(c.isspace() for c in v):
            raise ValueError("must not contain whitespace")
        return v


class ChatCompletionResponse(BaseModel):
    answer: str
    session_id: str
    retrieval_metadata: dict[str, Any]
    answerability: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    verification_rejections: list[str] = Field(default_factory=list)
    finish_reason: str = "stop"


class HealthResponse(BaseModel):
    status: str
    components: dict[str, bool]


class ConfigResponse(BaseModel):
    retrieval: dict[str, Any]
    core_model: dict[str, Any]
    frontier_llm: dict[str, Any]
