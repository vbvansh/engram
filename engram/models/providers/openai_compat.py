"""OpenAI-compatible provider.

A single adapter that talks to any service exposing the OpenAI Chat
Completions API shape. Point `api_base` at the right URL and it works
against all of:

  - **OpenAI**            `https://api.openai.com/v1`            models: gpt-4o-mini, gpt-4o
  - **Groq**              `https://api.groq.com/openai/v1`       models: llama-3.3-70b-versatile
  - **Google Gemini**     `https://generativelanguage.googleapis.com/v1beta/openai`
                                                                  models: gemini-2.0-flash, gemini-2.5-flash
  - **Ollama (local)**    `http://localhost:11434/v1`            models: qwen2.5:7b, llama3.2:3b
  - **OpenRouter**        `https://openrouter.ai/api/v1`         models: any OpenRouter catalog entry
  - **Together.ai**       `https://api.together.xyz/v1`
  - **DeepSeek**          `https://api.deepseek.com`
  - any other OpenAI-API-compatible endpoint

Structured output uses the native `response_format={"type": "json_object"}`
when the backend supports it (OpenAI, Groq, Together), otherwise falls
back to prompt-level JSON hinting (Ollama, some Gemini setups).

Every call goes through the shared `@resilient` decorator (3 attempts,
exponential backoff, per-provider circuit breaker).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

try:
    import openai
except ImportError as err:  # pragma: no cover
    raise SystemExit("`openai` package is not installed. Run `pip install openai`.") from err

from engram.config import CoreModelConfig, FrontierLlmConfig
from engram.models.core import CompletionResult, CoreModelError, CoreModelProvider
from engram.models.frontier import FrontierLLMProvider, FrontierVerdict
from engram.models.request_context import opencode_headers
from engram.resilience import resilient

log = logging.getLogger(__name__)


# Transient errors from openai-python >= 1.x.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


_CORE_SYSTEM_SUFFIX = (
    "\n\nYou must respond with a single JSON object that matches the output schema. "
    "Emit only JSON — no prose, no markdown fences, no commentary."
)

_FRONTIER_SYSTEM = (
    "You are the final-answer generator in the Engram memory system. You receive a "
    "Minimal Sufficient Context (MSC) block and a user query.\n\n"
    "Emit exactly one JSON object, no prose, no markdown fences:\n"
    "{\n"
    '  "verdict": "ANSWER" | "NEED_MORE",\n'
    '  "answer": "final answer text (required when verdict=ANSWER)",\n'
    '  "reason": "why more context is needed (required when verdict=NEED_MORE)",\n'
    '  "suggested_queries": ["focused vector query", "..."],\n'
    '  "suggested_depth": "L1" | "L2" | "L3" | "L4"\n'
    "}\n\n"
    "Use ANSWER when the MSC contains sufficient information. Use NEED_MORE only if "
    "a specific missing fact blocks the answer; cite what is missing in `reason`.\n"
    "Do not fabricate citations — only reference information present in the MSC.\n"
    "Annotations like [ACTIVE], [HISTORICAL: ...], [LOW_CONFIDENCE: 0.45] tell you "
    "the status of each retrieved memory; prefer ACTIVE and note uncertainty when "
    "using LOW_CONFIDENCE."
)


def _frontier_verdict(data: Any, raw_text: str) -> str:
    """Read the verdict of a frontier reply, tolerating near-miss labels.

    Models sometimes echo a word from the context instead of the requested
    label: gpt-4.1-mini wrote "ANSWERABLE" (the evidence state shown in the
    MSC) on ~1% of LoCoMo questions while giving a valid answer. A reply with
    an answer text counts as ANSWER; anything else is still rejected.
    """
    if not isinstance(data, dict):
        raise CoreModelError(f"frontier returned invalid verdict: {raw_text[:200]!r}")
    verdict = str(data.get("verdict") or "").strip().upper()
    if verdict in {"ANSWER", "NEED_MORE"}:
        return verdict
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return "ANSWER"
    raise CoreModelError(f"frontier returned invalid verdict: {raw_text[:200]!r}")


def _correction_max_tokens(value: int) -> int:
    """Give a JSON-correction retry room for hidden reasoning tokens.

    Reasoning-capable gateways can consume the configured output budget before
    emitting the JSON object.  Keep the first request inexpensive, but make a
    malformed/empty-output retry large enough to finish the response.
    """

    return min(max(int(value) * 2, 4096), 8192)


def _supports_json_mode(api_base: str | None) -> bool:
    """Backends where we can safely ask for `response_format=json_object`.

    Ollama chokes on unknown response_format; some Gemini shims do too.
    When in doubt, return False and rely on the prompt hint instead.
    """
    if not api_base:
        return True  # canonical OpenAI
    host = api_base.lower()
    if "ollama" in host or "localhost" in host or "127.0.0.1" in host:
        return False
    return "generativelanguage.googleapis.com" not in host


class OpenAICompatCoreProvider(CoreModelProvider):
    """Core Model via any OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, cfg: CoreModelConfig) -> None:
        if not cfg.api_key:
            raise RuntimeError("core_model.api_key is not set")
        self.cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base,  # None → api.openai.com/v1
            timeout=cfg.timeout_seconds,
            max_retries=0,  # we handle retries via @resilient
        )
        self._use_json_mode = _supports_json_mode(cfg.api_base)
        # Per-provider breaker key so one backend outage does not open another.
        host = (cfg.api_base or "openai").split("//", 1)[-1].split("/", 1)[0]
        self._breaker_key = f"openai_compat_core_{host}"

    @resilient(
        breaker="openai_compat_core",
        failure_threshold=5,
        cool_down=30.0,
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        retry_on=_RETRYABLE,
        log_context="openai_compat_core.complete",
    )
    def _call_chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ):
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_path,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        sys_text = system_prompt + _CORE_SYSTEM_SUFFIX
        if output_schema is not None:
            sys_text += "\n\nOutput schema:\n" + json.dumps(output_schema, indent=2)
        effective_max_tokens = max_tokens or self.cfg.max_tokens
        # Preserve the caller/configured value for the JSON-correction retry.
        # Some OpenAI-compatible models (for example OpenCode Go) only accept
        # a single temperature value, so silently switching it to 0.0 makes a
        # recoverable malformed-JSON response become a hard provider failure.
        effective_temperature = self.cfg.temperature if temperature is None else temperature
        started = time.perf_counter()
        try:
            resp = self._call_chat(
                system=sys_text,
                user=user_prompt,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )
        except openai.APIError as err:
            raise CoreModelError(f"openai-compat API error: {err}") from err
        latency_ms = (time.perf_counter() - started) * 1000
        raw_text = resp.choices[0].message.content or ""

        try:
            output = self.extract_json(raw_text)
        except CoreModelError:
            # One error-correcting retry for malformed JSON.
            log.info("openai_compat_core: malformed JSON, retrying with correction")
            retry_user = (
                f"{user_prompt}\n\nYour previous response was not valid JSON. "
                "Respond with ONLY the JSON object, no prose or fences."
            )
            resp = self._call_chat(
                system=sys_text,
                user=retry_user,
                max_tokens=_correction_max_tokens(effective_max_tokens),
                temperature=effective_temperature,
            )
            raw_text = resp.choices[0].message.content or ""
            output = self.extract_json(raw_text)
            latency_ms = (time.perf_counter() - started) * 1000

        usage = getattr(resp, "usage", None)
        return CompletionResult(
            output=output,
            raw_text=raw_text,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )


class OpenAIResponsesCoreProvider(CoreModelProvider):
    """Core Model through the OpenAI Responses API shape.

    This is used for gateways where a selected model, such as Muse Spark,
    is exposed through ``/responses`` but not ``/chat/completions``.
    """

    def __init__(self, cfg: CoreModelConfig) -> None:
        if not cfg.api_key:
            raise RuntimeError("core_model.api_key is not set")
        self.cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base,
            timeout=cfg.timeout_seconds,
            max_retries=0,
        )

    @resilient(
        breaker="openai_responses_core",
        failure_threshold=5,
        cool_down=30.0,
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        retry_on=_RETRYABLE,
        log_context="openai_responses_core.complete",
    )
    def _call_responses(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ):
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_path,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        # Muse Spark otherwise may consume the entire output budget on hidden
        # reasoning and return no visible JSON to the gate/extractor.
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}
        headers = opencode_headers(self.cfg.api_base)
        if headers:
            kwargs["extra_headers"] = headers
        return self._client.responses.create(**kwargs)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        sys_text = system_prompt + _CORE_SYSTEM_SUFFIX
        if output_schema is not None:
            sys_text += "\n\nOutput schema:\n" + json.dumps(output_schema, indent=2)
        effective_max_tokens = max_tokens or self.cfg.max_tokens
        effective_temperature = self.cfg.temperature if temperature is None else temperature
        started = time.perf_counter()
        try:
            resp = self._call_responses(
                system=sys_text,
                user=user_prompt,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )
        except openai.APIError as err:
            raise CoreModelError(f"openai-responses API error: {err}") from err
        # Let the normal JSON correction path handle an empty first response;
        # only the final retry should raise the richer provider diagnostic.
        raw_text = _response_output_text(resp)

        try:
            output = self.extract_json(raw_text)
        except CoreModelError:
            log.info("openai_responses_core: malformed JSON, retrying with correction")
            resp = self._call_responses(
                system=sys_text,
                user=(
                    f"{user_prompt}\n\nYour previous response was not valid JSON. "
                    "Respond with ONLY the JSON object, no prose or fences."
                ),
                max_tokens=_correction_max_tokens(effective_max_tokens),
                temperature=effective_temperature,
            )
            raw_text = _require_response_output_text(resp)
            output = self.extract_json(raw_text)

        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            output=output,
            raw_text=raw_text,
            tokens_in=getattr(usage, "input_tokens", None) if usage else None,
            tokens_out=getattr(usage, "output_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )


class OpenAICompatFrontierProvider(FrontierLLMProvider):
    """Frontier LLM via any OpenAI-compatible endpoint."""

    def __init__(self, cfg: FrontierLlmConfig) -> None:
        if not cfg.api_key:
            raise RuntimeError("frontier_llm.api_key is not set")
        self.cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=getattr(cfg, "api_base", None),
            timeout=60.0,
            max_retries=0,
        )
        self._use_json_mode = _supports_json_mode(getattr(cfg, "api_base", None))

    @resilient(
        breaker="openai_compat_frontier",
        failure_threshold=5,
        cool_down=30.0,
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        retry_on=_RETRYABLE,
        log_context="openai_compat_frontier.answer",
    )
    def _call_chat(self, *, system: str, user: str):
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_path,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def answer(
        self,
        *,
        system_prompt: str,
        msc: str,
        user_query: str,
        allow_need_more: bool = True,
    ) -> FrontierVerdict:
        system_text = (system_prompt or "") + "\n\n" + _FRONTIER_SYSTEM
        if not allow_need_more:
            system_text += (
                "\n\nThis is the final call. You MUST emit ANSWER with a best-effort "
                "answer even if context is incomplete."
            )
        user_text = f"<msc>\n{msc}\n</msc>\n\n<user_query>\n{user_query}\n</user_query>"
        started = time.perf_counter()
        try:
            resp = self._call_chat(system=system_text, user=user_text)
        except openai.APIError as err:
            raise CoreModelError(f"openai-compat frontier API error: {err}") from err
        latency_ms = (time.perf_counter() - started) * 1000
        raw_text = resp.choices[0].message.content or ""

        try:
            data = CoreModelProvider.extract_json(raw_text)
        except CoreModelError as err:
            raise CoreModelError(
                f"frontier returned unparseable output: {raw_text[:200]!r}"
            ) from err
        verdict = _frontier_verdict(data, raw_text)

        usage = getattr(resp, "usage", None)
        return FrontierVerdict(
            verdict=verdict,
            answer=data.get("answer"),
            reason=data.get("reason"),
            suggested_queries=data.get("suggested_queries") or [],
            suggested_depth=data.get("suggested_depth"),
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )

    def stream_answer(  # type: ignore[override]
        self,
        *,
        system_prompt: str,
        msc: str,
        user_query: str,
    ) -> Iterator[str]:
        """Stream the final ANSWER. Only call after a buffered verdict=ANSWER."""
        system_text = (
            (system_prompt or "") + "\n\nRespond with the final answer text only. "
            "No JSON, no 'ANSWER:' prefix, no markdown fences."
        )
        user_text = f"<msc>\n{msc}\n</msc>\n\n<user_query>\n{user_query}\n</user_query>"
        stream = self._client.chat.completions.create(
            model=self.cfg.model_path,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


class OpenAIResponsesFrontierProvider(FrontierLLMProvider):
    """Frontier LLM through OpenAI's Responses API shape.

    Some OpenAI-compatible gateways expose selected models only through
    ``/responses`` rather than ``/chat/completions``.  Keeping this adapter
    separate avoids silently sending an incompatible request to those models.
    """

    def __init__(self, cfg: FrontierLlmConfig) -> None:
        if not cfg.api_key:
            raise RuntimeError("frontier_llm.api_key is not set")
        self.cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=getattr(cfg, "api_base", None),
            timeout=60.0,
            max_retries=0,
        )

    @resilient(
        breaker="openai_responses_frontier",
        failure_threshold=5,
        cool_down=30.0,
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        retry_on=_RETRYABLE,
        log_context="openai_responses_frontier.answer",
    )
    def _call_responses(self, *, system: str, user: str):
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_path,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}
        headers = opencode_headers(getattr(self.cfg, "api_base", None))
        if headers:
            kwargs["extra_headers"] = headers
        return self._client.responses.create(**kwargs)

    def answer(
        self,
        *,
        system_prompt: str,
        msc: str,
        user_query: str,
        allow_need_more: bool = True,
    ) -> FrontierVerdict:
        system_text = (system_prompt or "") + "\n\n" + _FRONTIER_SYSTEM
        if not allow_need_more:
            system_text += (
                "\n\nThis is the final call. You MUST emit ANSWER with a best-effort "
                "answer even if context is incomplete."
            )
        user_text = f"<msc>\n{msc}\n</msc>\n\n<user_query>\n{user_query}\n</user_query>"
        started = time.perf_counter()
        try:
            resp = self._call_responses(system=system_text, user=user_text)
        except openai.APIError as err:
            raise CoreModelError(f"openai-responses frontier API error: {err}") from err
        latency_ms = (time.perf_counter() - started) * 1000
        raw_text = _response_output_text(resp)

        try:
            data = CoreModelProvider.extract_json(raw_text)
        except CoreModelError as err:
            raise CoreModelError(
                f"frontier returned unparseable output: {raw_text[:200]!r}"
            ) from err
        verdict = _frontier_verdict(data, raw_text)

        usage = getattr(resp, "usage", None)
        return FrontierVerdict(
            verdict=verdict,
            answer=data.get("answer"),
            reason=data.get("reason"),
            suggested_queries=data.get("suggested_queries") or [],
            suggested_depth=data.get("suggested_depth"),
            tokens_in=getattr(usage, "input_tokens", None) if usage else None,
            tokens_out=getattr(usage, "output_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )


def _response_output_text(response: Any) -> str:
    """Read output from standard and OpenAI-compatible Responses payloads."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
    return "".join(parts)


def _require_response_output_text(response: Any) -> str:
    """Return visible text or raise an actionable provider diagnostic."""
    text = _response_output_text(response)
    if text.strip():
        return text

    status = getattr(response, "status", "unknown")
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete else None
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    detail = f"status={status}"
    if reason:
        detail += f", incomplete_reason={reason}"
    if output_tokens is not None:
        detail += f", output_tokens={output_tokens}"
    raise CoreModelError(
        "openai-responses returned no visible output "
        f"({detail}); reduce reasoning effort or increase max_tokens"
    )
