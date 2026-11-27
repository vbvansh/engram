"""OpenAI-compatible provider — dispatch + JSON-mode handling + error mapping."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from engram.config import CoreModelConfig, FrontierLlmConfig
from engram.models.providers import (
    _OPENAI_COMPAT_BASES,
    build_core_provider,
    build_frontier_provider,
)
from engram.models.request_context import model_request_session, opencode_headers


def test_build_core_defaults_to_muse_spark_responses_adapter():
    cfg = CoreModelConfig(api_key="x")
    p = build_core_provider(cfg)
    from engram.models.providers.openai_compat import OpenAIResponsesCoreProvider

    assert isinstance(p, OpenAIResponsesCoreProvider)
    assert p.cfg.model_path == "muse-spark-1.3-contributor"
    assert p.cfg.api_base == "https://opencode.ai/zen/go/v1"


@pytest.mark.parametrize("provider", list(_OPENAI_COMPAT_BASES.keys()))
def test_build_core_accepts_all_openai_compat_providers(provider: str):
    cfg = CoreModelConfig(
        provider=provider,  # type: ignore[arg-type]
        api_key="sk-test",
        model_path="test-model",
    )
    p = build_core_provider(cfg)
    from engram.models.providers.openai_compat import OpenAICompatCoreProvider

    assert isinstance(p, OpenAICompatCoreProvider)


def test_build_core_rejects_unknown_provider():
    # The Literal prevents this at the Pydantic layer, but the factory
    # should also gate by string name.
    cfg = CoreModelConfig(provider="openai_responses", api_key="x")
    cfg.provider = "lolcats"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        build_core_provider(cfg)


def test_build_core_accepts_local_provider_without_loading_model():
    cfg = CoreModelConfig(provider="local", model_path="./models/engram-core-dpo")
    p = build_core_provider(cfg)
    from engram.models.providers.local_provider import LocalCoreProvider

    assert isinstance(p, LocalCoreProvider)
    assert p._model is None


def test_build_core_works_for_openai_responses():
    cfg = CoreModelConfig(
        provider="openai_responses",
        api_key="sk-test",
        api_base="https://example.test/v1",
        model_path="muse-spark-1.3-contributor",
    )
    p = build_core_provider(cfg)
    from engram.models.providers.openai_compat import OpenAIResponsesCoreProvider

    assert isinstance(p, OpenAIResponsesCoreProvider)


def test_openai_responses_core_returns_parsed_json():
    from engram.models.providers.openai_compat import OpenAIResponsesCoreProvider

    cfg = CoreModelConfig(
        provider="openai_responses",
        api_key="sk-test",
        api_base="https://example.test/v1",
        model_path="muse-spark-1.3-contributor",
        temperature=1.0,
    )
    provider = OpenAIResponsesCoreProvider(cfg)
    response = MagicMock()
    response.output_text = json.dumps({"store": True, "reason": "durable fact"})
    response.usage = MagicMock(input_tokens=12, output_tokens=7)
    provider._client.responses.create = MagicMock(return_value=response)

    result = provider.complete(system_prompt="system", user_prompt="input")

    assert result.output == {"store": True, "reason": "durable fact"}
    assert result.tokens_in == 12
    assert result.tokens_out == 7
    call = provider._client.responses.create.call_args
    assert call.kwargs["model"] == "muse-spark-1.3-contributor"
    assert call.kwargs["temperature"] == 1.0


def test_openai_responses_core_sends_opencode_session_header():
    from engram.models.providers.openai_compat import OpenAIResponsesCoreProvider

    provider = OpenAIResponsesCoreProvider(
        CoreModelConfig(
            provider="openai_responses",
            api_key="sk-test",
            api_base="https://opencode.ai/zen/go/v1",
            model_path="muse-spark-1.3-contributor",
            temperature=1.0,
        )
    )
    response = MagicMock()
    response.output_text = json.dumps({"store": True, "reason": "durable fact"})
    response.usage = MagicMock(input_tokens=12, output_tokens=7)
    provider._client.responses.create = MagicMock(return_value=response)

    with model_request_session("sess-chat-1"):
        provider.complete(system_prompt="system", user_prompt="input")

    assert provider._client.responses.create.call_args.kwargs["extra_headers"] == {
        "User-Agent": "engram/0.1.0",
        "x-opencode-session": "sess-chat-1",
    }


def test_opencode_session_header_hashes_non_header_safe_work_ids():
    with model_request_session("mem://user/entities/alice"):
        headers = opencode_headers("https://opencode.ai/zen/go/v1")

    assert headers["x-opencode-session"].startswith("engram-")
    assert "/" not in headers["x-opencode-session"]


def test_openai_compat_complete_returns_parsed_json():
    """The adapter should hand back parsed JSON from the chat completion."""
    from engram.models.providers.openai_compat import OpenAICompatCoreProvider

    cfg = CoreModelConfig(provider="openai", api_key="sk-test", model_path="gpt-4o-mini")
    provider = OpenAICompatCoreProvider(cfg)

    # Monkey-patch the underlying client
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({"store": True, "reason": "fact"}),
            )
        )
    ]
    fake_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
    provider._client.chat.completions.create = MagicMock(return_value=fake_response)

    result = provider.complete(
        system_prompt="[GATE] ...",
        user_prompt="...",
    )
    assert isinstance(result.output, dict)
    assert result.output["store"] is True
    assert result.tokens_in == 20
    assert result.tokens_out == 10


def test_openai_compat_retries_on_malformed_json():
    """The correction retry keeps the configured provider temperature."""
    from engram.models.providers.openai_compat import OpenAICompatCoreProvider

    cfg = CoreModelConfig(
        provider="openai",
        api_key="sk",
        model_path="gpt-4o-mini",
        temperature=1.0,
    )
    provider = OpenAICompatCoreProvider(cfg)

    bad = MagicMock()
    bad.choices = [MagicMock(message=MagicMock(content="I think the answer is yes"))]
    bad.usage = MagicMock(prompt_tokens=5, completion_tokens=5)
    good = MagicMock()
    good.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({"store": False, "reason": "pleasantry"}),
            )
        )
    ]
    good.usage = MagicMock(prompt_tokens=6, completion_tokens=6)
    provider._client.chat.completions.create = MagicMock(side_effect=[bad, good])

    result = provider.complete(system_prompt="[GATE] ...", user_prompt="...")
    assert result.output == {"store": False, "reason": "pleasantry"}
    assert provider._client.chat.completions.create.call_count == 2
    assert [
        call.kwargs["temperature"]
        for call in provider._client.chat.completions.create.call_args_list
    ] == [1.0, 1.0]


def test_openai_compat_disables_json_mode_for_ollama():
    """Ollama doesn't reliably honour response_format; our adapter should skip it."""
    from engram.models.providers.openai_compat import OpenAICompatCoreProvider

    cfg = CoreModelConfig(
        provider="ollama",
        api_key="sk",
        api_base="http://localhost:11434/v1",
        model_path="qwen2.5:7b",
    )
    provider = OpenAICompatCoreProvider(cfg)
    assert provider._use_json_mode is False


def test_build_frontier_works_for_openai():
    cfg = FrontierLlmConfig(provider="openai", api_key="sk", model_path="gpt-4o-mini")
    p = build_frontier_provider(cfg)
    from engram.models.providers.openai_compat import OpenAICompatFrontierProvider

    assert isinstance(p, OpenAICompatFrontierProvider)


def test_build_frontier_works_for_openai_responses():
    cfg = FrontierLlmConfig(
        provider="openai_responses",
        api_key="sk-test",
        api_base="https://example.test/v1",
        model_path="muse-spark-1.3-contributor",
    )
    p = build_frontier_provider(cfg)
    from engram.models.providers.openai_compat import OpenAIResponsesFrontierProvider

    assert isinstance(p, OpenAIResponsesFrontierProvider)


def test_openai_responses_frontier_parses_verdict():
    from engram.models.providers.openai_compat import OpenAIResponsesFrontierProvider

    cfg = FrontierLlmConfig(
        provider="openai_responses",
        api_key="sk-test",
        api_base="https://example.test/v1",
        model_path="muse-spark-1.3-contributor",
        temperature=1.0,
        reasoning_effort="low",
    )
    provider = OpenAIResponsesFrontierProvider(cfg)
    response = MagicMock()
    response.output_text = json.dumps({"verdict": "ANSWER", "answer": "Stored answer"})
    response.usage = MagicMock(input_tokens=12, output_tokens=7)
    provider._client.responses.create = MagicMock(return_value=response)

    result = provider.answer(system_prompt="system", msc="context", user_query="question")

    assert result.verdict == "ANSWER"
    assert result.answer == "Stored answer"
    assert result.tokens_in == 12
    assert result.tokens_out == 7
    call = provider._client.responses.create.call_args
    assert call.kwargs["model"] == "muse-spark-1.3-contributor"
    assert call.kwargs["temperature"] == 1.0
    assert call.kwargs["reasoning"] == {"effort": "low"}


def test_openai_responses_frontier_sends_opencode_session_header():
    from engram.models.providers.openai_compat import OpenAIResponsesFrontierProvider

    provider = OpenAIResponsesFrontierProvider(
        FrontierLlmConfig(
            provider="openai_responses",
            api_key="sk-test",
            api_base="https://opencode.ai/zen/go/v1",
            model_path="muse-spark-1.3-contributor",
            temperature=1.0,
        )
    )
    response = MagicMock()
    response.output_text = json.dumps({"verdict": "ANSWER", "answer": "Stored answer"})
    response.usage = MagicMock(input_tokens=12, output_tokens=7)
    provider._client.responses.create = MagicMock(return_value=response)

    with model_request_session("sess-chat-1"):
        provider.answer(system_prompt="system", msc="context", user_query="question")

    assert provider._client.responses.create.call_args.kwargs["extra_headers"] == {
        "User-Agent": "engram/0.1.0",
        "x-opencode-session": "sess-chat-1",
    }


def test_openai_compat_frontier_maps_api_errors_to_core_model_error():
    from engram.models.core import CoreModelError
    from engram.models.providers.openai_compat import OpenAICompatFrontierProvider

    cfg = FrontierLlmConfig(
        provider="openai",
        api_key="sk",
        model_path="gpt-4o-mini",
    )
    provider = OpenAICompatFrontierProvider(cfg)
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    provider._call_chat = MagicMock(  # type: ignore[method-assign]
        side_effect=openai.APIConnectionError(request=request)
    )

    with pytest.raises(CoreModelError, match="frontier API error"):
        provider.answer(system_prompt="", msc="", user_query="test")


def test_frontier_verdict_accepts_an_echoed_label_when_an_answer_is_given():
    import pytest

    from engram.models.core import CoreModelError
    from engram.models.providers.openai_compat import _frontier_verdict

    assert _frontier_verdict({"verdict": "ANSWER", "answer": "x"}, "") == "ANSWER"
    assert _frontier_verdict({"verdict": "need_more", "answer": ""}, "") == "NEED_MORE"
    # gpt-4.1-mini echoed the MSC's evidence state instead of "ANSWER".
    assert _frontier_verdict({"verdict": "ANSWERABLE", "answer": "ten years ago"}, "") == "ANSWER"
    with pytest.raises(CoreModelError):
        _frontier_verdict({"verdict": "ANSWERABLE", "answer": ""}, "raw")
    with pytest.raises(CoreModelError):
        _frontier_verdict(["not", "a", "dict"], "raw")
