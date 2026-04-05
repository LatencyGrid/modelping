"""Tests for provider classes with mocked httpx responses."""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from modelping.models import RunResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sse_lines(*chunks: dict) -> list[str]:
    """Build SSE lines from JSON chunks."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


def _async_iter(items):
    """Return an async iterator over items."""
    async def _gen():
        for item in items:
            yield item
    return _gen()


class MockStreamResponse:
    """Mock for httpx streaming context manager."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            request = MagicMock()
            response = MagicMock()
            response.status_code = self.status_code
            response.text = "Error"
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def aiter_lines(self):
        return _async_iter(self._lines)


class MockAsyncClient:
    """Mock for httpx.AsyncClient context manager."""

    def __init__(self, stream_response: MockStreamResponse):
        self._stream_response = stream_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, *args, **kwargs):
        return self._stream_response


# ── OpenAI ────────────────────────────────────────────────────────────────────

class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.openai import OpenAIProvider

        chunks = [
            {"choices": [{"delta": {"content": "Neural"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " networks"}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 25},
            },
        ]
        lines = _make_sse_lines(*chunks)
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("modelping.providers.openai.httpx.AsyncClient", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.measure("gpt-4o", "test prompt")

        assert result.error is None
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.ttft_ms >= 0
        assert result.total_ms >= 0
        assert result.input_tokens == 12
        assert result.tokens_generated == 25

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.openai import OpenAIProvider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider()
        result = await provider.measure("gpt-4o", "test")
        assert result.error is not None
        assert "OPENAI_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch):
        from modelping.providers.openai import OpenAIProvider

        mock_resp = MockStreamResponse([], status_code=401)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
        with patch("modelping.providers.openai.httpx.AsyncClient", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.measure("gpt-4o", "test")

        assert result.error is not None
        assert "401" in result.error


# ── Anthropic ─────────────────────────────────────────────────────────────────

class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.anthropic import AnthropicProvider

        chunks = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 15}}},
            {"type": "content_block_start", "index": 0},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Neural"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " networks"}},
            {"type": "message_delta", "usage": {"output_tokens": 30}},
            {"type": "message_stop"},
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("modelping.providers.anthropic.httpx.AsyncClient", return_value=mock_client):
            provider = AnthropicProvider()
            result = await provider.measure("claude-3-5-sonnet-20241022", "test")

        assert result.error is None
        assert result.provider == "anthropic"
        assert result.ttft_ms >= 0
        assert result.input_tokens == 15
        assert result.tokens_generated == 30

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.anthropic import AnthropicProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicProvider()
        result = await provider.measure("claude-3-5-sonnet-20241022", "test")
        assert result.error is not None
        assert "ANTHROPIC_API_KEY" in result.error


# ── Google ────────────────────────────────────────────────────────────────────

class TestGoogleProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.google import GoogleProvider

        chunks = [
            {
                "candidates": [{"content": {"parts": [{"text": "Neural networks"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            },
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("modelping.providers.google.httpx.AsyncClient", return_value=mock_client):
            provider = GoogleProvider()
            result = await provider.measure("gemini-2.0-flash", "test")

        assert result.error is None
        assert result.provider == "google"
        assert result.ttft_ms >= 0
        assert result.input_tokens == 10
        assert result.tokens_generated == 20

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.google import GoogleProvider

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GoogleProvider()
        result = await provider.measure("gemini-2.0-flash", "test")
        assert result.error is not None


# ── Groq ──────────────────────────────────────────────────────────────────────

class TestGroqProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.groq import GroqProvider

        chunks = [
            {"choices": [{"delta": {"content": "Fast"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "x_groq": {"usage": {"prompt_tokens": 8, "completion_tokens": 15}},
            },
        ]
        lines = _make_sse_lines(*chunks)
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with patch("modelping.providers.groq.httpx.AsyncClient", return_value=mock_client):
            provider = GroqProvider()
            result = await provider.measure("llama-3.3-70b-versatile", "test")

        assert result.error is None
        assert result.provider == "groq"
        assert result.input_tokens == 8
        assert result.tokens_generated == 15

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.groq import GroqProvider

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqProvider()
        result = await provider.measure("llama-3.3-70b-versatile", "test")
        assert result.error is not None


# ── Fireworks ─────────────────────────────────────────────────────────────────

class TestFireworksProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.fireworks import FireworksProvider

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            },
        ]
        lines = _make_sse_lines(*chunks)
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        with patch("modelping.providers.fireworks.httpx.AsyncClient", return_value=mock_client):
            provider = FireworksProvider()
            result = await provider.measure(
                "accounts/fireworks/models/llama-v3p1-70b-instruct", "test"
            )

        assert result.error is None
        assert result.provider == "fireworks"

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.fireworks import FireworksProvider

        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        provider = FireworksProvider()
        result = await provider.measure("accounts/fireworks/models/llama-v3p1-70b-instruct", "test")
        assert result.error is not None


# ── Together ──────────────────────────────────────────────────────────────────

class TestTogetherProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.together import TogetherProvider

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            },
        ]
        lines = _make_sse_lines(*chunks)
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
        with patch("modelping.providers.together.httpx.AsyncClient", return_value=mock_client):
            provider = TogetherProvider()
            result = await provider.measure("meta-llama/Llama-3.3-70B-Instruct-Turbo", "test")

        assert result.error is None
        assert result.provider == "together"

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.together import TogetherProvider

        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        provider = TogetherProvider()
        result = await provider.measure("meta-llama/Llama-3.3-70B-Instruct-Turbo", "test")
        assert result.error is not None


# ── Mistral ───────────────────────────────────────────────────────────────────

class TestMistralProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.mistral import MistralProvider

        chunks = [
            {"choices": [{"delta": {"content": "A neural"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 20},
            },
        ]
        lines = _make_sse_lines(*chunks)
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("modelping.providers.mistral.httpx.AsyncClient", return_value=mock_client):
            provider = MistralProvider()
            result = await provider.measure("mistral-large-latest", "test")

        assert result.error is None
        assert result.provider == "mistral"
        assert result.tokens_generated == 20

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.mistral import MistralProvider

        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        provider = MistralProvider()
        result = await provider.measure("mistral-large-latest", "test")
        assert result.error is not None


# ── Cohere ────────────────────────────────────────────────────────────────────

class TestCohereProvider:
    @pytest.mark.asyncio
    async def test_successful_measure(self, monkeypatch):
        from modelping.providers.cohere import CohereProvider

        events = [
            {"type": "message-start", "id": "abc"},
            {"type": "content-start", "index": 0},
            {
                "type": "content-delta",
                "index": 0,
                "delta": {"message": {"content": {"type": "text", "text": "Neural"}}},
            },
            {
                "type": "message-end",
                "delta": {
                    "finish_reason": "COMPLETE",
                    "usage": {"billed_units": {"input_tokens": 9, "output_tokens": 18}},
                },
            },
        ]
        lines = [json.dumps(e) for e in events]
        mock_resp = MockStreamResponse(lines)
        mock_client = MockAsyncClient(mock_resp)

        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        with patch("modelping.providers.cohere.httpx.AsyncClient", return_value=mock_client):
            provider = CohereProvider()
            result = await provider.measure("command-r-plus", "test")

        assert result.error is None
        assert result.provider == "cohere"
        assert result.input_tokens == 9
        assert result.tokens_generated == 18

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        from modelping.providers.cohere import CohereProvider

        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        provider = CohereProvider()
        result = await provider.measure("command-r-plus", "test")
        assert result.error is not None


# ── BaseProvider helpers ──────────────────────────────────────────────────────

class TestBaseProviderHelpers:
    def test_make_run_result_tokens_per_sec(self):
        from modelping.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        result = provider._make_run_result(
            model="gpt-4o",
            ttft_ms=100.0,
            total_ms=2000.0,
            tokens_generated=200,
            input_tokens=50,
        )
        assert result.tokens_per_sec == pytest.approx(100.0)  # 200 / 2s

    def test_error_result(self):
        from modelping.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        result = provider._error_result("gpt-4o", "something went wrong")
        assert result.error == "something went wrong"
        assert result.ttft_ms == 0.0
        assert result.tokens_generated == 0
