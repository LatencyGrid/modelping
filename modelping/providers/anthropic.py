"""Anthropic provider — uses Anthropic Messages API with streaming."""

from __future__ import annotations

import json
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1"

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, "ANTHROPIC_API_KEY not set")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        ttft_ms = 0.0
        total_ms = 0.0
        tokens_generated = 0
        input_tokens = 0
        first_token = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            event_type = event.get("type", "")

                            if event_type == "content_block_delta":
                                if not first_token:
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta" and delta.get("text"):
                                        ttft_ms = (time.perf_counter() - start) * 1000
                                        first_token = True

                            elif event_type == "message_delta":
                                usage = event.get("usage", {})
                                tokens_generated = usage.get("output_tokens", tokens_generated)

                            elif event_type == "message_start":
                                msg = event.get("message", {})
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", input_tokens)

            total_ms = (time.perf_counter() - start) * 1000

        except httpx.HTTPStatusError as e:
            return self._error_result(model, f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return self._error_result(model, str(e))

        if not first_token:
            ttft_ms = total_ms

        return self._make_run_result(
            model=model,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=tokens_generated,
            input_tokens=input_tokens,
        )
