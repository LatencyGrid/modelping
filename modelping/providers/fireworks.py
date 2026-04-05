"""Fireworks AI provider — OpenAI-compatible API with streaming."""

from __future__ import annotations

import json
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider


class FireworksProvider(BaseProvider):
    name = "fireworks"
    api_key_env = "FIREWORKS_API_KEY"
    base_url = "https://api.fireworks.ai/inference/v1"

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, "FIREWORKS_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
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
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if not first_token:
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("delta", {}).get("content"):
                                ttft_ms = (time.perf_counter() - start) * 1000
                                first_token = True

                        if chunk.get("usage"):
                            usage = chunk["usage"]
                            tokens_generated = usage.get("completion_tokens", tokens_generated)
                            input_tokens = usage.get("prompt_tokens", input_tokens)

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
