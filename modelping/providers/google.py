"""Google Gemini provider — uses Gemini API directly (not Vertex) with streaming."""

from __future__ import annotations

import json
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    name = "google"
    api_key_env = "GOOGLE_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, "GOOGLE_API_KEY not set")

        url = (
            f"{self.base_url}/models/{model}:streamGenerateContent"
            f"?key={api_key}&alt=sse"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
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
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and not first_token:
                                text = parts[0].get("text", "")
                                if text:
                                    ttft_ms = (time.perf_counter() - start) * 1000
                                    first_token = True

                        # Extract usage metadata
                        usage = chunk.get("usageMetadata", {})
                        if usage:
                            tokens_generated = usage.get("candidatesTokenCount", tokens_generated)
                            input_tokens = usage.get("promptTokenCount", input_tokens)

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
