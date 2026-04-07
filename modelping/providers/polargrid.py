"""PolarGrid provider — OpenAI-compatible edge inference API.

Auth flow (per Sev, 2026-04-06):
  1. Exchange API key for JWT: POST api.polargrid.ai/v1/auth/token
  2. Use JWT as Bearer token against edge nodes: https://api.{node}.edge.polargrid.ai:55111

Supports: LLM chat completions, STT, TTS (same endpoint shapes as OpenAI)
"""

from __future__ import annotations

import json
import time
import threading

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider

# Central auth endpoint
AUTH_URL = "https://api.polargrid.ai/v1/auth/token"

# Default region — Toronto (your live node)
DEFAULT_REGION = "yto-01"

REGION_URLS = {
    "yto-01": "https://api.yto-01.edge.polargrid.ai:55111",  # Toronto
    "yvr-02": "https://api.yvr-02.edge.polargrid.ai:55111",  # Vancouver
    "yul-01": "https://api.yul-01.edge.polargrid.ai:55111",  # Montreal
}

# Simple in-process JWT cache (per api_key)
_jwt_cache: dict[str, tuple[str, float]] = {}
_jwt_lock = threading.Lock()
JWT_TTL = 55 * 60  # 55 min — refresh before typical 60 min expiry


def _get_jwt(api_key: str) -> str:
    """Exchange API key for a JWT, with in-process caching."""
    now = time.time()
    with _jwt_lock:
        cached = _jwt_cache.get(api_key)
        if cached and now < cached[1]:
            return cached[0]

    resp = httpx.post(
        AUTH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    jwt = resp.json()["token"]

    with _jwt_lock:
        _jwt_cache[api_key] = (jwt, now + JWT_TTL)

    return jwt


class PolarGridProvider(BaseProvider):
    name = "polargrid"
    api_key_env = "POLARGRID_API_KEY"

    def __init__(self, region: str = DEFAULT_REGION):
        self.region = region
        self.base_url = REGION_URLS.get(region, REGION_URLS[DEFAULT_REGION])

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, "POLARGRID_API_KEY not set")

        try:
            jwt = _get_jwt(api_key)
        except Exception as e:
            return self._error_result(model, f"JWT exchange failed: {e}")

        headers = {
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        ttft_ms = 0.0
        total_ms = 0.0
        tokens_generated = 0
        input_tokens = 0
        first_token = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0, verify=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
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
            return self._error_result(model, f"HTTP {e.response.status_code}")
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
