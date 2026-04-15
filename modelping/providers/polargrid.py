"""PolarGrid provider — OpenAI-compatible edge inference API.

Supports remote edge regions and local inference servers.

Environment variables:
  POLARGRID_API_KEY    — Bearer token / JWT for auth
  POLARGRID_REGION     — Region ID (tor-01, yvr-01, …) or "local" (default: tor-01)
  POLARGRID_BASE_URL   — Custom base URL, overrides region lookup
                         e.g. http://24.84.229.106:8000  or  http://localhost:8000
  POLARGRID_VERIFY_SSL — Set to "false" to skip TLS verification (default: true)
"""

from __future__ import annotations

import json
import os
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider

# Default region — Toronto (your live node)
DEFAULT_REGION = "tor-01"

REGION_URLS: dict[str, str] = {
    "tor-01": "https://api.tor-01.edge.polargrid.ai:55111",
    "yvr-01": "https://api.yvr-01.edge.polargrid.ai:55111",
    "ymq-01": "https://api.ymq-01.edge.polargrid.ai:55111",
    "was-01": "https://api.was-01.edge.polargrid.ai:55111",
}

# Friendly model name → actual Triton/vLLM model ID on the server
MODEL_ID_MAP: dict[str, str] = {
    "polargrid/llama-3.1-8b": "Meta-Llama-3.1-8B-Instruct",
    "polargrid/llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct",
}


def _resolve_base_url(region: str) -> str:
    """Resolve base URL from env override, region name, or default."""
    env_url = os.environ.get("POLARGRID_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return REGION_URLS.get(region, REGION_URLS[DEFAULT_REGION])


def _verify_ssl() -> bool:
    """Return False when user opts out of TLS verification (self-signed certs)."""
    val = os.environ.get("POLARGRID_VERIFY_SSL", "true").strip().lower()
    return val not in ("false", "0", "no")


class PolarGridProvider(BaseProvider):
    name = "polargrid"
    api_key_env = "POLARGRID_API_KEY"

    def __init__(self, region: str | None = None):
        self.region = region or os.environ.get("POLARGRID_REGION", DEFAULT_REGION)
        self.base_url = _resolve_base_url(self.region)
        self.verify_ssl = _verify_ssl()

    @property
    def effective_base_url(self) -> str:
        """CLI --base-url > POLARGRID_BASE_URL env > region lookup."""
        if self._base_url_override:
            return self._base_url_override
        return self.base_url

    def resolve_model(self, model: str) -> str:
        """CLI --model-id > MODEL_ID_MAP > model as-is."""
        if self._model_id:
            return self._model_id
        return MODEL_ID_MAP.get(model, model)

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Map friendly name → server model ID
        model_id = self.resolve_model(model)

        payload = {
            "model": model_id,
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
            verify = self._verify_ssl and self.verify_ssl
            async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
                async with client.stream(
                    "POST",
                    f"{self.effective_base_url}/v1/chat/completions",
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
