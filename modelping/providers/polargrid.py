"""PolarGrid provider — OpenAI-compatible edge inference API.

Base URL: https://api.{region}.edge.polargrid.ai:55111
Supports: LLM chat completions, STT, TTS (same endpoint shapes as OpenAI)
"""

from __future__ import annotations

import json
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider

DEFAULT_REGION = "yvr-01"

REGION_URLS = {
    "yvr-01": "https://api.yvr-01.edge.polargrid.ai:55111",
    "yul-01": "https://api.yul-01.edge.polargrid.ai:55111",
    "yto-01": "https://api.yto-01.edge.polargrid.ai:55111",
}


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

        # Exchange API key for JWT
        try:
            async with httpx.AsyncClient(timeout=15.0) as auth_client:
                auth_resp = await auth_client.post(
                    "https://api.polargrid.ai/v1/auth/token",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                auth_resp.raise_for_status()
                jwt = auth_resp.json()["token"]
        except Exception as e:
            return self._error_result(model, f"Auth failed: {e}")

        headers = {
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # Resolve actual model ID from config (e.g. "Llama-3.3-70B-Instruct")
        from modelping.config import MODELS
        model_cfg = MODELS.get(model, {})
        api_model = model_cfg.get("model_id", model.removeprefix("polargrid/"))
        payload = {
            "model": api_model,
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
            async with httpx.AsyncClient(timeout=60.0, verify=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    buf = b""
                    async for raw in response.aiter_bytes():
                        buf += raw
                        # Server sends literal \n\n (escaped) as SSE delimiter
                        delim = b"\\n\\n" if b"\\n\\n" in buf else b"\n"
                        while delim in buf:
                            line_bytes, buf = buf.split(delim, 1)
                            line = line_bytes.decode("utf-8", errors="replace").strip()
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
                                    tokens_generated += 1
                                    continue

                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("delta", {}).get("content"):
                                tokens_generated += 1

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
