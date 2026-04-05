"""Deepgram Aura TTS provider."""

from __future__ import annotations

import time

import httpx

from modelping.models import TTSRunResult
from modelping.providers.tts.base import BaseTTSProvider


class DeepgramTTSProvider(BaseTTSProvider):
    name = "deepgram_tts"
    api_key_env = "DEEPGRAM_API_KEY"
    base_url = "https://api.deepgram.com/v1/speak"

    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, len(text), "DEEPGRAM_API_KEY not set")

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        ttfb_ms = 0.0
        total_bytes = 0
        first_chunk = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}?model={model}",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        if chunk and not first_chunk:
                            ttfb_ms = (time.perf_counter() - start) * 1000
                            first_chunk = True
                        total_bytes += len(chunk)
            total_ms = (time.perf_counter() - start) * 1000

        except httpx.HTTPStatusError as e:
            return self._error_result(
                model, len(text),
                f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:
            return self._error_result(model, len(text), str(e))

        if not first_chunk:
            ttfb_ms = total_ms

        # Linear PCM 16-bit at 24kHz (Deepgram Aura default)
        num_samples = total_bytes / 2  # 16-bit = 2 bytes per sample
        audio_duration_ms = (num_samples / 24000) * 1000

        return self._make_result(
            model=model,
            text_chars=len(text),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        )
