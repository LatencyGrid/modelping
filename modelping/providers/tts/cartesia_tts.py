"""Cartesia TTS provider."""

from __future__ import annotations

import time

import httpx

from modelping.models import TTSRunResult
from modelping.providers.tts.base import BaseTTSProvider

DEFAULT_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Barbershop Man (public)


class CartesiaTTSProvider(BaseTTSProvider):
    name = "cartesia_tts"
    api_key_env = "CARTESIA_API_KEY"
    base_url = "https://api.cartesia.ai/tts/bytes"

    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, len(text), "CARTESIA_API_KEY not set")

        headers = {
            "X-API-Key": api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": model,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": DEFAULT_VOICE_ID,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
        }

        ttfb_ms = 0.0
        total_bytes = 0
        first_chunk = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
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

        # PCM f32le at 44100 Hz: 4 bytes per sample, 1 channel
        num_samples = total_bytes / 4
        audio_duration_ms = (num_samples / 44100) * 1000

        return self._make_result(
            model=model,
            text_chars=len(text),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        )
