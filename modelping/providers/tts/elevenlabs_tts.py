"""ElevenLabs TTS provider."""

from __future__ import annotations

import time

import httpx

from modelping.models import TTSRunResult
from modelping.providers.tts.base import BaseTTSProvider

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel


class ElevenLabsTTSProvider(BaseTTSProvider):
    name = "elevenlabs_tts"
    api_key_env = "ELEVENLABS_API_KEY"
    base_url = "https://api.elevenlabs.io/v1/text-to-speech"

    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, len(text), "ELEVENLABS_API_KEY not set")

        url = f"{self.base_url}/{DEFAULT_VOICE_ID}/stream"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        ttfb_ms = 0.0
        total_bytes = 0
        first_chunk = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
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

        # ElevenLabs returns MP3 ~128kbps; estimate duration
        audio_duration_ms = (total_bytes * 8 / 128_000) * 1000

        return self._make_result(
            model=model,
            text_chars=len(text),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        )
