"""PlayHT TTS provider."""

from __future__ import annotations

import os
import time

import httpx

from modelping.models import TTSRunResult
from modelping.providers.tts.base import BaseTTSProvider


class PlayHTTTSProvider(BaseTTSProvider):
    name = "playht_tts"
    api_key_env = "PLAYHT_API_KEY"
    base_url = "https://api.play.ht/api/v2/tts/stream"

    def get_user_id(self) -> str | None:
        return os.environ.get("PLAYHT_USER_ID") or None

    def is_configured(self) -> bool:
        return bool(self.get_api_key() and self.get_user_id())

    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        api_key = self.get_api_key()
        user_id = self.get_user_id()
        if not api_key:
            return self._error_result(model, len(text), "PLAYHT_API_KEY not set")
        if not user_id:
            return self._error_result(model, len(text), "PLAYHT_USER_ID not set")

        headers = {
            "AUTHORIZATION": api_key,
            "X-USER-ID": user_id,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "voice": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
            "output_format": "mp3",
            "voice_engine": model,
            "quality": "medium",
        }

        ttfb_ms = 0.0
        total_bytes = 0
        first_chunk = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
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

        # MP3 ~128kbps estimate
        audio_duration_ms = (total_bytes * 8 / 128_000) * 1000

        return self._make_result(
            model=model,
            text_chars=len(text),
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        )
