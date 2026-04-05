"""Groq Whisper STT provider."""

from __future__ import annotations

import time

import httpx

from modelping.models import STTRunResult
from modelping.providers.stt.base import BaseSTTProvider
from modelping.utils.audio import get_audio_duration_ms


class GroqSTTProvider(BaseSTTProvider):
    name = "groq_stt"
    api_key_env = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio_path: str, model: str) -> STTRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, 0.0, "GROQ_API_KEY not set")

        audio_duration_ms = get_audio_duration_ms(audio_path)

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": ("audio.wav", audio_data, "audio/wav")},
                    data={"model": model, "response_format": "json"},
                )
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result = response.json()
                text = result.get("text", "")
                word_count = len(text.split()) if text.strip() else 0

        except httpx.HTTPStatusError as e:
            return self._error_result(
                model, audio_duration_ms,
                f"HTTP {e.response.status_code}"
            )
        except Exception as e:
            return self._error_result(model, audio_duration_ms, str(e))

        return self._make_result(
            model=model,
            audio_duration_ms=audio_duration_ms,
            transcription_latency_ms=elapsed_ms,
            word_count=word_count,
        )
