"""Deepgram Nova STT provider."""

from __future__ import annotations

import time

import httpx

from modelping.models import STTRunResult
from modelping.providers.stt.base import BaseSTTProvider
from modelping.utils.audio import get_audio_duration_ms


class DeepgramSTTProvider(BaseSTTProvider):
    name = "deepgram_stt"
    api_key_env = "DEEPGRAM_API_KEY"
    base_url = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, audio_path: str, model: str) -> STTRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, 0.0, "DEEPGRAM_API_KEY not set")

        audio_duration_ms = get_audio_duration_ms(audio_path)

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                response = await client.post(
                    f"{self.base_url}?model={model}&smart_format=true",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "audio/wav",
                    },
                    content=audio_data,
                )
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result = response.json()
                text = (
                    result
                    .get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )
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
