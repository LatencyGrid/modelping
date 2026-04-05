"""Gladia STT provider."""

from __future__ import annotations

import asyncio
import time

import httpx

from modelping.models import STTRunResult
from modelping.providers.stt.base import BaseSTTProvider
from modelping.utils.audio import get_audio_duration_ms

POLL_INTERVAL = 1.0
POLL_TIMEOUT = 120.0


class GladiaSTTProvider(BaseSTTProvider):
    name = "gladia_stt"
    api_key_env = "GLADIA_API_KEY"
    base_url = "https://api.gladia.io/v2"

    async def transcribe(self, audio_path: str, model: str) -> STTRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, 0.0, "GLADIA_API_KEY not set")

        audio_duration_ms = get_audio_duration_ms(audio_path)
        headers = {"x-gladia-key": api_key}

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 1: Upload audio
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                upload_resp = await client.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    files={"audio": ("audio.wav", audio_data, "audio/wav")},
                )
                upload_resp.raise_for_status()
                audio_url = upload_resp.json()["audio_url"]

                # Step 2: Submit pre-recorded transcription
                submit_resp = await client.post(
                    f"{self.base_url}/pre-recorded",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"audio_url": audio_url},
                )
                submit_resp.raise_for_status()
                submit_data = submit_resp.json()
                result_url = submit_data.get("result_url") or submit_data.get("id")

                # Step 3: Poll for result
                deadline = time.perf_counter() + POLL_TIMEOUT
                while True:
                    if time.perf_counter() > deadline:
                        return self._error_result(model, audio_duration_ms, "Polling timeout exceeded")
                    await asyncio.sleep(POLL_INTERVAL)
                    # result_url may be a full URL or just an ID
                    poll_url = result_url if result_url.startswith("http") else f"{self.base_url}/pre-recorded/{result_url}"
                    poll_resp = await client.get(poll_url, headers=headers)
                    poll_resp.raise_for_status()
                    poll_data = poll_resp.json()
                    status = poll_data.get("status")
                    if status == "done":
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        text = (
                            poll_data.get("result", {})
                            .get("transcription", {})
                            .get("full_transcript", "")
                        )
                        word_count = len(text.split()) if text.strip() else 0
                        break
                    elif status == "error":
                        return self._error_result(
                            model, audio_duration_ms,
                            f"Gladia error: {poll_data.get('error', 'unknown')}"
                        )

        except httpx.HTTPStatusError as e:
            return self._error_result(
                model, audio_duration_ms,
                f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:
            return self._error_result(model, audio_duration_ms, str(e))

        return self._make_result(
            model=model,
            audio_duration_ms=audio_duration_ms,
            transcription_latency_ms=elapsed_ms,
            word_count=word_count,
        )
