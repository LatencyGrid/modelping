"""AssemblyAI STT provider — async submit + poll pattern."""

from __future__ import annotations

import asyncio
import time

import httpx

from modelping.models import STTRunResult
from modelping.providers.stt.base import BaseSTTProvider
from modelping.utils.audio import get_audio_duration_ms

POLL_INTERVAL = 1.0
POLL_TIMEOUT = 120.0


class AssemblyAISTTProvider(BaseSTTProvider):
    name = "assemblyai_stt"
    api_key_env = "ASSEMBLYAI_API_KEY"
    base_url = "https://api.assemblyai.com/v2"

    async def transcribe(self, audio_path: str, model: str) -> STTRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, 0.0, "ASSEMBLYAI_API_KEY not set")

        audio_duration_ms = get_audio_duration_ms(audio_path)
        headers = {"authorization": api_key, "content-type": "application/json"}

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 1: Upload audio
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                upload_resp = await client.post(
                    f"{self.base_url}/upload",
                    headers={"authorization": api_key, "Content-Type": "application/octet-stream"},
                    content=audio_data,
                )
                upload_resp.raise_for_status()
                upload_url = upload_resp.json()["upload_url"]

                # Step 2: Submit transcription
                payload: dict = {"audio_url": upload_url}
                if model != "default":
                    payload["speech_model"] = model
                submit_resp = await client.post(
                    f"{self.base_url}/transcript",
                    json=payload,
                    headers=headers,
                )
                submit_resp.raise_for_status()
                transcript_id = submit_resp.json()["id"]

                # Step 3: Poll until complete
                deadline = time.perf_counter() + POLL_TIMEOUT
                while True:
                    if time.perf_counter() > deadline:
                        return self._error_result(
                            model, audio_duration_ms, "Polling timeout exceeded"
                        )
                    await asyncio.sleep(POLL_INTERVAL)
                    poll_resp = await client.get(
                        f"{self.base_url}/transcript/{transcript_id}",
                        headers=headers,
                    )
                    poll_resp.raise_for_status()
                    poll_data = poll_resp.json()
                    status = poll_data.get("status")
                    if status == "completed":
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        text = poll_data.get("text", "")
                        word_count = len(text.split()) if text.strip() else 0
                        break
                    elif status == "error":
                        return self._error_result(
                            model, audio_duration_ms,
                            f"AssemblyAI error: {poll_data.get('error', 'unknown')}"
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
