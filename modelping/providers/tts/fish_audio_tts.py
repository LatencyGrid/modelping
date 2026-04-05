"""Fish Audio TTS provider."""

from __future__ import annotations

import struct
import time

import httpx

from modelping.models import TTSRunResult
from modelping.providers.tts.base import BaseTTSProvider

# A well-known public reference model ID from Fish Audio
DEFAULT_REFERENCE_ID = "54a5170264694bfc8e9ad98df7bd89c3"


class FishAudioTTSProvider(BaseTTSProvider):
    name = "fish_audio_tts"
    api_key_env = "FISH_AUDIO_API_KEY"
    base_url = "https://api.fish.audio/v1/tts"

    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, len(text), "FISH_AUDIO_API_KEY not set")

        # Fish Audio uses msgpack for the request
        try:
            import msgpack  # type: ignore
        except ImportError:
            # Fall back to JSON if msgpack not available
            return await self._synthesize_json(text, model, api_key)

        payload_data = {
            "text": text,
            "reference_id": DEFAULT_REFERENCE_ID,
            "format": "mp3",
            "mp3_bitrate": 128,
        }
        packed = msgpack.packb(payload_data)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/msgpack",
        }

        return await self._stream_request(text, model, headers, packed, content_type="msgpack")

    async def _synthesize_json(self, text: str, model: str, api_key: str) -> TTSRunResult:
        """Fallback using JSON if msgpack not available."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "reference_id": DEFAULT_REFERENCE_ID,
            "format": "mp3",
            "mp3_bitrate": 128,
        }

        return await self._stream_request(text, model, headers, None, json_payload=payload)

    async def _stream_request(
        self,
        text: str,
        model: str,
        headers: dict,
        content: bytes | None,
        content_type: str = "json",
        json_payload: dict | None = None,
    ) -> TTSRunResult:
        ttfb_ms = 0.0
        total_bytes = 0
        first_chunk = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                if content is not None:
                    request = client.stream("POST", self.base_url, headers=headers, content=content)
                else:
                    request = client.stream("POST", self.base_url, headers=headers, json=json_payload)

                async with request as response:
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
