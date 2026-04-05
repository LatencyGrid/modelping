"""BaseTTSProvider abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from modelping.models import TTSRunResult


class BaseTTSProvider(ABC):
    name: str
    api_key_env: str

    def get_api_key(self) -> str | None:
        from modelping.config import get_tts_api_key
        return get_tts_api_key(self.name)

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    @abstractmethod
    async def synthesize(self, text: str, model: str) -> TTSRunResult:
        """
        Synthesize speech and measure latency.

        Must measure TTFB (time to first audio byte) via streaming.
        Use time.perf_counter() at request start, capture timestamp
        of first audio chunk received.

        Args:
            text: Text to synthesize
            model: Model identifier

        Returns:
            TTSRunResult with ttfb_ms, total_ms, audio_duration_ms, realtime_factor
        """
        ...

    def _make_result(
        self,
        model: str,
        text_chars: int,
        ttfb_ms: float,
        total_ms: float,
        audio_duration_ms: float,
        error: str | None = None,
    ) -> TTSRunResult:
        realtime_factor = (audio_duration_ms / total_ms) if total_ms > 0 else 0.0
        return TTSRunResult(
            provider=self.name,
            model=model,
            text_chars=text_chars,
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
            realtime_factor=realtime_factor,
            timestamp=datetime.now(timezone.utc),
            error=error,
        )

    def _error_result(self, model: str, text_chars: int, error: str) -> TTSRunResult:
        return self._make_result(
            model=model,
            text_chars=text_chars,
            ttfb_ms=0.0,
            total_ms=0.0,
            audio_duration_ms=0.0,
            error=error,
        )

    @staticmethod
    def _estimate_audio_duration_ms(audio_bytes: bytes, sample_rate: int = 22050, channels: int = 1, bits: int = 16) -> float:
        """Estimate audio duration from raw PCM bytes."""
        bytes_per_sample = bits // 8
        num_samples = len(audio_bytes) / (bytes_per_sample * channels)
        return (num_samples / sample_rate) * 1000.0
