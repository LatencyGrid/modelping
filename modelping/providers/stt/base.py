"""BaseSTTProvider abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from modelping.models import STTRunResult


class BaseSTTProvider(ABC):
    name: str
    api_key_env: str

    def get_api_key(self) -> str | None:
        from modelping.config import get_stt_api_key
        return get_stt_api_key(self.name)

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    @abstractmethod
    async def transcribe(self, audio_path: str, model: str) -> STTRunResult:
        """
        Transcribe audio file and measure latency.

        Args:
            audio_path: Path to WAV audio file
            model: Model identifier to use

        Returns:
            STTRunResult with transcription_latency_ms, word_count, etc.
        """
        ...

    def _make_result(
        self,
        model: str,
        audio_duration_ms: float,
        transcription_latency_ms: float,
        word_count: int,
        ttft_ms: float | None = None,
        error: str | None = None,
    ) -> STTRunResult:
        return STTRunResult(
            provider=self.name,
            model=model,
            audio_duration_ms=audio_duration_ms,
            transcription_latency_ms=transcription_latency_ms,
            ttft_ms=ttft_ms,
            word_count=word_count,
            timestamp=datetime.now(timezone.utc),
            error=error,
        )

    def _error_result(self, model: str, audio_duration_ms: float, error: str) -> STTRunResult:
        return self._make_result(
            model=model,
            audio_duration_ms=audio_duration_ms,
            transcription_latency_ms=0.0,
            word_count=0,
            error=error,
        )
