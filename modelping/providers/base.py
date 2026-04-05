"""BaseProvider abstract class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from modelping.config import get_api_key
from modelping.models import RunResult


class BaseProvider(ABC):
    name: str
    api_key_env: str

    def get_api_key(self) -> str | None:
        return get_api_key(self.name)

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    @abstractmethod
    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        """
        Measure latency for a single inference call using streaming.

        Must use streaming to accurately capture TTFT.
        Record time.perf_counter() at request start, then capture
        the timestamp of the first chunk received.

        Returns a RunResult with ttft_ms, total_ms, tokens_generated, etc.
        """
        ...

    def _make_run_result(
        self,
        model: str,
        ttft_ms: float,
        total_ms: float,
        tokens_generated: int,
        input_tokens: int,
        error: str | None = None,
    ) -> RunResult:
        tokens_per_sec = (
            tokens_generated / (total_ms / 1000.0)
            if total_ms > 0 and tokens_generated > 0
            else 0.0
        )
        return RunResult(
            model=model,
            provider=self.name,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=tokens_generated,
            tokens_per_sec=tokens_per_sec,
            input_tokens=input_tokens,
            timestamp=datetime.now(timezone.utc),
            error=error,
        )

    def _error_result(self, model: str, error: str) -> RunResult:
        return self._make_run_result(
            model=model,
            ttft_ms=0.0,
            total_ms=0.0,
            tokens_generated=0,
            input_tokens=0,
            error=error,
        )
