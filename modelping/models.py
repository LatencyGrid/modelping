"""Pydantic schemas for run results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class RunResult(BaseModel):
    model: str
    provider: str
    ttft_ms: float
    total_ms: float
    tokens_generated: int
    tokens_per_sec: float
    input_tokens: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class AggregatedResult(BaseModel):
    model: str
    provider: str
    runs: int
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    throughput_p50: float  # tokens/sec
    total_p50: float
    cost_per_1k_input: float
    cost_per_1k_output: float
    error_rate: float


class STTRunResult(BaseModel):
    provider: str
    model: str
    audio_duration_ms: float       # length of test audio
    transcription_latency_ms: float  # time from request to full transcript
    ttft_ms: Optional[float] = None  # time to first partial transcript (if streaming supported)
    word_count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class TTSRunResult(BaseModel):
    provider: str
    model: str
    text_chars: int                # length of input text
    ttfb_ms: float                 # time to first audio BYTE (most important)
    total_ms: float                # time to complete audio
    audio_duration_ms: float       # duration of generated audio
    realtime_factor: float         # audio_duration / total_ms (>1 = faster than realtime)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class PipelineRunResult(BaseModel):
    stt_provider: str
    stt_model: str
    llm_provider: str
    llm_model: str
    tts_provider: str
    tts_model: str
    stt_latency_ms: float
    llm_ttft_ms: float
    tts_ttfb_ms: float
    total_pipeline_ms: float       # stt + llm + tts end to end
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
