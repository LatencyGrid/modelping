"""Pipeline orchestration — STT → LLM → TTS sequential benchmark."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from modelping.config import (
    MODELS,
    STT_MODELS,
    TTS_MODELS,
    TTS_DEFAULT_TEXT,
    get_stt_api_key,
    get_tts_api_key,
    get_api_key,
)
from modelping.models import PipelineRunResult
from modelping.providers.stt import get_stt_provider
from modelping.providers.tts import get_tts_provider
from modelping.providers import get_provider
from modelping.utils.audio import get_test_audio_path


async def run_pipeline(
    stt_model_key: str,
    llm_model_key: str,
    tts_model_key: str,
    *,
    base_url: str | None = None,
    verify_ssl: bool = True,
    model_id: str | None = None,
) -> PipelineRunResult:
    """
    Run a single STT → LLM → TTS pipeline benchmark.

    Pipeline runs sequentially (matches real-world voice assistant flow):
    1. STT: transcribe test audio → get text
    2. LLM: send transcript as prompt → get first token (TTFT)
    3. TTS: synthesize LLM response → get first audio byte (TTFB)

    Returns PipelineRunResult with per-stage latencies.
    """
    stt_cfg = STT_MODELS.get(stt_model_key)
    llm_cfg = MODELS.get(llm_model_key)
    tts_cfg = TTS_MODELS.get(tts_model_key)

    if not stt_cfg:
        return _error_pipeline(stt_model_key, llm_model_key, tts_model_key, f"Unknown STT model: {stt_model_key}")
    if not llm_cfg:
        return _error_pipeline(stt_model_key, llm_model_key, tts_model_key, f"Unknown LLM model: {llm_model_key}")
    if not tts_cfg:
        return _error_pipeline(stt_model_key, llm_model_key, tts_model_key, f"Unknown TTS model: {tts_model_key}")

    stt_provider_name = stt_cfg["provider"]
    stt_model_id = stt_cfg["model_id"]
    llm_provider_name = llm_cfg["provider"]
    tts_provider_name = tts_cfg["provider"]
    tts_model_id = tts_cfg["model_id"]

    # Check API keys (--base-url bypasses key check)
    if not base_url:
        if not get_stt_api_key(stt_provider_name):
            return _error_pipeline(stt_model_key, llm_model_key, tts_model_key,
                                   f"No API key for STT provider: {stt_provider_name}")
        if not get_api_key(llm_provider_name):
            return _error_pipeline(stt_model_key, llm_model_key, tts_model_key,
                                   f"No API key for LLM provider: {llm_provider_name}")
        if not get_tts_api_key(tts_provider_name):
            return _error_pipeline(stt_model_key, llm_model_key, tts_model_key,
                                   f"No API key for TTS provider: {tts_provider_name}")

    override_kwargs = dict(base_url=base_url, verify_ssl=verify_ssl, model_id=model_id)
    audio_path = get_test_audio_path()
    pipeline_start = time.perf_counter()

    # ── Stage 1: STT ─────────────────────────────────────────────────────────
    stt_provider = get_stt_provider(stt_provider_name, **override_kwargs)
    stt_result = await stt_provider.transcribe(audio_path, stt_model_id)

    if stt_result.error:
        return _error_pipeline(
            stt_model_key, llm_model_key, tts_model_key,
            f"STT error: {stt_result.error}"
        )

    stt_latency_ms = stt_result.transcription_latency_ms
    transcript = "The quick brown fox jumps over the lazy dog."  # Use fixed text for LLM

    # ── Stage 2: LLM ─────────────────────────────────────────────────────────
    llm_provider = get_provider(llm_provider_name, **override_kwargs)
    llm_result = await llm_provider.measure(
        llm_model_key,
        f"Given this transcription: '{transcript}' — respond in one sentence.",
        max_tokens=50,
    )

    if llm_result.error:
        return _error_pipeline(
            stt_model_key, llm_model_key, tts_model_key,
            f"LLM error: {llm_result.error}"
        )

    llm_ttft_ms = llm_result.ttft_ms

    # ── Stage 3: TTS ─────────────────────────────────────────────────────────
    tts_provider = get_tts_provider(tts_provider_name, **override_kwargs)
    tts_result = await tts_provider.synthesize(TTS_DEFAULT_TEXT, tts_model_id)

    if tts_result.error:
        return _error_pipeline(
            stt_model_key, llm_model_key, tts_model_key,
            f"TTS error: {tts_result.error}"
        )

    tts_ttfb_ms = tts_result.ttfb_ms
    total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

    return PipelineRunResult(
        stt_provider=stt_provider_name,
        stt_model=stt_model_key,
        llm_provider=llm_provider_name,
        llm_model=llm_model_key,
        tts_provider=tts_provider_name,
        tts_model=tts_model_key,
        stt_latency_ms=stt_latency_ms,
        llm_ttft_ms=llm_ttft_ms,
        tts_ttfb_ms=tts_ttfb_ms,
        total_pipeline_ms=total_pipeline_ms,
        timestamp=datetime.now(timezone.utc),
    )


async def run_pipeline_matrix(
    stt_models: list[str],
    llm_models: list[str],
    tts_models: list[str],
    runs: int = 1,
    *,
    base_url: str | None = None,
    verify_ssl: bool = True,
    model_id: str | None = None,
) -> list[PipelineRunResult]:
    """Run all combinations of STT × LLM × TTS pipelines."""
    results = []
    for stt in stt_models:
        for llm in llm_models:
            for tts in tts_models:
                for _ in range(runs):
                    result = await run_pipeline(
                        stt, llm, tts,
                        base_url=base_url, verify_ssl=verify_ssl, model_id=model_id,
                    )
                    results.append(result)
    return results


def _error_pipeline(
    stt_model: str,
    llm_model: str,
    tts_model: str,
    error: str,
) -> PipelineRunResult:
    stt_cfg = STT_MODELS.get(stt_model, {})
    llm_cfg = MODELS.get(llm_model, {})
    tts_cfg = TTS_MODELS.get(tts_model, {})
    return PipelineRunResult(
        stt_provider=stt_cfg.get("provider", "unknown"),
        stt_model=stt_model,
        llm_provider=llm_cfg.get("provider", "unknown"),
        llm_model=llm_model,
        tts_provider=tts_cfg.get("provider", "unknown"),
        tts_model=tts_model,
        stt_latency_ms=0.0,
        llm_ttft_ms=0.0,
        tts_ttfb_ms=0.0,
        total_pipeline_ms=0.0,
        timestamp=datetime.now(timezone.utc),
        error=error,
    )
