"""Async orchestration — run providers concurrently."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from modelping.config import MODELS, get_api_key
from modelping.metrics import aggregate_results
from modelping.models import AggregatedResult, RunResult
from modelping.providers import get_provider


async def run_model(
    model: str,
    prompt: str,
    runs: int = 5,
    max_tokens: int = 100,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> AggregatedResult:
    """Run N iterations for a single model and return aggregated results."""
    model_cfg = MODELS.get(model)
    if not model_cfg:
        # Return error result for unknown model
        from modelping.models import AggregatedResult
        return AggregatedResult(
            model=model,
            provider="unknown",
            runs=0,
            ttft_p50=0.0,
            ttft_p95=0.0,
            ttft_p99=0.0,
            throughput_p50=0.0,
            total_p50=0.0,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            error_rate=1.0,
        )

    provider_name = model_cfg["provider"]
    provider = get_provider(provider_name)

    results: list[RunResult] = []
    for i in range(runs):
        if progress_callback:
            progress_callback(model, i + 1, runs)
        result = await provider.measure(model, prompt, max_tokens=max_tokens)
        results.append(result)

    return aggregate_results(results, model)


async def run_models(
    models: list[str],
    prompt: str,
    runs: int = 5,
    max_tokens: int = 100,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[AggregatedResult]:
    """Run all models concurrently and return aggregated results."""
    tasks = [
        run_model(model, prompt, runs, max_tokens, progress_callback)
        for model in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    aggregated: list[AggregatedResult] = []
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            model_cfg = MODELS.get(model, {})
            aggregated.append(
                AggregatedResult(
                    model=model,
                    provider=model_cfg.get("provider", "unknown"),
                    runs=0,
                    ttft_p50=0.0,
                    ttft_p95=0.0,
                    ttft_p99=0.0,
                    throughput_p50=0.0,
                    total_p50=0.0,
                    cost_per_1k_input=model_cfg.get("input_cost", 0.0),
                    cost_per_1k_output=model_cfg.get("output_cost", 0.0),
                    error_rate=1.0,
                )
            )
        else:
            aggregated.append(result)  # type: ignore[arg-type]

    return aggregated
