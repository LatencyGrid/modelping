"""TTFT, throughput, and percentile calculations."""

from __future__ import annotations

import statistics
from typing import Sequence

from modelping.models import AggregatedResult, RunResult


def percentile(data: Sequence[float], p: float) -> float:
    """Compute the p-th percentile of data (0-100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    # Linear interpolation
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    fraction = index - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])


def aggregate_results(results: list[RunResult], model: str) -> AggregatedResult:
    """Aggregate multiple RunResult into an AggregatedResult with percentiles."""
    from modelping.config import MODELS

    successful = [r for r in results if r.error is None]
    error_count = len([r for r in results if r.error is not None])
    total_runs = len(results)
    error_rate = error_count / total_runs if total_runs > 0 else 1.0

    provider = results[0].provider if results else ""
    model_cfg = MODELS.get(model, {})

    if not successful:
        return AggregatedResult(
            model=model,
            provider=provider,
            runs=total_runs,
            ttft_p50=0.0,
            ttft_p95=0.0,
            ttft_p99=0.0,
            throughput_p50=0.0,
            total_p50=0.0,
            cost_per_1k_input=model_cfg.get("input_cost", 0.0),
            cost_per_1k_output=model_cfg.get("output_cost", 0.0),
            error_rate=error_rate,
        )

    ttfts = [r.ttft_ms for r in successful]
    throughputs = [r.tokens_per_sec for r in successful]
    totals = [r.total_ms for r in successful]

    return AggregatedResult(
        model=model,
        provider=provider,
        runs=total_runs,
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_p99=percentile(ttfts, 99),
        throughput_p50=percentile(throughputs, 50),
        total_p50=percentile(totals, 50),
        cost_per_1k_input=model_cfg.get("input_cost", 0.0),
        cost_per_1k_output=model_cfg.get("output_cost", 0.0),
        error_rate=error_rate,
    )
