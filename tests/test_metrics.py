"""Tests for metrics.py — percentile calculations, no API calls."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from modelping.metrics import percentile, aggregate_results
from modelping.models import RunResult, AggregatedResult


class TestPercentile:
    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 99) == 42.0

    def test_p50_even(self):
        data = [10.0, 20.0, 30.0, 40.0]
        result = percentile(data, 50)
        assert 20.0 <= result <= 30.0  # between 2nd and 3rd

    def test_p50_odd(self):
        data = [10.0, 20.0, 30.0]
        result = percentile(data, 50)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_p0_is_min(self):
        data = [5.0, 10.0, 15.0, 20.0]
        assert percentile(data, 0) == 5.0

    def test_p100_is_max(self):
        data = [5.0, 10.0, 15.0, 20.0]
        assert percentile(data, 100) == 20.0

    def test_p95_above_median(self):
        data = list(range(1, 101))  # 1..100
        p50 = percentile(data, 50)
        p95 = percentile(data, 95)
        assert p95 > p50

    def test_interpolation(self):
        data = [0.0, 100.0]
        assert percentile(data, 50) == pytest.approx(50.0)

    def test_sorted_order_irrelevant(self):
        data = [30.0, 10.0, 20.0]
        assert percentile(data, 50) == pytest.approx(percentile([10.0, 20.0, 30.0], 50))


def _make_run_result(model="gpt-4o", provider="openai", ttft=100.0, total=500.0, tps=80.0, error=None):
    return RunResult(
        model=model,
        provider=provider,
        ttft_ms=ttft,
        total_ms=total,
        tokens_generated=int(tps * total / 1000),
        tokens_per_sec=tps,
        input_tokens=50,
        timestamp=datetime.now(timezone.utc),
        error=error,
    )


class TestAggregateResults:
    def test_basic_aggregation(self):
        runs = [
            _make_run_result(ttft=100.0, tps=80.0),
            _make_run_result(ttft=200.0, tps=60.0),
            _make_run_result(ttft=150.0, tps=70.0),
        ]
        result = aggregate_results(runs, "gpt-4o")
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.runs == 3
        assert result.error_rate == 0.0
        assert 100.0 <= result.ttft_p50 <= 200.0
        assert result.ttft_p95 >= result.ttft_p50
        assert result.throughput_p50 > 0

    def test_all_errors_returns_error_result(self):
        runs = [
            _make_run_result(error="timeout"),
            _make_run_result(error="timeout"),
        ]
        result = aggregate_results(runs, "gpt-4o")
        assert result.error_rate == 1.0
        assert result.ttft_p50 == 0.0
        assert result.ttft_p95 == 0.0

    def test_partial_errors(self):
        runs = [
            _make_run_result(ttft=100.0),
            _make_run_result(error="timeout"),
            _make_run_result(ttft=200.0),
        ]
        result = aggregate_results(runs, "gpt-4o")
        assert result.error_rate == pytest.approx(1 / 3)
        assert result.runs == 3

    def test_cost_from_model_registry(self):
        runs = [_make_run_result()]
        result = aggregate_results(runs, "gpt-4o")
        assert result.cost_per_1k_input == pytest.approx(2.50)
        assert result.cost_per_1k_output == pytest.approx(10.00)

    def test_empty_runs(self):
        result = aggregate_results([], "gpt-4o")
        assert result.error_rate == 1.0
        assert result.runs == 0

    def test_single_run(self):
        runs = [_make_run_result(ttft=123.0, tps=99.0)]
        result = aggregate_results(runs, "gpt-4o")
        assert result.ttft_p50 == pytest.approx(123.0)
        assert result.ttft_p95 == pytest.approx(123.0)
        assert result.ttft_p99 == pytest.approx(123.0)
