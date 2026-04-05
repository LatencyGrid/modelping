"""Rich/JSON/CSV renderers for results."""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from modelping.models import AggregatedResult

console = Console()


def _color_by_rank(values: list[float], idx: int, reverse: bool = False) -> str:
    """
    Return a Rich color based on rank relative to other values.
    reverse=True means lower is better (e.g., TTFT).
    """
    if not values or len(values) == 1:
        return "white"
    sorted_vals = sorted(values, reverse=not reverse)
    val = values[idx]
    rank_idx = sorted_vals.index(val)
    n = len(sorted_vals)
    if n <= 2:
        if rank_idx == 0:
            return "green"
        return "red"
    third = n // 3
    if rank_idx < third:
        return "green"
    elif rank_idx < 2 * third:
        return "yellow"
    else:
        return "red"


def render_table(
    results: list[AggregatedResult],
    runs: int,
    prompt_tokens: int,
    elapsed: float,
) -> None:
    """Render a Rich table of results to the terminal."""
    console.print()
    console.print(
        Panel(
            f"[bold]modelping[/bold]  •  {runs} runs  •  prompt: {prompt_tokens} tokens",
            border_style="dim",
        )
    )
    console.print()

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        show_footer=False,
        padding=(0, 1),
    )
    table.add_column("Model", style="", no_wrap=False, min_width=30)
    table.add_column("Provider", style="dim")
    table.add_column("TTFT P50", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Cost/1M out", justify="right")

    # Gather values for coloring
    ttft_p50s = [r.ttft_p50 if r.error_rate < 1.0 else float("inf") for r in results]
    ttft_p95s = [r.ttft_p95 if r.error_rate < 1.0 else float("inf") for r in results]
    throughputs = [r.throughput_p50 if r.error_rate < 1.0 else 0.0 for r in results]

    for i, r in enumerate(results):
        if r.error_rate >= 1.0:
            table.add_row(
                r.model,
                r.provider,
                "[red]error[/red]",
                "[red]error[/red]",
                "[red]error[/red]",
                f"${r.cost_per_1k_output:.2f}",
            )
            continue

        ttft_color = _color_by_rank(
            [v for v in ttft_p50s if v != float("inf")],
            [v for v in ttft_p50s if v != float("inf")].index(r.ttft_p50) if r.ttft_p50 in ttft_p50s else 0,
            reverse=True,  # lower TTFT = better = green
        )
        tps_vals = [v for v in throughputs if v > 0]
        tps_color = _color_by_rank(
            tps_vals,
            tps_vals.index(r.throughput_p50) if r.throughput_p50 in tps_vals else 0,
            reverse=False,  # higher throughput = better = green
        )

        ttft50_color = _color_by_rank(
            [v for v in ttft_p50s if v != float("inf")],
            [j for j, v in enumerate(ttft_p50s) if v != float("inf") and results[j].ttft_p50 == r.ttft_p50][0] if any(results[j].ttft_p50 == r.ttft_p50 for j in range(len(results))) else 0,
            reverse=True,
        )

        table.add_row(
            r.model,
            r.provider,
            f"[{ttft50_color}]{r.ttft_p50:.0f}ms[/{ttft50_color}]",
            f"{r.ttft_p95:.0f}ms",
            f"[{tps_color}]{r.throughput_p50:.1f}[/{tps_color}]",
            f"${r.cost_per_1k_output:.2f}",
        )

    console.print(table)

    success_count = sum(1 for r in results if r.error_rate < 1.0)
    console.print(
        f"[green]✓[/green] {success_count}/{len(results)} models tested  •  {elapsed:.1f}s total"
    )
    console.print()


def render_json(results: list[AggregatedResult]) -> None:
    """Print results as JSON to stdout."""
    data = [r.model_dump() for r in results]
    # Convert datetime-ish objects to str
    print(json.dumps(data, indent=2, default=str))


def render_csv(results: list[AggregatedResult]) -> None:
    """Print results as CSV to stdout."""
    if not results:
        return
    output = io.StringIO()
    fields = list(results[0].model_dump().keys())
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r.model_dump())
    print(output.getvalue(), end="")


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return max(1, len(prompt) // 4)
