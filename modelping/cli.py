"""Typer CLI entrypoint for modelping."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from modelping.config import (
    DEFAULT_PROMPT,
    MODELS,
    STT_MODELS,
    TTS_MODELS,
    TTS_DEFAULT_TEXT,
    PROVIDER_KEY_ENV,
    STT_PROVIDER_KEY_ENV,
    TTS_PROVIDER_KEY_ENV,
    get_api_key,
    get_stt_api_key,
    get_tts_api_key,
    get_configured_providers,
    get_models_for_provider,
    get_unconfigured_providers,
    get_configured_stt_providers,
    get_configured_tts_providers,
)
from modelping.output import (
    estimate_prompt_tokens,
    render_csv,
    render_json,
    render_table,
)
from modelping.runner import run_models
from modelping.utils.audio import get_test_audio_path

app = typer.Typer(
    name="modelping",
    help="⚡ Latency benchmarks for LLM inference providers.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


@app.command("run")
def run_cmd(
    model_names: Optional[List[str]] = typer.Argument(
        None, help="Model(s) to benchmark. Omit to use --all or --provider."
    ),
    all_models: bool = typer.Option(False, "--all", help="Test all configured models."),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Test all models from a specific provider."
    ),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per model."),
    prompt: str = typer.Option(
        DEFAULT_PROMPT, "--prompt", "-p", help="Custom prompt to use."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    output_csv: bool = typer.Option(False, "--csv", help="Output results as CSV."),
    fail_above_ttft: Optional[float] = typer.Option(
        None,
        "--fail-above-ttft",
        help="Exit 1 if any model's P95 TTFT exceeds this value (ms). Useful for CI.",
    ),
    max_tokens: int = typer.Option(100, "--max-tokens", help="Max tokens to generate."),
) -> None:
    """Run latency benchmarks for one or more models."""
    # Determine which models to test
    target_models: list[str] = []

    if model_names:
        for m in model_names:
            if m not in MODELS:
                err_console.print(f"[red]Unknown model:[/red] {m}")
                err_console.print(f"Run [bold]modelping models[/bold] to see available models.")
                raise typer.Exit(1)
        target_models = list(model_names)
    elif all_models:
        target_models = list(MODELS.keys())
    elif provider:
        if provider not in PROVIDER_KEY_ENV:
            err_console.print(f"[red]Unknown provider:[/red] {provider}")
            raise typer.Exit(1)
        target_models = get_models_for_provider(provider)
        if not target_models:
            err_console.print(f"[yellow]No models found for provider:[/yellow] {provider}")
            raise typer.Exit(1)
    else:
        err_console.print(
            "[yellow]No models specified.[/yellow] Use model names, --all, or --provider."
        )
        raise typer.Exit(1)

    # Filter to configured providers; warn about skipped ones
    configured = get_configured_providers()
    skipped = [m for m in target_models if MODELS.get(m, {}).get("provider") not in configured]
    target_models = [m for m in target_models if MODELS.get(m, {}).get("provider") in configured]

    if skipped and not output_json and not output_csv:
        err_console.print(
            f"[yellow]⚠ Skipping {len(skipped)} model(s) — API keys not configured:[/yellow]"
        )
        for m in skipped:
            p = MODELS[m]["provider"]
            err_console.print(f"  [dim]{m}[/dim] (needs {PROVIDER_KEY_ENV[p]})")

    if not target_models:
        err_console.print("[red]No configured models to test. Set API keys in .env or environment.[/red]")
        raise typer.Exit(1)

    prompt_tokens = estimate_prompt_tokens(prompt)
    start = time.perf_counter()

    if not output_json and not output_csv:
        err_console.print(
            f"[dim]Running {runs} × {len(target_models)} model(s) concurrently...[/dim]"
        )

    results = asyncio.run(run_models(target_models, prompt, runs=runs, max_tokens=max_tokens))
    elapsed = time.perf_counter() - start

    if output_json:
        render_json(results)
    elif output_csv:
        render_csv(results)
    else:
        render_table(results, runs=runs, prompt_tokens=prompt_tokens, elapsed=elapsed)

    # CI fail-above-ttft check
    if fail_above_ttft is not None:
        failures = [r for r in results if r.ttft_p95 > fail_above_ttft and r.error_rate < 1.0]
        if failures:
            err_console.print(
                f"[red]✗ FAIL:[/red] {len(failures)} model(s) exceeded P95 TTFT of {fail_above_ttft:.0f}ms:"
            )
            for f in failures:
                err_console.print(f"  {f.model}: P95={f.ttft_p95:.0f}ms")
            raise typer.Exit(1)


@app.command("models")
def models_cmd(
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Filter by provider name."
    ),
    show_costs: bool = typer.Option(False, "--costs", help="Show cost per 1M tokens."),
) -> None:
    """List all supported models."""
    table = Table(title="Supported Models", show_lines=False)
    table.add_column("Model", style="bold", no_wrap=False)
    table.add_column("Provider")
    table.add_column("Status")
    if show_costs:
        table.add_column("Input $/1M", justify="right")
        table.add_column("Output $/1M", justify="right")

    configured_providers = set(get_configured_providers())

    for model, cfg in MODELS.items():
        if provider and cfg["provider"] != provider:
            continue
        is_configured = cfg["provider"] in configured_providers
        status = "[green]✓ ready[/green]" if is_configured else "[dim]no key[/dim]"
        if show_costs:
            table.add_row(
                model,
                cfg["provider"],
                status,
                f"${cfg['input_cost']:.2f}",
                f"${cfg['output_cost']:.2f}",
            )
        else:
            table.add_row(model, cfg["provider"], status)

    console.print(table)

    unconfigured = get_unconfigured_providers()
    if unconfigured:
        console.print(
            f"\n[dim]Providers without API keys: {', '.join(unconfigured)}[/dim]"
        )
        console.print(
            "[dim]Set keys in .env file or environment variables. See .env.example.[/dim]"
        )


@app.command("stt")
def stt_cmd(
    model_keys: Optional[List[str]] = typer.Argument(
        None,
        help="STT model key(s) to benchmark (e.g. groq/whisper-large-v3). Omit for all configured.",
    ),
    runs: int = typer.Option(1, "--runs", "-r", help="Number of runs per model."),
    audio: Optional[str] = typer.Option(
        None, "--audio", help="Path to audio file (WAV). Defaults to built-in test audio."
    ),
) -> None:
    """Benchmark Speech-to-Text providers."""
    from modelping.providers.stt import get_stt_provider

    audio_path = audio or get_test_audio_path()
    configured_providers = set(get_configured_stt_providers())

    if model_keys:
        targets = list(model_keys)
        for k in targets:
            if k not in STT_MODELS:
                err_console.print(f"[red]Unknown STT model:[/red] {k}")
                err_console.print("Available: " + ", ".join(STT_MODELS.keys()))
                raise typer.Exit(1)
    else:
        targets = [k for k, cfg in STT_MODELS.items() if cfg["provider"] in configured_providers]
        if not targets:
            err_console.print("[red]No STT providers configured. Set API keys in .env[/red]")
            raise typer.Exit(1)

    # Filter unconfigured
    skipped = [k for k in targets if STT_MODELS[k]["provider"] not in configured_providers]
    targets = [k for k in targets if STT_MODELS[k]["provider"] in configured_providers]

    if skipped:
        err_console.print(f"[yellow]⚠ Skipping {len(skipped)} STT model(s) — API keys not set:[/yellow]")
        for k in skipped:
            p = STT_MODELS[k]["provider"]
            err_console.print(f"  [dim]{k}[/dim] (needs {STT_PROVIDER_KEY_ENV.get(p, '?')})")

    if not targets:
        err_console.print("[red]No configured STT models to test.[/red]")
        raise typer.Exit(1)

    err_console.print(f"[dim]Running STT benchmark: {len(targets)} model(s), {runs} run(s) each...[/dim]")
    err_console.print(f"[dim]Audio: {audio_path}[/dim]")

    async def _run() -> list:
        from modelping.models import STTRunResult
        all_results: list[STTRunResult] = []
        for model_key in targets:
            cfg = STT_MODELS[model_key]
            provider = get_stt_provider(cfg["provider"])
            for _ in range(runs):
                r = await provider.transcribe(audio_path, cfg["model_id"])
                r = r.model_copy(update={"model": model_key})  # use display key
                all_results.append(r)
        return all_results

    results = asyncio.run(_run())

    # Render table
    table = Table(title="STT Benchmark Results", show_lines=False)
    table.add_column("Provider/Model", style="bold")
    table.add_column("Latency", justify="right")
    table.add_column("Audio Dur", justify="right")
    table.add_column("RTF", justify="right", help="Real-time factor (latency/audio)")
    table.add_column("Words", justify="right")
    table.add_column("Status")

    for r in results:
        if r.error:
            table.add_row(
                r.model, "—", "—", "—", "—",
                f"[red]✗ {r.error[:40]}[/red]"
            )
        else:
            rtf = r.transcription_latency_ms / r.audio_duration_ms if r.audio_duration_ms > 0 else 0
            table.add_row(
                r.model,
                f"{r.transcription_latency_ms:.0f}ms",
                f"{r.audio_duration_ms:.0f}ms",
                f"{rtf:.2f}x",
                str(r.word_count),
                "[green]✓[/green]",
            )

    console.print(table)
    successes = [r for r in results if not r.error]
    if successes:
        fastest = min(successes, key=lambda r: r.transcription_latency_ms)
        console.print(
            f"\n[dim]✓ {len(successes)}/{len(results)} succeeded  •  fastest: {fastest.model} @ {fastest.transcription_latency_ms:.0f}ms[/dim]"
        )


@app.command("tts")
def tts_cmd(
    model_keys: Optional[List[str]] = typer.Argument(
        None,
        help="TTS model key(s) to benchmark (e.g. elevenlabs/flash). Omit for all configured.",
    ),
    runs: int = typer.Option(1, "--runs", "-r", help="Number of runs per model."),
    text: str = typer.Option(
        TTS_DEFAULT_TEXT, "--text", "-t", help="Text to synthesize."
    ),
) -> None:
    """Benchmark Text-to-Speech providers."""
    from modelping.providers.tts import get_tts_provider

    configured_providers = set(get_configured_tts_providers())

    if model_keys:
        targets = list(model_keys)
        for k in targets:
            if k not in TTS_MODELS:
                err_console.print(f"[red]Unknown TTS model:[/red] {k}")
                err_console.print("Available: " + ", ".join(TTS_MODELS.keys()))
                raise typer.Exit(1)
    else:
        targets = [k for k, cfg in TTS_MODELS.items() if cfg["provider"] in configured_providers]
        if not targets:
            err_console.print("[red]No TTS providers configured. Set API keys in .env[/red]")
            raise typer.Exit(1)

    # Filter unconfigured
    skipped = [k for k in targets if TTS_MODELS[k]["provider"] not in configured_providers]
    targets = [k for k in targets if TTS_MODELS[k]["provider"] in configured_providers]

    if skipped:
        err_console.print(f"[yellow]⚠ Skipping {len(skipped)} TTS model(s) — API keys not set:[/yellow]")
        for k in skipped:
            p = TTS_MODELS[k]["provider"]
            err_console.print(f"  [dim]{k}[/dim] (needs {TTS_PROVIDER_KEY_ENV.get(p, '?')})")

    if not targets:
        err_console.print("[red]No configured TTS models to test.[/red]")
        raise typer.Exit(1)

    err_console.print(f"[dim]Running TTS benchmark: {len(targets)} model(s), {runs} run(s) each...[/dim]")
    err_console.print(f"[dim]Text ({len(text)} chars): {text[:60]}...[/dim]")

    async def _run() -> list:
        from modelping.models import TTSRunResult
        all_results: list[TTSRunResult] = []
        for model_key in targets:
            cfg = TTS_MODELS[model_key]
            provider = get_tts_provider(cfg["provider"])
            for _ in range(runs):
                r = await provider.synthesize(text, cfg["model_id"])
                r = r.model_copy(update={"model": model_key})  # use display key
                all_results.append(r)
        return all_results

    results = asyncio.run(_run())

    # Render table
    table = Table(title="TTS Benchmark Results", show_lines=False)
    table.add_column("Provider/Model", style="bold")
    table.add_column("TTFB", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Audio Dur", justify="right")
    table.add_column("RTF", justify="right", help="audio_duration / total_ms (>1 = faster than real-time)")
    table.add_column("Status")

    for r in results:
        if r.error:
            table.add_row(
                r.model, "—", "—", "—", "—",
                f"[red]✗ {r.error[:40]}[/red]"
            )
        else:
            rtf_color = "green" if r.realtime_factor >= 1.0 else "yellow"
            table.add_row(
                r.model,
                f"[bold]{r.ttfb_ms:.0f}ms[/bold]",
                f"{r.total_ms:.0f}ms",
                f"{r.audio_duration_ms:.0f}ms",
                f"[{rtf_color}]{r.realtime_factor:.1f}x[/{rtf_color}]",
                "[green]✓[/green]",
            )

    console.print(table)
    successes = [r for r in results if not r.error]
    if successes:
        fastest = min(successes, key=lambda r: r.ttfb_ms)
        console.print(
            f"\n[dim]✓ {len(successes)}/{len(results)} succeeded  •  fastest TTFB: {fastest.model} @ {fastest.ttfb_ms:.0f}ms[/dim]"
        )


@app.command("pipeline")
def pipeline_cmd(
    stt: Optional[str] = typer.Option(
        None, "--stt", help="STT model key, or 'all' for matrix. Default: groq/whisper-large-v3"
    ),
    llm: Optional[str] = typer.Option(
        None, "--llm", help="LLM model key, or 'all' for matrix. Default: gpt-4o-mini"
    ),
    tts: Optional[str] = typer.Option(
        None, "--tts", help="TTS model key, or 'all' for matrix. Default: cartesia/sonic-2"
    ),
    runs: int = typer.Option(1, "--runs", "-r", help="Number of runs per pipeline combination."),
) -> None:
    """
    Benchmark the full STT → LLM → TTS voice pipeline.

    This is the hero feature — measures end-to-end latency of a voice assistant pipeline.
    """
    from modelping.pipeline_runner import run_pipeline_matrix
    from modelping.config import get_configured_stt_providers, get_configured_tts_providers

    # Resolve STT models
    configured_stt = set(get_configured_stt_providers())
    configured_llm = set(get_configured_providers())
    configured_tts = set(get_configured_tts_providers())

    if stt == "all":
        stt_models = [k for k, cfg in STT_MODELS.items() if cfg["provider"] in configured_stt]
    elif stt:
        stt_models = [stt]
    else:
        # Default: first configured STT
        stt_models = [k for k, cfg in STT_MODELS.items() if cfg["provider"] in configured_stt][:1]
        if not stt_models:
            stt_models = ["groq/whisper-large-v3"]

    # Resolve LLM models
    if llm == "all":
        llm_models = [k for k, cfg in MODELS.items() if cfg["provider"] in configured_llm]
    elif llm:
        llm_models = [llm]
    else:
        # Default: first configured LLM, prefer gpt-4o-mini
        if "gpt-4o-mini" in MODELS and get_api_key("openai"):
            llm_models = ["gpt-4o-mini"]
        else:
            llm_models = [k for k, cfg in MODELS.items() if cfg["provider"] in configured_llm][:1]
            if not llm_models:
                llm_models = ["gpt-4o-mini"]

    # Resolve TTS models
    if tts == "all":
        tts_models = [k for k, cfg in TTS_MODELS.items() if cfg["provider"] in configured_tts]
    elif tts:
        tts_models = [tts]
    else:
        # Default: first configured TTS, prefer cartesia
        if "cartesia/sonic-2" in TTS_MODELS and get_tts_api_key("cartesia_tts"):
            tts_models = ["cartesia/sonic-2"]
        else:
            tts_models = [k for k, cfg in TTS_MODELS.items() if cfg["provider"] in configured_tts][:1]
            if not tts_models:
                tts_models = ["cartesia/sonic-2"]

    total_combos = len(stt_models) * len(llm_models) * len(tts_models) * runs
    console.print(
        Panel(
            f"[bold]modelping pipeline[/bold]  •  {total_combos} run(s)",
            style="bold blue",
        )
    )
    err_console.print(
        f"[dim]STT: {stt_models}  |  LLM: {llm_models}  |  TTS: {tts_models}[/dim]"
    )

    results = asyncio.run(run_pipeline_matrix(stt_models, llm_models, tts_models, runs=runs))

    # Render table
    table = Table(show_lines=False)
    table.add_column("STT", style="bold")
    table.add_column("LLM", style="bold")
    table.add_column("TTS", style="bold")
    table.add_column("STT", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("TTS", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Status")

    successes = []
    for r in results:
        if r.error:
            table.add_row(
                r.stt_model, r.llm_model, r.tts_model,
                "—", "—", "—", "—",
                f"[red]✗ {r.error[:30]}[/red]",
            )
        else:
            successes.append(r)
            table.add_row(
                r.stt_model, r.llm_model, r.tts_model,
                f"{r.stt_latency_ms:.0f}ms",
                f"{r.llm_ttft_ms:.0f}ms",
                f"{r.tts_ttfb_ms:.0f}ms",
                f"[bold]{r.total_pipeline_ms:.0f}ms[/bold]",
                "[green]✓[/green]",
            )

    console.print(table)

    if successes:
        fastest = min(successes, key=lambda r: r.total_pipeline_ms)
        console.print(
            f"\n[green]✓ {len(successes)} pipeline(s) tested  •  fastest total: {fastest.total_pipeline_ms:.0f}ms[/green]"
        )
        console.print(
            f"  [dim]({fastest.stt_model} + {fastest.llm_model} + {fastest.tts_model})[/dim]"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
