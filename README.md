# modelping

**⚡ Latency benchmarks for LLM inference providers**

Think `ping`, but for AI models. Measure TTFT (Time to First Token), throughput, and cost across every major LLM provider — from a single command.

---

## Installation

```bash
pip install modelping
```

Or install from source:

```bash
git clone https://github.com/yourorg/modelping
cd modelping
pip install -e .
```

## Quick Start

```bash
# Copy and configure API keys
cp .env.example .env
# Edit .env with your keys

# Benchmark three models head-to-head
modelping run gpt-4o claude-3-5-sonnet-20241022 gemini-2.0-flash
```

## Usage

### Benchmark specific models

```bash
modelping run gpt-4o claude-3-5-sonnet-20241022 gemini-2.0-flash
```

### Benchmark all configured models

```bash
modelping run --all
```

### Benchmark all models from one provider

```bash
modelping run --provider openai
modelping run --provider groq
```

### More runs for better percentile accuracy

```bash
modelping run gpt-4o --runs 10
```

### Custom prompt

```bash
modelping run gpt-4o --prompt "Write a haiku about distributed systems."
```

### Machine-readable output

```bash
modelping run gpt-4o --json        # JSON to stdout
modelping run gpt-4o --csv         # CSV to stdout
```

### CI/CD: fail if latency is too high

```bash
modelping run gpt-4o --fail-above-ttft 500   # exit 1 if P95 TTFT > 500ms
```

### List available models

```bash
modelping models                        # all models
modelping models --provider anthropic   # filtered by provider
modelping models --costs                # show pricing
```

## Sample Output

```
╭─────────────────────────────────────────────────────────────────────────────────╮
│  modelping  •  5 runs  •  prompt: 64 tokens                                     │
╰─────────────────────────────────────────────────────────────────────────────────╯

 Model                          Provider     TTFT P50   TTFT P95   Tok/s   Cost/1M
 ─────────────────────────────────────────────────────────────────────────────────
 gpt-4o                         openai        312ms      489ms     82.3    $10.00
 claude-3-5-sonnet-20241022     anthropic     198ms      234ms     71.1    $15.00
 gemini-2.0-flash               google         89ms      134ms    143.2     $0.40
 llama-3.3-70b-versatile        groq           42ms       67ms    312.4     $0.79

✓ 4/4 models tested  •  12.3s total
```

Colors: 🟢 green = fastest, 🟡 yellow = mid, 🔴 red = slowest (relative to the tested set).

## Supported Models

| Model | Provider | Input $/1M | Output $/1M |
|-------|----------|-----------|------------|
| gpt-4o | openai | $2.50 | $10.00 |
| gpt-4o-mini | openai | $0.15 | $0.60 |
| o3-mini | openai | $1.10 | $4.40 |
| claude-3-5-sonnet-20241022 | anthropic | $3.00 | $15.00 |
| claude-3-haiku-20240307 | anthropic | $0.25 | $1.25 |
| gemini-2.0-flash | google | $0.10 | $0.40 |
| gemini-1.5-pro | google | $1.25 | $5.00 |
| llama-3.3-70b-versatile | groq | $0.59 | $0.79 |
| mixtral-8x7b-32768 | groq | $0.24 | $0.24 |
| accounts/fireworks/models/llama-v3p1-70b-instruct | fireworks | $0.90 | $0.90 |
| meta-llama/Llama-3.3-70B-Instruct-Turbo | together | $0.88 | $0.88 |
| mistral-large-latest | mistral | $2.00 | $6.00 |
| mistral-small-latest | mistral | $0.10 | $0.30 |
| command-r-plus | cohere | $2.50 | $10.00 |
| command-r | cohere | $0.15 | $0.60 |

## Configuration

Set API keys in a `.env` file in your working directory:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
FIREWORKS_API_KEY=fw_...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...
```

modelping auto-detects configured providers and skips (with a warning) any without keys set.

## CI/CD Example

### GitHub Actions

```yaml
name: LLM Latency Check

on:
  schedule:
    - cron: '0 */6 * * *'   # every 6 hours
  workflow_dispatch:

jobs:
  latency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install modelping
        run: pip install modelping

      - name: Run latency benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          modelping run gpt-4o claude-3-5-sonnet-20241022 --runs 3 --fail-above-ttft 1000

      - name: Export JSON results
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          modelping run --provider openai --json > results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: latency-results
          path: results.json
```

## What It Measures

| Metric | Description |
|--------|-------------|
| **TTFT P50** | Median time to first token (ms) |
| **TTFT P95** | 95th percentile TTFT — what most users experience |
| **Tok/s** | Median output throughput (tokens per second) |
| **Cost/1M** | Output cost per 1 million tokens |

All measurements use **real streaming requests** — TTFT is captured at the byte level when the first content token arrives.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding a new provider.

PRs welcome for:
- New providers
- New models / updated pricing
- Output improvements
- Bug fixes

## License

MIT
