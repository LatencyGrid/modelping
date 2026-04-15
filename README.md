# modelping

⚡ Benchmark LLM, STT, TTS, and full voice pipeline latency across every major AI provider.

One tool. Every provider. The metrics that actually matter.

---

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/modelping)
![GitHub Stars](https://img.shields.io/github/stars/LatencyGrid/modelping)

---

## What It Measures

| Category     | Metrics                                                  |
| ------------ | -------------------------------------------------------- |
| **LLM**      | Time to First Token (TTFT) P50/P95/P99, tokens/sec, cost |
| **STT**      | Transcription latency, time to first partial transcript  |
| **TTS**      | Time to First Audio Byte (TTFB), realtime factor         |
| **Pipeline** | Full STT→LLM→TTS end-to-end latency (the hero metric)    |

All measurements use **real streaming requests** — latency is captured at the byte level.

---

## Installation

```bash
uv pip install modelping
```

Or run directly without installing:

```bash
uvx modelping run gpt-4o --runs 3
```

Or install from source:

```bash
git clone https://github.com/LatencyGrid/modelping
cd modelping
uv sync
```

---

## Quick Start

```bash
# Copy and configure API keys
cp .env.example .env
# Edit .env with your keys

# Run the full voice pipeline benchmark (hero feature)
modelping pipeline --stt groq-whisper-large-v3 --llm llama-3.3-70b-versatile --tts cartesia-sonic-2

# Benchmark LLMs head-to-head
modelping run gpt-4o claude-3-5-sonnet-20241022 gemini-2.5-flash

# Benchmark STT providers
modelping stt

# Benchmark TTS providers
modelping tts
```

---

## Custom / Self-Hosted Servers

Point modelping at **any** OpenAI-compatible inference server — local, self-hosted, or custom endpoints. No API keys needed.

```bash
# Benchmark a local vLLM / Ollama / TGI server
modelping run gpt-4o \
  --base-url http://24.84.229.106:8000 \
  --model-id Meta-Llama-3.1-8B-Instruct \
  --no-verify-ssl

# STT against a local Whisper server
modelping stt groq/whisper-large-v3 \
  --base-url http://localhost:8080 \
  --model-id whisper-large-v3 \
  --no-verify-ssl

# TTS against a local server
modelping tts openai/tts-1 \
  --base-url http://localhost:9000 \
  --model-id kokoro \
  --no-verify-ssl

# Full pipeline against local infrastructure
modelping pipeline \
  --stt groq/whisper-large-v3 \
  --llm gpt-4o-mini \
  --tts cartesia/sonic-2 \
  --base-url http://192.168.1.100:8000 \
  --model-id my-custom-model \
  --no-verify-ssl
```

| Flag              | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `--base-url`      | Override the provider's API endpoint (e.g. `http://host:8000`) |
| `--model-id`      | Override the model name sent to the server                     |
| `--no-verify-ssl` | Skip TLS certificate verification (self-signed certs)          |

When `--base-url` is set, API key checks are bypassed — you don't need any keys configured.

---

## Sample Output

### Pipeline (STT→LLM→TTS)

```
$ modelping pipeline --stt groq-whisper-large-v3 --llm llama-3.3-70b-versatile --tts cartesia-sonic-2

╭──────────────────────────────────────────────────────────────────────────────╮
│  modelping pipeline  •  3 runs                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

 STT                     LLM                      TTS              STT    LLM    TTS   Total
 ─────────────────────────────────────────────────────────────────────────────────────────────
 groq/whisper-large-v3   groq/llama-3.3-70b        cartesia/sonic-2  182ms  44ms   91ms   317ms

✓ Pipeline tested  •  fastest total: 317ms
```

### LLM

```
$ modelping run gpt-4o claude-3-5-sonnet-20241022 gemini-2.5-flash llama-3.3-70b-versatile

╭─────────────────────────────────────────────────────────────────────────────────╮
│  modelping  •  5 runs  •  prompt: 64 tokens                                     │
╰─────────────────────────────────────────────────────────────────────────────────╯

 Model                          Provider     TTFT P50   TTFT P95   Tok/s   Cost/1M
 ─────────────────────────────────────────────────────────────────────────────────
 llama-3.3-70b-versatile        groq           42ms       67ms    312.4     $0.79
 gemini-2.5-flash               google        936ms     1100ms     45.2     $0.60
 claude-3-5-sonnet-20241022     anthropic     198ms      234ms     71.1    $15.00
 gpt-4o                         openai        312ms      489ms     82.3    $10.00

✓ 4 models tested  •  12.3s total
```

Colors: 🟢 green = fastest, 🟡 yellow = mid, 🔴 red = slowest (relative to the tested set).

### STT

```
$ modelping stt --runs 3

 Model                    Provider     Latency P50   Latency P95   Words
 ────────────────────────────────────────────────────────────────────────
 whisper-large-v3-turbo   groq            180ms         210ms        9
 nova-2                   deepgram        240ms         290ms        9
 whisper-1                openai          890ms        1100ms        9

✓ 3 providers tested
```

### TTS

```
$ modelping tts --runs 3

 Model                    Provider      TTFB P50   TTFB P95   Realtime
 ──────────────────────────────────────────────────────────────────────
 sonic-2                  cartesia         89ms      112ms      14.2x
 eleven_flash_v2_5        elevenlabs      210ms      267ms       8.1x
 aura-asteria-en          deepgram        198ms      245ms       9.3x
 tts-1                    openai          312ms      398ms       6.4x

✓ 4 providers tested
```

---

## Full CLI Reference

```bash
# LLM benchmarks
modelping run gpt-4o claude-3-5-sonnet-20241022 gemini-2.5-flash
modelping run --all
modelping run --provider groq
modelping run gpt-4o --runs 10
modelping run gpt-4o --prompt "custom prompt"
modelping run gpt-4o --json
modelping run gpt-4o --csv
modelping run gpt-4o --fail-above-ttft 500

# STT benchmarks
modelping stt
modelping stt groq-whisper-large-v3 deepgram-nova-2
modelping stt --runs 5

# TTS benchmarks
modelping tts
modelping tts cartesia-sonic-2 elevenlabs-flash
modelping tts --runs 5
modelping tts --text "Custom text to synthesize"

# Pipeline benchmark (full STT→LLM→TTS)
modelping pipeline
modelping pipeline --stt groq-whisper-large-v3 --llm gpt-4o-mini --tts cartesia-sonic-2
modelping pipeline --stt all --llm all --tts all
modelping pipeline --runs 3

# List available models
modelping models
modelping models --provider anthropic

# Custom / self-hosted servers (works with any command)
modelping run gpt-4o --base-url http://24.84.229.106:8000 --model-id Meta-Llama-3.1-8B-Instruct --no-verify-ssl
modelping stt groq/whisper-large-v3 --base-url http://localhost:8080 --model-id whisper-large-v3
modelping tts openai/tts-1 --base-url http://localhost:9000 --model-id kokoro
modelping pipeline --base-url http://192.168.1.100:8000 --model-id my-model --no-verify-ssl
```

---

## Supported Providers

### LLM

| Model                                             | Provider  | Input $/1M | Output $/1M |
| ------------------------------------------------- | --------- | ---------- | ----------- |
| gpt-4o                                            | openai    | $2.50      | $10.00      |
| gpt-4o-mini                                       | openai    | $0.15      | $0.60       |
| o3-mini                                           | openai    | $1.10      | $4.40       |
| claude-3-5-sonnet-20241022                        | anthropic | $3.00      | $15.00      |
| claude-3-haiku-20240307                           | anthropic | $0.25      | $1.25       |
| gemini-2.5-flash                                  | google    | $0.15      | $0.60       |
| llama-3.3-70b-versatile                           | groq      | $0.59      | $0.79       |
| mixtral-8x7b-32768                                | groq      | $0.24      | $0.24       |
| accounts/fireworks/models/llama-v3p1-70b-instruct | fireworks | $0.90      | $0.90       |
| meta-llama/Llama-3.3-70B-Instruct-Turbo           | together  | $0.88      | $0.88       |
| mistral-large-latest                              | mistral   | $2.00      | $6.00       |
| mistral-small-latest                              | mistral   | $0.10      | $0.30       |
| command-r-plus                                    | cohere    | $2.50      | $10.00      |
| command-r                                         | cohere    | $0.15      | $0.60       |

### STT

| Model                      | Provider   |
| -------------------------- | ---------- |
| whisper-large-v3           | groq       |
| whisper-large-v3-turbo     | groq       |
| distil-whisper-large-v3-en | groq       |
| whisper-1                  | openai     |
| gpt-4o-transcribe          | openai     |
| nova-2                     | deepgram   |
| nova-3                     | deepgram   |
| best                       | assemblyai |
| nano                       | assemblyai |
| (default)                  | gladia     |

### TTS

| Model                  | Provider   |
| ---------------------- | ---------- |
| eleven_flash_v2_5      | elevenlabs |
| eleven_multilingual_v2 | elevenlabs |
| sonic-2                | cartesia   |
| sonic-english          | cartesia   |
| tts-1                  | openai     |
| tts-1-hd               | openai     |
| (streaming)            | fish-audio |
| PlayDialog             | playht     |
| Play3.0-mini           | playht     |
| aura-asteria-en        | deepgram   |
| aura-luna-en           | deepgram   |
| blizzard               | lmnt       |
| aurora                 | lmnt       |

---

## Configuration

Set API keys in a `.env` file in your working directory:

```env
# LLM providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
FIREWORKS_API_KEY=fw_...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...

# STT providers
DEEPGRAM_API_KEY=...
ASSEMBLYAI_API_KEY=...
GLADIA_API_KEY=...

# TTS providers
ELEVENLABS_API_KEY=...
CARTESIA_API_KEY=...
FISH_AUDIO_API_KEY=...
PLAYHT_API_KEY=...
PLAYHT_USER_ID=...
LMNT_API_KEY=...
```

modelping auto-detects configured providers and skips (with a warning) any without keys set.

---

## CI/CD Example

### GitHub Actions

```yaml
name: AI Latency Check

on:
  schedule:
    - cron: "0 */6 * * *" # every 6 hours
  workflow_dispatch:

jobs:
  latency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Install modelping
        run: uv pip install modelping --system

      - name: Run LLM latency benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          modelping run gpt-4o claude-3-5-sonnet-20241022 --runs 3 --fail-above-ttft 1000

      - name: Check voice pipeline latency
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CARTESIA_API_KEY: ${{ secrets.CARTESIA_API_KEY }}
        run: modelping pipeline --stt groq-whisper-large-v3 --llm gpt-4o-mini --tts cartesia-sonic-2 --fail-above-ttft 500

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

---

## Roadmap

- [x] LLM benchmarking (TTFT, throughput, cost)
- [x] STT benchmarking (transcription latency)
- [x] TTS benchmarking (time to first audio byte)
- [x] Full STT→LLM→TTS pipeline benchmark
- [ ] Community leaderboard — submit anonymous results, see global rankings
- [ ] Web UI — run benchmarks from your browser (bring your own keys)
- [ ] Self-hosted / open source model endpoints
- [ ] Historical tracking and latency alerts

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding a new provider.

PRs welcome for:

- New providers
- New models / updated pricing
- Output improvements
- Bug fixes

```bash
git clone https://github.com/LatencyGrid/modelping
cd modelping
uv sync
```

---

## License

MIT
