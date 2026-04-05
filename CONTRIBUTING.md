# Contributing to modelping

## Adding a New Provider

modelping makes it straightforward to add support for a new LLM inference provider. Here's how:

### 1. Create the provider file

Add a new file at `modelping/providers/<name>.py`. Implement the `BaseProvider` abstract class:

```python
"""Your Provider Name — short description."""

from __future__ import annotations

import json
import time

import httpx

from modelping.models import RunResult
from modelping.providers.base import BaseProvider


class YourProvider(BaseProvider):
    name = "yourprovider"           # must be unique, lowercase
    api_key_env = "YOUR_API_KEY"    # env var name for the API key
    base_url = "https://api.yourprovider.com/v1"

    async def measure(self, model: str, prompt: str, max_tokens: int = 100) -> RunResult:
        api_key = self.get_api_key()
        if not api_key:
            return self._error_result(model, "YOUR_API_KEY not set")

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        ttft_ms = 0.0
        total_ms = 0.0
        tokens_generated = 0
        input_tokens = 0
        first_token = False

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions",
                                         headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        # Parse SSE / newline-delimited JSON
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)

                        # Capture TTFT on the first content token
                        if not first_token:
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                ttft_ms = (time.perf_counter() - start) * 1000
                                first_token = True

                        # Extract token usage when available
                        if chunk.get("usage"):
                            tokens_generated = chunk["usage"].get("completion_tokens", tokens_generated)
                            input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)

            total_ms = (time.perf_counter() - start) * 1000

        except httpx.HTTPStatusError as e:
            return self._error_result(model, f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return self._error_result(model, str(e))

        if not first_token:
            ttft_ms = total_ms  # fallback

        return self._make_run_result(
            model=model,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=tokens_generated,
            input_tokens=input_tokens,
        )
```

### Key rules:
- **Always use streaming** — this is required to measure TTFT accurately.
- Record `time.perf_counter()` before the request, then capture the first-chunk time.
- Use `self._error_result()` for graceful error handling.
- Use `self._make_run_result()` to construct the final `RunResult`.

### 2. Register the provider

In `modelping/providers/__init__.py`, add your provider to the imports and `PROVIDER_MAP`:

```python
from modelping.providers.yourprovider import YourProvider

PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    ...
    "yourprovider": YourProvider,
}
```

### 3. Add models to the registry

In `modelping/config.py`, add your models to `MODELS`:

```python
MODELS = {
    ...
    "your-model-name": {
        "provider": "yourprovider",
        "input_cost": 1.00,   # cost per 1M input tokens in USD
        "output_cost": 3.00,  # cost per 1M output tokens in USD
    },
}
```

Also add your API key env var to `PROVIDER_KEY_ENV`:

```python
PROVIDER_KEY_ENV = {
    ...
    "yourprovider": "YOUR_API_KEY",
}
```

### 4. Add tests

Create or extend `tests/test_providers.py` with a class for your provider. Use the existing mock patterns — no real API calls in tests.

### 5. Document the env var

Add to `.env.example`:

```
YOUR_API_KEY=
```

---

That's it! Open a PR with all four changes and the tests passing.
