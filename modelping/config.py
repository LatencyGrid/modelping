"""Key management and model registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from cwd or project root
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.home() / ".env")

DEFAULT_PROMPT = "Explain the concept of neural networks in one paragraph."

TTS_DEFAULT_TEXT = "The inference delivery network routes your request to the lowest latency node automatically."

MODELS: dict[str, dict[str, Any]] = {
    # OpenAI
    "gpt-4o": {"provider": "openai", "input_cost": 2.50, "output_cost": 10.00},
    "gpt-4o-mini": {"provider": "openai", "input_cost": 0.15, "output_cost": 0.60},
    "o3-mini": {"provider": "openai", "input_cost": 1.10, "output_cost": 4.40},
    # Anthropic
    "claude-sonnet-4-5-20250929": {"provider": "anthropic", "input_cost": 3.00, "output_cost": 15.00},
    "claude-haiku-4-5-20251001": {"provider": "anthropic", "input_cost": 0.25, "output_cost": 1.25},
    # Google
    "gemini-2.0-flash": {"provider": "google", "input_cost": 0.10, "output_cost": 0.40},
    "gemini-1.5-pro": {"provider": "google", "input_cost": 1.25, "output_cost": 5.00},
    # Groq
    "llama-3.3-70b-versatile": {"provider": "groq", "input_cost": 0.59, "output_cost": 0.79},
    "llama-3.1-8b-instant": {"provider": "groq", "input_cost": 0.05, "output_cost": 0.08},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"provider": "groq", "input_cost": 0.11, "output_cost": 0.34},
    "moonshotai/kimi-k2-instruct": {"provider": "groq", "input_cost": 1.00, "output_cost": 3.00},
    # Fireworks
    "accounts/fireworks/models/llama-v3p1-70b-instruct": {
        "provider": "fireworks",
        "input_cost": 0.90,
        "output_cost": 0.90,
    },
    # Together
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "provider": "together",
        "input_cost": 0.88,
        "output_cost": 0.88,
    },
    # Mistral
    "mistral-large-latest": {"provider": "mistral", "input_cost": 2.00, "output_cost": 6.00},
    "mistral-small-latest": {"provider": "mistral", "input_cost": 0.10, "output_cost": 0.30},
    # Cohere
    "command-r-plus": {"provider": "cohere", "input_cost": 2.50, "output_cost": 10.00},
    "command-r": {"provider": "cohere", "input_cost": 0.15, "output_cost": 0.60},
}

STT_MODELS: dict[str, dict[str, Any]] = {
    # Groq Whisper
    "groq/whisper-large-v3": {"provider": "groq_stt", "model_id": "whisper-large-v3"},
    "groq/whisper-large-v3-turbo": {"provider": "groq_stt", "model_id": "whisper-large-v3-turbo"},
    "openai/gpt-4o-transcribe": {"provider": "openai_stt", "model_id": "gpt-4o-transcribe"},
    # Deepgram
    "deepgram/nova-2": {"provider": "deepgram_stt", "model_id": "nova-2"},
    "deepgram/nova-3": {"provider": "deepgram_stt", "model_id": "nova-3"},
    # AssemblyAI
    "assemblyai/universal-3-pro": {"provider": "assemblyai_stt", "model_id": "universal-3-pro"},
    "assemblyai/universal-2": {"provider": "assemblyai_stt", "model_id": "universal-2"},
    # Gladia
    "gladia/default": {"provider": "gladia_stt", "model_id": "default"},
}

TTS_MODELS: dict[str, dict[str, Any]] = {
    # ElevenLabs
    "elevenlabs/flash": {"provider": "elevenlabs_tts", "model_id": "eleven_flash_v2_5"},
    "elevenlabs/multilingual": {"provider": "elevenlabs_tts", "model_id": "eleven_multilingual_v2"},
    # Cartesia
    "cartesia/sonic-2": {"provider": "cartesia_tts", "model_id": "sonic-2"},
    "cartesia/sonic-english": {"provider": "cartesia_tts", "model_id": "sonic-english"},
    # OpenAI TTS
    "openai-tts/tts-1": {"provider": "openai_tts", "model_id": "tts-1"},
    "openai-tts/tts-1-hd": {"provider": "openai_tts", "model_id": "tts-1-hd"},
    # Fish Audio
    "fish-audio/default": {"provider": "fish_audio_tts", "model_id": "default"},
    # PlayHT
    "playht/play-dialog": {"provider": "playht_tts", "model_id": "PlayDialog"},
    "playht/play3-mini": {"provider": "playht_tts", "model_id": "Play3.0-mini"},
    # Deepgram TTS
    "deepgram-tts/asteria": {"provider": "deepgram_tts", "model_id": "aura-asteria-en"},
    "deepgram-tts/luna": {"provider": "deepgram_tts", "model_id": "aura-luna-en"},
    # LMNT
    "lmnt/blizzard": {"provider": "lmnt_tts", "model_id": "blizzard"},
    "lmnt/aurora": {"provider": "lmnt_tts", "model_id": "aurora"},
}

PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}

STT_PROVIDER_KEY_ENV: dict[str, str] = {
    "groq_stt": "GROQ_API_KEY",
    "openai_stt": "OPENAI_API_KEY",
    "deepgram_stt": "DEEPGRAM_API_KEY",
    "assemblyai_stt": "ASSEMBLYAI_API_KEY",
    "gladia_stt": "GLADIA_API_KEY",
}

TTS_PROVIDER_KEY_ENV: dict[str, str] = {
    "elevenlabs_tts": "ELEVENLABS_API_KEY",
    "cartesia_tts": "CARTESIA_API_KEY",
    "openai_tts": "OPENAI_API_KEY",
    "fish_audio_tts": "FISH_AUDIO_API_KEY",
    "playht_tts": "PLAYHT_API_KEY",
    "deepgram_tts": "DEEPGRAM_API_KEY",
    "lmnt_tts": "LMNT_API_KEY",
}


def get_api_key(provider: str) -> str | None:
    """Return API key for a provider, or None if not set."""
    env_var = PROVIDER_KEY_ENV.get(provider)
    if not env_var:
        return None
    return os.environ.get(env_var) or None


def get_stt_api_key(provider: str) -> str | None:
    """Return API key for an STT provider, or None if not set."""
    env_var = STT_PROVIDER_KEY_ENV.get(provider)
    if not env_var:
        return None
    return os.environ.get(env_var) or None


def get_tts_api_key(provider: str) -> str | None:
    """Return API key for a TTS provider, or None if not set."""
    env_var = TTS_PROVIDER_KEY_ENV.get(provider)
    if not env_var:
        return None
    return os.environ.get(env_var) or None


def get_models_for_provider(provider: str) -> list[str]:
    """Return all model names for a given provider."""
    return [m for m, cfg in MODELS.items() if cfg["provider"] == provider]


def get_configured_providers() -> list[str]:
    """Return providers that have API keys set."""
    return [p for p in PROVIDER_KEY_ENV if get_api_key(p)]


def get_unconfigured_providers() -> list[str]:
    """Return providers without API keys."""
    return [p for p in PROVIDER_KEY_ENV if not get_api_key(p)]


def get_configured_stt_providers() -> list[str]:
    """Return STT providers that have API keys set."""
    return [p for p in STT_PROVIDER_KEY_ENV if get_stt_api_key(p)]


def get_configured_tts_providers() -> list[str]:
    """Return TTS providers that have API keys set."""
    return [p for p in TTS_PROVIDER_KEY_ENV if get_tts_api_key(p)]
