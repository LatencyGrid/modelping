"""TTS provider implementations."""

from modelping.providers.tts.base import BaseTTSProvider
from modelping.providers.tts.elevenlabs_tts import ElevenLabsTTSProvider
from modelping.providers.tts.cartesia_tts import CartesiaTTSProvider
from modelping.providers.tts.openai_tts import OpenAITTSProvider
from modelping.providers.tts.fish_audio_tts import FishAudioTTSProvider
from modelping.providers.tts.playht_tts import PlayHTTTSProvider
from modelping.providers.tts.deepgram_tts import DeepgramTTSProvider
from modelping.providers.tts.lmnt_tts import LMNTTTSProvider

TTS_PROVIDER_MAP: dict[str, type[BaseTTSProvider]] = {
    "elevenlabs_tts": ElevenLabsTTSProvider,
    "cartesia_tts": CartesiaTTSProvider,
    "openai_tts": OpenAITTSProvider,
    "fish_audio_tts": FishAudioTTSProvider,
    "playht_tts": PlayHTTTSProvider,
    "deepgram_tts": DeepgramTTSProvider,
    "lmnt_tts": LMNTTTSProvider,
}


def get_tts_provider(
    name: str,
    *,
    base_url: str | None = None,
    verify_ssl: bool = True,
    model_id: str | None = None,
) -> BaseTTSProvider:
    """Instantiate a TTS provider by name, optionally applying CLI overrides."""
    cls = TTS_PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown TTS provider: {name}")
    provider = cls()
    provider.apply_overrides(base_url=base_url, verify_ssl=verify_ssl, model_id=model_id)
    return provider


__all__ = [
    "BaseTTSProvider",
    "ElevenLabsTTSProvider",
    "CartesiaTTSProvider",
    "OpenAITTSProvider",
    "FishAudioTTSProvider",
    "PlayHTTTSProvider",
    "DeepgramTTSProvider",
    "LMNTTTSProvider",
    "TTS_PROVIDER_MAP",
    "get_tts_provider",
]
