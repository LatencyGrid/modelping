"""STT provider implementations."""

from modelping.providers.stt.base import BaseSTTProvider
from modelping.providers.stt.groq_stt import GroqSTTProvider
from modelping.providers.stt.openai_stt import OpenAISTTProvider
from modelping.providers.stt.deepgram_stt import DeepgramSTTProvider
from modelping.providers.stt.assemblyai_stt import AssemblyAISTTProvider
from modelping.providers.stt.gladia_stt import GladiaSTTProvider

STT_PROVIDER_MAP: dict[str, type[BaseSTTProvider]] = {
    "groq_stt": GroqSTTProvider,
    "openai_stt": OpenAISTTProvider,
    "deepgram_stt": DeepgramSTTProvider,
    "assemblyai_stt": AssemblyAISTTProvider,
    "gladia_stt": GladiaSTTProvider,
}


def get_stt_provider(name: str) -> BaseSTTProvider:
    """Instantiate an STT provider by name."""
    cls = STT_PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown STT provider: {name}")
    return cls()


__all__ = [
    "BaseSTTProvider",
    "GroqSTTProvider",
    "OpenAISTTProvider",
    "DeepgramSTTProvider",
    "AssemblyAISTTProvider",
    "GladiaSTTProvider",
    "STT_PROVIDER_MAP",
    "get_stt_provider",
]
