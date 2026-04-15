"""Provider implementations."""

from modelping.providers.base import BaseProvider
from modelping.providers.openai import OpenAIProvider
from modelping.providers.anthropic import AnthropicProvider
from modelping.providers.google import GoogleProvider
from modelping.providers.groq import GroqProvider
from modelping.providers.fireworks import FireworksProvider
from modelping.providers.together import TogetherProvider
from modelping.providers.mistral import MistralProvider
from modelping.providers.cohere import CohereProvider
from modelping.providers.polargrid import PolarGridProvider

PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "fireworks": FireworksProvider,
    "together": TogetherProvider,
    "mistral": MistralProvider,
    "cohere": CohereProvider,
    "polargrid": PolarGridProvider,
}


def get_provider(
    name: str,
    *,
    base_url: str | None = None,
    verify_ssl: bool = True,
    model_id: str | None = None,
) -> BaseProvider:
    """Instantiate a provider by name, optionally applying CLI overrides."""
    cls = PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}")
    provider = cls()
    provider.apply_overrides(base_url=base_url, verify_ssl=verify_ssl, model_id=model_id)
    return provider


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "FireworksProvider",
    "TogetherProvider",
    "MistralProvider",
    "CohereProvider",
    "PROVIDER_MAP",
    "get_provider",
]
