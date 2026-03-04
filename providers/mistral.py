"""Mistral AI model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class MistralModelProvider(OpenAICompatibleProvider):
    """Mistral AI API provider (api.mistral.ai)."""

    FRIENDLY_NAME = "Mistral AI"

    MODEL_CAPABILITIES = {
        "mistral-large-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="mistral-large-latest",
            friendly_name="Mistral (Large 3)",
            context_window=128_000,
            max_output_tokens=32768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.7),
            description="Mistral Large 3 (128K context) - 675B total/41B active params MoE, top-tier reasoning and coding",
            aliases=["mistral-large", "mistral", "large3"],
            intelligence_score=16,
        ),
        "magistral-medium-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="magistral-medium-latest",
            friendly_name="Mistral (Magistral Medium)",
            context_window=40_000,
            max_output_tokens=16384,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Magistral Medium - Enterprise reasoning model with multi-step logic and transparent chain-of-thought",
            aliases=["magistral-medium", "magistral"],
            intelligence_score=15,
        ),
        "magistral-small-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="magistral-small-latest",
            friendly_name="Mistral (Magistral Small)",
            context_window=40_000,
            max_output_tokens=16384,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Magistral Small - Fast reasoning model for lightweight multi-step logic tasks",
            aliases=["magistral-small"],
            intelligence_score=13,
        ),
        "mistral-small-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="mistral-small-latest",
            friendly_name="Mistral (Small 3.1)",
            context_window=128_000,
            max_output_tokens=32768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.7),
            description="Mistral Small 3.1 (128K context) - Efficient model for general tasks with vision support",
            aliases=["mistral-small"],
            intelligence_score=12,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Mistral provider with API key."""
        kwargs.setdefault("base_url", "https://api.mistral.ai/v1")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.MISTRAL

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in (
            "magistral-medium-latest",
            "magistral-small-latest",
        )
