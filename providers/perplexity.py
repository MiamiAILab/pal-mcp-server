"""Perplexity AI (Sonar) model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class PerplexityModelProvider(OpenAICompatibleProvider):
    """Perplexity AI API provider (api.perplexity.ai)."""

    FRIENDLY_NAME = "Perplexity AI"

    MODEL_CAPABILITIES = {
        "sonar-pro": ModelCapabilities(
            provider=ProviderType.PERPLEXITY,
            model_name="sonar-pro",
            friendly_name="Perplexity (Sonar Pro)",
            context_window=200_000,
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Sonar Pro - Enhanced search-augmented generation with advanced filtering and citations",
            aliases=["sonar-pro", "pplx-pro"],
            intelligence_score=14,
        ),
        "sonar": ModelCapabilities(
            provider=ProviderType.PERPLEXITY,
            model_name="sonar",
            friendly_name="Perplexity (Sonar)",
            context_window=128_000,
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Sonar - Fast search-augmented generation with web citations",
            aliases=["pplx", "perplexity"],
            intelligence_score=12,
        ),
        "sonar-reasoning-pro": ModelCapabilities(
            provider=ProviderType.PERPLEXITY,
            model_name="sonar-reasoning-pro",
            friendly_name="Perplexity (Sonar Reasoning Pro)",
            context_window=128_000,
            max_output_tokens=8192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Sonar Reasoning Pro - Step-by-step logical analysis with search-augmented generation",
            aliases=["sonar-reasoning", "pplx-reasoning"],
            intelligence_score=15,
        ),
        "sonar-deep-research": ModelCapabilities(
            provider=ProviderType.PERPLEXITY,
            model_name="sonar-deep-research",
            friendly_name="Perplexity (Sonar Deep Research)",
            context_window=128_000,
            max_output_tokens=8192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Sonar Deep Research - Comprehensive iterative analysis with multi-step research",
            aliases=["deep-research", "pplx-research"],
            intelligence_score=16,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Perplexity provider with API key."""
        kwargs.setdefault("base_url", "https://api.perplexity.ai")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.PERPLEXITY

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in (
            "sonar-reasoning-pro",
            "sonar-deep-research",
        )
