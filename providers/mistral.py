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
            context_window=256_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.7),
            description="Mistral Large 3 (256K context) - 41B active/675B total MoE flagship, multimodal with agentic tools",
            aliases=["mistral-large", "mistral", "large3", "large"],
            intelligence_score=17,
            allow_code_generation=True,
        ),
        "magistral-medium-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="magistral-medium-latest",
            friendly_name="Mistral (Magistral Medium 1.2)",
            context_window=128_000,
            max_output_tokens=40_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Magistral Medium 1.2 (128K context) - Frontier multimodal reasoning with always-on thinking mode",
            aliases=["magistral-medium", "magistral", "magistral-1.2"],
            intelligence_score=16,
        ),
        "magistral-small-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="magistral-small-latest",
            friendly_name="Mistral (Magistral Small 1.2)",
            context_window=128_000,
            max_output_tokens=40_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Magistral Small 1.2 (128K context) - 24B open-source reasoning model (Apache 2.0) with vision",
            aliases=["magistral-small"],
            intelligence_score=13,
        ),
        "mistral-medium-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="mistral-medium-latest",
            friendly_name="Mistral (Medium 3.1)",
            context_window=128_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.7),
            description="Mistral Medium 3.1 (128K context) - Balanced performance/cost multimodal model",
            aliases=["mistral-medium", "medium"],
            intelligence_score=14,
        ),
        "mistral-small-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="mistral-small-latest",
            friendly_name="Mistral (Small 4)",
            context_window=128_000,
            max_output_tokens=32_768,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.7),
            description="Mistral Small 4 (128K context) - Unifies Magistral + Devstral + Small-Instruct (Apache 2.0)",
            aliases=["mistral-small", "small"],
            intelligence_score=13,
        ),
        "devstral-medium-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="devstral-medium-latest",
            friendly_name="Mistral (Devstral 2)",
            context_window=256_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Devstral 2 (256K context) - Flagship coding-agent model with multi-file edits",
            aliases=["devstral", "devstral-medium"],
            intelligence_score=15,
            allow_code_generation=True,
        ),
        "devstral-small-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="devstral-small-latest",
            friendly_name="Mistral (Devstral Small 2)",
            context_window=256_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Devstral Small 2 (256K context) - 24B open-source coding agent (Apache 2.0)",
            aliases=["devstral-small"],
            intelligence_score=13,
        ),
        "codestral-latest": ModelCapabilities(
            provider=ProviderType.MISTRAL,
            model_name="codestral-latest",
            friendly_name="Mistral (Codestral 25.08)",
            context_window=256_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),
            description="Codestral 25.08 (256K context) - FIM/completion specialist for IDE code generation",
            aliases=["codestral"],
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
            "mistral-small-latest",
        )
