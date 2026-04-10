"""Moonshot AI (Kimi K2.5) model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    FixedTemperatureConstraint,
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class MoonshotModelProvider(OpenAICompatibleProvider):
    """Moonshot AI Kimi API provider (api.moonshot.ai)."""

    FRIENDLY_NAME = "Moonshot AI"

    # Model configurations using ModelCapabilities objects
    MODEL_CAPABILITIES = {
        "kimi-k2.5": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2.5",
            friendly_name="Moonshot AI (Kimi K2.5)",
            context_window=262_144,
            max_output_tokens=65_535,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Kimi K2.5 (256K context) - Flagship native-multimodal MoE (1T/32B active) with instant + thinking modes, tool use, web search",
            aliases=["k2.5", "kimi2.5", "kimi", "moonshot"],
            intelligence_score=18,
            allow_code_generation=True,
        ),
        "kimi-k2-thinking": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2-thinking",
            friendly_name="Moonshot AI (Kimi K2 Thinking)",
            context_window=262_144,
            max_output_tokens=32_768,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Kimi K2 Thinking (256K context) - Dedicated deep reasoning model with multi-step tool calls and long CoT traces",
            aliases=["k2-thinking", "kimi-thinking"],
            intelligence_score=16,
        ),
        "kimi-k2-thinking-turbo": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2-thinking-turbo",
            friendly_name="Moonshot AI (Kimi K2 Thinking Turbo)",
            context_window=262_144,
            max_output_tokens=32_768,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Kimi K2 Thinking Turbo (256K context) - Fast reasoning variant, lower latency than K2 Thinking",
            aliases=["k2-thinking-turbo", "kimi-turbo"],
            intelligence_score=15,
        ),
        "kimi-k2-turbo-preview": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2-turbo-preview",
            friendly_name="Moonshot AI (Kimi K2 Turbo)",
            context_window=262_144,
            max_output_tokens=16_384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Kimi K2 Turbo (256K context) - Speed-optimized non-thinking K2 for high-throughput agents",
            aliases=["k2-turbo", "kimi-k2-turbo"],
            intelligence_score=14,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Moonshot AI provider with API key."""
        # Set Moonshot AI base URL (platform.moonshot.ai)
        kwargs.setdefault("base_url", "https://api.moonshot.ai/v1")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.MOONSHOT

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in (
            "kimi-k2.5",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        )
