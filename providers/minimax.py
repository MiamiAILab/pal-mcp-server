"""MiniMax AI model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class MiniMaxModelProvider(OpenAICompatibleProvider):
    """MiniMax AI API provider (api.minimax.io)."""

    FRIENDLY_NAME = "MiniMax AI"

    # Model configurations using ModelCapabilities objects
    MODEL_CAPABILITIES = {
        "MiniMax-M2.7": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2.7",
            friendly_name="MiniMax AI (M2.7)",
            context_window=204_800,
            max_output_tokens=131_072,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2.7 (200K context, 128K output) - Self-evolving agentic SOTA rivaling Claude Opus/GPT-5 on long-chain tool use",
            aliases=["m2.7", "minimax", "minimax-m2.7"],
            intelligence_score=19,
            allow_code_generation=True,
        ),
        "MiniMax-M2.7-highspeed": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2.7-highspeed",
            friendly_name="MiniMax AI (M2.7 Highspeed)",
            context_window=204_800,
            max_output_tokens=131_072,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2.7 Highspeed (200K context) - Low-latency variant of M2.7 with same quality",
            aliases=["m2.7-fast", "minimax-fast"],
            intelligence_score=18,
            allow_code_generation=True,
        ),
        "MiniMax-M2.5": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2.5",
            friendly_name="MiniMax AI (M2.5)",
            context_window=204_800,
            max_output_tokens=65_536,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2.5 (200K context) - SWE-Bench Verified 80.2%, BrowseComp 76.3%; 230B MoE / 10B active",
            aliases=["m2.5"],
            intelligence_score=17,
            allow_code_generation=True,
        ),
        "MiniMax-M2.5-highspeed": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2.5-highspeed",
            friendly_name="MiniMax AI (M2.5 Highspeed)",
            context_window=204_800,
            max_output_tokens=65_536,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2.5 Highspeed (200K context) - Fast-path variant of M2.5",
            aliases=["m2.5-fast"],
            intelligence_score=16,
            allow_code_generation=True,
        ),
        "MiniMax-M2.1": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2.1",
            friendly_name="MiniMax AI (M2.1)",
            context_window=204_800,
            max_output_tokens=131_072,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2.1 (200K context) - Multi-language coding focus with strong VIBE-Web/Android scores",
            aliases=["m2.1"],
            intelligence_score=15,
        ),
        "MiniMax-M2": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="MiniMax-M2",
            friendly_name="MiniMax AI (M2)",
            context_window=204_800,
            max_output_tokens=131_072,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            description="MiniMax M2 (200K context) - Original agent-native coding model; 230B MoE / 10B active",
            aliases=["m2", "minimax-coding"],
            intelligence_score=14,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize MiniMax AI provider with API key."""
        # Set MiniMax AI base URL (OpenAI-compatible endpoint)
        kwargs.setdefault("base_url", "https://api.minimax.io/v1")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.MINIMAX
