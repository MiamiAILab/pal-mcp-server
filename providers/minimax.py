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
        "minimax-m2.5": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="minimax-m2.5",
            friendly_name="MiniMax AI (M2.5)",
            context_window=200_000,
            max_output_tokens=16384,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="MiniMax M2.5 (200K context) - SOTA coding (SWE-Bench 80.2%), agentic tool use, search. Released Feb 2026",
            aliases=["m2.5", "minimax"],
            intelligence_score=18,
            allow_code_generation=True,
        ),
        "minimax-m2": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="minimax-m2",
            friendly_name="MiniMax AI (M2)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="MiniMax M2 (200K context) - Agent-native model optimized for coding and agentic workflows",
            aliases=["m2", "minimax-coding"],
        ),
        "minimax-text-01": ModelCapabilities(
            provider=ProviderType.MINIMAX,
            model_name="minimax-text-01",
            friendly_name="MiniMax AI (Text-01)",
            context_window=4_000_000,  # 4M tokens
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="MiniMax Text-01 (4M context) - 456B parameter model with 45.9B activated per token for long-context processing",
            aliases=["text-01", "minimax-01"],
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
