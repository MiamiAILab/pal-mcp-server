"""Together.ai model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class TogetherModelProvider(OpenAICompatibleProvider):
    """Together.ai API provider (api.together.xyz)."""

    FRIENDLY_NAME = "Together AI"

    # Model configurations using ModelCapabilities objects
    MODEL_CAPABILITIES = {
        "Qwen/Qwen3.5-397B-A17B": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="Qwen/Qwen3.5-397B-A17B",
            friendly_name="Together AI (Qwen3.5 397B)",
            context_window=262_144,
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
            description="Qwen3.5 397B (256K context) - Latest Qwen MoE reasoning model with 17B active params, replaces Qwen3 235B",
            aliases=["qwen3-thinking", "qwen3-235b", "qwen-thinking", "qwen3.5", "qwen3.5-397b"],
            intelligence_score=19,
        ),
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            friendly_name="Together AI (Qwen3 Coder 480B)",
            context_window=262_144,
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
            description="Qwen3 Coder 480B (256K context) - Largest open coding model, 35B active params",
            aliases=["qwen3-coder", "qwen-coder", "qwen3-coder-480b"],
            intelligence_score=19,
            allow_code_generation=True,
        ),
        "Qwen/Qwen3-Coder-Next-FP8": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="Qwen/Qwen3-Coder-Next-FP8",
            friendly_name="Together AI (Qwen3 Coder Next)",
            context_window=262_144,
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
            description="Qwen3 Coder Next (256K context) - Latest Qwen coding model",
            aliases=["qwen3-coder-next", "qwen-coder-next"],
            intelligence_score=19,
            allow_code_generation=True,
        ),
        "Qwen/Qwen3-Next-80B-A3B-Thinking": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
            friendly_name="Together AI (Qwen3 Next 80B Thinking)",
            context_window=262_144,
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
            description="Qwen3 Next 80B Thinking (256K context) - Efficient MoE thinking model with only 3B active params",
            aliases=["qwen3-next", "qwen3-80b", "qwen-next"],
            intelligence_score=15,
        ),
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            friendly_name="Together AI (Qwen3 235B Instruct)",
            context_window=262_144,
            max_output_tokens=16384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Qwen3 235B Instruct (256K context) - Non-thinking variant optimized for throughput",
            aliases=["qwen3-instruct", "qwen3-235b-instruct"],
            intelligence_score=17,
        ),
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            friendly_name="Together AI (Llama 4 Maverick)",
            context_window=1_048_576,
            max_output_tokens=16384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Llama 4 Maverick (1M context) - 128-expert MoE model with 17B active/400B total params, multimodal",
            aliases=["llama4-maverick", "llama-4-maverick", "maverick"],
            intelligence_score=16,
        ),
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelCapabilities(
            provider=ProviderType.TOGETHER,
            model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            friendly_name="Together AI (Llama 4 Scout)",
            context_window=262_144,
            max_output_tokens=16384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Llama 4 Scout (256K context) - 16-expert MoE model with 17B active/109B total params, multimodal",
            aliases=["llama4-scout", "llama-4-scout", "scout", "llama4"],
            intelligence_score=14,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Together.ai provider with API key."""
        kwargs.setdefault("base_url", "https://api.together.xyz/v1")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.TOGETHER

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in (
            "Qwen/Qwen3.5-397B-A17B",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "Qwen/Qwen3-Coder-Next-FP8",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        )
