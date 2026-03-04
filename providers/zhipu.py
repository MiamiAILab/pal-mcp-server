"""Zhipu AI (BigModel/GLM-4) model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)


class ZhipuModelProvider(OpenAICompatibleProvider):
    """Zhipu AI GLM-4 API provider (open.bigmodel.cn)."""

    FRIENDLY_NAME = "Zhipu AI"

    # Model configurations using ModelCapabilities objects
    MODEL_CAPABILITIES = {
        "glm-4-flash": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4-flash",
            friendly_name="Zhipu AI (GLM-4 Flash)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=4096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,  # Text-only
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4 Flash (128K context) - Fast and cost-effective model for general tasks",
            aliases=["glm4-flash", "glm-flash", "zhipu-flash"],
        ),
        "glm-4": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4",
            friendly_name="Zhipu AI (GLM-4)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,  # Text-only
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4 (128K context) - Flagship model with balanced performance and capability",
            aliases=["glm4", "zhipu", "zhipu-glm4"],
        ),
        "glm-4-plus": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4-plus",
            friendly_name="Zhipu AI (GLM-4 Plus)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=16384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,  # Text-only
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4 Plus (128K context) - Enhanced model with improved reasoning and longer outputs",
            aliases=["glm4-plus", "glm-plus", "zhipu-plus"],
        ),
        "glm-4v": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4v",
            friendly_name="Zhipu AI (GLM-4V)",
            context_window=128_000,  # 128K tokens
            max_output_tokens=4096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,  # Vision capability
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4V (128K context) - Multimodal model with vision capabilities",
            aliases=["glm4v", "glm-vision", "zhipu-vision"],
        ),
        "glm-5": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-5",
            friendly_name="Zhipu AI (GLM-5)",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-5 (200K context, 128K output) - Flagship 744B MoE reasoning model for complex system engineering and agentic tasks",
            aliases=["glm5", "zhipu-5"],
            intelligence_score=18,
        ),
        "glm-4.7": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4.7",
            friendly_name="Zhipu AI (GLM-4.7)",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4.7 (200K context) - 400B param model with strong coding (84.9% LiveCodeBench) and reasoning",
            aliases=["glm47", "zhipu-4.7"],
            intelligence_score=17,
        ),
        "glm-4.7-flash": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4.7-flash",
            friendly_name="Zhipu AI (GLM-4.7 Flash)",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4.7 Flash (200K context) - Free-tier fast model with strong coding and reasoning",
            aliases=["glm47-flash", "zhipu-flash-4.7"],
            intelligence_score=14,
        ),
        "glm-4.6": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4.6",
            friendly_name="Zhipu AI (GLM-4.6)",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4.6 (200K context) - Coding-focused flagship model",
            aliases=["glm46", "zhipu-4.6"],
            intelligence_score=16,
        ),
        "glm-4.6v": ModelCapabilities(
            provider=ProviderType.ZHIPU,
            model_name="glm-4.6v",
            friendly_name="Zhipu AI (GLM-4.6V)",
            context_window=128_000,
            max_output_tokens=4096,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="GLM-4.6V (128K context) - Vision model with reasoning capabilities",
            aliases=["glm46v", "zhipu-vision-4.6"],
            intelligence_score=15,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Zhipu AI provider with API key."""
        # Set Zhipu AI base URL
        kwargs.setdefault("base_url", "https://open.bigmodel.cn/api/paas/v4")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ZHIPU
