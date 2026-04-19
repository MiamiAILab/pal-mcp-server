"""Alibaba Cloud Model Studio (DashScope / Qwen) provider implementation."""

import logging
from typing import Optional

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class AlibabaModelProvider(OpenAICompatibleProvider):
    """Alibaba Cloud Model Studio (Qwen) provider via OpenAI-compatible endpoint."""

    FRIENDLY_NAME = "Alibaba Model Studio"

    MODEL_CAPABILITIES = {
        "qwen3.6-plus": ModelCapabilities(
            provider=ProviderType.ALIBABA,
            model_name="qwen3.6-plus",
            friendly_name="Alibaba (Qwen3.6 Plus)",
            context_window=262_144,
            max_output_tokens=32_768,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Qwen3.6 Plus (256K context) - Flagship native vision-language model with significant coding capability leap over 3.5 series",
            aliases=["qwen3.6", "qwen-plus-latest", "qwen-flagship"],
            intelligence_score=17,
            allow_code_generation=True,
        ),
        "qwen3-max": ModelCapabilities(
            provider=ProviderType.ALIBABA,
            model_name="qwen3-max",
            friendly_name="Alibaba (Qwen3 Max)",
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
            description="Qwen3 Max (256K context) - Previous flagship general-purpose LLM, state-of-the-art agent programming and tool invocation",
            aliases=["qwen-max", "qwen3max"],
            intelligence_score=16,
            allow_code_generation=True,
        ),
        "qwen3.6-flash": ModelCapabilities(
            provider=ProviderType.ALIBABA,
            model_name="qwen3.6-flash",
            friendly_name="Alibaba (Qwen3.6 Flash)",
            context_window=131_072,
            max_output_tokens=16_384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Qwen3.6 Flash (128K context) - Cost-optimized native vision-language model, excels at agentic coding",
            aliases=["qwen-flash", "qwen3.6flash"],
            intelligence_score=13,
            allow_code_generation=True,
        ),
        "qwen3-vl-plus": ModelCapabilities(
            provider=ProviderType.ALIBABA,
            model_name="qwen3-vl-plus",
            friendly_name="Alibaba (Qwen3 VL Plus)",
            context_window=131_072,
            max_output_tokens=16_384,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Qwen3 VL Plus (128K context) - Vision-language model with thinking + non-thinking modes, strong OS-World visual agent benchmarks",
            aliases=["qwen-vl", "qwen3-vl", "qwen-vision"],
            intelligence_score=14,
        ),
        "qwen3-coder-plus": ModelCapabilities(
            provider=ProviderType.ALIBABA,
            model_name="qwen3-coder-plus",
            friendly_name="Alibaba (Qwen3 Coder Plus)",
            context_window=262_144,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
            description="Qwen3 Coder Plus (256K context) - Dedicated coding agent with strong tool invocation and repo-level understanding",
            aliases=["qwen-coder", "qwen3-coder"],
            intelligence_score=15,
            allow_code_generation=True,
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Alibaba Model Studio provider."""
        base_url = get_env("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        kwargs.setdefault("base_url", base_url)
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        return ProviderType.ALIBABA

    def supports_thinking_mode(self, model_name: str) -> bool:
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in (
            "qwen3.6-plus",
            "qwen3-max",
            "qwen3-vl-plus",
        )
