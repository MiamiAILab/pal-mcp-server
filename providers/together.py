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
            # Reasoning model: emits a hidden <think> trace before its final
            # answer. At 16384 it exhausted the budget mid-trace and returned
            # finish_reason=length with empty visible content (silent empty
            # verdicts in consensus). Raised to 65536 so the trace + final
            # answer both fit. Genesis fix 2026-05-30 (SOL-338 adjacent).
            max_output_tokens=65536,
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
        # Removed 2026-06-27 (GENESIS-096): "Qwen/Qwen3-Next-80B-A3B-Thinking"
        # is listed in the Together catalog but is NON-SERVERLESS (probed: HTTP
        # 400 "Unable to access non-serverless model ... create a dedicated
        # endpoint"). It cannot be invoked on the shared serverless path consensus
        # uses. Removed rather than repointed — provisioning a paid dedicated
        # endpoint is a cost decision for Mario; the serverless Qwen/Qwen3.5-397B-
        # A17B (qwen3.5-397b) already covers the Qwen thinking seat.
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
        )
