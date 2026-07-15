"""Moonshot AI (Kimi K2.5) model provider implementation."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    FixedTemperatureConstraint,
    ModelCapabilities,
    ProviderType,
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
        # Repointed 2026-06-27 (GENESIS-096): the prior "kimi-k2-thinking",
        # "kimi-k2-thinking-turbo", and "kimi-k2-turbo-preview" entries all 404'd
        # against the live Moonshot catalog (probed: api.moonshot.ai/v1/models ->
        # kimi-k2.5, kimi-k2.6, kimi-k2.7-code, kimi-k2.7-code-highspeed). The two
        # turbo entries were removed (no live equivalent); the dedicated-thinking
        # entry is repointed to kimi-k2.6, the newest live thinking-capable model
        # (probed finish_reason=stop, visible content, reasoning_content present).
        # Legacy thinking aliases retained so existing fallback references resolve.
        "kimi-k2.6": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2.6",
            friendly_name="Moonshot AI (Kimi K2.6)",
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
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Kimi K2.6 (256K context) - Deep-reasoning MoE with long CoT traces and multi-step tool calls (live successor to the retired k2-thinking line)",
            aliases=["k2.6", "kimi2.6", "k2-thinking", "kimi-thinking"],
            intelligence_score=17,
        ),
        # Added 2026-07-12 (Genesis, GENESIS-097 batch): Kimi K2.7 Code, the
        # dedicated code-lane seat. Probed live against api.moonshot.ai/v1/models
        # (catalog includes kimi-k2.7-code + kimi-k2.7-code-highspeed alongside
        # kimi-k2.5 and kimi-k2.6, all retained). Code-generation flagship of the
        # Kimi line.
        #
        # Renamed -direct 2026-07-15 (Alan SM ruling on PR #12 frontier-seat
        # governance flag): kimi-k2.7-code is "US-brokered only" — the bare
        # canonical name + clean aliases (kimi-k2.7-code/k2.7-code/kimi-k2.7/
        # k2.7/kimi-code) now resolve to the US-brokered OpenRouter seat
        # (moonshotai/kimi-k2.7-code, pinned deepinfra, conf/openrouter_models.json).
        # This DIRECT-CHINA entry (api.moonshot.ai) = LOW-sensitivity ONLY per
        # evidence-regime policy §4 (Chinese-lineage frontier seat, in
        # GEOPOLITICAL_PROVIDERS -> excluded on sensitive content). Kept only so
        # explicit -direct references resolve; it must NOT be the default seat.
        # Same GENESIS-096 pattern as MiniMax-M2-direct. NOTE: MOONSHOT outranks
        # OPENROUTER in registry PROVIDER_PRIORITY_ORDER, so stripping the clean
        # aliases here is what makes the US-broker pin actually win.
        "kimi-k2.7-code-direct": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2.7-code",
            friendly_name="Moonshot AI (Kimi K2.7 Code direct)",
            context_window=262_144,
            max_output_tokens=65_535,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Kimi K2.7 Code (256K context) - Dedicated agentic-coding MoE. DIRECT-CHINA (api.moonshot.ai) = LOW-sensitivity ONLY. Renamed -direct 2026-07-15 (Alan SM, PR #12) so the bare `kimi-k2.7-code`/`k2.7`/`kimi-code` seat resolves to US-brokered OpenRouter (moonshotai/kimi-k2.7-code, deepinfra-pinned, MODERATE-eligible).",
            aliases=["k2.7-code-direct", "kimi-code-direct"],
            intelligence_score=18,
            allow_code_generation=True,
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
            "kimi-k2.6",
            "kimi-k2.7-code-direct",
        )
