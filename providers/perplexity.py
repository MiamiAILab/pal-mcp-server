"""Perplexity AI (Sonar) model provider implementation."""

import logging
import re
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)

logger = logging.getLogger(__name__)

# Patterns the underlying Llama-based Sonar model emits when its search
# backend fails to ground the response (the so-called "fail-open" mode).
# We only flag a response as fail-open when the content STARTS with one of
# these patterns AND no citations were returned. Conservative on purpose —
# false positives would force unnecessary retries.
_SONAR_FAIL_OPEN_PATTERNS = [
    r"^\s*i (?:cannot|can't|am unable to) access (?:real-?time|live|current|recent)",
    r"^\s*i (?:do not|don't) have access to (?:real-?time|live|current|recent)",
    r"^\s*i (?:cannot|can't|am unable to) (?:browse|search) (?:the )?(?:web|internet)",
    r"^\s*i (?:do not|don't) have the ability to (?:access|browse|search)",
    r"^\s*as an ai(?:[, ]).*(?:cannot|don't have|unable)",
    r"^\s*unfortunately,?\s+i (?:cannot|can't|don't have)",
]
_SONAR_FAIL_OPEN_REGEX = re.compile("|".join(_SONAR_FAIL_OPEN_PATTERNS), re.IGNORECASE)


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

    def _postprocess_response(
        self,
        raw_response,
        model_response: ModelResponse,
        resolved_model: str,
    ) -> ModelResponse:
        """Surface Sonar grounding signals and detect fail-open responses.

        Perplexity Sonar returns a ``citations`` field at the response root
        when its web-search backend successfully grounds the answer. When the
        backend hiccups (rate-limit, transient outage), the underlying
        Llama-based model still answers — but with a generic refusal pattern
        and no citations. zen-mcp-server's OpenAI-compatible response capture
        does not see these extension fields, so without this hook a fail-open
        response is structurally indistinguishable from a successful one.

        We do two things here:

        1. Always attach ``citations``, ``citations_count``, and
           ``grounding_active`` to ``model_response.metadata`` so callers can
           inspect grounding status programmatically (transparency layer).
        2. If grounding is inactive AND the content matches a known fail-open
           pattern, raise ``RuntimeError``. The surrounding retry loop in
           ``OpenAICompatibleProvider._attempt`` will retry with progressive
           delays; if all 4 attempts fail-open, the error propagates and the
           caller can fall back to a different tool (WebSearch, etc.).
        """

        # Citations may surface as an attribute (newer SDKs) or only via
        # ``model_dump()`` (older SDKs that drop unknown fields). Try both.
        citations = getattr(raw_response, "citations", None)
        if citations is None:
            try:
                citations = raw_response.model_dump().get("citations")
            except Exception:
                citations = None

        citations_list = list(citations) if citations else []
        citations_count = len(citations_list)
        grounding_active = citations_count > 0

        model_response.metadata["citations"] = citations_list
        model_response.metadata["citations_count"] = citations_count
        model_response.metadata["grounding_active"] = grounding_active

        if not grounding_active:
            content = model_response.content or ""
            if _SONAR_FAIL_OPEN_REGEX.search(content[:400]):
                raise RuntimeError(
                    f"Perplexity Sonar returned a response without grounding "
                    f"citations and the content matches a known fail-open "
                    f"refusal pattern (model={resolved_model}). The search "
                    f"backend likely hiccupped. Retrying."
                )

        return model_response
