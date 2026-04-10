"""
Content Sensitivity Service

Detects sensitive business content in user prompts and returns provider
exclusion sets for data sovereignty protection. When enabled, prompts
containing financial, strategic, legal, or compliance keywords automatically
exclude providers hosted in jurisdictions with data sovereignty concerns
(e.g., Chinese AI providers).

Environment Variables:
- CONTENT_SENSITIVITY_FILTERING: Set to "true" to enable (default: disabled)
- SENSITIVE_KEYWORDS: Optional comma-separated list of additional keywords
"""

import logging
import re
from typing import Optional

from providers.shared import ProviderType
from utils.env import get_env

logger = logging.getLogger(__name__)

# Providers hosted in jurisdictions with data sovereignty concerns.
# When sensitive content is detected, these providers are excluded.
GEOPOLITICAL_PROVIDERS: frozenset[ProviderType] = frozenset(
    {
        ProviderType.MINIMAX,
        ProviderType.MOONSHOT,
        ProviderType.ZHIPU,
        ProviderType.CUSTOM,  # DeepSeek via custom endpoint
    }
)

# Default sensitive keywords organized by category.
# Matching is case-insensitive with word boundaries.
_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "investor",
        "valuation",
        "revenue",
        "cap table",
        "burn rate",
        "runway",
        "funding round",
        "pre-seed",
        "seed round",
        "series a",
        "series b",
        "pitch deck",
        "term sheet",
        "safe note",
        "convertible note",
        "due diligence",
        "financial projection",
        "profit margin",
        "balance sheet",
        "income statement",
        "cash flow",
    ],
    "strategic": [
        "proprietary",
        "trade secret",
        "competitive advantage",
        "product roadmap",
        "market positioning",
        "acquisition target",
        "exit strategy",
        "competitive analysis",
    ],
    "legal": [
        "nda",
        "non-disclosure",
        "confidential agreement",
        "licensing agreement",
        "ip assignment",
        "patent filing",
        "litigation",
        "compliance requirement",
        "regulatory filing",
    ],
    "personnel": [
        "salary",
        "compensation package",
        "equity grant",
        "vesting schedule",
        "termination",
        "performance review",
        "offer letter",
    ],
    "compliance": [
        "hipaa",
        "gdpr",
        "soc 2",
        "social security number",
        "credit card number",
    ],
}


class ContentSensitivityService:
    """Detects sensitive content and determines provider exclusions.

    When ``CONTENT_SENSITIVITY_FILTERING`` is set to ``"true"``, this service
    scans user prompts for keywords that indicate sensitive business content
    and returns the set of providers that should be excluded from model
    selection.

    When the feature flag is not set or is ``"false"``, all methods return
    safe no-op values (no sensitivity detected, no exclusions).
    """

    def __init__(self) -> None:
        self._enabled = get_env("CONTENT_SENSITIVITY_FILTERING", "").lower() == "true"
        self._patterns: list[tuple[str, re.Pattern]] = []

        if self._enabled:
            self._patterns = self._compile_patterns()
            logger.info(
                "Content sensitivity filtering ENABLED (%d keywords across %d categories)",
                len(self._patterns),
                len(_DEFAULT_KEYWORDS),
            )
        else:
            logger.debug("Content sensitivity filtering disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_sensitive(self, text: str) -> tuple[bool, list[str]]:
        """Check whether *text* contains sensitive content.

        Returns:
            A ``(is_sensitive, matched_keywords)`` tuple.  When the feature
            is disabled, always returns ``(False, [])``.
        """
        if not self._enabled or not text:
            return False, []

        text_lower = text.lower()
        matched: list[str] = []

        for keyword, pattern in self._patterns:
            if pattern.search(text_lower):
                matched.append(keyword)

        return bool(matched), matched

    @staticmethod
    def get_excluded_providers() -> frozenset[ProviderType]:
        """Return the set of providers to exclude for sensitive content."""
        return GEOPOLITICAL_PROVIDERS

    @staticmethod
    def is_provider_excluded(provider_type: ProviderType) -> bool:
        """Check if a single provider is in the geopolitical exclusion set."""
        return provider_type in GEOPOLITICAL_PROVIDERS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> list[tuple[str, re.Pattern]]:
        """Build compiled regex patterns from default + custom keywords."""
        keywords: list[str] = []

        # Collect defaults
        for category_words in _DEFAULT_KEYWORDS.values():
            keywords.extend(category_words)

        # Add custom keywords from env var (additive)
        custom = get_env("SENSITIVE_KEYWORDS", "")
        if custom:
            for kw in custom.split(","):
                cleaned = kw.strip().lower()
                if cleaned and cleaned not in keywords:
                    keywords.append(cleaned)

        # Compile with word boundaries to reduce false positives
        patterns: list[tuple[str, re.Pattern]] = []
        for kw in keywords:
            try:
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                patterns.append((kw, pattern))
            except re.error:
                logger.warning("Invalid sensitivity keyword pattern: %s", kw)

        return patterns


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_sensitivity_service: Optional[ContentSensitivityService] = None


def get_sensitivity_service() -> ContentSensitivityService:
    """Return the global ``ContentSensitivityService`` singleton."""
    global _sensitivity_service
    if _sensitivity_service is None:
        _sensitivity_service = ContentSensitivityService()
    return _sensitivity_service
