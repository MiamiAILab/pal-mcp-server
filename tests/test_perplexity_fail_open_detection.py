"""Tests for Perplexity Sonar fail-open detection.

Sonar models combine a Llama-based LLM with a web-search backend. When the
search backend has a transient outage, the LLM still answers — but with a
generic refusal pattern and no citations. Without explicit detection, this
fail-open response is structurally indistinguishable from a successful
grounded one at the OpenAI-compatible API layer.

These tests exercise the postprocess hook added in providers/perplexity.py
that surfaces grounding signals into ``ModelResponse.metadata`` and raises
when a fail-open response is detected (so the surrounding retry loop fires).

See ``docs/fallback-policy.md`` for the design rationale.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.perplexity import (
    _SONAR_FAIL_OPEN_REGEX,
    PerplexityModelProvider,
)
from providers.shared import ModelResponse, ProviderType


def _make_model_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        usage={},
        model_name="sonar-deep-research",
        friendly_name="Perplexity AI",
        provider=ProviderType.PERPLEXITY,
        metadata={
            "finish_reason": "stop",
            "model": "sonar-deep-research",
            "id": "test-id",
            "created": 0,
        },
    )


class TestSonarFailOpenRegex:
    """The regex itself — conservative, matches only known refusal openings."""

    @pytest.mark.parametrize(
        "content",
        [
            "I cannot access real-time information about that.",
            "I can't access live data right now.",
            "I am unable to access current data.",
            "I do not have access to real-time information.",
            "I don't have access to live information.",
            "I cannot browse the web at this time.",
            "I can't search the internet right now.",
            "I do not have the ability to access external sources.",
            "I don't have the ability to browse the web.",
            "As an AI, I cannot retrieve real-time data.",
            "Unfortunately, I cannot access live data for you.",
            "Unfortunately, I don't have real-time information available.",
        ],
    )
    def test_matches_known_failopen_openings(self, content: str) -> None:
        assert _SONAR_FAIL_OPEN_REGEX.search(content) is not None

    @pytest.mark.parametrize(
        "content",
        [
            "Here is the latest information based on my web search results.",
            "According to recent reports, the situation is...",
            "The company announced on April 23 that...",
            "I cannot guarantee accuracy, but here is what I found.",
            "Citations: [1] example.com, [2] news.example.org",
        ],
    )
    def test_does_not_match_grounded_responses(self, content: str) -> None:
        assert _SONAR_FAIL_OPEN_REGEX.search(content) is None


class TestPerplexityPostprocess:
    """The postprocess hook — citation surfacing + fail-open raising."""

    def setup_method(self) -> None:
        self.provider = PerplexityModelProvider(api_key="test-key")

    def test_grounded_response_attaches_citations_metadata(self) -> None:
        raw = SimpleNamespace(citations=["https://example.com/a", "https://example.org/b"])
        model_response = _make_model_response(
            "The latest data shows X. According to recent sources..."
        )

        result = self.provider._postprocess_response(
            raw, model_response, "sonar-deep-research"
        )

        assert result.metadata["citations_count"] == 2
        assert result.metadata["grounding_active"] is True
        assert result.metadata["citations"] == [
            "https://example.com/a",
            "https://example.org/b",
        ]

    def test_failopen_response_raises_runtime_error(self) -> None:
        raw = SimpleNamespace(citations=None)
        model_response = _make_model_response(
            "I cannot access real-time information about recent events."
        )

        with pytest.raises(RuntimeError, match="fail-open"):
            self.provider._postprocess_response(
                raw, model_response, "sonar-deep-research"
            )

    def test_no_citations_but_no_failopen_pattern_passes(self) -> None:
        # Edge case: zero citations but content doesn't match a refusal
        # pattern. We attach the metadata flags but DO NOT raise — false
        # positives would force unnecessary retries on legitimate edge cases.
        raw = SimpleNamespace(citations=[])
        model_response = _make_model_response(
            "Based on my training data, the principle of least privilege means..."
        )

        result = self.provider._postprocess_response(
            raw, model_response, "sonar"
        )

        assert result.metadata["citations_count"] == 0
        assert result.metadata["grounding_active"] is False
        # No RuntimeError raised — content didn't match fail-open pattern.

    def test_failopen_with_citations_does_not_raise(self) -> None:
        # If citations are present, the response is grounded by definition,
        # even if the content somehow contains refusal-like phrasing.
        raw = SimpleNamespace(citations=["https://example.com"])
        model_response = _make_model_response(
            "I cannot access real-time information directly, but my search "
            "found the following relevant sources."
        )

        result = self.provider._postprocess_response(
            raw, model_response, "sonar-pro"
        )

        assert result.metadata["grounding_active"] is True
        # No RuntimeError because grounding_active gates the raise.

    def test_citations_via_model_dump_fallback(self) -> None:
        # Older OpenAI SDKs drop unknown extension fields from the response
        # object but preserve them in model_dump(). The provider must read
        # both paths.
        raw = Mock(spec=[])  # Empty spec — no auto-attrs
        raw.model_dump = Mock(return_value={"citations": ["https://example.com"]})
        model_response = _make_model_response("Some grounded content here.")

        result = self.provider._postprocess_response(
            raw, model_response, "sonar-pro"
        )

        assert result.metadata["citations_count"] == 1
        assert result.metadata["grounding_active"] is True
