"""Tests for content sensitivity detection and geopolitical provider filtering."""

import os
from unittest.mock import patch

import pytest

from providers.shared import ProviderType


# ---------------------------------------------------------------------------
# Fixture: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure a fresh ContentSensitivityService for each test."""
    import utils.content_sensitivity as cs

    cs._sensitivity_service = None
    yield
    cs._sensitivity_service = None


# ---------------------------------------------------------------------------
# Detection tests (feature enabled)
# ---------------------------------------------------------------------------


class TestSensitivityDetection:
    """Tests for keyword detection when the feature is enabled."""

    @pytest.fixture(autouse=True)
    def _enable(self):
        with patch.dict(os.environ, {"CONTENT_SENSITIVITY_FILTERING": "true"}):
            yield

    def test_detects_financial_keywords(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("Review the investor pipeline for our pre-seed round")
        assert is_sensitive
        assert "investor" in kw
        assert "pre-seed" in kw

    def test_detects_legal_keywords(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("We need to review the NDA before signing")
        assert is_sensitive
        assert "nda" in kw

    def test_detects_strategic_keywords(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("This is proprietary technology and a trade secret")
        assert is_sensitive
        assert "proprietary" in kw
        assert "trade secret" in kw

    def test_detects_personnel_keywords(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("Discuss the salary and equity grant for the new hire")
        assert is_sensitive
        assert "salary" in kw
        assert "equity grant" in kw

    def test_no_false_positive_on_general_code(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive(
            "Refactor the authentication module to use OAuth2 flow"
        )
        assert not is_sensitive
        assert kw == []

    def test_no_false_positive_on_architecture(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive(
            "Design a microservice event-driven architecture with Redis pub/sub"
        )
        assert not is_sensitive
        assert kw == []

    def test_case_insensitive(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("INVESTOR meeting about VALUATION")
        assert is_sensitive
        assert "investor" in kw
        assert "valuation" in kw

    def test_empty_input(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("")
        assert not is_sensitive
        assert kw == []

    def test_none_safe(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        # Passing None-ish empty value should not crash
        is_sensitive, kw = svc.is_sensitive("")
        assert not is_sensitive

    def test_word_boundary_prevents_false_positive(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        # "investigate" contains "invest" but not "investor"
        is_sensitive, kw = svc.is_sensitive("We need to investigate the bug in production")
        assert not is_sensitive

    def test_multi_word_keywords(self):
        from utils.content_sensitivity import get_sensitivity_service

        svc = get_sensitivity_service()
        is_sensitive, kw = svc.is_sensitive("The cap table needs updating before the next round")
        assert is_sensitive
        assert "cap table" in kw


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """Tests for the CONTENT_SENSITIVITY_FILTERING feature flag."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTENT_SENSITIVITY_FILTERING", None)
            from utils.content_sensitivity import get_sensitivity_service

            svc = get_sensitivity_service()
            is_sensitive, kw = svc.is_sensitive("investor valuation cap table")
            assert not is_sensitive
            assert kw == []

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"CONTENT_SENSITIVITY_FILTERING": "false"}):
            from utils.content_sensitivity import get_sensitivity_service

            svc = get_sensitivity_service()
            is_sensitive, kw = svc.is_sensitive("investor valuation cap table")
            assert not is_sensitive

    def test_enabled_when_true(self):
        with patch.dict(os.environ, {"CONTENT_SENSITIVITY_FILTERING": "true"}):
            from utils.content_sensitivity import get_sensitivity_service

            svc = get_sensitivity_service()
            is_sensitive, kw = svc.is_sensitive("investor valuation")
            assert is_sensitive


# ---------------------------------------------------------------------------
# Custom keywords from env var
# ---------------------------------------------------------------------------


class TestCustomKeywords:
    """Tests for SENSITIVE_KEYWORDS env var."""

    def test_custom_keywords_additive(self):
        with patch.dict(
            os.environ,
            {
                "CONTENT_SENSITIVITY_FILTERING": "true",
                "SENSITIVE_KEYWORDS": "moonshot,secret project",
            },
        ):
            from utils.content_sensitivity import get_sensitivity_service

            svc = get_sensitivity_service()
            # Custom keyword should trigger
            is_sensitive, kw = svc.is_sensitive("This is about the secret project")
            assert is_sensitive
            assert "secret project" in kw

            # Default keywords should still work
            is_sensitive2, kw2 = svc.is_sensitive("Review the investor pipeline")
            assert is_sensitive2
            assert "investor" in kw2


# ---------------------------------------------------------------------------
# Provider exclusion tests
# ---------------------------------------------------------------------------


class TestProviderExclusion:
    """Tests for geopolitical provider identification."""

    def test_excluded_providers_set(self):
        from utils.content_sensitivity import GEOPOLITICAL_PROVIDERS

        assert ProviderType.MINIMAX in GEOPOLITICAL_PROVIDERS
        assert ProviderType.MOONSHOT in GEOPOLITICAL_PROVIDERS
        assert ProviderType.ZHIPU in GEOPOLITICAL_PROVIDERS
        assert ProviderType.CUSTOM in GEOPOLITICAL_PROVIDERS

    def test_non_excluded_providers(self):
        from utils.content_sensitivity import GEOPOLITICAL_PROVIDERS

        assert ProviderType.OPENAI not in GEOPOLITICAL_PROVIDERS
        assert ProviderType.GOOGLE not in GEOPOLITICAL_PROVIDERS
        assert ProviderType.XAI not in GEOPOLITICAL_PROVIDERS
        assert ProviderType.MISTRAL not in GEOPOLITICAL_PROVIDERS
        assert ProviderType.PERPLEXITY not in GEOPOLITICAL_PROVIDERS
        assert ProviderType.TOGETHER not in GEOPOLITICAL_PROVIDERS

    def test_is_provider_excluded(self):
        from utils.content_sensitivity import ContentSensitivityService

        assert ContentSensitivityService.is_provider_excluded(ProviderType.ZHIPU)
        assert not ContentSensitivityService.is_provider_excluded(ProviderType.OPENAI)
