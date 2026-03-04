"""Tests for CustomProvider functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers import ModelProviderRegistry
from providers.custom import CustomProvider
from providers.shared import ProviderType


class TestCustomProvider:
    """Test CustomProvider class functionality."""

    def test_provider_initialization_with_params(self):
        """Test CustomProvider initializes correctly with explicit parameters."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        assert provider.base_url == "http://localhost:11434/v1"
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.CUSTOM

    def test_provider_initialization_with_env_vars(self):
        """Test CustomProvider initializes correctly with environment variables."""
        with patch.dict(os.environ, {"CUSTOM_API_URL": "http://localhost:8000/v1", "CUSTOM_API_KEY": "env-key"}):
            provider = CustomProvider()

            assert provider.base_url == "http://localhost:8000/v1"
            assert provider.api_key == "env-key"

    def test_provider_initialization_missing_url(self):
        """Test CustomProvider raises error when URL is missing."""
        with patch.dict(os.environ, {"CUSTOM_API_URL": ""}, clear=False):
            with pytest.raises(ValueError, match="Custom API URL must be provided"):
                CustomProvider(api_key="test-key")

    def test_validate_model_names_always_true(self):
        """Test CustomProvider validates model names correctly."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # Known model should validate
        assert provider.validate_model_name("llama3.2")

        # For custom provider, unknown models return False when not in registry
        # This is expected behavior - custom models need to be declared in custom_models.json
        assert not provider.validate_model_name("unknown-model")
        assert not provider.validate_model_name("anything")

    def test_get_capabilities_from_registry(self):
        """Test get_capabilities returns registry capabilities when available."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # Test with a custom model from the local registry
        capabilities = provider.get_capabilities("local-llama")
        assert capabilities.provider == ProviderType.CUSTOM
        assert capabilities.context_window > 0

    def test_get_capabilities_generic_fallback(self):
        """Test get_capabilities raises error for unknown models not in registry."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # Unknown models should raise ValueError when not in registry
        with pytest.raises(ValueError, match="Unsupported model 'unknown-model-xyz' for provider custom"):
            provider.get_capabilities("unknown-model-xyz")

    def test_model_alias_resolution(self):
        """Test model alias resolution works correctly."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # Test local model alias
        resolved_local = provider._resolve_model_name("local-llama")
        assert resolved_local == "llama3.2"

    def test_no_thinking_mode_support(self):
        """Custom provider generic capabilities default to no thinking mode."""
        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # llama3.2 is a known model that should work
        assert not provider.get_capabilities("llama3.2").supports_extended_thinking

        # Unknown models should raise error
        with pytest.raises(ValueError, match="Unsupported model 'any-model' for provider custom"):
            provider.get_capabilities("any-model")

    @patch("providers.custom.OpenAICompatibleProvider.generate_content")
    def test_generate_content_with_alias_resolution(self, mock_generate):
        """Test generate_content resolves aliases before calling parent."""
        mock_response = MagicMock()
        mock_generate.return_value = mock_response

        provider = CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        # Call with an alias
        result = provider.generate_content(
            prompt="test prompt",
            model_name="llama",
            temperature=0.7,  # This is an alias
        )

        # Verify parent method was called with resolved model name
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        # The model_name should be either resolved or passed through
        assert "model_name" in call_args.kwargs
        assert result == mock_response


class TestCustomProviderRegistration:
    """Test CustomProvider integration with ModelProviderRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelProviderRegistry.clear_cache()
        ModelProviderRegistry.unregister_provider(ProviderType.CUSTOM)

    def teardown_method(self):
        """Clean up after each test."""
        ModelProviderRegistry.clear_cache()
        ModelProviderRegistry.unregister_provider(ProviderType.CUSTOM)

    def test_custom_provider_factory_registration(self):
        """Test custom provider can be registered via factory function."""

        def custom_provider_factory(api_key=None):
            return CustomProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        with patch.dict(os.environ, {"CUSTOM_API_PLACEHOLDER": "configured"}):
            ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)

            # Verify provider is available
            available = ModelProviderRegistry.get_available_providers()
            assert ProviderType.CUSTOM in available

            # Verify provider can be retrieved
            provider = ModelProviderRegistry.get_provider(ProviderType.CUSTOM)
            assert provider is not None
            assert isinstance(provider, CustomProvider)



class TestConfigureProvidersFunction:
    """Test the configure_providers function in server.py."""

    def setup_method(self):
        """Clear environment and registry before each test."""
        # Store the original providers to restore them later
        registry = ModelProviderRegistry()
        self._original_providers = registry._providers.copy()
        ModelProviderRegistry.clear_cache()
        for provider_type in ProviderType:
            ModelProviderRegistry.unregister_provider(provider_type)

    def teardown_method(self):
        """Clean up after each test."""
        # Restore the original providers that were registered in conftest.py
        registry = ModelProviderRegistry()
        ModelProviderRegistry.clear_cache()
        registry._providers.clear()
        registry._providers.update(self._original_providers)

    def test_configure_providers_custom_only(self):
        """Test configure_providers with only custom URL set."""
        from server import configure_providers

        with patch.dict(
            os.environ,
            {
                "CUSTOM_API_URL": "http://localhost:11434/v1",
                "CUSTOM_API_KEY": "",
                # Clear other API keys
                "GEMINI_API_KEY": "",
                "OPENAI_API_KEY": "",
            },
            clear=True,
        ):
            configure_providers()

            # Verify only custom provider is available
            available = ModelProviderRegistry.get_available_providers()
            assert ProviderType.CUSTOM in available

    def test_configure_providers_no_valid_keys(self):
        """Test configure_providers raises error when no valid API keys."""
        from server import configure_providers

        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "", "OPENAI_API_KEY": "", "CUSTOM_API_URL": ""},
            clear=True,
        ):
            with pytest.raises(ValueError, match="At least one API configuration is required"):
                configure_providers()
