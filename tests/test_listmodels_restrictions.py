"""Test listmodels tool respects model restrictions."""

import asyncio
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from providers.base import ModelProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ModelCapabilities, ProviderType
from tools.listmodels import ListModelsTool


class TestListModelsRestrictions(unittest.TestCase):
    """Test that listmodels handles provider listing correctly."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing registry state
        ModelProviderRegistry.clear_cache()

        # Create mock Gemini provider
        self.mock_gemini = MagicMock(spec=ModelProvider)
        self.mock_gemini.provider_type = ProviderType.GOOGLE
        self.mock_gemini.list_models.return_value = ["gemini-2.5-flash", "gemini-2.5-pro"]
        self.mock_gemini.get_capabilities_by_rank.return_value = []

    def tearDown(self):
        """Clean up after tests."""
        ModelProviderRegistry.clear_cache()
        # Clean up environment variables
        for key in ["GEMINI_API_KEY"]:
            os.environ.pop(key, None)

    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "gemini-test-key",
        },
    )
    @patch.object(ModelProviderRegistry, "get_available_models")
    @patch.object(ModelProviderRegistry, "get_provider")
    def test_listmodels_shows_configured_providers(self, mock_get_provider, mock_get_models):
        """Test that listmodels shows configured providers correctly."""
        # Mock provider registry
        def get_provider_side_effect(provider_type, force_new=False):
            if provider_type == ProviderType.GOOGLE:
                return self.mock_gemini
            return None

        mock_get_provider.side_effect = get_provider_side_effect

        # Mock available models
        mock_get_models.return_value = {
            "gemini-2.5-flash": ProviderType.GOOGLE,
            "gemini-2.5-pro": ProviderType.GOOGLE,
        }

        # Create tool and execute
        tool = ListModelsTool()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_contents = loop.run_until_complete(tool.execute({}))
        loop.close()

        # Extract text content from result
        result_text = result_contents[0].text

        # Parse JSON response
        result_json = json.loads(result_text)
        result = result_json["content"]

        # Check that Gemini section exists
        self.assertIn("Google Gemini", result)


if __name__ == "__main__":
    unittest.main()
