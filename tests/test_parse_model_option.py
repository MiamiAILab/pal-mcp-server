"""Tests for parse_model_option function."""

from server import parse_model_option


class TestParseModelOption:
    """Test cases for model option parsing."""

    def test_ollama_tag_parsed_as_option(self):
        """Test that Ollama tags are parsed as options."""
        model, option = parse_model_option("llama3.2:latest")
        assert model == "llama3.2"
        assert option == "latest"

    def test_consensus_stance_parsed_as_option(self):
        """Test that consensus stances are parsed as options."""
        model, option = parse_model_option("o3:for")
        assert model == "o3"
        assert option == "for"

        model, option = parse_model_option("gemini-2.5-pro:against")
        assert model == "gemini-2.5-pro"
        assert option == "against"

    def test_slash_model_suffix_parsed_as_option(self):
        """Test that suffixes on slash-prefixed models are parsed as options."""
        model, option = parse_model_option("openai/gpt-4:custom-tag")
        assert model == "openai/gpt-4"
        assert option == "custom-tag"

    def test_plain_model_name(self):
        """Test plain model names without colons."""
        model, option = parse_model_option("gpt-4")
        assert model == "gpt-4"
        assert option is None

    def test_url_not_parsed(self):
        """Test that URLs are not parsed for options."""
        model, option = parse_model_option("http://localhost:8080")
        assert model == "http://localhost:8080"
        assert option is None

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        model, option = parse_model_option("  llama3.2 : latest  ")
        assert model == "llama3.2"
        assert option == "latest"

    def test_colon_suffix_parsed_as_option(self):
        """Test that colon suffixes on slash-prefixed models are parsed as options."""
        model, option = parse_model_option("openai/gpt-3.5-turbo:free")
        assert model == "openai/gpt-3.5-turbo"
        assert option == "free"

        model, option = parse_model_option("openai/gpt-3.5-turbo:FREE")
        assert model == "openai/gpt-3.5-turbo"
        assert option == "FREE"
