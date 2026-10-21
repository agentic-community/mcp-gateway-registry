"""Unit tests for remote embeddings model-name resolution (issue #1826)."""

import pytest

from registry.utils.embeddings_provider import (
    PROVIDER_AZURE,
    PROVIDER_BEDROCK_COHERE,
    PROVIDER_BEDROCK_TITAN,
    PROVIDER_OPENAI,
    resolve_remote_model,
)


@pytest.mark.unit
class TestSupportedNames:
    """Every model name the adapters serve resolves to a provider and native ID."""

    @pytest.mark.parametrize(
        ("model_name", "provider", "model_id"),
        [
            ("openai/text-embedding-3-small", PROVIDER_OPENAI, "text-embedding-3-small"),
            ("openai/text-embedding-ada-002", PROVIDER_OPENAI, "text-embedding-ada-002"),
            # Custom gateways use display names with spaces and parentheses.
            ("openai/My Embedding Model (v2)", PROVIDER_OPENAI, "My Embedding Model (v2)"),
            ("azure/my-deployment", PROVIDER_AZURE, "my-deployment"),
            (
                "bedrock/amazon.titan-embed-text-v1",
                PROVIDER_BEDROCK_TITAN,
                "amazon.titan-embed-text-v1",
            ),
            (
                "bedrock/amazon.titan-embed-text-v2:0",
                PROVIDER_BEDROCK_TITAN,
                "amazon.titan-embed-text-v2:0",
            ),
            (
                "bedrock/amazon.titan-embed-g1-text-02",
                PROVIDER_BEDROCK_TITAN,
                "amazon.titan-embed-g1-text-02",
            ),
            ("bedrock/cohere.embed-english-v3", PROVIDER_BEDROCK_COHERE, "cohere.embed-english-v3"),
            (
                "bedrock/cohere.embed-multilingual-v3",
                PROVIDER_BEDROCK_COHERE,
                "cohere.embed-multilingual-v3",
            ),
        ],
    )
    def test_resolves(self, model_name, provider, model_id):
        resolved = resolve_remote_model(model_name)
        assert resolved.provider == provider
        assert resolved.model_id == model_id

    def test_prefix_is_case_insensitive_and_id_keeps_case(self):
        resolved = resolve_remote_model("OpenAI/Text-Embedding-3-Small")
        assert resolved.provider == PROVIDER_OPENAI
        assert resolved.model_id == "Text-Embedding-3-Small"

    def test_surrounding_whitespace_ignored(self):
        assert resolve_remote_model("  openai/m  ").model_id == "m"


@pytest.mark.unit
class TestRejectedNames:
    """Unsupported names fail with a message that says what to use instead."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("cohere/embed-english-v3.0", "bedrock/cohere.embed-english-v3"),
            ("voyage/voyage-3", "openai/<model> with EMBEDDINGS_API_BASE"),
            ("voyage-3-large", "openai/<model> with EMBEDDINGS_API_BASE"),
        ],
    )
    def test_removed_prefix_names_replacement(self, model_name, expected):
        with pytest.raises(ValueError) as exc_info:
            resolve_remote_model(model_name)
        assert "removed with LiteLLM" in str(exc_info.value)
        assert expected in str(exc_info.value)

    def test_unknown_prefix_points_at_openai_compatible(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_remote_model("mistral/mistral-embed")
        assert "no supported provider prefix" in str(exc_info.value)
        assert "openai/<model> with EMBEDDINGS_API_BASE" in str(exc_info.value)

    def test_bare_name_points_at_sentence_transformers(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_PROVIDER=sentence-transformers"):
            resolve_remote_model("all-MiniLM-L6-v2")

    def test_unsupported_bedrock_model(self):
        with pytest.raises(ValueError, match="Amazon Titan text embeddings and Cohere"):
            resolve_remote_model("bedrock/amazon.titan-embed-image-v1")

    def test_empty_model_after_prefix(self):
        with pytest.raises(ValueError, match="nothing follows"):
            resolve_remote_model("openai/")

    def test_error_names_the_setting(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_MODEL_NAME='mistral/x'"):
            resolve_remote_model("mistral/x")
