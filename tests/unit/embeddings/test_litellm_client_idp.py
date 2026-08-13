"""Unit tests for LiteLLMClient with token_provider (IdP auth mode)."""

import os
from unittest.mock import MagicMock, patch

from registry.embeddings.client import LiteLLMClient, create_embeddings_client


class TestLiteLLMClientTokenProvider:
    """Tests for the token_provider integration in LiteLLMClient."""

    def test_no_static_env_when_provider_present(self):
        """When token_provider is set, OPENAI_API_KEY should NOT be set."""
        os.environ.pop("OPENAI_API_KEY", None)
        provider_fn = MagicMock(return_value="dynamic-token")

        client = LiteLLMClient(
            model_name="openai/text-embedding-3-small",
            api_key="should-not-be-set-in-env",
            api_base="https://embeddings.example.com/v1",
            embedding_dimension=1536,
            token_provider=provider_fn,
        )

        assert os.environ.get("OPENAI_API_KEY") is None
        assert client.token_provider is provider_fn

    def test_static_env_set_when_no_provider(self):
        """Without token_provider, api_key sets the env var (existing behavior)."""
        os.environ.pop("OPENAI_API_KEY", None)

        client = LiteLLMClient(
            model_name="openai/text-embedding-3-small",
            api_key="sk-static-key",
            embedding_dimension=1536,
        )

        assert os.environ.get("OPENAI_API_KEY") == "sk-static-key"
        os.environ.pop("OPENAI_API_KEY", None)

    @patch("litellm.embedding")
    def test_encode_injects_fresh_token(self, mock_embedding):
        """token_provider() is called on each encode() and passed as api_key."""
        call_count = 0

        def provider_fn():
            nonlocal call_count
            call_count += 1
            return f"token-{call_count}"

        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, key: (
            [{"embedding": [0.1] * 384}] if key == "data" else None
        )
        mock_embedding.return_value = mock_response

        client = LiteLLMClient(
            model_name="openai/mock-model",
            api_base="https://embeddings.example.com/v1",
            embedding_dimension=384,
            token_provider=provider_fn,
        )

        client.encode(["hello"])
        assert mock_embedding.call_args[1]["api_key"] == "token-1"

        client.encode(["world"])
        assert mock_embedding.call_args[1]["api_key"] == "token-2"

        assert call_count == 2

    @patch("litellm.embedding")
    def test_encode_no_api_key_when_no_provider(self, mock_embedding):
        """Without token_provider, api_key is not passed in kwargs."""
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, key: (
            [{"embedding": [0.1] * 384}] if key == "data" else None
        )
        mock_embedding.return_value = mock_response

        client = LiteLLMClient(
            model_name="openai/mock-model",
            api_base="https://embeddings.example.com/v1",
            embedding_dimension=384,
        )

        client.encode(["test"])
        assert "api_key" not in mock_embedding.call_args[1]


class TestCreateEmbeddingsClientWithTokenProvider:
    """Tests for the factory function passing token_provider."""

    def test_factory_passes_token_provider_to_litellm(self):
        provider_fn = MagicMock(return_value="test-token")

        client = create_embeddings_client(
            provider="litellm",
            model_name="openai/text-embedding-3-small",
            api_base="https://embeddings.example.com/v1",
            embedding_dimension=1536,
            token_provider=provider_fn,
        )

        assert isinstance(client, LiteLLMClient)
        assert client.token_provider is provider_fn

    def test_factory_ignores_token_provider_for_sentence_transformers(self):
        """token_provider is not applicable to sentence-transformers."""
        provider_fn = MagicMock(return_value="test-token")

        with patch("registry.embeddings.client.SentenceTransformersClient") as mock_st:
            mock_st.return_value = MagicMock()
            client = create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                token_provider=provider_fn,
            )
            mock_st.assert_called_once()
            call_kwargs = mock_st.call_args[1]
            assert "token_provider" not in call_kwargs

    def test_factory_litellm_without_token_provider(self):
        """Default behavior: no token_provider passed."""
        client = create_embeddings_client(
            provider="litellm",
            model_name="openai/text-embedding-3-small",
            embedding_dimension=1536,
        )
        assert isinstance(client, LiteLLMClient)
        assert client.token_provider is None
