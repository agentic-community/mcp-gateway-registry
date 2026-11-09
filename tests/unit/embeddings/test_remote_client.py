"""Unit tests for RemoteEmbeddingsClient (issue #1826).

Covers what the client decides itself: which adapter a model name selects, which
credential is used, the environment-variable fallbacks LiteLLM honored, and the
dimension check. Adapter request shapes are covered in test_adapters.py.
"""

import logging
from unittest.mock import (
    MagicMock,
    patch,
)

import httpx
import numpy as np
import pytest

from registry.embeddings.client import (
    LiteLLMClient,
    RemoteEmbeddingsClient,
)
from registry.utils.embeddings_provider import (
    PROVIDER_AZURE,
    PROVIDER_BEDROCK_COHERE,
    PROVIDER_BEDROCK_TITAN,
    PROVIDER_OPENAI,
)

API_BASE: str = "https://embeddings.example.com/v1"
PROVIDER_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Keep provider credentials from the developer's shell out of these tests."""
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _envelope(
    vectors: list[list[float]],
) -> httpx.Response:
    """Return an OpenAI envelope response for the given vectors."""
    request = httpx.Request("POST", API_BASE + "/embeddings")
    data = [{"embedding": v, "index": i, "object": "embedding"} for i, v in enumerate(vectors)]
    return httpx.Response(200, json={"data": data}, request=request)


@pytest.mark.unit
class TestModelSelection:
    """The model-name prefix picks the adapter."""

    @pytest.mark.parametrize(
        ("model_name", "provider", "model_id"),
        [
            ("openai/text-embedding-3-small", PROVIDER_OPENAI, "text-embedding-3-small"),
            ("azure/my-deployment", PROVIDER_AZURE, "my-deployment"),
            (
                "bedrock/amazon.titan-embed-text-v2:0",
                PROVIDER_BEDROCK_TITAN,
                "amazon.titan-embed-text-v2:0",
            ),
            ("bedrock/cohere.embed-english-v3", PROVIDER_BEDROCK_COHERE, "cohere.embed-english-v3"),
        ],
    )
    def test_prefix_selects_provider(self, model_name, provider, model_id):
        client = RemoteEmbeddingsClient(model_name=model_name)
        assert client.remote_model.provider == provider
        assert client.remote_model.model_id == model_id

    def test_unsupported_prefix_rejected_at_construction(self):
        with pytest.raises(ValueError, match="removed with LiteLLM"):
            RemoteEmbeddingsClient(model_name="cohere/embed-english-v3.0")

    def test_raw_array_accepts_any_prefix(self):
        # Before #1826 raw_array posted to EMBEDDINGS_API_BASE whatever the prefix.
        client = RemoteEmbeddingsClient(
            model_name="my-proxy/model-x", api_base=API_BASE, response_format="raw_array"
        )
        assert client.remote_model.provider == PROVIDER_OPENAI
        assert client.remote_model.model_id == "model-x"

    def test_litellm_client_name_is_an_alias(self):
        assert LiteLLMClient is RemoteEmbeddingsClient


@pytest.mark.unit
class TestCredentials:
    """Token provider beats static key beats the provider's environment variable."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_token_provider_wins(self, mock_post):
        mock_post.return_value = _envelope([[0.1]])
        provider = MagicMock(return_value="fresh-token")
        client = RemoteEmbeddingsClient(
            model_name="openai/m", api_key="static", api_base=API_BASE, token_provider=provider
        )
        client.encode(["a"])
        client.encode(["b"])
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer fresh-token"
        assert provider.call_count == 2

    @patch("registry.embeddings.adapters.httpx.post")
    def test_static_key_used_without_provider(self, mock_post):
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(model_name="openai/m", api_key="static", api_base=API_BASE).encode(
            ["a"]
        )
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer static"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_openai_api_key_env_fallback(self, mock_post, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(model_name="openai/m", api_base=API_BASE).encode(["a"])
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-from-env"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_no_credential_sends_no_auth_header(self, mock_post):
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(model_name="openai/m", api_base=API_BASE).encode(["a"])
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]

    def test_static_key_not_written_to_environment(self):
        # LiteLLM needed OPENAI_API_KEY set; the adapters take the key directly.
        import os

        RemoteEmbeddingsClient(model_name="openai/m", api_key="sk-static")
        assert os.environ.get("OPENAI_API_KEY") is None


@pytest.mark.unit
class TestOpenAIBase:
    """API base resolution for OpenAI-compatible endpoints."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_openai_base_url_env_fallback(self, mock_post, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(model_name="openai/m", api_key="k").encode(["a"])
        assert mock_post.call_args.args[0] == "https://proxy.example.com/v1/embeddings"

    def test_raw_array_requires_api_base(self):
        client = RemoteEmbeddingsClient(model_name="openai/m", response_format="raw_array")
        with pytest.raises(RuntimeError, match="requires EMBEDDINGS_API_BASE"):
            client.encode(["a"])


@pytest.mark.unit
class TestAzure:
    """Azure needs a base URL and honors the AZURE_* variables LiteLLM read."""

    def test_missing_api_base_raises(self):
        client = RemoteEmbeddingsClient(model_name="azure/d", api_key="k")
        with pytest.raises(RuntimeError, match="EMBEDDINGS_API_BASE"):
            client.encode(["a"])

    @patch("registry.embeddings.adapters.httpx.post")
    def test_env_fallbacks(self, mock_post, monkeypatch):
        monkeypatch.setenv("AZURE_API_BASE", "https://res.openai.azure.com")
        monkeypatch.setenv("AZURE_API_KEY", "az-env")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(model_name="azure/d").encode(["a"])
        assert mock_post.call_args.args[0].endswith(
            "/openai/deployments/d/embeddings?api-version=2024-10-21"
        )
        assert mock_post.call_args.kwargs["headers"]["api-key"] == "az-env"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_default_api_version(self, mock_post):
        mock_post.return_value = _envelope([[0.1]])
        RemoteEmbeddingsClient(
            model_name="azure/d", api_key="k", api_base="https://res.openai.azure.com"
        ).encode(["a"])
        assert mock_post.call_args.args[0].endswith("api-version=2025-02-01-preview")


@pytest.mark.unit
class TestBedrockDispatch:
    """Bedrock models go to the matching adapter with the client's region and limits."""

    @patch("registry.embeddings.client.embed_bedrock_titan")
    def test_titan_adapter(self, mock_titan):
        mock_titan.return_value = [[0.1, 0.2]]
        client = RemoteEmbeddingsClient(
            model_name="bedrock/amazon.titan-embed-text-v1",
            aws_region="eu-west-1",
            request_timeout_seconds=45,
            max_retries=4,
        )
        client.encode(["a"])
        assert mock_titan.call_args.kwargs == {
            "texts": ["a"],
            "model_id": "amazon.titan-embed-text-v1",
            "region": "eu-west-1",
            "timeout_seconds": 45,
            "max_retries": 4,
        }

    @patch("registry.embeddings.client.embed_bedrock_cohere")
    def test_cohere_adapter(self, mock_cohere):
        mock_cohere.return_value = [[0.1]]
        RemoteEmbeddingsClient(model_name="bedrock/cohere.embed-english-v3").encode(["a"])
        assert mock_cohere.called


@pytest.mark.unit
class TestEncodeAndDimension:
    """encode returns a float32 array, wraps errors, and validates the dimension once."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_returns_float32_array(self, mock_post):
        mock_post.return_value = _envelope([[0.1, 0.2], [0.3, 0.4]])
        result = RemoteEmbeddingsClient(model_name="openai/m", api_base=API_BASE).encode(["a", "b"])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)

    @patch("registry.embeddings.adapters.httpx.post")
    def test_errors_are_wrapped(self, mock_post):
        request = httpx.Request("POST", API_BASE)
        mock_post.return_value = httpx.Response(
            401, json={"message": "Unauthorized"}, request=request
        )
        client = RemoteEmbeddingsClient(model_name="openai/m", api_base=API_BASE, max_retries=0)
        with pytest.raises(
            RuntimeError, match="Failed to generate embeddings via openai: .*HTTP 401"
        ):
            client.encode(["a"])

    @patch("registry.embeddings.adapters.httpx.post")
    def test_dimension_mismatch_warns(self, mock_post, caplog):
        mock_post.return_value = _envelope([[0.1, 0.2, 0.3]])
        client = RemoteEmbeddingsClient(
            model_name="openai/m", api_base=API_BASE, embedding_dimension=2
        )
        with caplog.at_level(logging.WARNING):
            client.encode(["a"])
        assert "dimension mismatch" in caplog.text
        assert client.get_embedding_dimension() == 3

    def test_configured_dimension_used_before_any_call(self):
        client = RemoteEmbeddingsClient(model_name="openai/m", embedding_dimension=1536)
        assert client.get_embedding_dimension() == 1536

    @patch("registry.embeddings.adapters.httpx.post")
    def test_dimension_probe_when_unknown(self, mock_post):
        mock_post.return_value = _envelope([[0.1, 0.2, 0.3, 0.4]])
        client = RemoteEmbeddingsClient(model_name="openai/m", api_base=API_BASE)
        assert client.get_embedding_dimension() == 4
