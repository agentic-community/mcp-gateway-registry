"""Unit tests for operator-configured extra headers on embeddings requests (issue #1819).

Since #1826 every OpenAI-compatible and Azure request goes through one httpx code
path in adapters.py, so headers are applied in one place. These tests drive the
client and assert on the headers handed to httpx.post.
"""

import logging
from unittest.mock import (
    MagicMock,
    patch,
)

import httpx
import pytest

from registry.embeddings.client import (
    RemoteEmbeddingsClient,
    SentenceTransformersClient,
    create_embeddings_client,
)

API_BASE: str = "https://embeddings.example.com/v1"
TENANT_HEADERS: dict[str, str] = {"X-Tenant-Id": "research"}


def _envelope_response() -> httpx.Response:
    """Return an OpenAI envelope with one 2-dim vector."""
    request = httpx.Request("POST", API_BASE + "/embeddings")
    return httpx.Response(
        200,
        json={"data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}]},
        request=request,
    )


def _raw_array_response() -> httpx.Response:
    """Return a bare array with one 2-dim vector."""
    request = httpx.Request("POST", API_BASE + "/embeddings")
    return httpx.Response(200, json=[[0.1, 0.2]], request=request)


def _client(
    extra_headers: dict[str, str] | None,
    response_format: str = "openai",
    model_name: str = "openai/mock-model",
) -> RemoteEmbeddingsClient:
    """Build a client with a dynamic token so no env var is read or written."""
    return RemoteEmbeddingsClient(
        model_name=model_name,
        api_base=API_BASE,
        embedding_dimension=2,
        token_provider=MagicMock(return_value="dynamic-token"),
        response_format=response_format,
        extra_headers=extra_headers,
    )


def _sent_headers(
    mock_post: MagicMock,
) -> dict[str, str]:
    """Return the headers of the last httpx.post call."""
    return mock_post.call_args.kwargs["headers"]


@pytest.mark.unit
class TestEnvelopePath:
    """Headers on the default OpenAI-envelope path."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_sends_extra_headers(self, mock_post):
        mock_post.return_value = _envelope_response()
        _client(TENANT_HEADERS).encode(["hi"])
        assert _sent_headers(mock_post)["X-Tenant-Id"] == "research"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_keeps_managed_headers(self, mock_post):
        mock_post.return_value = _envelope_response()
        _client(TENANT_HEADERS).encode(["hi"])
        headers = _sent_headers(mock_post)
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["Authorization"] == "Bearer dynamic-token"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_unset_sends_only_managed_headers(self, mock_post):
        mock_post.return_value = _envelope_response()
        _client(None).encode(["hi"])
        assert set(_sent_headers(mock_post)) == {"Content-Type", "Accept", "Authorization"}

    @patch("registry.embeddings.adapters.httpx.post")
    def test_managed_headers_win_in_any_casing(self, mock_post):
        # Bypasses Settings validation on purpose: proves the backstop.
        mock_post.return_value = _envelope_response()
        _client({"content-type": "text/plain", "AUTHORIZATION": "Bearer evil"}).encode(["hi"])
        names = [name.lower() for name in _sent_headers(mock_post)]
        assert names.count("content-type") == 1
        assert names.count("authorization") == 1
        assert _sent_headers(mock_post)["Authorization"] == "Bearer dynamic-token"

    @patch("registry.embeddings.adapters.httpx.post")
    def test_constructor_copies_the_map(self, mock_post):
        mock_post.return_value = _envelope_response()
        caller_map = {"X-Tenant-Id": "research"}
        client = _client(caller_map)
        caller_map["X-Tenant-Id"] = "changed"
        caller_map["X-Added"] = "later"
        client.encode(["hi"])
        headers = _sent_headers(mock_post)
        assert headers["X-Tenant-Id"] == "research"
        assert "X-Added" not in headers


@pytest.mark.unit
class TestRawArrayPath:
    """Headers on the raw_array path, which shares the same request code."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_sends_extra_headers(self, mock_post):
        mock_post.return_value = _raw_array_response()
        _client(TENANT_HEADERS, response_format="raw_array").encode(["hi"])
        assert _sent_headers(mock_post)["X-Tenant-Id"] == "research"


@pytest.mark.unit
class TestAzurePath:
    """Headers on Azure OpenAI requests, beside the api-key credential."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_sends_extra_headers_with_api_key(self, mock_post):
        mock_post.return_value = _envelope_response()
        client = RemoteEmbeddingsClient(
            model_name="azure/my-deployment",
            api_key="az-key",
            api_base="https://res.openai.azure.com",
            extra_headers=TENANT_HEADERS,
        )
        client.encode(["hi"])
        headers = _sent_headers(mock_post)
        assert headers["X-Tenant-Id"] == "research"
        assert headers["api-key"] == "az-key"


@pytest.mark.unit
class TestFactory:
    """create_embeddings_client forwards the map to the remote client only."""

    def test_forwards_extra_headers_to_remote_client(self):
        client = create_embeddings_client(
            provider="litellm",
            model_name="openai/mock-model",
            api_base=API_BASE,
            extra_headers=TENANT_HEADERS,
        )
        assert isinstance(client, RemoteEmbeddingsClient)
        assert client.extra_headers == TENANT_HEADERS

    def test_warns_for_sentence_transformers(self, caplog):
        with caplog.at_level(logging.WARNING):
            client = create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                extra_headers=TENANT_HEADERS,
            )
        assert isinstance(client, SentenceTransformersClient)
        assert "EMBEDDINGS_EXTRA_HEADER_<N>" in caplog.text

    def test_silent_when_unset_for_sentence_transformers(self, caplog):
        with caplog.at_level(logging.WARNING):
            create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
            )
        assert "EMBEDDINGS_EXTRA_HEADER" not in caplog.text


@pytest.mark.unit
class TestLogHygiene:
    """Header values never reach logs; INFO carries only the count."""

    @patch("registry.embeddings.adapters.httpx.post")
    def test_header_values_never_logged(self, mock_post, caplog):
        mock_post.return_value = _envelope_response()
        caplog.set_level(logging.DEBUG)
        _client({"X-Tenant-Id": "very-private-value"}).encode(["hi"])
        for record in caplog.records:
            assert "very-private-value" not in record.getMessage()

    def test_info_log_reports_count_only(self, caplog):
        caplog.set_level(logging.INFO)
        _client({"X-Tenant-Id": "research", "X-Route": "v2"})
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        count_lines = [m for m in info_messages if "extra header(s)" in m]
        assert count_lines == ["Embeddings client configured with 2 extra header(s)"]
        assert not any("X-Tenant-Id" in m for m in info_messages)
