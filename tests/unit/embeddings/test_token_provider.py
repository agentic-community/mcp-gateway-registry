"""Unit tests for EmbeddingsTokenProvider (OAuth2 client-credentials)."""

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from registry.embeddings.token_provider import (
    DEFAULT_EXPIRES_IN_SECONDS,
    EmbeddingsTokenProvider,
    _require_idp_settings,
)


class TestRequireIdpSettings:
    """Tests for the _require_idp_settings validation helper."""

    def test_raises_when_all_missing(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_IDP_TOKEN_ENDPOINT"):
            _require_idp_settings(None, None, None)

    def test_raises_when_partial_missing(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_IDP_CLIENT_SECRET"):
            _require_idp_settings("https://idp.example.com/token", "client-id", None)

    def test_raises_when_not_https(self):
        with pytest.raises(ValueError, match="https://"):
            _require_idp_settings("http://idp.example.com/token", "client-id", "secret")

    def test_passes_with_valid_settings(self):
        _require_idp_settings("https://idp.example.com/token", "client-id", "secret")

    def test_allow_insecure_permits_loopback_http(self):
        # http:// is allowed for local dev only when the host is loopback.
        _require_idp_settings(
            "http://localhost:8080/token", "client-id", "secret", allow_insecure=True
        )
        _require_idp_settings(
            "http://127.0.0.1:8080/token", "client-id", "secret", allow_insecure=True
        )

    def test_allow_insecure_rejects_remote_http(self):
        # allow_insecure must NOT permit a remote http:// endpoint (cleartext secret).
        with pytest.raises(ValueError, match="loopback"):
            _require_idp_settings(
                "http://idp.example.com/token", "client-id", "secret", allow_insecure=True
            )


class TestEmbeddingsTokenProviderInit:
    """Tests for provider initialization validation."""

    def test_rejects_http_endpoint(self):
        with pytest.raises(ValueError, match="https://"):
            EmbeddingsTokenProvider(
                token_endpoint="http://insecure.example.com/token",
                client_id="cid",
                client_secret="csecret",
            )

    def test_accepts_https_endpoint(self):
        provider = EmbeddingsTokenProvider(
            token_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csecret",
        )
        assert provider._token_endpoint == "https://idp.example.com/token"
        provider.close()


class TestEmbeddingsTokenProviderGetToken:
    """Tests for token acquisition, caching, and refresh."""

    def _make_provider(self, **kwargs):
        defaults = {
            "token_endpoint": "https://idp.example.com/token",
            "client_id": "test-client",
            "client_secret": "test-secret",
            "scope": "api://test/.default",
            "timeout_seconds": 5,
        }
        defaults.update(kwargs)
        return EmbeddingsTokenProvider(**defaults)

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_returns_token_on_success(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "fresh-token-123",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        token = provider.get_token()

        assert token == "fresh-token-123"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "client_credentials"
        assert call_kwargs[1]["data"]["client_id"] == "test-client"
        assert call_kwargs[1]["data"]["client_secret"] == "test-secret"
        assert call_kwargs[1]["data"]["scope"] == "api://test/.default"

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_raises_on_malformed_response(self, mock_client_class):
        # A 200 with a non-numeric expires_in (ValueError from int()) must be
        # wrapped in the domain RuntimeError, not surfaced as a raw parse error.
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "x", "expires_in": "not-a-number"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="Malformed IdP token response"):
            provider.get_token()

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_caches_until_near_expiry(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        token1 = provider.get_token()
        token2 = provider.get_token()

        assert token1 == "cached-token"
        assert token2 == "cached-token"
        assert mock_client.post.call_count == 1

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_refreshes_after_expiry(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "first-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        provider.get_token()

        provider._token_expiry = datetime.now(UTC) - timedelta(seconds=10)

        mock_response.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 3600,
        }
        token = provider.get_token()

        assert token == "refreshed-token"
        assert mock_client.post.call_count == 2

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_raises_on_http_error(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="status 401"):
            provider.get_token()

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_raises_on_network_error(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="Network error"):
            provider.get_token()

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_raises_on_missing_access_token_in_response(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {"token_type": "Bearer", "expires_in": 3600}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="missing 'access_token'"):
            provider.get_token()

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_omits_scope_when_none(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "no-scope-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider(scope=None)
        provider.get_token()

        call_data = mock_client.post.call_args[1]["data"]
        assert "scope" not in call_data

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_secret_never_logged(self, mock_client_class, caplog):
        """The client secret and access token must never appear in logs."""
        secret = "super-secret-value-xyz"
        token_value = "access-token-abc-123"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": token_value,
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        with caplog.at_level(logging.DEBUG, logger="registry.embeddings.token_provider"):
            provider = self._make_provider(client_secret=secret)
            provider.get_token()

        full_log = caplog.text
        assert secret not in full_log
        assert token_value not in full_log

    @patch("registry.embeddings.token_provider.httpx.Client")
    def test_defaults_expires_in_when_missing(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "tk"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = self._make_provider()
        provider.get_token()

        expected_expiry = datetime.now(UTC) + timedelta(seconds=DEFAULT_EXPIRES_IN_SECONDS)
        assert abs((provider._token_expiry - expected_expiry).total_seconds()) < 5
