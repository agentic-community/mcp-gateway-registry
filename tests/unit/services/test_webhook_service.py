"""Unit tests for the registration webhook notification service."""

import hashlib
import hmac
import json
import logging
from unittest.mock import (
    AsyncMock,
    patch,
)

import httpx
import pytest

from registry.services.webhook_service import (
    SIGNATURE_HEADER,
    _build_auth_headers,
    _sign_body,
    send_registration_webhook,
)

SAMPLE_CARD = {
    "name": "test-server",
    "path": "test/server",
    "description": "A test server",
}


def _sent_payload(mock_client):
    """Parse the JSON body sent via httpx content= from a mocked client."""
    call = mock_client.post.call_args
    body = call.kwargs["content"]
    return json.loads(body)


class TestBuildAuthHeaders:
    """Tests for _build_auth_headers."""

    def test_authorization_header_prepends_bearer(self):
        """Bearer prefix is added when header is Authorization."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_auth_token = "my-secret-token"
            mock_settings.registration_webhook_auth_header = "Authorization"

            headers = _build_auth_headers()

            assert headers == {"Authorization": "Bearer my-secret-token"}

    def test_custom_header_sends_token_as_is(self):
        """Custom header names send the token without Bearer prefix."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_auth_token = "my-api-key"
            mock_settings.registration_webhook_auth_header = "X-API-Key"

            headers = _build_auth_headers()

            assert headers == {"X-API-Key": "my-api-key"}

    def test_no_token_returns_empty_dict(self):
        """No auth headers when token is not configured."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"

            headers = _build_auth_headers()

            assert headers == {}

    def test_authorization_header_case_insensitive(self):
        """Bearer prefix added regardless of Authorization casing."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_auth_token = "tok"
            mock_settings.registration_webhook_auth_header = "AUTHORIZATION"

            headers = _build_auth_headers()

            assert headers == {"AUTHORIZATION": "Bearer tok"}


class TestSendRegistrationWebhook:
    """Tests for send_registration_webhook."""

    @pytest.mark.asyncio
    async def test_registration_event_payload(self):
        """Webhook is called with correct payload for a registration event."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch("registry.services.webhook_service.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="registration",
                registration_type="server",
                card_data=SAMPLE_CARD,
                performed_by="alice",
            )

            mock_client.post.assert_called_once()
            payload = _sent_payload(mock_client)

            assert payload["event_type"] == "registration"
            assert payload["registration_type"] == "server"
            assert payload["performed_by"] == "alice"
            assert payload["card"] == SAMPLE_CARD
            assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_deletion_event_payload(self):
        """Webhook is called with correct payload for a deletion event."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch("registry.services.webhook_service.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="deletion",
                registration_type="agent",
                card_data=SAMPLE_CARD,
                performed_by="bob",
            )

            payload = _sent_payload(mock_client)

            assert payload["event_type"] == "deletion"
            assert payload["registration_type"] == "agent"
            assert payload["performed_by"] == "bob"

    @pytest.mark.asyncio
    async def test_failure_does_not_propagate(self):
        """Webhook HTTP errors are logged but not raised."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch("registry.services.webhook_service.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="registration",
                registration_type="server",
                card_data=SAMPLE_CARD,
            )

    @pytest.mark.asyncio
    async def test_timeout_does_not_propagate(self):
        """Webhook timeout is logged but not raised."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch("registry.services.webhook_service.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 5
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="registration",
                registration_type="skill",
                card_data=SAMPLE_CARD,
            )

    @pytest.mark.asyncio
    async def test_no_url_configured_skips_webhook(self):
        """Webhook is not called when URL is not configured."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_url = None

            with patch("registry.services.webhook_service.httpx.AsyncClient") as mock_async:
                await send_registration_webhook(
                    event_type="registration",
                    registration_type="server",
                    card_data=SAMPLE_CARD,
                )

                mock_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_url_skips_webhook(self):
        """Webhook is not called when URL is empty string."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_url = ""

            with patch("registry.services.webhook_service.httpx.AsyncClient") as mock_async:
                await send_registration_webhook(
                    event_type="registration",
                    registration_type="server",
                    card_data=SAMPLE_CARD,
                )

                mock_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_url_logs_warning(self, caplog):
        """A WARNING is logged when webhook URL uses HTTP instead of HTTPS."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch("registry.services.webhook_service.httpx.AsyncClient", return_value=mock_client),
            caplog.at_level(logging.WARNING, logger="registry.services.webhook_service"),
        ):
            mock_settings.registration_webhook_url = "http://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="registration",
                registration_type="server",
                card_data=SAMPLE_CARD,
            )

            assert any("HTTP (not HTTPS)" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_invalid_url_scheme_rejected(self, caplog):
        """URLs with non-http(s) schemes are rejected and logged as error."""
        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            caplog.at_level(logging.ERROR, logger="registry.services.webhook_service"),
        ):
            mock_settings.registration_webhook_url = "ftp://example.com/webhook"

            with patch("registry.services.webhook_service.httpx.AsyncClient") as mock_async:
                await send_registration_webhook(
                    event_type="registration",
                    registration_type="server",
                    card_data=SAMPLE_CARD,
                )

                mock_async.assert_not_called()
                assert any(
                    "Invalid webhook URL scheme" in record.message for record in caplog.records
                )

    @pytest.mark.asyncio
    async def test_local_runtime_env_redacted_in_payload(self):
        """local_runtime.env values must be masked before being
        sent to the external webhook endpoint, mirroring the registration gate
        sanitizer."""
        from unittest.mock import AsyncMock

        local_card = {
            "server_name": "local-server",
            "deployment": "local",
            "local_runtime": {
                "type": "npx",
                "package": "@acme/mcp",
                "env": {"LOG_LEVEL": "info", "API_KEY": "${API_KEY}"},
                "args": ["--api-key", "sk-realsecret"],
                "required_env": ["API_KEY"],
            },
            "auth_credential": "should-be-stripped",
        }

        captured_payload = {}

        async def _capture_post(url, content, headers):
            import json as _json

            captured_payload.update(_json.loads(content))
            response = MagicMock()
            response.status_code = 200
            return response

        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_url = "https://hooks.example.com/recv"
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_auth_token = ""
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value.post = _capture_post
            mock_client.__aexit__.return_value = None

            with patch(
                "registry.services.webhook_service.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await send_registration_webhook(
                    event_type="registration",
                    registration_type="server",
                    card_data=local_card,
                )

        sent_card = captured_payload["card"]
        # Top-level credential fields stripped (existing behavior).
        assert "auth_credential" not in sent_card
        # local_runtime.env values masked, keys preserved.
        rt = sent_card["local_runtime"]
        assert set(rt["env"].keys()) == {"LOG_LEVEL", "API_KEY"}
        assert all(v == "<redacted>" for v in rt["env"].values())
        # args fully masked.
        assert rt["args"] == ["<redacted>", "<redacted>"]
        # Non-sensitive fields pass through.
        assert rt["package"] == "@acme/mcp"
        assert rt["required_env"] == ["API_KEY"]


class TestSignBody:
    """Tests for _sign_body (HMAC-SHA256 webhook signing, Issue #1330)."""

    def test_returns_none_when_no_secret(self):
        """No signature is produced when the signing secret is unset."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_signing_secret = None
            assert _sign_body(b'{"a":1}') is None

    def test_matches_known_hmac_vector(self):
        """Signature equals the HMAC-SHA256 of the body for the configured secret."""
        with patch("registry.services.webhook_service.settings") as mock_settings:
            mock_settings.registration_webhook_signing_secret = "sekret"
            body = b'{"event_type":"registration"}'
            expected = "sha256=" + hmac.new(b"sekret", body, hashlib.sha256).hexdigest()
            assert _sign_body(body) == expected


class TestWebhookSigningAndExtraFields:
    """Signature header + extra_fields behavior on send_registration_webhook."""

    def _mock_client(self):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    @pytest.mark.asyncio
    async def test_signature_header_present_and_verifiable(self):
        """When a secret is set, X-Registry-Signature matches the sent bytes."""
        mock_client = self._mock_client()
        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch(
                "registry.services.webhook_service.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = "topsecret"

            await send_registration_webhook(
                event_type="registration",
                registration_type="server",
                card_data=SAMPLE_CARD,
            )

            call = mock_client.post.call_args
            body = call.kwargs["content"]
            headers = call.kwargs["headers"]
            expected = "sha256=" + hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
            assert headers[SIGNATURE_HEADER] == expected
            assert hmac.compare_digest(headers[SIGNATURE_HEADER], expected)

    @pytest.mark.asyncio
    async def test_no_signature_header_when_secret_unset(self):
        """No X-Registry-Signature header when the signing secret is unset."""
        mock_client = self._mock_client()
        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch(
                "registry.services.webhook_service.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            await send_registration_webhook(
                event_type="registration",
                registration_type="server",
                card_data=SAMPLE_CARD,
            )

            headers = mock_client.post.call_args.kwargs["headers"]
            assert SIGNATURE_HEADER not in headers

    @pytest.mark.asyncio
    async def test_extra_fields_merged_into_envelope(self):
        """extra_fields are merged at the top level (e.g. scan_complete payload)."""
        mock_client = self._mock_client()
        with (
            patch("registry.services.webhook_service.settings") as mock_settings,
            patch(
                "registry.services.webhook_service.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            mock_settings.registration_webhook_url = "https://example.com/webhook"
            mock_settings.registration_webhook_auth_token = None
            mock_settings.registration_webhook_auth_header = "Authorization"
            mock_settings.registration_webhook_timeout_seconds = 10
            mock_settings.registration_webhook_signing_secret = None

            scan = {"is_safe": True, "severity_counts": {"critical": 0, "high": 0}}
            await send_registration_webhook(
                event_type="scan_complete",
                registration_type="server",
                card_data=SAMPLE_CARD,
                extra_fields={"scan": scan},
            )

            payload = _sent_payload(mock_client)
            assert payload["event_type"] == "scan_complete"
            assert payload["scan"] == scan
