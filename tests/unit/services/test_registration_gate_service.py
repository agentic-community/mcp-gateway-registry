"""Unit tests for the registration gate (admission control) service."""

import logging
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx

from registry.schemas.registration_gate_models import (
    RegistrationGateAuthType,
    RegistrationGateRequest,
    RegistrationGateResponse,
    RegistrationGateResult,
)
from registry.services.registration_gate_service import (
    GATE_ERROR_MAX_LENGTH,
    _acquire_oauth2_token,
    _build_auth_headers,
    _extract_request_headers,
    _is_gate_configured,
    _sanitize_payload,
    _truncate_error,
    check_registration_gate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

SETTINGS_PATH = "registry.services.registration_gate_service.settings"
HTTPX_CLIENT_PATH = "registry.services.registration_gate_service.httpx.AsyncClient"
ASYNCIO_SLEEP_PATH = "registry.services.registration_gate_service.asyncio.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_headers(
    headers: dict[str, str],
) -> list[tuple[bytes, bytes]]:
    """Convert a plain dict to ASGI raw header tuples.

    Args:
        headers: Dict of header name to value.

    Returns:
        List of (name_bytes, value_bytes) tuples.
    """
    return [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in headers.items()
    ]


def _make_mock_settings(
    gate_enabled: bool = True,
    gate_url: str = "https://gate.example.com/check",
    auth_type: str = "none",
    auth_credential: str = "",
    auth_header_name: str = "X-Api-Key",
    timeout_seconds: int = 5,
    max_retries: int = 2,
    oauth2_token_url: str = "",
    oauth2_client_id: str = "",
    oauth2_client_secret: str = "",
    oauth2_scope: str = "",
) -> MagicMock:
    """Build a MagicMock that mimics the settings object.

    Args:
        gate_enabled: Whether the gate is enabled.
        gate_url: URL of the gate endpoint.
        auth_type: Auth type string.
        auth_credential: Credential string.
        auth_header_name: Header name for api_key auth.
        timeout_seconds: Per-request timeout.
        max_retries: Max retries on transient failures.
        oauth2_token_url: OAuth2 token endpoint URL.
        oauth2_client_id: OAuth2 client ID.
        oauth2_client_secret: OAuth2 client secret.
        oauth2_scope: OAuth2 scope parameter.

    Returns:
        MagicMock configured with the given values.
    """
    mock = MagicMock()
    mock.registration_gate_enabled = gate_enabled
    mock.registration_gate_url = gate_url
    mock.registration_gate_auth_type = auth_type
    mock.registration_gate_auth_credential = auth_credential
    mock.registration_gate_auth_header_name = auth_header_name
    mock.registration_gate_timeout_seconds = timeout_seconds
    mock.registration_gate_max_retries = max_retries
    mock.registration_gate_oauth2_token_url = oauth2_token_url
    mock.registration_gate_oauth2_client_id = oauth2_client_id
    mock.registration_gate_oauth2_client_secret = oauth2_client_secret
    mock.registration_gate_oauth2_scope = oauth2_scope
    return mock


def _make_mock_http_client(
    response: AsyncMock | None = None,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Build an AsyncMock that acts as httpx.AsyncClient context manager.

    Args:
        response: Mock response to return from post().
        side_effect: Exception to raise on post().

    Returns:
        AsyncMock configured as an async context manager.
    """
    mock_client = AsyncMock()
    if side_effect:
        mock_client.post = AsyncMock(side_effect=side_effect)
    elif response is not None:
        mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Build a MagicMock that mimics an httpx.Response.

    Args:
        status_code: HTTP status code.
        json_data: Dict returned by response.json().
        text: Text returned by response.text.

    Returns:
        MagicMock configured as an HTTP response.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    if json_data is not None:
        mock_response.json = MagicMock(return_value=json_data)
    else:
        mock_response.json = MagicMock(side_effect=ValueError("No JSON"))
    return mock_response


# ===========================================================================
# Model tests
# ===========================================================================


class TestRegistrationGateRequest:
    """Tests for the RegistrationGateRequest Pydantic model."""

    def test_valid_construction(self):
        """Model can be constructed with all required fields."""
        req = RegistrationGateRequest(
            asset_type="server",
            operation="register",
            source_api="/api/v1/servers",
            registration_payload={"name": "my-server"},
            request_headers={"host": "localhost"},
        )

        assert req.asset_type == "server"
        assert req.operation == "register"
        assert req.source_api == "/api/v1/servers"
        assert req.registration_payload == {"name": "my-server"}
        assert req.request_headers == {"host": "localhost"}

    def test_default_request_headers(self):
        """request_headers defaults to empty dict when not provided."""
        req = RegistrationGateRequest(
            asset_type="agent",
            operation="update",
            source_api="/api/v1/agents",
            registration_payload={},
        )

        assert req.request_headers == {}

    def test_serialization_round_trip(self):
        """Model serializes to JSON and deserializes back correctly."""
        req = RegistrationGateRequest(
            asset_type="skill",
            operation="register",
            source_api="/api/v1/skills",
            registration_payload={"name": "my-skill", "version": "1.0"},
            request_headers={"content-type": "application/json"},
        )

        json_str = req.model_dump_json()
        restored = RegistrationGateRequest.model_validate_json(json_str)

        assert restored.asset_type == req.asset_type
        assert restored.registration_payload == req.registration_payload


class TestRegistrationGateResponse:
    """Tests for the RegistrationGateResponse Pydantic model."""

    def test_allowed_response(self):
        """Response with status='allowed' and no error."""
        resp = RegistrationGateResponse(status="allowed")

        assert resp.status == "allowed"
        assert resp.error is None

    def test_denied_response_with_error(self):
        """Response with status='denied' and an error message."""
        resp = RegistrationGateResponse(
            status="denied",
            error="Server name is reserved",
        )

        assert resp.status == "denied"
        assert resp.error == "Server name is reserved"


class TestRegistrationGateResult:
    """Tests for the RegistrationGateResult Pydantic model."""

    def test_allowed_result(self):
        """Result with allowed=True and no error."""
        result = RegistrationGateResult(
            allowed=True,
            error_message=None,
            gate_status_code=200,
            attempts=1,
        )

        assert result.allowed is True
        assert result.error_message is None
        assert result.gate_status_code == 200
        assert result.attempts == 1

    def test_denied_result(self):
        """Result with allowed=False and error message."""
        result = RegistrationGateResult(
            allowed=False,
            error_message="Policy violation: name is blacklisted",
            gate_status_code=403,
            attempts=1,
        )

        assert result.allowed is False
        assert result.error_message == "Policy violation: name is blacklisted"
        assert result.gate_status_code == 403

    def test_defaults(self):
        """Default values for optional fields."""
        result = RegistrationGateResult(allowed=True)

        assert result.error_message is None
        assert result.gate_status_code is None
        assert result.attempts == 0


class TestRegistrationGateAuthType:
    """Tests for the RegistrationGateAuthType enum."""

    def test_enum_values(self):
        """Enum contains expected values."""
        assert RegistrationGateAuthType.NONE == "none"
        assert RegistrationGateAuthType.API_KEY == "api_key"
        assert RegistrationGateAuthType.BEARER == "bearer"


# ===========================================================================
# _sanitize_payload tests
# ===========================================================================


class TestSanitizePayload:
    """Tests for _sanitize_payload."""

    def test_removes_exact_sensitive_field_names(self):
        """Fields in SENSITIVE_FIELD_NAMES are removed."""
        payload = {
            "name": "my-server",
            "auth_credential": "secret123",
            "auth_credential_encrypted": "enc456",
            "auth_header_name": "X-Secret",
            "description": "A test server",
        }

        result = _sanitize_payload(payload)

        assert "auth_credential" not in result
        assert "auth_credential_encrypted" not in result
        assert "auth_header_name" not in result
        assert result["name"] == "my-server"
        assert result["description"] == "A test server"

    def test_removes_fields_matching_sensitive_substrings(self):
        """Fields containing any sensitive substring are removed."""
        payload = {
            "name": "my-server",
            "user_credential": "cred",
            "db_secret": "s3cret",
            "auth_token": "tok",
            "user_password": "pw",
            "my_api_key": "key123",
            "description": "safe",
        }

        result = _sanitize_payload(payload)

        assert "user_credential" not in result
        assert "db_secret" not in result
        assert "auth_token" not in result
        assert "user_password" not in result
        assert "my_api_key" not in result
        assert result["name"] == "my-server"
        assert result["description"] == "safe"

    def test_substring_matching_is_case_insensitive(self):
        """Sensitive substrings are matched via lowercased key."""
        payload = {
            "MyCredential": "hidden",
            "DB_SECRET": "hidden",
            "AuthToken": "hidden",
            "safe_field": "visible",
        }

        result = _sanitize_payload(payload)

        # "MyCredential" lowercased contains "credential"
        assert "MyCredential" not in result
        # "AuthToken" lowercased contains "token"
        assert "AuthToken" not in result
        assert result["safe_field"] == "visible"

    def test_preserves_all_non_sensitive_fields(self):
        """Non-sensitive fields are preserved exactly."""
        payload = {
            "name": "test",
            "description": "A server",
            "tags": ["prod", "ml"],
            "num_tools": 5,
            "proxy_pass_url": "http://localhost:8080",
        }

        result = _sanitize_payload(payload)

        assert result == payload

    def test_empty_payload(self):
        """Empty payload returns empty dict."""
        result = _sanitize_payload({})

        assert result == {}

    def test_all_sensitive_payload(self):
        """Payload with only sensitive fields returns empty dict."""
        payload = {
            "auth_credential": "secret",
            "user_token": "tok",
            "db_password": "pw",
        }

        result = _sanitize_payload(payload)

        assert result == {}


# ===========================================================================
# _build_auth_headers tests
# ===========================================================================


class TestBuildAuthHeaders:
    """Tests for _build_auth_headers (async)."""

    async def test_returns_empty_when_auth_type_none(self):
        """No headers when auth_type is 'none'."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "none"
            mock_settings.registration_gate_auth_credential = ""

            headers = await _build_auth_headers()

            assert headers == {}

    async def test_returns_bearer_header(self):
        """Bearer token header when auth_type is 'bearer'."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "bearer"
            mock_settings.registration_gate_auth_credential = "my-jwt-token"

            headers = await _build_auth_headers()

            assert headers == {"Authorization": "Bearer my-jwt-token"}

    async def test_returns_api_key_header(self):
        """Custom API key header when auth_type is 'api_key'."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "api_key"
            mock_settings.registration_gate_auth_credential = "key-abc-123"
            mock_settings.registration_gate_auth_header_name = "X-Api-Key"

            headers = await _build_auth_headers()

            assert headers == {"X-Api-Key": "key-abc-123"}

    async def test_api_key_with_custom_header_name(self):
        """API key uses the configured header name."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "api_key"
            mock_settings.registration_gate_auth_credential = "my-key"
            mock_settings.registration_gate_auth_header_name = "X-Custom-Auth"

            headers = await _build_auth_headers()

            assert headers == {"X-Custom-Auth": "my-key"}

    async def test_bearer_with_empty_credential_returns_empty(self):
        """No headers when bearer auth has empty credential."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "bearer"
            mock_settings.registration_gate_auth_credential = ""

            headers = await _build_auth_headers()

            assert headers == {}

    async def test_api_key_with_empty_credential_returns_empty(self):
        """No headers when api_key auth has empty credential."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "api_key"
            mock_settings.registration_gate_auth_credential = ""
            mock_settings.registration_gate_auth_header_name = "X-Api-Key"

            headers = await _build_auth_headers()

            assert headers == {}

    async def test_oauth2_success_returns_bearer_header(self):
        """OAuth2 auth type acquires token and returns Bearer header."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "oauth2_client_credentials"

            with patch(
                "registry.services.registration_gate_service._acquire_oauth2_token",
                new_callable=AsyncMock,
                return_value="dynamic-token-xyz",
            ):
                headers = await _build_auth_headers()

            assert headers == {"Authorization": "Bearer dynamic-token-xyz"}

    async def test_oauth2_token_failure_returns_empty(self):
        """OAuth2 auth type returns empty dict when token acquisition fails."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_auth_type = "oauth2_client_credentials"

            with patch(
                "registry.services.registration_gate_service._acquire_oauth2_token",
                new_callable=AsyncMock,
                return_value=None,
            ):
                headers = await _build_auth_headers()

            assert headers == {}


# ===========================================================================
# _extract_request_headers tests
# ===========================================================================


class TestExtractRequestHeaders:
    """Tests for _extract_request_headers."""

    def test_converts_raw_asgi_headers(self):
        """Raw byte tuples are decoded to string dict."""
        raw = _make_raw_headers({
            "host": "example.com",
            "content-type": "application/json",
        })

        result = _extract_request_headers(raw)

        assert result["host"] == "example.com"
        assert result["content-type"] == "application/json"

    def test_filters_authorization_header(self):
        """The 'authorization' header is excluded."""
        raw = _make_raw_headers({
            "authorization": "Bearer secret-token",
            "host": "example.com",
        })

        result = _extract_request_headers(raw)

        assert "authorization" not in result
        assert result["host"] == "example.com"

    def test_filters_cookie_header(self):
        """The 'cookie' header is excluded."""
        raw = _make_raw_headers({
            "cookie": "session=abc123",
            "accept": "application/json",
        })

        result = _extract_request_headers(raw)

        assert "cookie" not in result
        assert result["accept"] == "application/json"

    def test_filters_csrf_token_header(self):
        """The 'x-csrf-token' header is excluded."""
        raw = _make_raw_headers({
            "x-csrf-token": "csrf-value",
            "user-agent": "test-client",
        })

        result = _extract_request_headers(raw)

        assert "x-csrf-token" not in result
        assert result["user-agent"] == "test-client"

    def test_filters_multiple_sensitive_headers(self):
        """All sensitive headers are excluded simultaneously."""
        raw = _make_raw_headers({
            "authorization": "Bearer tok",
            "cookie": "sess=123",
            "x-csrf-token": "csrf",
            "host": "example.com",
            "x-request-id": "req-001",
        })

        result = _extract_request_headers(raw)

        assert len(result) == 2
        assert result["host"] == "example.com"
        assert result["x-request-id"] == "req-001"

    def test_empty_headers(self):
        """Empty header list returns empty dict."""
        result = _extract_request_headers([])

        assert result == {}

    def test_header_names_are_lowercased(self):
        """Header names are lowercased during extraction."""
        raw = [
            (b"Host", b"example.com"),
            (b"Content-Type", b"application/json"),
        ]

        result = _extract_request_headers(raw)

        assert "host" in result
        assert "content-type" in result


# ===========================================================================
# _is_gate_configured tests
# ===========================================================================


class TestIsGateConfigured:
    """Tests for _is_gate_configured."""

    def test_returns_false_when_disabled(self):
        """Gate is not configured when disabled."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_enabled = False

            assert _is_gate_configured() is False

    def test_returns_false_when_enabled_but_url_empty(self, caplog):
        """Gate is not configured when enabled but URL is empty."""
        with (
            patch(SETTINGS_PATH) as mock_settings,
            caplog.at_level(
                logging.WARNING,
                logger="registry.services.registration_gate_service",
            ),
        ):
            mock_settings.registration_gate_enabled = True
            mock_settings.registration_gate_url = ""

            assert _is_gate_configured() is False
            assert any(
                "no URL is configured" in record.message
                for record in caplog.records
            )

    def test_returns_true_when_enabled_and_url_set(self):
        """Gate is configured when enabled and URL is present."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_enabled = True
            mock_settings.registration_gate_url = "https://gate.example.com"

            assert _is_gate_configured() is True


# ===========================================================================
# _truncate_error tests
# ===========================================================================


class TestTruncateError:
    """Tests for _truncate_error."""

    def test_short_message_unchanged(self):
        """Messages under the max length are returned as-is."""
        msg = "Registration denied"

        assert _truncate_error(msg) == msg

    def test_exact_limit_unchanged(self):
        """Message exactly at the limit is not truncated."""
        msg = "x" * GATE_ERROR_MAX_LENGTH

        assert _truncate_error(msg) == msg
        assert len(_truncate_error(msg)) == GATE_ERROR_MAX_LENGTH

    def test_over_limit_is_truncated(self):
        """Message over the limit is truncated with ellipsis."""
        msg = "y" * (GATE_ERROR_MAX_LENGTH + 100)

        result = _truncate_error(msg)

        assert len(result) == GATE_ERROR_MAX_LENGTH + 3
        assert result.endswith("...")
        assert result[:GATE_ERROR_MAX_LENGTH] == "y" * GATE_ERROR_MAX_LENGTH

    def test_empty_message(self):
        """Empty string is returned as-is."""
        assert _truncate_error("") == ""


# ===========================================================================
# check_registration_gate tests
# ===========================================================================


class TestCheckRegistrationGate:
    """Tests for check_registration_gate (public entry point)."""

    async def test_returns_allowed_when_gate_not_configured(self):
        """Immediately returns allowed=True when gate is disabled."""
        with patch(SETTINGS_PATH) as mock_settings:
            mock_settings.registration_gate_enabled = False

            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is True
            assert result.error_message is None
            assert result.gate_status_code is None
            assert result.attempts == 0

    async def test_calls_gate_when_configured_and_returns_allowed(self):
        """Gate returns allowed on 200 response."""
        mock_response = _make_mock_response(status_code=200)
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings(
            gate_enabled=True,
            gate_url="https://gate.example.com/check",
            auth_type="none",
        )

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "my-server"},
                raw_headers=_make_raw_headers({"host": "localhost"}),
            )

            assert result.allowed is True
            assert result.gate_status_code == 200
            assert result.attempts == 1

    async def test_returns_denied_on_403_with_json_error(self):
        """Gate returns denied with error message from JSON body on 403."""
        mock_response = _make_mock_response(
            status_code=403,
            json_data={"status": "denied", "error": "Name is reserved"},
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/v1/agents",
                registration_payload={"name": "reserved-name"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.error_message == "Name is reserved"
            assert result.gate_status_code == 403
            assert result.attempts == 1

    async def test_returns_denied_on_403_with_raw_text(self):
        """Gate returns denied with raw text when JSON parsing fails on 403."""
        mock_response = _make_mock_response(
            status_code=403,
            text="Forbidden by policy",
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="update",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.error_message == "Forbidden by policy"
            assert result.gate_status_code == 403

    async def test_returns_denied_on_403_default_message_when_no_body(self):
        """Gate returns default denial message when 403 has empty body and invalid JSON."""
        mock_response = _make_mock_response(
            status_code=403,
            text="",
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.error_message == "Registration denied by policy"

    async def test_sanitizes_payload_before_sending(self):
        """Sensitive fields are removed from the payload sent to gate."""
        mock_response = _make_mock_response(status_code=200)
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={
                    "name": "my-server",
                    "auth_credential": "secret",
                    "description": "A server",
                },
                raw_headers=[],
            )

            call_kwargs = mock_client.post.call_args
            sent_content = call_kwargs.kwargs.get("content", "")
            assert "secret" not in sent_content
            assert "my-server" in sent_content

    async def test_filters_sensitive_headers_before_sending(self):
        """Sensitive request headers are excluded from gate payload."""
        mock_response = _make_mock_response(status_code=200)
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/v1/agents",
                registration_payload={"name": "agent1"},
                raw_headers=_make_raw_headers({
                    "host": "localhost",
                    "authorization": "Bearer secret-token",
                    "x-request-id": "req-001",
                }),
            )

            call_kwargs = mock_client.post.call_args
            sent_content = call_kwargs.kwargs.get("content", "")
            assert "secret-token" not in sent_content
            assert "localhost" in sent_content


# ===========================================================================
# _call_gate_endpoint tests (via check_registration_gate)
# ===========================================================================


class TestCallGateEndpoint:
    """Tests for _call_gate_endpoint retry and error handling."""

    async def test_timeout_exhausts_retries_and_returns_denied(self):
        """Timeout on all attempts results in fail-closed denial."""
        mock_client = _make_mock_http_client(
            side_effect=httpx.TimeoutException("timed out"),
        )
        mock_settings = _make_mock_settings(max_retries=1)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert "unavailable" in result.error_message
            assert "fail-closed" in result.error_message
            # 1 initial attempt + 1 retry = 2 total
            assert result.attempts == 2

    async def test_connection_error_exhausts_retries_and_returns_denied(self):
        """Connection error on all attempts results in fail-closed denial."""
        mock_client = _make_mock_http_client(
            side_effect=httpx.ConnectError("connection refused"),
        )
        mock_settings = _make_mock_settings(max_retries=1)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/v1/agents",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert "unavailable" in result.error_message
            assert result.attempts == 2

    async def test_unexpected_status_code_triggers_retry(self):
        """Unexpected status codes (e.g. 500) trigger retries."""
        mock_response_500 = _make_mock_response(status_code=500, text="Internal error")
        mock_client = _make_mock_http_client(response=mock_response_500)
        mock_settings = _make_mock_settings(max_retries=1)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="skill",
                operation="register",
                source_api="/api/v1/skills",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.attempts == 2
            assert mock_client.post.call_count == 2

    async def test_retry_succeeds_on_second_attempt(self):
        """Gate call succeeds on retry after first attempt fails."""
        mock_response_ok = _make_mock_response(status_code=200)
        mock_response_500 = _make_mock_response(status_code=500, text="error")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_500, mock_response_ok],
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_settings = _make_mock_settings(max_retries=1)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is True
            assert result.gate_status_code == 200
            assert result.attempts == 2

    async def test_no_retries_when_max_retries_is_zero(self):
        """Only one attempt when max_retries is 0."""
        mock_client = _make_mock_http_client(
            side_effect=httpx.TimeoutException("timed out"),
        )
        mock_settings = _make_mock_settings(max_retries=0)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.attempts == 1
            assert mock_client.post.call_count == 1

    async def test_backoff_sleep_called_between_retries(self):
        """Exponential backoff sleep is called between retry attempts."""
        mock_client = _make_mock_http_client(
            side_effect=httpx.TimeoutException("timed out"),
        )
        mock_settings = _make_mock_settings(max_retries=2)

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
        ):
            await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            # With max_retries=2, total attempts=3
            # Sleep is called after attempt 1 and attempt 2 (not after the last)
            assert mock_sleep.call_count == 2
            # First backoff: 0.5 * 2^0 = 0.5
            mock_sleep.assert_any_call(0.5)
            # Second backoff: 0.5 * 2^1 = 1.0
            mock_sleep.assert_any_call(1.0)

    async def test_includes_bearer_auth_in_gate_request(self):
        """Bearer auth headers are included in gate HTTP request."""
        mock_response = _make_mock_response(status_code=200)
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings(
            auth_type="bearer",
            auth_credential="jwt-token-xyz",
        )

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            call_kwargs = mock_client.post.call_args
            headers_sent = call_kwargs.kwargs.get("headers", {})
            assert headers_sent.get("Authorization") == "Bearer jwt-token-xyz"

    async def test_403_with_long_error_is_truncated(self):
        """Long error messages from 403 responses are truncated."""
        long_error = "x" * 1000
        mock_response = _make_mock_response(
            status_code=403,
            json_data={"status": "denied", "error": long_error},
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert len(result.error_message) == GATE_ERROR_MAX_LENGTH + 3
            assert result.error_message.endswith("...")

    async def test_403_json_without_error_field_uses_default(self):
        """When 403 JSON has no error field, default denial message is used."""
        mock_response = _make_mock_response(
            status_code=403,
            json_data={"status": "denied"},
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings()

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="server",
                operation="register",
                source_api="/api/v1/servers",
                registration_payload={"name": "test"},
                raw_headers=[],
            )

            assert result.allowed is False
            assert result.error_message == "Registration denied by policy"


# ===========================================================================
# _acquire_oauth2_token tests
# ===========================================================================


ACQUIRE_TOKEN_SETTINGS_PATH = SETTINGS_PATH


class TestAcquireOAuth2Token:
    """Tests for _acquire_oauth2_token."""

    def _make_oauth2_settings(
        self,
        token_url: str = "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
        client_id: str = "test-client-id",
        client_secret: str = "test-client-secret",
        scope: str = "api://app-id/.default",
        timeout: int = 5,
    ) -> MagicMock:
        """Build mock settings for OAuth2 token acquisition."""
        mock = MagicMock()
        mock.registration_gate_oauth2_token_url = token_url
        mock.registration_gate_oauth2_client_id = client_id
        mock.registration_gate_oauth2_client_secret = client_secret
        mock.registration_gate_oauth2_scope = scope
        mock.registration_gate_timeout_seconds = timeout
        return mock

    async def test_success_returns_access_token(self):
        """Happy path: token endpoint returns 200 with access_token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9",
            "token_type": "Bearer",
            "expires_in": 3600,
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token == "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9"

    async def test_includes_scope_when_configured(self):
        """Scope parameter is included in form data when set."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "access_token": "token123",
            "token_type": "Bearer",
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                ACQUIRE_TOKEN_SETTINGS_PATH,
                self._make_oauth2_settings(scope="api://my-app/.default"),
            ),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            await _acquire_oauth2_token()

        call_kwargs = mock_client.post.call_args
        sent_data = call_kwargs.kwargs.get("data", {})
        assert sent_data["scope"] == "api://my-app/.default"
        assert sent_data["grant_type"] == "client_credentials"

    async def test_excludes_scope_when_empty(self):
        """Scope parameter is omitted when config is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "access_token": "token123",
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                ACQUIRE_TOKEN_SETTINGS_PATH,
                self._make_oauth2_settings(scope=""),
            ),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            await _acquire_oauth2_token()

        call_kwargs = mock_client.post.call_args
        sent_data = call_kwargs.kwargs.get("data", {})
        assert "scope" not in sent_data

    async def test_non_200_returns_none(self):
        """Non-200 response from token endpoint returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token is None

    async def test_missing_access_token_field_returns_none(self):
        """200 response without access_token field returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "token_type": "Bearer",
            "expires_in": 3600,
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token is None

    async def test_timeout_returns_none(self):
        """Timeout from token endpoint returns None."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timed out"),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token is None

    async def test_connection_error_returns_none(self):
        """Connection error to token endpoint returns None."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused"),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token is None

    async def test_unexpected_exception_returns_none(self):
        """Unexpected exception returns None."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=Exception("unexpected error"),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(ACQUIRE_TOKEN_SETTINGS_PATH, self._make_oauth2_settings()),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
        ):
            token = await _acquire_oauth2_token()

        assert token is None


# ===========================================================================
# check_registration_gate with OAuth2 auth tests
# ===========================================================================


class TestCheckRegistrationGateOAuth2:
    """Tests for gate calls with oauth2_client_credentials auth type."""

    async def test_gate_fails_closed_on_token_acquisition_failure(self):
        """Registration is blocked when OAuth2 token cannot be acquired."""
        mock_settings = _make_mock_settings(
            auth_type="oauth2_client_credentials",
            oauth2_token_url="https://login.example.com/token",
            oauth2_client_id="client-id",
            oauth2_client_secret="client-secret",
        )

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(
                "registry.services.registration_gate_service._acquire_oauth2_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/agents/register",
                registration_payload={"name": "test-agent"},
                raw_headers=[],
            )

        assert result.allowed is False
        assert "OAuth2 token" in result.error_message
        assert result.attempts == 0

    async def test_gate_succeeds_with_oauth2_token(self):
        """Gate call succeeds with dynamically acquired OAuth2 token."""
        mock_response = _make_mock_response(status_code=200)
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings(
            auth_type="oauth2_client_credentials",
            oauth2_token_url="https://login.example.com/token",
            oauth2_client_id="client-id",
            oauth2_client_secret="client-secret",
        )

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(
                "registry.services.registration_gate_service._acquire_oauth2_token",
                new_callable=AsyncMock,
                return_value="dynamic-token-abc",
            ),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/agents/register",
                registration_payload={"name": "test-agent"},
                raw_headers=[],
            )

        assert result.allowed is True
        assert result.gate_status_code == 200

        call_kwargs = mock_client.post.call_args
        headers_sent = call_kwargs.kwargs.get("headers", {})
        assert headers_sent.get("Authorization") == "Bearer dynamic-token-abc"

    async def test_gate_denied_with_oauth2_token(self):
        """Gate denies registration even with valid OAuth2 token."""
        mock_response = _make_mock_response(
            status_code=403,
            json_data={"status": "denied", "error": "Policy violation"},
        )
        mock_client = _make_mock_http_client(response=mock_response)
        mock_settings = _make_mock_settings(
            auth_type="oauth2_client_credentials",
            oauth2_token_url="https://login.example.com/token",
            oauth2_client_id="client-id",
            oauth2_client_secret="client-secret",
        )

        with (
            patch(SETTINGS_PATH, mock_settings),
            patch(
                "registry.services.registration_gate_service._acquire_oauth2_token",
                new_callable=AsyncMock,
                return_value="valid-token",
            ),
            patch(HTTPX_CLIENT_PATH, return_value=mock_client),
            patch(ASYNCIO_SLEEP_PATH, new_callable=AsyncMock),
        ):
            result = await check_registration_gate(
                asset_type="agent",
                operation="register",
                source_api="/api/agents/register",
                registration_payload={"name": "test-agent"},
                raw_headers=[],
            )

        assert result.allowed is False
        assert result.error_message == "Policy violation"
        assert result.gate_status_code == 403


# ===========================================================================
# RegistrationGateAuthType enum update test
# ===========================================================================


class TestRegistrationGateAuthTypeOAuth2:
    """Tests for the OAuth2 addition to RegistrationGateAuthType enum."""

    def test_oauth2_client_credentials_enum_value(self):
        """Enum contains the new oauth2_client_credentials value."""
        assert (
            RegistrationGateAuthType.OAUTH2_CLIENT_CREDENTIALS
            == "oauth2_client_credentials"
        )
