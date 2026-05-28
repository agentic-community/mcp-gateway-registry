"""
Unit tests for auth_server/models/custom_authorizer.py
        and auth_server/services/custom_authorizer.py

Tests cover:
  - AuthorizerMode enum parsing and string comparison
  - All Pydantic model constructors (required vs optional fields)
  - mask_sensitive_headers (all sensitive headers, Bearer preview, case)
  - CustomAuthorizerClient (success, deny-200, deny-403, timeout,
    network error, 5xx, malformed JSON, API key forwarding)
  - get_authorizer_mode (default, valid values, invalid fallback, caching)
  - get_custom_authorizer_client (native→None, singleton, API key)
  - build_custom_auth_payload (with/without native auth, header masking,
    X-Body forwarding)
  - validate_custom_authorizer_config (all mode × URL combinations)
  - _reset_globals (test isolation)
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# AUTHORIZER MODE
# =============================================================================


class TestAuthorizerMode:
    """AuthorizerMode enum parsing and string comparison."""

    def test_parse_native(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode("native") == AuthorizerMode.NATIVE

    def test_parse_custom(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode("custom") == AuthorizerMode.CUSTOM

    def test_parse_both(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode("both") == AuthorizerMode.BOTH

    def test_invalid_raises_value_error(self):
        from models.custom_authorizer import AuthorizerMode
        with pytest.raises(ValueError):
            AuthorizerMode("invalid")

    def test_str_comparison_native(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode.NATIVE == "native"

    def test_str_comparison_custom(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode.CUSTOM == "custom"

    def test_str_comparison_both(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode.BOTH == "both"

    def test_values_are_lowercase_strings(self):
        from models.custom_authorizer import AuthorizerMode
        assert AuthorizerMode.NATIVE.value == "native"
        assert AuthorizerMode.CUSTOM.value == "custom"
        assert AuthorizerMode.BOTH.value == "both"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class TestCustomAuthRequest:
    """CustomAuthRequest construction and defaults."""

    def test_required_fields_only(self):
        from models.custom_authorizer import CustomAuthRequest
        req = CustomAuthRequest(
            method="GET",
            path="/api/servers",
            original_url="http://example.com/api/servers",
            client_ip="10.0.1.50",
        )
        assert req.method == "GET"
        assert req.path == "/api/servers"
        assert req.client_ip == "10.0.1.50"
        assert req.query_params == {}
        assert req.headers == {}
        assert req.body is None

    def test_all_fields(self):
        from models.custom_authorizer import CustomAuthRequest
        req = CustomAuthRequest(
            method="POST",
            path="/api/servers/register",
            original_url="http://example.com/api/servers/register?foo=bar",
            client_ip="192.168.1.100",
            query_params={"foo": "bar"},
            headers={"Content-Type": "application/json"},
            body='{"name": "my-server"}',
        )
        assert req.query_params == {"foo": "bar"}
        assert req.headers == {"Content-Type": "application/json"}
        assert req.body == '{"name": "my-server"}'

    def test_missing_required_field_raises(self):
        from models.custom_authorizer import CustomAuthRequest
        with pytest.raises(Exception):
            CustomAuthRequest(method="GET", path="/x")  # missing original_url and client_ip


class TestNativeAuthResult:
    """NativeAuthResult construction and defaults."""

    def test_only_valid_required(self):
        from models.custom_authorizer import NativeAuthResult
        result = NativeAuthResult(valid=True)
        assert result.valid is True
        assert result.username is None
        assert result.scopes == []
        assert result.groups == []
        assert result.auth_method is None
        assert result.client_id is None

    def test_all_fields_populated(self):
        from models.custom_authorizer import NativeAuthResult
        result = NativeAuthResult(
            valid=True,
            username="alice@example.com",
            scopes=["read:servers", "execute:servers"],
            groups=["registry-admins"],
            auth_method="keycloak",
            client_id="mcp-gateway-web",
        )
        assert result.username == "alice@example.com"
        assert len(result.scopes) == 2
        assert result.groups == ["registry-admins"]
        assert result.auth_method == "keycloak"
        assert result.client_id == "mcp-gateway-web"

    def test_invalid_native_result(self):
        from models.custom_authorizer import NativeAuthResult
        result = NativeAuthResult(valid=False, username="unknown")
        assert result.valid is False


class TestCustomAuthContext:
    """CustomAuthContext auto-timestamp and defaults."""

    def test_auto_timestamp_generated(self):
        from models.custom_authorizer import CustomAuthContext
        ctx = CustomAuthContext(request_id="req-1")
        assert ctx.timestamp  # non-empty
        assert ctx.gateway_version == "1.0.0"
        assert ctx.request_id == "req-1"

    def test_explicit_timestamp_accepted(self):
        from models.custom_authorizer import CustomAuthContext
        ctx = CustomAuthContext(timestamp="2026-01-01T00:00:00Z", request_id="req-2")
        assert ctx.timestamp == "2026-01-01T00:00:00Z"

    def test_custom_gateway_version(self):
        from models.custom_authorizer import CustomAuthContext
        ctx = CustomAuthContext(request_id="req-3", gateway_version="2.5.1")
        assert ctx.gateway_version == "2.5.1"


class TestCustomAuthorizerPayload:
    """CustomAuthorizerPayload with and without native auth."""

    def _make_request(self):
        from models.custom_authorizer import CustomAuthRequest
        return CustomAuthRequest(
            method="GET", path="/test", original_url="http://h/test", client_ip="1.2.3.4"
        )

    def _make_context(self):
        from models.custom_authorizer import CustomAuthContext
        return CustomAuthContext(request_id="req-x")

    def test_without_native_auth(self):
        from models.custom_authorizer import CustomAuthorizerPayload
        payload = CustomAuthorizerPayload(
            request=self._make_request(),
            native_auth_result=None,
            context=self._make_context(),
        )
        assert payload.native_auth_result is None

    def test_with_native_auth(self):
        from models.custom_authorizer import CustomAuthorizerPayload, NativeAuthResult
        payload = CustomAuthorizerPayload(
            request=self._make_request(),
            native_auth_result=NativeAuthResult(valid=True, username="alice"),
            context=self._make_context(),
        )
        assert payload.native_auth_result.username == "alice"

    def test_serialises_to_json(self):
        from models.custom_authorizer import CustomAuthorizerPayload
        payload = CustomAuthorizerPayload(
            request=self._make_request(),
            context=self._make_context(),
        )
        json_str = payload.model_dump_json()
        assert '"method"' in json_str
        assert '"request_id"' in json_str


class TestCustomAuthorizerResponse:
    """CustomAuthorizerResponse construction."""

    def test_authorized_response(self):
        from models.custom_authorizer import CustomAuthorizerResponse
        resp = CustomAuthorizerResponse(authorized=True, metadata={"policy": "allow-all"})
        assert resp.authorized is True
        assert resp.error is None
        assert resp.metadata == {"policy": "allow-all"}

    def test_denied_response_with_error(self):
        from models.custom_authorizer import CustomAuthErrorDetail, CustomAuthorizerResponse
        err = CustomAuthErrorDetail(
            code="POLICY_VIOLATION",
            message="User not permitted",
            details={"username": "bob"},
        )
        resp = CustomAuthorizerResponse(authorized=False, error=err)
        assert resp.authorized is False
        assert resp.error.code == "POLICY_VIOLATION"
        assert resp.error.details == {"username": "bob"}

    def test_minimal_denied_response(self):
        from models.custom_authorizer import CustomAuthErrorDetail, CustomAuthorizerResponse
        resp = CustomAuthorizerResponse(
            authorized=False,
            error=CustomAuthErrorDetail(code="DENIED", message="No access"),
        )
        assert resp.error.details is None


# =============================================================================
# HEADER MASKING
# =============================================================================


class TestMaskSensitiveHeaders:
    """mask_sensitive_headers — coverage of all sensitive headers."""

    def test_authorization_bearer_gets_preview(self):
        from services.custom_authorizer import mask_sensitive_headers
        token = "eyJhbGciOiJSUzI1NiJ9.payload.signature_value"
        masked = mask_sensitive_headers({"Authorization": f"Bearer {token}"})
        assert "***MASKED***" in masked["Authorization"]
        # First 4 chars of credential preserved for traceability
        assert masked["Authorization"].startswith("Bearer eyJh")
        # Full token body must not be present
        assert token[4:] not in masked["Authorization"]

    def test_authorization_non_bearer_fully_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({"Authorization": "Basic dXNlcjpwYXNz"})
        assert masked["Authorization"] == "***MASKED***"

    def test_cookie_fully_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({"Cookie": "session=abc; csrf=xyz"})
        assert masked["Cookie"] == "***MASKED***"

    def test_x_authorization_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({"X-Authorization": "Bearer secret"})
        assert "***MASKED***" in masked["X-Authorization"]

    def test_set_cookie_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({"Set-Cookie": "session=abc; HttpOnly"})
        assert masked["Set-Cookie"] == "***MASKED***"

    def test_proxy_authorization_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({"Proxy-Authorization": "Bearer proxy-token"})
        assert "***MASKED***" in masked["Proxy-Authorization"]

    def test_non_sensitive_headers_unchanged(self):
        from services.custom_authorizer import mask_sensitive_headers
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": "req-123",
            "User-Agent": "httpx/0.25.0",
            "Accept": "*/*",
        }
        masked = mask_sensitive_headers(headers)
        assert masked == headers

    def test_mixed_headers_only_sensitive_masked(self):
        from services.custom_authorizer import mask_sensitive_headers
        headers = {
            "Authorization": "Bearer mysecret",
            "Content-Type": "application/json",
            "X-Request-ID": "req-999",
        }
        masked = mask_sensitive_headers(headers)
        assert "***MASKED***" in masked["Authorization"]
        assert masked["Content-Type"] == "application/json"
        assert masked["X-Request-ID"] == "req-999"

    def test_case_insensitive_key_matching(self):
        from services.custom_authorizer import mask_sensitive_headers
        masked = mask_sensitive_headers({
            "AUTHORIZATION": "Bearer uppercase",
            "cookie": "lower=case",
        })
        assert "***MASKED***" in masked["AUTHORIZATION"]
        assert masked["cookie"] == "***MASKED***"

    def test_original_dict_not_mutated(self):
        from services.custom_authorizer import mask_sensitive_headers
        original = {"Authorization": "Bearer secret", "X-Request-ID": "r1"}
        original_copy = dict(original)
        mask_sensitive_headers(original)
        assert original == original_copy

    def test_empty_headers_returns_empty(self):
        from services.custom_authorizer import mask_sensitive_headers
        assert mask_sensitive_headers({}) == {}


# =============================================================================
# CUSTOM AUTHORIZER CLIENT
# =============================================================================


def _make_payload():
    """Helper — builds a minimal CustomAuthorizerPayload."""
    from models.custom_authorizer import CustomAuthContext, CustomAuthRequest, CustomAuthorizerPayload
    return CustomAuthorizerPayload(
        request=CustomAuthRequest(
            method="GET", path="/test", original_url="http://h/test", client_ip="1.2.3.4"
        ),
        context=CustomAuthContext(request_id="req-t"),
    )


class TestCustomAuthorizerClient:
    """CustomAuthorizerClient.authorize — all success and fail-closed paths."""

    @pytest.mark.asyncio
    async def test_returns_authorized_true_on_200(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(
            200, json={"authorized": True, "metadata": {"policy": "allow"}}
        ))
        result = await client.authorize(_make_payload())
        assert result.authorized is True
        assert result.metadata == {"policy": "allow"}

    @pytest.mark.asyncio
    async def test_returns_denied_on_200_with_authorized_false(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(
            200, json={"authorized": False, "error": {"code": "DENIED", "message": "Blocked"}}
        ))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "DENIED"

    @pytest.mark.asyncio
    async def test_accepts_403_as_valid_deny(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(
            403, json={"authorized": False, "error": {"code": "FORBIDDEN", "message": "No"}}
        ))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_fail_closed_on_timeout(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize", timeout=1.0)
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_fail_closed_on_connect_error(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "UNREACHABLE"

    @pytest.mark.asyncio
    async def test_fail_closed_on_5xx(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(500, text="Internal Error"))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "UNEXPECTED_STATUS"

    @pytest.mark.asyncio
    async def test_fail_closed_on_malformed_json(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(200, text="not json at all"))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "MALFORMED_RESPONSE"

    @pytest.mark.asyncio
    async def test_fail_closed_on_unexpected_exception(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=RuntimeError("unexpected crash"))
        result = await client.authorize(_make_payload())
        assert result.authorized is False
        assert result.error.code == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_api_key_sent_as_bearer(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize", api_key="my-secret-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(200, json={"authorized": True}))
        await client.authorize(_make_payload())
        call_headers = client._client.post.call_args.kwargs["headers"]
        assert call_headers["Authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_no_api_key_sends_no_auth_header(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize", api_key=None)
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(200, json={"authorized": True}))
        await client.authorize(_make_payload())
        call_headers = client._client.post.call_args.kwargs["headers"]
        assert "Authorization" not in call_headers

    @pytest.mark.asyncio
    async def test_payload_serialised_as_json(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=httpx.Response(200, json={"authorized": True}))
        payload = _make_payload()
        await client.authorize(payload)
        call_kwargs = client._client.post.call_args.kwargs
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        body_bytes = call_kwargs["content"]
        assert '"method"' in body_bytes
        assert '"request_id"' in body_bytes

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        from services.custom_authorizer import CustomAuthorizerClient
        client = CustomAuthorizerClient(url="http://mock/authorize")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        await client.close()
        client._client.aclose.assert_called_once()


# =============================================================================
# GET AUTHORIZER MODE
# =============================================================================


class TestGetAuthorizerMode:
    """get_authorizer_mode — env reading, caching, fallback."""

    def setup_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def teardown_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def test_default_is_native_when_env_not_set(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.delenv("AUTHORIZER_MODE", raising=False)
        assert get_authorizer_mode() == AuthorizerMode.NATIVE

    def test_parses_custom(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        assert get_authorizer_mode() == AuthorizerMode.CUSTOM

    def test_parses_both(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "both")
        assert get_authorizer_mode() == AuthorizerMode.BOTH

    def test_case_insensitive(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "BOTH")
        assert get_authorizer_mode() == AuthorizerMode.BOTH

    def test_invalid_falls_back_to_native(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "garbage")
        assert get_authorizer_mode() == AuthorizerMode.NATIVE

    def test_result_is_cached(self, monkeypatch):
        from services.custom_authorizer import get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        first = get_authorizer_mode()
        monkeypatch.setenv("AUTHORIZER_MODE", "both")  # change env after first call
        second = get_authorizer_mode()
        assert first is second  # same object returned from cache


# =============================================================================
# GET CUSTOM AUTHORIZER CLIENT
# =============================================================================


class TestGetCustomAuthorizerClient:
    """get_custom_authorizer_client — None in native mode, singleton otherwise."""

    def setup_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def teardown_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def test_returns_none_in_native_mode(self, monkeypatch):
        from services.custom_authorizer import get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "native")
        assert get_custom_authorizer_client() is None

    def test_returns_client_in_custom_mode(self, monkeypatch):
        from services.custom_authorizer import CustomAuthorizerClient, get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_TIMEOUT", "3")
        client = get_custom_authorizer_client()
        assert isinstance(client, CustomAuthorizerClient)
        assert client._url == "http://localhost:8090/authorize"
        assert client._timeout == 3.0

    def test_returns_client_in_both_mode(self, monkeypatch):
        from services.custom_authorizer import CustomAuthorizerClient, get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "both")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        client = get_custom_authorizer_client()
        assert isinstance(client, CustomAuthorizerClient)

    def test_singleton_same_instance_on_repeated_calls(self, monkeypatch):
        from services.custom_authorizer import get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        c1 = get_custom_authorizer_client()
        c2 = get_custom_authorizer_client()
        assert c1 is c2

    def test_api_key_passed_to_client(self, monkeypatch):
        from services.custom_authorizer import get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_API_KEY", "super-secret")
        client = get_custom_authorizer_client()
        assert client._api_key == "super-secret"

    def test_empty_api_key_env_becomes_none(self, monkeypatch):
        from services.custom_authorizer import get_custom_authorizer_client
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_API_KEY", "")
        client = get_custom_authorizer_client()
        assert client._api_key is None


# =============================================================================
# BUILD CUSTOM AUTH PAYLOAD
# =============================================================================


def _make_mock_request(
    method="GET",
    path="/api/servers",
    url_str="http://auth.example.com/validate",
    headers=None,
    query_params=None,
    client_ip="10.0.1.50",
):
    """Build a MagicMock that looks like a FastAPI Request."""
    req = MagicMock()
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.url.__str__ = MagicMock(return_value=url_str)
    req.headers = headers or {"Content-Type": "application/json", "X-Request-ID": "req-1"}
    req.query_params = query_params or {}
    return req, client_ip


class TestBuildCustomAuthPayload:
    """build_custom_auth_payload — payload construction and header masking."""

    def test_basic_payload_without_native_auth(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request()
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r1")
        assert payload.native_auth_result is None
        assert payload.request.method == "GET"
        assert payload.request.client_ip == ip
        assert payload.context.request_id == "r1"

    def test_payload_with_native_auth(self):
        from services.custom_authorizer import build_custom_auth_payload
        from models.custom_authorizer import NativeAuthResult
        req, ip = _make_mock_request()
        native = NativeAuthResult(
            valid=True, username="alice", scopes=["read:servers"], groups=["admins"],
            auth_method="keycloak", client_id="web-client",
        )
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=native, request_id="r2")
        assert payload.native_auth_result.username == "alice"
        assert payload.native_auth_result.scopes == ["read:servers"]

    def test_authorization_header_is_masked(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={
            "Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
            "Content-Type": "application/json",
        })
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r3")
        assert "***MASKED***" in payload.request.headers["Authorization"]

    def test_cookie_header_is_masked(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={"Cookie": "session=abc123"})
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r4")
        assert payload.request.headers["Cookie"] == "***MASKED***"

    def test_non_sensitive_headers_preserved(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={
            "Content-Type": "application/json",
            "X-Request-ID": "req-999",
        })
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r5")
        assert payload.request.headers["Content-Type"] == "application/json"
        assert payload.request.headers["X-Request-ID"] == "req-999"

    def test_x_body_header_forwarded_as_body(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={
            "Content-Type": "application/json",
            "X-Body": '{"jsonrpc":"2.0","method":"tools/list","id":1}',
        })
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r6")
        assert payload.request.body == '{"jsonrpc":"2.0","method":"tools/list","id":1}'

    def test_absent_x_body_results_in_none_body(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={"Content-Type": "application/json"})
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r7")
        assert payload.request.body is None

    def test_empty_x_body_header_becomes_none(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(headers={"X-Body": ""})
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r8")
        assert payload.request.body is None

    def test_query_params_forwarded(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request(query_params={"limit": "10", "offset": "0"})
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="r9")
        assert payload.request.query_params == {"limit": "10", "offset": "0"}

    def test_context_request_id_matches(self):
        from services.custom_authorizer import build_custom_auth_payload
        req, ip = _make_mock_request()
        with patch("services.custom_authorizer.get_client_ip", return_value=ip):
            payload = build_custom_auth_payload(request=req, native_auth_result=None, request_id="custom-id-42")
        assert payload.context.request_id == "custom-id-42"


# =============================================================================
# VALIDATE CUSTOM AUTHORIZER CONFIG
# =============================================================================


class TestValidateCustomAuthorizerConfig:
    """validate_custom_authorizer_config — startup validation logic."""

    def setup_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def teardown_method(self):
        from services.custom_authorizer import _reset_globals
        _reset_globals()

    def test_native_mode_no_url_does_not_raise(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "native")
        monkeypatch.delenv("CUSTOM_AUTHORIZER_URL", raising=False)
        validate_custom_authorizer_config()  # must not raise

    def test_custom_mode_missing_url_raises(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.delenv("CUSTOM_AUTHORIZER_URL", raising=False)
        with pytest.raises(ValueError, match="CUSTOM_AUTHORIZER_URL"):
            validate_custom_authorizer_config()

    def test_both_mode_missing_url_raises(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "both")
        monkeypatch.delenv("CUSTOM_AUTHORIZER_URL", raising=False)
        with pytest.raises(ValueError, match="CUSTOM_AUTHORIZER_URL"):
            validate_custom_authorizer_config()

    def test_custom_mode_blank_url_raises(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "   ")
        with pytest.raises(ValueError, match="CUSTOM_AUTHORIZER_URL"):
            validate_custom_authorizer_config()

    def test_custom_mode_with_valid_https_url_passes(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "https://custom.example.com/authorize")
        validate_custom_authorizer_config()  # must not raise

    def test_both_mode_with_valid_url_passes(self, monkeypatch):
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "both")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "https://custom.example.com/authorize")
        validate_custom_authorizer_config()  # must not raise

    def test_localhost_http_url_passes_without_warning_required(self, monkeypatch):
        """localhost HTTP is acceptable (dev environment)."""
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        validate_custom_authorizer_config()  # must not raise

    def test_plain_http_non_localhost_still_passes_but_logs_warning(self, monkeypatch, caplog):
        """Non-localhost HTTP should pass validation but emit a warning."""
        from services.custom_authorizer import validate_custom_authorizer_config
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://internal.corp.com/authorize")
        with caplog.at_level(logging.WARNING):
            validate_custom_authorizer_config()
        assert any("HTTPS" in r.message or "http" in r.message.lower() for r in caplog.records)


# =============================================================================
# RESET GLOBALS (TEST ISOLATION)
# =============================================================================


class TestResetGlobals:
    """_reset_globals — verifies test isolation helper works correctly."""

    def test_reset_clears_mode_singleton(self, monkeypatch):
        from services.custom_authorizer import _reset_globals, get_authorizer_mode
        from models.custom_authorizer import AuthorizerMode
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        get_authorizer_mode()  # populate cache

        _reset_globals()
        monkeypatch.setenv("AUTHORIZER_MODE", "native")
        assert get_authorizer_mode() == AuthorizerMode.NATIVE  # cache was cleared
        _reset_globals()

    def test_reset_clears_client_singleton(self, monkeypatch):
        from services.custom_authorizer import (
            _reset_globals, get_custom_authorizer_client, CustomAuthorizerClient
        )
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        monkeypatch.setenv("CUSTOM_AUTHORIZER_URL", "http://localhost:8090/authorize")
        c1 = get_custom_authorizer_client()

        _reset_globals()
        monkeypatch.setenv("AUTHORIZER_MODE", "custom")
        c2 = get_custom_authorizer_client()
        assert c1 is not c2  # a brand-new instance was created
        _reset_globals()
