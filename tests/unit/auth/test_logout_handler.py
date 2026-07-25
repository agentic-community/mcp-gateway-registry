"""Tests for the OAuth2 logout handler S2S flow (issue #1503).

The logout handler must never place the id_token_hint JWT in a browser-facing
URL. Instead it calls the auth-server directly over the internal network
(settings.auth_server_url) and redirects the browser straight to the IdP logout
URL returned in the Location header. These tests verify:

  - the id_token_hint travels only on the internal S2S call, never in the
    browser redirect;
  - X-Forwarded-Host/Proto are forwarded so auth-server's redirect_uri
    same-origin validation approves the post-logout URI;
  - logout always completes locally (cookie cleared, /login fallback) when the
    S2S call fails, returns a non-redirect status, or yields an unsafe Location;
  - the internal auth_server_url is used for the S2S hop, not the public
    auth_server_external_url (which sits behind the WAF).
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from registry.auth import routes
from registry.auth.routes import _is_safe_logout_redirect, logout_handler
from registry.core.config import Settings

_SESSION_COOKIE = "mcp_gateway_session"
_ID_TOKEN = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl"
_IDP_LOGOUT_URL = (
    "https://login.microsoftonline.com/common/oauth2/v2.0/logout"
    "?post_logout_redirect_uri=https%3A%2F%2Fexample.com%2Flogout"
)


@pytest.fixture(autouse=True)
def _trust_example_host():
    """Trust example.com so redirect_uri construction does not fail closed."""
    with patch.object(
        Settings,
        "trusted_external_hosts_set",
        new_callable=PropertyMock,
        return_value={"example.com", "localhost", "localhost:7860"},
    ):
        yield


def _make_request(host: str = "example.com", scheme: str = "https") -> MagicMock:
    """Create a mock Request with an https example.com origin."""
    request = MagicMock()
    header_dict = {
        "host": host,
        "x-cloudfront-forwarded-proto": "",
        "x-forwarded-proto": "https" if scheme == "https" else "",
    }
    request.headers = MagicMock()
    request.headers.get = lambda key, default="": header_dict.get(key, default)
    request.url = MagicMock()
    request.url.scheme = scheme
    request.state = MagicMock()
    return request


def _mock_s2s_client(response=None, exc=None):
    """Build a MagicMock standing in for httpx.AsyncClient(...) as a context manager."""
    client_factory = MagicMock()
    instance = AsyncMock()
    if exc is not None:
        instance.get.side_effect = exc
    else:
        instance.get.return_value = response
    client_factory.return_value.__aenter__ = AsyncMock(return_value=instance)
    client_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return client_factory, instance


def _s2s_response(status_code=302, location=_IDP_LOGOUT_URL):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"location": location} if location is not None else {}
    return resp


def _oauth2_session():
    return {
        "auth_method": "oauth2",
        "provider": "entra",
        "id_token": _ID_TOKEN,
        "session_id": "sess-123",
    }


class TestIsSafeLogoutRedirect:
    """Unit tests for the defense-in-depth Location validator."""

    def test_absolute_https_allowed(self):
        assert _is_safe_logout_redirect("https://login.microsoftonline.com/logout") is True

    def test_absolute_http_allowed(self):
        assert _is_safe_logout_redirect("http://keycloak.example.com/logout") is True

    def test_relative_path_allowed(self):
        assert _is_safe_logout_redirect("/login") is True

    def test_javascript_scheme_rejected(self):
        assert _is_safe_logout_redirect("javascript:alert(1)") is False

    def test_data_scheme_rejected(self):
        assert _is_safe_logout_redirect("data:text/html,<script>") is False

    def test_empty_rejected(self):
        assert _is_safe_logout_redirect("") is False


class TestLogoutHandlerS2S:
    """Behavioral tests for the S2S logout flow."""

    async def _run(self, client_factory, session_data=None):
        """Invoke logout_handler with session resolution and delete patched."""
        request = _make_request()
        with (
            patch.object(routes, "httpx") as mock_httpx,
            patch(
                "registry.auth.dependencies.resolve_session_from_cookie",
                new=AsyncMock(return_value=session_data or _oauth2_session()),
            ),
            patch(
                "registry.auth.session_store.delete_session",
                new=AsyncMock(return_value=True),
            ),
        ):
            mock_httpx.AsyncClient = client_factory
            response = await logout_handler(request, session="cookie-value")
        return response

    @pytest.mark.asyncio
    async def test_id_token_hint_only_on_s2s_call_never_in_browser_url(self):
        """The JWT must appear on the internal S2S call, not the browser redirect."""
        factory, instance = _mock_s2s_client(_s2s_response())
        response = await self._run(factory)

        # Browser is redirected to the IdP URL from Location, which carries no
        # id_token_hint from our side (the IdP embeds it internally).
        assert response.status_code == 303
        assert response.headers["location"] == _IDP_LOGOUT_URL
        assert _ID_TOKEN not in response.headers["location"]

        # The id_token_hint was passed only on the server-to-server GET.
        _, kwargs = instance.get.call_args
        assert kwargs["params"]["id_token_hint"] == _ID_TOKEN

    @pytest.mark.asyncio
    async def test_s2s_uses_internal_url_not_external(self):
        """S2S hop must target auth_server_url (internal), not the WAF-fronted external URL."""
        factory, instance = _mock_s2s_client(_s2s_response())
        with (
            patch.object(routes.settings, "auth_server_url", "http://auth-server:8888"),
            patch.object(routes.settings, "auth_server_external_url", "https://example.com"),
        ):
            await self._run(factory)

        args, _ = instance.get.call_args
        assert args[0] == "http://auth-server:8888/oauth2/logout/entra"

    @pytest.mark.asyncio
    async def test_forwarded_headers_sent(self):
        """X-Forwarded-Host/Proto must be forwarded for same-origin redirect_uri validation."""
        factory, instance = _mock_s2s_client(_s2s_response())
        await self._run(factory)

        _, kwargs = instance.get.call_args
        assert kwargs["headers"]["X-Forwarded-Host"] == "example.com"
        assert kwargs["headers"]["X-Forwarded-Proto"] == "https"

    @pytest.mark.asyncio
    async def test_post_logout_redirect_uri_lands_on_logout_screen(self):
        """The post-logout redirect_uri lands on /logout, which renders the
        'Successfully Logged Out' confirmation screen. This URL must be registered
        as a valid sign-out redirect in the IdP (Cognito rejects an unregistered
        logout_uri with 'Required parameters missing' -- issue #1503); the
        same-origin fix lets the https public /logout through on the S2S hop."""
        factory, instance = _mock_s2s_client(_s2s_response())
        await self._run(factory)

        _, kwargs = instance.get.call_args
        redirect_uri = kwargs["params"]["redirect_uri"]
        assert redirect_uri.endswith("/logout"), redirect_uri

    @pytest.mark.asyncio
    async def test_cookie_cleared_on_success(self):
        """The session cookie must be cleared on the IdP redirect response."""
        factory, _ = _mock_s2s_client(_s2s_response())
        response = await self._run(factory)
        set_cookie = response.headers.get("set-cookie", "")
        assert _SESSION_COOKIE in set_cookie

    @pytest.mark.asyncio
    async def test_fallback_to_login_when_s2s_raises(self):
        """A failed S2S call must still complete logout locally via /login."""
        factory, _ = _mock_s2s_client(exc=Exception("connection refused"))
        response = await self._run(factory)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_fallback_when_non_redirect_status(self):
        """A non-3xx auth-server response falls back to /login."""
        factory, _ = _mock_s2s_client(_s2s_response(status_code=500, location=None))
        response = await self._run(factory)
        assert response.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_fallback_when_location_missing(self):
        """A 3xx with no Location header falls back to /login."""
        factory, _ = _mock_s2s_client(_s2s_response(location=None))
        response = await self._run(factory)
        assert response.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_unsafe_location_rejected(self):
        """A javascript: Location from the auth-server must not be followed."""
        factory, _ = _mock_s2s_client(_s2s_response(location="javascript:alert(1)"))
        response = await self._run(factory)
        assert response.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_malformed_id_token_not_forwarded(self):
        """A non-JWT id_token must not be placed in the S2S params."""
        factory, instance = _mock_s2s_client(_s2s_response())
        session = _oauth2_session()
        session["id_token"] = "not-a-jwt"
        await self._run(factory, session_data=session)

        _, kwargs = instance.get.call_args
        assert "id_token_hint" not in kwargs["params"]

    @pytest.mark.asyncio
    async def test_non_oauth2_session_skips_s2s(self):
        """A non-OAuth2 session logs out locally without any S2S call."""
        factory, instance = _mock_s2s_client(_s2s_response())
        session = {"auth_method": "session", "session_id": "sess-9"}
        response = await self._run(factory, session_data=session)

        instance.get.assert_not_called()
        assert response.headers["location"] == "/login"
