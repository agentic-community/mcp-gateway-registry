"""Tests for CSRF token validation with Bearer token bypass."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from registry.auth.csrf import (
    generate_csrf_token,
    verify_csrf_token_flexible,
    verify_csrf_token_header_only,
)


@pytest.fixture
def mock_session_resolver(monkeypatch):
    """Mock the cookie -> session_id resolver used by the CSRF dependency.

    Default resolves any non-empty cookie to session_id 'sid-1'. Tests that
    want an unresolvable cookie can set `mock_session_resolver.next_value = None`.
    """

    class _Stub:
        next_value: dict | None = {"session_id": "sid-1", "username": "u"}

    stub = _Stub()

    async def _fake_resolve(cookie_value: str):
        return stub.next_value

    monkeypatch.setattr("registry.auth.csrf.resolve_session_from_cookie", _fake_resolve)
    return stub


def _make_request(
    cookies: dict | None = None,
    headers: dict | None = None,
    form_data: dict | None = None,
):
    """Create a mock Request object with optional cookies, headers, and form data.

    The cookies travel in the ``cookie`` header, which the session cookie
    reader parses; ``scope`` holds its per-request cache.
    """
    request = MagicMock()
    request.scope = {}

    header_dict = dict(headers or {})
    if cookies:
        header_dict["cookie"] = "; ".join(f"{name}={value}" for name, value in cookies.items())
    request.headers = Headers(header_dict)

    request.form = AsyncMock(return_value=form_data or {})
    return request


class TestVerifyCsrfTokenFlexibleBypass:
    """Tests for the session-cookie-based CSRF bypass."""

    @pytest.mark.asyncio
    async def test_skip_csrf_when_no_session_cookie(self):
        """No session cookie means non-browser client, CSRF check is skipped."""
        request = _make_request(cookies={}, headers={})
        await verify_csrf_token_flexible(request)

    @pytest.mark.asyncio
    async def test_skip_csrf_for_bearer_token_client(self):
        """Bearer token client with no cookies should skip CSRF."""
        request = _make_request(
            cookies={},
            headers={"Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.test"},
        )
        await verify_csrf_token_flexible(request)


class TestVerifyCsrfTokenFlexibleEnforcement:
    """Tests for CSRF enforcement when session cookie is present."""

    @pytest.mark.asyncio
    async def test_reject_when_session_cookie_but_no_csrf_token(self, mock_session_resolver):
        """Session cookie present (and resolvable) but no CSRF token → 403."""
        request = _make_request(
            cookies={"mcp_gateway_session": "test-session"},
            headers={},
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf_token_flexible(request)

        assert exc_info.value.status_code == 403
        assert "no token provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_when_session_cookie_and_invalid_csrf_token(self, mock_session_resolver):
        """Resolvable session cookie + invalid CSRF token → 403."""
        request = _make_request(
            cookies={"mcp_gateway_session": "test-session"},
            headers={"X-CSRF-Token": "invalid-token-value"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf_token_flexible(request)

        assert exc_info.value.status_code == 403
        assert "invalid token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_pass_when_session_cookie_and_valid_csrf_header(self, mock_session_resolver):
        """Resolvable session cookie + valid CSRF token in header should pass."""
        csrf_token = generate_csrf_token("sid-1")

        request = _make_request(
            cookies={"mcp_gateway_session": "opaque-cookie"},
            headers={"X-CSRF-Token": csrf_token},
        )

        await verify_csrf_token_flexible(request)

    @pytest.mark.asyncio
    async def test_pass_when_session_cookie_and_valid_csrf_form(self, mock_session_resolver):
        """Resolvable session cookie + valid CSRF token in form data should pass."""
        csrf_token = generate_csrf_token("sid-1")

        request = _make_request(
            cookies={"mcp_gateway_session": "opaque-cookie"},
            headers={},
            form_data={"csrf_token": csrf_token},
        )

        await verify_csrf_token_flexible(request)

    @pytest.mark.asyncio
    async def test_header_token_takes_precedence_over_form(self, mock_session_resolver):
        """X-CSRF-Token header should be checked before form data."""
        valid_token = generate_csrf_token("sid-1")

        request = _make_request(
            cookies={"mcp_gateway_session": "opaque-cookie"},
            headers={"X-CSRF-Token": valid_token},
            form_data={"csrf_token": "wrong-token"},
        )

        await verify_csrf_token_flexible(request)


class TestVerifyCsrfTokenHeaderOnly:
    """Tests for verify_csrf_token_header_only (used by GET endpoints like
    connect-config).

    Regression guard for the bug where this dependency compared the CSRF token
    against the RAW session cookie blob instead of the resolved session_id. The
    token is signed with the resolved session_id, so the raw-cookie comparison
    always failed -> 403 on every cookie-authenticated GET. The happy-path test
    below (valid token + resolvable cookie must PASS) is what was missing.
    """

    @pytest.mark.asyncio
    async def test_skip_csrf_when_no_session_cookie(self):
        """No session cookie (Bearer-token / CLI client) skips the CSRF check."""
        request = _make_request(cookies={}, headers={})
        await verify_csrf_token_header_only(request)

    @pytest.mark.asyncio
    async def test_pass_when_cookie_resolves_and_token_valid(self, mock_session_resolver):
        """The regression case: a valid CSRF token (signed with the resolved
        session_id) plus a resolvable session cookie must PASS.

        The cookie value is intentionally DIFFERENT from the session_id to prove
        the dependency resolves the cookie rather than comparing the raw blob.
        """
        # mock_session_resolver resolves any cookie to session_id 'sid-1'.
        csrf_token = generate_csrf_token("sid-1")

        request = _make_request(
            cookies={"mcp_gateway_session": "opaque-cookie-blob-not-the-session-id"},
            headers={"X-CSRF-Token": csrf_token},
        )

        await verify_csrf_token_header_only(request)

    @pytest.mark.asyncio
    async def test_reject_when_cookie_resolves_but_no_token(self, mock_session_resolver):
        """Resolvable session cookie but missing X-CSRF-Token header → 403."""
        request = _make_request(
            cookies={"mcp_gateway_session": "test-session"},
            headers={},
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf_token_header_only(request)

        assert exc_info.value.status_code == 403
        assert "X-CSRF-Token header required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_when_cookie_resolves_and_token_invalid(self, mock_session_resolver):
        """Resolvable session cookie + invalid CSRF token → 403."""
        request = _make_request(
            cookies={"mcp_gateway_session": "test-session"},
            headers={"X-CSRF-Token": "invalid-token-value"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf_token_header_only(request)

        assert exc_info.value.status_code == 403
        assert "invalid token" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check",
    [verify_csrf_token_flexible, verify_csrf_token_header_only],
    ids=["flexible", "header-only"],
)
async def test_refuse_when_cookie_present_but_unresolvable(mock_session_resolver, check):
    """A session cookie that resolves to no session (legacy, expired, tampered,
    store down) is refused, never skipped: the request carries ambient
    credentials, and nothing proves it is not a session another reader accepts."""
    mock_session_resolver.next_value = None
    request = _make_request(
        cookies={"mcp_gateway_session": "stale-or-tampered"},
        headers={"X-CSRF-Token": generate_csrf_token("sid-1")},
    )

    with pytest.raises(HTTPException) as exc_info:
        await check(request)

    assert exc_info.value.status_code == 403
