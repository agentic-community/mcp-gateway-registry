"""Tests for CSRF token validation with Bearer token bypass."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

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
    """Create a mock Request object with optional cookies, headers, and form data."""
    request = MagicMock()
    request.cookies = cookies or {}

    header_dict = headers or {}
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: header_dict.get(key, default)

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
    async def test_pass_when_session_cookie_and_valid_csrf_header(self):
        """Session cookie + valid CSRF token in header should pass."""
        session_id = "test-session-id"
        csrf_token = generate_csrf_token(session_id)

        request = _make_request(
            cookies={"mcp_gateway_session": session_id},
            headers={"X-CSRF-Token": csrf_token},
        )

        await verify_csrf_token_flexible(request)

    @pytest.mark.asyncio
    async def test_pass_when_session_cookie_and_valid_csrf_form(self):
        """Session cookie + valid CSRF token in form data should pass."""
        session_id = "test-session-id"
        csrf_token = generate_csrf_token(session_id)

        request = _make_request(
            cookies={"mcp_gateway_session": session_id},
            headers={},
            form_data={"csrf_token": csrf_token},
        )

        await verify_csrf_token_flexible(request)

    @pytest.mark.asyncio
    async def test_header_token_takes_precedence_over_form(self):
        """X-CSRF-Token header should be checked before form data."""
        session_id = "test-session-id"
        valid_token = generate_csrf_token(session_id)

        request = _make_request(
            cookies={"mcp_gateway_session": session_id},
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
    async def test_skip_when_cookie_present_but_unresolvable(self, mock_session_resolver):
        """Cookie present but unresolvable (legacy/expired/tampered) skips the
        check (the downstream auth dependency rejects with 401 anyway)."""
        mock_session_resolver.next_value = None
        request = _make_request(
            cookies={"mcp_gateway_session": "stale-or-tampered"},
            headers={},
        )

        await verify_csrf_token_header_only(request)
