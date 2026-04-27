"""Tests for CSRF token validation with Bearer token bypass."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from registry.auth.csrf import (
    generate_csrf_token,
    verify_csrf_token_flexible,
)


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
    async def test_reject_when_session_cookie_but_no_csrf_token(self):
        """Session cookie present but no CSRF token should return 403."""
        request = _make_request(
            cookies={"mcp_gateway_session": "test-session"},
            headers={},
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_csrf_token_flexible(request)

        assert exc_info.value.status_code == 403
        assert "no token provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_when_session_cookie_and_invalid_csrf_token(self):
        """Session cookie + invalid CSRF token should return 403."""
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
