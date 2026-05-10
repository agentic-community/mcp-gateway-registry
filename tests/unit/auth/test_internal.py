"""Unit tests for registry/auth/internal.py header-parsing helper.

The public ``validate_internal_auth`` FastAPI dependency is exercised
end-to-end by ``tests/auth_server/unit/test_server.py::TestInternalRouterGate``
which calls through the full HTTP stack and the router-level gate.

These tests hit the private helper ``_validate_authorization_header``
directly so a regression in the header-parsing logic fails a focused
unit test rather than an HTTP meta-test. Localized, fast, and targeted.

Issue #998.
"""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from registry.auth.internal import _validate_authorization_header


class TestValidateAuthorizationHeader:
    """Direct tests for the header-parsing helper."""

    def test_none_raises_401_missing_header(self) -> None:
        """When no Authorization header is present on the request, the
        helper must reject with 401 and the "Missing authorization header"
        detail (so the router-level dependency returns a consistent error
        to callers)."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_authorization_header(None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing authorization header"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    def test_empty_string_raises_401_missing_header(self) -> None:
        """An empty string is semantically the same as no header; the
        helper must treat it the same way. Otherwise an upstream bug
        that substitutes '' for None could silently leak through."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_authorization_header("")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing authorization header"

    def test_basic_auth_scheme_raises_401_unsupported(self) -> None:
        """Non-Bearer schemes must be rejected. This is the defense against
        a caller mistakenly sending HTTP Basic to an endpoint that requires
        the signed internal JWT."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_authorization_header("Basic YWxpY2U6cGFzcw==")
        assert exc_info.value.status_code == 401
        assert "Unsupported authentication scheme" in exc_info.value.detail

    def test_bearer_with_empty_token_raises_401_invalid(self) -> None:
        """'Bearer ' (trailing space, empty token) passes the ``startswith``
        check but fails inside ``_validate_bearer_token`` because pyjwt
        cannot decode an empty string. Must surface as a 401, not a 500.

        Needs SECRET_KEY set for ``_validate_bearer_token`` to proceed
        past its own config check; the actual JWT decode is what fails."""
        with patch.dict(os.environ, {"SECRET_KEY": "x" * 32}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header("Bearer ")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"
