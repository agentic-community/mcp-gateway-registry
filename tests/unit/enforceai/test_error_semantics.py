"""
Unit tests for EnforceAI error semantics (401/403/503 mapping).
"""

import pytest

from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    ForbiddenError,
    UnauthorizedError,
)


@pytest.mark.unit
class TestErrorSemantics:
    """Test suite for EnforceAI error status mapping."""

    def test_unauthorized_maps_to_401(self):
        error = UnauthorizedError("missing token")

        assert error.status_code == 401
        assert error.as_http_exception().status_code == 401

    def test_forbidden_maps_to_403(self):
        error = ForbiddenError("missing X-Agent-Id")

        assert error.status_code == 403
        assert error.as_http_exception().status_code == 403

    def test_dependency_unavailable_maps_to_503(self):
        error = DependencyUnavailableError("db read failed")

        assert error.status_code == 503
        assert error.as_http_exception().status_code == 503

    def test_public_message_override_does_not_change_status(self):
        error = ForbiddenError(
            "detailed internal message",
            public_message="Forbidden",
        )

        assert error.status_code == 403
        assert error.as_http_exception().detail == "Forbidden"

