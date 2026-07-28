"""Security regression tests: a management-plane mutation cannot suppress its
own audit record by shadowing the "/health" substring.

Threat: the audit middleware skips logging for health-check traffic when
``audit_log_health_checks`` is False. It used to do so with a
``"/health" in path.lower()`` SUBSTRING test on ``request.url.path``. The server
management routes embed the caller-chosen server path in the request URL via
``{service_path:path}`` / ``{path:path}`` (e.g. ``POST /api/toggle/<path>``,
``POST /api/servers/<path>/rescan``, ``PATCH /api/servers/<path>/auth-credential``).
So an attacker who registered a server named ``health`` could issue MUTATING
admin actions whose URL contained ``/health`` — the substring test matched and
the audit record for the mutation was silently skipped.

The invariants these tests protect:
  1. A genuine monitoring endpoint (``/health``, ``/api/health`` and its
     sub-tree, ``/api/federation/health`` ...) is still suppressed when
     ``log_health_checks`` is False — legitimate behaviour preserved.
  2. A management-plane path that merely CONTAINS "health" (a server named
     ``health``) is NOT treated as a health check and IS audited — the bypass
     is closed.
  3. Registering a server under a monitoring-route name (``health`` etc.) is
     rejected at the source (fail closed), so the collision can never exist.
"""

import pytest

from registry.audit.middleware import AuditMiddleware, _is_health_check_path
from registry.exceptions import UrlValidationError
from registry.utils.url_guard import validate_server_path


class TestGenuineHealthEndpointsStillSuppressed:
    """Legitimate health suppression must be preserved (default flag off)."""

    def _middleware(self) -> AuditMiddleware:
        from unittest.mock import MagicMock

        return AuditMiddleware(app=MagicMock(), audit_logger=MagicMock())

    @pytest.mark.parametrize(
        "path",
        [
            "/health",
            "/api/health",
            "/api/health/",
            "/api/health/ws/health_status",
            "/api/health/ws/stats",
            "/api/federation/health",
            "/api/admin/ans/health",
            "/api/internal/healthcheck",
            "/api/servers/health",
        ],
    )
    def test_real_health_endpoints_not_logged(self, path):
        """Known monitoring routes are still skipped when logging is off."""
        assert _is_health_check_path(path) is True
        assert self._middleware()._should_log(path) is False

    def test_health_endpoints_logged_when_enabled(self):
        """The opt-in flag still forces health traffic to be audited."""
        from unittest.mock import MagicMock

        mw = AuditMiddleware(app=MagicMock(), audit_logger=MagicMock(), log_health_checks=True)
        assert mw._should_log("/health") is True
        assert mw._should_log("/api/health/ws/health_status") is True


class TestManagementMutationContainingHealthIsAudited:
    """A URL that merely contains "health" must NOT bypass the audit trail."""

    def _middleware(self) -> AuditMiddleware:
        from unittest.mock import MagicMock

        return AuditMiddleware(app=MagicMock(), audit_logger=MagicMock())

    @pytest.mark.parametrize(
        "path",
        [
            "/api/toggle/health",
            "/api/edit/health",
            "/api/servers/health/rescan",
            "/api/servers/health/auth-credential",
            "/api/refresh/health",
            "/api/servers/health/versions/default",
        ],
    )
    def test_management_mutation_paths_are_audited(self, path):
        """BYPASS CLOSED: a mutation against a server named "health" is audited.

        These paths embed the caller-chosen ``health`` server path; the old
        substring test skipped them from the audit trail. They must now be
        classified as non-health and logged.
        """
        assert _is_health_check_path(path) is False
        assert self._middleware()._should_log(path) is True

    def test_substring_only_paths_are_not_treated_as_health(self):
        """A path that only *contains* "health" is not a health endpoint."""
        assert _is_health_check_path("/api/toggle/healthy-server") is False
        assert _is_health_check_path("/api/servers/health-monitor/rescan") is False
        assert _is_health_check_path("/api/edit/my-health-check") is False


class TestServerNameShadowingHealthRejected:
    """Fail closed at the source: a server cannot be named after a health route."""

    @pytest.mark.parametrize(
        "path",
        [
            "/health",
            "health",
            "/Health",
            "/HEALTH",
            "/healthcheck",
            "/metrics",
            "/well-known",
            "/.well-known",
        ],
    )
    def test_reserved_monitoring_names_rejected(self, path):
        """Registration under a monitoring-route name is rejected."""
        with pytest.raises(UrlValidationError):
            validate_server_path(path)

    def test_ordinary_server_path_still_allowed(self):
        """A benign path that merely contains the word health is registerable."""
        # Must not raise — the reservation is exact-name, not substring.
        validate_server_path("/healthy-app")
        validate_server_path("/github")
        validate_server_path("/team/health-dashboard")
