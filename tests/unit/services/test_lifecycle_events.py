"""Unit tests for lifecycle event helpers (Issue #1330)."""

from unittest.mock import patch

import pytest

from registry.schemas.security import SecurityScanResult
from registry.services import lifecycle_events as le
from registry.services.lifecycle_events import (
    EnforcedStatusError,
    _build_scan_fields,
    _sanitize_scan_error,
    enforce_registration_status,
)


def _scan_result(is_safe: bool, critical: int = 0, high: int = 0) -> SecurityScanResult:
    return SecurityScanResult(
        server_url="https://example.com/mcp",
        server_path="/x",
        scan_timestamp="2026-06-25T00:00:00Z",
        is_safe=is_safe,
        critical_issues=critical,
        high_severity=high,
    )


class TestSanitizeScanError:
    """Tests for _sanitize_scan_error."""

    def test_none_returns_none(self):
        assert _sanitize_scan_error(None) is None

    def test_collapses_whitespace(self):
        assert _sanitize_scan_error("a\n   b\t c") == "a b c"

    def test_truncates_to_max_len(self):
        assert len(_sanitize_scan_error("x" * 500)) == 200


class TestBuildScanFields:
    """Tests for _build_scan_fields."""

    def test_safe_result(self):
        fields = _build_scan_fields(_scan_result(True), {"tags": []}, None, False)
        assert fields["is_safe"] is True
        assert fields["severity_counts"]["critical"] == 0
        assert fields["tags_applied"] == []
        assert fields["auto_disabled"] is False
        assert fields["scan_error"] is None

    def test_unsafe_result_with_tag_and_disable(self):
        entry = {"tags": ["security-pending", "other"]}
        fields = _build_scan_fields(_scan_result(False, critical=2), entry, None, True)
        assert fields["is_safe"] is False
        assert fields["severity_counts"]["critical"] == 2
        assert fields["tags_applied"] == ["security-pending"]
        assert fields["auto_disabled"] is True

    def test_error_result(self):
        fields = _build_scan_fields(None, {"tags": []}, "RuntimeError: boom", False)
        assert fields["is_safe"] is None
        assert fields["severity_counts"] == {}
        assert fields["scan_error"] == "RuntimeError: boom"


class TestEnforceRegistrationStatus:
    """Tests for enforce_registration_status."""

    def test_unset_policy_is_passthrough(self):
        with patch.object(le.settings, "registration_enforced_status", None):
            assert enforce_registration_status(None, "server") is None
            assert enforce_registration_status("beta", "server") == "beta"

    def test_missing_status_forced_to_enforced(self):
        with patch.object(le.settings, "registration_enforced_status", "draft"):
            assert enforce_registration_status(None, "server") == "draft"

    def test_matching_status_accepted(self):
        with patch.object(le.settings, "registration_enforced_status", "draft"):
            assert enforce_registration_status("draft", "agent") == "draft"
            # Case-insensitive match.
            assert enforce_registration_status("DRAFT", "agent") == "draft"

    def test_mismatched_status_raises(self):
        with patch.object(le.settings, "registration_enforced_status", "draft"):
            with pytest.raises(EnforcedStatusError, match="draft"):
                enforce_registration_status("active", "skill")


class TestUserCanChangeLifecycleStatus:
    """Tests for user_can_change_lifecycle_status."""

    def test_admin_always_allowed(self):
        assert le.user_can_change_lifecycle_status("svc", {"is_admin": True}) is True

    def test_non_admin_without_permission_denied(self):
        ctx = {"is_admin": False, "ui_permissions": {}}
        assert le.user_can_change_lifecycle_status("svc", ctx) is False

    def test_non_admin_with_permission_allowed(self):
        ctx = {
            "is_admin": False,
            "ui_permissions": {"change_lifecycle_status": ["all"]},
        }
        assert le.user_can_change_lifecycle_status("svc", ctx) is True
