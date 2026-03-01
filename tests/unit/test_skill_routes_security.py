"""
Tests for skill security scan API endpoints and registration integration.

# Feature: skill-scanner-integration
# Property 4: Unsafe skill disabling and tagging

**Validates: Requirements 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 4.5, 8.4**
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from registry.schemas.skill_security import SkillSecurityScanResult


VALID_ANALYZERS = ["static", "behavioral", "llm", "meta", "virustotal", "ai-defense"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_skill(path="/test-skill", tags=None, skill_md_url="https://example.com/SKILL.md"):
    """Create a mock SkillCard."""
    mock = MagicMock()
    mock.path = path
    mock.name = "test-skill"
    mock.tags = tags or []
    mock.skill_md_url = skill_md_url
    mock.skill_md_raw_url = None
    mock.visibility = "public"
    mock.owner = "testuser"
    mock.allowed_groups = []
    return mock


def _make_unsafe_scan_result(skill_path, critical=1, high=1):
    """Create an unsafe SkillSecurityScanResult."""
    return SkillSecurityScanResult(
        skill_path=skill_path,
        scan_timestamp="2026-02-16T10:00:00Z",
        is_safe=False,
        critical_issues=critical,
        high_severity=high,
        analyzers_used=["static"],
        raw_output={},
        scan_failed=False,
    )


def _make_safe_scan_result(skill_path):
    """Create a safe SkillSecurityScanResult."""
    return SkillSecurityScanResult(
        skill_path=skill_path,
        scan_timestamp="2026-02-16T10:00:00Z",
        is_safe=True,
        critical_issues=0,
        high_severity=0,
        analyzers_used=["static"],
        raw_output={},
        scan_failed=False,
    )


# ---------------------------------------------------------------------------
# Property 4: Unsafe skill disabling and tagging
# ---------------------------------------------------------------------------


def _unsafe_result_strategy():
    """Strategy for generating unsafe scan results."""
    return st.builds(
        SkillSecurityScanResult,
        skill_path=st.from_regex(r"/[a-z][a-z0-9\-]{0,20}", fullmatch=True),
        scan_timestamp=st.just("2026-02-16T10:00:00Z"),
        is_safe=st.just(False),
        critical_issues=st.integers(min_value=0, max_value=10),
        high_severity=st.integers(min_value=1, max_value=10),
        analyzers_used=st.just(["static"]),
        raw_output=st.just({}),
        scan_failed=st.just(False),
    )


class TestUnsafeSkillDisablingAndTagging:
    """Property 4: Unsafe skill disabling and tagging."""

    @given(scan_result=_unsafe_result_strategy())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_unsafe_skill_disabled_and_tagged(self, scan_result):
        """When scan is unsafe and blocking is enabled, skill is disabled and tagged."""
        from registry.api.skill_routes import _perform_skill_security_scan_on_registration

        mock_skill = _make_mock_skill(path=scan_result.skill_path)
        mock_service = AsyncMock()
        mock_service.toggle_skill = AsyncMock()
        mock_service.update_skill = AsyncMock()

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.scan_on_registration = True
        mock_config.block_unsafe_skills = True
        mock_config.add_security_pending_tag = True

        mock_scanner = MagicMock()
        mock_scanner.get_scan_config.return_value = mock_config
        mock_scanner.scan_skill = AsyncMock(return_value=scan_result)

        with patch(
            "registry.services.skill_scanner.skill_scanner_service",
            mock_scanner,
        ):
            await _perform_skill_security_scan_on_registration(mock_skill, mock_service)

        mock_service.toggle_skill.assert_called_once_with(scan_result.skill_path, enabled=False)
        mock_service.update_skill.assert_called_once()
        call_args = mock_service.update_skill.call_args
        assert "security-pending" in call_args[0][1]["tags"]


# ---------------------------------------------------------------------------
# Unit tests for API endpoints
# ---------------------------------------------------------------------------


class TestGetSkillSecurityScan:
    """Tests for GET /api/skills/{path}/security-scan."""

    @pytest.mark.asyncio
    async def test_returns_scan_result_when_exists(self):
        """Returns scan result for a skill with existing scan data."""
        from registry.api.skill_routes import get_skill_security_scan

        mock_skill = _make_mock_skill()
        mock_result = {"skill_path": "/test-skill", "is_safe": True}

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=mock_skill)

        mock_scanner = MagicMock()
        mock_scanner.get_scan_result = AsyncMock(return_value=mock_result)

        user_context = {"is_admin": True, "username": "admin", "groups": []}

        with (
            patch("registry.api.skill_routes.get_skill_service", return_value=mock_service),
            patch("registry.services.skill_scanner.skill_scanner_service", mock_scanner),
        ):
            result = await get_skill_security_scan(
                user_context=user_context,
                skill_path="test-skill",
            )

        assert result["is_safe"] is True

    @pytest.mark.asyncio
    async def test_returns_no_results_message_when_none(self):
        """Returns message when no scan results exist."""
        from registry.api.skill_routes import get_skill_security_scan

        mock_skill = _make_mock_skill()
        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=mock_skill)

        mock_scanner = MagicMock()
        mock_scanner.get_scan_result = AsyncMock(return_value=None)

        user_context = {"is_admin": True, "username": "admin", "groups": []}

        with (
            patch("registry.api.skill_routes.get_skill_service", return_value=mock_service),
            patch("registry.services.skill_scanner.skill_scanner_service", mock_scanner),
        ):
            result = await get_skill_security_scan(
                user_context=user_context,
                skill_path="test-skill",
            )

        assert "No security scan results available" in result["message"]

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_skill(self):
        """Returns 404 when skill does not exist."""
        from fastapi import HTTPException
        from registry.api.skill_routes import get_skill_security_scan

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=None)

        user_context = {"is_admin": True, "username": "admin", "groups": []}

        with patch("registry.api.skill_routes.get_skill_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                await get_skill_security_scan(
                    user_context=user_context,
                    skill_path="nonexistent",
                )

        assert exc_info.value.status_code == 404


class TestRescanSkill:
    """Tests for POST /api/skills/{path}/rescan."""

    @pytest.mark.asyncio
    async def test_non_admin_returns_403(self):
        """Non-admin user gets 403 on rescan."""
        from fastapi import HTTPException
        from registry.api.skill_routes import rescan_skill

        user_context = {"is_admin": False, "username": "user", "groups": []}
        mock_request = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await rescan_skill(
                http_request=mock_request,
                user_context=user_context,
                skill_path="test-skill",
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_skill(self):
        """Returns 404 when skill does not exist."""
        from fastapi import HTTPException
        from registry.api.skill_routes import rescan_skill

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=None)

        user_context = {"is_admin": True, "username": "admin", "groups": []}
        mock_request = MagicMock()

        with patch("registry.api.skill_routes.get_skill_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                await rescan_skill(
                    http_request=mock_request,
                    user_context=user_context,
                    skill_path="nonexistent",
                )

        assert exc_info.value.status_code == 404


class TestRegistrationWithScanning:
    """Tests for scan-on-registration behavior."""

    @pytest.mark.asyncio
    async def test_scanning_skipped_when_disabled(self):
        """Security scan is skipped when scan_on_registration is disabled."""
        from registry.api.skill_routes import _perform_skill_security_scan_on_registration

        mock_skill = _make_mock_skill()
        mock_service = AsyncMock()

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.scan_on_registration = False

        mock_scanner = MagicMock()
        mock_scanner.get_scan_config.return_value = mock_config
        mock_scanner.scan_skill = AsyncMock()

        with patch(
            "registry.services.skill_scanner.skill_scanner_service",
            mock_scanner,
        ):
            await _perform_skill_security_scan_on_registration(mock_skill, mock_service)

        mock_scanner.scan_skill.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_skill_not_disabled(self):
        """Safe skill is not disabled after scan."""
        from registry.api.skill_routes import _perform_skill_security_scan_on_registration

        mock_skill = _make_mock_skill()
        mock_service = AsyncMock()
        mock_service.toggle_skill = AsyncMock()

        safe_result = _make_safe_scan_result("/test-skill")

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.scan_on_registration = True
        mock_config.block_unsafe_skills = True
        mock_config.add_security_pending_tag = True

        mock_scanner = MagicMock()
        mock_scanner.get_scan_config.return_value = mock_config
        mock_scanner.scan_skill = AsyncMock(return_value=safe_result)

        with patch(
            "registry.services.skill_scanner.skill_scanner_service",
            mock_scanner,
        ):
            await _perform_skill_security_scan_on_registration(mock_skill, mock_service)

        mock_service.toggle_skill.assert_not_called()
