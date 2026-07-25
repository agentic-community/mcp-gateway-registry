"""
Tests for skill security scan API endpoints and registration integration.

# Feature: skill-scanner-integration
# Property 4: Unsafe skill disabling and tagging

**Validates: Requirements 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 4.5, 8.4**
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


class TestUserCanModifySkill:
    """Dual-gate authorization for skill mutation (scope AND owner/admin).

    The caller must hold the canonical per-resource skill-mutation scope
    (modify/delete/toggle_skill for the skill name, or "all") AND be admin or
    owner. Previously this was owner-or-admin only, which left the seeded
    skill-mutation scopes inert. The scope half matches every other family; the
    always-required-scope ownership half follows the stricter server/custom-entity
    model (see _user_can_modify_skill for why this is stricter than agents).
    """

    def _skill(self, owner="alice", name="test-skill"):
        skill = MagicMock()
        skill.name = name
        skill.owner = owner
        return skill

    def test_admin_bypasses_both_checks(self):
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {"is_admin": True, "username": "admin", "ui_permissions": {}}
        assert _user_can_modify_skill(self._skill(), ctx) is True

    def test_owner_with_scope_allowed(self):
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {
            "is_admin": False,
            "username": "alice",
            "ui_permissions": {"modify_skill": ["all"]},
        }
        assert _user_can_modify_skill(self._skill(owner="alice"), ctx) is True

    def test_owner_without_scope_denied(self):
        # The key regression this fixes: ownership alone is no longer enough.
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {"is_admin": False, "username": "alice", "ui_permissions": {}}
        assert _user_can_modify_skill(self._skill(owner="alice"), ctx) is False

    def test_scope_without_ownership_denied(self):
        # Holding the scope but not owning the skill (and not admin) is denied:
        # the ownership half of the dual gate still applies to non-admins.
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {
            "is_admin": False,
            "username": "bob",
            "ui_permissions": {"modify_skill": ["all"]},
        }
        assert _user_can_modify_skill(self._skill(owner="alice"), ctx) is False

    def test_per_action_scope_resolution(self):
        # Each action resolves to its own scope; a modify grant does not satisfy
        # a delete/toggle check.
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {
            "is_admin": False,
            "username": "alice",
            "ui_permissions": {"modify_skill": ["all"]},
        }
        skill = self._skill(owner="alice")
        assert _user_can_modify_skill(skill, ctx, action="modify") is True
        assert _user_can_modify_skill(skill, ctx, action="delete") is False
        assert _user_can_modify_skill(skill, ctx, action="toggle") is False

    def test_per_resource_scope_grant(self):
        # A grant scoped to the specific skill name (not "all") also satisfies
        # the scope half.
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {
            "is_admin": False,
            "username": "alice",
            "ui_permissions": {"delete_skill": ["test-skill"]},
        }
        assert _user_can_modify_skill(self._skill(owner="alice"), ctx, action="delete") is True

    def test_owner_and_username_both_none_still_denied_without_scope(self):
        # Defensive: owner == username is trivially True when both are None, but
        # the scope half fails closed first, so a scopeless caller is still denied.
        from registry.api.skill_routes import _user_can_modify_skill

        ctx = {"is_admin": False, "username": None, "ui_permissions": {}}
        assert _user_can_modify_skill(self._skill(owner=None), ctx) is False


class TestSkillMutationEndpointsWireCorrectAction:
    """Regression guard: each mutation endpoint must resolve to ITS OWN scope.

    The endpoints call ``_user_can_modify_skill(existing, ctx, action=...)``. A
    regression that dropped the action (defaulting delete/toggle back to
    "modify") would pass every direct-call test above, so these exercise the
    endpoints end-to-end: a non-admin owner holding ONLY ``modify_skill`` must be
    denied on delete and toggle (the modify grant must not satisfy them). This is
    the same cross-action bleed that previously bit agent toggle/modify.
    """

    def _skill(self, owner="alice", name="test-skill"):
        skill = MagicMock()
        skill.name = name
        skill.owner = owner
        skill.path = "/test-skill"
        return skill

    def _owner_ctx_with_modify_only(self):
        return {
            "is_admin": False,
            "username": "alice",
            "ui_permissions": {"modify_skill": ["all"]},
        }

    @pytest.mark.asyncio
    async def test_delete_denied_with_modify_only_grant(self):
        from fastapi import HTTPException

        from registry.api.skill_routes import delete_skill

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=self._skill())
        mock_request = MagicMock()

        with patch("registry.api.skill_routes.get_skill_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                await delete_skill(
                    http_request=mock_request,
                    user_context=self._owner_ctx_with_modify_only(),
                    skill_path="test-skill",
                )

        assert exc_info.value.status_code == 403
        mock_service.delete_skill.assert_not_called()

    @pytest.mark.asyncio
    async def test_toggle_denied_with_modify_only_grant(self):
        from fastapi import HTTPException

        from registry.api.skill_routes import ToggleStateRequest, toggle_skill

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=self._skill())
        mock_request = MagicMock()

        with patch("registry.api.skill_routes.get_skill_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                await toggle_skill(
                    http_request=mock_request,
                    request=ToggleStateRequest(enabled=False),
                    user_context=self._owner_ctx_with_modify_only(),
                    skill_path="test-skill",
                )

        assert exc_info.value.status_code == 403
        mock_service.toggle_skill.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_allowed_with_delete_grant(self):
        # Positive control: the matching grant lets the owner through.
        from registry.api.skill_routes import delete_skill

        mock_service = AsyncMock()
        mock_service.get_skill = AsyncMock(return_value=self._skill())
        mock_service.delete_skill = AsyncMock(return_value=True)
        mock_request = MagicMock()

        ctx = {
            "is_admin": False,
            "username": "alice",
            "ui_permissions": {"delete_skill": ["all"]},
        }

        with (
            patch("registry.api.skill_routes.get_skill_service", return_value=mock_service),
            patch("registry.api.skill_routes.send_registration_webhook", new=AsyncMock()),
        ):
            await delete_skill(
                http_request=mock_request,
                user_context=ctx,
                skill_path="test-skill",
            )

        mock_service.delete_skill.assert_awaited_once()


class TestSkillManagementNotAdminConferring:
    """Finding #4: skill-management scopes must not confer registry admin.

    An admin granting a non-admin skill-manager group the seeded skill scope set
    (publish/modify/delete/toggle_skill: ["all"]) must NOT silently promote that
    group to full registry admin.
    """

    def test_skill_mutation_scopes_excluded(self):
        from registry.auth.privileged_constants import is_admin_conferring_action

        assert is_admin_conferring_action("publish_skill") is False
        assert is_admin_conferring_action("modify_skill") is False
        assert is_admin_conferring_action("delete_skill") is False
        assert is_admin_conferring_action("toggle_skill") is False

    def test_list_skills_still_non_conferring(self):
        from registry.auth.privileged_constants import is_admin_conferring_action

        assert is_admin_conferring_action("list_skills") is False

    def test_real_admin_actions_still_confer(self):
        # The exclusion must be surgical: genuine management scopes still confer.
        from registry.auth.privileged_constants import is_admin_conferring_action

        assert is_admin_conferring_action("register_service") is True
        assert is_admin_conferring_action("modify_agent") is True
        assert is_admin_conferring_action("delete_service") is True

    def test_user_is_admin_deriver_honors_exclusion(self):
        # The per-request admin check must honor the exclusion, not just the
        # predicate in isolation: a skill-manager grant does not promote to admin.
        from registry.auth.dependencies import _user_is_admin

        assert _user_is_admin({"modify_skill": ["all"]}) is False
        assert _user_is_admin({"publish_skill": ["all"]}) is False
        assert (
            _user_is_admin(
                {
                    "publish_skill": ["all"],
                    "modify_skill": ["all"],
                    "delete_skill": ["all"],
                    "toggle_skill": ["all"],
                    "list_skills": ["all"],
                }
            )
            is False
        )

    def test_user_is_admin_still_true_for_genuine_admin_scope(self):
        # A context that ALSO holds a genuine admin scope must still be admin: the
        # skill exclusion must not poison an otherwise-admin grant set.
        from registry.auth.dependencies import _user_is_admin

        assert _user_is_admin({"modify_skill": ["all"], "register_service": ["all"]}) is True

    def test_repo_write_guard_deriver_honors_exclusion(self):
        # The privileged-write guard (repo layer) defers to the same predicate, so
        # skill scopes are not admin-conferring there either.
        from registry.repositories.documentdb.scope_repository import _grants_admin

        assert _grants_admin({"publish_skill": ["all"]}) is False
        assert _grants_admin({"modify_skill": ["all"]}) is False
        assert _grants_admin({"register_service": ["all"]}) is True
