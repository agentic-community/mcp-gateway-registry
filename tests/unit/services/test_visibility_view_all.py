"""Shared read-visibility helpers honor ``view_all`` (issue #2099); redaction does not."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.core.config import DeploymentMode
from registry.services.visibility import (
    can_skip_read_visibility,
    should_redact_backend_urls,
    user_can_access_agent_from_doc,
    user_can_access_custom_entity,
    user_can_access_server,
    user_can_access_server_from_doc,
    user_can_access_skill,
)

OTHER_OWNER: str = "someone-else"
OTHER_GROUPS: list[str] = ["other-group"]


def _context(
    view_all: bool,
) -> dict:
    """A non-admin context with no grants except, optionally, view_all."""
    ui_permissions = {"view_all": ["all"]} if view_all else {}
    return {
        "username": "ro-admin",
        "is_admin": False,
        "groups": ["registry-read-only-admin"],
        "accessible_servers": [],
        "accessible_agents": [],
        "accessible_skills": [],
        "ui_permissions": ui_permissions,
    }


def _agent_doc(
    visibility: str,
) -> dict:
    return {
        "path": "/hidden-agent",
        "visibility": visibility,
        "registered_by": OTHER_OWNER,
        "allowed_groups": OTHER_GROUPS,
    }


@pytest.mark.unit
class TestCanSkipReadVisibility:
    def test_holder_skips(self):
        assert can_skip_read_visibility(_context(view_all=True)) is True

    def test_admin_skips(self):
        assert can_skip_read_visibility({"is_admin": True}) is True

    def test_plain_user_does_not_skip(self):
        assert can_skip_read_visibility(_context(view_all=False)) is False

    def test_missing_context_does_not_skip(self):
        assert can_skip_read_visibility(None) is False


@pytest.mark.unit
class TestAgentVisibility:
    @pytest.mark.parametrize("visibility", ["private", "group-restricted", "public"])
    def test_holder_sees_every_agent(self, visibility):
        assert (
            user_can_access_agent_from_doc(
                _agent_doc(visibility), _context(True), honor_view_all=True
            )
            is True
        )

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    def test_bypass_is_off_by_default(self, visibility):
        """A caller that does not opt in (any write) never gets the bypass."""
        assert user_can_access_agent_from_doc(_agent_doc(visibility), _context(True)) is False

    @pytest.mark.parametrize("visibility", ["private", "group-restricted", "public"])
    def test_plain_user_without_grant_sees_none(self, visibility):
        assert user_can_access_agent_from_doc(_agent_doc(visibility), _context(False)) is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestSkillAndCustomEntityVisibility:
    @pytest.mark.parametrize("visibility", ["private", "group"])
    async def test_holder_sees_hidden_skill(self, visibility):
        result = await user_can_access_skill(
            "/hidden-skill",
            visibility,
            OTHER_OWNER,
            OTHER_GROUPS,
            _context(True),
            honor_view_all=True,
        )
        assert result is True

    @pytest.mark.parametrize("visibility", ["private", "group"])
    async def test_plain_user_does_not_see_hidden_skill(self, visibility):
        result = await user_can_access_skill(
            "/hidden-skill", visibility, OTHER_OWNER, OTHER_GROUPS, _context(False)
        )
        assert result is False

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    async def test_holder_sees_hidden_record(self, visibility):
        result = await user_can_access_custom_entity(
            visibility, OTHER_OWNER, OTHER_GROUPS, _context(True), honor_view_all=True
        )
        assert result is True

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    async def test_record_bypass_off_by_default(self, visibility):
        result = await user_can_access_custom_entity(
            visibility, OTHER_OWNER, OTHER_GROUPS, _context(True)
        )
        assert result is False

    async def test_skill_bypass_off_by_default(self):
        result = await user_can_access_skill(
            "/hidden-skill", "private", OTHER_OWNER, OTHER_GROUPS, _context(True)
        )
        assert result is False

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    async def test_plain_user_does_not_see_hidden_record(self, visibility):
        result = await user_can_access_custom_entity(
            visibility, OTHER_OWNER, OTHER_GROUPS, _context(False)
        )
        assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestServerVisibility:
    async def test_holder_sees_server_without_grant(self):
        result = await user_can_access_server("/srv", "srv", _context(True), honor_view_all=True)
        assert result is True

    def test_holder_sees_server_from_doc(self):
        result = user_can_access_server_from_doc("/srv", "srv", _context(True), honor_view_all=True)
        assert result is True

    def test_server_from_doc_bypass_off_by_default(self):
        assert user_can_access_server_from_doc("/srv", "srv", _context(True)) is False

    async def test_plain_user_without_grant_denied(self):
        assert await user_can_access_server("/srv", "srv", _context(False)) is False

    async def test_server_bypass_off_by_default(self):
        """The OpenAPI my-credential writes, and any future write, do not opt in."""
        service = MagicMock()
        service.user_can_access_server_path = AsyncMock(return_value=False)
        with patch("registry.services.server_service.server_service", service):
            result = await user_can_access_server("/srv", "srv", _context(True))
        assert result is False


@pytest.mark.unit
class TestRedactionUnchanged:
    def test_holder_still_redacted(self):
        with patch(
            "registry.services.visibility.settings.deployment_mode",
            DeploymentMode.WITH_GATEWAY,
        ):
            assert should_redact_backend_urls(_context(True)) is True
