"""Group-restricted visibility matches group names in canonical form.

The login filter stores canonical group names in the session (``team``), while
``allowed_groups`` on a record may hold the Keycloak full-path form (``/team``)
or the reverse. Every visibility check compares through
``registry.utils.group_names.groups_intersect`` so both forms match
(issue #1689).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from registry.schemas.skill_models import VisibilityEnum
from registry.services.skill_service import SkillService
from registry.services.visibility import (
    user_can_access_agent,
    user_can_access_agent_from_doc,
    user_can_access_custom_entity,
    user_can_access_skill,
)

pytestmark = [pytest.mark.unit]


def _ctx(groups: list[str]) -> dict:
    """Non-admin caller with every agent and skill discoverable."""
    return {
        "is_admin": False,
        "username": "bob",
        "groups": groups,
        "accessible_agents": ["all"],
        "accessible_skills": ["all"],
    }


class TestAgentVisibility:
    async def test_agent_card_with_full_path_group_is_visible(self):
        card = SimpleNamespace(
            visibility="group-restricted", registered_by="alice", allowed_groups=["/team"]
        )
        with patch(
            "registry.services.agent_service.agent_service.get_agent_info",
            new=AsyncMock(return_value=card),
        ):
            assert await user_can_access_agent("/agents/a", _ctx(["team"])) is True
            assert await user_can_access_agent("/agents/a", _ctx(["other"])) is False

    def test_agent_doc_with_full_path_group_is_visible(self):
        doc = {"path": "/agents/a", "visibility": "group-restricted", "allowed_groups": ["/team"]}
        assert user_can_access_agent_from_doc(doc, _ctx(["team"])) is True

    def test_nested_agent_card_doc_matches_bare_claim_against_full_path(self):
        doc = {
            "path": "/agents/a",
            "agent_card": {"visibility": "group-restricted", "allowed_groups": ["team"]},
        }
        assert user_can_access_agent_from_doc(doc, _ctx(["/team"])) is True


class TestSkillAndCustomEntityVisibility:
    async def test_group_skill_with_full_path_group_is_visible(self):
        assert await user_can_access_skill("/s", "group", "alice", ["/team"], _ctx(["team"]))

    async def test_group_skill_denies_non_member(self):
        assert not await user_can_access_skill("/s", "group", "alice", ["/team"], _ctx(["x"]))

    async def test_custom_entity_with_full_path_group_is_visible(self):
        assert await user_can_access_custom_entity(
            "group-restricted", "alice", ["/team"], _ctx(["team"])
        )

    async def test_custom_entity_blank_group_grants_nothing(self):
        assert not await user_can_access_custom_entity(
            "group-restricted", "alice", ["/"], _ctx([""])
        )


class TestListSkillsForUser:
    async def test_group_skill_with_full_path_group_is_listed(self):
        member_only = SimpleNamespace(
            name="hr-skill", visibility=VisibilityEnum.GROUP, owner="alice", allowed_groups=["/hr"]
        )
        service = SkillService.__new__(SkillService)
        with patch.object(service, "list_skills", new=AsyncMock(return_value=[member_only])):
            assert await service.list_skills_for_user(_ctx(["hr"])) == [member_only]
            assert await service.list_skills_for_user(_ctx(["eng"])) == []
