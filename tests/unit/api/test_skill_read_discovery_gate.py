"""Unit tests for the list_skills DISCOVERY gate on the single-skill READ path.

The eight single-skill read/consume routes (GET /skills/{path}, /content,
/integrity, /health, /tools, /rating, /security-scan, and the rate write) all
share ``_user_can_access_skill``. That helper used to check only visibility
(admin -> True, PUBLIC -> True, ...), so a caller whose ``list_skills`` grant is
empty could still read -- and, via ``/content``, fetch with the skill's stored
credential -- essentially every skill directly, bypassing the type-level
discovery gate that the list path and the semantic-search branch enforce.

These tests exercise ``_user_can_access_skill`` directly and assert the
discovery gate now layers on top of visibility (fail closed), matching
``list_skills_for_user`` and ``search_routes``.
"""

import logging
from types import SimpleNamespace

import pytest

from registry.api.skill_routes import _user_can_access_skill
from registry.schemas.skill_models import VisibilityEnum

logger = logging.getLogger(__name__)


def _skill(
    name: str = "my-skill",
    visibility: VisibilityEnum = VisibilityEnum.PUBLIC,
    owner: str = "alice",
    groups: list[str] | None = None,
) -> SimpleNamespace:
    """Minimal skill exposing the attributes ``_user_can_access_skill`` reads."""
    return SimpleNamespace(
        name=name,
        visibility=visibility,
        owner=owner,
        allowed_groups=groups or [],
    )


def _ctx(
    *,
    is_admin: bool = False,
    username: str = "bob",
    groups: list[str] | None = None,
    list_skills: list[str] | None = None,
) -> dict:
    """Build a user context with an optional ``list_skills`` ui_permission."""
    ui_permissions: dict[str, list[str]] = {}
    if list_skills is not None:
        ui_permissions["list_skills"] = list_skills
    return {
        "is_admin": is_admin,
        "username": username,
        "groups": groups or [],
        "ui_permissions": ui_permissions,
    }


@pytest.mark.unit
class TestSkillReadDiscoveryGate:
    def test_no_list_grant_denies_public(self):
        """Non-admin with no list_skills grant is denied a PUBLIC skill."""
        skill = _skill(visibility=VisibilityEnum.PUBLIC)
        assert _user_can_access_skill(skill, _ctx(list_skills=[])) is False

    def test_missing_ui_permissions_denies(self):
        """A context with no ui_permissions at all fails closed."""
        skill = _skill(visibility=VisibilityEnum.PUBLIC)
        ctx = {"is_admin": False, "username": "bob", "groups": []}
        assert _user_can_access_skill(skill, ctx) is False

    def test_no_list_grant_denies_own_private(self):
        """Even the owner of a PRIVATE skill is denied without a list grant.

        Visibility alone would have allowed the owner; the discovery gate now
        precedes it, so lacking ``list_skills`` denies discovery entirely.
        """
        skill = _skill(visibility=VisibilityEnum.PRIVATE, owner="alice")
        ctx = _ctx(username="alice", list_skills=[])
        assert _user_can_access_skill(skill, ctx) is False

    def test_all_grant_allows_public(self):
        """list_skills:[all] passes the gate; PUBLIC visibility then allows."""
        skill = _skill(visibility=VisibilityEnum.PUBLIC)
        assert _user_can_access_skill(skill, _ctx(list_skills=["all"])) is True

    def test_named_grant_allows_that_skill(self):
        """A named list_skills grant surfaces exactly that skill."""
        skill = _skill(name="my-skill", visibility=VisibilityEnum.PUBLIC)
        assert _user_can_access_skill(skill, _ctx(list_skills=["my-skill"])) is True

    def test_named_grant_other_skill_denied(self):
        """A named grant for a different skill does not open this one."""
        skill = _skill(name="my-skill", visibility=VisibilityEnum.PUBLIC)
        assert _user_can_access_skill(skill, _ctx(list_skills=["other-skill"])) is False

    def test_gate_open_but_visibility_still_applies(self):
        """Discovery grant does not bypass per-record visibility.

        bob may discover the skill (all-grant) but a PRIVATE skill owned by
        alice is still denied to him.
        """
        skill = _skill(name="my-skill", visibility=VisibilityEnum.PRIVATE, owner="alice")
        ctx = _ctx(username="bob", list_skills=["all"])
        assert _user_can_access_skill(skill, ctx) is False

    def test_gate_open_owner_sees_own_private(self):
        """With the gate open, the owner still sees their PRIVATE skill."""
        skill = _skill(name="my-skill", visibility=VisibilityEnum.PRIVATE, owner="alice")
        ctx = _ctx(username="alice", list_skills=["all"])
        assert _user_can_access_skill(skill, ctx) is True

    def test_gate_open_group_member_sees_group_skill(self):
        """Group visibility still works once the discovery gate is open."""
        skill = _skill(
            name="my-skill",
            visibility=VisibilityEnum.GROUP,
            owner="alice",
            groups=["secret-group"],
        )
        ctx = _ctx(username="bob", groups=["secret-group"], list_skills=["all"])
        assert _user_can_access_skill(skill, ctx) is True

    def test_admin_bypasses_gate(self):
        """Admin reads any skill regardless of ui_permissions."""
        skill = _skill(visibility=VisibilityEnum.PRIVATE, owner="someone-else")
        assert _user_can_access_skill(skill, _ctx(is_admin=True, list_skills=[])) is True
