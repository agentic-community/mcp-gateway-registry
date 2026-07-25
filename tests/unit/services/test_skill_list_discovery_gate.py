"""Unit tests for the list_skills DISCOVERY gate in list_skills_for_user.

Skills now have a type-level discovery gate (parity with list_service /
list_agents / list_<type>_entity): a caller with no list_skills grant sees ZERO
skills -- including public ones -- BEFORE the per-record visibility check. Admin
bypasses. These tests exercise the real SkillService.list_skills_for_user with
list_skills mocked, asserting the gate layers correctly on top of visibility.
"""

import logging
import uuid

import pytest

from registry.schemas.skill_models import SkillInfo, VisibilityEnum
from registry.services.skill_service import SkillService

logger = logging.getLogger(__name__)


def _skill(name: str, visibility: VisibilityEnum, owner: str = "someone", groups=None) -> SkillInfo:
    return SkillInfo(
        id=str(uuid.uuid4()),
        path=f"/skills/{name}",
        name=name,
        description="d",
        skill_md_url="https://example.com/SKILL.md",
        skill_md_raw_url=None,
        repository_url=None,
        tags=[],
        author=None,
        version=None,
        metadata=None,
        compatibility=None,
        target_agents=[],
        is_enabled=True,
        visibility=visibility,
        allowed_groups=groups or [],
        registry_name="local",
        owner=owner,
        auth_scheme="none",
        auth_header_name=None,
        num_stars=0,
        rating_details=[],
        health_status="unknown",
        last_checked_time=None,
        status="active",
    )


@pytest.fixture
def all_skills():
    return [
        _skill("public-one", VisibilityEnum.PUBLIC),
        _skill("private-alice", VisibilityEnum.PRIVATE, owner="alice"),
        _skill("group-secret", VisibilityEnum.GROUP, owner="alice", groups=["secret-group"]),
    ]


async def _list(monkeypatch, all_skills, user_context):
    svc = SkillService()

    async def _fake_list_skills(include_disabled=False, tag=None):
        return list(all_skills)

    monkeypatch.setattr(svc, "list_skills", _fake_list_skills)
    return await svc.list_skills_for_user(user_context)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSkillDiscoveryGate:
    async def test_no_list_skills_grant_sees_nothing_including_public(
        self, monkeypatch, all_skills
    ):
        """A non-admin with no list_skills grant sees ZERO skills (incl. public)."""
        ctx = {
            "username": "alice",  # even the owner of private/group skills
            "is_admin": False,
            "groups": ["secret-group"],
            "accessible_skills": [],  # no list_skills grant
        }
        result = await _list(monkeypatch, all_skills, ctx)
        assert result == []

    async def test_anonymous_sees_nothing(self, monkeypatch, all_skills):
        """Anonymous (no context) has no grant -> sees nothing (strict parity)."""
        result = await _list(monkeypatch, all_skills, None)
        assert result == []

    async def test_all_grant_then_visibility_applies(self, monkeypatch, all_skills):
        """list_skills:[all] passes the gate; visibility still filters per record."""
        ctx = {
            "username": "bob",  # not the owner; not in the group
            "is_admin": False,
            "groups": [],
            "accessible_skills": ["all"],
        }
        result = await _list(monkeypatch, all_skills, ctx)
        # Gate passes for all; visibility leaves only the public skill for bob.
        assert {s.name for s in result} == {"public-one"}

    async def test_all_grant_owner_sees_own_and_group(self, monkeypatch, all_skills):
        """With the gate open, owner+group visibility returns all three."""
        ctx = {
            "username": "alice",
            "is_admin": False,
            "groups": ["secret-group"],
            "accessible_skills": ["all"],
        }
        result = await _list(monkeypatch, all_skills, ctx)
        assert {s.name for s in result} == {"public-one", "private-alice", "group-secret"}

    async def test_named_grant_gates_by_skill_name(self, monkeypatch, all_skills):
        """A named list_skills grant only surfaces those named (still public-only by visibility)."""
        ctx = {
            "username": "bob",
            "is_admin": False,
            "groups": [],
            "accessible_skills": ["public-one"],  # only this skill discoverable
        }
        result = await _list(monkeypatch, all_skills, ctx)
        assert {s.name for s in result} == {"public-one"}

    async def test_named_grant_for_hidden_skill_still_visibility_gated(
        self, monkeypatch, all_skills
    ):
        """Discovering a name you can't see by visibility yields nothing.

        bob is granted discovery of private-alice, but visibility (private,
        owned by alice) still denies him -- gate is ON TOP of visibility.
        """
        ctx = {
            "username": "bob",
            "is_admin": False,
            "groups": [],
            "accessible_skills": ["private-alice"],
        }
        result = await _list(monkeypatch, all_skills, ctx)
        assert result == []

    async def test_admin_bypasses_gate(self, monkeypatch, all_skills):
        """Admin sees everything regardless of accessible_skills."""
        ctx = {"username": "admin", "is_admin": True, "groups": [], "accessible_skills": []}
        result = await _list(monkeypatch, all_skills, ctx)
        assert {s.name for s in result} == {"public-one", "private-alice", "group-secret"}
