"""Unit tests for the list_skills DISCOVERY gate on the ARD skill branches.

The ARD adapter's server and agent branches enforce the type-level discovery
gate (``accessible_servers`` / ``accessible_agents`` baked into their visibility
helpers). The skill branch in both ``search_and_scope`` and ``browse`` called
``user_can_access_skill`` -- a visibility-only check -- with no preceding
``list_skills`` gate, so a caller with no skill-discovery grant could still
enumerate every skill (defaulting to PUBLIC) through the ARD catalog.

These tests assert the skill branch now applies the same ``list_skills``
discovery gate (via ``user_has_asset_permission``) before the visibility check,
matching the server/agent branches and ``search_routes``. Fails closed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from registry.services import ard_search_service as s


def _skill_hit(path: str, skill_name: str) -> dict:
    """A PUBLIC skill search hit (visibility allows everyone)."""
    return {
        "path": path,
        "skill_name": skill_name,
        "visibility": "public",
        "owner": "alice",
        "allowed_groups": [],
        "tags": [],
        "relevance_score": 0.9,
        "description": "d",
    }


def _skill_doc(name: str, path: str) -> SimpleNamespace:
    """A PUBLIC skill repo document (SkillCard-shaped) for the browse path.

    Mirrors the real model: the canonical discovery key is ``name`` (SkillCard
    has no ``skill_name`` attribute), which is what the ``list_skills`` grant is
    keyed on.
    """
    return SimpleNamespace(
        path=path,
        name=name,
        visibility="public",
        owner="alice",
        allowed_groups=[],
        tags=[],
        tools=[],
        version=None,
        updated_at=None,
        description="d",
    )


def _ctx(*, is_admin: bool = False, list_skills: list[str] | None = None) -> dict:
    ui_permissions: dict[str, list[str]] = {}
    if list_skills is not None:
        ui_permissions["list_skills"] = list_skills
    return {"username": "bob", "is_admin": is_admin, "groups": [], "ui_permissions": ui_permissions}


def _raw_two_skills() -> dict:
    return {
        "servers": [],
        "agents": [],
        "skills": [
            _skill_hit("/skills/alpha", "alpha"),
            _skill_hit("/skills/beta", "beta"),
        ],
    }


async def _search(user_context: dict):
    mock_repo = AsyncMock()
    mock_repo.search = AsyncMock(return_value=_raw_two_skills())
    with (
        patch.object(s, "get_search_repository", return_value=mock_repo),
        patch.object(s, "_resolve_publisher_domain", return_value="reg.example.com"),
        patch.object(s, "_build_origin_map", AsyncMock(return_value=({}, []))),
    ):
        return await s.search_and_scope(
            "q", None, None, 10, user_context, "http://h/api/ard/search"
        )


async def _browse(user_context: dict):
    mock_repo = AsyncMock()
    mock_repo.list_filtered = AsyncMock(
        return_value=[_skill_doc("alpha", "/skills/alpha"), _skill_doc("beta", "/skills/beta")]
    )
    with (
        patch.object(s, "get_skill_repository", return_value=mock_repo),
        patch.object(s, "_resolve_publisher_domain", return_value="reg.example.com"),
    ):
        return await s.browse(["type=skill"], "identifier", 0, 10, user_context, "http://h")


@pytest.mark.unit
@pytest.mark.asyncio
class TestArdSearchSkillDiscoveryGate:
    async def test_no_grant_scopes_out_all_public_skills(self):
        """No list_skills grant -> zero skill results, both counted as scoped."""
        results, scoped_out, _ = await _search(_ctx(list_skills=[]))
        assert results == []
        assert scoped_out == 2

    async def test_all_grant_returns_all(self):
        """list_skills:[all] passes the gate for both PUBLIC skills."""
        results, scoped_out, _ = await _search(_ctx(list_skills=["all"]))
        assert {r.display_name for r in results} == {"alpha", "beta"}
        assert scoped_out == 0

    async def test_named_grant_returns_only_named(self):
        """A named grant surfaces only that skill; the other is scoped out."""
        results, scoped_out, _ = await _search(_ctx(list_skills=["alpha"]))
        assert {r.display_name for r in results} == {"alpha"}
        assert scoped_out == 1

    async def test_admin_bypasses_gate(self):
        """Admin sees all skills regardless of ui_permissions."""
        results, scoped_out, _ = await _search(_ctx(is_admin=True, list_skills=[]))
        assert {r.display_name for r in results} == {"alpha", "beta"}
        assert scoped_out == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestArdBrowseSkillDiscoveryGate:
    @staticmethod
    def _leaves(entries) -> set[str]:
        """The trailing path segment of each entry's record url (browse maps a
        skill's display name from its path, so assert on the stable url leaf)."""
        return {e.url.rsplit("/", 1)[-1] for e in entries}

    async def test_no_grant_hides_all_public_skills(self):
        """No list_skills grant -> browse returns zero skills."""
        entries, total = await _browse(_ctx(list_skills=[]))
        assert entries == []
        assert total == 0

    async def test_all_grant_returns_all(self):
        entries, total = await _browse(_ctx(list_skills=["all"]))
        assert self._leaves(entries) == {"alpha", "beta"}
        assert total == 2

    async def test_named_grant_returns_only_named(self):
        entries, total = await _browse(_ctx(list_skills=["beta"]))
        assert self._leaves(entries) == {"beta"}
        assert total == 1

    async def test_admin_bypasses_gate(self):
        entries, total = await _browse(_ctx(is_admin=True, list_skills=[]))
        assert self._leaves(entries) == {"alpha", "beta"}
        assert total == 2
