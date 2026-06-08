"""Tests for ``find_by_identity_url`` on the three entity repository bases.

The default implementations in ``registry/repositories/interfaces.py`` are
the ones exercised here — DocumentDB and file backends inherit them
directly. We use minimal in-memory fakes so the tests don't require
a running database or the full file-backed repo plumbing.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from registry.repositories.interfaces import (
    AgentRepositoryBase,
    ServerRepositoryBase,
    SkillRepositoryBase,
)
from registry.utils.url_normalize import (
    ENTITY_TYPE_AGENT,
    ENTITY_TYPE_SERVER,
    ENTITY_TYPE_SKILL,
    normalize_identity_url,
)


class _FakeServerRepo(ServerRepositoryBase):
    """In-memory fake exposing only the methods needed for the test."""

    def __init__(self, servers: dict[str, dict[str, Any]]):
        self._servers = servers

    async def list_all(self, exclude_tool_list=False) -> dict[str, dict[str, Any]]:
        return dict(self._servers)

    # Abstract methods that are not exercised by these tests — stub out
    # with no-ops so the class can be instantiated.
    async def get(self, path):  # pragma: no cover - unused
        return self._servers.get(path)

    async def list_paginated(self, skip=0, limit=100, exclude_tool_list=False):  # pragma: no cover
        return {}

    async def list_by_ids(self, paths):  # pragma: no cover
        return {p: self._servers[p] for p in paths if p in self._servers}

    async def list_by_source(self, source):  # pragma: no cover
        return {}

    async def create(self, server_info):  # pragma: no cover
        return False

    async def update(self, path, server_info):  # pragma: no cover
        return False

    async def delete(self, path):  # pragma: no cover
        return False

    async def delete_with_versions(self, path):  # pragma: no cover
        return 0

    async def get_state(self, path):  # pragma: no cover
        return False

    async def get_all_states(self):  # pragma: no cover
        return {}

    async def set_state(self, path, enabled):  # pragma: no cover
        return False

    async def load_all(self):  # pragma: no cover
        return None

    async def count(self):  # pragma: no cover
        return len(self._servers)

    async def update_field(self, path, field, value):  # pragma: no cover
        return False

    async def find_with_filter(self, filter_dict, *, limit=None):  # pragma: no cover
        return {}


class _FakeAgentRepo(AgentRepositoryBase):
    def __init__(self, agents: list[Any]):
        self._agents = list(agents)

    async def list_all(self):
        return list(self._agents)

    async def get(self, path):  # pragma: no cover
        for a in self._agents:
            if getattr(a, "path", None) == path:
                return a
        return None

    async def list_paginated(self, skip=0, limit=100):  # pragma: no cover
        return []

    async def create(self, agent):  # pragma: no cover
        return agent

    async def update(self, path, updates):  # pragma: no cover
        return None

    async def delete(self, path):  # pragma: no cover
        return False

    async def get_state(self, path):  # pragma: no cover
        return False

    async def get_all_states(self):  # pragma: no cover
        return {}

    async def set_state(self, path, enabled):  # pragma: no cover
        return False

    async def load_all(self):  # pragma: no cover
        return None

    async def count(self):  # pragma: no cover
        return len(self._agents)

    async def update_field(self, path, field, value):  # pragma: no cover
        return False

    async def find_with_filter(self, filter_dict, *, limit=None):  # pragma: no cover
        return {}


class _FakeSkillRepo(SkillRepositoryBase):
    def __init__(self, skills: list[Any]):
        self._skills = list(skills)

    async def list_all(self, skip=0, limit=100):
        return list(self._skills)

    async def ensure_indexes(self):  # pragma: no cover
        return None

    async def get(self, path):  # pragma: no cover
        for s in self._skills:
            if getattr(s, "path", None) == path:
                return s
        return None

    async def list_paginated(self, skip=0, limit=100):  # pragma: no cover
        return []

    async def list_filtered(self, **kwargs):  # pragma: no cover
        return []

    async def create(self, skill):  # pragma: no cover
        return skill

    async def update(self, path, updates):  # pragma: no cover
        return None

    async def delete(self, path):  # pragma: no cover
        return False

    async def get_state(self, path):  # pragma: no cover
        return False

    async def set_state(self, path, enabled):  # pragma: no cover
        return False

    async def create_many(self, skills):  # pragma: no cover
        return list(skills)

    async def update_many(self, updates):  # pragma: no cover
        return 0

    async def count(self):  # pragma: no cover
        return len(self._skills)


def _agent(path: str, url: str, **extra: Any) -> Any:
    """Build an AgentCard-like object with the minimum the helper inspects."""
    obj = MagicMock()
    obj.path = path
    obj.url = url
    obj.model_dump.return_value = {"path": path, "url": url, **extra}
    return obj


def _skill(path: str, skill_md_url: str, **extra: Any) -> Any:
    """Build a SkillCard-like object with the minimum the helper inspects."""
    obj = MagicMock()
    obj.path = path
    obj.skill_md_url = skill_md_url
    obj.model_dump.return_value = {
        "path": path,
        "skill_md_url": skill_md_url,
        **extra,
    }
    return obj


@pytest.mark.asyncio
class TestServerFindByIdentityUrl:
    async def test_finds_exact_match(self) -> None:
        repo = _FakeServerRepo(
            {
                "/foo": {
                    "server_name": "foo",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
                "/bar": {
                    "server_name": "bar",
                    "proxy_pass_url": "https://other.example.com/mcp",
                },
            }
        )
        identity = normalize_identity_url("https://api.example.com/mcp", ENTITY_TYPE_SERVER)
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/foo"
        assert result["server_name"] == "foo"

    async def test_scheme_collapse(self) -> None:
        """A registration on http://x finds the existing https://x entry."""
        repo = _FakeServerRepo(
            {
                "/foo": {
                    "server_name": "foo",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            }
        )
        identity = normalize_identity_url("http://api.example.com/mcp", ENTITY_TYPE_SERVER)
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/foo"

    async def test_no_match_returns_none(self) -> None:
        repo = _FakeServerRepo(
            {
                "/foo": {
                    "server_name": "foo",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            }
        )
        identity = normalize_identity_url("https://different.example.com/mcp", ENTITY_TYPE_SERVER)
        result = await repo.find_by_identity_url(identity)
        assert result is None

    async def test_skips_servers_without_proxy_url(self) -> None:
        repo = _FakeServerRepo(
            {
                "/foo": {"server_name": "foo", "proxy_pass_url": None},
                "/bar": {
                    "server_name": "bar",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            }
        )
        identity = normalize_identity_url("https://api.example.com/mcp", ENTITY_TYPE_SERVER)
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/bar"

    async def test_empty_identity_returns_none(self) -> None:
        repo = _FakeServerRepo({})
        assert await repo.find_by_identity_url("") is None
        # type: ignore[arg-type] — defensive guard against None at runtime
        assert await repo.find_by_identity_url(None) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestAgentFindByIdentityUrl:
    async def test_finds_exact_match(self) -> None:
        repo = _FakeAgentRepo(
            [
                _agent("/agents/foo", "https://api.example.com/agent"),
                _agent("/agents/bar", "https://other.example.com/agent"),
            ]
        )
        identity = normalize_identity_url("https://api.example.com/agent", ENTITY_TYPE_AGENT)
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/agents/foo"

    async def test_no_match_returns_none(self) -> None:
        repo = _FakeAgentRepo([_agent("/agents/foo", "https://api.example.com/agent")])
        identity = normalize_identity_url("https://nope.example.com/agent", ENTITY_TYPE_AGENT)
        assert await repo.find_by_identity_url(identity) is None

    async def test_skips_agents_without_url(self) -> None:
        repo = _FakeAgentRepo(
            [
                _agent("/agents/foo", ""),
                _agent("/agents/bar", "https://api.example.com/agent"),
            ]
        )
        identity = normalize_identity_url("https://api.example.com/agent", ENTITY_TYPE_AGENT)
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/agents/bar"


@pytest.mark.asyncio
class TestSkillFindByIdentityUrl:
    async def test_finds_exact_github_match(self) -> None:
        repo = _FakeSkillRepo(
            [
                _skill(
                    "/skills/foo",
                    "https://github.com/org/repo/blob/main/.claude/skills/foo/SKILL.md",
                ),
                _skill(
                    "/skills/bar",
                    "https://github.com/org/repo/blob/main/.claude/skills/bar/SKILL.md",
                ),
            ]
        )
        identity = normalize_identity_url(
            "https://github.com/org/repo/blob/main/.claude/skills/foo/SKILL.md",
            ENTITY_TYPE_SKILL,
        )
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/skills/foo"

    async def test_git_suffix_equivalence(self) -> None:
        """Registering ``repo.git`` finds an existing ``repo`` entry."""
        repo = _FakeSkillRepo(
            [
                _skill("/skills/foo", "https://github.com/org/repo"),
            ]
        )
        identity = normalize_identity_url(
            "https://github.com/org/repo.git",
            ENTITY_TYPE_SKILL,
        )
        result = await repo.find_by_identity_url(identity)
        assert result is not None
        assert result["path"] == "/skills/foo"

    async def test_path_case_difference_is_distinct(self) -> None:
        """GitHub paths are case-sensitive — different case is a different skill."""
        repo = _FakeSkillRepo(
            [
                _skill("/skills/foo", "https://github.com/Org/Repo/blob/main/SKILL.md"),
            ]
        )
        identity = normalize_identity_url(
            "https://github.com/org/repo/blob/main/SKILL.md",
            ENTITY_TYPE_SKILL,
        )
        assert await repo.find_by_identity_url(identity) is None
