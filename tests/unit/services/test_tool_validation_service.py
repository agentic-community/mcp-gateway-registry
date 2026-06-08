"""Unit tests for registry/services/tool_validation_service.py."""

from unittest.mock import AsyncMock

import pytest

from registry.schemas.skill_models import SkillCard, ToolReference
from registry.services.tool_validation_service import (
    ToolValidationService,
    get_tool_validation_service,
)


def _make_skill(tools: list[ToolReference] | None) -> SkillCard:
    return SkillCard(
        name="test-skill",
        description="A test skill",
        path="/skills/test-skill",
        skill_md_url="https://example.com/SKILL.md",
        allowed_tools=tools or [],
    )


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.list_all = AsyncMock(return_value={})
    repo.get_all_states = AsyncMock(return_value={})
    return repo


@pytest.fixture
def service(mock_repo):
    svc = ToolValidationService()
    svc._server_repo = mock_repo
    return svc


class TestValidateToolsAvailable:
    async def test_no_required_tools_returns_available(self, service, mock_repo):
        result = await service.validate_tools_available(_make_skill(None))
        assert result.all_available is True
        assert result.missing_tools == []
        mock_repo.list_all.assert_not_called()

    async def test_all_tools_found(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"tool_list": [{"name": "Read"}, {"name": "Write"}]},
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": True}

        skill = _make_skill([ToolReference(tool_name="Read")])
        result = await service.validate_tools_available(skill)

        assert result.all_available is True
        assert result.available_tools == ["Read"]
        assert result.mcp_servers_required == ["/servers/fs"]

    async def test_missing_tool_reported(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"tool_list": [{"name": "Read"}]},
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": True}

        skill = _make_skill([ToolReference(tool_name="Bash")])
        result = await service.validate_tools_available(skill)

        assert result.all_available is False
        assert result.missing_tools == ["Bash"]

    async def test_disabled_server_skipped_when_enabled_only(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"tool_list": [{"name": "Read"}]},
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": False}

        skill = _make_skill([ToolReference(tool_name="Read")])
        result = await service.validate_tools_available(skill, enabled_only=True)

        assert result.all_available is False
        assert "Read" in result.missing_tools

    async def test_disabled_server_used_when_not_enabled_only(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"tool_list": [{"name": "Read"}]},
        }

        skill = _make_skill([ToolReference(tool_name="Read")])
        result = await service.validate_tools_available(skill, enabled_only=False)

        assert result.all_available is True
        mock_repo.get_all_states.assert_not_called()

    async def test_tool_without_name_ignored(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"tool_list": [{"name": ""}, {"name": "Read"}]},
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": True}

        skill = _make_skill([ToolReference(tool_name="Read")])
        result = await service.validate_tools_available(skill)
        assert result.all_available is True


class TestGetToolsWithServers:
    async def test_tool_mapped_to_servers(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {
                "server_name": "Filesystem",
                "tool_list": [{"name": "Read"}],
            },
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": True}

        result = await service.get_tools_with_servers([ToolReference(tool_name="Read")])

        assert len(result) == 1
        assert result[0]["tool_name"] == "Read"
        assert result[0]["servers"][0]["path"] == "/servers/fs"
        assert result[0]["servers"][0]["name"] == "Filesystem"
        assert result[0]["servers"][0]["is_enabled"] is True

    async def test_tool_not_found_has_empty_servers(self, service, mock_repo):
        mock_repo.list_all.return_value = {
            "/servers/fs": {"server_name": "Filesystem", "tool_list": [{"name": "Read"}]},
        }
        mock_repo.get_all_states.return_value = {"/servers/fs": True}

        result = await service.get_tools_with_servers([ToolReference(tool_name="Bash")])
        assert result[0]["servers"] == []

    async def test_empty_tool_refs(self, service, mock_repo):
        result = await service.get_tools_with_servers([])
        assert result == []


class TestGetServerRepoLazyInit:
    async def test_lazy_init_uses_factory(self, monkeypatch):
        sentinel = AsyncMock()
        monkeypatch.setattr(
            "registry.services.tool_validation_service.get_server_repository",
            lambda: sentinel,
        )
        svc = ToolValidationService()
        assert svc._get_server_repo() is sentinel
        assert svc._get_server_repo() is sentinel


class TestSingleton:
    def test_returns_same_instance(self):
        first = get_tool_validation_service()
        second = get_tool_validation_service()
        assert first is second
