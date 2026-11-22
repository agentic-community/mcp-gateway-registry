"""Tool access rows: who can see and who can call each tool (issue #2102)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.schemas.tool_access_models import ToolAccessRow
from registry.schemas.virtual_server_models import (
    ToolMapping,
    VirtualServerConfig,
)
from registry.services import tool_access_service

DOCS_PATH: str = "/docs/"
OPENAPI_PATH: str = "/petstore/"
SYNCED_PATH: str = "/peer-docs/"
VIRTUAL_PATH: str = "/virtual/eng"

SERVER_RULES: dict[str, list[dict]] = {
    "admins": [{"server": "*", "methods": ["all"], "tools": ["*"]}],
    "ro-admins": [],
    "docs-readers": [{"server": "docs", "methods": ["initialize"], "tools": ["search"]}],
    "docs-callers": [{"server": "docs", "methods": [], "tools": ["install"]}],
    "session-only": [{"server": "docs", "methods": ["initialize"], "tools": []}],
    "eng-users": [],
    "virtual-users": [
        {"server": "virtual/eng", "methods": ["initialize"], "tools": ["docs_search"]},
    ],
}

UI_PERMISSIONS: dict[str, dict] = {
    "admins": {"register_service": ["all"], "list_service": ["all"]},
    "ro-admins": {"view_all": ["all"]},
    "docs-readers": {"list_service": ["docs"]},
    "docs-callers": {},
    "session-only": {},
    "eng-users": {},
    "virtual-users": {"list_virtual_server": [VIRTUAL_PATH]},
}

GROUP_MAPPINGS: dict[str, list[str]] = {
    "admins": ["mcp-registry-admin"],
    "docs-readers": ["eng-docs"],
}


def _servers() -> dict[str, dict[str, Any]]:
    """An MCP server, an OpenAPI MCP server and a server synced from a peer."""
    return {
        DOCS_PATH: {
            "path": DOCS_PATH,
            "server_name": "Docs",
            "is_enabled": True,
            "registered_by": "alice",
            "tool_list": [{"name": "search"}, {"name": "install"}],
            "tool_overrides": {"install": {"blocked": True, "source": "admin"}},
            "proxy_pass_url": "http://internal-docs:8000/mcp",
        },
        OPENAPI_PATH: {
            "path": OPENAPI_PATH,
            "server_name": "Petstore",
            "is_enabled": False,
            "registered_by": "carol",
            "metadata": {"openapi_server": {"slug": "petstore"}},
            "tool_list": [{"name": "list_pets"}],
        },
        SYNCED_PATH: {
            "path": SYNCED_PATH,
            "server_name": "Peer docs",
            "is_enabled": True,
            "registered_by": "alice",
            "sync_metadata": {"source_peer": "peer-a"},
            "tool_list": [{"name": "lookup"}],
        },
    }


def _virtual() -> VirtualServerConfig:
    """A virtual server exposing docs' search as docs_search."""
    return VirtualServerConfig(
        path=VIRTUAL_PATH,
        server_name="Engineering tools",
        tool_mappings=[
            ToolMapping(tool_name="search", alias="docs_search", backend_server_path=DOCS_PATH),
        ],
        required_scopes=["eng-users"],
        created_by="bob",
    )


@pytest.fixture
def patched_sources():
    """Patch the scope, server and virtual server reads the service makes."""
    scope_repo = MagicMock()
    scope_repo.list_scope_names = AsyncMock(return_value=list(SERVER_RULES))
    scope_repo.get_server_scopes_bulk = AsyncMock(return_value=SERVER_RULES)
    scope_repo.get_ui_scopes_bulk = AsyncMock(return_value=UI_PERMISSIONS)
    scope_repo.get_all_group_mappings = AsyncMock(return_value=GROUP_MAPPINGS)
    servers = MagicMock()
    servers.get_all_servers = AsyncMock(side_effect=lambda: _servers())
    virtual = MagicMock()
    virtual.list_virtual_server_configs = AsyncMock(return_value=[_virtual()])
    tool_access_service.clear_cache()
    with (
        patch.object(tool_access_service, "get_scope_repository", return_value=scope_repo),
        patch.object(tool_access_service, "server_service", servers),
        patch.object(tool_access_service, "get_virtual_server_service", return_value=virtual),
    ):
        yield scope_repo, servers, virtual
    tool_access_service.clear_cache()


async def _rows(
    owner: str | None = None,
) -> dict[tuple[str, str], ToolAccessRow]:
    tool_access_service.clear_cache()
    data = await tool_access_service.build_listing(owner=owner)
    return {(b.row.server_path, b.row.tool_name): b.row for b in data.rows}


def _names(
    names: list[str],
) -> list[str]:
    return list(names)


@pytest.mark.unit
@pytest.mark.asyncio
class TestServerRows:
    """MCP and OpenAPI MCP server rows."""

    async def test_one_row_per_tool_on_every_server_type(self, patched_sources):
        rows = await _rows()
        assert set(rows) == {
            (DOCS_PATH, "search"),
            (DOCS_PATH, "install"),
            (OPENAPI_PATH, "list_pets"),
            (SYNCED_PATH, "lookup"),
            (VIRTUAL_PATH, "docs_search"),
        }
        assert rows[(OPENAPI_PATH, "list_pets")].server_type == "openapi"
        assert rows[(OPENAPI_PATH, "list_pets")].enabled is False
        assert rows[(DOCS_PATH, "search")].server_type == "mcp"

    async def test_call_alone_lists_groups_with_session_and_tool(self, patched_sources):
        row = (await _rows())[(DOCS_PATH, "search")]
        assert _names(row.can_call.alone) == ["admins", "docs-readers"]
        assert row.can_call.together == []

    async def test_groups_needed_together(self, patched_sources):
        row = (await _rows())[(DOCS_PATH, "install")]
        assert _names(row.can_call.alone) == ["admins"]
        session, call = row.can_call.together
        assert "initialize" in session.label
        assert _names(session.groups) == ["docs-readers", "session-only"]
        assert _names(call.groups) == ["docs-callers"]

    async def test_blocked_tool_is_marked(self, patched_sources):
        row = (await _rows())[(DOCS_PATH, "install")]
        assert row.blocked is True
        assert row.blocked_source == "admin"
        assert (await _rows())[(DOCS_PATH, "search")].blocked is False

    async def test_can_see_follows_registry_reads(self, patched_sources):
        row = (await _rows())[(DOCS_PATH, "search")]
        assert _names(row.can_see.alone) == ["admins", "docs-readers", "ro-admins"]

    async def test_view_all_sees_but_never_calls(self, patched_sources):
        for row in (await _rows()).values():
            assert "ro-admins" not in _names(row.can_call.alone)
            for condition in row.can_call.together:
                assert "ro-admins" not in _names(condition.groups)

    async def test_idp_groups_and_admin_flag(self, patched_sources):
        groups = (await tool_access_service.build_listing()).groups
        assert groups["docs-readers"].idp_groups == ["eng-docs"]
        assert groups["admins"].is_admin is True
        assert groups["docs-readers"].is_admin is False

    async def test_rows_carry_no_backend_url(self, patched_sources):
        for row in (await _rows()).values():
            assert "internal-docs" not in row.model_dump_json()
            assert "proxy_pass_url" not in row.model_dump()


@pytest.mark.unit
@pytest.mark.asyncio
class TestVirtualRows:
    """A tool reached through a virtual server."""

    async def test_alias_and_backend_tool(self, patched_sources):
        row = (await _rows())[(VIRTUAL_PATH, "docs_search")]
        assert row.server_type == "virtual"
        assert row.backend_server_path == DOCS_PATH
        assert row.backend_tool_name == "search"

    async def test_no_group_calls_it_alone(self, patched_sources):
        row = (await _rows())[(VIRTUAL_PATH, "docs_search")]
        assert row.can_call.alone == []

    async def test_together_lists_every_gateway_check(self, patched_sources):
        row = (await _rows())[(VIRTUAL_PATH, "docs_search")]
        by_label = {c.label: _names(c.groups) for c in row.can_call.together}
        assert by_label == {
            f"Open a session on {VIRTUAL_PATH} (initialize)": ["admins", "virtual-users"],
            f"Call docs_search on {VIRTUAL_PATH}": ["admins", "virtual-users"],
            f"Open a session on backend {DOCS_PATH} (initialize)": [
                "admins",
                "docs-readers",
                "session-only",
            ],
            f"Call search on backend {DOCS_PATH}": ["admins", "docs-readers"],
            "Member of eng-users": ["eng-users"],
        }

    async def test_blocked_backend_tool_marks_row(self, patched_sources):
        _, servers, _ = patched_sources
        changed = _servers()
        changed[DOCS_PATH]["tool_overrides"] = {
            "search": {"blocked": True, "source": "security_scan"}
        }
        servers.get_all_servers = AsyncMock(return_value=changed)
        row = (await _rows())[(VIRTUAL_PATH, "docs_search")]
        assert row.blocked is True
        assert row.blocked_source == "security_scan"


@pytest.mark.unit
@pytest.mark.asyncio
class TestOwnerAndQueries:
    """Owner-only builds and the number of reads per build."""

    async def test_owner_gets_only_their_servers(self, patched_sources):
        rows = await _rows(owner="alice")
        assert {path for path, _ in rows} == {DOCS_PATH}

    async def test_synced_server_has_no_local_owner(self, patched_sources):
        rows = await _rows(owner="alice")
        assert (SYNCED_PATH, "lookup") not in rows

    async def test_virtual_owner_is_created_by(self, patched_sources):
        rows = await _rows(owner="bob")
        assert set(rows) == {(VIRTUAL_PATH, "docs_search")}

    async def test_unknown_owner_gets_nothing(self, patched_sources):
        assert await _rows(owner="nobody") == {}

    async def test_each_source_read_once(self, patched_sources):
        scope_repo, servers, virtual = patched_sources
        await tool_access_service.build_listing()
        await tool_access_service.build_listing()
        scope_repo.list_scope_names.assert_awaited_once()
        scope_repo.get_server_scopes_bulk.assert_awaited_once()
        scope_repo.get_ui_scopes_bulk.assert_awaited_once()
        scope_repo.get_all_group_mappings.assert_awaited_once()
        servers.get_all_servers.assert_awaited_once()
        virtual.list_virtual_server_configs.assert_awaited_once()

    async def test_cache_expires(self, patched_sources, monkeypatch):
        scope_repo, _, _ = patched_sources
        await tool_access_service.build_listing()
        monkeypatch.setattr(tool_access_service, "CACHE_SECONDS", -1.0)
        await tool_access_service.build_listing()
        assert scope_repo.list_scope_names.await_count == 2

    async def test_index_matches_full_scan(self, patched_sources):
        from registry.common.tool_grants import server_names_match

        groups = await tool_access_service._load_groups()
        for path in (DOCS_PATH, VIRTUAL_PATH, "/elsewhere/", "*"):
            found = tool_access_service._rules_for_server(groups, path)
            for group in groups:
                expected = [r for r in group.rules if server_names_match(r.get("server"), path)]
                assert sorted(map(str, found.rules_by_group[group.name])) == sorted(
                    map(str, expected)
                )

    async def test_no_groups_gives_rows_with_no_access(self, patched_sources):
        scope_repo, _, _ = patched_sources
        scope_repo.list_scope_names = AsyncMock(return_value=[])
        row = (await _rows())[(DOCS_PATH, "search")]
        assert row.can_see.alone == []
        assert row.can_call.alone == []
