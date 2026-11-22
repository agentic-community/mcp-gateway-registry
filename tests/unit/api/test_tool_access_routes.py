"""GET /api/tool-access: caller tiers, filters, paging (issue #2102).

The last class checks that "can see" agrees with GET /api/servers: for each
group, the tools the real server list returns to a member of only that group
equal the rows that list the group in can_see.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.schemas.tool_access_models import (
    AccessCondition,
    AccessSummary,
    GroupRef,
    ToolAccessRow,
)
from registry.services import tool_access_service
from registry.services.tool_access_service import (
    BuiltRow,
    ListingData,
)
from tests.unit.services.test_tool_access_service import (
    DOCS_PATH,
    OPENAPI_PATH,
    SYNCED_PATH,
    patched_sources,  # noqa: F401  (pytest fixture)
)

ENDPOINT: str = "/api/tool-access"


def _context(
    username: str,
    is_admin: bool = False,
    view_all: bool = False,
) -> dict[str, Any]:
    """A caller context with only the flags the route reads."""
    return {
        "username": username,
        "is_admin": is_admin,
        "ui_permissions": {"view_all": ["all"]} if view_all else {},
        "scopes": [],
        "groups": [],
    }


def _row(
    path: str,
    tool: str,
    server_type: str = "mcp",
    enabled: bool = True,
    group: str = "readers",
    idp_group: str = "idp-readers",
) -> ToolAccessRow:
    return ToolAccessRow(
        tool_name=tool,
        server_name=path.strip("/").title(),
        server_path=path,
        server_type=server_type,
        enabled=enabled,
        can_see=AccessSummary(alone=[group]),
        can_call=AccessSummary(together=[AccessCondition(label="Call", groups=[group])]),
    )


BUILT: list[BuiltRow] = [
    BuiltRow(_row("/alpha/", "a1"), owner="alice"),
    BuiltRow(_row("/alpha/", "a2", group="writers", idp_group="idp-writers"), owner="alice"),
    BuiltRow(_row("/beta/", "b1", server_type="openapi", enabled=False), owner="bob"),
    BuiltRow(_row("/virtual/gamma", "g1", server_type="virtual"), owner="bob"),
]


GROUPS: dict[str, GroupRef] = {
    "readers": GroupRef(group="readers", idp_groups=["idp-readers"]),
    "writers": GroupRef(group="writers", idp_groups=["idp-writers"]),
}


def _client_for(
    context: dict[str, Any],
) -> Iterator[tuple[TestClient, AsyncMock]]:
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    async def fake_build(owner: str | None = None) -> ListingData:
        rows = list(BUILT) if owner is None else [b for b in BUILT if b.owner == owner]
        return ListingData(rows=rows, groups=GROUPS)

    build = AsyncMock(side_effect=fake_build)
    app.dependency_overrides[nginx_proxied_auth] = lambda: context
    with patch.object(tool_access_service, "build_listing", build):
        yield TestClient(app), build
    app.dependency_overrides.clear()


@pytest.fixture
def admin_client():
    yield from _client_for(_context("root", is_admin=True))


def _paths(
    response,
) -> list[tuple[str, str]]:
    return [(r["server_path"], r["tool_name"]) for r in response.json()["rows"]]


@pytest.fixture
def owner_view_owned(monkeypatch):
    """Turn on the owner view (TOOL_ACCESS_OWNER_VIEW=owned)."""
    from registry.core.config import settings

    monkeypatch.setattr(settings, "tool_access_owner_view", "owned")


@pytest.mark.unit
class TestCallerTiers:
    """Who gets which rows."""

    def test_admin_gets_every_row(self, admin_client):
        client, build = admin_client
        body = client.get(ENDPOINT).json()
        assert body["total_count"] == 4
        build.assert_awaited_once_with()

    def test_view_all_holder_gets_every_row(self):
        for client, _ in _client_for(_context("ro", view_all=True)):
            assert client.get(ENDPOINT).json()["total_count"] == 4

    def test_owner_gets_nothing_by_default(self):
        for client, build in _client_for(_context("alice")):
            response = client.get(ENDPOINT)
            assert response.status_code == 200
            assert response.json()["total_count"] == 0
            build.assert_not_awaited()

    def test_view_all_holder_unaffected_by_owner_view(self):
        for client, _ in _client_for(_context("ro", view_all=True)):
            assert client.get(ENDPOINT).json()["total_count"] == 4

    def test_owner_gets_own_rows_only(self, owner_view_owned):
        for client, build in _client_for(_context("alice")):
            response = client.get(ENDPOINT)
            assert _paths(response) == [("/alpha/", "a1"), ("/alpha/", "a2")]
            build.assert_awaited_once_with(owner="alice")

    def test_non_owner_gets_empty_200(self, owner_view_owned):
        for client, _ in _client_for(_context("nobody")):
            response = client.get(ENDPOINT)
            assert response.status_code == 200
            assert response.json()["rows"] == []
            assert response.json()["total_count"] == 0

    def test_caller_without_username_gets_nothing(self, owner_view_owned):
        for client, build in _client_for(_context("")):
            assert client.get(ENDPOINT).json()["total_count"] == 0
            build.assert_not_awaited()

    def test_build_failure_is_a_500_with_a_message(self):
        for client, build in _client_for(_context("root", is_admin=True)):
            build.side_effect = RuntimeError("db down")
            response = client.get(ENDPOINT)
            assert response.status_code == 500
            assert "Could not load tool access" in response.json()["detail"]


@pytest.mark.unit
class TestFilters:
    """Search, filters and paging."""

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"q": "A2"}, [("/alpha/", "a2")]),
            ({"q": "idp-writers"}, [("/alpha/", "a2")]),
            ({"q": "beta"}, [("/beta/", "b1")]),
            ({"server_path": "alpha"}, [("/alpha/", "a1"), ("/alpha/", "a2")]),
            ({"server_type": "virtual"}, [("/virtual/gamma", "g1")]),
            ({"group": "writers"}, [("/alpha/", "a2")]),
            ({"group": "idp-writers"}, [("/alpha/", "a2")]),
            ({"group": "write"}, []),
            (
                {"include_disabled": "false"},
                [("/alpha/", "a1"), ("/alpha/", "a2"), ("/virtual/gamma", "g1")],
            ),
        ],
    )
    def test_filter(self, admin_client, params, expected):
        client, _ = admin_client
        assert _paths(client.get(ENDPOINT, params=params)) == expected

    def test_page_carries_details_only_for_its_groups(self, admin_client):
        client, _ = admin_client
        body = client.get(ENDPOINT, params={"q": "a1"}).json()
        assert body["groups"] == {
            "readers": {"group": "readers", "idp_groups": ["idp-readers"], "is_admin": False}
        }

    def test_default_page_size_is_20(self, admin_client):
        client, _ = admin_client
        assert client.get(ENDPOINT).json()["limit"] == 20

    def test_pages_do_not_overlap(self, admin_client):
        client, _ = admin_client
        first = client.get(ENDPOINT, params={"limit": 3}).json()
        second = client.get(ENDPOINT, params={"limit": 3, "offset": 3}).json()
        assert first["has_next"] is True
        assert second["has_next"] is False
        assert len(first["rows"]) == 3
        assert len(second["rows"]) == 1
        assert first["total_count"] == second["total_count"] == 4

    @pytest.mark.parametrize(
        "params",
        [{"limit": 0}, {"limit": 501}, {"offset": -1}, {"server_type": "bogus"}, {"q": "a" * 201}],
    )
    def test_invalid_query_is_422(self, admin_client, params):
        client, _ = admin_client
        assert client.get(ENDPOINT, params=params).status_code == 422


def _server_list_tools(
    context: dict[str, Any],
    servers: dict[str, dict[str, Any]],
) -> set[tuple[str, str]]:
    """Ask the real GET /api/servers which tools this context sees."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    service = MagicMock()
    service.get_all_servers = AsyncMock(return_value=servers)
    service.get_all_servers_with_permissions = AsyncMock(
        side_effect=lambda allowed, **_kw: {
            path: info
            for path, info in servers.items()
            if "*" in allowed or any(a.strip("/") == path.strip("/") for a in allowed)
        }
    )
    scanner = MagicMock()
    scanner.get_scan_summaries = AsyncMock(return_value={})
    app.dependency_overrides[nginx_proxied_auth] = lambda: context
    try:
        with (
            patch("registry.api.server_routes.server_service", service),
            patch("registry.api.server_routes.security_scanner_service", scanner),
            patch(
                "registry.api.server_routes.annotate_blocked_tools",
                AsyncMock(side_effect=lambda _path, tools: tools),
            ),
        ):
            # status=active forces the full-fetch path that restricted users take.
            response = TestClient(app).get(
                "/api/servers", params={"include_tools": "true", "status": "active", "limit": 2000}
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    seen = set()
    for server in response.json()["servers"]:
        for tool in server.get("tool_list") or []:
            seen.add((server["path"], tool["name"]))
    return seen


@pytest.mark.unit
@pytest.mark.asyncio
class TestCanSeeMatchesServerList:
    """can_see agrees with what GET /api/servers shows each group."""

    async def test_each_group(self, patched_sources):
        from tests.unit.services.test_tool_access_service import _servers

        groups = await tool_access_service._load_groups()
        built = (await tool_access_service.build_listing()).rows
        mcp_paths = {DOCS_PATH, OPENAPI_PATH, SYNCED_PATH}
        for group in groups:
            expected = {
                (b.row.server_path, b.row.tool_name)
                for b in built
                if b.row.server_path in mcp_paths and group.name in b.row.can_see.alone
            }
            context = dict(group.context, username=f"member-of-{group.name}")
            assert _server_list_tools(context, _servers()) == expected, group.name


@pytest.mark.unit
class TestOwnerViewSetting:
    """TOOL_ACCESS_OWNER_VIEW parsing."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("", "none"), ("none", "none"), ("owned", "owned"), (" OWNED ", "owned")],
    )
    def test_values(self, monkeypatch, raw, expected):
        from registry.core.config import Settings

        monkeypatch.setenv("TOOL_ACCESS_OWNER_VIEW", raw)
        assert Settings(_env_file=None).tool_access_owner_view == expected

    def test_default_is_none(self, monkeypatch):
        from registry.core.config import Settings

        monkeypatch.delenv("TOOL_ACCESS_OWNER_VIEW", raising=False)
        assert Settings(_env_file=None).tool_access_owner_view == "none"

    def test_unknown_value_stops_startup(self, monkeypatch):
        from pydantic import ValidationError

        from registry.core.config import Settings

        monkeypatch.setenv("TOOL_ACCESS_OWNER_VIEW", "everyone")
        with pytest.raises(ValidationError, match="tool_access_owner_view"):
            Settings(_env_file=None)
