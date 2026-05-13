"""Integration tests for tool-level access enforcement (Issue #1026).

These tests exercise the four REST endpoints that render tool lists plus
the semantic-search variants under three personas:

- Admin (is_admin=True)
- Restricted consumer (per-server allowlist of one tool)
- Empty allowlist user (explicit empty set)

Rather than bootstrapping the full FastAPI app (which pulls in nginx
config generation, health service, Keycloak, Redis and DocumentDB),
these tests call the endpoint handler functions directly with mocked
module-level service singletons. That isolates the filter application
path while still exercising each endpoint's full handler logic.

Mapped to testing.md sections 1.1.1 through 1.1.12 plus 5.3.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# =============================================================================
# PERSONAS
# =============================================================================


ADMIN_CONTEXT: dict[str, Any] = {
    "username": "admin",
    "is_admin": True,
    "scopes": ["tla-admin"],
    "groups": ["registry-admins"],
    "accessible_servers": ["*"],
    "accessible_services": ["all"],
    "accessible_tools": {"*": {"*"}},
    "accessible_agents": ["all"],
    "ui_permissions": {"list_service": ["all"]},
    "auth_method": "session",
}


RESTRICTED_CONTEXT: dict[str, Any] = {
    "username": "alice",
    "is_admin": False,
    "scopes": ["tla-consumer-restricted"],
    "groups": ["tla-consumer-restricted"],
    "accessible_servers": ["current_time"],
    "accessible_services": ["current_time"],
    "accessible_tools": {"current_time": {"current_time_by_timezone"}},
    "accessible_agents": [],
    "ui_permissions": {"list_service": ["current_time"]},
    "auth_method": "session",
}


EMPTY_CONTEXT: dict[str, Any] = {
    "username": "empty-user",
    "is_admin": False,
    "scopes": ["tla-consumer-empty"],
    "groups": ["tla-consumer-empty"],
    "accessible_servers": ["current_time"],
    "accessible_services": ["current_time"],
    "accessible_tools": {"current_time": set()},
    "accessible_agents": [],
    "ui_permissions": {"list_service": ["current_time"]},
    "auth_method": "session",
}


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def current_time_server_info() -> dict[str, Any]:
    """Server info dict for the `current_time` backend with three tools."""
    return {
        "id": "srv-current-time",
        "server_name": "current_time",
        "path": "/current_time/",
        "description": "Time helpers",
        "proxy_pass_url": "http://current-time:8000/",
        "is_enabled": True,
        "tags": ["time"],
        "num_tools": 3,
        "license": "MIT",
        "health_status": "healthy",
        "metadata": {},
        "version": "v1.0.0",
        "status": "active",
        "tool_list": [
            {"name": "current_time_by_timezone", "description": "tz"},
            {"name": "current_time_utc", "description": "utc"},
            {"name": "current_time_epoch", "description": "epoch"},
        ],
        "is_active": True,
    }


@pytest.fixture
def fininfo_server_info() -> dict[str, Any]:
    """Second backend server with two distinct tools."""
    return {
        "id": "srv-fininfo",
        "server_name": "fininfo",
        "path": "/fininfo/",
        "description": "Financial info",
        "proxy_pass_url": "http://fininfo:8000/",
        "is_enabled": True,
        "tags": ["finance"],
        "num_tools": 2,
        "license": "MIT",
        "health_status": "healthy",
        "metadata": {},
        "version": "v1.0.0",
        "status": "active",
        "tool_list": [
            {"name": "get_stock", "description": "stock"},
            {"name": "get_forex", "description": "forex"},
        ],
        "is_active": True,
    }


@pytest.fixture
def all_servers_map(
    current_time_server_info, fininfo_server_info
) -> dict[str, dict[str, Any]]:
    """Map of path -> server_info used by get_all_servers mocks."""
    return {
        "/current_time/": current_time_server_info,
        "/fininfo/": fininfo_server_info,
    }


# =============================================================================
# /tools/{service_path} - restricted and unauthorized paths
# =============================================================================


class TestGetServiceToolsEndpoint:
    """Exercises the handler for GET /tools/{service_path}."""

    @pytest.mark.asyncio
    async def test_get_tools_service_path_restricted_user_prunes_list(
        self, current_time_server_info
    ):
        """Restricted user receives only the tools in their allowlist."""
        from registry.api.server_routes import get_service_tools

        # Arrange: patch module-level singletons the handler imports.
        fake_server_service = AsyncMock()
        fake_server_service.get_server_info = AsyncMock(return_value=current_time_server_info)
        fake_server_service.user_can_access_server_path = AsyncMock(return_value=True)
        fake_server_service.is_service_enabled = AsyncMock(return_value=True)
        fake_server_service.update_server = AsyncMock(return_value=True)

        fake_mcp = AsyncMock()
        fake_mcp.get_tools_from_server_with_server_info = AsyncMock(
            return_value=current_time_server_info["tool_list"]
        )

        fake_faiss = AsyncMock()
        fake_faiss.add_or_update_service = AsyncMock()

        with (
            patch("registry.api.server_routes.server_service", fake_server_service),
            patch("registry.core.mcp_client.mcp_client_service", fake_mcp),
            patch("registry.search.service.faiss_service", fake_faiss),
        ):
            # Act
            result = await get_service_tools("current_time", RESTRICTED_CONTEXT)

        # Assert
        assert result["cached"] is False
        names = [t["name"] for t in result["tools"]]
        assert names == ["current_time_by_timezone"]

    @pytest.mark.asyncio
    async def test_get_tools_service_path_unauthorized_server_returns_403(
        self, fininfo_server_info
    ):
        """Restricted user gets 403 for a server they cannot access."""
        from fastapi import HTTPException

        from registry.api.server_routes import get_service_tools

        # Arrange
        fake_server_service = AsyncMock()
        fake_server_service.get_server_info = AsyncMock(return_value=fininfo_server_info)
        fake_server_service.user_can_access_server_path = AsyncMock(return_value=False)

        with patch("registry.api.server_routes.server_service", fake_server_service):
            # Act / Assert
            with pytest.raises(HTTPException) as excinfo:
                await get_service_tools("fininfo", RESTRICTED_CONTEXT)
            assert excinfo.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_tools_service_path_cached_fallback_is_also_filtered(
        self, current_time_server_info
    ):
        """When live fetch fails and we fall back to cached tools, the filter still runs."""
        from registry.api.server_routes import get_service_tools

        # Arrange
        fake_server_service = AsyncMock()
        fake_server_service.get_server_info = AsyncMock(return_value=current_time_server_info)
        fake_server_service.user_can_access_server_path = AsyncMock(return_value=True)
        fake_server_service.is_service_enabled = AsyncMock(return_value=True)

        fake_mcp = AsyncMock()
        # Live fetch returns None -> handler falls back to cached tool_list.
        fake_mcp.get_tools_from_server_with_server_info = AsyncMock(return_value=None)

        fake_faiss = AsyncMock()

        with (
            patch("registry.api.server_routes.server_service", fake_server_service),
            patch("registry.core.mcp_client.mcp_client_service", fake_mcp),
            patch("registry.search.service.faiss_service", fake_faiss),
        ):
            # Act
            result = await get_service_tools("current_time", RESTRICTED_CONTEXT)

        # Assert
        assert result["cached"] is True
        names = [t["name"] for t in result["tools"]]
        assert names == ["current_time_by_timezone"]


# =============================================================================
# /tools/all
# =============================================================================


class TestGetToolsAllEndpoint:
    """Exercises the special `/all` branch of GET /tools/{service_path}."""

    @pytest.mark.asyncio
    async def test_get_tools_all_prunes_per_server_and_skips_empty_servers(
        self, all_servers_map
    ):
        """Restricted user: only allowed tools appear; fully-filtered servers are skipped."""
        from registry.api.server_routes import get_service_tools

        # Arrange
        fake_server_service = AsyncMock()
        fake_server_service.get_all_servers_with_permissions = AsyncMock(
            return_value=all_servers_map
        )
        fake_server_service.get_all_servers = AsyncMock(return_value=all_servers_map)

        fake_mcp = AsyncMock()
        fake_faiss = AsyncMock()

        with (
            patch("registry.api.server_routes.server_service", fake_server_service),
            patch("registry.core.mcp_client.mcp_client_service", fake_mcp),
            patch("registry.search.service.faiss_service", fake_faiss),
        ):
            # Act
            result = await get_service_tools("all", RESTRICTED_CONTEXT)

        # Assert
        names = sorted(t["name"] for t in result["tools"])
        assert names == ["current_time_by_timezone"]
        # fininfo had zero allowed tools, so it must be skipped entirely.
        assert "/fininfo/" not in result["servers"]
        # current_time aggregation retains the single allowed tool.
        ct_names = [t["name"] for t in result["servers"]["/current_time/"]]
        assert ct_names == ["current_time_by_timezone"]

    @pytest.mark.asyncio
    async def test_get_tools_all_admin_sees_full_aggregate(self, all_servers_map):
        """Admin persona sees every tool on every server."""
        from registry.api.server_routes import get_service_tools

        # Arrange
        fake_server_service = AsyncMock()
        fake_server_service.get_all_servers = AsyncMock(return_value=all_servers_map)
        fake_server_service.get_all_servers_with_permissions = AsyncMock(
            return_value=all_servers_map
        )

        fake_mcp = AsyncMock()
        fake_faiss = AsyncMock()

        with (
            patch("registry.api.server_routes.server_service", fake_server_service),
            patch("registry.core.mcp_client.mcp_client_service", fake_mcp),
            patch("registry.search.service.faiss_service", fake_faiss),
        ):
            # Act
            result = await get_service_tools("all", ADMIN_CONTEXT)

        # Assert
        names = sorted(t["name"] for t in result["tools"])
        assert names == [
            "current_time_by_timezone",
            "current_time_epoch",
            "current_time_utc",
            "get_forex",
            "get_stock",
        ]


# =============================================================================
# /servers (direct filter_tools_for_user integration)
# =============================================================================


class TestGetServersEndpointFilter:
    """Validates the filter application at the /servers response builder layer.

    The full `get_servers_json` handler pulls in nginx, health service and
    pagination machinery that is out of scope here. These tests exercise
    the same call that the handler makes (`filter_tools_for_user` on the
    cached `tool_list`) to lock in the behaviour from the caller's view.
    """

    def test_get_servers_admin_sees_all_tools_and_num_tools_matches(
        self, current_time_server_info
    ):
        """Admin sees the full tool_list; num_tools equals the list length."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Act
        filtered = filter_tools_for_user(
            current_time_server_info["server_name"],
            current_time_server_info["tool_list"],
            ADMIN_CONTEXT,
            endpoint="servers",
        )

        # Assert
        assert len(filtered) == 3
        assert [t["name"] for t in filtered] == [
            "current_time_by_timezone",
            "current_time_utc",
            "current_time_epoch",
        ]

    def test_get_servers_restricted_user_sees_only_allowed_and_num_tools_recomputed(
        self, current_time_server_info
    ):
        """Restricted user: num_tools equals the filtered list length."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Act
        filtered = filter_tools_for_user(
            current_time_server_info["server_name"],
            current_time_server_info["tool_list"],
            RESTRICTED_CONTEXT,
            endpoint="servers",
        )

        # Assert
        assert len(filtered) == 1
        assert filtered[0]["name"] == "current_time_by_timezone"

    def test_get_servers_empty_allowlist_returns_zero_tools(
        self, current_time_server_info
    ):
        """Empty allowlist user sees num_tools==0 on an otherwise accessible server."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Act
        filtered = filter_tools_for_user(
            current_time_server_info["server_name"],
            current_time_server_info["tool_list"],
            EMPTY_CONTEXT,
            endpoint="servers",
        )

        # Assert
        assert filtered == []


# =============================================================================
# Semantic search
# =============================================================================


class TestSemanticSearchFilter:
    """Validates filter application at the three semantic-search sites.

    As with /servers above, these tests call the same filter the handler
    calls rather than booting the FAISS + search repo machinery.
    """

    def test_semantic_search_per_server_matching_tools_pruned(
        self, current_time_server_info
    ):
        """matching_tools from FAISS is pruned to the user's allowlist."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Arrange: FAISS-shaped entries use `tool_name`.
        raw_matching = [
            {"tool_name": "current_time_by_timezone", "relevance_score": 0.9},
            {"tool_name": "current_time_utc", "relevance_score": 0.8},
        ]

        # Act
        allowed = filter_tools_for_user(
            current_time_server_info["server_name"],
            raw_matching,
            RESTRICTED_CONTEXT,
            endpoint="semantic_search",
        )

        # Assert
        assert len(allowed) == 1
        assert allowed[0]["tool_name"] == "current_time_by_timezone"

    def test_semantic_search_server_num_tools_recomputed_after_prune(
        self, current_time_server_info
    ):
        """Blocker 3: num_tools is the length of the filtered full tool_list."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Act
        allowed_full = filter_tools_for_user(
            current_time_server_info["server_name"],
            current_time_server_info["tool_list"],
            RESTRICTED_CONTEXT,
            endpoint="semantic_search",
        )

        # Assert
        assert len(allowed_full) == 1

    def test_semantic_search_top_level_tool_results_pruned(self):
        """Top-level tool results are filtered via tool_allowed_for_user."""
        from registry.auth.tool_filter import tool_allowed_for_user

        # Arrange
        raw_tools = [
            {
                "tool_name": "current_time_by_timezone",
                "server_name": "current_time",
                "server_path": "/current_time/",
            },
            {
                "tool_name": "current_time_utc",
                "server_name": "current_time",
                "server_path": "/current_time/",
            },
        ]

        # Act: mimic the loop in search_routes.semantic_search that filters
        # top-level ToolSearchResult entries.
        kept = [
            tool
            for tool in raw_tools
            if tool_allowed_for_user(
                tool["server_name"], tool["tool_name"], RESTRICTED_CONTEXT
            )
        ]

        # Assert
        assert len(kept) == 1
        assert kept[0]["tool_name"] == "current_time_by_timezone"

    def test_semantic_search_virtual_server_matching_tools_pruned_and_num_tools_recomputed(
        self, current_time_server_info
    ):
        """Virtual server matching_tools are filtered and num_tools recomputed."""
        from registry.auth.tool_filter import filter_tools_for_user

        # Arrange: virtual server wraps current_time; matching_tools came
        # from the FAISS search and point at the underlying server name.
        vs_matching = [
            {"tool_name": "current_time_by_timezone", "relevance_score": 0.95},
            {"tool_name": "current_time_utc", "relevance_score": 0.80},
        ]
        # The virtual server's metadata.tool_list mirrors current_time's list.
        vs_tool_list = current_time_server_info["tool_list"]

        # Act
        allowed_vs_matching = filter_tools_for_user(
            "current_time", vs_matching, RESTRICTED_CONTEXT, endpoint="semantic_search"
        )
        allowed_vs_full = filter_tools_for_user(
            "current_time",
            vs_tool_list,
            RESTRICTED_CONTEXT,
            endpoint="semantic_search",
        )

        # Assert
        assert [t["tool_name"] for t in allowed_vs_matching] == [
            "current_time_by_timezone"
        ]
        assert len(allowed_vs_full) == 1

    def test_semantic_search_admin_no_filtering_applied(
        self, current_time_server_info
    ):
        """Admin requests leave all FAISS entries intact."""
        from registry.auth.tool_filter import filter_tools_for_user, tool_allowed_for_user

        # Arrange
        raw_matching = [
            {"tool_name": "current_time_by_timezone"},
            {"tool_name": "current_time_utc"},
            {"tool_name": "current_time_epoch"},
        ]

        # Act
        allowed = filter_tools_for_user(
            current_time_server_info["server_name"],
            raw_matching,
            ADMIN_CONTEXT,
            endpoint="semantic_search",
        )
        top_level_kept = [
            t
            for t in raw_matching
            if tool_allowed_for_user("current_time", t["tool_name"], ADMIN_CONTEXT)
        ]

        # Assert
        assert allowed == raw_matching
        assert top_level_kept == raw_matching
