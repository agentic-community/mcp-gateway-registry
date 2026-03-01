"""Unit tests for virtual server service layer."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from registry.exceptions import (
    VirtualServerAlreadyExistsError,
    VirtualServerNotFoundError,
    VirtualServerValidationError,
)
from registry.schemas.virtual_server_models import (
    CreateVirtualServerRequest,
    ToolMapping,
    UpdateVirtualServerRequest,
    VirtualServerConfig,
)
from registry.services.virtual_server_service import (
    VirtualServerService,
    _generate_path_from_name,
    _get_effective_tool_name,
    _get_unique_backends,
)

# --- Unit tests for helper functions ---


class TestHelperFunctions:
    """Tests for private helper functions."""

    def test_generate_path_from_name(self):
        """Test path generation from server name."""
        assert _generate_path_from_name("Dev Essentials") == "/virtual/dev-essentials"

    def test_generate_path_special_chars(self):
        """Test path generation strips special characters."""
        assert _generate_path_from_name("My Server (v2)!") == "/virtual/my-server-v2"

    def test_generate_path_multiple_spaces(self):
        """Test path generation handles multiple spaces."""
        assert _generate_path_from_name("Dev   Essentials") == "/virtual/dev-essentials"

    def test_generate_path_empty_fallback(self):
        """Test path generation with empty name falls back."""
        assert _generate_path_from_name("!!!") == "/virtual/virtual-server"

    def test_get_effective_tool_name_with_alias(self):
        """Test effective name returns alias when set."""
        mapping = ToolMapping(
            tool_name="search",
            alias="github_search",
            backend_server_path="/github",
        )
        assert _get_effective_tool_name(mapping) == "github_search"

    def test_get_effective_tool_name_without_alias(self):
        """Test effective name returns original when no alias."""
        mapping = ToolMapping(
            tool_name="search",
            backend_server_path="/github",
        )
        assert _get_effective_tool_name(mapping) == "search"

    def test_get_unique_backends(self):
        """Test extracting unique backend paths."""
        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/github"),
            ToolMapping(tool_name="issues", backend_server_path="/github"),
            ToolMapping(tool_name="tickets", backend_server_path="/jira"),
        ]
        backends = _get_unique_backends(mappings)
        assert set(backends) == {"/github", "/jira"}


# --- Unit tests for service validation ---


class TestVirtualServerServiceValidation:
    """Tests for VirtualServerService validation logic."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repositories."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_validate_unique_tool_names_no_duplicates(self, service):
        """Test validation passes with unique tool names."""
        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/github"),
            ToolMapping(tool_name="tickets", backend_server_path="/jira"),
        ]
        # Should not raise
        service._validate_unique_tool_names(mappings)

    @pytest.mark.asyncio
    async def test_validate_unique_tool_names_duplicate_detected(self, service):
        """Test validation fails with duplicate tool names."""
        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/github"),
            ToolMapping(tool_name="search", backend_server_path="/jira"),
        ]
        with pytest.raises(VirtualServerValidationError, match="Duplicate tool names"):
            service._validate_unique_tool_names(mappings)

    @pytest.mark.asyncio
    async def test_validate_unique_tool_names_alias_resolves_conflict(self, service):
        """Test that aliases resolve name conflicts."""
        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/github"),
            ToolMapping(
                tool_name="search",
                alias="jira_search",
                backend_server_path="/jira",
            ),
        ]
        # Should not raise because alias makes names unique
        service._validate_unique_tool_names(mappings)

    @pytest.mark.asyncio
    async def test_validate_tool_mappings_missing_backend(self, service, mock_server_repo):
        """Test validation fails when backend server doesn't exist."""
        mock_server_repo.get.return_value = None

        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/nonexistent"),
        ]
        with pytest.raises(VirtualServerValidationError, match="does not exist"):
            await service._validate_tool_mappings(mappings)

    @pytest.mark.asyncio
    async def test_validate_tool_mappings_missing_tool(self, service, mock_server_repo):
        """Test validation fails when tool doesn't exist in backend."""
        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {"name": "create_issue", "description": "Create issue"},
            ],
        }

        mappings = [
            ToolMapping(tool_name="nonexistent_tool", backend_server_path="/github"),
        ]
        with pytest.raises(VirtualServerValidationError, match="not found in backend"):
            await service._validate_tool_mappings(mappings)

    @pytest.mark.asyncio
    async def test_validate_tool_mappings_valid(self, service, mock_server_repo):
        """Test validation passes with valid tool mappings."""
        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {"name": "search", "description": "Search repos"},
            ],
        }

        mappings = [
            ToolMapping(tool_name="search", backend_server_path="/github"),
        ]
        # Should not raise
        await service._validate_tool_mappings(mappings)

    @pytest.mark.asyncio
    async def test_validate_tool_mappings_version_not_found(self, service, mock_server_repo):
        """Test validation fails when pinned version doesn't exist."""
        # First call: server exists
        # Second call: version doc doesn't exist
        mock_server_repo.get.side_effect = [
            {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search"},
                ],
            },
            None,  # Version doc not found
        ]

        mappings = [
            ToolMapping(
                tool_name="search",
                backend_server_path="/github",
                backend_version="v99.0.0",
            ),
        ]
        with pytest.raises(VirtualServerValidationError, match="Version"):
            await service._validate_tool_mappings(mappings)


# --- Unit tests for service CRUD operations ---


class TestVirtualServerServiceCRUD:
    """Tests for VirtualServerService CRUD operations."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_create_virtual_server(self, service, mock_vs_repo):
        """Test creating a virtual server."""
        request = CreateVirtualServerRequest(
            server_name="Dev Essentials",
            path="/virtual/dev-essentials",
            description="Tools for development",
        )

        created = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
            description="Tools for development",
        )
        mock_vs_repo.create.return_value = created

        result = await service.create_virtual_server(request, created_by="admin")

        mock_vs_repo.create.assert_called_once()
        assert result.path == "/virtual/dev-essentials"
        assert result.server_name == "Dev Essentials"

    @pytest.mark.asyncio
    async def test_create_virtual_server_auto_generates_path(self, service, mock_vs_repo):
        """Test that path is auto-generated from name when not provided."""
        request = CreateVirtualServerRequest(
            server_name="My Cool Server",
        )

        mock_vs_repo.create.return_value = VirtualServerConfig(
            path="/virtual/my-cool-server",
            server_name="My Cool Server",
        )

        result = await service.create_virtual_server(request, created_by="admin")
        call_args = mock_vs_repo.create.call_args[0][0]
        assert call_args.path == "/virtual/my-cool-server"

    @pytest.mark.asyncio
    async def test_list_virtual_servers(self, service, mock_vs_repo):
        """Test listing virtual servers."""
        mock_vs_repo.list_all.return_value = [
            VirtualServerConfig(
                path="/virtual/dev",
                server_name="Dev",
                tool_mappings=[
                    ToolMapping(tool_name="search", backend_server_path="/github"),
                ],
            ),
        ]

        result = await service.list_virtual_servers()

        assert len(result) == 1
        assert result[0].path == "/virtual/dev"
        assert result[0].tool_count == 1
        assert result[0].backend_count == 1

    @pytest.mark.asyncio
    async def test_get_virtual_server(self, service, mock_vs_repo):
        """Test getting a single virtual server."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )

        result = await service.get_virtual_server("/virtual/dev")
        assert result is not None
        assert result.server_name == "Dev"

    @pytest.mark.asyncio
    async def test_get_virtual_server_not_found(self, service, mock_vs_repo):
        """Test getting a nonexistent virtual server returns None."""
        mock_vs_repo.get.return_value = None
        result = await service.get_virtual_server("/virtual/nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_virtual_server(self, service, mock_vs_repo):
        """Test deleting a virtual server."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            is_enabled=False,
        )
        mock_vs_repo.delete.return_value = True

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock):
            result = await service.delete_virtual_server("/virtual/dev")

        assert result is True
        mock_vs_repo.delete.assert_called_once_with("/virtual/dev")

    @pytest.mark.asyncio
    async def test_delete_virtual_server_not_found(self, service, mock_vs_repo):
        """Test deleting a nonexistent virtual server raises error."""
        mock_vs_repo.get.return_value = None

        with pytest.raises(VirtualServerNotFoundError):
            await service.delete_virtual_server("/virtual/nonexistent")

    @pytest.mark.asyncio
    async def test_toggle_virtual_server_enable(self, service, mock_vs_repo):
        """Test enabling a virtual server."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/github"),
            ],
        )
        mock_vs_repo.set_state.return_value = True

        with (
            patch.object(service, "_validate_tool_mappings", new_callable=AsyncMock),
            patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock),
        ):
            result = await service.toggle_virtual_server("/virtual/dev", True)

        assert result is True
        mock_vs_repo.set_state.assert_called_once_with("/virtual/dev", True)

    @pytest.mark.asyncio
    async def test_toggle_enable_with_no_tools_fails(self, service, mock_vs_repo):
        """Test enabling fails when no tool mappings configured."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[],
        )

        with pytest.raises(VirtualServerValidationError, match="no tool mappings"):
            await service.toggle_virtual_server("/virtual/dev", True)

    @pytest.mark.asyncio
    async def test_toggle_virtual_server_disable(self, service, mock_vs_repo):
        """Test disabling a virtual server."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            is_enabled=True,
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/github"),
            ],
        )
        mock_vs_repo.set_state.return_value = True

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock):
            result = await service.toggle_virtual_server("/virtual/dev", False)

        assert result is True
        mock_vs_repo.set_state.assert_called_once_with("/virtual/dev", False)

    @pytest.mark.asyncio
    async def test_toggle_virtual_server_not_found(self, service, mock_vs_repo):
        """Test toggling a nonexistent virtual server raises error."""
        mock_vs_repo.get.return_value = None

        with pytest.raises(VirtualServerNotFoundError):
            await service.toggle_virtual_server("/virtual/nonexistent", True)

    @pytest.mark.asyncio
    async def test_update_virtual_server_happy_path(self, service, mock_vs_repo):
        """Test updating a virtual server with valid data."""
        existing = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            description="Old description",
        )
        mock_vs_repo.get.return_value = existing

        updated = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev Updated",
            description="New description",
        )
        mock_vs_repo.update.return_value = updated

        request = UpdateVirtualServerRequest(
            server_name="Dev Updated",
            description="New description",
        )

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock):
            result = await service.update_virtual_server("/virtual/dev", request)

        assert result is not None
        assert result.server_name == "Dev Updated"
        mock_vs_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_virtual_server_not_found(self, service, mock_vs_repo):
        """Test updating a nonexistent virtual server raises error."""
        mock_vs_repo.get.return_value = None

        request = UpdateVirtualServerRequest(description="New description")

        with pytest.raises(VirtualServerNotFoundError):
            await service.update_virtual_server("/virtual/nonexistent", request)

    @pytest.mark.asyncio
    async def test_update_virtual_server_with_new_tool_mappings(
        self, service, mock_vs_repo, mock_server_repo
    ):
        """Test updating tool_mappings validates backend servers."""
        existing = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )
        mock_vs_repo.get.return_value = existing

        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {"name": "search", "description": "Search repos"},
            ],
        }

        updated = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/github"),
            ],
        )
        mock_vs_repo.update.return_value = updated

        request = UpdateVirtualServerRequest(
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/github"),
            ],
        )

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock):
            result = await service.update_virtual_server("/virtual/dev", request)

        assert result is not None
        mock_server_repo.get.assert_called()

    @pytest.mark.asyncio
    async def test_create_virtual_server_duplicate_path(self, service, mock_vs_repo):
        """Test creating virtual server with duplicate path raises error."""
        mock_vs_repo.create.side_effect = VirtualServerAlreadyExistsError("/virtual/dev-essentials")

        request = CreateVirtualServerRequest(
            server_name="Dev Essentials",
            path="/virtual/dev-essentials",
        )

        with pytest.raises(VirtualServerAlreadyExistsError):
            await service.create_virtual_server(request, created_by="admin")

    @pytest.mark.asyncio
    async def test_create_virtual_server_invalid_backend(self, service, mock_server_repo):
        """Test creating virtual server with invalid backend raises validation error."""
        mock_server_repo.get.return_value = None

        request = CreateVirtualServerRequest(
            server_name="Dev Essentials",
            path="/virtual/dev-essentials",
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/nonexistent"),
            ],
        )

        with pytest.raises(VirtualServerValidationError, match="does not exist"):
            await service.create_virtual_server(request, created_by="admin")


# --- Unit tests for tool resolution ---


class TestToolResolution:
    """Tests for tool resolution logic."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create service with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_resolve_tools(self, service, mock_vs_repo, mock_server_repo):
        """Test resolving tools from virtual server config."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    alias="github_search",
                    backend_server_path="/github",
                ),
            ],
        )

        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {
                    "name": "search",
                    "description": "Search repos",
                    "inputSchema": {"type": "object"},
                },
            ],
        }

        tools = await service.resolve_tools("/virtual/dev")

        assert len(tools) == 1
        assert tools[0].name == "github_search"
        assert tools[0].original_name == "search"
        assert tools[0].description == "Search repos"

    @pytest.mark.asyncio
    async def test_resolve_tools_with_description_override(
        self, service, mock_vs_repo, mock_server_repo
    ):
        """Test that description_override replaces original description."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    backend_server_path="/github",
                    description_override="Custom description",
                ),
            ],
        )

        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {
                    "name": "search",
                    "description": "Original description",
                    "inputSchema": {},
                },
            ],
        }

        tools = await service.resolve_tools("/virtual/dev")

        assert len(tools) == 1
        assert tools[0].description == "Custom description"

    @pytest.mark.asyncio
    async def test_resolve_tools_not_found(self, service, mock_vs_repo):
        """Test resolving tools for nonexistent server raises error."""
        mock_vs_repo.get.return_value = None

        with pytest.raises(VirtualServerNotFoundError):
            await service.resolve_tools("/virtual/nonexistent")

    @pytest.mark.asyncio
    async def test_resolve_tools_with_scope_overrides(
        self, service, mock_vs_repo, mock_server_repo
    ):
        """Test that scope overrides are applied to resolved tools."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    backend_server_path="/github",
                ),
            ],
            tool_scope_overrides=[
                {
                    "tool_alias": "search",
                    "required_scopes": ["github:read"],
                },
            ],
        )

        mock_server_repo.get.return_value = {
            "server_name": "GitHub",
            "tool_list": [
                {
                    "name": "search",
                    "description": "Search",
                    "inputSchema": {},
                },
            ],
        }

        tools = await service.resolve_tools("/virtual/dev")

        assert len(tools) == 1
        assert tools[0].required_scopes == ["github:read"]


# --- Unit tests for nginx trigger ---


class TestNginxTrigger:
    """Tests verifying nginx config regeneration is triggered on changes."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_create_triggers_nginx_reload(self, service, mock_vs_repo):
        """Test that creating a virtual server triggers nginx reload."""
        request = CreateVirtualServerRequest(
            server_name="Dev Tools",
            path="/virtual/dev-tools",
        )
        mock_vs_repo.create.return_value = VirtualServerConfig(
            path="/virtual/dev-tools",
            server_name="Dev Tools",
        )

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock) as mock_reload:
            await service.create_virtual_server(request, created_by="admin")
            mock_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_triggers_nginx_reload(self, service, mock_vs_repo):
        """Test that deleting a virtual server triggers nginx reload."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )
        mock_vs_repo.delete.return_value = True

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock) as mock_reload:
            await service.delete_virtual_server("/virtual/dev")
            mock_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_triggers_nginx_reload(self, service, mock_vs_repo):
        """Test that toggling a virtual server triggers nginx reload."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            tool_mappings=[
                ToolMapping(tool_name="search", backend_server_path="/github"),
            ],
        )
        mock_vs_repo.set_state.return_value = True

        with (
            patch.object(service, "_validate_tool_mappings", new_callable=AsyncMock),
            patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock) as mock_reload,
        ):
            await service.toggle_virtual_server("/virtual/dev", True)
            mock_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_triggers_nginx_reload(self, service, mock_vs_repo):
        """Test that updating a virtual server triggers nginx reload."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )
        mock_vs_repo.update.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev Updated",
        )

        request = UpdateVirtualServerRequest(server_name="Dev Updated")

        with patch.object(service, "_trigger_nginx_reload", new_callable=AsyncMock) as mock_reload:
            await service.update_virtual_server("/virtual/dev", request)
            mock_reload.assert_called_once()


# --- Unit tests for ToolCatalogService ---


class TestToolCatalogService:
    """Tests for ToolCatalogService aggregation logic."""

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def catalog_service(self, mock_server_repo):
        """Create ToolCatalogService with mocked repository."""
        with patch(
            "registry.services.tool_catalog_service.get_server_repository",
            return_value=mock_server_repo,
        ):
            from registry.services.tool_catalog_service import ToolCatalogService

            svc = ToolCatalogService()
            return svc

    @pytest.mark.asyncio
    async def test_catalog_aggregates_from_multiple_servers(
        self, catalog_service, mock_server_repo
    ):
        """Test that catalog aggregates tools from multiple enabled servers."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                    {"name": "create_issue", "description": "Create issue"},
                ],
            },
            "/jira": {
                "server_name": "Jira",
                "tool_list": [
                    {"name": "get_ticket", "description": "Get ticket"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog()

        assert len(catalog) == 3
        tool_names = [entry.tool_name for entry in catalog]
        assert "search" in tool_names
        assert "create_issue" in tool_names
        assert "get_ticket" in tool_names

    @pytest.mark.asyncio
    async def test_catalog_filters_disabled_servers(self, catalog_service, mock_server_repo):
        """Test that catalog excludes tools from disabled servers."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/jira": {
                "server_name": "Jira",
                "tool_list": [
                    {"name": "get_ticket", "description": "Get ticket"},
                ],
            },
        }
        # GitHub enabled, Jira disabled
        mock_server_repo.get_state.side_effect = [True, False]

        catalog = await catalog_service.get_tool_catalog()

        assert len(catalog) == 1
        assert catalog[0].tool_name == "search"
        assert catalog[0].server_path == "/github"

    @pytest.mark.asyncio
    async def test_catalog_filters_by_server_path(self, catalog_service, mock_server_repo):
        """Test that catalog can filter by server_path."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/jira": {
                "server_name": "Jira",
                "tool_list": [
                    {"name": "get_ticket", "description": "Get ticket"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog(server_path_filter="/github")

        assert len(catalog) == 1
        assert catalog[0].server_path == "/github"

    @pytest.mark.asyncio
    async def test_catalog_skips_version_documents(self, catalog_service, mock_server_repo):
        """Test that catalog skips version documents (paths with ':')."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/github:v1.5.0": {
                "server_name": "GitHub v1.5.0",
                "tool_list": [
                    {"name": "search", "description": "Search repos v1.5"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog()

        assert len(catalog) == 1
        assert catalog[0].server_path == "/github"

    @pytest.mark.asyncio
    async def test_catalog_empty_when_no_servers(self, catalog_service, mock_server_repo):
        """Test that catalog returns empty list when no servers exist."""
        mock_server_repo.list_all.return_value = {}

        catalog = await catalog_service.get_tool_catalog()

        assert catalog == []

    @pytest.mark.asyncio
    async def test_catalog_includes_available_versions(self, catalog_service, mock_server_repo):
        """Test that catalog entries include available versions."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "version": "v2.0.0",
                "other_version_ids": ["/github:v1.5.0"],
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog()

        assert len(catalog) == 1
        assert "v2.0.0" in catalog[0].available_versions
        assert "v1.5.0" in catalog[0].available_versions

    @pytest.mark.asyncio
    async def test_catalog_skips_tools_without_name(self, catalog_service, mock_server_repo):
        """Test that catalog skips tool entries that have no name."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                    {"name": "", "description": "Unnamed tool"},
                    {"description": "No name field"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog()

        assert len(catalog) == 1
        assert catalog[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_catalog_filters_by_user_scopes(self, catalog_service, mock_server_repo):
        """Test that catalog filters out servers the user lacks scopes for."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "required_scopes": ["read:repos"],
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/jira": {
                "server_name": "Jira",
                "required_scopes": ["admin:jira"],
                "tool_list": [
                    {"name": "get_ticket", "description": "Get ticket"},
                ],
            },
            "/slack": {
                "server_name": "Slack",
                "tool_list": [
                    {"name": "send_message", "description": "Send message"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        # User has read:repos but not admin:jira
        catalog = await catalog_service.get_tool_catalog(user_scopes=["read:repos"])

        assert len(catalog) == 2
        tool_names = [entry.tool_name for entry in catalog]
        assert "search" in tool_names
        assert "send_message" in tool_names
        assert "get_ticket" not in tool_names

    @pytest.mark.asyncio
    async def test_catalog_no_filtering_when_scopes_none(self, catalog_service, mock_server_repo):
        """Test that passing user_scopes=None returns all servers (no filtering)."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "required_scopes": ["admin:everything"],
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog(user_scopes=None)

        assert len(catalog) == 1
        assert catalog[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_catalog_empty_scopes_filters_restricted_servers(
        self, catalog_service, mock_server_repo
    ):
        """Test that user with empty scopes is filtered out from restricted servers."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "required_scopes": ["read:repos"],
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/slack": {
                "server_name": "Slack",
                "tool_list": [
                    {"name": "send_message", "description": "Send message"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog(user_scopes=[])

        assert len(catalog) == 1
        assert catalog[0].tool_name == "send_message"

    @pytest.mark.asyncio
    async def test_catalog_user_with_all_scopes_sees_all(self, catalog_service, mock_server_repo):
        """Test that user with all required scopes sees all servers."""
        mock_server_repo.list_all.return_value = {
            "/github": {
                "server_name": "GitHub",
                "required_scopes": ["read:repos"],
                "tool_list": [
                    {"name": "search", "description": "Search repos"},
                ],
            },
            "/jira": {
                "server_name": "Jira",
                "required_scopes": ["admin:jira", "read:projects"],
                "tool_list": [
                    {"name": "get_ticket", "description": "Get ticket"},
                ],
            },
        }
        mock_server_repo.get_state.return_value = True

        catalog = await catalog_service.get_tool_catalog(
            user_scopes=["read:repos", "admin:jira", "read:projects"]
        )

        assert len(catalog) == 2


# --- Unit tests for nginx reload failure handling ---


class TestNginxReloadFailureHandling:
    """Tests verifying CRUD operations succeed even when nginx reload fails."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_create_succeeds_when_nginx_reload_fails(
        self,
        service,
        mock_vs_repo,
    ):
        """Test that create succeeds even if nginx reload returns False."""
        request = CreateVirtualServerRequest(
            server_name="Dev Tools",
            path="/virtual/dev-tools",
        )
        created = VirtualServerConfig(
            path="/virtual/dev-tools",
            server_name="Dev Tools",
        )
        mock_vs_repo.create.return_value = created

        with patch.object(
            service,
            "_trigger_nginx_reload",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await service.create_virtual_server(request, created_by="admin")

        # CRUD operation should succeed regardless
        assert result.path == "/virtual/dev-tools"
        mock_vs_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_nginx_reload_returns_false_on_exception(self, service):
        """Test that _trigger_nginx_reload returns False when an exception occurs."""
        mock_nginx = AsyncMock()
        mock_server_svc = AsyncMock()
        mock_server_svc.get_enabled_services = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )

        with (
            patch(
                "registry.core.nginx_service.nginx_service",
                mock_nginx,
            ),
            patch(
                "registry.services.server_service.server_service",
                mock_server_svc,
            ),
        ):
            result = await service._trigger_nginx_reload()

        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_nginx_reload_logs_error_on_failure(self, service, caplog):
        """Test that _trigger_nginx_reload logs error when it fails."""
        mock_nginx = AsyncMock()
        mock_server_svc = AsyncMock()
        mock_server_svc.get_enabled_services = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )

        with (
            patch(
                "registry.core.nginx_service.nginx_service",
                mock_nginx,
            ),
            patch(
                "registry.services.server_service.server_service",
                mock_server_svc,
            ),
            caplog.at_level(logging.ERROR),
        ):
            result = await service._trigger_nginx_reload()

        assert result is False
        assert any(
            "Failed to regenerate nginx config" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_trigger_nginx_reload_returns_true_on_success(self, service):
        """Test that _trigger_nginx_reload returns True on success."""
        mock_nginx = AsyncMock()
        mock_nginx.generate_config_async = AsyncMock(return_value=True)
        mock_server_svc = AsyncMock()
        mock_server_svc.get_enabled_services = AsyncMock(return_value=[])

        with (
            patch(
                "registry.core.nginx_service.nginx_service",
                mock_nginx,
            ),
            patch(
                "registry.services.server_service.server_service",
                mock_server_svc,
            ),
        ):
            result = await service._trigger_nginx_reload()

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_succeeds_when_nginx_reload_fails(
        self,
        service,
        mock_vs_repo,
    ):
        """Test that delete succeeds even if nginx reload fails."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )
        mock_vs_repo.delete.return_value = True

        with patch.object(
            service,
            "_trigger_nginx_reload",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await service.delete_virtual_server("/virtual/dev")

        assert result is True
        mock_vs_repo.delete.assert_called_once_with("/virtual/dev")

    @pytest.mark.asyncio
    async def test_update_succeeds_when_nginx_reload_fails(
        self,
        service,
        mock_vs_repo,
    ):
        """Test that update succeeds even if nginx reload fails."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
        )
        updated = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev Updated",
        )
        mock_vs_repo.update.return_value = updated

        request = UpdateVirtualServerRequest(server_name="Dev Updated")

        with patch.object(
            service,
            "_trigger_nginx_reload",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await service.update_virtual_server("/virtual/dev", request)

        assert result is not None
        assert result.server_name == "Dev Updated"


# --- Unit tests for path auto-generation collision ---


class TestPathAutoGenerationCollision:
    """Tests for path auto-generation and collision handling."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_auto_generated_path_collision_raises_error(
        self,
        service,
        mock_vs_repo,
    ):
        """Test that auto-generated path collision raises VirtualServerAlreadyExistsError."""
        mock_vs_repo.create.side_effect = VirtualServerAlreadyExistsError("/virtual/my-cool-server")

        request = CreateVirtualServerRequest(
            server_name="My Cool Server",
            # No explicit path -- will be auto-generated as /virtual/my-cool-server
        )

        with pytest.raises(VirtualServerAlreadyExistsError) as exc_info:
            await service.create_virtual_server(request, created_by="admin")

        assert "/virtual/my-cool-server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_explicit_path_collision_raises_error(
        self,
        service,
        mock_vs_repo,
    ):
        """Test that explicit path collision raises VirtualServerAlreadyExistsError."""
        mock_vs_repo.create.side_effect = VirtualServerAlreadyExistsError("/virtual/dev-essentials")

        request = CreateVirtualServerRequest(
            server_name="Dev Essentials",
            path="/virtual/dev-essentials",
        )

        with pytest.raises(VirtualServerAlreadyExistsError):
            await service.create_virtual_server(request, created_by="admin")

    @pytest.mark.asyncio
    async def test_auto_generate_path_produces_valid_slug(self):
        """Test various name-to-path conversions produce valid slugs."""
        test_cases = [
            ("Dev Essentials", "/virtual/dev-essentials"),
            ("My Server (v2)!", "/virtual/my-server-v2"),
            ("  spaces  everywhere  ", "/virtual/spaces-everywhere"),
            ("UPPERCASE", "/virtual/uppercase"),
            ("with---dashes", "/virtual/with-dashes"),
            ("a", "/virtual/a"),
        ]
        for name, expected_path in test_cases:
            result = _generate_path_from_name(name)
            assert result == expected_path, f"Failed for name='{name}'"


# --- Unit tests for nginx reload lock serialization ---


class TestNginxReloadLock:
    """Tests verifying nginx reload lock serializes concurrent operations."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_reload_lock_exists(self):
        """Test that the module-level nginx reload lock is an asyncio.Lock."""
        import asyncio

        from registry.services.virtual_server_service import _nginx_reload_lock

        assert isinstance(_nginx_reload_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_reloads_are_serialized(self, service):
        """Test that concurrent nginx reloads are serialized by the lock."""
        import asyncio

        call_order = []

        async def mock_generate(*args, **kwargs):
            call_order.append("start")
            await asyncio.sleep(0.01)
            call_order.append("end")
            return True

        mock_nginx = AsyncMock()
        mock_nginx.generate_config_async = mock_generate
        mock_server_svc = AsyncMock()
        mock_server_svc.get_enabled_services = AsyncMock(return_value=[])

        with (
            patch(
                "registry.core.nginx_service.nginx_service",
                mock_nginx,
            ),
            patch(
                "registry.services.server_service.server_service",
                mock_server_svc,
            ),
        ):
            # Launch two reloads concurrently
            results = await asyncio.gather(
                service._trigger_nginx_reload(),
                service._trigger_nginx_reload(),
            )

        # Both should succeed
        assert all(results)
        # The lock ensures serialization: start-end-start-end, not start-start-end-end
        assert call_order == ["start", "end", "start", "end"]


# --- Unit tests for rating functionality ---


class TestVirtualServerRating:
    """Tests for VirtualServerService rating operations."""

    @pytest.fixture
    def mock_vs_repo(self):
        """Create mock virtual server repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_server_repo(self):
        """Create mock server repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_vs_repo, mock_server_repo):
        """Create VirtualServerService with mocked repos."""
        with (
            patch(
                "registry.services.virtual_server_service.get_virtual_server_repository",
                return_value=mock_vs_repo,
            ),
            patch(
                "registry.services.virtual_server_service.get_server_repository",
                return_value=mock_server_repo,
            ),
        ):
            svc = VirtualServerService()
            return svc

    @pytest.mark.asyncio
    async def test_rate_virtual_server_new_rating(self, service, mock_vs_repo):
        """Test rating a virtual server for the first time."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            num_stars=0.0,
            rating_details=[],
        )
        mock_vs_repo.update_rating.return_value = True

        result = await service.rate_virtual_server(
            path="/virtual/dev",
            username="testuser",
            rating=4,
        )

        assert result["average_rating"] == 4.0
        assert result["is_new_rating"] is True
        assert result["total_ratings"] == 1
        mock_vs_repo.update_rating.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_virtual_server_update_existing(self, service, mock_vs_repo):
        """Test updating an existing rating."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            num_stars=4.0,
            rating_details=[{"user": "testuser", "rating": 4}],
        )
        mock_vs_repo.update_rating.return_value = True

        result = await service.rate_virtual_server(
            path="/virtual/dev",
            username="testuser",
            rating=5,
        )

        assert result["average_rating"] == 5.0
        assert result["is_new_rating"] is False
        assert result["total_ratings"] == 1

    @pytest.mark.asyncio
    async def test_rate_virtual_server_multiple_users(self, service, mock_vs_repo):
        """Test rating with multiple users."""
        mock_vs_repo.get.return_value = VirtualServerConfig(
            path="/virtual/dev",
            server_name="Dev",
            num_stars=4.0,
            rating_details=[{"user": "user1", "rating": 4}],
        )
        mock_vs_repo.update_rating.return_value = True

        result = await service.rate_virtual_server(
            path="/virtual/dev",
            username="user2",
            rating=5,
        )

        assert result["average_rating"] == 4.5
        assert result["is_new_rating"] is True
        assert result["total_ratings"] == 2

    @pytest.mark.asyncio
    async def test_rate_virtual_server_not_found(self, service, mock_vs_repo):
        """Test rating a nonexistent virtual server raises error."""
        mock_vs_repo.get.return_value = None

        with pytest.raises(VirtualServerNotFoundError):
            await service.rate_virtual_server(
                path="/virtual/nonexistent",
                username="testuser",
                rating=4,
            )

    @pytest.mark.asyncio
    async def test_rate_virtual_server_invalid_rating_low(self, service):
        """Test rating with value below minimum raises error."""
        with pytest.raises(ValueError, match="between 1 and 5"):
            await service.rate_virtual_server(
                path="/virtual/dev",
                username="testuser",
                rating=0,
            )

    @pytest.mark.asyncio
    async def test_rate_virtual_server_invalid_rating_high(self, service):
        """Test rating with value above maximum raises error."""
        with pytest.raises(ValueError, match="between 1 and 5"):
            await service.rate_virtual_server(
                path="/virtual/dev",
                username="testuser",
                rating=6,
            )

    @pytest.mark.asyncio
    async def test_get_virtual_server_rating(self, service, mock_vs_repo):
        """Test getting rating information."""
        mock_vs_repo.get_rating.return_value = {
            "num_stars": 4.5,
            "rating_details": [
                {"user": "user1", "rating": 4},
                {"user": "user2", "rating": 5},
            ],
        }

        result = await service.get_virtual_server_rating("/virtual/dev")

        assert result["num_stars"] == 4.5
        assert len(result["rating_details"]) == 2
        mock_vs_repo.get_rating.assert_called_once_with("/virtual/dev")

    @pytest.mark.asyncio
    async def test_get_virtual_server_rating_not_found(self, service, mock_vs_repo):
        """Test getting rating for nonexistent virtual server raises error."""
        mock_vs_repo.get_rating.return_value = None

        with pytest.raises(VirtualServerNotFoundError):
            await service.get_virtual_server_rating("/virtual/nonexistent")

    @pytest.mark.asyncio
    async def test_get_virtual_server_rating_no_ratings(self, service, mock_vs_repo):
        """Test getting rating for server with no ratings."""
        mock_vs_repo.get_rating.return_value = {
            "num_stars": 0.0,
            "rating_details": [],
        }

        result = await service.get_virtual_server_rating("/virtual/dev")

        assert result["num_stars"] == 0.0
        assert result["rating_details"] == []

    @pytest.mark.asyncio
    async def test_list_virtual_servers_includes_rating(self, service, mock_vs_repo):
        """Test that list_virtual_servers includes rating info."""
        mock_vs_repo.list_all.return_value = [
            VirtualServerConfig(
                path="/virtual/dev",
                server_name="Dev",
                tool_mappings=[
                    ToolMapping(tool_name="search", backend_server_path="/github"),
                ],
                num_stars=4.5,
                rating_details=[{"user": "user1", "rating": 4}],
            ),
        ]

        result = await service.list_virtual_servers()

        assert len(result) == 1
        assert result[0].num_stars == 4.5
        assert len(result[0].rating_details) == 1
