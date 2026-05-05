"""Unit tests for virtual server Pydantic models."""

import pytest
from pydantic import ValidationError

from registry.schemas.virtual_server_models import (
    CreateVirtualServerRequest,
    ResolvedTool,
    ToggleVirtualServerRequest,
    ToolCatalogEntry,
    ToolMapping,
    ToolScopeOverride,
    UpdateVirtualServerRequest,
    VirtualServerConfig,
    VirtualServerInfo,
)


class TestToolMapping:
    """Tests for ToolMapping model."""

    def test_valid_tool_mapping(self):
        """Test creating a valid tool mapping."""
        mapping = ToolMapping(
            tool_name="search",
            backend_server_path="/github",
        )
        assert mapping.tool_name == "search"
        assert mapping.backend_server_path == "/github"
        assert mapping.alias is None
        assert mapping.backend_version is None
        assert mapping.description_override is None

    def test_tool_mapping_with_alias(self):
        """Test tool mapping with alias and version pin."""
        mapping = ToolMapping(
            tool_name="search",
            alias="github_search",
            backend_server_path="/github",
            backend_version="v1.5.0",
            description_override="Search GitHub repos",
        )
        assert mapping.alias == "github_search"
        assert mapping.backend_version == "v1.5.0"
        assert mapping.description_override == "Search GitHub repos"

    def test_tool_mapping_requires_tool_name(self):
        """Test that tool_name is required."""
        with pytest.raises(ValidationError):
            ToolMapping(
                backend_server_path="/github",
            )

    def test_tool_mapping_requires_backend_path(self):
        """Test that backend_server_path is required."""
        with pytest.raises(ValidationError):
            ToolMapping(
                tool_name="search",
            )

    def test_backend_path_must_start_with_slash(self):
        """Test that backend_server_path must start with /."""
        with pytest.raises(ValidationError, match="must start with '/'"):
            ToolMapping(
                tool_name="search",
                backend_server_path="github",
            )

    def test_backend_path_empty_string_rejected(self):
        """Test that empty backend_server_path is rejected."""
        with pytest.raises(ValidationError):
            ToolMapping(
                tool_name="search",
                backend_server_path="",
            )

    def test_tool_name_empty_string_rejected(self):
        """Test that empty tool_name is rejected."""
        with pytest.raises(ValidationError):
            ToolMapping(
                tool_name="",
                backend_server_path="/github",
            )


class TestToolScopeOverride:
    """Tests for ToolScopeOverride model."""

    def test_valid_scope_override(self):
        """Test creating a valid scope override."""
        override = ToolScopeOverride(
            tool_alias="github_search",
            required_scopes=["tools:github:read"],
        )
        assert override.tool_alias == "github_search"
        assert override.required_scopes == ["tools:github:read"]

    def test_multiple_scopes(self):
        """Test ToolScopeOverride with multiple scopes."""
        override = ToolScopeOverride(
            tool_alias="get_data",
            required_scopes=["read:data", "write:data"],
        )
        assert len(override.required_scopes) == 2

    def test_scope_override_requires_scopes(self):
        """Test that required_scopes must be non-empty."""
        with pytest.raises(ValidationError):
            ToolScopeOverride(
                tool_alias="search",
                required_scopes=[],
            )

    def test_empty_tool_alias_rejected(self):
        """Test that empty tool_alias is rejected."""
        with pytest.raises(ValidationError):
            ToolScopeOverride(
                tool_alias="",
                required_scopes=["read:data"],
            )


class TestVirtualServerConfig:
    """Tests for VirtualServerConfig model."""

    def test_valid_config(self):
        """Test creating a valid virtual server config."""
        config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
        )
        assert config.path == "/virtual/dev-essentials"
        assert config.server_name == "Dev Essentials"
        assert config.description == ""
        assert config.tool_mappings == []
        assert config.required_scopes == []
        assert config.tool_scope_overrides == []
        assert config.is_enabled is False
        assert config.tags == []
        assert config.supported_transports == ["streamable-http"]

    def test_full_config(self):
        """Test creating a config with all fields."""
        config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
            description="Tools for everyday development",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    backend_server_path="/github",
                ),
                ToolMapping(
                    tool_name="create_issue",
                    alias="jira_create_issue",
                    backend_server_path="/jira",
                ),
            ],
            required_scopes=["dev-team"],
            tool_scope_overrides=[
                ToolScopeOverride(
                    tool_alias="jira_create_issue",
                    required_scopes=["jira:write"],
                ),
            ],
            is_enabled=True,
            tags=["dev", "productivity"],
            created_by="admin",
        )
        assert len(config.tool_mappings) == 2
        assert len(config.tool_scope_overrides) == 1
        assert config.is_enabled is True
        assert config.created_by == "admin"

    def test_path_must_start_with_virtual(self):
        """Test that path must start with /virtual/."""
        with pytest.raises(ValidationError, match="must start with '/virtual/'"):
            VirtualServerConfig(
                path="/my-server",
                server_name="Test",
            )

    def test_path_requires_name_after_virtual(self):
        """Test that path must have a name after /virtual/."""
        with pytest.raises(ValidationError, match="must have a name"):
            VirtualServerConfig(
                path="/virtual/",
                server_name="Test",
            )

    def test_path_name_must_be_lowercase_alphanumeric(self):
        """Test that path segment must be lowercase alphanumeric."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/My_Server",
                server_name="Test",
            )

    def test_path_uppercase_rejected(self):
        """Test that uppercase characters in path are rejected."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/DevTools",
                server_name="Test",
            )

    def test_path_special_chars_rejected(self):
        """Test that special characters in path are rejected."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/dev_tools",
                server_name="Test",
            )

    def test_path_name_allows_hyphens(self):
        """Test that path segment allows single hyphens."""
        config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
        )
        assert config.path == "/virtual/dev-essentials"

    def test_path_name_allows_multi_segment_hyphens(self):
        """Test valid path with multiple hyphenated segments."""
        config = VirtualServerConfig(
            path="/virtual/dev-tools-v2",
            server_name="Dev Tools V2",
        )
        assert config.path == "/virtual/dev-tools-v2"

    def test_path_name_disallows_consecutive_hyphens(self):
        """Test that path segment disallows consecutive hyphens."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/dev--essentials",
                server_name="Test",
            )

    def test_path_leading_hyphen_rejected(self):
        """Test that leading hyphen in path segment is rejected."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/-devtools",
                server_name="Test",
            )

    def test_path_trailing_hyphen_rejected(self):
        """Test that trailing hyphen in path segment is rejected."""
        with pytest.raises(ValidationError, match="lowercase alphanumeric"):
            VirtualServerConfig(
                path="/virtual/devtools-",
                server_name="Test",
            )

    def test_server_name_cannot_be_empty(self):
        """Test that server_name cannot be empty."""
        with pytest.raises(ValidationError):
            VirtualServerConfig(
                path="/virtual/test",
                server_name="",
            )

    def test_server_name_strips_whitespace(self):
        """Test that server name is stripped of whitespace."""
        config = VirtualServerConfig(
            path="/virtual/test",
            server_name="  Dev Essentials  ",
        )
        assert config.server_name == "Dev Essentials"

    def test_server_name_whitespace_only_rejected(self):
        """Test that whitespace-only server name is rejected."""
        with pytest.raises(ValidationError, match="empty or whitespace-only"):
            VirtualServerConfig(
                path="/virtual/test",
                server_name="   ",
            )

    def test_default_is_enabled_false(self):
        """Test that is_enabled defaults to False."""
        config = VirtualServerConfig(
            path="/virtual/test",
            server_name="Test",
        )
        assert config.is_enabled is False

    def test_default_tags_empty(self):
        """Test that tags defaults to empty list."""
        config = VirtualServerConfig(
            path="/virtual/test",
            server_name="Test",
        )
        assert config.tags == []

    def test_default_supported_transports(self):
        """Test that supported_transports defaults to streamable-http."""
        config = VirtualServerConfig(
            path="/virtual/test",
            server_name="Test",
        )
        assert config.supported_transports == ["streamable-http"]

    def test_default_timestamps_set(self):
        """Test that created_at and updated_at are set by default."""
        config = VirtualServerConfig(
            path="/virtual/test",
            server_name="Test",
        )
        assert config.created_at is not None
        assert config.updated_at is not None

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization round trip."""
        config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
            description="Testing serialization",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    alias="gh-search",
                    backend_server_path="/github",
                ),
            ],
            tags=["dev"],
            is_enabled=True,
        )

        json_data = config.model_dump(mode="json")
        restored = VirtualServerConfig(**json_data)

        assert restored.path == config.path
        assert restored.server_name == config.server_name
        assert restored.description == config.description
        assert len(restored.tool_mappings) == 1
        assert restored.tool_mappings[0].tool_name == "search"
        assert restored.tool_mappings[0].alias == "gh-search"
        assert restored.tags == ["dev"]
        assert restored.is_enabled is True

    def test_null_description_coerced_to_empty_string(self):
        """MongoDB docs with description=null must deserialize (PR #932 fix).

        Older records persisted a literal null for description. The Pydantic
        validator added in PR #932 coerces that to the empty string on read
        so the repository does not silently drop the document.
        """
        config = VirtualServerConfig(
            path="/virtual/legacy-null",
            server_name="Legacy",
            description=None,
        )
        assert config.description == ""

    def test_missing_description_defaults_to_empty_string(self):
        """Description field is optional with a default of empty string."""
        config = VirtualServerConfig(
            path="/virtual/no-description",
            server_name="No Description",
        )
        assert config.description == ""


class TestVirtualServerInfo:
    """Tests for VirtualServerInfo model."""

    def test_valid_info(self):
        """Test creating a valid info summary."""
        info = VirtualServerInfo(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
            tool_count=5,
            backend_count=2,
            backend_paths=["/github", "/jira"],
            is_enabled=True,
        )
        assert info.tool_count == 5
        assert info.backend_count == 2
        assert len(info.backend_paths) == 2

    def test_info_defaults(self):
        """Test VirtualServerInfo default values."""
        info = VirtualServerInfo(
            path="/virtual/test",
            server_name="Test",
        )
        assert info.tool_count == 0
        assert info.backend_count == 0
        assert info.backend_paths == []
        assert info.is_enabled is False
        assert info.tags == []
        assert info.created_by is None
        assert info.created_at is None

    def test_null_description_coerced_to_empty_string(self):
        """Symmetric defensive coercion on VirtualServerInfo (PR #932 fix)."""
        info = VirtualServerInfo(
            path="/virtual/legacy-null",
            server_name="Legacy",
            description=None,
        )
        assert info.description == ""


class TestCreateVirtualServerRequest:
    """Tests for CreateVirtualServerRequest model."""

    def test_minimal_request(self):
        """Test creating request with only required fields."""
        req = CreateVirtualServerRequest(
            server_name="Dev Essentials",
        )
        assert req.server_name == "Dev Essentials"
        assert req.path is None
        assert req.description == ""
        assert req.tool_mappings == []
        assert req.required_scopes == []
        assert req.tags == []

    def test_full_request(self):
        """Test creating request with all fields."""
        req = CreateVirtualServerRequest(
            server_name="Dev Essentials",
            path="/virtual/dev-essentials",
            description="Tools for development",
            tool_mappings=[
                ToolMapping(
                    tool_name="search",
                    backend_server_path="/github",
                ),
            ],
            required_scopes=["dev-team"],
            tags=["dev"],
        )
        assert req.path == "/virtual/dev-essentials"
        assert len(req.tool_mappings) == 1

    def test_default_supported_transports(self):
        """Test CreateVirtualServerRequest default supported transports."""
        req = CreateVirtualServerRequest(
            server_name="My Server",
        )
        assert req.supported_transports == ["streamable-http"]


class TestUpdateVirtualServerRequest:
    """Tests for UpdateVirtualServerRequest model."""

    def test_partial_update(self):
        """Test creating request with partial fields."""
        req = UpdateVirtualServerRequest(
            description="Updated description",
        )
        assert req.description == "Updated description"
        assert req.server_name is None
        assert req.tool_mappings is None

    def test_update_all_none(self):
        """Test UpdateVirtualServerRequest with no fields set."""
        req = UpdateVirtualServerRequest()
        data = req.model_dump(exclude_unset=True)
        assert data == {}

    def test_update_exclude_unset(self):
        """Test that exclude_unset only includes provided fields."""
        req = UpdateVirtualServerRequest(
            server_name="New Name",
            description="New description",
        )
        data = req.model_dump(exclude_unset=True)
        assert "server_name" in data
        assert "description" in data
        assert "tool_mappings" not in data
        assert "tags" not in data

    def test_update_with_tool_mappings(self):
        """Test UpdateVirtualServerRequest with tool_mappings update."""
        req = UpdateVirtualServerRequest(
            tool_mappings=[
                ToolMapping(
                    tool_name="new_tool",
                    backend_server_path="/new-backend",
                ),
            ],
        )
        data = req.model_dump(exclude_unset=True)
        assert "tool_mappings" in data
        assert len(data["tool_mappings"]) == 1


class TestToggleVirtualServerRequest:
    """Tests for ToggleVirtualServerRequest model."""

    def test_toggle_enabled(self):
        """Test toggle request with enabled=True."""
        req = ToggleVirtualServerRequest(enabled=True)
        assert req.enabled is True

    def test_toggle_disabled(self):
        """Test toggle request with enabled=False."""
        req = ToggleVirtualServerRequest(enabled=False)
        assert req.enabled is False

    def test_toggle_requires_enabled(self):
        """Test that enabled field is required."""
        with pytest.raises(ValidationError):
            ToggleVirtualServerRequest()


class TestToolCatalogEntry:
    """Tests for ToolCatalogEntry model."""

    def test_valid_entry(self):
        """Test creating a valid catalog entry."""
        entry = ToolCatalogEntry(
            tool_name="search",
            server_path="/github",
            server_name="GitHub",
            description="Search GitHub repositories",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            available_versions=["v1.0.0", "v1.5.0"],
        )
        assert entry.tool_name == "search"
        assert len(entry.available_versions) == 2

    def test_entry_defaults(self):
        """Test ToolCatalogEntry default values."""
        entry = ToolCatalogEntry(
            tool_name="get_data",
            server_path="/github",
        )
        assert entry.server_name == ""
        assert entry.description == ""
        assert entry.input_schema == {}
        assert entry.available_versions == []


class TestResolvedTool:
    """Tests for ResolvedTool model."""

    def test_valid_resolved_tool(self):
        """Test creating a valid resolved tool."""
        tool = ResolvedTool(
            name="github_search",
            original_name="search",
            backend_server_path="/github",
            backend_version="v1.5.0",
            description="Search GitHub repos",
            input_schema={"type": "object"},
            required_scopes=["github:read"],
        )
        assert tool.name == "github_search"
        assert tool.original_name == "search"
        assert tool.backend_version == "v1.5.0"
        assert tool.required_scopes == ["github:read"]

    def test_resolved_tool_defaults(self):
        """Test ResolvedTool default values."""
        tool = ResolvedTool(
            name="get_data",
            original_name="get_data",
            backend_server_path="/github",
        )
        assert tool.backend_version is None
        assert tool.description == ""
        assert tool.input_schema == {}
        assert tool.required_scopes == []

    def test_resolved_tool_with_alias(self):
        """Test ResolvedTool where name differs from original_name (aliased)."""
        tool = ResolvedTool(
            name="fetch_data",
            original_name="get_data",
            backend_server_path="/github",
        )
        assert tool.name == "fetch_data"
        assert tool.original_name == "get_data"
