"""Tests for MCP Registry server.json schema detection and transformation."""

import pytest

from registry.schemas.mcp_registry_schema import McpRegistryServerJson
from registry.schemas.mcp_registry_transform import (
    _slugify,
    is_mcp_registry_schema,
    transform_mcp_registry_to_internal,
)


class TestIsMcpRegistrySchema:
    """Tests for schema detection."""

    def test_positive_with_full_url(self):
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
        }
        assert is_mcp_registry_schema(data) is True

    def test_positive_with_partial_url(self):
        data = {
            "$schema": "https://modelcontextprotocol/registry/server.schema.json",
            "name": "test",
        }
        assert is_mcp_registry_schema(data) is True

    def test_negative_no_schema(self):
        data = {"name": "test", "path": "/test"}
        assert is_mcp_registry_schema(data) is False

    def test_negative_different_schema(self):
        data = {"$schema": "https://json-schema.org/draft/2020-12/schema", "name": "test"}
        assert is_mcp_registry_schema(data) is False

    def test_negative_non_string_schema(self):
        data = {"$schema": 123, "name": "test"}
        assert is_mcp_registry_schema(data) is False


class TestSlugify:
    """Tests for path slug generation."""

    def test_namespaced_name(self):
        assert _slugify("io.example/calculator-mcp") == "io-example-calculator-mcp"

    def test_simple_name(self):
        assert _slugify("my-server") == "my-server"

    def test_spaces_and_caps(self):
        assert _slugify("My Cool Server") == "my-cool-server"

    def test_special_chars(self):
        assert _slugify("server@v1.0!beta") == "server-v1-0-beta"

    def test_multiple_separators(self):
        assert _slugify("io.example///multi") == "io-example-multi"


class TestTransformRemoteServer:
    """Tests for transforming remote server configs."""

    def test_full_hybrid_json(self):
        """Transform a hybrid JSON with both upstream and gateway fields."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/calculator-mcp",
            "title": "Calculator MCP Server",
            "description": "MCP server providing arithmetic operations",
            "version": "1.0.0",
            "repository": {"url": "https://github.com/example-org/calc", "source": "github"},
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": "https://gateway-dev.example.com/mcp",
                    "headers": [
                        {"name": "Authorization", "value": "Bearer {token}", "isSecret": True}
                    ],
                }
            ],
            "proxy_pass_url": "https://gateway-dev.example.com/mcp",
            "tags": ["calculator", "mcp"],
            "num_tools": 3,
            "tool_list": [{"name": "echo", "description": "Echo tool"}],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["path"] == "/io-example-calculator-mcp"
        assert result["name"] == "Calculator MCP Server"
        assert result["server_name"] == "Calculator MCP Server"
        assert result["description"] == "MCP server providing arithmetic operations"
        assert result["proxy_pass_url"] == "https://gateway-dev.example.com/mcp"
        assert result["deployment"] == "remote"
        assert result["auth_scheme"] == "bearer"
        assert result["num_tools"] == 3
        assert result["tool_list"] == [{"name": "echo", "description": "Echo tool"}]
        assert "calculator" in result["tags"]
        assert "mcp-registry-spec" in result["tags"]

    def test_proxy_url_from_remotes(self):
        """proxy_pass_url derived from remotes[0].url when not explicit."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/server",
            "remotes": [{"type": "streamable-http", "url": "https://remote.example.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["proxy_pass_url"] == "https://remote.example.com/mcp"

    def test_explicit_path_overrides_derived(self):
        """Explicit path field overrides name-derived path."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/calculator-mcp",
            "path": "/my-custom-path",
            "remotes": [{"type": "streamable-http", "url": "https://example.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["path"] == "/my-custom-path"

    def test_name_used_when_no_title(self):
        """server_name falls back to name when title is missing."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/my-server",
            "remotes": [{"type": "sse", "url": "https://example.com/sse"}],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["server_name"] == "io.example/my-server"

    def test_supported_transports_from_remotes(self):
        """Transports derived from remotes when not explicit."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "remotes": [
                {"type": "streamable-http", "url": "https://a.com/mcp"},
                {"type": "sse", "url": "https://a.com/sse"},
            ],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["supported_transports"] == ["streamable-http", "sse"]
        assert result["transport"] == "streamable-http"

    def test_visibility_and_groups(self):
        """Visibility and allowed_groups passed through."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
            "visibility": "group-restricted",
            "allowed_groups": ["engineering", "platform"],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["visibility"] == "group-restricted"
        assert result["allowed_groups"] == ["engineering", "platform"]


class TestTransformLocalServer:
    """Tests for transforming local/stdio server configs."""

    def test_packages_only_infers_local(self):
        """A packages-only config with stdio transport becomes local."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/local-tool",
            "packages": [
                {
                    "registryType": "pypi",
                    "identifier": "local-tool",
                    "version": "2.0.0",
                    "transport": {"type": "stdio"},
                    "runtimeHint": "uvx",
                    "environmentVariables": [
                        {"name": "API_KEY", "isRequired": True, "isSecret": True},
                        {"name": "REGION", "default": "us-east-1", "isRequired": False},
                    ],
                }
            ],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["deployment"] == "local"
        assert result["proxy_pass_url"] is None
        assert result["local_runtime"]["type"] == "uvx"
        assert result["local_runtime"]["package"] == "local-tool"
        assert result["local_runtime"]["version"] == "2.0.0"
        assert "API_KEY" in result["local_runtime"]["required_env"]
        assert result["local_runtime"]["env"]["REGION"] == "us-east-1"

    def test_explicit_deployment_local(self):
        """Explicit deployment=local forces local even with remotes."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "deployment": "local",
            "packages": [
                {
                    "registryType": "npm",
                    "identifier": "@example/tool",
                    "version": "1.0.0",
                    "transport": {"type": "stdio"},
                    "runtimeHint": "npx",
                }
            ],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["deployment"] == "local"
        assert result["local_runtime"]["type"] == "npx"
        assert result["local_runtime"]["package"] == "@example/tool"


class TestMetadataPreservation:
    """Tests for metadata preservation of upstream fields."""

    def test_repository_preserved(self):
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "repository": {"url": "https://github.com/org/repo", "source": "github"},
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)
        spec = result["metadata"]["mcp_registry_spec"]

        assert spec["repository"]["url"] == "https://github.com/org/repo"
        assert spec["repository"]["source"] == "github"

    def test_packages_preserved(self):
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "packages": [
                {"registryType": "pypi", "identifier": "my-tool", "version": "1.0.0"}
            ],
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)
        spec = result["metadata"]["mcp_registry_spec"]

        assert len(spec["packages"]) == 1
        assert spec["packages"][0]["registryType"] == "pypi"

    def test_meta_preserved(self):
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "_meta": {"io.modelcontextprotocol.registry/publisher-provided": {"tool": "manual"}},
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)
        spec = result["metadata"]["mcp_registry_spec"]

        assert "_meta" in spec
        assert "io.modelcontextprotocol.registry/publisher-provided" in spec["_meta"]

    def test_original_name_preserved(self):
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "io.example/my-namespaced-server",
            "title": "My Server",
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)
        spec = result["metadata"]["mcp_registry_spec"]

        assert spec["original_name"] == "io.example/my-namespaced-server"

    def test_user_metadata_merged(self):
        """User-provided metadata is preserved alongside mcp_registry_spec."""
        data = {
            "$schema": "https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json",
            "name": "test",
            "metadata": {"custom_field": "custom_value", "team": "platform"},
            "remotes": [{"type": "streamable-http", "url": "https://a.com/mcp"}],
        }

        result = transform_mcp_registry_to_internal(data)

        assert result["metadata"]["custom_field"] == "custom_value"
        assert result["metadata"]["team"] == "platform"
        assert "mcp_registry_spec" in result["metadata"]


class TestPydanticModel:
    """Tests for the McpRegistryServerJson Pydantic model validation."""

    def test_minimal_valid(self):
        data = {"name": "test-server"}
        model = McpRegistryServerJson.model_validate(data)
        assert model.name == "test-server"

    def test_full_valid(self):
        data = {
            "$schema": "https://example.com/modelcontextprotocol/registry/schema.json",
            "name": "io.example/server",
            "title": "Example Server",
            "description": "A test server",
            "version": "2.0.0",
            "repository": {"url": "https://github.com/org/repo", "source": "github"},
            "packages": [{"registryType": "pypi", "identifier": "pkg", "version": "1.0.0"}],
            "remotes": [{"type": "sse", "url": "https://a.com/sse"}],
            "_meta": {"key": "value"},
            "tags": ["tag1", "tag2"],
            "visibility": "public",
        }
        model = McpRegistryServerJson.model_validate(data)
        assert model.schema_url == "https://example.com/modelcontextprotocol/registry/schema.json"
        assert model.name == "io.example/server"
        assert model.title == "Example Server"
        assert model.meta == {"key": "value"}
        assert len(model.packages) == 1
        assert model.packages[0].registry_type == "pypi"
