"""Unit tests for MCP Apps detection (registry/core/mcp_apps.py)."""

import pytest
from mcp import Client
from mcp.types import Tool

from registry.core.mcp_apps import (
    MCP_APPS_EXTENSION_ID,
    MCP_APPS_MAX_URI_LENGTH,
    MCP_APPS_MIME_TYPE,
    count_mcp_app_tools,
    is_model_visible,
    mcp_app_signature,
    mcp_apps_client_extensions,
    model_visible_tools,
    stored_ui_resource_uri,
    tool_ui_fields,
)
from registry.core.mcp_client import _extract_tool_details


def _tool(
    meta: dict | None = None,
    name: str = "open_view",
) -> Tool:
    """Build an SDK tool with the given _meta."""
    return Tool.model_validate(
        {"name": name, "inputSchema": {"type": "object"}, "_meta": meta},
    )


@pytest.mark.unit
class TestToolUiFields:
    """Tests for tool_ui_fields."""

    def test_reads_spec_resource_uri_and_visibility(self):
        """A tool with _meta.ui.resourceUri keeps the address and visibility."""
        tool = _tool({"ui": {"resourceUri": "ui://budget/view.html", "visibility": ["app"]}})

        result = tool_ui_fields(tool)

        assert result == {
            "ui_resource_uri": "ui://budget/view.html",
            "ui_visibility": ["app"],
        }

    def test_defaults_visibility_to_model_and_app(self):
        """A missing visibility list gets the spec default."""
        tool = _tool({"ui": {"resourceUri": "ui://budget/view.html"}})

        assert tool_ui_fields(tool)["ui_visibility"] == ["model", "app"]

    def test_drops_unknown_visibility_values(self):
        """Values the spec does not define are removed."""
        tool = _tool({"ui": {"resourceUri": "ui://a/b", "visibility": ["app", "admin"]}})

        assert tool_ui_fields(tool)["ui_visibility"] == ["app"]

    def test_reads_deprecated_flat_key(self):
        """Older servers use the flat _meta["ui/resourceUri"] key."""
        tool = _tool({"ui/resourceUri": "ui://legacy/page"})

        assert tool_ui_fields(tool)["ui_resource_uri"] == "ui://legacy/page"

    def test_tool_without_meta_has_no_ui_fields(self):
        """A plain tool gets no MCP Apps keys."""
        assert tool_ui_fields(_tool(None)) == {}

    @pytest.mark.parametrize(
        "uri",
        [
            "https://example.com/page.html",
            "ui://",
            "",
            42,
            ["ui://a/b"],
            "ui://" + "a" * MCP_APPS_MAX_URI_LENGTH,
        ],
    )
    def test_rejects_invalid_resource_uri(self, uri):
        """Anything other than a bounded ui:// string is ignored."""
        tool = _tool({"ui": {"resourceUri": uri}})

        assert tool_ui_fields(tool) == {}

    def test_does_not_store_other_meta_keys(self):
        """Only the two validated fields leave the backend's _meta."""
        tool = _tool({"ui": {"resourceUri": "ui://a/b", "csp": {"connectDomains": ["x"]}}})

        assert set(tool_ui_fields(tool)) == {"ui_resource_uri", "ui_visibility"}


@pytest.mark.unit
class TestCountAndSignature:
    """Tests for count_mcp_app_tools and mcp_app_signature."""

    def test_counts_only_valid_ui_tools(self):
        """Tools without a page, or with a caller-supplied bad address, do not count."""
        tool_list = [
            {"name": "a", "ui_resource_uri": "ui://x/a"},
            {"name": "b"},
            {"name": "c", "ui_resource_uri": "javascript:alert(1)"},
            "not-a-dict",
        ]

        assert count_mcp_app_tools(tool_list) == 1

    def test_count_handles_none(self):
        """A server with no stored tool list counts zero."""
        assert count_mcp_app_tools(None) == 0

    def test_signature_ignores_order(self):
        """Two lists with the same UI tools in a different order match."""
        first = [
            {"name": "a", "ui_resource_uri": "ui://x/a", "ui_visibility": ["model"]},
            {"name": "b", "ui_resource_uri": "ui://x/b", "ui_visibility": ["app"]},
        ]

        assert mcp_app_signature(first) == mcp_app_signature(list(reversed(first)))

    def test_signature_changes_when_tool_gains_page(self):
        """Adding a page to an existing tool changes the signature."""
        before = [{"name": "a"}]
        after = [{"name": "a", "ui_resource_uri": "ui://x/a", "ui_visibility": ["model"]}]

        assert mcp_app_signature(before) != mcp_app_signature(after)


@pytest.mark.unit
class TestModelVisibility:
    """Tests for is_model_visible, model_visible_tools and stored_ui_resource_uri."""

    def test_app_only_tool_is_hidden(self):
        """A tool with ui_visibility ["app"] is not model-visible."""
        assert is_model_visible({"name": "save", "ui_visibility": ["app"]}) is False

    def test_model_and_app_tool_is_visible(self):
        """A tool the model and the page share stays visible."""
        assert is_model_visible({"name": "open", "ui_visibility": ["model", "app"]}) is True

    def test_plain_tool_is_visible(self):
        """An ordinary tool with no ui_visibility stays visible."""
        assert is_model_visible({"name": "plain"}) is True

    def test_model_visible_tools_keeps_order(self):
        """Hidden tools are dropped and the rest keep their order."""
        tools = [
            {"name": "a"},
            {"name": "b", "ui_visibility": ["app"]},
            {"name": "c", "ui_visibility": ["model"]},
        ]

        assert [t["name"] for t in model_visible_tools(tools)] == ["a", "c"]

    def test_stored_uri_rejects_forged_value(self):
        """A caller-supplied non-ui:// value is not returned."""
        assert stored_ui_resource_uri({"ui_resource_uri": "https://evil.example"}) is None
        assert stored_ui_resource_uri({"ui_resource_uri": "ui://a/b"}) == "ui://a/b"


@pytest.mark.unit
class TestDiscoveryIntegration:
    """Tests that discovery declares the extension and keeps the fields."""

    def test_client_advertises_mcp_apps_extension(self):
        """The SDK client built with our extensions advertises the UI capability."""
        client = Client("http://127.0.0.1:1/mcp", extensions=mcp_apps_client_extensions())

        assert client._folded_extensions.ad == {
            MCP_APPS_EXTENSION_ID: {"mimeTypes": [MCP_APPS_MIME_TYPE]},
        }

    def test_extract_tool_details_keeps_ui_fields(self):
        """_extract_tool_details stores the UI fields next to the schema."""
        tools = [
            _tool({"ui": {"resourceUri": "ui://budget/view.html"}}, name="budget"),
            _tool(None, name="plain"),
        ]

        details = _extract_tool_details(tools)

        assert details[0]["ui_resource_uri"] == "ui://budget/view.html"
        assert "ui_resource_uri" not in details[1]
