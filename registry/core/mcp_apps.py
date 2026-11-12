"""MCP Apps detection for registered MCP servers.

MCP Apps (SEP-1865) lets a tool point at an interactive HTML page that the
chat client renders. The tool carries ``_meta.ui.resourceUri`` with a
``ui://`` address, and the client fetches that page with ``resources/read``.

The registry never renders these pages. It only records which tools have one,
so the UI can mark servers that ship an MCP App. Spec (stable 2026-01-26):
https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx
"""

import logging
from typing import Any

from mcp.client.extension import ClientExtension
from mcp.types import Tool

logger = logging.getLogger(__name__)

# The extension a client declares in its capabilities to say it can render
# MCP Apps. A server that follows the spec only lists its UI tools to a client
# that declares it, so the health check must declare it too.
MCP_APPS_EXTENSION_ID: str = "io.modelcontextprotocol/ui"
MCP_APPS_MIME_TYPE: str = "text/html;profile=mcp-app"
MCP_APPS_URI_SCHEME: str = "ui://"
# The spec does not cap the URI. This bound stops a backend from storing an
# arbitrarily large string in every tool record.
MCP_APPS_MAX_URI_LENGTH: int = 2048
# Who can call a tool: the model, the app's page, or both (the spec default).
MCP_APPS_VISIBILITY_VALUES: tuple[str, ...] = ("model", "app")
# Older servers put the address in a flat key that the spec now deprecates.
MCP_APPS_DEPRECATED_URI_KEY: str = "ui/resourceUri"

# Keys this module adds to a stored tool dict.
TOOL_UI_RESOURCE_URI_KEY: str = "ui_resource_uri"
TOOL_UI_VISIBILITY_KEY: str = "ui_visibility"


def _is_valid_ui_resource_uri(
    value: Any,
) -> bool:
    """Return True when value is a usable ui:// resource address.

    Args:
        value: The candidate taken from the tool's _meta.

    Returns:
        True for a non-empty string that starts with ui:// and fits the length cap.
    """
    if not isinstance(value, str):
        return False
    if not value.startswith(MCP_APPS_URI_SCHEME):
        return False
    if len(value) <= len(MCP_APPS_URI_SCHEME):
        return False
    return len(value) <= MCP_APPS_MAX_URI_LENGTH


def _clean_visibility(
    value: Any,
) -> list[str]:
    """Keep only the visibility values the spec defines.

    Args:
        value: The tool's _meta.ui.visibility, which may be missing or malformed.

    Returns:
        The valid values in spec order, or the spec default when none are valid.
    """
    if not isinstance(value, list):
        return list(MCP_APPS_VISIBILITY_VALUES)
    cleaned = [item for item in MCP_APPS_VISIBILITY_VALUES if item in value]
    if not cleaned:
        return list(MCP_APPS_VISIBILITY_VALUES)
    return cleaned


def _ui_block(
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Return the tool's _meta.ui object, or an empty dict when it is not one."""
    ui = meta.get("ui")
    if isinstance(ui, dict):
        return ui
    return {}


class _McpAppsUiExtension(ClientExtension):
    """Declares MCP Apps support when the registry connects to a server."""

    identifier = MCP_APPS_EXTENSION_ID

    def settings(self) -> dict[str, Any]:
        """Advertise the one page format the spec defines."""
        return {"mimeTypes": [MCP_APPS_MIME_TYPE]}


def mcp_apps_client_extensions() -> list[ClientExtension]:
    """Return the client extensions the registry declares during discovery.

    Returns:
        A list holding the MCP Apps UI extension.
    """
    return [_McpAppsUiExtension()]


def tool_ui_fields(
    tool: Tool,
) -> dict[str, Any]:
    """Pull the MCP Apps fields from one SDK tool.

    Only a validated address and visibility list are kept. The rest of _meta
    comes from the backend and is never stored.

    Args:
        tool: A tool returned by tools/list.

    Returns:
        {"ui_resource_uri": ..., "ui_visibility": [...]} when the tool has a
        valid UI page, otherwise an empty dict.
    """
    meta = tool.meta if isinstance(tool.meta, dict) else {}
    ui = _ui_block(meta)
    uri = ui.get("resourceUri")
    if uri is None:
        uri = meta.get(MCP_APPS_DEPRECATED_URI_KEY)
    if uri is None:
        return {}
    if not _is_valid_ui_resource_uri(uri):
        logger.debug(f"Ignoring invalid MCP Apps resource URI on tool {tool.name}")
        return {}
    return {
        TOOL_UI_RESOURCE_URI_KEY: uri,
        TOOL_UI_VISIBILITY_KEY: _clean_visibility(ui.get("visibility")),
    }


def count_mcp_app_tools(
    tool_list: list[dict[str, Any]] | None,
) -> int:
    """Count the stored tools that open an MCP App page.

    A registration request can supply its own tool list, so the address is
    re-checked here rather than trusted because the key is present.

    Args:
        tool_list: Tool dicts as the registry stores them.

    Returns:
        The number of tools with a valid ui_resource_uri.
    """
    count = 0
    for tool in tool_list or []:
        if isinstance(tool, dict) and stored_ui_resource_uri(tool):
            count += 1
    return count


def stored_ui_resource_uri(
    tool: dict[str, Any],
) -> str | None:
    """Return a stored tool's ui:// address if it is valid, else None.

    Use this, not the raw key, wherever the address leaves the registry. A
    registration request can supply its own tool list, so the stored value is
    not trusted just because discovery normally validates it.

    Args:
        tool: A tool dict as the registry stores it.

    Returns:
        The address, or None when the tool has no valid page.
    """
    uri = tool.get(TOOL_UI_RESOURCE_URI_KEY)
    if _is_valid_ui_resource_uri(uri):
        return uri
    return None


def is_model_visible(
    tool: dict[str, Any],
) -> bool:
    """Return False only for a tool the spec hides from the model.

    A tool whose ui_visibility is ["app"] exists for its MCP App page to call.
    The spec says the model must not see it, so model-facing discovery (search,
    the virtual-server tool picker) leaves it out. A tool with no
    ui_visibility is an ordinary tool and stays visible.

    Args:
        tool: A tool dict as the registry stores it.

    Returns:
        True unless the tool has a ui_visibility list without "model".
    """
    visibility = tool.get(TOOL_UI_VISIBILITY_KEY)
    if not isinstance(visibility, list):
        return True
    return "model" in visibility


def model_visible_tools(
    tool_list: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Drop the tools that only an MCP App page may call.

    Args:
        tool_list: Tool dicts as the registry stores them.

    Returns:
        The tools the model is allowed to see, in their original order.
    """
    visible: list[dict[str, Any]] = []
    for tool in tool_list or []:
        if isinstance(tool, dict) and is_model_visible(tool):
            visible.append(tool)
    return visible


def mcp_app_signature(
    tool_list: list[dict[str, Any]] | None,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Summarize which tools have a UI page, for change detection.

    The health check rewrites the stored tool list only when something changed.
    Comparing two signatures tells it when a tool gained, lost or moved its page,
    which a tool count alone would miss.

    Args:
        tool_list: Tool dicts as the registry stores them.

    Returns:
        Sorted (tool name, resource URI, visibility) entries for the UI tools.
    """
    entries: list[tuple[str, str, tuple[str, ...]]] = []
    for tool in tool_list or []:
        if not isinstance(tool, dict):
            continue
        uri = tool.get(TOOL_UI_RESOURCE_URI_KEY)
        if not uri:
            continue
        visibility = tuple(tool.get(TOOL_UI_VISIBILITY_KEY) or [])
        entries.append((str(tool.get("name", "")), str(uri), visibility))
    return sorted(entries)
