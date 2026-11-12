"""Read a stored tool's input schema, and turn stored tools into MCP tool entries.

Tools reach a server's ``tool_list`` from two writers. A registration or an
MCP ``tools/list`` copy uses the MCP field name ``inputSchema``. Health-loop
discovery (``registry/core/mcp_client.py``) has always written ``schema``, the
alias of ``ToolDescription.tool_schema``. Readers that looked only for
``inputSchema`` saw an empty schema for every discovered tool, so virtual
server mappings and the tool catalog lost the argument definitions.

The MCP spec requires every tool's ``inputSchema`` to have ``"type": "object"``
at the root. ``mcp_input_schema`` applies that rule. Virtual server mappings
(``registry/core/nginx_service.py``) and the pre-consent tool list (issue
#1986) both use it, so a tool's schema reads the same on either route.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

INPUT_SCHEMA_KEYS: tuple[str, ...] = ("inputSchema", "input_schema", "schema")

# JSON Schema keywords that hold reusable definitions (2020-12 and draft-07).
SCHEMA_DEFINITION_KEYS: tuple[str, ...] = ("$defs", "definitions")


def _wrap_non_object_schema(
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Nest a non-object schema under properties.value, keeping its definitions.

    References such as "#/$defs/Item" resolve against the document root. The
    definitions therefore move to the new root, or the references would break.
    """
    inner = dict(input_schema)
    wrapper: dict[str, Any] = {"type": "object", "properties": {"value": inner}}
    for key in SCHEMA_DEFINITION_KEYS:
        if key in inner:
            wrapper[key] = inner.pop(key)
    return wrapper


def _stored_description(
    tool: dict[str, Any],
) -> str:
    """Return a stored tool's description, whichever writer stored it.

    A registration stores ``description``. Health discovery stores the parsed
    docstring under ``parsed_description.main``.
    """
    description = tool.get("description")
    if isinstance(description, str) and description:
        return description
    parsed = tool.get("parsed_description")
    if isinstance(parsed, dict):
        main = parsed.get("main")
        if isinstance(main, str):
            return main
    return ""


def stored_input_schema(
    tool: dict[str, Any],
) -> dict[str, Any]:
    """Return the tool's input schema, or an empty dict when it has none.

    Args:
        tool: One entry of a server's stored ``tool_list``.

    Returns:
        The first non-empty dict found under ``inputSchema``, ``input_schema``
        or ``schema``, in that order.
    """
    for key in INPUT_SCHEMA_KEYS:
        value = tool.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def mcp_input_schema(
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Ensure inputSchema conforms to MCP spec by adding 'type': 'object' if missing.

    The MCP spec requires all tool inputSchema definitions to have "type": "object"
    at the top level. This function ensures backend tool schemas are compliant.

    Args:
        input_schema: The input schema from a backend tool

    Returns:
        MCP-compliant schema with "type": "object" at top level
    """
    if not input_schema:
        return {"type": "object", "properties": {}}

    # If schema already has "type": "object", return as-is
    if input_schema.get("type") == "object":
        return input_schema

    # A type list that allows an object (for example ["object", "null"]) already
    # describes the arguments object. MCP 2026-07-28 requires the root type to
    # be exactly "object", so narrow it instead of nesting the arguments.
    root_type = input_schema.get("type")
    if isinstance(root_type, list) and "object" in root_type:
        schema_copy = input_schema.copy()
        schema_copy["type"] = "object"
        return schema_copy

    # If schema has "type" but it's not "object", wrap it
    if "type" in input_schema:
        logger.warning(
            f"Tool inputSchema has non-object type '{input_schema.get('type')}'. "
            "Wrapping in object schema to comply with MCP spec."
        )
        return _wrap_non_object_schema(input_schema)

    # If no "type" field but has "properties", add "type": "object"
    if "properties" in input_schema or "additionalProperties" in input_schema:
        schema_copy = input_schema.copy()
        schema_copy["type"] = "object"
        return schema_copy

    # Default: wrap unknown schema structure
    logger.warning(
        "Tool inputSchema missing 'type' field and has unexpected structure. "
        "Adding 'type': 'object' to comply with MCP spec."
    )
    schema_copy = input_schema.copy()
    schema_copy["type"] = "object"
    return schema_copy


def stored_tool_to_mcp(
    tool: object,
) -> dict[str, Any] | None:
    """Convert one stored ``tool_list`` entry to an MCP ``tools/list`` entry.

    Only ``name``, ``description`` and ``inputSchema`` are copied, the same
    fields a virtual server mapping publishes for a backend that has not been
    connected yet.

    Args:
        tool: One entry of a server's stored ``tool_list``.

    Returns:
        A dict with ``name``, ``description`` and ``inputSchema``, or None when
        the entry is not a dict or has no usable name.

    Example:
        >>> stored_tool_to_mcp({"name": "t", "parsed_description": {"main": "Do it"}})
        {'name': 't', 'description': 'Do it', 'inputSchema': {'type': 'object', 'properties': {}}}
    """
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "name": name,
        "description": _stored_description(tool),
        "inputSchema": mcp_input_schema(stored_input_schema(tool)),
    }
