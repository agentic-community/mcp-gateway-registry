"""Read a stored tool's input schema, whichever key it was saved under.

Tools reach a server's ``tool_list`` from two writers. A registration or an
MCP ``tools/list`` copy uses the MCP field name ``inputSchema``. Health-loop
discovery (``registry/core/mcp_client.py``) has always written ``schema``, the
alias of ``ToolDescription.tool_schema``. Readers that looked only for
``inputSchema`` saw an empty schema for every discovered tool, so virtual
server mappings and the tool catalog lost the argument definitions.
"""

from typing import Any

INPUT_SCHEMA_KEYS: tuple[str, ...] = ("inputSchema", "input_schema", "schema")


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
