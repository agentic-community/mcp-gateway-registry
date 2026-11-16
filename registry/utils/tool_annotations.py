"""Capture and read the MCP tool fields that describe behavior and output.

An MCP server's ``tools/list`` can give each tool a display ``title``, an
``annotations`` object with behavior hints (``readOnlyHint``,
``destructiveHint``, ``idempotentHint``, ``openWorldHint``) and an
``outputSchema``. The registry used to drop all three. This module copies them
into the stored tool dict and validates them again wherever a reader forwards
them, because a registrant can also write ``tool_list`` directly.

The server declares these values and the registry does not verify them. No
authorization or approval decision may read them.

This module must not import the MCP SDK: the auth server imports
``registry.utils.tool_schema``, which imports this module, and the auth-server
image has no SDK. SDK objects are read through ``getattr``.

MCP spec: https://modelcontextprotocol.io/specification/2025-11-25/server/tools
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

ANNOTATION_HINT_KEYS: tuple[str, ...] = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)
TITLE_KEY: str = "title"
ANNOTATIONS_KEY: str = "annotations"
OUTPUT_SCHEMA_KEY: str = "outputSchema"
MAX_TOOL_TITLE_LENGTH: int = 256


def _clean_title(
    value: object,
) -> str | None:
    """Return a usable title, or None for a non-string, blank or oversized value.

    Args:
        value: A title as received or stored.

    Returns:
        The stripped title, or None.
    """
    if not isinstance(value, str):
        return None
    title = value.strip()
    if not title or len(title) > MAX_TOOL_TITLE_LENGTH:
        return None
    return title


def _clean_annotations(
    raw: object,
) -> dict[str, Any] | None:
    """Keep the standard hint keys whose value is a bool, plus a valid title.

    Unknown keys and wrongly typed values are dropped. A string "true" is not a
    bool, so it is dropped rather than coerced.

    Args:
        raw: An annotations object as received or stored.

    Returns:
        A dict with only valid keys, or None when nothing valid is left.
    """
    if not isinstance(raw, dict):
        return None
    cleaned: dict[str, Any] = {}
    title = _clean_title(raw.get(TITLE_KEY))
    if title:
        cleaned[TITLE_KEY] = title
    for key in ANNOTATION_HINT_KEYS:
        value = raw.get(key)
        if isinstance(value, bool):
            cleaned[key] = value
    if not cleaned:
        return None
    return cleaned


def _sdk_annotations(
    tool: object,
) -> dict[str, Any] | None:
    """Read an SDK Tool's annotations as a dict with MCP (camelCase) keys.

    ``exclude_unset`` keeps an explicit False and leaves out a hint the server
    never sent. ``exclude_none`` treats an explicit null as absent.

    Args:
        tool: An SDK ``Tool`` from tools/list.

    Returns:
        The cleaned annotations, or None when the tool has none.
    """
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return None
    model_dump = getattr(annotations, "model_dump", None)
    if model_dump is None:
        return _clean_annotations(annotations)
    dumped = model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
    return _clean_annotations(dumped)


def tool_metadata_fields(
    tool: object,
) -> dict[str, Any]:
    """Return the title, annotations and outputSchema of an SDK Tool.

    Only the keys the tool has are returned, so a tool without them keeps the
    stored shape it had before this field set existed.

    Args:
        tool: An SDK ``Tool`` from tools/list.

    Returns:
        A dict with any of ``title``, ``annotations`` and ``outputSchema``.

    Example:
        >>> tool_metadata_fields(sdk_tool)  # readOnlyHint=False was sent
        {'annotations': {'readOnlyHint': False}}
    """
    fields: dict[str, Any] = {}
    title = _clean_title(getattr(tool, "title", None))
    if title:
        fields[TITLE_KEY] = title
    annotations = _sdk_annotations(tool)
    if annotations:
        fields[ANNOTATIONS_KEY] = annotations
    output_schema = getattr(tool, "output_schema", None)
    if isinstance(output_schema, dict):
        fields[OUTPUT_SCHEMA_KEY] = output_schema
    return fields


def stored_tool_annotations(
    tool: object,
) -> dict[str, Any] | None:
    """Return a stored tool's annotations after validation.

    Args:
        tool: One entry of a server's stored ``tool_list``.

    Returns:
        The valid annotations, or None.
    """
    if not isinstance(tool, dict):
        return None
    raw = tool.get(ANNOTATIONS_KEY)
    if raw is None:
        return None
    cleaned = _clean_annotations(raw)
    if cleaned is None:
        logger.debug(f"Dropped invalid stored annotations for tool {tool.get('name')!r}")
    return cleaned


def stored_tool_metadata(
    tool: object,
) -> dict[str, Any]:
    """Return a stored tool's title, annotations and outputSchema after validation.

    Use this wherever a stored tool is forwarded to a client, so a value a
    registrant wrote by hand is checked the same way a discovered one is.

    Args:
        tool: One entry of a server's stored ``tool_list``.

    Returns:
        A dict with any of ``title``, ``annotations`` and ``outputSchema``.
    """
    if not isinstance(tool, dict):
        return {}
    fields: dict[str, Any] = {}
    title = _clean_title(tool.get(TITLE_KEY))
    if title:
        fields[TITLE_KEY] = title
    annotations = stored_tool_annotations(tool)
    if annotations:
        fields[ANNOTATIONS_KEY] = annotations
    output_schema = tool.get(OUTPUT_SCHEMA_KEY)
    if isinstance(output_schema, dict):
        fields[OUTPUT_SCHEMA_KEY] = output_schema
    return fields


def tool_metadata_signature(
    tool_list: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    """Summarize each tool's title, annotations and outputSchema, for change detection.

    The health check compares the stored and the fetched signatures and rewrites
    the tool list when they differ. Tools with none of the three fields are left
    out, so a server without them yields [] on both sides and is not rewritten.

    Args:
        tool_list: Tool dicts as the registry stores them.

    Returns:
        Sorted (tool name, canonical JSON of the fields) pairs.
    """
    entries: list[tuple[str, str]] = []
    for tool in tool_list or []:
        fields = stored_tool_metadata(tool)
        if not fields:
            continue
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        entries.append((str(tool.get("name", "")), canonical))
    return sorted(entries)
