"""Hash and compare MCP tool lists, for tool definition drift detection.

The health loop and the live tools route fetch a server's ``tools/list`` and
need to know whether it differs from the stored copy. This module reduces each
tool to the validated client projection, including optional metadata and MCP
Apps page and audience fields. It also reports added, removed and changed tools
by name. Discovery and registration representations share the same projection.

This module must not import the MCP SDK, for the same reason as
``registry.utils.tool_annotations``: it may end up on the auth server's import
path, and the auth-server image has no SDK.
"""

import hashlib
import json
from typing import Any

from pydantic import (
    BaseModel,
    Field,
)

from .tool_schema import stored_tool_to_mcp

# Bump when the set of fields the hash covers changes, so every stored list
# compares as changed once instead of matching by accident.
TOOL_HASH_FIELDS_VERSION: str = "2"


class ToolListChanges(BaseModel):
    """Which tools differ between two tool lists, by name."""

    added: list[str] = Field(
        default_factory=list,
        description="Names present only in the new list",
    )
    removed: list[str] = Field(
        default_factory=list,
        description="Names present only in the old list",
    )
    changed: list[str] = Field(
        default_factory=list,
        description="Names in both lists whose definition hash differs",
    )

    @property
    def names_changed(self) -> bool:
        """True when a tool was added or removed, which changes scopes."""
        return bool(self.added or self.removed)

    @property
    def any_change(self) -> bool:
        """True when any tool was added, removed or changed."""
        return bool(self.added or self.removed or self.changed)


def _hashes_by_name(
    tool_list: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Map each tool name to its definition hash. A repeated name keeps the last tool."""
    hashes: dict[str, str] = {}
    for tool in tool_list or []:
        if not isinstance(tool, dict):
            continue
        hashes[str(tool.get("name", ""))] = tool_definition_hash(tool)
    return hashes


def tool_definition_hash(
    tool: dict[str, Any],
) -> str:
    """Return the SHA-256 hex digest of one tool's canonical JSON.

    Args:
        tool: One tool dict as the registry stores it.

    Returns:
        A 64-character hex digest.
    """
    text = json.dumps(
        stored_tool_to_mcp(tool),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tool_list_hash(
    tool_list: list[dict[str, Any]] | None,
) -> str:
    """Return one SHA-256 hex digest for a whole tool list.

    Tool order does not affect the result, and entries that are not dicts are
    skipped. An empty or missing list hashes to the same value every time.

    Args:
        tool_list: Tool dicts as the registry stores them.

    Returns:
        A 64-character hex digest.

    Example:
        >>> tool_list_hash([{"name": "a"}, {"name": "b"}]) == tool_list_hash(
        ...     [{"name": "b"}, {"name": "a"}]
        ... )
        True
    """
    entries: list[str] = []
    for tool in tool_list or []:
        if isinstance(tool, dict):
            entries.append(tool_definition_hash(tool))
    entries.sort()
    joined = TOOL_HASH_FIELDS_VERSION + "\n" + "\n".join(entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compare_tool_lists(
    old_list: list[dict[str, Any]] | None,
    new_list: list[dict[str, Any]] | None,
) -> ToolListChanges:
    """Report which tools were added, removed or changed, by name.

    Args:
        old_list: The stored tool list.
        new_list: The freshly fetched tool list.

    Returns:
        A ToolListChanges with sorted name lists.
    """
    old_by_name = _hashes_by_name(old_list)
    new_by_name = _hashes_by_name(new_list)
    added = sorted(set(new_by_name) - set(old_by_name))
    removed = sorted(set(old_by_name) - set(new_by_name))
    changed: list[str] = []
    for name in sorted(set(old_by_name) & set(new_by_name)):
        if old_by_name[name] != new_by_name[name]:
            changed.append(name)
    return ToolListChanges(added=added, removed=removed, changed=changed)
