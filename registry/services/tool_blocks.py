"""Apply per-tool security blocks to read projections.

A tool blocked by the security scanner (or by an admin) is refused on
``tools/call`` and hidden from the proxy's ``tools/list``. Every other read
surface used to ignore the block, which leaked in two different ways:

1. ``POST /api/search/semantic`` returned the blocked tool AND its description.
   ``intelligent_tool_finder`` is built on that endpoint, so an agent searching
   for a capability got the blocked tool's description handed to the model. A
   tool earns a ``HIGH:PROMPT INJECTION`` block precisely because its
   description carries the injection, so this routed around the block entirely.
2. ``GET /api/servers`` reported the tool as present with no indication it was
   blocked, so the UI told an operator a tool was available while every call to
   it returned 403.

Those two want opposite treatments, which is why there are two functions here:

- Model-facing projections (search, tool catalog, virtual servers) **hide** the
  tool. The model must never see the description.
- Operator-facing projections (server listing and detail, which feed the admin
  UI) **keep** the entry and mark it. Hiding it there would leave nobody able to
  see which tools were blocked or why, since there is no management UI yet.

Both read block state fresh per request. There is no cache to invalidate, which
matches how the proxy enforces on ``tools/call``.
"""

import logging
from typing import Any

from ..repositories.factory import get_server_repository

logger = logging.getLogger(__name__)


def _tool_name(tool: dict[str, Any]) -> str | None:
    """Pull the tool name out of either projection shape.

    Search results use ``tool_name``; server documents use ``name``.
    """
    if not isinstance(tool, dict):
        return None
    name = tool.get("tool_name") or tool.get("name")
    return name if isinstance(name, str) and name else None


async def _blocked_map(server_path: str) -> dict[str, dict[str, Any]]:
    """Blocked tool name -> its override entry, for one server.

    Returns an empty map when the path is unknown or nothing is blocked. A
    lookup failure is logged and treated as "nothing blocked": these are read
    projections, and failing the whole listing closed would take the registry UI
    down over a per-tool annotation. Enforcement still fails closed on
    ``tools/call``, which is the control that matters.
    """
    if not server_path:
        return {}
    try:
        overrides = await get_server_repository().get_tool_overrides(server_path)
    except Exception as exc:
        logger.error(f"tool_blocks: override lookup failed for {server_path}: {exc}")
        return {}
    return {
        name: entry
        for name, entry in (overrides or {}).items()
        if isinstance(entry, dict) and entry.get("blocked") is True
    }


async def hide_blocked_tools(
    server_path: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Drop blocked tools from a model-facing projection.

    Use this wherever the result can reach an LLM: semantic search, the tool
    catalog, virtual-server tool lists.

    Args:
        server_path: Registered server path, for example "/context7".
        tools: The tool entries to filter.

    Returns:
        The tools that are not blocked.
    """
    if not tools:
        return []

    blocked = await _blocked_map(server_path)
    if not blocked:
        return list(tools)

    kept = [t for t in tools if _tool_name(t) not in blocked]
    dropped = len(tools) - len(kept)
    if dropped:
        logger.info(
            f"tool_blocks: hid {dropped} blocked tool(s) from a read projection for "
            f"{server_path}: {sorted(blocked)}"
        )
    return kept


async def annotate_blocked_tools(
    server_path: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Mark blocked tools in an operator-facing projection.

    Adds ``blocked`` (always, so the field is never missing) and
    ``block_reason`` / ``block_source`` on the blocked ones, letting the UI grey
    the entry and say why rather than presenting it as callable.

    Args:
        server_path: Registered server path, for example "/context7".
        tools: The tool entries to annotate.

    Returns:
        Fresh copies of the entries, annotated. The inputs are not mutated.
    """
    if not tools:
        return []

    blocked = await _blocked_map(server_path)
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        entry = dict(tool)
        override = blocked.get(_tool_name(tool) or "")
        entry["blocked"] = override is not None
        if override is not None:
            entry["block_reason"] = override.get("reason")
            entry["block_source"] = override.get("source")
        out.append(entry)
    return out


async def blocked_names_for(
    server_path: str,
    cache: dict[str, set[str]],
) -> set[str]:
    """Blocked tool names for one server, memoised for the current request.

    The flat tool-search results iterate tools across many servers, so a lookup
    per tool would issue one query per row. ``cache`` is a plain dict the caller
    creates per request and passes back in; it holds one entry per server path.

    Args:
        server_path: Registered server path, for example "/context7".
        cache: Per-request memo, mutated in place.

    Returns:
        The blocked tool names for that server.
    """
    if server_path in cache:
        return cache[server_path]
    names = set((await _blocked_map(server_path)).keys())
    cache[server_path] = names
    return names
