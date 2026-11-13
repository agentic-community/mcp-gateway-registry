"""Turn one OpenAPI spec into a FastMCP server, with the target's filters and overrides.

The preview and the hub build both call ``convert_target``, so what the admin
sees in the preview is what the hub serves.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx2
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import (
    MCPType,
    OpenAPITool,
)
from fastmcp.utilities.openapi import HTTPRoute
from mcp_types import ToolAnnotations
from models import (
    PreviewRequest,
    PreviewResponse,
    TargetOptions,
)

logger = logging.getLogger(__name__)


HTTP_METHODS: tuple[str, ...] = (
    "get",
    "put",
    "post",
    "delete",
    "options",
    "head",
    "patch",
    "trace",
)
READ_ONLY_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})
DESTRUCTIVE_METHODS: frozenset[str] = frozenset({"DELETE"})
MAX_DESCRIPTION_OVERRIDE_CHARS: int = 2000
UNSAFE_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.\-]+")
MAX_BASE_TOOL_NAME_CHARS: int = 56
# The preview never sends a request; this base URL only satisfies the client.
PREVIEW_BASE_URL: str = "http://preview.invalid"


class ConversionError(Exception):
    """The spec could not be converted. The message is safe to return to an admin."""


def _operation_keys(
    method: str,
    path: str,
    operation_id: str | None,
) -> set[str]:
    """The names an include_operations entry may use for one operation."""
    keys = {f"{method.upper()} {path}"}
    if operation_id:
        keys.add(operation_id)
    return keys


def _is_included(
    options: TargetOptions,
    method: str,
    path: str,
    operation_id: str | None,
    tags: list[str],
) -> bool:
    """True when the target's filters keep this operation. No filters keeps everything."""
    if not options.include_tags and not options.include_operations:
        return True
    if set(tags) & set(options.include_tags):
        return True
    return bool(_operation_keys(method, path, operation_id) & set(options.include_operations))


def _route_filter(
    options: TargetOptions,
) -> Callable[[HTTPRoute, MCPType], MCPType | None]:
    """Build FastMCP's route_map_fn: every kept route is a tool, the rest are excluded."""

    def _map(
        route: HTTPRoute,
        default: MCPType,
    ) -> MCPType | None:
        if _is_included(options, route.method, route.path, route.operation_id, route.tags):
            return MCPType.TOOL
        return MCPType.EXCLUDE

    return _map


def _annotations_for(
    method: str,
) -> ToolAnnotations:
    """MCP tool annotations derived from the HTTP method."""
    upper = method.upper()
    if upper in READ_ONLY_METHODS:
        return ToolAnnotations(read_only_hint=True)
    if upper in DESTRUCTIVE_METHODS:
        return ToolAnnotations(read_only_hint=False, destructive_hint=True)
    return ToolAnnotations(read_only_hint=False, destructive_hint=False)


def _closed_input_schema(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return the input schema with additionalProperties set to false (prototype finding 7)."""
    closed = dict(parameters)
    closed.setdefault("type", "object")
    closed["additionalProperties"] = False
    return closed


def _safe_tool_name(
    name: str,
) -> str:
    """Keep a renamed tool to the characters MCP clients accept."""
    return UNSAFE_TOOL_NAME_CHARS.sub("_", name).strip("_")[:MAX_BASE_TOOL_NAME_CHARS] or "tool"


def _component_adjuster(
    options: TargetOptions,
) -> Callable[[HTTPRoute, Any], None]:
    """Build FastMCP's mcp_component_fn: overrides, annotations and a closed input schema."""

    def _adjust(
        route: HTTPRoute,
        component: Any,
    ) -> None:
        if not isinstance(component, OpenAPITool):
            return
        # FastMCP's mcp_names only matches operationId; this also renames an
        # operation without one, keyed "METHOD /path" (prototype finding 8).
        rename = options.tool_name_overrides.get(f"{route.method.upper()} {route.path}")
        if rename:
            component.name = _safe_tool_name(rename)
        override = options.description_overrides.get(component.name)
        if override:
            component.description = override[:MAX_DESCRIPTION_OVERRIDE_CHARS]
        component.annotations = _annotations_for(route.method)
        component.parameters = _closed_input_schema(component.parameters)

    return _adjust


def _build_sub_server(
    spec: dict[str, Any],
    options: TargetOptions,
    client: httpx2.AsyncClient,
) -> FastMCP:
    """Synchronous FastMCP conversion. Runs in a worker thread."""
    return FastMCP.from_openapi(
        openapi_spec=spec,
        client=client,
        name=options.name,
        route_map_fn=_route_filter(options),
        mcp_component_fn=_component_adjuster(options),
        mcp_names=options.tool_name_overrides,
        validate_output=options.keep_output_schemas,
        strict_input_validation=True,
    )


async def convert_target(
    spec: dict[str, Any],
    options: TargetOptions,
    client: httpx2.AsyncClient,
    timeout_seconds: float,
) -> tuple[FastMCP, list[str]]:
    """Convert a spec into a FastMCP server and list its tool names, without the prefix.

    Raises:
        ConversionError: When FastMCP rejects the spec or takes longer than
            ``timeout_seconds``.
    """
    try:
        sub = await asyncio.wait_for(
            asyncio.to_thread(_build_sub_server, spec, options, client),
            timeout=timeout_seconds,
        )
        tools = await sub.list_tools(run_middleware=False)
    except TimeoutError as e:
        raise ConversionError(f"conversion took longer than {timeout_seconds:g} seconds") from e
    except Exception as e:
        raise ConversionError(f"{type(e).__name__}: {str(e)[:500]}") from e
    return sub, sorted(tool.name for tool in tools)


def _iter_operations(
    spec: dict[str, Any],
) -> list[tuple[str, str, str | None, list[str]]]:
    """List (method, path, operationId, tags) for every operation in the spec."""
    operations = []
    paths = spec.get("paths") or {}
    if not isinstance(paths, dict):
        return operations
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in HTTP_METHODS:
            operation = item.get(method)
            if not isinstance(operation, dict):
                continue
            tags = [t for t in operation.get("tags") or [] if isinstance(t, str)]
            operation_id = operation.get("operationId")
            operations.append(
                (method, str(path), operation_id if isinstance(operation_id, str) else None, tags)
            )
    return operations


def tag_counts(
    spec: dict[str, Any],
) -> dict[str, int]:
    """Operation count per tag, so an admin can choose include_tags."""
    counts: dict[str, int] = {}
    for _method, _path, _operation_id, tags in _iter_operations(spec):
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _unmatched_warnings(
    spec: dict[str, Any],
    options: TargetOptions,
    tool_names: list[str],
) -> list[str]:
    """Warnings for filter and override entries that match nothing."""
    operations = _iter_operations(spec)
    all_tags = {tag for *_rest, tags in operations for tag in tags}
    all_keys: set[str] = set()
    for method, path, operation_id, _tags in operations:
        all_keys |= _operation_keys(method, path, operation_id)

    warnings = [
        f"include_tags entry '{t}' matches no tag in the spec"
        for t in options.include_tags
        if t not in all_tags
    ]
    warnings += [
        f"include_operations entry '{o}' matches no operation"
        for o in options.include_operations
        if o not in all_keys
    ]
    warnings += [
        f"tool_name_overrides key '{k}' matches no operationId or 'METHOD /path'"
        for k in options.tool_name_overrides
        if k not in all_keys
    ]
    warnings += [
        f"description_overrides key '{k}' matches no tool"
        for k in options.description_overrides
        if k not in tool_names
    ]
    return warnings


async def preview(
    request: PreviewRequest,
    timeout_seconds: float,
) -> PreviewResponse:
    """Convert a spec with no network I/O and report the tools it would produce."""
    options = request.model_copy(update={"name": request.target_name})
    async with httpx2.AsyncClient(base_url=PREVIEW_BASE_URL) as client:
        _sub, names = await convert_target(request.spec, options, client, timeout_seconds)
    return PreviewResponse(
        tools=[f"{request.target_name}_{name}" for name in names],
        tags=tag_counts(request.spec),
        warnings=_unmatched_warnings(request.spec, options, names),
    )
