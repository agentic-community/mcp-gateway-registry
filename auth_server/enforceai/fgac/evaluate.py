from __future__ import annotations

from typing import Optional

from .models import (
    Decision,
    ScopeCatalog,
    ToolPolicy,
)
from ..identity import (
    IdentityContext,
)

WILDCARD_VALUES: frozenset[str] = frozenset({"*", "all"})


def _server_matches(
    *,
    policy_server: str,
    request_server: str,
) -> bool:
    if policy_server.strip().lower() in WILDCARD_VALUES:
        return True
    return policy_server == request_server


def _normalize_allowed_tools(
    allowed_tools: Optional[list[str]],
) -> Optional[frozenset[str]]:
    if allowed_tools is None:
        return None

    normalized: list[str] = []
    for tool in allowed_tools:
        stripped = tool.strip()
        if not stripped:
            continue
        normalized.append(stripped)

    if not normalized:
        return frozenset()

    return frozenset(normalized)


def _evaluate_tool(
    *,
    identity: IdentityContext,
    catalog: ScopeCatalog,
    server: str,
    method: str,
    tool: str,
    allowed_tools: Optional[list[str]] = None,
) -> Decision:
    normalized_server = server.strip()
    normalized_method = method.strip()
    normalized_tool = tool.strip()

    if not normalized_server or not normalized_method or not normalized_tool:
        return Decision(
            allowed=False,
            reason="no_matching_policy",
        )

    allowed_tools_set = _normalize_allowed_tools(allowed_tools)
    if allowed_tools_set is not None and normalized_tool not in allowed_tools_set:
        return Decision(
            allowed=False,
            reason="tool_not_in_allowed_tools",
        )

    saw_known_scope: bool = False
    for scope_name in identity.scopes:
        scope = catalog.get_scope(scope_name)
        if scope is None:
            continue

        saw_known_scope = True
        for server_permission in scope.server_permissions:
            if not _server_matches(
                policy_server=server_permission.server,
                request_server=normalized_server,
            ):
                continue

            if not server_permission.methods.allows(method=normalized_method):
                continue

            if server_permission.tools is None:
                continue

            if not server_permission.tools.allows(tool=normalized_tool):
                continue

            return Decision(
                allowed=True,
                reason="allowed",
                matched_scope=scope_name,
            )

    if not saw_known_scope and identity.scopes:
        return Decision(
            allowed=False,
            reason="unknown_scope",
        )

    return Decision(
        allowed=False,
        reason="no_matching_policy",
    )


def evaluate_tool_call(
    *,
    identity: IdentityContext,
    catalog: ScopeCatalog,
    server: str,
    tool: str,
    allowed_tools: Optional[list[str]] = None,
) -> Decision:
    """Evaluate whether the identity may call a tool on a specific server."""

    return _evaluate_tool(
        identity=identity,
        catalog=catalog,
        server=server,
        method="tools/call",
        tool=tool,
        allowed_tools=allowed_tools,
    )


def evaluate_tool_visibility(
    *,
    identity: IdentityContext,
    catalog: ScopeCatalog,
    server: str,
    tool: str,
    allowed_tools: Optional[list[str]] = None,
) -> Decision:
    """Evaluate whether a tool should be visible in tools/list.

    Stage 5 rule: only list tools that are callable (tools/list must be a
    subset of tools/call).
    """

    return evaluate_tool_call(
        identity=identity,
        catalog=catalog,
        server=server,
        tool=tool,
        allowed_tools=allowed_tools,
    )


def resolve_callable_tools_for_server(
    *,
    identity: IdentityContext,
    catalog: ScopeCatalog,
    server: str,
    allowed_tools: Optional[list[str]] = None,
) -> ToolPolicy:
    """Return the set of tools callable on `server` under `tools/call`.

    This is used to filter `tools/list` responses so that listed tools are a
    subset of callable tools.
    """

    normalized_server = server.strip()
    if not normalized_server:
        return ToolPolicy(
            all_tools=False,
            tools=frozenset(),
        )

    allowed_tools_set = _normalize_allowed_tools(allowed_tools)

    allow_all: bool = False
    allowed: set[str] = set()

    for scope_name in identity.scopes:
        scope = catalog.get_scope(scope_name)
        if scope is None:
            continue

        for server_permission in scope.server_permissions:
            if not _server_matches(
                policy_server=server_permission.server,
                request_server=normalized_server,
            ):
                continue

            if not server_permission.methods.allows(method="tools/call"):
                continue

            if server_permission.tools is None:
                continue

            if server_permission.tools.all_tools:
                allow_all = True
                break

            allowed.update(server_permission.tools.tools)

        if allow_all:
            break

    if allow_all:
        if allowed_tools_set is None:
            return ToolPolicy(
                all_tools=True,
                tools=frozenset(),
            )

        return ToolPolicy(
            all_tools=False,
            tools=allowed_tools_set,
        )

    if allowed_tools_set is not None:
        allowed = allowed.intersection(allowed_tools_set)

    return ToolPolicy(
        all_tools=False,
        tools=frozenset(allowed),
    )
