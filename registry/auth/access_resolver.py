"""Per-request scope resolution: servers + tools in a single scope-repo pass.

Used by `registry.auth.dependencies` to populate `user_context` with both
`accessible_servers` and `accessible_tools` without walking the scope
repository twice. The resolver is intentionally fail-closed: on any scope
repo error, callers observe an empty `UserAccess()` and every downstream
filter treats that as "no access".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


ToolAllowlist = dict[str, set[str]]


_WILDCARD_VALUES: frozenset[str] = frozenset(("all", "*"))


@dataclass(frozen=True)
class UserAccess:
    """Resolved per-request access bag.

    Returned by `resolve_scope_access(user_scopes)`. Grows additively as
    future resource types (agents, skills) gain scope-driven filtering.

    Attributes:
        servers: List of server names the user can access. Contains the
            sentinel "*" when any scope grants cross-server wildcard via
            `server: "*"` / `server: "all"`.
        tools: Mapping of server_name to the set of allowed tool names.
            `{"*"}` means wildcard on that server (no pruning). A missing
            entry or empty set means "no tools allowed" (fail-closed).
    """

    servers: list[str] = field(default_factory=list)
    tools: ToolAllowlist = field(default_factory=dict)


async def resolve_scope_access(
    user_scopes: list[str],
) -> UserAccess:
    """Resolve `accessible_servers` and `accessible_tools` in one loop.

    Wildcard semantics (mirrors auth_server/server.py):
        - `server: "*"` / `server: "all"` at the rule level is a
          cross-server wildcard. The sentinel "*" is registered in both
          `servers` and `tools` so downstream callers can short-circuit
          without knowing the concrete server name.
        - `tools: ["all"]` / `tools: ["*"]` is a per-server wildcard and
          is recorded as `{"*"}`.
        - A missing `tools` key or a non-list value means "no tools
          allowed" on that server (fail-closed). A startup audit in
          `audit_legacy_scopes_on_startup` warns operators about the
          legacy shape so they can migrate before the tightening bites.

    Wildcards are sticky across scopes: once any scope grants wildcard on
    a server, later scopes cannot narrow it. Non-wildcard tool sets are
    unioned across scopes.

    Args:
        user_scopes: Flat list of user scope names.

    Returns:
        UserAccess. On scope-repo error, returns an empty UserAccess().
    """
    try:
        from ..repositories.factory import get_scope_repository

        scope_repo = get_scope_repository()
    except Exception as exc:
        logger.error(
            "resolve_scope_access: scope repo unavailable: %s", exc, exc_info=True
        )
        return UserAccess()

    servers: set[str] = set()
    tools: ToolAllowlist = {}

    for scope in user_scopes:
        try:
            scope_config = await scope_repo.get_server_scopes(scope)
        except Exception as exc:
            logger.error(
                "resolve_scope_access: get_server_scopes(%s) failed: %s", scope, exc
            )
            continue

        if not scope_config:
            continue

        for rule in scope_config:
            if not isinstance(rule, dict):
                continue
            server_name = rule.get("server")
            if not server_name:
                continue

            # Cross-server wildcard: server: "*" or server: "all". Promote
            # to sentinel "*" in both servers and tools so downstream
            # short-circuits work regardless of the requested server.
            if str(server_name) in _WILDCARD_VALUES:
                servers.add("*")
                tools["*"] = {"*"}
                continue

            servers.add(server_name)

            tool_rules = rule.get("tools")
            if tool_rules is None:
                # Fail-closed for legacy scopes missing the tools key.
                if server_name not in tools:
                    tools[server_name] = set()
                continue

            # Some UIs store a wildcard as a bare string ("*" or "all")
            # instead of a single-item list. Treat both forms as wildcard
            # so the resolver agrees with what auth_server's
            # validate_server_tool_access already accepts.
            if isinstance(tool_rules, str):
                if tool_rules in _WILDCARD_VALUES:
                    tools[server_name] = {"*"}
                else:
                    # Single tool name written as a bare string.
                    existing_str = tools.get(server_name)
                    if existing_str == {"*"}:
                        continue
                    merged_str = existing_str if existing_str is not None else set()
                    merged_str.add(tool_rules)
                    tools[server_name] = merged_str
                continue

            if not isinstance(tool_rules, list):
                # Malformed rule (e.g. dict / int): treat as no allowlist.
                if server_name not in tools:
                    tools[server_name] = set()
                continue

            existing = tools.get(server_name)
            if existing == {"*"}:
                # Wildcard already sticky from a prior scope; keep it.
                continue

            has_wildcard = any(str(t) in _WILDCARD_VALUES for t in tool_rules)
            if has_wildcard:
                tools[server_name] = {"*"}
                continue

            merged: set[str] = existing if existing is not None else set()
            merged.update(str(t) for t in tool_rules)
            tools[server_name] = merged

    tool_restricted = sum(1 for v in tools.values() if v != {"*"})
    logger.info(
        "resolve_scope_access: scopes=%d servers=%d tool_restricted_servers=%d",
        len(user_scopes),
        len(servers),
        tool_restricted,
    )

    return UserAccess(servers=sorted(servers), tools=tools)


async def get_user_accessible_servers(
    user_scopes: list[str],
) -> list[str]:
    """Return the list of server names accessible to the given scopes.

    Thin wrapper around `resolve_scope_access` that preserves the
    historical signature used across the codebase (dependencies.py,
    user_can_access_server, tests, CLIs).
    """
    access = await resolve_scope_access(user_scopes)
    return access.servers


async def get_user_accessible_tools(
    user_scopes: list[str],
) -> dict[str, set[str]]:
    """Return the tool allowlist map for the given scopes.

    Parallel wrapper for symmetry with get_user_accessible_servers and
    for test ergonomics.
    """
    access = await resolve_scope_access(user_scopes)
    return access.tools


async def audit_legacy_scopes_on_startup() -> int:
    """One-time scan at boot for legacy scope documents.

    Emits a WARN per scope rule where:
        - `tools` key is absent (legacy shape), OR
        - `tools` is an empty list AND the scope declares a method
          set that includes tools/call (call will always fail, list
          will always be empty).

    Returns:
        The number of warnings emitted so tests / metrics can assert.
        On scope-repo error, returns 0 and logs an error.
    """
    try:
        from ..repositories.factory import get_scope_repository

        scope_repo = get_scope_repository()
    except Exception as exc:
        logger.error(
            "audit_legacy_scopes_on_startup: scope repo unavailable: %s", exc
        )
        return 0

    try:
        scope_names = await scope_repo.list_scope_names()
    except Exception as exc:
        logger.error(
            "audit_legacy_scopes_on_startup: list_scope_names failed: %s", exc
        )
        return 0

    warnings_emitted = 0
    for scope_name in scope_names:
        try:
            rules = await scope_repo.get_server_scopes(scope_name)
        except Exception as exc:
            logger.error(
                "audit_legacy_scopes_on_startup: get_server_scopes(%s) failed: %s",
                scope_name,
                exc,
            )
            continue

        for rule in rules or []:
            if not isinstance(rule, dict) or "server" not in rule:
                continue
            server_name = rule.get("server")
            tools = rule.get("tools")
            methods = rule.get("methods") or []

            if tools is None:
                logger.warning(
                    "legacy_scope_missing_tools scope=%s server=%s methods=%s "
                    "(post-upgrade this will deny all tools/list and tools/call; "
                    "migrate to tools: ['all'] if wildcard was intended)",
                    scope_name,
                    server_name,
                    methods,
                )
                warnings_emitted += 1
            elif (
                isinstance(tools, list)
                and not tools
                and (
                    "tools/call" in methods
                    or "all" in methods
                    or "*" in methods
                )
            ):
                logger.warning(
                    "empty_tools_list_with_call_method scope=%s server=%s",
                    scope_name,
                    server_name,
                )
                warnings_emitted += 1

    if warnings_emitted:
        logger.warning(
            "Legacy scope audit: %d warnings. See log lines above for "
            "specific scope/server pairs. Update the mcp-scopes collection "
            "before upgrading.",
            warnings_emitted,
        )
    else:
        logger.info("Legacy scope audit: no issues found.")
    return warnings_emitted
