"""Build the tool access listing: who can see and who can call each tool (issue #2102).

The registry stores access as scope documents, one per group: each maps
identity provider groups to server_access rules and ui_permissions. This
service runs those rules in reverse. It loads every group once, builds the
user context a member of only that group would get, and asks the same checks
the registry and the gateway use whether that group passes each condition for
each tool.

A condition is one check a caller must pass, such as "can open a session on
the server" or "can call this tool". A user's access merges all their groups,
so a caller passes a condition when any one of their groups does. The summary
for a tool lists the groups that pass every condition by themselves
("alone"), and, when that misses callers, the groups that pass each condition
("together": one group from each).

The listing is informational. It writes nothing and changes no access check.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..auth.access_resolver import access_from_rules
from ..auth.asset_permissions import can_view_all
from ..auth.dependencies import (
    _user_is_admin,
    get_accessible_services_for_user,
    ui_permissions_from_configs,
)
from ..auth.tool_filter import tool_visible_without_audit
from ..common.tool_grants import (
    SERVER_WILDCARD,
    normalize_server_name,
    rule_grants_mcp_method,
    rule_grants_tool_call,
)
from ..repositories.factory import get_scope_repository
from ..schemas.tool_access_models import (
    AccessCondition,
    AccessSummary,
    GroupRef,
    ToolAccessRow,
)
from ..schemas.virtual_server_models import VirtualServerConfig
from .server_service import server_service
from .virtual_server_service import get_virtual_server_service
from .visibility import (
    user_can_access_server_from_doc,
    user_can_list_virtual_server,
)

logger = logging.getLogger(__name__)

INITIALIZE_METHOD: str = "initialize"
SERVER_TYPE_MCP: str = "mcp"
SERVER_TYPE_OPENAPI: str = "openapi"
SERVER_TYPE_VIRTUAL: str = "virtual"


@dataclass
class GroupInfo:
    """One registry group, its rules, and the context a sole member would get."""

    name: str
    idp_groups: list[str]
    rules: list[dict[str, Any]]
    context: dict[str, Any]
    is_admin: bool
    # The group's rules keyed by slash-stripped server name, built once so a
    # per-server lookup does not rescan every rule (admin groups hold thousands).
    rules_by_server: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def ref(self) -> GroupRef:
        """Return the response form of this group."""
        return GroupRef(group=self.name, idp_groups=self.idp_groups, is_admin=self.is_admin)


@dataclass
class BuiltRow:
    """A listing row plus who owns its server, which the response never shows."""

    row: ToolAccessRow
    owner: str | None = None


@dataclass
class ListingData:
    """Every row, plus each group's identity provider names and admin flag."""

    rows: list[BuiltRow]
    groups: dict[str, GroupRef]


# Each search keystroke in the UI requests a page, so the built listing is
# reused for a short time instead of rebuilt per request.
CACHE_SECONDS: float = 30.0
_cache: tuple[float, ListingData] | None = None
_cache_lock = asyncio.Lock()


# A condition: a label for display and a check run on one group.
Condition = tuple[str, Callable[[GroupInfo], bool]]


@dataclass
class _ServerRules:
    """Each group's rules that name one server, so per-tool checks scan only those."""

    server_key: str
    rules_by_group: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _group_context(
    name: str,
    scope_rules: dict[str, list[dict]],
    ui_configs: dict[str, dict],
) -> dict[str, Any]:
    """Build the user context a member of only this group would get.

    Uses the same helpers as _derive_user_context, so is_admin, view_all and the
    accessible_* values match what a real member receives.
    """
    access = access_from_rules([name], scope_rules)
    ui_permissions = ui_permissions_from_configs([name], ui_configs)
    return {
        "username": "",
        "groups": [],
        "scopes": [name],
        "accessible_servers": access.servers,
        "accessible_tools": access.tools,
        "accessible_services": get_accessible_services_for_user(ui_permissions),
        "ui_permissions": ui_permissions,
        "is_admin": _user_is_admin(ui_permissions),
    }


async def _load_groups() -> list[GroupInfo]:
    """Load every group with four bulk reads, whatever the number of groups."""
    scope_repo = get_scope_repository()
    names = sorted(await scope_repo.list_scope_names())
    if not names:
        logger.warning("Tool access listing found no groups in the scopes collection")
        return []
    scope_rules = await scope_repo.get_server_scopes_bulk(names)
    ui_configs = await scope_repo.get_ui_scopes_bulk(names)
    mappings = await scope_repo.get_all_group_mappings()

    groups = []
    for name in names:
        context = _group_context(name, scope_rules, ui_configs)
        rules = [r for r in scope_rules.get(name) or [] if isinstance(r, dict)]
        groups.append(
            GroupInfo(
                name=name,
                idp_groups=sorted(mappings.get(name) or []),
                rules=rules,
                context=context,
                is_admin=bool(context["is_admin"]),
                rules_by_server=_index_rules(rules),
            )
        )
    return groups


def _index_rules(
    rules: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Key rules by slash-stripped server name, the way server_names_match compares."""
    index: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        server = normalize_server_name(rule.get("server"))
        if isinstance(server, str) and server:
            index.setdefault(server, []).append(rule)
    return index


def _rules_for_server(
    groups: list[GroupInfo],
    server_path: str,
) -> _ServerRules:
    """Collect each group's rules that name this server or the '*' wildcard.

    Same result as filtering every rule with server_names_match, read from the
    per-group index instead.
    """
    server_key = server_path.strip("/")
    found = _ServerRules(server_key=server_key)
    for group in groups:
        named = group.rules_by_server.get(server_key, [])
        wildcard = (
            group.rules_by_server.get(SERVER_WILDCARD, []) if server_key != SERVER_WILDCARD else []
        )
        found.rules_by_group[group.name] = named + wildcard
    return found


def _connect_check(
    server_rules: _ServerRules,
) -> Callable[[GroupInfo], bool]:
    """Check that a group's own rules allow initialize on the server."""

    def check(group: GroupInfo) -> bool:
        rules = server_rules.rules_by_group.get(group.name, [])
        return any(
            rule_grants_mcp_method(rule, server_rules.server_key, INITIALIZE_METHOD)
            for rule in rules
        )

    return check


def _call_check(
    server_rules: _ServerRules,
    tool_name: str,
) -> Callable[[GroupInfo], bool]:
    """Check that a group's own rules allow tools/call for this tool."""

    def check(group: GroupInfo) -> bool:
        rules = server_rules.rules_by_group.get(group.name, [])
        return any(
            rule_grants_tool_call(rule, server_rules.server_key, tool_name) for rule in rules
        )

    return check


def _summarize(
    groups: list[GroupInfo],
    conditions: list[Condition],
) -> AccessSummary:
    """Turn per-group checks into the 'alone' and 'together' lists.

    'alone' holds groups that pass every condition. 'together' is filled only
    when every condition has at least one passing group outside 'alone': only
    then can a combination of groups reach access that no single group gives.
    """
    passed = {group.name: [check(group) for _, check in conditions] for group in groups}
    alone = [group for group in groups if all(passed[group.name])]
    alone_names = {group.name for group in alone}

    together: list[AccessCondition] = []
    for index, (label, _) in enumerate(conditions):
        extra = [
            group.name
            for group in groups
            if passed[group.name][index] and group.name not in alone_names
        ]
        together.append(AccessCondition(label=label, groups=extra))
    if len(conditions) < 2 or not all(condition.groups for condition in together):
        together = []
    return AccessSummary(alone=[group.name for group in alone], together=together)


def _server_listed_check(
    server_path: str,
) -> Callable[[GroupInfo], bool]:
    """Mirror the row filter in GET /api/servers (list_service grants)."""
    technical_name = server_path.strip("/")

    def check(group: GroupInfo) -> bool:
        ctx = group.context
        if ctx["is_admin"] or can_view_all(ctx):
            return True
        services = [s.strip("/") for s in ctx.get("accessible_services") or []]
        return "all" in services or technical_name in services

    return check


def _server_fetched_check(
    server_path: str,
) -> Callable[[GroupInfo], bool]:
    """Mirror get_all_servers_with_permissions: a rule must name the server's path."""
    technical_name = server_path.strip("/")

    def check(group: GroupInfo) -> bool:
        ctx = group.context
        if ctx["is_admin"] or can_view_all(ctx):
            return True
        servers = ctx.get("accessible_servers") or []
        return "*" in servers or any(s.strip("/") == technical_name for s in servers)

    return check


def _tool_visible_check(
    server_info: dict[str, Any],
    tool_name: str,
) -> Callable[[GroupInfo], bool]:
    """Mirror filter_tools_for_user for one tool."""
    server_name = server_info.get("server_name") or server_info["path"]

    def check(group: GroupInfo) -> bool:
        return tool_visible_without_audit(
            server_name, tool_name, group.context, server_path=server_info["path"]
        )

    return check


def _server_type(
    server_info: dict[str, Any],
) -> str:
    """Return 'openapi' for a server the OpenAPI MCP service owns, else 'mcp'."""
    metadata = server_info.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("openapi_server"):
        return SERVER_TYPE_OPENAPI
    return SERVER_TYPE_MCP


def _block_state(
    server_info: dict[str, Any] | None,
    tool_name: str,
) -> tuple[bool, str | None]:
    """Return (blocked, source) for a tool from the server's tool_overrides."""
    overrides = (server_info or {}).get("tool_overrides") or {}
    override = overrides.get(tool_name) if isinstance(overrides, dict) else None
    if isinstance(override, dict) and override.get("blocked"):
        return True, override.get("source")
    return False, None


def _server_owner(
    server_info: dict[str, Any],
) -> str | None:
    """Return the local owner of a server, or None for a server synced from a peer.

    A synced server's registered_by names a user of the peer registry, who may
    share a username with a different local user.
    """
    if server_info.get("sync_metadata"):
        return None
    return server_info.get("registered_by") or None


def _tool_names(
    server_info: dict[str, Any],
) -> list[str]:
    """Return the names in a server's stored tool list, skipping malformed entries."""
    names = []
    for tool in server_info.get("tool_list") or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"]:
            names.append(tool["name"])
    return sorted(set(names))


def _server_rows(
    server_info: dict[str, Any],
    groups: list[GroupInfo],
) -> list[BuiltRow]:
    """Build one row per tool on an MCP or OpenAPI MCP server."""
    path = server_info["path"]
    server_rules = _rules_for_server(groups, path)
    connect = _connect_check(server_rules)
    listed = _server_listed_check(path)
    fetched = _server_fetched_check(path)
    owner = _server_owner(server_info)

    rows = []
    for tool_name in _tool_names(server_info):
        blocked, source = _block_state(server_info, tool_name)
        see_conditions: list[Condition] = [
            ("Listed in the registry (list_service)", listed),
            (f"Has a server_access rule for {path}", fetched),
            (f"Allowed to see {tool_name}", _tool_visible_check(server_info, tool_name)),
        ]
        call_conditions: list[Condition] = [
            (f"Open a session on {path} (initialize)", connect),
            (f"Call {tool_name} on {path}", _call_check(server_rules, tool_name)),
        ]
        row = ToolAccessRow(
            tool_name=tool_name,
            server_name=server_info.get("server_name") or path,
            server_path=path,
            server_type=_server_type(server_info),
            enabled=bool(server_info.get("is_enabled")),
            blocked=blocked,
            blocked_source=source,
            can_see=_summarize(groups, see_conditions),
            can_call=_summarize(groups, call_conditions),
        )
        rows.append(BuiltRow(row=row, owner=owner))
    return rows


def _required_group_checks(
    names: list[str],
    admin_passes: bool,
) -> list[Condition]:
    """One condition per required scope name: the caller must be in that group.

    The gateway's Lua router requires every listed scope and lets nobody
    bypass it. The registry's own view of a virtual server lets admins see a
    scope-guarded tool, so admin_passes is True only for the "can see" side.
    """
    checks: list[Condition] = []
    for name in sorted(set(names)):

        def check(group: GroupInfo, required: str = name) -> bool:
            return group.name == required or (admin_passes and group.is_admin)

        checks.append((f"Member of {name}", check))
    return checks


def _virtual_see_conditions(
    config: VirtualServerConfig,
    backend: dict[str, Any] | None,
    mapping_tool: str,
    tool_scopes: list[str],
) -> list[Condition]:
    """Mirror the registry's virtual server read and its tool list filter."""
    path = config.path

    def listed(group: GroupInfo) -> bool:
        return user_can_list_virtual_server(group.context, path, honor_view_all=True)

    if backend is None:
        return [("Virtual server listed", listed), ("Backend server exists", lambda _g: False)]
    backend_path = backend["path"]
    backend_name = backend.get("server_name") or backend_path

    def backend_access(group: GroupInfo) -> bool:
        return user_can_access_server_from_doc(
            backend_path, backend_name, group.context, honor_view_all=True
        )

    conditions: list[Condition] = [
        (f"Virtual server {path} listed (list_virtual_server)", listed),
        (f"Has access to backend {backend_path}", backend_access),
        (
            f"Allowed to see {mapping_tool} on {backend_path}",
            _tool_visible_check(backend, mapping_tool),
        ),
    ]
    return conditions + _required_group_checks(tool_scopes, admin_passes=True)


def _virtual_call_conditions(
    config: VirtualServerConfig,
    groups: list[GroupInfo],
    alias: str,
    backend_path: str,
    backend_tool: str,
    tool_scopes: list[str],
) -> list[Condition]:
    """Mirror every gateway check on a tools/call through a virtual server.

    The virtual server's location checks initialize and the alias on its own
    path, the Lua router checks the required scopes, and the router then
    authorizes initialize and the original tool name on the backend's path.
    """
    virtual_rules = _rules_for_server(groups, config.path)
    backend_rules = _rules_for_server(groups, backend_path)
    conditions: list[Condition] = [
        (f"Open a session on {config.path} (initialize)", _connect_check(virtual_rules)),
        (f"Call {alias} on {config.path}", _call_check(virtual_rules, alias)),
        (f"Open a session on backend {backend_path} (initialize)", _connect_check(backend_rules)),
        (
            f"Call {backend_tool} on backend {backend_path}",
            _call_check(backend_rules, backend_tool),
        ),
    ]
    required = list(config.required_scopes) + list(tool_scopes)
    return conditions + _required_group_checks(required, admin_passes=False)


def _virtual_rows(
    config: VirtualServerConfig,
    servers: dict[str, dict[str, Any]],
    groups: list[GroupInfo],
) -> list[BuiltRow]:
    """Build one row per tool mapping on a virtual server."""
    overrides = {o.tool_alias: list(o.required_scopes) for o in config.tool_scope_overrides}
    rows = []
    for mapping in config.tool_mappings:
        alias = mapping.alias or mapping.tool_name
        backend = _find_server(servers, mapping.backend_server_path)
        backend_path = backend["path"] if backend else mapping.backend_server_path
        if backend is None:
            logger.warning(
                "Virtual server %s maps %s to missing backend %s",
                config.path,
                alias,
                mapping.backend_server_path,
            )
        tool_scopes = overrides.get(alias, [])
        blocked, source = _block_state(backend, mapping.tool_name)
        see = _virtual_see_conditions(config, backend, mapping.tool_name, tool_scopes)
        call = _virtual_call_conditions(
            config, groups, alias, backend_path, mapping.tool_name, tool_scopes
        )
        row = ToolAccessRow(
            tool_name=alias,
            server_name=config.server_name,
            server_path=config.path,
            server_type=SERVER_TYPE_VIRTUAL,
            enabled=bool(config.is_enabled),
            blocked=blocked,
            blocked_source=source,
            backend_server_path=backend_path,
            backend_tool_name=mapping.tool_name,
            can_see=_summarize(groups, see),
            can_call=_summarize(groups, call),
        )
        rows.append(BuiltRow(row=row, owner=config.created_by or None))
    return rows


def _find_server(
    servers: dict[str, dict[str, Any]],
    path: str,
) -> dict[str, Any] | None:
    """Find a server by path, tolerating trailing-slash differences."""
    wanted = path.strip("/")
    for server_path, server_info in servers.items():
        if server_path.strip("/") == wanted:
            return server_info
    return None


async def _load_servers() -> dict[str, dict[str, Any]]:
    """Load every active server with its tool list, keyed by path."""
    servers = await server_service.get_all_servers()
    for path, server_info in servers.items():
        server_info.setdefault("path", path)
    return servers


async def _build_listing() -> ListingData:
    """Build every row, sorted by server name, path, then tool."""
    started = time.monotonic()
    groups = await _load_groups()
    servers = await _load_servers()
    virtual_configs = await get_virtual_server_service().list_virtual_server_configs()

    rows: list[BuiltRow] = []
    for server_info in servers.values():
        rows.extend(_server_rows(server_info, groups))
    for config in virtual_configs:
        rows.extend(_virtual_rows(config, servers, groups))

    rows.sort(key=lambda b: (b.row.server_name.lower(), b.row.server_path, b.row.tool_name))
    logger.info(
        "Tool access listing built: groups=%d servers=%d virtual_servers=%d rows=%d in %.2fs",
        len(groups),
        len(servers),
        len(virtual_configs),
        len(rows),
        time.monotonic() - started,
    )
    return ListingData(rows=rows, groups={group.name: group.ref() for group in groups})


async def build_listing(
    owner: str | None = None,
) -> ListingData:
    """Return the listing, reusing a build up to CACHE_SECONDS old.

    Args:
        owner: When set, keep only rows for servers and virtual servers this
            username owns.

    Returns:
        ListingData with the rows the filter keeps and every group's details.
    """
    global _cache
    async with _cache_lock:
        if _cache is None or time.monotonic() - _cache[0] > CACHE_SECONDS:
            _cache = (time.monotonic(), await _build_listing())
        data = _cache[1]
    if owner is None:
        return data
    return ListingData(rows=[b for b in data.rows if b.owner == owner], groups=data.groups)


def clear_cache() -> None:
    """Drop the cached listing (tests, and anyone who needs a fresh build)."""
    global _cache
    _cache = None
