from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from ..errors import (
    DependencyUnavailableError,
)
from .models import (
    AgentActionPermission,
    MethodPolicy,
    ScopeCatalog,
    ScopeDefinition,
    ServerPermission,
    ToolPolicy,
    UIActionPermission,
)

RESERVED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"UI-Scopes", "group_mappings"})
WILDCARD_VALUES: frozenset[str] = frozenset({"*", "all"})


def _get_mtime_ns(
    path: Path,
) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError as exc:
        raise DependencyUnavailableError(
            f"Failed to stat scope catalog at {path}",
            public_message="Scope catalog unavailable",
        ) from exc


def _read_text(
    path: Path,
) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DependencyUnavailableError(
            f"Failed to read scope catalog at {path}",
            public_message="Scope catalog unavailable",
        ) from exc


def _load_yaml_object(
    *,
    path: Path,
) -> Any:
    raw = _read_text(path)
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in scope catalog: {exc}") from exc


def _ensure_mapping(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping/object")
    return value


def _ensure_list(
    value: Any,
    *,
    label: str,
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _normalize_str(
    value: Any,
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be a non-empty string")
    return stripped


def _normalize_str_list(
    value: Any,
    *,
    label: str,
    allow_none: bool = False,
) -> list[str]:
    if value is None and allow_none:
        return []

    if isinstance(value, str):
        return [_normalize_str(value, label=label)]

    raw_list = _ensure_list(value, label=label)
    normalized: list[str] = []
    for idx, item in enumerate(raw_list):
        normalized.append(
            _normalize_str(
                item,
                label=f"{label}[{idx}]",
            )
        )
    return normalized


def _parse_method_policy(
    value: Any,
    *,
    label: str,
) -> MethodPolicy:
    items = _normalize_str_list(value, label=label)
    lowered = {item.lower() for item in items}
    if lowered & WILDCARD_VALUES:
        return MethodPolicy(
            all_methods=True,
            methods=frozenset(),
        )

    return MethodPolicy(
        all_methods=False,
        methods=frozenset(items),
    )


def _parse_tool_policy(
    value: Any,
    *,
    label: str,
) -> ToolPolicy:
    if isinstance(value, str) and value.strip() in WILDCARD_VALUES:
        return ToolPolicy(
            all_tools=True,
            tools=frozenset(),
        )

    items = _normalize_str_list(value, label=label)
    lowered = {item.lower() for item in items}
    if lowered & WILDCARD_VALUES:
        return ToolPolicy(
            all_tools=True,
            tools=frozenset(),
        )

    return ToolPolicy(
        all_tools=False,
        tools=frozenset(items),
    )


def _parse_server_permission(
    entry: dict[str, Any],
    *,
    scope_name: str,
    index: int,
) -> ServerPermission:
    server = _normalize_str(
        entry.get("server"),
        label=f"{scope_name}[{index}].server",
    )
    methods = _parse_method_policy(
        entry.get("methods"),
        label=f"{scope_name}[{index}].methods",
    )
    tools_raw = entry.get("tools")
    tools: Optional[ToolPolicy] = None
    if tools_raw is not None:
        tools = _parse_tool_policy(
            tools_raw,
            label=f"{scope_name}[{index}].tools",
        )

    return ServerPermission(
        server=server,
        methods=methods,
        tools=tools,
    )


def _parse_agent_permissions(
    entry: dict[str, Any],
    *,
    scope_name: str,
    index: int,
) -> list[AgentActionPermission]:
    agents_value = _ensure_mapping(
        entry.get("agents"),
        label=f"{scope_name}[{index}].agents",
    )
    actions_value = _ensure_list(
        agents_value.get("actions"),
        label=f"{scope_name}[{index}].agents.actions",
    )

    parsed: list[AgentActionPermission] = []
    for action_index, raw_action in enumerate(actions_value):
        action_mapping = _ensure_mapping(
            raw_action,
            label=f"{scope_name}[{index}].agents.actions[{action_index}]",
        )
        action_name = _normalize_str(
            action_mapping.get("action"),
            label=f"{scope_name}[{index}].agents.actions[{action_index}].action",
        )
        resources = _normalize_str_list(
            action_mapping.get("resources"),
            label=f"{scope_name}[{index}].agents.actions[{action_index}].resources",
        )
        parsed.append(
            AgentActionPermission(
                action=action_name,
                resources=tuple(resources),
            )
        )

    return parsed


def _parse_scope_definition(
    scope_name: str,
    value: Any,
) -> ScopeDefinition:
    entries = _ensure_list(value, label=f"scope[{scope_name}]")

    server_permissions: list[ServerPermission] = []
    agent_permissions: list[AgentActionPermission] = []

    for index, raw_entry in enumerate(entries):
        entry = _ensure_mapping(
            raw_entry,
            label=f"{scope_name}[{index}]",
        )
        if "server" in entry:
            server_permissions.append(
                _parse_server_permission(
                    entry,
                    scope_name=scope_name,
                    index=index,
                )
            )
            continue

        if "agents" in entry:
            agent_permissions.extend(
                _parse_agent_permissions(
                    entry,
                    scope_name=scope_name,
                    index=index,
                )
            )
            continue

        raise ValueError(f"{scope_name}[{index}] must contain 'server' or 'agents'")

    return ScopeDefinition(
        name=scope_name,
        server_permissions=tuple(server_permissions),
        agent_permissions=tuple(agent_permissions),
    )


def _parse_ui_scopes(
    value: Any,
) -> dict[str, dict[str, UIActionPermission]]:
    ui_scopes_mapping = _ensure_mapping(value, label="UI-Scopes")
    parsed: dict[str, dict[str, UIActionPermission]] = {}
    for group_name, group_value in ui_scopes_mapping.items():
        group = _normalize_str(group_name, label="UI-Scopes key")
        actions_mapping = _ensure_mapping(group_value, label=f"UI-Scopes[{group}]")
        group_permissions: dict[str, UIActionPermission] = {}
        for action, resources_value in actions_mapping.items():
            action_name = _normalize_str(
                action,
                label=f"UI-Scopes[{group}] action",
            )
            resources = _normalize_str_list(
                resources_value,
                label=f"UI-Scopes[{group}].{action_name}",
                allow_none=True,
            )
            group_permissions[action_name] = UIActionPermission(
                action=action_name,
                resources=tuple(resources),
            )
        parsed[group] = group_permissions
    return parsed


def _parse_group_mappings(
    value: Any,
) -> dict[str, tuple[str, ...]]:
    mappings = _ensure_mapping(value, label="group_mappings")
    parsed: dict[str, tuple[str, ...]] = {}
    for group_name, scopes_value in mappings.items():
        group = _normalize_str(group_name, label="group_mappings key")
        scopes = _normalize_str_list(
            scopes_value,
            label=f"group_mappings[{group}]",
        )
        parsed[group] = tuple(scopes)
    return parsed


def _validate_references(
    *,
    ui_scopes: dict[str, dict[str, UIActionPermission]],
    group_mappings: dict[str, tuple[str, ...]],
    scopes: dict[str, ScopeDefinition],
) -> None:
    known: set[str] = set(scopes.keys())
    known |= set(ui_scopes.keys())

    missing: list[str] = []
    for group_name, mapped_scopes in group_mappings.items():
        for scope_name in mapped_scopes:
            if scope_name not in known:
                missing.append(f"{group_name} -> {scope_name}")

    if missing:
        display = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"group_mappings references unknown scopes: {display}{suffix}")


def default_scopes_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scopes.yml"


def _canonicalize_path(
    path: Path,
) -> Path:
    return path.expanduser().resolve()


@lru_cache(maxsize=8)
def _load_scope_catalog_cached(
    path_str: str,
    mtime_ns: int,
) -> ScopeCatalog:
    del mtime_ns
    path = _canonicalize_path(Path(path_str))
    payload = _load_yaml_object(path=path)
    root = _ensure_mapping(payload, label="scope catalog root")

    if "UI-Scopes" not in root:
        raise ValueError("Scope catalog missing required key: UI-Scopes")
    if "group_mappings" not in root:
        raise ValueError("Scope catalog missing required key: group_mappings")

    ui_scopes = _parse_ui_scopes(root.get("UI-Scopes"))
    group_mappings = _parse_group_mappings(root.get("group_mappings"))

    scopes: dict[str, ScopeDefinition] = {}
    for key, value in root.items():
        if key in RESERVED_TOP_LEVEL_KEYS:
            continue
        scope_name = _normalize_str(key, label="scope name")
        scopes[scope_name] = _parse_scope_definition(scope_name, value)

    _validate_references(
        ui_scopes=ui_scopes,
        group_mappings=group_mappings,
        scopes=scopes,
    )

    return ScopeCatalog(
        path=path,
        ui_scopes=ui_scopes,
        group_mappings=group_mappings,
        scopes=scopes,
    )


def clear_scope_catalog_cache() -> None:
    _load_scope_catalog_cached.cache_clear()


def load_scope_catalog(
    *,
    path: Optional[Path] = None,
) -> ScopeCatalog:
    """Load and validate the scope catalog from YAML, with caching.

    Args:
        path: Optional override path. When omitted, uses `auth_server/scopes.yml`.

    Returns:
        Parsed scope catalog with stable, typed accessors.

    Raises:
        DependencyUnavailableError: Catalog cannot be read.
        ValueError: Catalog YAML/schema is invalid.
    """
    resolved = _canonicalize_path(path or default_scopes_catalog_path())
    return _load_scope_catalog_cached(
        str(resolved),
        _get_mtime_ns(resolved),
    )
