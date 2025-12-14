from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

DecisionReason = Literal[
    "allowed",
    "tool_not_in_allowed_tools",
    "unknown_scope",
    "no_matching_policy",
]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: DecisionReason
    matched_scope: Optional[str] = None


@dataclass(frozen=True)
class MethodPolicy:
    all_methods: bool
    methods: frozenset[str]

    def allows(
        self,
        *,
        method: str,
    ) -> bool:
        if self.all_methods:
            return True
        return method in self.methods


@dataclass(frozen=True)
class ToolPolicy:
    all_tools: bool
    tools: frozenset[str]

    def allows(
        self,
        *,
        tool: str,
    ) -> bool:
        if self.all_tools:
            return True
        return tool in self.tools


@dataclass(frozen=True)
class ServerPermission:
    server: str
    methods: MethodPolicy
    tools: Optional[ToolPolicy]


@dataclass(frozen=True)
class AgentActionPermission:
    action: str
    resources: tuple[str, ...]


@dataclass(frozen=True)
class UIActionPermission:
    action: str
    resources: tuple[str, ...]


@dataclass(frozen=True)
class ScopeDefinition:
    name: str
    server_permissions: tuple[ServerPermission, ...]
    agent_permissions: tuple[AgentActionPermission, ...]


@dataclass(frozen=True)
class ScopeCatalog:
    path: Path
    ui_scopes: dict[str, dict[str, UIActionPermission]]
    group_mappings: dict[str, tuple[str, ...]]
    scopes: dict[str, ScopeDefinition]

    def get_scope(
        self,
        scope_name: str,
    ) -> Optional[ScopeDefinition]:
        return self.scopes.get(scope_name)
