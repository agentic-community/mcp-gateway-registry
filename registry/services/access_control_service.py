from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_SCOPES_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "auth_server" / "scopes.yml"
)


def _load_yaml_config(
    config_path: Path,
) -> Optional[dict[str, Any]]:
    if not config_path.exists():
        logger.warning(f"Scopes config not found at {config_path}")
        return None

    with open(config_path, "r") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Scopes config must be a YAML mapping")

    return loaded


class AccessControlService:
    """Access control helpers for registry UI/API.

    Note: EnforceAI runtime authorization lives in `auth_server` and is
    implemented separately. This service exists to support the upstream
    registry-side behavior and its unit tests.
    """

    def __init__(
        self,
        *,
        scopes_config_path: Path = DEFAULT_SCOPES_CONFIG_PATH,
    ) -> None:
        self._scopes_config_path = scopes_config_path
        self._scopes_config: Optional[dict[str, Any]] = _load_yaml_config(
            self._scopes_config_path,
        )

    def reload_config(self) -> None:
        self._scopes_config = _load_yaml_config(
            self._scopes_config_path,
        )

    def get_user_scopes(
        self,
        groups: list[str],
    ) -> list[str]:
        scopes: list[str] = []

        if "mcp-admin" in groups:
            scopes.extend(
                [
                    "mcp-servers-unrestricted/read",
                    "mcp-servers-unrestricted/execute",
                ]
            )

        if "mcp-user" in groups:
            scopes.append("mcp-servers-restricted/read")

        if any(group.startswith("mcp-server-") for group in groups):
            scopes.append("mcp-servers-restricted/execute")

        seen: set[str] = set()
        unique_scopes: list[str] = []
        for scope in scopes:
            if scope in seen:
                continue
            seen.add(scope)
            unique_scopes.append(scope)

        return unique_scopes

    def get_accessible_servers(
        self,
        groups: list[str],
    ) -> set[str]:
        if self._scopes_config is None:
            return set()

        accessible: set[str] = set()
        scopes = self.get_user_scopes(groups)
        for scope in scopes:
            rules = self._scopes_config.get(scope, [])
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                server_name = rule.get("server")
                if isinstance(server_name, str) and server_name:
                    accessible.add(server_name)
        return accessible

    def can_user_access_server(
        self,
        server_name: str,
        groups: list[str],
    ) -> bool:
        if self._scopes_config is None:
            return True

        return server_name in self.get_accessible_servers(groups)

