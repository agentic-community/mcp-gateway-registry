"""
File-based repository for authorization scopes storage.

Extracts all scopes.yml file I/O logic while maintaining identical behavior.
"""

import logging
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..interfaces import ScopeRepositoryBase

logger = logging.getLogger(__name__)


class FileScopeRepository(ScopeRepositoryBase):
    """File-based implementation of scope repository."""

    def __init__(self):
        self._scopes_data: Dict[str, Any] = {}
        self._scopes_file = Path("/app/auth_server/scopes.yml")
        self._alt_scopes_file = Path("/app/auth_server/auth_config/scopes.yml")

    async def load_all(self) -> None:
        """Load all scopes from scopes.yml file."""
        # Check primary location first, then alternative (EFS mount)
        if self._scopes_file.exists():
            scopes_file = self._scopes_file
        elif self._alt_scopes_file.exists():
            scopes_file = self._alt_scopes_file
            logger.info(f"Using alternative scopes file at {scopes_file}")
        else:
            logger.error(f"Scopes file not found at {self._scopes_file} or {self._alt_scopes_file}")
            self._scopes_data = {}
            return

        logger.info(f"Loading scopes from {scopes_file}...")

        try:
            with open(scopes_file, 'r') as f:
                self._scopes_data = yaml.safe_load(f)

            if not isinstance(self._scopes_data, dict):
                logger.warning("Invalid scopes file format, resetting")
                self._scopes_data = {}
            else:
                logger.info(f"Successfully loaded scopes from {scopes_file}")

        except Exception as e:
            logger.error(f"Failed to read scopes file: {e}", exc_info=True)
            self._scopes_data = {}

    async def _save_scopes(self) -> bool:
        """Save scopes data to file."""
        try:
            backup_file = self._scopes_file.with_suffix('.backup')

            shutil.copy2(self._scopes_file, backup_file)

            class NoAnchorDumper(yaml.SafeDumper):
                def ignore_aliases(self, data):
                    return True

            with open(self._scopes_file, 'w') as f:
                yaml.dump(
                    self._scopes_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    Dumper=NoAnchorDumper
                )

            logger.info(f"Successfully updated scopes file at {self._scopes_file}")

            if backup_file.exists():
                backup_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to write scopes file: {e}", exc_info=True)
            if backup_file.exists():
                shutil.copy2(backup_file, self._scopes_file)
                logger.info("Restored scopes file from backup")
            return False

    async def get_ui_scopes(
        self,
        group_name: str,
    ) -> Dict[str, Any]:
        """Get UI scopes for a Keycloak group."""
        ui_scopes = self._scopes_data.get("UI-Scopes", {})
        return ui_scopes.get(group_name, {})

    async def get_group_mappings(
        self,
        keycloak_group: str,
    ) -> List[str]:
        """Get scope names mapped to a Keycloak group."""
        group_mappings = self._scopes_data.get("group_mappings", {})
        return group_mappings.get(keycloak_group, [])

    async def get_server_scopes(
        self,
        scope_name: str,
    ) -> List[Dict[str, Any]]:
        """Get server access rules for a scope."""
        return self._scopes_data.get(scope_name, [])

    async def add_server_scope(
        self,
        server_path: str,
        scope_name: str,
        methods: List[str],
        tools: Optional[List[str]] = None,
    ) -> bool:
        """Add scope for a server."""
        try:
            server_name = server_path.lstrip('/')

            server_entry = {
                "server": server_name,
                "methods": methods,
                "tools": tools
            }

            if scope_name not in self._scopes_data:
                logger.warning(f"Scope section {scope_name} not found in scopes.yml")
                return False

            if not isinstance(self._scopes_data[scope_name], list):
                logger.warning(f"Scope section {scope_name} is not a list")
                return False

            existing = [s for s in self._scopes_data[scope_name]
                       if s.get('server') == server_name]

            if existing:
                idx = self._scopes_data[scope_name].index(existing[0])
                self._scopes_data[scope_name][idx] = server_entry
                logger.info(f"Updated existing server {server_path} in scope {scope_name}")
            else:
                self._scopes_data[scope_name].append(server_entry)
                logger.info(f"Added server {server_path} to scope {scope_name}")

            return await self._save_scopes()

        except Exception as e:
            logger.error(f"Failed to add server scope: {e}", exc_info=True)
            return False

    async def remove_server_scope(
        self,
        server_path: str,
        scope_name: str,
    ) -> bool:
        """Remove scope for a server."""
        try:
            server_name = server_path.lstrip('/')

            if scope_name not in self._scopes_data:
                logger.warning(f"Scope section {scope_name} not found")
                return False

            if not isinstance(self._scopes_data[scope_name], list):
                logger.warning(f"Scope section {scope_name} is not a list")
                return False

            original_length = len(self._scopes_data[scope_name])
            self._scopes_data[scope_name] = [
                s for s in self._scopes_data[scope_name]
                if s.get('server') != server_name
            ]

            if len(self._scopes_data[scope_name]) < original_length:
                logger.info(f"Removed server {server_path} from scope {scope_name}")
                return await self._save_scopes()
            else:
                logger.warning(f"Server {server_path} not found in scope {scope_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove server scope: {e}", exc_info=True)
            return False


    async def create_group(
        self,
        group_name: str,
        description: str = "",
    ) -> bool:
        """Create a new group in scopes."""
        try:
            if group_name in self._scopes_data:
                logger.warning(f"Group {group_name} already exists in scopes.yml")
                return False

            self._scopes_data[group_name] = []
            logger.info(f"Created new group entry: {group_name}")

            if "group_mappings" not in self._scopes_data:
                self._scopes_data["group_mappings"] = {}

            if group_name not in self._scopes_data["group_mappings"]:
                self._scopes_data["group_mappings"][group_name] = [group_name]
                logger.info(f"Added {group_name} to group_mappings (self-mapping)")

            if "UI-Scopes" not in self._scopes_data:
                self._scopes_data["UI-Scopes"] = {}

            if group_name not in self._scopes_data["UI-Scopes"]:
                self._scopes_data["UI-Scopes"][group_name] = {
                    "list_service": []
                }
                logger.info(f"Added {group_name} to UI-Scopes with empty list_service")

            return await self._save_scopes()

        except Exception as e:
            logger.error(f"Failed to create group {group_name}: {e}", exc_info=True)
            return False


    async def delete_group(
        self,
        group_name: str,
        remove_from_mappings: bool = True,
    ) -> bool:
        """Delete a group from scopes."""
        try:
            if group_name not in self._scopes_data:
                logger.warning(f"Group {group_name} not found in scopes.yml")
                return False

            if isinstance(self._scopes_data[group_name], list) and len(self._scopes_data[group_name]) > 0:
                server_count = len(self._scopes_data[group_name])
                logger.warning(f"Group {group_name} has {server_count} servers assigned")

            del self._scopes_data[group_name]
            logger.info(f"Removed group {group_name} from scopes.yml")

            if remove_from_mappings and "group_mappings" in self._scopes_data:
                modified_mappings = False
                for mapped_group, mapped_scopes in self._scopes_data["group_mappings"].items():
                    if group_name in mapped_scopes:
                        self._scopes_data["group_mappings"][mapped_group].remove(group_name)
                        logger.info(f"Removed {group_name} from group_mappings[{mapped_group}]")
                        modified_mappings = True

                if modified_mappings:
                    logger.info("Updated group_mappings after group deletion")

            return await self._save_scopes()

        except Exception as e:
            logger.error(f"Failed to delete group {group_name}: {e}", exc_info=True)
            return False


    async def import_group(
        self,
        group_name: str,
        description: str = "",
        server_access: list = None,
        group_mappings: list = None,
        ui_permissions: dict = None,
    ) -> bool:
        """
        Import a complete group definition.

        Args:
            group_name: Name of the group
            description: Description of the group
            server_access: List of server access definitions
            group_mappings: List of group names this group maps to
            ui_permissions: Dictionary of UI permissions
        """
        try:
            # Set defaults
            if server_access is None:
                server_access = []
            if group_mappings is None:
                group_mappings = [group_name]
            if ui_permissions is None:
                ui_permissions = {"list_service": []}

            # Update server_access
            self._scopes_data[group_name] = server_access

            # Update group_mappings
            if "group_mappings" not in self._scopes_data:
                self._scopes_data["group_mappings"] = {}
            self._scopes_data["group_mappings"][group_name] = group_mappings

            # Update UI-Scopes
            if "UI-Scopes" not in self._scopes_data:
                self._scopes_data["UI-Scopes"] = {}
            self._scopes_data["UI-Scopes"][group_name] = ui_permissions

            logger.info(f"Imported complete group definition for {group_name}")
            return await self._save_scopes()

        except Exception as e:
            logger.error(f"Failed to import group {group_name}: {e}", exc_info=True)
            return False


    async def get_group(self, group_name: str) -> Dict[str, Any]:
        """Get full details of a specific group."""
        try:
            if group_name not in self._scopes_data:
                logger.warning(f"Group {group_name} not found in scopes.yml")
                return None

            # Get server_access from main scopes data
            server_access = self._scopes_data.get(group_name, [])

            # Get group_mappings
            group_mappings = self._scopes_data.get("group_mappings", {}).get(group_name, [group_name])

            # Get ui_permissions
            ui_permissions = self._scopes_data.get("UI-Scopes", {}).get(group_name, {})

            result = {
                "scope_name": group_name,
                "scope_type": "server_scope",
                "description": "",  # File-based doesn't have separate description field
                "server_access": server_access,
                "group_mappings": group_mappings,
                "ui_permissions": ui_permissions,
                "created_at": "",
                "updated_at": ""
            }

            logger.info(f"Retrieved full group details for {group_name} from scopes.yml")
            return result

        except Exception as e:
            logger.error(f"Failed to get group {group_name}: {e}", exc_info=True)
            return None


    async def list_groups(
        self,
    ) -> Dict[str, Any]:
        """List all groups with server counts."""
        try:
            groups = {}

            for key, value in self._scopes_data.items():
                if key in ["UI-Scopes", "group_mappings"]:
                    continue

                if isinstance(value, list):
                    server_count = len(value)
                    server_names = [s.get("server", "unknown") for s in value if isinstance(s, dict)]

                    groups[key] = {
                        "name": key,
                        "server_count": server_count,
                        "servers": server_names,
                        "in_mappings": []
                    }

            if "group_mappings" in self._scopes_data:
                for mapped_group, mapped_scopes in self._scopes_data["group_mappings"].items():
                    for scope in mapped_scopes:
                        if scope in groups:
                            groups[scope]["in_mappings"].append(mapped_group)

            logger.info(f"Found {len(groups)} groups in scopes.yml")

            return {
                "total_count": len(groups),
                "groups": groups
            }

        except Exception as e:
            logger.error(f"Failed to list groups: {e}", exc_info=True)
            return {
                "total_count": 0,
                "groups": {},
                "error": str(e)
            }


    async def group_exists(
        self,
        group_name: str,
    ) -> bool:
        """Check if a group exists."""
        try:
            return group_name in self._scopes_data
        except Exception as e:
            logger.error(f"Error checking if group exists: {e}", exc_info=True)
            return False


    async def add_server_to_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """Add server to group's UI scopes list_service."""
        try:
            if "UI-Scopes" not in self._scopes_data:
                self._scopes_data["UI-Scopes"] = {}

            if group_name not in self._scopes_data["UI-Scopes"]:
                self._scopes_data["UI-Scopes"][group_name] = {"list_service": []}

            if "list_service" not in self._scopes_data["UI-Scopes"][group_name]:
                self._scopes_data["UI-Scopes"][group_name]["list_service"] = []

            if server_name not in self._scopes_data["UI-Scopes"][group_name]["list_service"]:
                self._scopes_data["UI-Scopes"][group_name]["list_service"].append(server_name)
                logger.info(f"Added {server_name} to UI-Scopes[{group_name}].list_service")
                return await self._save_scopes()
            else:
                logger.info(f"Server {server_name} already in UI-Scopes[{group_name}].list_service")
                return True

        except Exception as e:
            logger.error(f"Failed to add server to UI scopes: {e}", exc_info=True)
            return False


    async def remove_server_from_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """Remove server from group's UI scopes list_service."""
        try:
            if "UI-Scopes" not in self._scopes_data:
                logger.warning("UI-Scopes section not found")
                return False

            if group_name not in self._scopes_data["UI-Scopes"]:
                logger.warning(f"Group {group_name} not found in UI-Scopes")
                return False

            if "list_service" not in self._scopes_data["UI-Scopes"][group_name]:
                logger.warning(f"list_service not found in UI-Scopes[{group_name}]")
                return False

            if server_name in self._scopes_data["UI-Scopes"][group_name]["list_service"]:
                self._scopes_data["UI-Scopes"][group_name]["list_service"].remove(server_name)
                logger.info(f"Removed {server_name} from UI-Scopes[{group_name}].list_service")
                return await self._save_scopes()
            else:
                logger.warning(f"Server {server_name} not found in UI-Scopes[{group_name}].list_service")
                return False

        except Exception as e:
            logger.error(f"Failed to remove server from UI scopes: {e}", exc_info=True)
            return False


    async def add_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """Add a scope to group mappings."""
        try:
            if "group_mappings" not in self._scopes_data:
                self._scopes_data["group_mappings"] = {}

            if group_name not in self._scopes_data["group_mappings"]:
                self._scopes_data["group_mappings"][group_name] = []

            if scope_name not in self._scopes_data["group_mappings"][group_name]:
                self._scopes_data["group_mappings"][group_name].append(scope_name)
                logger.info(f"Added scope {scope_name} to group_mappings[{group_name}]")
                return await self._save_scopes()
            else:
                logger.info(f"Scope {scope_name} already in group_mappings[{group_name}]")
                return True

        except Exception as e:
            logger.error(f"Failed to add group mapping: {e}", exc_info=True)
            return False


    async def remove_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """Remove a scope from group mappings."""
        try:
            if "group_mappings" not in self._scopes_data:
                logger.warning("group_mappings section not found")
                return False

            if group_name not in self._scopes_data["group_mappings"]:
                logger.warning(f"Group {group_name} not found in group_mappings")
                return False

            if scope_name in self._scopes_data["group_mappings"][group_name]:
                self._scopes_data["group_mappings"][group_name].remove(scope_name)
                logger.info(f"Removed scope {scope_name} from group_mappings[{group_name}]")
                return await self._save_scopes()
            else:
                logger.warning(f"Scope {scope_name} not found in group_mappings[{group_name}]")
                return False

        except Exception as e:
            logger.error(f"Failed to remove group mapping: {e}", exc_info=True)
            return False


    async def get_all_group_mappings(
        self,
    ) -> Dict[str, List[str]]:
        """Get all group mappings."""
        try:
            return self._scopes_data.get("group_mappings", {})
        except Exception as e:
            logger.error(f"Failed to get group mappings: {e}", exc_info=True)
            return {}


    async def add_server_to_multiple_scopes(
        self,
        server_path: str,
        scope_names: List[str],
        methods: List[str],
        tools: List[str],
    ) -> bool:
        """Add server to multiple scopes at once."""
        try:
            success = True
            for scope_name in scope_names:
                result = await self.add_server_scope(
                    server_path,
                    scope_name,
                    methods,
                    tools
                )
                if not result:
                    logger.warning(f"Failed to add server {server_path} to scope {scope_name}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Failed to add server to multiple scopes: {e}", exc_info=True)
            return False


    async def remove_server_from_all_scopes(
        self,
        server_path: str,
    ) -> bool:
        """Remove server from all scopes."""
        try:
            server_name = server_path.lstrip('/')

            sections = [
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
                "mcp-servers-restricted/read",
                "mcp-servers-restricted/execute"
            ]

            modified = False
            for section in sections:
                if section in self._scopes_data:
                    original_length = len(self._scopes_data[section])
                    self._scopes_data[section] = [
                        s for s in self._scopes_data[section]
                        if s.get('server') != server_name
                    ]

                    if len(self._scopes_data[section]) < original_length:
                        logger.info(f"Removed server {server_path} from section {section}")
                        modified = True

            if modified:
                return await self._save_scopes()
            else:
                logger.warning(f"Server {server_path} not found in any scope sections")
                return False

        except Exception as e:
            logger.error(f"Failed to remove server from all scopes: {e}", exc_info=True)
            return False
