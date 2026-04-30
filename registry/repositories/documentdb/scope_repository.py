"""DocumentDB-based repository for authorization scopes storage."""

import logging
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ..interfaces import ScopeRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class DocumentDBScopeRepository(ScopeRepositoryBase):
    """DocumentDB implementation of scope repository using embedded documents."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_scopes")
        self._scopes_cache: dict[str, Any] = {}

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def load_all(self) -> None:
        """Load all scopes from DocumentDB."""
        logger.info(f"Loading scopes from DocumentDB collection: {self._collection_name}")
        collection = await self._get_collection()

        try:
            cursor = collection.find({})
            self._scopes_cache = {
                "UI-Scopes": {},
                "group_mappings": {},
            }

            async for doc in cursor:
                scope_name = doc.get("_id")

                # UI permissions: scope_name -> ui_permissions
                if doc.get("ui_permissions"):
                    self._scopes_cache["UI-Scopes"][scope_name] = doc.get("ui_permissions", {})

                # Group mappings: keycloak_group -> [scope_names]
                # Build reverse mapping from scope's group_mappings list
                for keycloak_group in doc.get("group_mappings", []):
                    if keycloak_group not in self._scopes_cache["group_mappings"]:
                        self._scopes_cache["group_mappings"][keycloak_group] = []
                    if scope_name not in self._scopes_cache["group_mappings"][keycloak_group]:
                        self._scopes_cache["group_mappings"][keycloak_group].append(scope_name)

                # Scope definitions: scope_name -> [access_rules]
                if doc.get("server_access"):
                    self._scopes_cache[scope_name] = doc.get("server_access", [])

            logger.info("Loaded scopes from DocumentDB")
        except Exception as e:
            logger.error(f"Error loading scopes from DocumentDB: {e}", exc_info=True)
            self._scopes_cache = {"UI-Scopes": {}, "group_mappings": {}}

    async def get_ui_scopes(
        self,
        group_name: str,
    ) -> dict[str, Any]:
        """Get UI scopes for a Keycloak group - queries DocumentDB directly."""
        logger.debug(f"DocumentDB READ: Getting UI scopes for group '{group_name}' from DB")
        collection = await self._get_collection()

        try:
            group_doc = await collection.find_one({"_id": group_name})
            if not group_doc:
                logger.debug(f"DocumentDB READ: Group '{group_name}' not found")
                return {}

            scopes = group_doc.get("ui_permissions", {})
            logger.debug(f"DocumentDB READ: Found {len(scopes)} UI scopes for group '{group_name}'")
            return scopes
        except Exception as e:
            logger.error(f"Error getting UI scopes for group '{group_name}': {e}", exc_info=True)
            return {}

    async def get_group_mappings(
        self,
        keycloak_group: str,
    ) -> list[str]:
        """Get scope names mapped to a group (Keycloak group name or Entra ID group Object ID).

        The scopes collection stores documents with:
        - _id: scope name (e.g., 'registry-admins')
        - group_mappings: list of group identifiers that have this scope

        This method finds all scopes where the given group appears in group_mappings.
        """
        logger.debug(f"DocumentDB READ: Getting group mappings for '{keycloak_group}' from DB")
        collection = await self._get_collection()

        try:
            # Find all scope documents where group_mappings array contains this group
            cursor = collection.find({"group_mappings": keycloak_group})
            scope_names = [doc["_id"] async for doc in cursor]

            logger.debug(
                f"DocumentDB READ: Found {len(scope_names)} scopes for group "
                f"'{keycloak_group}': {scope_names}"
            )
            return scope_names
        except Exception as e:
            logger.error(f"Error getting group mappings for '{keycloak_group}': {e}", exc_info=True)
            return []

    async def get_server_scopes(
        self,
        scope_name: str,
    ) -> list[dict[str, Any]]:
        """Get server access rules for a scope - queries DocumentDB directly."""
        logger.debug(
            f"DocumentDB READ: Getting server access rules for scope '{scope_name}' from DB"
        )
        collection = await self._get_collection()

        try:
            # Find the group document that contains this scope
            group_doc = await collection.find_one({"_id": scope_name})
            if not group_doc:
                logger.debug(f"DocumentDB READ: Scope '{scope_name}' not found")
                return []

            # Extract server access rules from the server_access array
            server_access = group_doc.get("server_access", [])

            # Flatten the access rules from all scope entries
            # Handle two formats:
            # 1. New format: {"scope_name": "...", "access_rules": [...]}
            # 2. Old/direct format: {"server": "...", "methods": [...], "tools": [...]}
            all_rules = []
            for scope_entry in server_access:
                # Check if this entry has "access_rules" (new format)
                if "access_rules" in scope_entry:
                    access_rules = scope_entry.get("access_rules", [])
                    all_rules.extend(access_rules)
                # Check if this entry is a direct server access rule (old format)
                elif "server" in scope_entry:
                    all_rules.append(scope_entry)
                # Skip entries that are not server access rules (e.g., agent permissions)

            logger.debug(
                f"DocumentDB READ: Found {len(all_rules)} access rules for scope '{scope_name}'"
            )
            return all_rules
        except Exception as e:
            logger.error(f"Error getting server scopes for '{scope_name}': {e}", exc_info=True)
            return []

    async def add_server_scope(
        self,
        server_path: str,
        scope_name: str,
        methods: list[str],
        tools: list[str] | None = None,
    ) -> bool:
        """Add scope for a server."""
        try:
            collection = await self._get_collection()
            server_name = server_path.lstrip("/")

            server_entry = {"server": server_name, "methods": methods, "tools": tools}

            result = await collection.update_one(
                {"_id": scope_name},
                {
                    "$push": {
                        "server_access": {
                            "$each": [{"scope_name": scope_name, "access_rules": [server_entry]}]
                        }
                    },
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Scope '{scope_name}' not found")
                return False

            self._scopes_cache.setdefault(scope_name, []).append(server_entry)

            logger.info(f"Added server '{server_name}' to scope '{scope_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add server scope in DocumentDB: {e}", exc_info=True)
            return False

    async def remove_server_scope(
        self,
        server_path: str,
        scope_name: str,
    ) -> bool:
        """Remove scope for a server."""
        try:
            collection = await self._get_collection()
            server_name = server_path.lstrip("/")

            result = await collection.update_one(
                {"_id": scope_name},
                {
                    "$pull": {
                        "server_access": {
                            "scope_name": scope_name,
                            "access_rules.server": server_name,
                        }
                    },
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Scope '{scope_name}' not found")
                return False

            if scope_name in self._scopes_cache:
                self._scopes_cache[scope_name] = [
                    s for s in self._scopes_cache[scope_name] if s.get("server") != server_name
                ]

            logger.info(f"Removed server '{server_name}' from scope '{scope_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to remove server scope in DocumentDB: {e}", exc_info=True)
            return False

    async def create_group(
        self,
        group_name: str,
        description: str = "",
    ) -> bool:
        """Create a new group in scopes."""
        try:
            collection = await self._get_collection()

            doc = {
                "_id": group_name,
                "scope_type": "group",
                "description": description,
                "server_access": [],
                "group_mappings": [],
                "ui_permissions": {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            await collection.insert_one(doc)

            self._scopes_cache.setdefault("UI-Scopes", {})[group_name] = {}
            self._scopes_cache.setdefault("group_mappings", {})[group_name] = []

            logger.info(f"Created group '{group_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create group in DocumentDB: {e}", exc_info=True)
            return False

    async def delete_group(
        self,
        group_name: str,
        remove_from_mappings: bool = True,
    ) -> bool:
        """Delete a group from scopes."""
        try:
            collection = await self._get_collection()

            result = await collection.delete_one({"_id": group_name})

            if result.deleted_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            self._scopes_cache.get("UI-Scopes", {}).pop(group_name, None)
            self._scopes_cache.get("group_mappings", {}).pop(group_name, None)

            logger.info(f"Deleted group '{group_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete group in DocumentDB: {e}", exc_info=True)
            return False

    async def get_group(
        self,
        group_name: str,
    ) -> dict[str, Any]:
        """Get full details of a specific group."""
        collection = await self._get_collection()

        try:
            group_doc = await collection.find_one({"_id": group_name})
            if not group_doc:
                return None

            group_doc["scope_name"] = group_doc.pop("_id")
            return group_doc
        except Exception as e:
            logger.error(f"Error getting group '{group_name}' from DocumentDB: {e}", exc_info=True)
            return None

    async def list_groups(self) -> dict[str, Any]:
        """List all groups with server counts."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({})
            groups = {}
            async for doc in cursor:
                group_name = doc.get("_id")
                server_count = len(doc.get("server_access", []))
                groups[group_name] = {
                    "server_count": server_count,
                    "ui_scopes": doc.get("ui_permissions", {}),
                    "mappings": doc.get("group_mappings", []),
                }
            return groups
        except Exception as e:
            logger.error(f"Error listing groups from DocumentDB: {e}", exc_info=True)
            return {}

    async def group_exists(
        self,
        group_name: str,
    ) -> bool:
        """Check if a group exists."""
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({"_id": group_name})
            return count > 0
        except Exception as e:
            logger.error(f"Error checking group existence in DocumentDB: {e}", exc_info=True)
            return False

    async def add_server_to_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """Add server to group's UI scopes list_service."""
        try:
            collection = await self._get_collection()

            result = await collection.update_one(
                {"_id": group_name},
                {
                    "$addToSet": {"ui_permissions.list_service": server_name},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            logger.info(f"Added server '{server_name}' to UI scopes for group '{group_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add server to UI scopes in DocumentDB: {e}", exc_info=True)
            return False

    async def remove_server_from_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """Remove server from group's UI scopes list_service."""
        try:
            collection = await self._get_collection()

            result = await collection.update_one(
                {"_id": group_name},
                {
                    "$pull": {"ui_permissions.list_service": server_name},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            logger.info(f"Removed server '{server_name}' from UI scopes for group '{group_name}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to remove server from UI scopes in DocumentDB: {e}", exc_info=True
            )
            return False

    async def add_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """Add a scope to group mappings."""
        try:
            collection = await self._get_collection()

            result = await collection.update_one(
                {"_id": group_name},
                {
                    "$addToSet": {"group_mappings": scope_name},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            logger.info(f"Added mapping '{scope_name}' to group '{group_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add group mapping in DocumentDB: {e}", exc_info=True)
            return False

    async def remove_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """Remove a scope from group mappings."""
        try:
            collection = await self._get_collection()

            result = await collection.update_one(
                {"_id": group_name},
                {
                    "$pull": {"group_mappings": scope_name},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            logger.info(f"Removed mapping '{scope_name}' from group '{group_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to remove group mapping in DocumentDB: {e}", exc_info=True)
            return False

    async def get_all_group_mappings(self) -> dict[str, list[str]]:
        """Get all group mappings."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({})
            mappings = {}
            async for doc in cursor:
                group_name = doc.get("_id")
                mappings[group_name] = doc.get("group_mappings", [])
            return mappings
        except Exception as e:
            logger.error(f"Error getting all group mappings from DocumentDB: {e}", exc_info=True)
            return {}

    async def add_server_to_multiple_scopes(
        self,
        server_path: str,
        scope_names: list[str],
        methods: list[str],
        tools: list[str],
    ) -> bool:
        """Add server to multiple scopes at once."""
        try:
            for scope_name in scope_names:
                success = await self.add_server_scope(server_path, scope_name, methods, tools)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to add server to multiple scopes: {e}", exc_info=True)
            return False

    async def remove_server_from_all_scopes(
        self,
        server_path: str,
    ) -> bool:
        """Remove server from all scopes."""
        try:
            collection = await self._get_collection()
            server_name = server_path.lstrip("/")

            result = await collection.update_many(
                {},
                {
                    "$pull": {"server_access": {"access_rules.server": server_name}},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            for scope_name in list(self._scopes_cache.keys()):
                if scope_name not in ["UI-Scopes", "group_mappings"]:
                    self._scopes_cache[scope_name] = [
                        s for s in self._scopes_cache[scope_name] if s.get("server") != server_name
                    ]

            logger.info(f"Removed server '{server_name}' from all scopes")
            return True
        except Exception as e:
            logger.error(
                f"Failed to remove server from all scopes in DocumentDB: {e}", exc_info=True
            )
            return False

    async def import_group(
        self,
        group_name: str,
        description: str = "",
        server_access: list = None,
        group_mappings: list = None,
        ui_permissions: dict = None,
        agent_access: list = None,
    ) -> bool:
        """
        Import a complete group definition.

        Args:
            group_name: Name of the group
            description: Description of the group
            server_access: List of server access definitions
            group_mappings: List of group names this group maps to
            ui_permissions: Dictionary of UI permissions
            agent_access: List of agent paths this group can access

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            # Set defaults
            if server_access is None:
                server_access = []
            if group_mappings is None:
                group_mappings = [group_name]
            if ui_permissions is None:
                ui_permissions = {"list_service": []}
            if agent_access is None:
                agent_access = []

            # Create the complete group document
            group_doc = {
                "_id": group_name,
                "scope_type": "group",
                "description": description,
                "server_access": server_access,
                "group_mappings": group_mappings,
                "ui_permissions": ui_permissions,
                "agent_access": agent_access,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            # Use replace_one with upsert=True to create or replace the entire document
            result = await collection.replace_one({"_id": group_name}, group_doc, upsert=True)

            # Update in-memory cache
            self._scopes_cache.setdefault("UI-Scopes", {})[group_name] = ui_permissions
            self._scopes_cache.setdefault("group_mappings", {})[group_name] = group_mappings

            # Update server access in cache
            for scope_entry in server_access:
                scope_name = scope_entry.get("scope_name")
                if scope_name:
                    if scope_name not in self._scopes_cache:
                        self._scopes_cache[scope_name] = []
                    self._scopes_cache[scope_name].extend(scope_entry.get("access_rules", []))

            if result.upserted_id:
                logger.info(f"Created new group '{group_name}' via import")
            else:
                logger.info(f"Updated existing group '{group_name}' via import")

            return True

        except Exception as e:
            logger.error(f"Failed to import group {group_name}: {e}", exc_info=True)
            return False
