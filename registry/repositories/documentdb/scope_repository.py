"""DocumentDB-based repository for authorization scopes storage."""

import logging
import re
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ...auth.privileged_constants import (
    ADMIN_ACTION_PREFIXES,
    PRIVILEGED_GRANTS,
    PRIVILEGED_SCOPE_NAMES,
    is_admin_conferring_action,
)
from ..interfaces import ScopeRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)

_GUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Last-line-of-defense reject set for server-scope writes. A server_path that
# normalizes to one of these collides with the cross-server wildcard sentinel
# and would grant access to every server. The registration guard
# registry.utils.validate_server_path already rejects these paths; this mirror
# ensures a caller that bypasses that guard still cannot inject a wildcard row.
#
# MUST stay in sync with registry.utils.url_guard._RESERVED_SERVER_PATH_NAMES
# and registry.auth.access_resolver._WILDCARD_VALUES. Mirrored locally rather
# than imported to keep this repository layer independent of the util layer.
_RESERVED_SERVER_PATH_NAMES: frozenset[str] = frozenset({"all", "*"})


def _looks_like_guid(
    value: str,
) -> bool:
    """Return True if the string matches the GUID/UUID shape."""
    return bool(_GUID_RE.match(value or ""))


# Admin-boundary constants are imported from registry.auth.privileged_constants
# (the dependency-free leaf module that is the single source of truth, shared
# with the admin-derivation rule in registry/auth/dependencies.py). Module-level
# aliases are kept so existing references and tests that import these private
# names from this module keep working.
#
# IMPORTANT: admin is conferred only by the literal "all" grant, NOT "*". "*" on
# a mutating action grants access to every server WITHOUT admin (see issue
# #663), which is why PRIVILEGED_GRANTS is {"all"} and not {"all", "*"}.
_PRIVILEGED_ACTION_PREFIXES = ADMIN_ACTION_PREFIXES
_PRIVILEGED_GRANTS = PRIVILEGED_GRANTS
_PRIVILEGED_SCOPE_NAMES = PRIVILEGED_SCOPE_NAMES


def _grants_admin(
    ui_permissions: dict | None,
) -> bool:
    """Return True if ui_permissions would confer admin privileges.

    Admin is conferred by any mutating UI action granted with "all" access,
    EXCEPT per-type custom-entity scopes (create_/modify_/delete_<type>_entity)
    and skill-management scopes (publish_/modify_/delete_/toggle_skill), which
    is_admin_conferring_action excludes. This is the same rule _user_is_admin
    uses to derive admin status per request (both defer to the shared
    is_admin_conferring_action so the write guard and the admin check cannot
    drift), so a group carrying genuinely-admin permissions promotes its members
    to admin.

    Args:
        ui_permissions: Dict mapping UI actions to lists of allowed resources.

    Returns:
        True if the permissions would confer admin, False otherwise.
    """
    if not ui_permissions:
        return False
    for action, resources in ui_permissions.items():
        if is_admin_conferring_action(action) and (_PRIVILEGED_GRANTS & set(resources or [])):
            return True
    return False


def _is_privileged_write(
    group_name: str,
    group_mappings: list | None,
    ui_permissions: dict | None,
) -> bool:
    """Return True if a group write would confer administrative access.

    Defense-in-depth that mirrors the service layer's
    ``_import_touches_privileged_scope`` so the deepest (repository) layer is
    also the most complete -- it catches all three admin-conferring vectors:

    1. The scope/group itself is a privileged name.
    2. It maps to a privileged group. ``_grants_admin``
       alone does NOT cover this.
    3. Its ui_permissions grant a mutating action to "all" (``_grants_admin``).

    Args:
        group_name: The scope/group being written.
        group_mappings: IdP group names this scope maps to.
        ui_permissions: UI permission grants for this scope.

    Returns:
        True if the write requires admin authorization.
    """
    if group_name in _PRIVILEGED_SCOPE_NAMES:
        return True
    for mapped in group_mappings or []:
        if mapped in _PRIVILEGED_SCOPE_NAMES:
            return True
    return _grants_admin(ui_permissions)


def _flatten_server_access(
    server_access: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten a scope document's ``server_access`` into a flat rule list.

    Handles three on-disk formats:
    1. Grouped format: ``{"scope_name": "...", "access_rules": [...]}`` (the inner
       rules are spliced in as-is; they may be server or agent rules).
    2. Direct server rule: ``{"server": "...", "methods": [...], "tools": [...]}``.
    3. Direct agent rule: ``{"agent": "...", "actions": [...]}`` -- the per-agent
       shape mirrors the server rule (``agent`` is the identifier, ``actions`` its
       siblings), consumed by ``validate_a2a_agent_access``.

    An entry that matches none of these is skipped. Shared by
    ``get_server_scopes`` and ``get_server_scopes_bulk`` so the single and batch
    paths produce byte-identical rule lists.
    """
    all_rules: list[dict[str, Any]] = []
    for scope_entry in server_access:
        if "access_rules" in scope_entry:
            all_rules.extend(scope_entry.get("access_rules", []))
        elif "server" in scope_entry:
            all_rules.append(scope_entry)
        elif "agent" in scope_entry:
            all_rules.append(scope_entry)
    return all_rules


def _server_access_has_invalid_server_rule(
    server_access: list[dict[str, Any]] | None,
) -> bool:
    """Return True if any server rule has a reserved-wildcard or empty ``server``.

    ``import_group`` writes ``server_access`` straight to the collection with
    ``replace_one`` and never routes through :meth:`add_server_scope`, so its
    sink guard does not apply. This scans the imported rules (in both the
    grouped ``access_rules`` shape and the direct ``server`` rule shape) so a
    group import cannot inject a degenerate server rule that
    :func:`registry.utils.validate_server_path` would have rejected at
    registration. Two cases are refused, mirroring that guard:

    1. ``server: "all"`` / ``server: "*"`` -- the resolver promotes these to a
       cross-server wildcard, granting access to every server.
    2. ``server: ""`` (or slashes-only) -- grants no access in the resolver, but
       the same path renders as a gateway-wide ``location /`` nginx block
       (issue #1501); reject it so this mirror stays in sync with the guard.

    The compare normalizes the same way the resolver's write path does
    (``lstrip("/")`` + trailing-slash strip + case-fold). Only rules that
    actually carry a ``server`` key are inspected; agent rules (which have an
    ``agent`` key and no ``server`` key) are left alone.

    Args:
        server_access: The ``server_access`` list from a group import.

    Returns:
        True if any server rule is a reserved wildcard or empty/slashes-only.
    """
    for rule in _flatten_server_access(server_access or []):
        if not isinstance(rule, dict) or "server" not in rule:
            continue
        # Coerce with str() exactly as the resolver does
        # (access_resolver.py: `str(server_name) in _WILDCARD_VALUES`) so the
        # scan cannot be blind to a non-string value the resolver would still
        # promote to a wildcard.
        normalized = str(rule.get("server") or "").strip("/").lower()
        if not normalized or normalized in _RESERVED_SERVER_PATH_NAMES:
            return True
    return False


def _backfill_is_idp_managed(
    doc: dict,
) -> bool:
    """Heuristic for legacy scope documents that predate issue #946.

    - Any GUID-shaped entry in group_mappings is a strong signal of an
      Entra-managed group, so treat the scope as IdP-managed.
    - Otherwise default to True to preserve pre-#946 behavior. The 403/404
      fall-through in PATCH/DELETE handles misclassified records safely.
    """
    for mapping in doc.get("group_mappings") or []:
        if isinstance(mapping, str) and _looks_like_guid(mapping):
            return True
    return True


class DocumentDBScopeRepository(ScopeRepositoryBase):
    """DocumentDB implementation of scope repository using embedded documents."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_scopes")
        self._scopes_cache: dict[str, Any] = {}
        self._indexes_created = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection, creating indexes on first access."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
            await self._ensure_indexes()
        return self._collection

    async def _ensure_indexes(self) -> None:
        """Create the multikey index on group_mappings if not present.

        ``get_group_mappings`` / ``get_group_mappings_bulk`` filter on
        ``group_mappings`` (``find({"group_mappings": ...})`` and the bulk
        ``$in``). Without an index on that array field each query is a full
        collection scan; the multikey index turns it into an index seek and
        is what makes the bulk ``$in`` actually cheap.
        """
        if self._indexes_created or self._collection is None:
            return
        try:
            await self._collection.create_index("group_mappings")
            self._indexes_created = True
            logger.info(f"Created group_mappings index for {self._collection_name}")
        except Exception as e:
            logger.warning(
                f"Could not create group_mappings index for {self._collection_name}: {e}"
            )

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

    async def get_group_mappings_bulk(
        self,
        groups: list[str],
    ) -> list[str]:
        """Union of scope names mapped to any of the given groups in one query.

        Collapses the per-group ``find`` fan-out into a single ``$in`` query,
        backed by the ``group_mappings`` index. Returns a de-duplicated,
        order-stable list of scope names.
        """
        unique = sorted({g for g in groups if g})
        if not unique:
            return []

        collection = await self._get_collection()
        try:
            cursor = collection.find({"group_mappings": {"$in": unique}})
            seen: set[str] = set()
            scope_names: list[str] = []
            async for doc in cursor:
                scope_id = doc["_id"]
                if scope_id not in seen:
                    seen.add(scope_id)
                    scope_names.append(scope_id)
            logger.debug(
                f"DocumentDB READ: bulk group mappings for {len(unique)} groups "
                f"-> {len(scope_names)} scopes"
            )
            return scope_names
        except Exception as e:
            logger.error(f"Error getting bulk group mappings: {e}", exc_info=True)
            return []

    async def get_all_mapped_group_names(self) -> set[str]:
        """Union of every scope document's group_mappings array.

        Uses a single projected query (not the in-memory cache) so the result
        reflects group mappings added after the process last loaded scopes.
        Returns an empty set on error so callers can fail open.
        """
        logger.debug("DocumentDB READ: Getting all mapped group names from DB")
        collection = await self._get_collection()

        names: set[str] = set()
        try:
            cursor = collection.find({}, {"group_mappings": 1})
            async for doc in cursor:
                names.update(doc.get("group_mappings") or [])
            logger.debug(f"DocumentDB READ: Found {len(names)} distinct mapped group names")
            return names
        except Exception as e:
            logger.error(f"Error getting all mapped group names: {e}", exc_info=True)
            return set()

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

            # Extract and flatten the access rules from the server_access array.
            all_rules = _flatten_server_access(group_doc.get("server_access", []))

            logger.debug(
                f"DocumentDB READ: Found {len(all_rules)} access rules for scope '{scope_name}'"
            )
            return all_rules
        except Exception as e:
            logger.error(f"Error getting server scopes for '{scope_name}': {e}", exc_info=True)
            return []

    async def get_server_scopes_bulk(
        self,
        scope_names: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch server access rules for many scopes in a single ``$in`` query.

        Collapses the per-scope ``find_one`` fan-out (one round-trip per
        scope) into one round-trip. On a remote cluster a user with many
        groups would otherwise pay one network latency per scope here.
        """
        unique = sorted({s for s in scope_names if s})
        if not unique:
            return {}

        collection = await self._get_collection()
        try:
            cursor = collection.find({"_id": {"$in": unique}})
            result: dict[str, list[dict[str, Any]]] = {}
            async for doc in cursor:
                rules = _flatten_server_access(doc.get("server_access", []))
                if rules:
                    result[doc["_id"]] = rules
            logger.debug(
                f"DocumentDB READ: bulk server scopes for {len(unique)} scopes "
                f"-> {len(result)} with rules"
            )
            return result
        except Exception as e:
            logger.error(f"Error getting bulk server scopes: {e}", exc_info=True)
            return {}

    async def get_ui_scopes_bulk(
        self,
        group_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch UI scopes for many groups/scopes in a single ``$in`` query.

        Batch equivalent of ``get_ui_scopes``; same round-trip collapsing
        rationale as ``get_server_scopes_bulk``.
        """
        unique = sorted({g for g in group_names if g})
        if not unique:
            return {}

        collection = await self._get_collection()
        try:
            cursor = collection.find({"_id": {"$in": unique}})
            result: dict[str, dict[str, Any]] = {}
            async for doc in cursor:
                ui_permissions = doc.get("ui_permissions", {})
                if ui_permissions:
                    result[doc["_id"]] = ui_permissions
            logger.debug(
                f"DocumentDB READ: bulk UI scopes for {len(unique)} groups "
                f"-> {len(result)} with permissions"
            )
            return result
        except Exception as e:
            logger.error(f"Error getting bulk UI scopes: {e}", exc_info=True)
            return {}

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

            # Last line of defense against writing a degenerate server path, even
            # if a caller bypassed validate_server_path. Fail closed (return
            # False + log, per this method's error convention).
            #
            # An empty/slashes-only server name is rejected too: it grants no
            # access in the resolver (falsy), but the same path renders as a
            # gateway-wide `location /` nginx block (issue #1501), so the
            # registration guard now rejects it and this mirror stays in sync.
            if not server_name.strip("/"):
                logger.error(
                    "Refusing to add server scope for empty/slashes-only server "
                    "name (from path '%s'): degenerate path, rejected at "
                    "registration",
                    server_path,
                )
                return False

            # Refuse to write a scope row whose server name collides with the
            # cross-server wildcard sentinel: it would grant access to every
            # server in the registry.
            if server_name.rstrip("/").lower() in _RESERVED_SERVER_PATH_NAMES:
                logger.error(
                    "Refusing to add server scope for reserved wildcard name "
                    "'%s' (from path '%s'): would grant cross-server access",
                    server_name,
                    server_path,
                )
                return False

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
        is_idp_managed: bool = True,
    ) -> bool:
        """Create a new group in scopes.

        Args:
            group_name: Name of the group.
            description: Optional description.
            is_idp_managed: Whether PATCH/DELETE should call the upstream IdP.
                Defaults to True to preserve pre-#946 behavior.
        """
        try:
            collection = await self._get_collection()

            doc = {
                "_id": group_name,
                "scope_type": "group",
                "description": description,
                "server_access": [],
                "group_mappings": [],
                "ui_permissions": {},
                "is_idp_managed": is_idp_managed,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            await collection.insert_one(doc)

            self._scopes_cache.setdefault("UI-Scopes", {})[group_name] = {}
            self._scopes_cache.setdefault("group_mappings", {})[group_name] = []

            logger.info(f"Created group '{group_name}' (is_idp_managed={is_idp_managed})")
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
        """Get full details of a specific group.

        Lazily backfills `is_idp_managed` for legacy documents that predate
        issue #946 so every call site sees a populated value.
        """
        collection = await self._get_collection()

        try:
            group_doc = await collection.find_one({"_id": group_name})
            if not group_doc:
                return None

            if "is_idp_managed" not in group_doc:
                backfilled = _backfill_is_idp_managed(group_doc)
                await collection.update_one(
                    {"_id": group_name},
                    {
                        "$set": {
                            "is_idp_managed": backfilled,
                            "updated_at": datetime.utcnow(),
                        }
                    },
                )
                group_doc["is_idp_managed"] = backfilled
                logger.info(
                    "scope_backfill operation=get group=%s is_idp_managed=%s",
                    group_name,
                    backfilled,
                )

            group_doc["scope_name"] = group_doc.pop("_id")
            return group_doc
        except Exception as e:
            logger.error(f"Error getting group '{group_name}' from DocumentDB: {e}", exc_info=True)
            return None

    async def list_groups(self) -> dict[str, Any]:
        """List all groups with server counts.

        Eagerly backfills `is_idp_managed` on any document that predates
        issue #946. The `$exists: False` filter returns nothing once the
        collection has been backfilled, so this is cheap on steady state.
        """
        collection = await self._get_collection()

        try:
            legacy_cursor = collection.find({"is_idp_managed": {"$exists": False}})
            legacy_count = 0
            async for legacy_doc in legacy_cursor:
                backfilled = _backfill_is_idp_managed(legacy_doc)
                await collection.update_one(
                    {"_id": legacy_doc["_id"]},
                    {
                        "$set": {
                            "is_idp_managed": backfilled,
                            "updated_at": datetime.utcnow(),
                        }
                    },
                )
                legacy_count += 1
                logger.info(
                    "scope_backfill operation=list group=%s is_idp_managed=%s",
                    legacy_doc["_id"],
                    backfilled,
                )
            if legacy_count:
                logger.info("scope_backfill batch_count=%d", legacy_count)

            cursor = collection.find({})
            groups = {}
            async for doc in cursor:
                group_name = doc.get("_id")
                server_count = len(doc.get("server_access", []))
                groups[group_name] = {
                    "server_count": server_count,
                    "ui_scopes": doc.get("ui_permissions", {}),
                    "mappings": doc.get("group_mappings", []),
                    "is_idp_managed": doc.get("is_idp_managed", True),
                    "description": doc.get("description"),
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

    async def merge_ui_permissions(
        self,
        group_name: str,
        ui_permissions: dict[str, list[str]],
    ) -> bool:
        """Merge ui_permission keys into a group's ui_permissions ($set per key)."""
        if not ui_permissions:
            return False
        try:
            collection = await self._get_collection()

            set_fields: dict[str, Any] = {
                f"ui_permissions.{key}": value for key, value in ui_permissions.items()
            }
            set_fields["updated_at"] = datetime.utcnow()

            result = await collection.update_one(
                {"_id": group_name},
                {"$set": set_fields},
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            # Keep the in-memory cache aligned so a subsequent read reflects the merge.
            cached = self._scopes_cache.setdefault("UI-Scopes", {}).setdefault(group_name, {})
            cached.update(ui_permissions)

            logger.info(
                "Merged %d ui_permission key(s) into group '%s': %s",
                len(ui_permissions),
                group_name,
                sorted(ui_permissions.keys()),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to merge ui_permissions in DocumentDB: {e}", exc_info=True)
            return False

    async def remove_ui_permission_keys(
        self,
        group_name: str,
        permission_keys: list[str],
    ) -> bool:
        """Unset ui_permission keys from a single group."""
        if not permission_keys:
            return False
        try:
            collection = await self._get_collection()

            unset_fields = {f"ui_permissions.{key}": "" for key in permission_keys}
            result = await collection.update_one(
                {"_id": group_name},
                {
                    "$unset": unset_fields,
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            if result.matched_count == 0:
                logger.error(f"Group '{group_name}' not found")
                return False

            cached = self._scopes_cache.get("UI-Scopes", {}).get(group_name)
            if isinstance(cached, dict):
                for key in permission_keys:
                    cached.pop(key, None)

            logger.info(
                "Removed %d ui_permission key(s) from group '%s': %s",
                len(permission_keys),
                group_name,
                sorted(permission_keys),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to remove ui_permission keys in DocumentDB: {e}", exc_info=True)
            return False

    async def remove_ui_permission_keys_from_all_groups(
        self,
        permission_keys: list[str],
    ) -> int:
        """Unset ui_permission keys from every group that holds any of them."""
        if not permission_keys:
            return 0
        try:
            collection = await self._get_collection()

            unset_fields = {f"ui_permissions.{key}": "" for key in permission_keys}
            # Only touch groups that actually carry at least one of the keys.
            match_filter = {
                "$or": [{f"ui_permissions.{key}": {"$exists": True}} for key in permission_keys]
            }
            result = await collection.update_many(
                match_filter,
                {
                    "$unset": unset_fields,
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            ui_cache = self._scopes_cache.get("UI-Scopes", {})
            for cached in ui_cache.values():
                if isinstance(cached, dict):
                    for key in permission_keys:
                        cached.pop(key, None)

            logger.info(
                "Swept %d ui_permission key(s) from %d group(s): %s",
                len(permission_keys),
                result.modified_count,
                sorted(permission_keys),
            )
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to sweep ui_permission keys in DocumentDB: {e}", exc_info=True)
            return 0

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

    async def list_scope_names(self) -> list[str]:
        """List all scope names (_id values) in the mcp-scopes collection.

        Used by the Issue #1026 legacy-scope startup audit. Projects only
        the _id field so it is cheap even on large scope collections.
        """
        try:
            collection = await self._get_collection()
            cursor = collection.find({}, projection={"_id": 1})
            scope_names: list[str] = []
            async for doc in cursor:
                scope_id = doc.get("_id")
                if isinstance(scope_id, str):
                    scope_names.append(scope_id)
            return scope_names
        except Exception as e:
            logger.error(f"Error listing scope names from DocumentDB: {e}", exc_info=True)
            return []

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
        is_idp_managed: bool = True,
        allow_privileged: bool = False,
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
            is_idp_managed: Whether PATCH/DELETE should call the upstream IdP.
                Defaults to True to preserve pre-#946 behavior for callers
                that do not explicitly pass the flag.
            allow_privileged: Whether to permit writing admin-conferring
                ui_permissions (mutating action with "all"/"*"). Defaults to
                False so untrusted/public callers cannot mint admin groups.
                Admin-gated callers (IAM management routes, IdP sync) pass True.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Defense in depth: refuse to write admin-conferring group definitions
            # unless the caller has explicitly enforced an admin check. The public
            # External API import path never passes allow_privileged=True. This
            # covers all three vectors (privileged name, privileged group_mapping,
            # admin-conferring ui_permissions) -- broader than ui_permissions
            # alone, so the deepest layer is also the most complete.
            if not allow_privileged and _is_privileged_write(
                group_name, group_mappings, ui_permissions
            ):
                logger.error(
                    f"Refusing to import privileged group '{group_name}' "
                    f"without allow_privileged=True (mappings={group_mappings})"
                )
                return False

            # Degenerate-server-rule sink guard for the import path. Unlike
            # add_server_scope, import_group writes server_access directly, so
            # it needs its own check. A server rule named "all"/"*" (cross-server
            # wildcard) or "" (renders as a gateway-wide `location /`, issue
            # #1501) is never legitimate, so this is refused unconditionally --
            # an admin cannot opt into it either. Fail closed (return False + log).
            if _server_access_has_invalid_server_rule(server_access):
                logger.error(
                    f"Refusing to import group '{group_name}': server_access "
                    "contains a reserved wildcard ('all'/'*') or empty server "
                    "name that would grant cross-server access or hijack the gateway"
                )
                return False

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
                "is_idp_managed": is_idp_managed,
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
