"""
Repository base classes for data access abstraction.

These abstract base classes define the contract that ALL repository implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..schemas.agent_models import AgentCard
from ..schemas.federation_schema import FederationConfig
from ..schemas.registry_card import RegistryCard

# Import skill models with try/except to avoid circular import issues
try:
    from ..schemas.skill_models import SkillCard
except ImportError:
    SkillCard = None

try:
    from ..schemas.virtual_server_models import VirtualServerConfig
except ImportError:
    VirtualServerConfig = None

try:
    from ..schemas.custom_entity_models import CustomEntityRecord, CustomTypeDescriptor
except ImportError:
    CustomEntityRecord = None
    CustomTypeDescriptor = None


class ServerRepositoryBase(ABC):
    """Abstract base class for MCP server data access."""

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """Get server by path."""
        pass

    @abstractmethod
    async def list_all(
        self,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """List all servers.

        Args:
            exclude_tool_list: If True, omit the heavy ``tool_list`` field from
                each document (DB-side projection) to cut transfer for callers
                that only need metadata. ``num_tools`` is unaffected.

        Returns:
            Dictionary mapping server path to server info.
        """
        pass

    @abstractmethod
    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """List servers with DB-level pagination.

        Args:
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.
            exclude_tool_list: If True, omit the heavy ``tool_list`` field from
                each document (DB-side projection). ``num_tools`` is unaffected.

        Returns:
            Dictionary mapping server path to server info for the requested page.
        """
        pass

    @abstractmethod
    async def list_by_ids(
        self,
        paths: list[str],
    ) -> dict[str, dict[str, Any]]:
        """List servers whose path (_id) is in the given set.

        Used to fetch only the servers a restricted user can access without
        scanning the full collection. Version documents (``{path}:{version}``)
        are never matched because only exact active paths are queried.

        Args:
            paths: Exact server paths to fetch.

        Returns:
            Dictionary mapping server path to server info for found paths.
        """
        pass

    @abstractmethod
    async def list_by_source(
        self,
        source: str,
    ) -> dict[str, dict[str, Any]]:
        """List all servers from a specific federation source.

        Args:
            source: Federation source identifier (e.g., "anthropic")

        Returns:
            Dictionary mapping server path to server info
        """
        pass

    @abstractmethod
    async def create(
        self,
        server_info: dict[str, Any],
    ) -> bool:
        """Create a new server."""
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        server_info: dict[str, Any],
    ) -> bool:
        """Update an existing server."""
        pass

    @abstractmethod
    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a server."""
        pass

    @abstractmethod
    async def delete_with_versions(
        self,
        path: str,
    ) -> int:
        """Delete a server and all its version documents.

        Deletes the active document at `path` and any version documents
        with IDs matching `{path}:{version}`.

        Args:
            path: Server base path (e.g., "/context7")

        Returns:
            Number of documents deleted (0 if none found)
        """
        pass

    @abstractmethod
    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get server enabled/disabled state."""
        pass

    @abstractmethod
    async def get_all_states(self) -> dict[str, bool]:
        """Get enabled/disabled state for all servers in a single query.

        Returns:
            Dict mapping server path to enabled (True) or disabled (False).
        """
        pass

    @abstractmethod
    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set server enabled/disabled state."""
        pass

    @abstractmethod
    async def load_all(self) -> None:
        """Load/reload all servers from storage."""
        pass

    @abstractmethod
    async def count(
        self,
        exclude_versions: bool = False,
    ) -> int:
        """Get total count of servers.

        Args:
            exclude_versions: When True, exclude version documents so the count
                reflects distinct servers only. Defaults to False to preserve
                the historical count of all documents.

        Returns:
            Total number of servers in the repository.
        """
        pass

    @abstractmethod
    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document.

        Args:
            path: Document path/ID
            field: Field name (supports dot notation for nested fields)
            value: New value (None to unset the field)

        Returns:
            True if document was found and updated
        """
        ...

    @abstractmethod
    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> dict[str, dict]:
        """Find documents matching a MongoDB-style filter.

        Args:
            filter_dict: MongoDB-style filter
            limit: optional cap on number of documents returned. None
                returns all matches (legacy behavior).

        Returns:
            Dict mapping path -> document data for matching documents
        """
        ...

    async def find_by_identity_url(
        self,
        identity_url: str,
    ) -> dict[str, Any] | None:
        """Find a server whose ``proxy_pass_url`` normalizes to ``identity_url``.

        Used by ``DuplicateCheckService`` to detect exact-URL
        collisions. ``identity_url`` is the canonical form returned by
        ``normalize_identity_url(url, "mcp_server")``.

        Default implementation scans ``list_all()`` and re-normalizes each
        candidate's ``proxy_pass_url``. Backends with index support
        (DocumentDB) override with a server-side filter.

        Args:
            identity_url: Pre-normalized identity URL (scheme-collapsed).

        Returns:
            The full document with ``path`` populated, or None if no match.
        """
        from ..utils.url_normalize import ENTITY_TYPE_SERVER, normalize_identity_url

        if not identity_url:
            return None
        all_servers = await self.list_all()
        for path, server_info in all_servers.items():
            candidate_url = server_info.get("proxy_pass_url")
            if not candidate_url:
                continue
            if normalize_identity_url(candidate_url, ENTITY_TYPE_SERVER) == identity_url:
                doc = dict(server_info)
                doc["path"] = path
                return doc
        return None


class AgentRepositoryBase(ABC):
    """Abstract base class for A2A agent data access."""

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> AgentCard | None:
        """Get agent by path."""
        pass

    @abstractmethod
    async def list_all(self) -> list[AgentCard]:
        """List all agents."""
        pass

    @abstractmethod
    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AgentCard]:
        """List agents with DB-level pagination.

        Args:
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.

        Returns:
            List of AgentCard objects for the requested page.
        """
        pass

    @abstractmethod
    async def create(
        self,
        agent: AgentCard,
    ) -> AgentCard:
        """Create a new agent."""
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> AgentCard:
        """Update an existing agent."""
        pass

    @abstractmethod
    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete an agent."""
        pass

    @abstractmethod
    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get agent enabled/disabled state."""
        pass

    @abstractmethod
    async def get_all_states(self) -> dict[str, bool]:
        """Get enabled/disabled state for all agents in a single query.

        Returns:
            Dict mapping agent path to enabled (True) or disabled (False).
        """
        pass

    @abstractmethod
    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set agent enabled/disabled state."""
        pass

    @abstractmethod
    async def load_all(self) -> None:
        """Load/reload all agents from storage."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of agents.

        Returns:
            Total number of agents in the repository.
        """
        pass

    @abstractmethod
    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document.

        Args:
            path: Document path/ID
            field: Field name (supports dot notation for nested fields)
            value: New value (None to unset the field)

        Returns:
            True if document was found and updated
        """
        ...

    @abstractmethod
    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> dict[str, dict]:
        """Find documents matching a MongoDB-style filter.

        Args:
            filter_dict: MongoDB-style filter
            limit: optional cap on number of documents returned. None
                returns all matches (legacy behavior).

        Returns:
            Dict mapping path -> document data for matching documents
        """
        ...

    async def find_by_identity_url(
        self,
        identity_url: str,
    ) -> dict[str, Any] | None:
        """Find an agent whose endpoint ``url`` normalizes to ``identity_url``.

        Used by ``DuplicateCheckService`` to detect exact-URL
        collisions. ``identity_url`` is the canonical form returned by
        ``normalize_identity_url(url, "a2a_agent")``.

        Default implementation scans ``list_all()``. Backends with index
        support (DocumentDB) override with a server-side filter.

        Args:
            identity_url: Pre-normalized identity URL (scheme-collapsed).

        Returns:
            A dict with the agent document fields plus ``path``, or None.
        """
        from ..utils.url_normalize import ENTITY_TYPE_AGENT, normalize_identity_url

        if not identity_url:
            return None
        agents = await self.list_all()
        for agent in agents:
            candidate_url = getattr(agent, "url", None)
            if not candidate_url:
                continue
            if normalize_identity_url(candidate_url, ENTITY_TYPE_AGENT) == identity_url:
                doc = agent.model_dump(mode="json") if hasattr(agent, "model_dump") else dict(agent)
                doc["path"] = getattr(agent, "path", doc.get("path"))
                return doc
        return None


class ScopeRepositoryBase(ABC):
    """
    Abstract base class for authorization scopes data access.

    Implementations:
    - DocumentDBScopeRepository: reads mcp-scopes collection
    """

    @abstractmethod
    async def get_ui_scopes(
        self,
        group_name: str,
    ) -> dict[str, Any]:
        """
        Get UI scopes for a Keycloak group.

        Args:
            group_name: Keycloak group name (e.g., "mcp-registry-admin")

        Returns:
            Dict with "agent_actions", "service_actions", "allowed_agents",
            "allowed_servers" keys. Returns empty dict if group not found.
        """
        pass

    @abstractmethod
    async def get_group_mappings(
        self,
        keycloak_group: str,
    ) -> list[str]:
        """
        Get scope names mapped to a Keycloak group.

        Args:
            keycloak_group: Keycloak group name

        Returns:
            List of scope names. Returns empty list if group not found.
        """
        pass

    async def get_group_mappings_bulk(
        self,
        groups: list[str],
    ) -> list[str]:
        """Get the union of scope names mapped to any of the given groups.

        Batch equivalent of ``get_group_mappings``. ``map_groups_to_scopes``
        on the auth hot path resolves a user's groups to scopes on every
        authenticated request; issuing one query per group serializes a
        round-trip per group — cheap on a local Mongo, seconds on a remote
        cluster for a user with many groups (same fan-out that
        ``get_server_scopes_bulk`` was added to fix). Backends that can fetch
        all mappings in a single query (DocumentDB ``$in``) override this; the
        default loops over the per-group method so in-memory backends (file)
        stay correct with no extra cost.

        Args:
            groups: Group names/IDs to resolve.

        Returns:
            De-duplicated list of scope names, order-preserving across the
            input groups. Empty list if none map.
        """
        seen: set[str] = set()
        result: list[str] = []
        for group in groups:
            for scope in await self.get_group_mappings(group):
                if scope not in seen:
                    seen.add(scope)
                    result.append(scope)
        return result

    @abstractmethod
    async def get_all_mapped_group_names(self) -> set[str]:
        """
        Get every IdP group name/ID that is mapped to at least one scope.

        This is the union of every scope document's ``group_mappings`` array,
        i.e. the set of groups the registry actually grants access through.
        Groups outside this set produce zero scopes, so they are safe to drop
        from a user's session at login.

        Returns:
            Set of group names/IDs. Returns an empty set if no scope has any
            group mappings.
        """
        pass

    @abstractmethod
    async def get_server_scopes(
        self,
        scope_name: str,
    ) -> list[dict[str, Any]]:
        """
        Get server access rules for a scope.

        Args:
            scope_name: Scope name (e.g., "mcp-servers-unrestricted/read")

        Returns:
            List of dicts with "server", "methods", "tools" keys.
            Returns empty list if scope not found.
        """
        pass

    async def get_server_scopes_bulk(
        self,
        scope_names: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Get server access rules for many scopes in one call.

        Batch equivalent of ``get_server_scopes``. On the cookie-auth hot
        path a user with many groups resolves to many scopes, and issuing
        one query per scope serializes a round-trip per scope — cheap on a
        local Mongo, seconds on a remote Atlas cluster (Issue: scope
        fan-out latency). Backends that can fetch all scopes in a single
        query (DocumentDB ``$in``) override this; the default loops over
        the per-scope method so in-memory backends (file) stay correct
        with no extra cost.

        Args:
            scope_names: Scope names to fetch.

        Returns:
            Dict mapping scope name to its access-rule list. Scopes with no
            rules are omitted (mirrors the empty-list return of the single
            getter, while keeping the map compact).
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for scope_name in scope_names:
            rules = await self.get_server_scopes(scope_name)
            if rules:
                result[scope_name] = rules
        return result

    async def get_ui_scopes_bulk(
        self,
        group_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get UI scopes for many groups/scopes in one call.

        Batch equivalent of ``get_ui_scopes`` — see ``get_server_scopes_bulk``
        for the latency rationale. Backends with a single-query path override
        this; the default loops over the per-scope getter.

        Args:
            group_names: Group/scope names to fetch.

        Returns:
            Dict mapping group/scope name to its UI-permissions dict. Names
            with no UI permissions are omitted.
        """
        result: dict[str, dict[str, Any]] = {}
        for group_name in group_names:
            scopes = await self.get_ui_scopes(group_name)
            if scopes:
                result[group_name] = scopes
        return result

    @abstractmethod
    async def load_all(self) -> None:
        """
        Load/reload all scopes from storage.
        Called once at application startup.
        """
        pass

    @abstractmethod
    async def add_server_scope(
        self,
        server_path: str,
        scope_name: str,
        methods: list[str],
        tools: list[str] | None = None,
    ) -> bool:
        """
        Add scope for a server.

        Args:
            server_path: Server path (e.g., "/currenttime")
            scope_name: Scope name
            methods: Allowed methods
            tools: Allowed tools (None = all tools)

        Returns:
            True if added successfully
        """
        pass

    @abstractmethod
    async def remove_server_scope(
        self,
        server_path: str,
        scope_name: str,
    ) -> bool:
        """
        Remove scope for a server.

        Args:
            server_path: Server path
            scope_name: Scope name to remove

        Returns:
            True if removed successfully
        """
        pass

    @abstractmethod
    async def create_group(
        self,
        group_name: str,
        description: str = "",
    ) -> bool:
        """
        Create a new group in scopes.

        Args:
            group_name: Name of the group to create
            description: Optional description for the group

        Returns:
            True if created successfully

        Note:
            This creates entries in both UI-Scopes and group_mappings sections.
        """
        pass

    @abstractmethod
    async def delete_group(
        self,
        group_name: str,
        remove_from_mappings: bool = True,
    ) -> bool:
        """
        Delete a group from scopes.

        Args:
            group_name: Name of the group to delete
            remove_from_mappings: If True, also remove from group_mappings

        Returns:
            True if deleted successfully

        Note:
            This removes the group's scope section and optionally its mappings.
        """
        pass

    @abstractmethod
    async def get_group(self, group_name: str) -> dict[str, Any]:
        """
        Get full details of a specific group.

        Args:
            group_name: Name of the group

        Returns:
            Dictionary with complete group information including:
            - scope_name: Name of the scope/group
            - scope_type: Type of scope (e.g., "server_scope")
            - description: Description of the group
            - server_access: List of server access definitions
            - group_mappings: List of group mappings
            - ui_permissions: UI permissions configuration
            - created_at: Creation timestamp
            - updated_at: Last update timestamp

        Returns None if group not found.
        """
        pass

    async def list_groups(self) -> dict[str, Any]:
        """
        List all groups with server counts.

        Returns:
            Dictionary mapping group names to their metadata including:
            - server_count: Number of servers in the group
            - ui_scopes: UI permission configuration
            - mappings: List of scope names mapped to this group

        Example:
            {
                "mcp-registry-admin": {
                    "server_count": 5,
                    "ui_scopes": {"list_agents": ["all"]},
                    "mappings": ["mcp-registry-admin", "mcp-servers-unrestricted/read"]
                }
            }
        """
        return {}

    @abstractmethod
    async def group_exists(
        self,
        group_name: str,
    ) -> bool:
        """
        Check if a group exists.

        Args:
            group_name: Name of the group to check

        Returns:
            True if group exists, False otherwise
        """
        pass

    @abstractmethod
    async def add_server_to_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """
        Add server to group's UI scopes list_service.

        Args:
            group_name: Name of the group
            server_name: Name of the server to add

        Returns:
            True if added successfully

        Note:
            This updates the UI-Scopes section to allow the server
            to appear in the UI for users in this group.
        """
        pass

    @abstractmethod
    async def remove_server_from_ui_scopes(
        self,
        group_name: str,
        server_name: str,
    ) -> bool:
        """
        Remove server from group's UI scopes list_service.

        Args:
            group_name: Name of the group
            server_name: Name of the server to remove

        Returns:
            True if removed successfully

        Note:
            This updates the UI-Scopes section to hide the server
            from the UI for users in this group.
        """
        pass

    @abstractmethod
    async def add_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """
        Add a scope to group mappings.

        Args:
            group_name: Name of the group
            scope_name: Name of the scope to map to the group

        Returns:
            True if added successfully

        Note:
            This updates the group_mappings section to associate
            a scope with a Keycloak group.
        """
        pass

    @abstractmethod
    async def remove_group_mapping(
        self,
        group_name: str,
        scope_name: str,
    ) -> bool:
        """
        Remove a scope from group mappings.

        Args:
            group_name: Name of the group
            scope_name: Name of the scope to remove from the group

        Returns:
            True if removed successfully

        Note:
            This updates the group_mappings section to disassociate
            a scope from a Keycloak group.
        """
        pass

    @abstractmethod
    async def get_all_group_mappings(self) -> dict[str, list[str]]:
        """
        Get all group mappings.

        Returns:
            Dictionary mapping scope names to lists of IdP/Keycloak groups
            that grant each scope.  Both the DocumentDB and file backends
            return this canonical shape so ``map_cognito_groups_to_scopes``
            can invert it uniformly.

        Example:
            {
                "mcp-registry-admin": [
                    "mcp-registry-admin"
                ],
                "mcp-servers-unrestricted/read": [
                    "mcp-registry-admin",
                    "registry-admins"
                ]
            }
        """
        pass

    @abstractmethod
    async def list_scope_names(self) -> list[str]:
        """
        List all known scope names (scope document ids).

        Used by the startup legacy-scope audit (Issue #1026) to enumerate
        scopes without loading their full bodies. Excludes well-known
        non-scope sections like 'UI-Scopes' and 'group_mappings' in the
        file backend; for DocumentDB this corresponds to the set of
        `_id` values in the mcp-scopes collection.

        Returns:
            List of scope names. Returns an empty list on repository error.
        """
        pass

    @abstractmethod
    async def add_server_to_multiple_scopes(
        self,
        server_path: str,
        scope_names: list[str],
        methods: list[str],
        tools: list[str],
    ) -> bool:
        """
        Add server to multiple scopes at once.

        Args:
            server_path: Server path (e.g., "/currenttime")
            scope_names: List of scope names to add the server to
            methods: Allowed methods for all scopes
            tools: Allowed tools for all scopes

        Returns:
            True if added successfully to all scopes

        Note:
            This is a bulk operation that atomically adds a server
            to multiple scope groups with the same permissions.
        """
        pass

    @abstractmethod
    async def remove_server_from_all_scopes(
        self,
        server_path: str,
    ) -> bool:
        """
        Remove server from all scopes.

        Args:
            server_path: Server path to remove

        Returns:
            True if removed successfully from all scopes

        Note:
            This is used during server deletion to clean up all
            scope references to the server.
        """
        pass


class SecurityScanRepositoryBase(ABC):
    """
    Abstract base class for security scan results data access.

    Implementations:
    - FileSecurityScanRepository: reads ~/mcp-gateway/security_scans/*.json
    - DocumentDBSecurityScanRepository: reads mcp-security-scans collection
    """

    @abstractmethod
    async def get(
        self,
        server_path: str,
    ) -> dict[str, Any] | None:
        """
        Get latest security scan result for a server.

        Args:
            server_path: Server path (e.g., "/currenttime")

        Returns:
            Security scan result dict if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_all(self) -> list[dict[str, Any]]:
        """
        List all security scan results.

        Returns:
            List of all security scan result dicts.
        """
        pass

    @abstractmethod
    async def list_latest(self) -> list[dict[str, Any]]:
        """
        List the latest security scan result per server path.

        Unlike list_all(), this collapses scan history (the DocumentDB backend
        stores one document per scan run) to a single most-recent document per
        path. List endpoints use this to attach a per-item scan summary without
        loading the entire scan history.

        Returns:
            List with at most one (the newest) scan result per server path.
        """
        pass

    @abstractmethod
    async def create(
        self,
        scan_result: dict[str, Any],
    ) -> bool:
        """
        Create/update a security scan result.

        Args:
            scan_result: Security scan result dict. Must contain "server_path".

        Returns:
            True if created successfully.
        """
        pass

    @abstractmethod
    async def get_latest(
        self,
        server_path: str,
    ) -> dict[str, Any] | None:
        """
        Get latest scan result for a server.

        Args:
            server_path: Server path

        Returns:
            Latest scan result if found, None otherwise.
        """
        pass

    @abstractmethod
    async def query_by_status(
        self,
        status: str,
    ) -> list[dict[str, Any]]:
        """
        Query scan results by status.

        Args:
            status: Scan status (e.g., "passed", "failed", "error")

        Returns:
            List of scan results with the given status.
        """
        pass

    @abstractmethod
    async def load_all(self) -> None:
        """
        Load/reload all security scan results from storage.
        Called once at application startup.
        """
        pass


class SkillSecurityScanRepositoryBase(ABC):
    """
    Abstract base class for skill security scan results data access.

    Implementations:
    - FileSkillSecurityScanRepository: reads ~/mcp-gateway/skill_security_scans/*.json
    - DocumentDBSkillSecurityScanRepository: reads mcp-skill-security-scans collection
    """

    @abstractmethod
    async def get(
        self,
        skill_path: str,
    ) -> dict[str, Any] | None:
        """
        Get latest security scan result for a skill.

        Args:
            skill_path: Skill path (e.g., "/skills/pdf-processing")

        Returns:
            Security scan result dict if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_all(self) -> list[dict[str, Any]]:
        """
        List all skill security scan results.

        Returns:
            List of all skill security scan result dicts.
        """
        pass

    @abstractmethod
    async def list_latest(self) -> list[dict[str, Any]]:
        """
        List the latest skill security scan result per skill path.

        Unlike list_all(), this collapses scan history (the DocumentDB backend
        stores one document per scan run) to a single most-recent document per
        path. List endpoints use this to attach a per-item scan summary without
        loading the entire scan history.

        Returns:
            List with at most one (the newest) scan result per skill path.
        """
        pass

    @abstractmethod
    async def create(
        self,
        scan_result: dict[str, Any],
    ) -> bool:
        """
        Create/update a skill security scan result.

        Args:
            scan_result: Skill security scan result dict. Must contain "skill_path".

        Returns:
            True if created successfully.
        """
        pass

    @abstractmethod
    async def get_latest(
        self,
        skill_path: str,
    ) -> dict[str, Any] | None:
        """
        Get latest scan result for a skill.

        Args:
            skill_path: Skill path

        Returns:
            Latest scan result if found, None otherwise.
        """
        pass

    @abstractmethod
    async def query_by_status(
        self,
        status: str,
    ) -> list[dict[str, Any]]:
        """
        Query scan results by status.

        Args:
            status: Scan status (e.g., "completed", "failed", "pending")

        Returns:
            List of scan results with the given status.
        """
        pass

    @abstractmethod
    async def load_all(self) -> None:
        """
        Load/reload all skill security scan results from storage.
        Called once at application startup.
        """
        pass


class SearchRepositoryBase(ABC):
    """Abstract base class for semantic/hybrid search using DocumentDB."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the search service."""
        pass

    @abstractmethod
    async def index_server(
        self,
        path: str,
        server_info: dict[str, Any],
        is_enabled: bool = False,
    ) -> None:
        """Index a server for search."""
        pass

    @abstractmethod
    async def index_agent(
        self,
        path: str,
        agent_card: AgentCard,
        is_enabled: bool = False,
    ) -> None:
        """Index an agent for search."""
        pass

    @abstractmethod
    async def remove_entity(
        self,
        path: str,
    ) -> bool:
        """Remove entity from search index.

        Returns:
            True if removal succeeded (including idempotent not-found).
            False if removal failed.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perform search."""
        pass

    async def index_skill(
        self,
        path: str,
        skill: Any,
        is_enabled: bool = False,
    ) -> None:
        """Index a skill for search.

        Default implementation is a no-op. Override in implementations
        that support skill indexing.

        Args:
            path: Skill path (e.g., /skills/pdf-processing)
            skill: SkillCard object
            is_enabled: Whether skill is enabled
        """
        return None

    async def index_virtual_server(
        self,
        path: str,
        virtual_server: Any,
        is_enabled: bool = False,
    ) -> None:
        """Index a virtual server for search.

        Default implementation is a no-op. Override in implementations
        that support virtual server indexing.

        Args:
            path: Virtual server path (e.g., /virtual/dev-essentials)
            virtual_server: VirtualServerConfig object
            is_enabled: Whether virtual server is enabled
        """
        return None

    async def index_custom_entity(
        self,
        record: Any,
        descriptor: Any,
    ) -> None:
        """Index a custom entity record for search.

        Default implementation is a no-op. Override in implementations
        that support custom entity indexing.

        Args:
            record: CustomEntityRecord to index.
            descriptor: CustomTypeDescriptor describing the record's type.
        """
        return None

    async def delete_custom_entity_index(
        self,
        path: str,
    ) -> None:
        """Remove a single custom entity record's embedding.

        Default implementation is a no-op. Override in implementations
        that support custom entity indexing.

        Args:
            path: Record path (e.g., /workflow/<uuid>).
        """
        return None

    async def delete_custom_entity_index_by_type(
        self,
        entity_type: str,
    ) -> int:
        """Bulk-remove all embeddings for a custom type.

        Default implementation is a no-op returning 0. Override in
        implementations that support custom entity indexing.

        Args:
            entity_type: The custom type name to purge.

        Returns:
            Number of embeddings removed.
        """
        return 0

    async def search_by_tags(
        self,
        tags: list[str],
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search entities by exact tag match (case-insensitive).

        Returns entities that have ALL specified tags. Default implementation
        falls back to search() with the tag names as query text.

        Args:
            tags: Required tags (all must match)
            entity_types: Optional entity type filter
            max_results: Max results per entity type

        Returns:
            Grouped search results dict
        """
        return await self.search(
            query=" ".join(tags),
            entity_types=entity_types,
            max_results=max_results,
        )

    async def get_all_tags(self) -> list[str]:
        """Return a sorted list of all unique tags across all indexed entities.

        Default implementation returns an empty list. Override in implementations
        that support tag retrieval.

        Returns:
            Sorted list of unique tag strings
        """
        return []


class PeerFederationRepositoryBase(ABC):
    """Abstract base class for peer federation storage."""

    @abstractmethod
    async def load_all(self) -> None:
        """Load/reload all peers and sync states from storage."""
        pass

    @abstractmethod
    async def get_peer(
        self,
        peer_id: str,
    ) -> Any | None:
        """Get peer configuration by ID."""
        pass

    @abstractmethod
    async def list_peers(
        self,
        enabled: bool | None = None,
    ) -> list[Any]:
        """List all peer configurations with optional filtering."""
        pass

    @abstractmethod
    async def create_peer(
        self,
        config: Any,
    ) -> Any:
        """Create a new peer configuration."""
        pass

    @abstractmethod
    async def update_peer(
        self,
        peer_id: str,
        updates: dict[str, Any],
    ) -> Any:
        """Update an existing peer configuration."""
        pass

    @abstractmethod
    async def delete_peer(
        self,
        peer_id: str,
    ) -> bool:
        """Delete a peer configuration and its sync status."""
        pass

    @abstractmethod
    async def get_sync_status(
        self,
        peer_id: str,
    ) -> Any | None:
        """Get sync status for a peer."""
        pass

    @abstractmethod
    async def update_sync_status(
        self,
        peer_id: str,
        status: Any,
    ) -> Any:
        """Update sync status for a peer."""
        pass

    @abstractmethod
    async def list_sync_statuses(self) -> list[Any]:
        """List all peer sync statuses."""
        pass


class FederationConfigRepositoryBase(ABC):
    """Abstract base class for federation configuration storage."""

    @abstractmethod
    async def get_config(self, config_id: str = "default") -> FederationConfig | None:
        """
        Get federation configuration by ID.

        Args:
            config_id: Configuration ID (default: "default")

        Returns:
            FederationConfig if found, None otherwise
        """
        pass

    @abstractmethod
    async def save_config(
        self, config: FederationConfig, config_id: str = "default"
    ) -> FederationConfig:
        """
        Save or update federation configuration.

        Args:
            config: Federation configuration to save
            config_id: Configuration ID (default: "default")

        Returns:
            Saved configuration with timestamps
        """
        pass

    @abstractmethod
    async def delete_config(self, config_id: str = "default") -> bool:
        """
        Delete federation configuration.

        Args:
            config_id: Configuration ID (default: "default")

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_configs(self) -> list[dict[str, Any]]:
        """
        List all federation configurations.

        Returns:
            List of config summaries with id, created_at, updated_at
        """
        pass


class SkillRepositoryBase(ABC):
    """Abstract base class for skill repository implementations."""

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes if not present."""
        pass

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> SkillCard | None:
        """Get a skill by path."""
        pass

    @abstractmethod
    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[SkillCard]:
        """List all skills with pagination.

        Args:
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return

        Returns:
            List of SkillCard objects
        """
        pass

    @abstractmethod
    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[SkillCard]:
        """List skills with DB-level pagination and deterministic ordering.

        Args:
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.

        Returns:
            List of SkillCard objects for the requested page.
        """
        pass

    @abstractmethod
    async def list_filtered(
        self,
        include_disabled: bool = False,
        tag: str | None = None,
        visibility: str | None = None,
        registry_name: str | None = None,
    ) -> list[SkillCard]:
        """List skills with database-level filtering."""
        pass

    @abstractmethod
    async def create(
        self,
        skill: SkillCard,
    ) -> SkillCard:
        """Create a new skill."""
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> SkillCard | None:
        """Update a skill."""
        pass

    @abstractmethod
    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a skill."""
        pass

    @abstractmethod
    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get skill enabled state."""
        pass

    @abstractmethod
    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set skill enabled state."""
        pass

    # Batch operations for federation sync
    @abstractmethod
    async def create_many(
        self,
        skills: list[SkillCard],
    ) -> list[SkillCard]:
        """Create multiple skills in single operation."""
        pass

    @abstractmethod
    async def update_many(
        self,
        updates: dict[str, dict[str, Any]],
    ) -> int:
        """Update multiple skills by path, return count."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of skills.

        Returns:
            Total number of skills in the repository.
        """
        pass

    async def find_by_identity_url(
        self,
        identity_url: str,
    ) -> dict[str, Any] | None:
        """Find a skill whose ``skill_md_url`` normalizes to ``identity_url``.

        Used by ``DuplicateCheckService`` to detect exact-URL
        collisions. ``identity_url`` is the canonical form returned by
        ``normalize_identity_url(url, "skill")``.

        Default implementation scans ``list_all()``. DocumentDB
        implementation overrides with a server-side filter when an
        identity index is present.

        Args:
            identity_url: Pre-normalized identity URL (canonical github form).

        Returns:
            A dict with the skill document fields plus ``path``, or None.
        """
        from ..utils.url_normalize import ENTITY_TYPE_SKILL, normalize_identity_url

        if not identity_url:
            return None
        skills = await self.list_all()
        for skill in skills:
            candidate_url = str(getattr(skill, "skill_md_url", "") or "")
            if not candidate_url:
                continue
            if normalize_identity_url(candidate_url, ENTITY_TYPE_SKILL) == identity_url:
                doc = skill.model_dump(mode="json") if hasattr(skill, "model_dump") else dict(skill)
                doc["path"] = getattr(skill, "path", doc.get("path"))
                return doc
        return None


class BackendSessionRepositoryBase(ABC):
    """Abstract base class for backend MCP session storage.

    Manages per-client backend sessions for virtual MCP servers.
    Each session maps a (client_session_id, backend_key) pair to the
    backend MCP server's session ID, enabling session isolation and
    persistence across L1 cache misses.
    """

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes (TTL on last_used_at, etc.)."""
        pass

    @abstractmethod
    async def get_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        user_id: str | None = None,
    ) -> str | None:
        """Get backend session ID and bump last_used_at atomically.

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            user_id: Authenticated user identity the session must belong to.
                When None, ownership is not enforced (legacy behavior).

        Returns:
            Backend session ID if found (and owned by ``user_id`` when given),
            None otherwise
        """
        pass

    @abstractmethod
    async def store_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        backend_session_id: str,
        user_id: str,
        virtual_server_path: str,
    ) -> None:
        """Store or update a backend session (upsert).

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            backend_session_id: Session ID from the backend MCP server
            user_id: User identity for audit
            virtual_server_path: Virtual server path
        """
        pass

    @abstractmethod
    async def delete_backend_session(
        self,
        client_session_id: str,
        backend_key: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a stale backend session.

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
            user_id: Authenticated user identity the session must belong to.
                When None, ownership is not enforced (legacy behavior).
        """
        pass

    @abstractmethod
    async def create_client_session(
        self,
        client_session_id: str,
        user_id: str,
        virtual_server_path: str,
    ) -> None:
        """Create a client session document for validation.

        Args:
            client_session_id: Generated client session ID
            user_id: User identity from auth context
            virtual_server_path: Virtual server path
        """
        pass

    @abstractmethod
    async def validate_client_session(
        self,
        client_session_id: str,
        user_id: str | None = None,
        virtual_server_path: str | None = None,
    ) -> bool:
        """Check if a client session exists, is owned by ``user_id``, and bump last_used_at.

        Args:
            client_session_id: Client-facing session ID
            user_id: Authenticated user identity the session must belong to.
                When None, ownership is not enforced (legacy behavior).
            virtual_server_path: Virtual server path the session was minted for.
                When None, the path is not enforced (legacy behavior).

        Returns:
            True if session exists (and matches ``user_id`` /
            ``virtual_server_path`` when given), False otherwise
        """
        pass


class VirtualServerRepositoryBase(ABC):
    """Abstract base class for virtual MCP server data access."""

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes if not present."""
        pass

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> VirtualServerConfig | None:
        """Get a virtual server by path.

        Args:
            path: Virtual server path (e.g., '/virtual/dev-essentials')

        Returns:
            VirtualServerConfig if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_all(self) -> list[VirtualServerConfig]:
        """List all virtual servers.

        Returns:
            List of all VirtualServerConfig objects
        """
        pass

    @abstractmethod
    async def list_enabled(self) -> list[VirtualServerConfig]:
        """List all enabled virtual servers.

        Returns:
            List of enabled VirtualServerConfig objects
        """
        pass

    @abstractmethod
    async def create(
        self,
        config: VirtualServerConfig,
    ) -> VirtualServerConfig:
        """Create a new virtual server.

        Args:
            config: Virtual server configuration

        Returns:
            Created VirtualServerConfig

        Raises:
            VirtualServerAlreadyExistsError: If path already exists
        """
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> VirtualServerConfig | None:
        """Update a virtual server.

        Args:
            path: Virtual server path
            updates: Fields to update

        Returns:
            Updated VirtualServerConfig if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a virtual server.

        Args:
            path: Virtual server path

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get virtual server enabled/disabled state.

        Args:
            path: Virtual server path

        Returns:
            True if enabled, False if disabled or not found
        """
        pass

    @abstractmethod
    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set virtual server enabled/disabled state.

        Args:
            path: Virtual server path
            enabled: New enabled state

        Returns:
            True if updated, False if not found
        """
        pass


class RegistryCardRepositoryBase(ABC):
    """Interface for Registry Card persistence."""

    @abstractmethod
    async def get(self) -> RegistryCard | None:
        """Retrieve the Registry Card."""
        pass

    @abstractmethod
    async def save(
        self,
        card: RegistryCard,
    ) -> RegistryCard:
        """Save or update the Registry Card."""
        pass

    @abstractmethod
    async def exists(self) -> bool:
        """Check if Registry Card exists."""
        pass


class CustomTypeRepositoryBase(ABC):
    """Abstract base class for custom entity type descriptor storage."""

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes if not present."""
        pass

    @abstractmethod
    async def create(
        self,
        descriptor: "CustomTypeDescriptor",
    ) -> "CustomTypeDescriptor":
        """Create a new type descriptor.

        Raises:
            CustomTypeAlreadyExistsError: If a type with this name exists.
        """
        pass

    @abstractmethod
    async def get(
        self,
        name: str,
    ) -> "CustomTypeDescriptor | None":
        """Get a type descriptor by name."""
        pass

    @abstractmethod
    async def list_all(self) -> "list[CustomTypeDescriptor]":
        """List all type descriptors, sorted by name."""
        pass

    @abstractmethod
    async def delete(
        self,
        name: str,
    ) -> bool:
        """Delete a type descriptor. Returns True if a document was removed."""
        pass

    @abstractmethod
    async def update_metadata(
        self,
        name: str,
        updates: dict[str, Any],
    ) -> "CustomTypeDescriptor | None":
        """Update mutable metadata (display_name/description) of a type.

        Only the keys present in ``updates`` are written; the immutable
        ``name``/``fields`` are never touched. Returns the updated descriptor,
        or None if no type with this name exists.
        """
        pass


class CustomEntityRepositoryBase(ABC):
    """Abstract base class for custom entity record storage (single collection)."""

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes if not present."""
        pass

    @abstractmethod
    async def create(
        self,
        record: "CustomEntityRecord",
    ) -> "CustomEntityRecord":
        """Create a new record (``_id`` = record.path)."""
        pass

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> "CustomEntityRecord | None":
        """Get a record by its synthetic path."""
        pass

    @abstractmethod
    async def list_paginated(
        self,
        entity_type: str,
        skip: int = 0,
        limit: int = 100,
        visibility_filter: dict[str, Any] | None = None,
    ) -> "list[CustomEntityRecord]":
        """List records of a type with DB-level pagination and optional filter.

        Args:
            entity_type: The custom type discriminator.
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.
            visibility_filter: Optional Mongo predicate composed with the
                type discriminator to restrict results to records the caller
                may see. None = no restriction (admin).
        """
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> "CustomEntityRecord | None":
        """Update a record via ``$set`` of the given fields."""
        pass

    @abstractmethod
    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete a single record. Returns True if a document was removed."""
        pass

    @abstractmethod
    async def delete_by_type(
        self,
        entity_type: str,
    ) -> int:
        """Bulk-delete all records of a type. Returns the deleted count."""
        pass

    @abstractmethod
    async def count(
        self,
        entity_type: str,
        visibility_filter: dict[str, Any] | None = None,
    ) -> int:
        """Count records of a type, applying the SAME optional filter as list."""
        pass
