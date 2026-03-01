"""
Repository base classes for data access abstraction.

These abstract base classes define the contract that ALL repository implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from ..schemas.agent_models import AgentCard
from ..schemas.federation_schema import FederationConfig

# Import skill models with try/except to avoid circular import issues
try:
    from ..schemas.skill_models import SkillCard
except ImportError:
    SkillCard = None

try:
    from ..schemas.virtual_server_models import VirtualServerConfig
except ImportError:
    VirtualServerConfig = None


class ServerRepositoryBase(ABC):
    """Abstract base class for MCP server data access."""

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> Optional[Dict[str, Any]]:
        """Get server by path."""
        pass

    @abstractmethod
    async def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all servers."""
        pass

    @abstractmethod
    async def list_by_source(
        self,
        source: str,
    ) -> Dict[str, Dict[str, Any]]:
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
        server_info: Dict[str, Any],
    ) -> bool:
        """Create a new server."""
        pass

    @abstractmethod
    async def update(
        self,
        path: str,
        server_info: Dict[str, Any],
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
    async def count(self) -> int:
        """Get total count of servers.

        Returns:
            Total number of servers in the repository.
        """
        pass


class AgentRepositoryBase(ABC):
    """Abstract base class for A2A agent data access."""

    @abstractmethod
    async def get(
        self,
        path: str,
    ) -> Optional[AgentCard]:
        """Get agent by path."""
        pass

    @abstractmethod
    async def list_all(self) -> List[AgentCard]:
        """List all agents."""
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
        updates: Dict[str, Any],
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


class ScopeRepositoryBase(ABC):
    """
    Abstract base class for authorization scopes data access.

    Implementations:
    - FileScopeRepository: reads auth_server/scopes.yml
    - DocumentDBScopeRepository: reads mcp-scopes collection
    """

    @abstractmethod
    async def get_ui_scopes(
        self,
        group_name: str,
    ) -> Dict[str, Any]:
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
    ) -> List[str]:
        """
        Get scope names mapped to a Keycloak group.

        Args:
            keycloak_group: Keycloak group name

        Returns:
            List of scope names. Returns empty list if group not found.
        """
        pass

    @abstractmethod
    async def get_server_scopes(
        self,
        scope_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Get server access rules for a scope.

        Args:
            scope_name: Scope name (e.g., "mcp-servers-unrestricted/read")

        Returns:
            List of dicts with "server", "methods", "tools" keys.
            Returns empty list if scope not found.
        """
        pass

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
        methods: List[str],
        tools: Optional[List[str]] = None,
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
    async def get_group(self, group_name: str) -> Dict[str, Any]:
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

    async def list_groups(self) -> Dict[str, Any]:
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
        pass

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
    async def get_all_group_mappings(self) -> Dict[str, List[str]]:
        """
        Get all group mappings.

        Returns:
            Dictionary mapping group names to lists of scope names.

        Example:
            {
                "mcp-registry-admin": [
                    "mcp-registry-admin",
                    "mcp-servers-unrestricted/read"
                ],
                "mcp-registry-user": [
                    "mcp-servers-unrestricted/read"
                ]
            }
        """
        pass

    @abstractmethod
    async def add_server_to_multiple_scopes(
        self,
        server_path: str,
        scope_names: List[str],
        methods: List[str],
        tools: List[str],
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
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest security scan result for a server.

        Args:
            server_path: Server path (e.g., "/currenttime")

        Returns:
            Security scan result dict if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_all(self) -> List[Dict[str, Any]]:
        """
        List all security scan results.

        Returns:
            List of all security scan result dicts.
        """
        pass

    @abstractmethod
    async def create(
        self,
        scan_result: Dict[str, Any],
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
    ) -> Optional[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
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
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest security scan result for a skill.

        Args:
            skill_path: Skill path (e.g., "/skills/pdf-processing")

        Returns:
            Security scan result dict if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_all(self) -> List[Dict[str, Any]]:
        """
        List all skill security scan results.

        Returns:
            List of all skill security scan result dicts.
        """
        pass

    @abstractmethod
    async def create(
        self,
        scan_result: Dict[str, Any],
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
    ) -> Optional[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
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
    """Abstract base class for semantic/hybrid search using FAISS or DocumentDB."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the search service."""
        pass

    @abstractmethod
    async def index_server(
        self,
        path: str,
        server_info: Dict[str, Any],
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
    ) -> None:
        """Remove entity from search index."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
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
        pass

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
        pass


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
    ) -> Optional[Any]:
        """Get peer configuration by ID."""
        pass

    @abstractmethod
    async def list_peers(
        self,
        enabled: Optional[bool] = None,
    ) -> List[Any]:
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
        updates: Dict[str, Any],
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
    ) -> Optional[Any]:
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
    async def list_sync_statuses(self) -> List[Any]:
        """List all peer sync statuses."""
        pass


class FederationConfigRepositoryBase(ABC):
    """Abstract base class for federation configuration storage."""

    @abstractmethod
    async def get_config(self, config_id: str = "default") -> Optional[FederationConfig]:
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
    async def list_configs(self) -> List[Dict[str, Any]]:
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
    ) -> Optional[SkillCard]:
        """Get a skill by path."""
        pass

    @abstractmethod
    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[SkillCard]:
        """List all skills with pagination.

        Args:
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return

        Returns:
            List of SkillCard objects
        """
        pass

    @abstractmethod
    async def list_filtered(
        self,
        include_disabled: bool = False,
        tag: Optional[str] = None,
        visibility: Optional[str] = None,
        registry_name: Optional[str] = None,
    ) -> List[SkillCard]:
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
        updates: Dict[str, Any],
    ) -> Optional[SkillCard]:
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
        skills: List[SkillCard],
    ) -> List[SkillCard]:
        """Create multiple skills in single operation."""
        pass

    @abstractmethod
    async def update_many(
        self,
        updates: Dict[str, Dict[str, Any]],
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
    ) -> Optional[str]:
        """Get backend session ID and bump last_used_at atomically.

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key

        Returns:
            Backend session ID if found, None otherwise
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
    ) -> None:
        """Delete a stale backend session.

        Args:
            client_session_id: Client-facing session ID
            backend_key: Backend location key
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
    ) -> bool:
        """Check if a client session exists and bump last_used_at.

        Args:
            client_session_id: Client-facing session ID

        Returns:
            True if session exists, False otherwise
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
    ) -> Optional[VirtualServerConfig]:
        """Get a virtual server by path.

        Args:
            path: Virtual server path (e.g., '/virtual/dev-essentials')

        Returns:
            VirtualServerConfig if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_all(self) -> List[VirtualServerConfig]:
        """List all virtual servers.

        Returns:
            List of all VirtualServerConfig objects
        """
        pass

    @abstractmethod
    async def list_enabled(self) -> List[VirtualServerConfig]:
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
        updates: Dict[str, Any],
    ) -> Optional[VirtualServerConfig]:
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
