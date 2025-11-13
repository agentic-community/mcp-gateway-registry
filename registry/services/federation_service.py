"""
Federation service for managing federated registry integrations.

Handles:
- Loading federation configuration
- Syncing servers from federated registries
- Caching federated server data with TTL
- Periodic sync scheduling
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.federation_schema import (
    AnthropicFederationConfig,
    FederatedServer,
    FederationConfig,
)
from .federation.anthropic_client import AnthropicFederationClient
from .federation.asor_client import AsorFederationClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class FederationService:
    """Service for managing federated registry integrations."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize federation service.

        Args:
            config_path: Path to federation.json config file
            cache_dir: Directory for caching federated server data
        """
        # Set default paths
        if config_path is None:
            config_path = os.getenv(
                "FEDERATION_CONFIG_PATH",
                "/app/config/federation.json"
            )
        if cache_dir is None:
            cache_dir = os.getenv(
                "FEDERATION_CACHE_DIR",
                "/app/.cache/federation"
            )

        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Initialize clients
        self.anthropic_client: Optional[AnthropicFederationClient] = None
        if self.config.anthropic.enabled:
            self.anthropic_client = AnthropicFederationClient(
                endpoint=self.config.anthropic.endpoint
            )

        self.asor_client: Optional[AsorFederationClient] = None
        if self.config.asor.enabled:
            # Extract tenant URL from endpoint or use default
            tenant_url = self.config.asor.endpoint.split("/api")[0] if "/api" in self.config.asor.endpoint else self.config.asor.endpoint

            self.asor_client = AsorFederationClient(
                endpoint=self.config.asor.endpoint,
                auth_env_var=self.config.asor.auth_env_var,
                tenant_url=tenant_url
            )

        # Cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        logger.info(f"Federation service initialized with config: {config_path}")
        if self.config.is_any_federation_enabled():
            logger.info(f"Enabled federations: {', '.join(self.config.get_enabled_federations())}")
        else:
            logger.info("No federations enabled")

    def _load_config(self) -> FederationConfig:
        """
        Load federation configuration from JSON file.

        Returns:
            FederationConfig instance
        """
        config_file = Path(self.config_path)

        if not config_file.exists():
            logger.warning(f"Federation config not found at {self.config_path}, using defaults")
            return FederationConfig()

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Remove JSON comments if present
            config_data.pop("_comment", None)
            config_data.pop("_description", None)

            config = FederationConfig(**config_data)
            logger.info(f"Loaded federation config from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load federation config: {e}")
            return FederationConfig()

    def sync_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sync servers from all enabled federated registries.

        Returns:
            Dictionary mapping source name to list of synced servers
        """
        results = {}

        if self.config.anthropic.enabled:
            logger.info("Syncing servers from Anthropic MCP Registry...")
            anthropic_servers = self._sync_anthropic()
            results["anthropic"] = anthropic_servers
            logger.info(f"Synced {len(anthropic_servers)} servers from Anthropic")

        # Sync ASOR agents
        logger.info("Syncing agents from ASOR...")
        asor_agents = self._sync_asor()
        results["asor"] = asor_agents
        logger.info(f"Synced {len(asor_agents)} agents from ASOR")

        return results

    def _sync_anthropic(self) -> List[Dict[str, Any]]:
        """
        Sync servers from Anthropic MCP Registry.

        Returns:
            List of synced server data
        """
        if not self.anthropic_client:
            logger.error("Anthropic client not initialized")
            return []

        # Fetch servers
        servers = self.anthropic_client.fetch_all_servers(
            self.config.anthropic.servers
        )

        # Save servers as files to external mount
        from pathlib import Path
        import json
        from ..core.config import settings
        
        for server_data in servers:
            try:
                # Create filename from server name
                server_name = server_data.get("server_name", "unknown-server")
                filename = server_name.replace("/", "-").replace(".", "-") + ".json"
                file_path = settings.servers_dir / filename
                
                # Save to file
                with open(file_path, "w") as f:
                    json.dump(server_data, f, indent=2)
                
                logger.info(f"Saved Anthropic server file: {server_name} -> {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save Anthropic server {server_data.get('server_name', 'unknown')}: {e}")

        # Cache the results
        for server in servers:
            server_name = server.get("server_name")
            if server_name:
                cache_key = f"anthropic:{server_name}"
                self._cache[cache_key] = server
                self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        # Persist cache to disk
        self._save_cache_to_disk("anthropic", servers)

        return servers

    def _sync_asor(self) -> List[Dict[str, Any]]:
        """
        Sync agents from Workday ASOR.

        Returns:
            List of synced agent data
        """
        if not self.asor_client:
            logger.error("ASOR client not initialized")
            return []

        # Fetch agents
        agents = self.asor_client.fetch_all_agents(
            self.config.asor.agents
        )

        # Register agents with the agent service
        from ..services.agent_service import agent_service
        from ..schemas.agent_models import AgentCard
        from datetime import datetime, timezone
        
        for agent_data in agents:
            # Extract agent info from ASOR data structure
            agent_name = agent_data.get("name", "Unknown ASOR Agent")
            agent_path = f"/{agent_name.lower().replace('_', '-')}"
            agent_url = agent_data.get("url", "")
            agent_description = agent_data.get("description", "Agent synced from ASOR")
            if agent_description == "None":
                agent_description = f"ASOR agent: {agent_name}"
            
            # Extract skills
            skills_data = agent_data.get("skills", [])
            skills = []
            for skill in skills_data:
                skills.append({
                    "name": skill.get("name", ""),
                    "description": skill.get("description", ""),
                    "id": skill.get("id", "")
                })
            
            # Convert ASOR agent data to AgentCard format
            agent_card = AgentCard(
                protocol_version="1.0",  # Required A2A field
                name=agent_name,
                path=agent_path,
                url=agent_url,
                description=agent_description,
                version=agent_data.get("version", "1.0.0"),
                provider="ASOR",  # Add provider field
                author="ASOR",
                license="Unknown",
                skills=skills,
                tags=["asor", "federated", "workday"],
                visibility="public",
                registered_by="asor-federation",
                registered_at=datetime.now(timezone.utc)
            )
            
            try:
                # Check if agent already exists
                if agent_path in agent_service.registered_agents:
                    logger.debug(f"ASOR agent {agent_path} already exists, skipping registration")
                    continue
                
                # Register the agent using the proper method
                agent_service.register_agent(agent_card)
                logger.info(f"Registered ASOR agent: {agent_card.name} at {agent_card.path}")
                
            except Exception as e:
                logger.error(f"Failed to register ASOR agent {agent_data.get('name', 'unknown')}: {e}")

        # Cache the results
        for agent in agents:
            agent_id = agent.get("name")  # Use 'name' instead of 'server_name'
            if agent_id:
                cache_key = f"asor:{agent_id}"
                self._cache[cache_key] = agent
                self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        # Persist cache to disk
        self._save_cache_to_disk("asor", agents)

        return agents

    def get_federated_servers(
        self,
        source: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get federated servers from cache or fetch if needed.

        Args:
            source: Filter by source (anthropic, asor, etc.) or None for all
            force_refresh: Force refresh from source even if cache is valid

        Returns:
            List of federated server data
        """
        servers = []

        if source is None or source == "anthropic":
            servers.extend(self._get_anthropic_servers(force_refresh))

        if source is None or source == "asor":
            servers.extend(self._get_asor_agents(force_refresh))

        return servers

    def _get_anthropic_servers(
        self,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get Anthropic servers from cache or fetch if needed.

        Args:
            force_refresh: Force refresh from source

        Returns:
            List of Anthropic server data
        """
        if not self.config.anthropic.enabled:
            return []

        # Check if cache needs refresh
        needs_refresh = force_refresh or self._is_cache_expired("anthropic")

        if needs_refresh:
            logger.info("Cache expired or force refresh, syncing from Anthropic...")
            return self._sync_anthropic()

        # Return from memory cache
        cached_servers = [
            server for key, server in self._cache.items()
            if key.startswith("anthropic:")
        ]

        if cached_servers:
            logger.debug(f"Returning {len(cached_servers)} servers from cache")
            return cached_servers

        # Try loading from disk cache
        disk_cache = self._load_cache_from_disk("anthropic")
        if disk_cache:
            logger.info(f"Loaded {len(disk_cache)} servers from disk cache")
            # Update memory cache
            for server in disk_cache:
                server_name = server.get("server_name")
                if server_name:
                    cache_key = f"anthropic:{server_name}"
                    self._cache[cache_key] = server
                    # Use cached_at timestamp if available
                    cached_at = server.get("cached_at")
                    if cached_at:
                        self._cache_timestamps[cache_key] = datetime.fromisoformat(cached_at)
            return disk_cache

        # No cache available, fetch from source
        logger.info("No cache available, fetching from Anthropic...")
        return self._sync_anthropic()

    def _get_asor_agents(
        self,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get ASOR agents from cache or fetch if needed.

        Args:
            force_refresh: Force refresh from source

        Returns:
            List of ASOR agent data
        """
        if not self.config.asor.enabled:
            return []

        # Check if cache needs refresh
        needs_refresh = force_refresh or self._is_cache_expired("asor")

        if needs_refresh:
            logger.info("Cache expired or force refresh, syncing from ASOR...")
            return self._sync_asor()

        # Return from memory cache
        cached_agents = [
            agent for key, agent in self._cache.items()
            if key.startswith("asor:")
        ]

        if cached_agents:
            logger.debug(f"Returning {len(cached_agents)} agents from cache")
            return cached_agents

        # Try loading from disk cache
        disk_cache = self._load_cache_from_disk("asor")
        if disk_cache:
            logger.info(f"Loaded {len(disk_cache)} agents from disk cache")
            # Update memory cache
            for agent in disk_cache:
                agent_id = agent.get("server_name")
                if agent_id:
                    cache_key = f"asor:{agent_id}"
                    self._cache[cache_key] = agent
                    # Use cached_at timestamp if available
                    cached_at = agent.get("cached_at")
                    if cached_at:
                        self._cache_timestamps[cache_key] = datetime.fromisoformat(cached_at)
            return disk_cache

        # No cache available, fetch from source
        logger.info("No cache available, fetching from ASOR...")
        return self._sync_asor()

    def _is_cache_expired(
        self,
        source: str
    ) -> bool:
        """
        Check if cache for a source is expired.

        Args:
            source: Source name (anthropic, asor, etc.)

        Returns:
            True if cache is expired or missing
        """
        # Get all cache keys for this source
        cache_keys = [key for key in self._cache_timestamps.keys() if key.startswith(f"{source}:")]

        if not cache_keys:
            return True

        # Check if any timestamp is older than TTL
        if source == "anthropic":
            ttl_seconds = self.config.anthropic.cache_ttl_seconds
        elif source == "asor":
            ttl_seconds = self.config.asor.cache_ttl_seconds
        else:
            ttl_seconds = 3600  # Default 1 hour

        now = datetime.now(timezone.utc)
        ttl = timedelta(seconds=ttl_seconds)

        for key in cache_keys:
            timestamp = self._cache_timestamps.get(key)
            if timestamp and (now - timestamp) > ttl:
                return True

        return False

    def _save_cache_to_disk(
        self,
        source: str,
        servers: List[Dict[str, Any]]
    ) -> None:
        """
        Save cache to disk for persistence across restarts.

        Args:
            source: Source name (anthropic, asor, etc.)
            servers: List of server data to cache
        """
        try:
            cache_file = self.cache_dir / f"{source}_cache.json"
            cache_data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "servers": servers
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Saved {len(servers)} servers to disk cache: {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_cache_from_disk(
        self,
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Load cache from disk.

        Args:
            source: Source name (anthropic, asor, etc.)

        Returns:
            List of cached server data or empty list
        """
        try:
            cache_file = self.cache_dir / f"{source}_cache.json"

            if not cache_file.exists():
                return []

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if cache is expired
            cached_at_str = cache_data.get("cached_at")
            if cached_at_str:
                cached_at = datetime.fromisoformat(cached_at_str)
                if source == "anthropic":
                    ttl_seconds = self.config.anthropic.cache_ttl_seconds
                elif source == "asor":
                    ttl_seconds = self.config.asor.cache_ttl_seconds
                else:
                    ttl_seconds = 3600

                age = (datetime.now(timezone.utc) - cached_at).total_seconds()
                if age > ttl_seconds:
                    logger.debug(f"Disk cache expired (age: {age}s, ttl: {ttl_seconds}s)")
                    return []

            servers = cache_data.get("servers", [])
            logger.debug(f"Loaded {len(servers)} servers from disk cache")
            return servers

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            return []

    def clear_cache(
        self,
        source: Optional[str] = None
    ) -> None:
        """
        Clear cache for specified source or all sources.

        Args:
            source: Source name or None for all sources
        """
        if source:
            # Clear specific source
            keys_to_remove = [key for key in self._cache.keys() if key.startswith(f"{source}:")]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

            # Remove disk cache
            cache_file = self.cache_dir / f"{source}_cache.json"
            if cache_file.exists():
                cache_file.unlink()

            logger.info(f"Cleared cache for source: {source}")
        else:
            # Clear all
            self._cache.clear()
            self._cache_timestamps.clear()

            # Remove all disk caches
            for cache_file in self.cache_dir.glob("*_cache.json"):
                cache_file.unlink()

            logger.info("Cleared all federation caches")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_cached_servers": len(self._cache),
            "sources": {}
        }

        for source in ["anthropic", "asor"]:
            source_keys = [key for key in self._cache.keys() if key.startswith(f"{source}:")]
            stats["sources"][source] = {
                "count": len(source_keys),
                "expired": self._is_cache_expired(source)
            }

        return stats


# Global instance
_federation_service: Optional[FederationService] = None


def get_federation_service() -> FederationService:
    """
    Get global federation service instance (singleton).

    Returns:
        FederationService instance
    """
    global _federation_service

    if _federation_service is None:
        _federation_service = FederationService()

    return _federation_service
