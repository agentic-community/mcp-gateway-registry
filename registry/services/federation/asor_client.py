"""
Workday ASOR (Agent Service Operating Registry) federation client.

Fetches agent configurations from Workday ASOR API and transforms them
to the gateway's internal format.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from .base_client import BaseFederationClient
from ...schemas.federation_schema import AsorAgentConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class AsorFederationClient(BaseFederationClient):
    """Client for fetching agents from Workday ASOR."""

    def __init__(
        self,
        endpoint: str,
        auth_type: str = "oauth2",
        auth_env_var: Optional[str] = None,
        tenant_url: Optional[str] = None,
        timeout_seconds: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize ASOR federation client.

        Args:
            endpoint: Base URL for ASOR API
            auth_type: Authentication type (oauth2, api-key)
            auth_env_var: Environment variable containing auth credentials
            tenant_url: Workday tenant URL (for authentication)
            timeout_seconds: HTTP request timeout
            retry_attempts: Number of retry attempts
        """
        super().__init__(endpoint, timeout_seconds, retry_attempts)
        self.auth_type = auth_type
        self.auth_env_var = auth_env_var
        self.tenant_url = tenant_url
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    def _get_access_token(self) -> Optional[str]:
        """
        Get or refresh OAuth2 access token from Workday.

        Returns:
            Access token or None if authentication fails
        """
        # Check if we have a valid cached token
        if self._access_token and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                logger.debug("Using cached access token")
                return self._access_token

        # Get credentials from environment
        if self.auth_env_var:
            credentials = os.getenv(self.auth_env_var)
            if credentials:
                # Parse credentials (format: client_id:client_secret)
                try:
                    client_id, client_secret = credentials.split(":", 1)
                except ValueError:
                    logger.error("ASOR credentials must be in format 'client_id:client_secret'")
                    return None
            else:
                logger.error(f"Environment variable {self.auth_env_var} not found")
                return None
        else:
            logger.error("No auth_env_var configured for ASOR")
            return None

        # Request token from Workday
        token_url = f"{self.tenant_url}/ccx/oauth2/token"

        logger.info(f"Requesting access token from Workday: {token_url}")

        # Workday uses standard OAuth2 client credentials flow
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            response = self.client.post(
                token_url,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)

            # Set expiry slightly before actual expiry (5 min buffer)
            self._token_expiry = datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(seconds=expires_in - 300)

            logger.info(f"Successfully obtained access token (expires in {expires_in}s)")
            return self._access_token

        except Exception as e:
            logger.error(f"Failed to obtain access token: {e}")
            return None

    def fetch_agent(
        self,
        agent_id: str,
        agent_config: Optional[AsorAgentConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single agent from ASOR.

        Args:
            agent_id: Agent ID in ASOR
            agent_config: Optional agent configuration

        Returns:
            Agent data dictionary or None if fetch fails
        """
        # Use custom endpoint if provided
        if agent_config and agent_config.endpoint:
            url = agent_config.endpoint
        else:
            # Construct endpoint from agent ID
<<<<<<< Updated upstream
            # ASOR API: GET /asor/v1/agentDefinition/{id} (singular, per OpenAPI spec)
            url = f"{self.endpoint}/agentDefinition/{agent_id}"
=======
            # ASOR API follows Workday REST pattern: /{serviceName}/{version}/{resource}/{id}
            # Example: /asor/v1/agentDefinitions/{id}
            url = f"{self.endpoint}/agentDefinitions/{agent_id}"
>>>>>>> Stashed changes

        # Get access token
        access_token = self._get_access_token()
        if not access_token:
            logger.error("Failed to authenticate with Workday")
            return None

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        # Make request
        logger.info(f"Fetching agent {agent_id} from ASOR")
        response = self._make_request(url, headers=headers)

        if not response:
            logger.error(f"Failed to fetch agent {agent_id}")
            return None

        # Transform response to internal format
        return self._transform_agent_response(response, agent_id, agent_config)

    def list_all_agents(self) -> List[Dict[str, Any]]:
        """
        List all agent definitions from ASOR.

        Returns:
            List of all agent definitions
        """
<<<<<<< Updated upstream
        # ASOR API: GET /asor/v1/agentDefinition (singular, per OpenAPI spec)
        url = f"{self.endpoint}/agentDefinition"
=======
        # ASOR API follows Workday REST pattern
        # Example: GET /asor/v1/agentDefinitions
        url = f"{self.endpoint}/agentDefinitions"
>>>>>>> Stashed changes

        # Get access token
        access_token = self._get_access_token()
        if not access_token:
            logger.error("Failed to authenticate with Workday")
            return []

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        # Make request
        logger.info("Listing all agents from ASOR")
        response = self._make_request(url, headers=headers)

        if not response:
            logger.error("Failed to list agents")
            return []

<<<<<<< Updated upstream
        # Response format per OpenAPI spec: {"data": [...], "total": N}
        if isinstance(response, dict) and "data" in response:
            agents = response.get("data", [])
            logger.info(f"Found {len(agents)} agents in ASOR (total: {response.get('total', 'unknown')})")
            return agents

        # Fallback for unexpected format
        agents = response if isinstance(response, list) else []
=======
        # Response should be a list of agent definitions
        agents = response if isinstance(response, list) else response.get("agents", [])
>>>>>>> Stashed changes
        logger.info(f"Found {len(agents)} agents in ASOR")
        return agents

    def fetch_all_agents(
        self,
        agent_configs: List[AsorAgentConfig]
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple agents from ASOR.

        Args:
            agent_configs: List of agent configurations

        Returns:
            List of agent data dictionaries
        """
        agents = []

        # If no configs provided, list all agents
        if not agent_configs:
            logger.info("No agent configs provided, listing all agents from ASOR")
            return self.list_all_agents()

        for config in agent_configs:
            if not config.enabled:
                logger.info(f"Skipping disabled agent: {config.id}")
                continue

            agent_data = self.fetch_agent(config.id, config)
            if agent_data:
                agents.append(agent_data)
            else:
                logger.warning(f"Failed to fetch agent: {config.id}")

        logger.info(f"Successfully fetched {len(agents)}/{len(agent_configs)} agents")
        return agents

    def fetch_server(
        self,
        server_name: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single server (agent) from ASOR.

        Args:
            server_name: Agent ID
            **kwargs: Additional parameters

        Returns:
            Server data dictionary
        """
        return self.fetch_agent(server_name, kwargs.get("agent_config"))

    def fetch_all_servers(
        self,
        server_names: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple servers (agents) from ASOR.

        Args:
            server_names: List of agent IDs
            **kwargs: Additional parameters

        Returns:
            List of server data dictionaries
        """
        # Convert server names to agent configs
        agent_configs = [
            AsorAgentConfig(id=name, endpoint=None, enabled=True)
            for name in server_names
        ]
        return self.fetch_all_agents(agent_configs)

    def _transform_agent_response(
        self,
        response: Dict[str, Any],
        agent_id: str,
        agent_config: Optional[AsorAgentConfig]
    ) -> Dict[str, Any]:
        """
        Transform ASOR API response to internal gateway format.

        Args:
            response: Raw response from ASOR API
            agent_id: Agent ID
            agent_config: Optional agent configuration

        Returns:
            Transformed agent data
        """
<<<<<<< Updated upstream
        # Extract agent details from ASOR agent card (per OpenAPI spec)
=======
        # Extract agent details from response
        # Note: Adjust field names based on actual ASOR API response structure
>>>>>>> Stashed changes
        name = response.get("name", agent_id)
        description = response.get("description", "")
        version = response.get("version", "1.0.0")

<<<<<<< Updated upstream
        # Extract endpoint/URL - ASOR uses "url" field
        endpoint = response.get("url")

        # Extract capabilities (ASOR capabilities object)
        capabilities_obj = response.get("capabilities", {})
        capabilities = {
            "streaming": capabilities_obj.get("streaming", False),
            "pushNotifications": capabilities_obj.get("pushNotifications", False),
            "stateTransitionHistory": capabilities_obj.get("stateTransitionHistory", False)
        }

        # Extract skills - ASOR agents have skills array
        skills = response.get("skills", [])

        # Extract workday resources from workdayConfig
        workday_config = response.get("workdayConfig", [])
        workday_resources = []
        for config in workday_config:
            resources = config.get("workdayResources", [])
            workday_resources.extend(resources)

        # Generate tags from skills
        tags = ["asor", "workday", "federated"]

        # Add skill tags
        for skill in skills:
            skill_tags = skill.get("tags", [])
            for tag_obj in skill_tags:
                tag = tag_obj.get("tag")
                if tag and tag not in tags:
                    tags.append(tag)

        # Add metadata category if provided
        if agent_config and agent_config.metadata:
            category = agent_config.metadata.get("category")
            if category and category not in tags:
                tags.append(category)

        # Count total tools from skills
        num_tools = len(workday_resources)

        # Build transformed agent object
        transformed = {
            "source": "asor",
            "server_name": f"asor/{name.lower().replace(' ', '-')}",
=======
        # Extract endpoint/URL
        endpoint = response.get("endpoint") or response.get("url")

        # Extract capabilities
        capabilities = response.get("capabilities", [])
        tools = response.get("tools", [])

        # Generate tags
        tags = ["asor", "workday", "federated"]
        if agent_config and agent_config.metadata:
            category = agent_config.metadata.get("category")
            if category:
                tags.append(category)

        # Build transformed agent object
        transformed = {
            "source": "asor",
            "server_name": f"asor/{agent_id}",
>>>>>>> Stashed changes
            "description": description,
            "version": version,
            "title": name,
            "proxy_pass_url": endpoint,
<<<<<<< Updated upstream
            "transport_type": "streamable-http" if capabilities.get("streaming") else "http",
            "requires_auth": True,  # ASOR agents require auth via Agent Gateway
=======
            "transport_type": "streamable-http",  # Assume HTTP transport
            "requires_auth": True,  # ASOR agents likely require auth
>>>>>>> Stashed changes
            "auth_headers": [],  # Auth handled by gateway
            "tags": tags,
            "metadata": {
                "original_response": response,
<<<<<<< Updated upstream
                "agent_id": response.get("id", agent_id),
                "capabilities": capabilities,
                "skills": skills,
                "workday_resources": workday_resources,
                "provider": response.get("provider", {}),
                "documentation_url": response.get("documentationUrl"),
                "icon_url": response.get("iconUrl"),
=======
                "agent_id": agent_id,
                "capabilities": capabilities,
                "tools": tools,
>>>>>>> Stashed changes
                "config_metadata": agent_config.metadata if agent_config else {}
            },
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "is_read_only": True,
<<<<<<< Updated upstream
            "attribution_label": "Workday ASOR",
            # Additional fields for compatibility
            "path": f"/asor-{name.lower().replace(' ', '-')}",
            "is_enabled": True,
            "health_status": "active",
            "num_tools": num_tools,
=======
            "attribution_label": "ASOR",
            # Additional fields for compatibility
            "path": f"/asor-{agent_id}",
            "is_enabled": True,
            "health_status": "unknown",
            "num_tools": len(tools) if tools else 0,
>>>>>>> Stashed changes
        }

        return transformed


# Import timedelta for token expiry calculation
from datetime import timedelta
