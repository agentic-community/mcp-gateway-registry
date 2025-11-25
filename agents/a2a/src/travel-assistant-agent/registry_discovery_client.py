"""Client for agent discovery through MCP Gateway Registry."""

import logging
import time
from typing import List, Optional

import aiohttp

from models import DiscoveredAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class RegistryDiscoveryClient:
    """Client for agent discovery through MCP Gateway Registry."""

    def __init__(
        self,
        registry_url: str,
        keycloak_url: str,
        client_id: str,
        client_secret: str,
        realm: str = "mcp-gateway",
    ) -> None:
        """Initialize registry discovery client.

        Args:
            registry_url: Base URL of the registry
            keycloak_url: Base URL of Keycloak
            client_id: M2M client ID
            client_secret: M2M client secret
            realm: Keycloak realm name
        """
        self.registry_url = registry_url.rstrip("/")
        self.keycloak_url = keycloak_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.realm = realm
        self.token: Optional[str] = None
        self.token_expires_at: float = 0

        logger.info(f"RegistryDiscoveryClient initialized for {registry_url}")

    async def _get_token(self) -> str:
        """Get or refresh JWT token from Keycloak using client credentials flow.

        Returns:
            JWT access token

        Raises:
            Exception: If token acquisition fails
        """
        current_time = time.time()
        if self.token and current_time < self.token_expires_at - 60:
            logger.debug("Using cached token")
            return self.token

        token_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"
        logger.debug(f"Requesting new token from {token_url}")

        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            try:
                async with session.post(token_url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Token request failed: {response.status} - {error_text}")
                        raise Exception(f"Failed to get token: {response.status}")

                    token_data = await response.json()
                    self.token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 300)
                    self.token_expires_at = current_time + expires_in

                    logger.info(f"Token acquired, expires in {expires_in}s")
                    return self.token

            except aiohttp.ClientError as e:
                logger.error(f"Network error getting token: {e}")
                raise Exception(f"Network error: {e}")

    async def discover_by_semantic_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[DiscoveredAgent]:
        """Discover agents using semantic search (natural language query).

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return

        Returns:
            List of discovered agents with relevance scores

        Raises:
            Exception: If discovery fails
        """
        logger.info(f"Semantic search: '{query}' (max_results={max_results})")

        token = await self._get_token()
        discovery_url = f"{self.registry_url}/api/search/semantic"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "query": query,
            "entity_types": ["agent"],
            "max_results": max_results
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    discovery_url,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Discovery failed: {response.status} - {error_text}")
                        raise Exception(f"Discovery failed: {response.status}")

                    result = await response.json()
                    agents_data = result.get("agents", [])

                    agents = [DiscoveredAgent(**agent) for agent in agents_data]
                    logger.info(f"Found {len(agents)} agents")

                    return agents

            except aiohttp.ClientError as e:
                logger.error(f"Network error during discovery: {e}")
                raise Exception(f"Network error: {e}")

    async def discover_by_skills(
        self,
        skills: List[str],
        tags: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> List[DiscoveredAgent]:
        """Discover agents by required skills and tags.

        Args:
            skills: Required skill names or IDs
            tags: Optional tag filters
            max_results: Maximum number of results to return

        Returns:
            List of discovered agents with relevance scores

        Raises:
            Exception: If discovery fails
        """
        logger.info(f"Skill-based search: skills={skills}, tags={tags}")

        token = await self._get_token()
        discovery_url = f"{self.registry_url}/api/agents/discover"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "skills": skills,
            "tags": tags or [],
            "max_results": max_results,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    discovery_url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Discovery failed: {response.status} - {error_text}")
                        raise Exception(f"Discovery failed: {response.status}")

                    result = await response.json()
                    agents_data = result.get("agents", [])

                    agents = [DiscoveredAgent(**agent) for agent in agents_data]
                    logger.info(f"Found {len(agents)} agents")

                    return agents

            except aiohttp.ClientError as e:
                logger.error(f"Network error during discovery: {e}")
                raise Exception(f"Network error: {e}")
