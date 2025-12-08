"""Factory for creating A2A tools from discovered agents."""

import logging
from typing import Dict, List, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
from strands import tool

from models import DiscoveredAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Global cache: agent_id -> A2AAgentTool instance
_remote_agent_cache: Dict[str, "A2AAgentTool"] = {}


class A2AAgentTool:
    """Wrapper for remote A2A agents - initialized once and reusable.
    
    This class wraps an A2A agent discovered from the registry, providing
    lazy initialization and reusable client connections.
    """

    def __init__(
        self,
        agent_url: str,
        agent_name: str,
        agent_id: str,
        auth_token: Optional[str] = None,
    ):
        """Initialize A2A agent tool wrapper.

        Args:
            agent_url: Full URL to the A2A agent endpoint
            agent_name: Human-readable name of the agent
            agent_id: Unique identifier (typically the registry path)
            auth_token: Optional JWT token for authentication
        """
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.auth_token = auth_token
        self.agent_card = None
        self.client = None
        self.httpx_client = None
        self._initialized = False
        logger.info(f"Created A2AAgentTool for: {agent_name} (ID: {agent_id})")

    async def _ensure_initialized(self):
        """Lazy initialization of A2A client - only happens on first invoke."""
        if self._initialized:
            return

        logger.info(f"Initializing A2A client for {self.agent_name} at {self.agent_url}")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Create persistent httpx client (not using context manager)
        self.httpx_client = httpx.AsyncClient(timeout=300, headers=headers)
        
        # Get agent card
        resolver = A2ACardResolver(httpx_client=self.httpx_client, base_url=self.agent_url)
        self.agent_card = await resolver.get_agent_card()

        # Create client with persistent httpx_client
        config = ClientConfig(httpx_client=self.httpx_client, streaming=False)
        factory = ClientFactory(config)
        self.client = factory.create(self.agent_card)

        self._initialized = True
        logger.info(f"A2A client initialized for {self.agent_name}")

    async def send_message(self, message: str) -> str:
        """Send a natural language message to the remote agent.

        Args:
            message: Natural language message to send to the agent

        Returns:
            Response text from the agent
        """
        await self._ensure_initialized()

        logger.info(f"Sending message to {self.agent_name}: {message[:100]}...")

        try:
            # Create A2A message
            msg = Message(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=message))],
                message_id=uuid4().hex,
            )

            # Send message and get response
            async for event in self.client.send_message(msg):
                if isinstance(event, Message):
                    response_text = ""
                    for part in event.parts:
                        if hasattr(part, "text"):
                            response_text += part.text
                    logger.info(f"Message sent successfully to {self.agent_name}")
                    return response_text

            return f"No response received from {self.agent_name}"

        except Exception as e:
            logger.error(f"Message failed: {e}", exc_info=True)
            return f"Error communicating with {self.agent_name}: {str(e)}"

    async def close(self):
        """Close the httpx client and cleanup resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            logger.info(f"Closed httpx client for {self.agent_name}")





def get_remote_agent_cache() -> Dict[str, A2AAgentTool]:
    """Get the global remote agent cache.
    
    Returns:
        Dictionary mapping agent IDs to A2AAgentTool instances
    """
    return _remote_agent_cache


async def clear_remote_agent_cache():
    """Clear all cached remote agents and cleanup resources."""
    global _remote_agent_cache
    count = len(_remote_agent_cache)
    
    # Close all httpx clients
    for agent_tool in _remote_agent_cache.values():
        await agent_tool.close()
    
    _remote_agent_cache.clear()
    logger.info(f"Cleared {count} agents from cache")


def cache_discovered_agents(
    agents: List[DiscoveredAgent],
    auth_token: Optional[str] = None,
) -> Dict[str, A2AAgentTool]:
    """Cache discovered agents as A2AAgentTool instances.
    
    Args:
        agents: List of discovered agents from registry
        auth_token: Optional JWT token for authentication
        
    Returns:
        Dictionary of newly cached agents (agent_id -> A2AAgentTool)
    """
    newly_cached = {}
    
    for agent in agents:
        agent_id = agent.path
        
        # Skip if already cached
        if agent_id in _remote_agent_cache:
            logger.info(f"Agent {agent_id} already cached, skipping")
            continue
        
        # Create and cache the A2A agent tool
        agent_tool = A2AAgentTool(
            agent_url=agent.url,
            agent_name=agent.name,
            agent_id=agent_id,
            auth_token=auth_token,
        )
        
        _remote_agent_cache[agent_id] = agent_tool
        newly_cached[agent_id] = agent_tool
        logger.info(f"Cached agent: {agent.name} (ID: {agent_id})")
    
    logger.info(f"Cached {len(newly_cached)} new agents. Total in cache: {len(_remote_agent_cache)}")
    return newly_cached






