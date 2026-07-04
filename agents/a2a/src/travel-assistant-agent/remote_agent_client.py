"""Client for communicating with remote A2A agents."""

import logging
from collections.abc import Callable
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
from models import DiscoveredAgent
from url_guard import UnsafeUrlError, assert_fetchable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Outbound-request timeout to a registry-discovered (untrusted) remote agent.
# Kept short so a slow or malicious endpoint cannot tie up a worker; the
# previous 300s let a hostile agent hold the connection open indefinitely.
_REMOTE_AGENT_TIMEOUT_SECONDS: float = 30.0


class RemoteAgentClient:
    """
    Client for communicating with a remote A2A agent.
    This class wraps an A2A agent discovered from the registry, providing
    lazy initialization and reusable client connections.

    Reference: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/agent-to-agent/
    """

    def __init__(
        self,
        agent_url: str,
        agent_name: str,
        agent_id: str,
        skills: list[str] | None = None,
        delegation_token: str | None = None,
    ):
        """Initialize a client for a registry-discovered remote agent.

        Args:
            agent_url: The remote agent's endpoint URL (registrant-controlled,
                not fully trusted). Validated against the SSRF guard before any
                outbound request.
            agent_name: Human-readable agent name (for logging).
            agent_id: Registry path/id used as the cache key.
            skills: Skill names the agent advertises.
            delegation_token: OPTIONAL, audience-restricted, short-lived token
                minted specifically for this target agent. It MUST NOT be the
                caller's registry-scoped token: forwarding the registry token to
                an untrusted remote agent would let that agent replay it against
                the registry API. When ``None`` (the default in this sample), no
                credential is sent to the remote agent.
        """
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.skills = skills or []
        # Per-target delegation credential only -- never the registry token.
        self.delegation_token = delegation_token
        self.agent_card = None
        self.client = None
        self.httpx_client = None
        self._initialized = False
        logger.info(
            f"Created RemoteAgentClient for: {agent_name} (ID: {agent_id}, Skills: {len(self.skills)})"
        )

    async def _ensure_initialized(self):
        if self._initialized:
            return

        logger.info(f"Initializing A2A client for {self.agent_name} at {self.agent_url}")

        # SSRF guard: the endpoint URL is registry-supplied and not fully
        # trusted. Validate (scheme + DNS-resolved public IP) before any fetch
        # so a malicious registered URL cannot pivot to loopback / RFC-1918 /
        # cloud-metadata targets. Fail closed on any validation error.
        try:
            assert_fetchable(self.agent_url)
        except UnsafeUrlError as exc:
            logger.error(f"Refusing to contact remote agent {self.agent_name}: {exc}")
            raise

        headers = {}
        if self.delegation_token:
            # Only an explicit per-target delegation token is ever attached.
            headers["Authorization"] = f"Bearer {self.delegation_token}"

        # Create persistent httpx client (not using context manager). Redirects
        # are disabled so a validated public host cannot 302 the request to an
        # internal address after the pre-fetch SSRF check (rebinding via
        # redirect); the short timeout bounds a hostile/slow endpoint.
        self.httpx_client = httpx.AsyncClient(
            timeout=_REMOTE_AGENT_TIMEOUT_SECONDS,
            headers=headers,
            follow_redirects=False,
        )

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
        # Send a natural language message to the remote agent.
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
            return f"Error communicating with {self.agent_name}: an internal error occurred"

    async def close(self):
        # Close the httpx client and cleanup resources
        if self.httpx_client:
            await self.httpx_client.aclose()
            logger.info(f"Closed httpx client for {self.agent_name}")


class RemoteAgentCache:
    def __init__(self):
        self._cache: dict[str, RemoteAgentClient] = {}
        logger.info("RemoteAgentCache initialized")

    def get(self, agent_id: str) -> RemoteAgentClient | None:
        return self._cache.get(agent_id)

    def get_all(self) -> dict[str, RemoteAgentClient]:
        return self._cache.copy()

    def add(self, agent_id: str, agent_client: RemoteAgentClient):
        self._cache[agent_id] = agent_client
        logger.info(f"Added agent to cache: {agent_id}")

    def cache_discovered_agents(
        self,
        agents: list[DiscoveredAgent],
        delegation_token_provider: Callable[[DiscoveredAgent], str | None] | None = None,
    ) -> dict[str, RemoteAgentClient]:
        """Cache discovered remote agents for later invocation.

        Args:
            agents: Agents returned by registry discovery.
            delegation_token_provider: OPTIONAL callback that mints an
                audience-restricted, short-lived delegation token bound to a
                specific target agent. The caller's registry token MUST NOT be
                passed here -- doing so would leak it to untrusted remote agents.
                When ``None`` (the default), no credential is attached to
                outbound A2A calls.

        Returns:
            The newly cached agent clients keyed by agent id.
        """
        newly_cached = {}

        for agent in agents:
            agent_id = agent.path

            # Skip if already cached
            if agent_id in self._cache:
                logger.info(f"Agent {agent_id} already cached, skipping")
                continue

            # Mint a per-target delegation token if a provider was supplied;
            # otherwise send no credential. Never reuse the registry token.
            delegation_token: str | None = None
            if delegation_token_provider is not None:
                delegation_token = delegation_token_provider(agent)

            # Create and cache the remote agent client
            agent_client = RemoteAgentClient(
                agent_url=agent.url,
                agent_name=agent.name,
                agent_id=agent_id,
                skills=agent.skill_names,
                delegation_token=delegation_token,
            )

            self._cache[agent_id] = agent_client
            newly_cached[agent_id] = agent_client
            logger.info(f"Cached agent: {agent.name} (ID: {agent_id})")

        logger.info(f"Cached {len(newly_cached)} new agents. Total in cache: {len(self._cache)}")
        return newly_cached

    async def clear(self):
        count = len(self._cache)
        for agent_client in self._cache.values():
            await agent_client.close()

        self._cache.clear()
        logger.info(f"Cleared {count} agents from cache")

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._cache
