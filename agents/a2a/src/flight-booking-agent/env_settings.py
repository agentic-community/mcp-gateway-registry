"""Environment settings for Flight Booking Agent."""

import logging
import os

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class EnvSettings:
    """Environment settings configuration."""

    def __init__(self) -> None:
        """Initialize environment settings."""
        self.db_path: str = os.getenv("DB_PATH", "/app/data/bookings.db")
        self.aws_region: str = os.getenv("AWS_REGION") or os.getenv(
            "AWS_DEFAULT_REGION", "us-east-1"
        )
        self.agent_name: str = os.getenv("AGENT_NAME", "flight-booking")
        self.agent_version: str = os.getenv("AGENT_VERSION", "1.0.0")

        # MCP Gateway Registry URL (TODO: replace later)
        self.mcp_registry_url: str = os.getenv("MCP_REGISTRY_URL", "http://localhost:7860")

        # Agent's public URL (AgentCore Runtime injects automatically)
        self.agent_url: str = os.getenv("AGENTCORE_RUNTIME_URL", "http://127.0.0.1:9000/")

        # Server configuration (fixed for A2A protocol).
        # Default to loopback so the agent is not exposed on all interfaces unless
        # the deployment explicitly opts in via AGENT_HOST (e.g. "0.0.0.0" inside a
        # container whose network is isolated by the runtime). JWT auth still gates
        # every request regardless of the bind address.
        self.host: str = os.getenv("AGENT_HOST", "127.0.0.1")
        self.port: int = 9000

        # Keycloak configuration for inbound JWT validation.
        self.keycloak_url: str = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
        self.keycloak_realm: str = os.getenv("KEYCLOAK_REALM", "mcp-gateway")

        # Expected audience for inbound bearer tokens. When unset, audience is not
        # enforced (set AGENT_AUDIENCE in production to bind tokens to this agent).
        self.agent_audience: str | None = os.getenv("AGENT_AUDIENCE") or None

        logger.info(
            f"EnvSettings initialized: agent_name={self.agent_name}, version={self.agent_version}"
        )
        logger.debug(f"Database path: {self.db_path}")
        logger.debug(f"Agent URL: {self.agent_url}")
