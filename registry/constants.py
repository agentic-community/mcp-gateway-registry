"""
Constants and enums for the MCP Gateway Registry.
"""

import os
from enum import Enum

from pydantic import BaseModel


class HealthStatus(str, Enum):
    """Health status constants for services."""

    HEALTHY = "healthy"
    HEALTHY_AUTH_EXPIRED = "healthy-auth-expired"
    UNHEALTHY_TIMEOUT = "unhealthy: timeout"
    UNHEALTHY_CONNECTION_ERROR = "unhealthy: connection error"
    UNHEALTHY_ENDPOINT_CHECK_FAILED = "unhealthy: endpoint check failed"
    UNHEALTHY_MISSING_PROXY_URL = "unhealthy: missing proxy URL"
    UNHEALTHY_URL_BLOCKED = "unhealthy: url blocked by SSRF guard"
    CHECKING = "checking"
    UNKNOWN = "unknown"
    LOCAL = "local"

    @classmethod
    def get_healthy_statuses(cls) -> list[str]:
        """Get list of statuses that should be considered healthy for nginx inclusion."""
        return [cls.HEALTHY, cls.HEALTHY_AUTH_EXPIRED]

    @classmethod
    def is_healthy(cls, status: str) -> bool:
        """Check if a status should be considered healthy."""
        return status in cls.get_healthy_statuses()


class TransportType(str, Enum):
    """Supported transport types for MCP servers."""

    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"
    STDIO = "stdio"


class DeploymentType(str, Enum):
    """Server deployment type: remote (HTTP-reachable) or local (stdio)."""

    REMOTE = "remote"
    LOCAL = "local"


class LocalRuntimeType(str, Enum):
    """Launcher type for local stdio MCP servers."""

    NPX = "npx"
    DOCKER = "docker"
    UVX = "uvx"
    COMMAND = "command"


class AuthScheme(str, Enum):
    """Authentication scheme for backend MCP servers."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"


# Auth header defaults
DEFAULT_API_KEY_HEADER: str = "X-API-Key"
DEFAULT_BEARER_HEADER: str = "Authorization"
VALID_AUTH_SCHEMES: list = ["none", "bearer", "api_key"]


class RegistryConstants(BaseModel):
    """Registry configuration constants."""

    class Config:
        """Pydantic config."""

        frozen = True

    # Health check settings
    DEFAULT_HEALTH_CHECK_TIMEOUT: int = 30
    HEALTH_CHECK_INTERVAL: int = 30

    # SSL certificate paths
    SSL_CERT_PATH: str = "/etc/ssl/certs/fullchain.pem"
    SSL_KEY_PATH: str = "/etc/ssl/private/privkey.pem"

    # Nginx settings
    NGINX_CONFIG_PATH: str = "/etc/nginx/conf.d/nginx_rev_proxy.conf"
    NGINX_TEMPLATE_HTTP_ONLY: str = "/app/docker/nginx_rev_proxy_http_only.conf"
    NGINX_TEMPLATE_HTTP_AND_HTTPS: str = "/app/docker/nginx_rev_proxy_http_and_https.conf"
    NGINX_TEMPLATE_HTTP_ONLY_LOCAL: str = "docker/nginx_rev_proxy_http_only.conf"
    NGINX_TEMPLATE_HTTP_AND_HTTPS_LOCAL: str = "docker/nginx_rev_proxy_http_and_https.conf"

    # Server settings
    DEFAULT_TRANSPORT: str = TransportType.STREAMABLE_HTTP
    SUPPORTED_TRANSPORTS: list[str] = [TransportType.STREAMABLE_HTTP, TransportType.SSE]

    # Anthropic Registry API constants
    ANTHROPIC_API_VERSION: str = "v0.1"
    ANTHROPIC_SERVER_NAMESPACE: str = "io.mcpgateway"
    ANTHROPIC_API_DEFAULT_LIMIT: int = 100
    ANTHROPIC_API_MAX_LIMIT: int = 1000

    # External Registry Tags
    # Comma-separated list of tags that identify external registry servers
    # Example: "anthropic-registry,workday-asor,custom-registry"
    EXTERNAL_REGISTRY_TAGS: str = os.getenv(
        "EXTERNAL_REGISTRY_TAGS", "anthropic-registry,workday-asor"
    )


# Global instance
REGISTRY_CONSTANTS = RegistryConstants()

# Maximum custom headers per server
MAX_CUSTOM_HEADERS_PER_SERVER: int = 10

# Header names the registry and gateway own. Custom headers with a name
# whose lowercased form appears here are rejected at registration time (and the
# egress hop strips them as a backstop for any bypass-written document).
#
# Three groups: (1) hop-by-hop / framing / content headers a proxy must control;
# (2) ingress credentials (authorization/x-authorization/cookie); (3) the
# GATEWAY-INTERNAL identity + routing + signed-token headers that /validate sets
# on the auth_request subresponse and nginx copies onto the proxied request.
# Group 3 is the #1391 credential-leak class: if an entity could register one of
# these as a custom (esp. overridable) upstream header, the caller could cause the
# gateway's own signed internal token / the authenticated user's identity to be
# forwarded to a registrant-controlled backend and replayed against the registry.
# They must NEVER be registrable or forwarded, in any form.
RESERVED_CUSTOM_HEADER_NAMES: frozenset[str] = frozenset(
    {
        # (1) hop-by-hop / framing / content
        "content-type",
        "content-length",
        "accept",
        "host",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-host",
        "x-real-ip",
        "set-cookie",
        # (2) ingress credentials
        "authorization",
        "x-authorization",
        "proxy-authorization",
        "cookie",
        # (3) gateway-internal identity / routing / signed-token headers
        "x-user",
        "x-username",
        "x-scopes",
        "x-auth-method",
        "x-groups",
        "x-client-id",
        "x-user-pool-id",
        "x-region",
        "x-original-url",
        "x-server-name",
        "x-tool-name",
        "x-internal-token",
        "x-internal-token-generic",
        "x-internal-token-registry",
        "x-generic-proxy-kind",
        "x-generic-streaming",
        "x-generic-has-upstream-auth",
        "x-resolved-upstream",
        "x-resolved-generic-upstream",
        "x-upstream-url",
        "x-body",
        "x-body-uninspectable",
    }
)
