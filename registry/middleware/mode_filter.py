"""
Middleware to filter API endpoints based on registry mode.

This middleware enforces registry mode restrictions by returning 403 Forbidden
for endpoints that are disabled in the current mode (e.g., skills-only mode
blocks /api/servers and /api/agents endpoints).
"""

import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings, RegistryMode
from ..core.metrics import MODE_BLOCKED_REQUESTS


logger = logging.getLogger(__name__)


# Endpoints that are always allowed regardless of mode
# These are administrative/infrastructure endpoints, not feature-specific
ALWAYS_ALLOWED_PREFIXES = (
    "/health",
    "/api/version",
    "/api/stats",
    "/api/config",
    "/api/auth/",
    "/api/tokens/",
    "/api/audit/",
    "/api/management/",
    "/oauth2/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/static/",
    "/assets/",
    "/_next/",
    "/validate",
    "/favicon",
)

# API endpoints allowed in skills-only mode
SKILLS_MODE_ALLOWED_PREFIXES = (
    "/api/skills",
    "/api/search/semantic",
)

# API endpoints allowed in mcp-servers-only mode
MCP_SERVERS_MODE_ALLOWED_PREFIXES = (
    "/api/servers",
    "/api/search/semantic",
    "/.well-known/mcp-servers",
)

# API endpoints allowed in agents-only mode
AGENTS_MODE_ALLOWED_PREFIXES = (
    "/api/agents",
    "/api/search/semantic",
)


def _is_path_allowed(path: str, mode: RegistryMode) -> bool:
    """Check if path is allowed for the given registry mode.

    Args:
        path: Request URL path
        mode: Current registry mode

    Returns:
        True if path is allowed, False otherwise
    """
    # Always allowed paths (auth, health, docs, static, etc.)
    for prefix in ALWAYS_ALLOWED_PREFIXES:
        if path.startswith(prefix):
            return True

    # Full mode allows everything
    if mode == RegistryMode.FULL:
        return True

    # Skills-only mode
    if mode == RegistryMode.SKILLS_ONLY:
        # Allow skills-related endpoints
        for prefix in SKILLS_MODE_ALLOWED_PREFIXES:
            if path.startswith(prefix):
                return True
        # Allow well-known but it will return empty list
        if path.startswith("/.well-known/"):
            return True
        # Block all other /api/* endpoints
        if path.startswith("/api/"):
            return False
        # Allow non-API paths (frontend, etc.)
        return True

    # MCP-servers-only mode
    if mode == RegistryMode.MCP_SERVERS_ONLY:
        for prefix in MCP_SERVERS_MODE_ALLOWED_PREFIXES:
            if path.startswith(prefix):
                return True
        if path.startswith("/api/"):
            return False
        return True

    # Agents-only mode
    if mode == RegistryMode.AGENTS_ONLY:
        for prefix in AGENTS_MODE_ALLOWED_PREFIXES:
            if path.startswith(prefix):
                return True
        if path.startswith("/api/"):
            return False
        return True

    # Unknown mode - allow by default
    return True


def _get_path_category(path: str) -> str:
    """Extract path category for metrics labeling.

    Args:
        path: Request URL path

    Returns:
        Category string for metrics (e.g., 'servers', 'agents', 'federation')
    """
    if path.startswith("/api/servers"):
        return "servers"
    if path.startswith("/api/agents"):
        return "agents"
    if path.startswith("/api/skills"):
        return "skills"
    if path.startswith("/api/federation") or path.startswith("/api/peers"):
        return "federation"
    if path.startswith("/api/"):
        parts = path.split("/")
        if len(parts) > 2:
            return parts[2]
    return "other"


class RegistryModeMiddleware(BaseHTTPMiddleware):
    """Middleware to filter requests based on registry mode."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and block if endpoint is disabled for current mode.

        Args:
            request: Incoming HTTP request
            call_next: Next handler in middleware chain

        Returns:
            Response from next handler or 403 if blocked
        """
        path = request.url.path
        mode = settings.registry_mode

        # Check if path is allowed for current mode
        if not _is_path_allowed(path, mode):
            # Log blocked request
            client_host = request.client.host if request.client else "unknown"
            logger.warning(
                f"Blocked request to '{path}' - endpoint disabled in {mode.value} mode. "
                f"Client: {client_host}"
            )

            # Increment metrics counter
            category = _get_path_category(path)
            MODE_BLOCKED_REQUESTS.labels(path_category=category, mode=mode.value).inc()

            return JSONResponse(
                status_code=403,
                content={
                    "detail": f"This endpoint is disabled in {mode.value} mode",
                    "error": "endpoint_disabled",
                    "registry_mode": mode.value,
                    "path": path,
                },
            )

        return await call_next(request)
