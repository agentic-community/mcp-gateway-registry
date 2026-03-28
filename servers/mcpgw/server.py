"""MCP Gateway Interaction Server (mcpgw).

This MCP server provides tools to interact with the MCP Gateway Registry API.
It acts as a thin protocol adapter, translating MCP tool calls into registry HTTP requests.

Supports two auth modes:
  - OAuth (OAuthProxy + Keycloak): set OIDC_ENABLED=true and provide Keycloak env vars.
    Exposes /.well-known/oauth-protected-resource for MCP clients (Cursor, VS Code).
  - Legacy bearer token: pass a Keycloak JWT via Authorization header directly.
"""

import logging
import os
import time
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from models import AgentInfo, RegistryStats, ServerInfo, SkillInfo, ToolSearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

REGISTRY_URL = os.getenv("REGISTRY_BASE_URL", "http://localhost")

MAX_QUERY_LENGTH: int = 500
MIN_TOP_N: int = 1
MAX_TOP_N: int = 50

logger.info(f"Registry URL: {REGISTRY_URL}")

# ---------------------------------------------------------------------------
# OAuth configuration (optional – enable via OIDC_ENABLED=true)
# ---------------------------------------------------------------------------
OIDC_ENABLED = os.getenv("OIDC_ENABLED", "").lower() in ("true", "1", "yes")

KEYCLOAK_INTERNAL_URL = os.getenv("KEYCLOAK_INTERNAL_URL", "http://keycloak:8080")
KEYCLOAK_EXTERNAL_URL = os.getenv("KEYCLOAK_EXTERNAL_URL", "http://localhost:18080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "mcp-gateway")
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "mcp-gateway-web")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "")
M2M_CLIENT_ID = os.getenv("M2M_CLIENT_ID", "mcp-gateway-m2m")
M2M_CLIENT_SECRET = os.getenv("M2M_CLIENT_SECRET", "")
MCPGW_BASE_URL = os.getenv("MCPGW_BASE_URL", "http://localhost:18003")
REGISTRY_API_TOKEN = os.getenv("REGISTRY_API_TOKEN", "")


class _M2MTokenManager:
    """Fetches and caches a Keycloak M2M token via client_credentials grant."""

    def __init__(self, token_url: str, client_id: str, client_secret: str) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._expires_at: float = 0

    async def get_token(self) -> str:
        if self._token and time.monotonic() < self._expires_at - 60:
            return self._token

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._expires_at = time.monotonic() + data.get("expires_in", 300)
            logger.info("Obtained fresh M2M token (expires_in=%s)", data.get("expires_in"))
            return self._token


_auth_provider = None
_m2m_manager: _M2MTokenManager | None = None
_realm_path = f"/realms/{KEYCLOAK_REALM}/protocol/openid-connect"

if M2M_CLIENT_ID and M2M_CLIENT_SECRET:
    _m2m_manager = _M2MTokenManager(
        token_url=f"{KEYCLOAK_INTERNAL_URL}{_realm_path}/token",
        client_id=M2M_CLIENT_ID,
        client_secret=M2M_CLIENT_SECRET,
    )
    logger.info("M2M token manager enabled (client=%s)", M2M_CLIENT_ID)

if OIDC_ENABLED:
    from fastmcp.server.auth.oauth_proxy import OAuthProxy
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    _auth_provider = OAuthProxy(
        upstream_authorization_endpoint=f"{KEYCLOAK_EXTERNAL_URL}{_realm_path}/auth",
        upstream_token_endpoint=f"{KEYCLOAK_INTERNAL_URL}{_realm_path}/token",
        upstream_revocation_endpoint=f"{KEYCLOAK_INTERNAL_URL}{_realm_path}/revoke",
        upstream_client_id=OIDC_CLIENT_ID,
        upstream_client_secret=OIDC_CLIENT_SECRET,
        token_verifier=JWTVerifier(
            jwks_uri=f"{KEYCLOAK_INTERNAL_URL}{_realm_path}/certs",
            issuer=f"{KEYCLOAK_EXTERNAL_URL}/realms/{KEYCLOAK_REALM}",
        ),
        base_url=MCPGW_BASE_URL,
        allowed_client_redirect_uris=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "cursor://anysphere.cursor-mcp/*",
            "vscode://anysphere.cursor-mcp/*",
        ],
        require_authorization_consent=False,
    )
    logger.info("OAuth enabled (OAuthProxy → Keycloak %s, realm=%s)", KEYCLOAK_EXTERNAL_URL, KEYCLOAK_REALM)
else:
    logger.info("OAuth disabled – using bearer-token passthrough with M2M for registry calls")

mcp = FastMCP("mcpgw", auth=_auth_provider)

if _auth_provider:
    from starlette.responses import RedirectResponse

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _redirect_protected_resource(_):  # noqa: ANN001
        """Redirect root well-known to the MCP-prefixed path (FastMCP path-prefix workaround)."""
        return RedirectResponse(
            url="/.well-known/oauth-protected-resource/mcp", status_code=302
        )


def _validate_top_n(top_n: int) -> int:
    """Validate top_n parameter is within acceptable bounds.

    Args:
        top_n: Number of results to return

    Returns:
        Validated top_n value

    Raises:
        ValueError: If top_n is out of bounds
    """
    if not isinstance(top_n, int) or top_n < MIN_TOP_N or top_n > MAX_TOP_N:
        raise ValueError(f"top_n must be an integer between {MIN_TOP_N} and {MAX_TOP_N}")
    return top_n


def _validate_query(query: str) -> str:
    """Validate query parameter.

    Args:
        query: Search query string

    Returns:
        Validated and trimmed query

    Raises:
        ValueError: If query is empty or too long
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")

    return query.strip()


def _extract_bearer_token(ctx: Context | None) -> str:
    """Extract bearer token from FastMCP context (legacy / no-OAuth mode).

    Supports both standard Authorization header and MCP Gateway's X-Authorization header.
    """
    if not ctx:
        raise ValueError("Authentication required: Context is None")

    try:
        if hasattr(ctx, "request_context") and ctx.request_context:
            request = ctx.request_context.request
            if request and hasattr(request, "headers"):
                auth_header = request.headers.get("authorization")
                if not auth_header:
                    auth_header = request.headers.get("x-authorization")
                if auth_header and auth_header.lower().startswith("bearer "):
                    return auth_header.split(" ", 1)[1]
                raise ValueError("Bearer token not found in Authorization or X-Authorization header")
            raise ValueError("Request object or headers not found in request_context")
        raise ValueError("request_context not available in Context")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to extract token: {e}", exc_info=True)
        raise ValueError(f"Failed to extract bearer token: {e}") from e


async def _get_registry_headers(ctx: Context | None) -> dict[str, str]:
    """Return headers for internal registry API calls.

    Priority: static API token > M2M service token > caller bearer token.
    """
    if REGISTRY_API_TOKEN:
        return {"Authorization": f"Bearer {REGISTRY_API_TOKEN}"}
    if _m2m_manager:
        token = await _m2m_manager.get_token()
        return {"X-Authorization": f"Bearer {token}"}
    token = _extract_bearer_token(ctx)
    return {"X-Authorization": f"Bearer {token}"}


@mcp.tool()
async def list_services(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all MCP servers registered in the gateway.

    Returns:
        Dictionary containing services, total_count, enabled_count, and status
    """
    logger.info("list_services called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/servers", headers=headers)
            response.raise_for_status()
            data = response.json()

        if isinstance(data, dict) and "servers" in data:
            servers = data["servers"]
        elif isinstance(data, list):
            servers = data
        else:
            servers = []

        services = []
        for s in servers:
            try:
                services.append(ServerInfo(**s).model_dump())
            except Exception as e:
                logger.warning(f"Failed to parse server {s.get('path', 'unknown')}: {e}")
        enabled_count = sum(1 for s in services if s.get("enabled"))

        return {
            "services": services,
            "total_count": len(services),
            "enabled_count": enabled_count,
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "services": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "services": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        return {
            "services": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def list_agents(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all agents registered in the gateway.

    Returns:
        Dictionary containing agents, total_count, and status
    """
    logger.info("list_agents called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/agents", headers=headers)
            response.raise_for_status()
            data = response.json()

        agents = data.get("agents", []) if isinstance(data, dict) else data
        agent_list = [AgentInfo(**a).model_dump() for a in agents]

        return {
            "agents": agent_list,
            "total_count": len(agent_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "agents": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "agents": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return {
            "agents": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def list_skills(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all skills registered in the gateway.

    Returns:
        Dictionary containing skills, total_count, and status
    """
    logger.info("list_skills called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/skills", headers=headers)
            response.raise_for_status()
            data = response.json()

        skills = data.get("skills", []) if isinstance(data, dict) else data
        skill_list = [SkillInfo(**s).model_dump() for s in skills]

        return {
            "skills": skill_list,
            "total_count": len(skill_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "skills": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "skills": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        return {
            "skills": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }



@mcp.tool()
async def get_skill_content(
    skill_name: str,
    resource_path: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Fetch skill content from the registry.

    Without resource_path: returns the full SKILL.md markdown and resource manifest.
    With resource_path: returns the content of a companion file (reference doc,
    script, agent config, etc.) validated against the stored manifest.

    Use this after list_skills or intelligent_tool_finder to retrieve the
    complete workflow instructions for a skill, or to read companion resources
    listed in the manifest.

    Args:
        skill_name: Name of the skill (e.g. "gerrit-workflow")
        resource_path: Optional relative path to a companion resource
                       (e.g. "references/architecture.md")

    Returns:
        Dictionary containing the skill name, content, source URL, and status
    """
    logger.info(
        "get_skill_content called: skill_name=%s resource_path=%s",
        skill_name,
        resource_path,
    )

    if not skill_name or not skill_name.strip():
        return {"error": "skill_name cannot be empty", "status": "failed"}

    skill_name = skill_name.strip()

    try:
        headers = await _get_registry_headers(ctx)
        url = f"{REGISTRY_URL}/api/skills/{skill_name}/content"
        params: dict[str, str] = {}
        if resource_path:
            params["resource"] = resource_path

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        result: dict[str, Any] = {
            "skill_name": skill_name,
            "source_url": data.get("url", ""),
            "content": data.get("content", ""),
            "status": "success",
        }
        if resource_path:
            result["resource_path"] = data.get("path", resource_path)
            result["resource_type"] = data.get("type", "")
        else:
            manifest = data.get("resource_manifest")
            if manifest:
                result["resources"] = manifest
        return result

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error fetching skill content: %s", e.response.status_code)
        return {"skill_name": skill_name, "error": f"HTTP {e.response.status_code}", "status": "failed"}
    except Exception as e:
        logger.error("Failed to get skill content: %s", e)
        return {"skill_name": skill_name, "error": str(e), "status": "failed"}



@mcp.tool()
async def intelligent_tool_finder(
    query: str,
    top_n: int = 5,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Search for tools using natural language semantic search.

    Args:
        query: Natural language description of what you want to do
        top_n: Number of results to return (default: 5, max: 50)

    Returns:
        Dictionary containing results, query, total_results, and status
    """
    logger.info(f"intelligent_tool_finder called: query={query}, top_n={top_n}")

    try:
        query = _validate_query(query)
        top_n = _validate_top_n(top_n)
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{REGISTRY_URL}/api/search/semantic",
                headers=headers,
                json={
                    "query": query,
                    "entity_types": ["mcp_server", "tool", "virtual_server"],
                    "max_results": top_n,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract servers array from response
        servers = data.get("servers", []) if isinstance(data, dict) else []

        # Flatten matching_tools from all servers into ToolSearchResult objects
        result_list = []
        for server in servers:
            server_path = server.get("path", "")
            server_name = server.get("server_name", "")
            for tool in server.get("matching_tools", []):
                result_list.append(
                    ToolSearchResult(
                        tool_name=tool.get("tool_name", ""),
                        server_name=server_name,
                        description=tool.get("description"),
                        score=tool.get("relevance_score"),
                        path=server_path,
                    ).model_dump()
                )

        # Enforce client-side limit (safety net in case registry returns more)
        result_list = result_list[:top_n]

        return {
            "results": result_list,
            "query": query,
            "total_results": len(result_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to search tools: {e}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def healthcheck(ctx: Context | None = None) -> dict[str, Any]:
    """
    Get registry health status and statistics.

    Returns:
        Dictionary containing health stats and status
    """
    logger.info("healthcheck called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/servers/health", headers=headers)
            response.raise_for_status()
            data = response.json()

        stats = RegistryStats(**data)
        return {**stats.model_dump(), "status": "success"}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "health_status": "error",
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "health_status": "error",
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            "health_status": "error",
            "error": str(e),
            "status": "failed",
        }


if __name__ == "__main__":
    import os

    logger.info("Starting mcpgw server")

    # Use HTTP transport if PORT is set (Docker container), otherwise stdio
    port = os.environ.get("PORT")
    if port:
        # Use configurable host with secure default (127.0.0.1)
        # Set HOST=0.0.0.0 in environment for Docker deployments
        host = os.environ.get("HOST", "127.0.0.1")
        logger.info(f"Running in HTTP mode on {host}:{port}")
        mcp.run(transport="streamable-http", host=host, port=int(port))
    else:
        logger.info("Running in stdio mode")
        mcp.run(transport="stdio")
