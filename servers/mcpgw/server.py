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
from logging_setup import setup_mcpgw_logging
from models import AgentInfo, RegistryStats, ServerInfo, SkillInfo, ToolSearchResult
from observability_bootstrap import init_meter_provider_if_needed, track_tool
from starlette.responses import JSONResponse

# Issue #1122: start the OTel Prometheus exporter listener so the in-cluster
# Prometheus can scrape mcpgw on :9464. No-op when
# OTEL_EXPORTER_PROMETHEUS_HOST is unset.
init_meter_provider_if_needed()

_log_file = setup_mcpgw_logging()
logger = logging.getLogger(__name__)
logger.info(
    "mcpgw logging configured: file=%s format=%s level=%s",
    _log_file,
    os.getenv("APP_LOG_FILE_FORMAT", "json"),
    os.getenv("APP_LOG_LEVEL", "INFO"),
)

REGISTRY_URL = os.getenv("REGISTRY_BASE_URL", "http://localhost")
REGISTRY_EXTERNAL_URL = os.getenv("REGISTRY_EXTERNAL_URL", "")

MAX_QUERY_LENGTH: int = 500
MIN_TOP_N: int = 1
MAX_TOP_N: int = 50

# Max number of withheld candidates itemized in a discovery receipt. The full
# count is always reported; this caps the listed near-miss items so the receipt
# stays compact and low-token for eval/agent-dev callers.
MAX_WITHHELD_ITEMS: int = 5

logger.info(f"Registry URL: {REGISTRY_URL}")
if REGISTRY_EXTERNAL_URL:
    logger.info(f"Registry External URL: {REGISTRY_EXTERNAL_URL}")

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
    logger.info(
        "OAuth enabled (OAuthProxy → Keycloak %s, realm=%s)", KEYCLOAK_EXTERNAL_URL, KEYCLOAK_REALM
    )
else:
    logger.info("OAuth disabled – using bearer-token passthrough with M2M for registry calls")

mcp = FastMCP(
    "AI Registry",
    instructions=(
        "This server connects you to an AI Registry containing MCP servers, "
        "tools, agents, and skills that you can discover and use. "
        "Start with search_registry to find relevant AI assets by describing "
        "what you need in natural language. Once you find a useful MCP server, "
        "you can connect to it directly via its endpoint URL. "
        "For Claude Code, add the server with: "
        "claude mcp add --transport http --scope user <name> <endpoint_url>. "
        "Adding a server usually requires restarting the AI assistant session "
        "for the new tools to take effect. "
        "For skills, use get_skill_content to retrieve the full instructions."
    ),
    auth=_auth_provider,
)


@mcp.custom_route("/health", methods=["GET"])
async def _health(_):  # noqa: ANN001
    """Local liveness/readiness probe. No external dependencies."""
    return JSONResponse({"status": "ok"})


if _auth_provider:
    from starlette.responses import RedirectResponse

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _redirect_protected_resource(_):  # noqa: ANN001
        """Redirect root well-known to the MCP-prefixed path (FastMCP path-prefix workaround)."""
        return RedirectResponse(url="/.well-known/oauth-protected-resource/mcp", status_code=302)


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


def _dedupe_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate discovery candidates while preserving order.

    The registry can return the same tool in both the top-level ``tools`` array
    and a server's ``matching_tools`` list. Without dedup the receipt counts that
    single tool twice, which inflates exposed_results and skews the withheld
    count. Candidates are keyed by (asset_type, service_path, name); on a
    collision the entry with the higher similarity_score is kept.
    """
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for candidate in candidates:
        key = (
            candidate.get("asset_type", ""),
            candidate.get("service_path", ""),
            candidate.get("name", ""),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            order.append(key)
            continue

        # Keep the higher score on collision (scores may be None).
        existing_score = existing.get("similarity_score") or 0
        new_score = candidate.get("similarity_score") or 0
        if new_score > existing_score:
            deduped[key] = candidate

    return [deduped[key] for key in order]


def _build_discovery_receipt(
    *,
    query: str,
    limit: int,
    exposed_results: list[dict[str, Any]],
    withheld_results: list[dict[str, Any]],
    status: str,
    stop_reason: str,
) -> dict[str, Any]:
    """Build a compact, privacy-safe receipt for dynamic discovery results.

    The receipt deliberately records shapes/counts and ranking metadata rather
    than raw tool arguments, tool outputs, or user data. It is an opt-in
    eval/agent-development signal; server-side audit should use logs or OTel.

    Withheld candidates are itemized (capped at MAX_WITHHELD_ITEMS) so an eval
    can see whether the right tool was starved out by a tight limit, not just
    how many were held back.
    """
    return {
        "event": "registry.discovery_receipt",
        "query": query,
        "limits": {"max_results": limit},
        "exposed_results": exposed_results,
        "withheld": {
            "candidate_result_count": len(withheld_results),
            "reason": "outside_intent_or_budget",
            "top_withheld": withheld_results[:MAX_WITHHELD_ITEMS],
        },
        "status": status,
        "stop_reason": stop_reason,
    }


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
                raise ValueError(
                    "Bearer token not found in Authorization or X-Authorization header"
                )
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
    Includes X-Forwarded-Host so the registry can construct correct external URLs.
    """
    if REGISTRY_API_TOKEN:
        headers = {"Authorization": f"Bearer {REGISTRY_API_TOKEN}"}
    elif _m2m_manager:
        token = await _m2m_manager.get_token()
        headers = {"X-Authorization": f"Bearer {token}"}
    else:
        token = _extract_bearer_token(ctx)
        headers = {"X-Authorization": f"Bearer {token}"}

    if REGISTRY_EXTERNAL_URL:
        from urllib.parse import urlparse

        parsed = urlparse(REGISTRY_EXTERNAL_URL)
        if parsed.hostname:
            headers["X-Forwarded-Host"] = parsed.netloc
            headers["X-Forwarded-Proto"] = parsed.scheme or "https"

    return headers


@mcp.tool()
@track_tool()
async def list_services(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all MCP servers registered in the registry. Use search_registry
    instead if you know what capability you need (faster, ranked results).

    Each server entry includes its endpoint URL, tools, and connection details.
    Use this for browsing the full catalog or when you need an unfiltered list.

    Returns:
        Dictionary containing services, total_count, enabled_count, and status
    """
    logger.info("list_services called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{REGISTRY_URL}/api/servers", headers=headers, params={"limit": 2000}
            )
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
@track_tool()
async def list_agents(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all agents registered in the registry. Use search_registry
    instead if you know what task you need an agent for (faster, ranked).

    Agents are autonomous services you can delegate tasks to. Each entry
    includes the agent's URL, capabilities, and skills.

    Returns:
        Dictionary containing agents, total_count, and status
    """
    logger.info("list_agents called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{REGISTRY_URL}/api/agents", headers=headers, params={"limit": 2000}
            )
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
@track_tool()
async def list_skills(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all skills registered in the registry. Use search_registry
    instead if you know what workflow you need (faster, ranked results).

    Skills are reusable workflow instructions (like slash commands) that
    you can load and execute. Use get_skill_content to retrieve the full
    markdown instructions for a discovered skill.

    Returns:
        Dictionary containing skills, total_count, and status
    """
    logger.info("list_skills called")

    try:
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{REGISTRY_URL}/api/skills", headers=headers, params={"limit": 2000}
            )
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
@track_tool()
async def get_skill_content(
    skill_name: str,
    resource_path: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Retrieve the full instructions for a skill. Call this after finding a
    skill via search_registry to get its complete workflow markdown.

    The returned content is a SKILL.md file containing step-by-step
    instructions you can follow to execute the workflow. Some skills also
    have companion resources (reference docs, scripts, configs) listed in
    the manifest that you can fetch with the resource_path parameter.

    Args:
        skill_name: Name of the skill (e.g. "pr-review", "mcp-builder")
        resource_path: Optional path to a companion resource file
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
        return {
            "skill_name": skill_name,
            "error": f"HTTP {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error("Failed to get skill content: %s", e)
        return {"skill_name": skill_name, "error": str(e), "status": "failed"}


@mcp.tool()
@track_tool()
async def search_registry(
    query: str,
    max_results: int = 10,
    include_discovery_receipt: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Discover AI assets (MCP servers, tools, agents, skills) by describing
    what you need. Use this as your first step when you need a capability
    you don't currently have.

    Results include connection details so you can use the discovered assets:
    - Servers: have an endpoint_url field you can connect to directly as an
      MCP server (e.g. add to mcp.json or claude_desktop_config.json)
    - Tools: individual capabilities within servers, with inputSchema
    - Agents: autonomous agents with a URL you can delegate tasks to
    - Skills: workflow instructions (use get_skill_content to fetch the full markdown)

    When a useful MCP server is found, use the endpoint_url to add it to
    the AI assistant's MCP configuration so its tools become available.
    For Claude Code, run:
      claude mcp add --transport http --scope user <server-name> <endpoint_url>
    Note: adding a server usually requires restarting the AI assistant
    session for the new tools to take effect.

    Examples:
        "search documentation" -> finds doc search servers
        "book flights hotels" -> finds travel booking tools
        "code review" -> finds PR review skills and agents

    Args:
        query: What capability or tool you are looking for (natural language)
        max_results: Number of results to return (default: 10, max: 50)
        include_discovery_receipt: Include compact eval metadata about exposed and withheld results

    Returns:
        Dictionary with servers, tools, agents, skills arrays and metadata
    """
    logger.info(f"search_registry called: query={query}, max_results={max_results}")

    try:
        query = _validate_query(query)
        max_results = _validate_top_n(max_results)
        headers = await _get_registry_headers(ctx)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{REGISTRY_URL}/api/search/semantic",
                headers=headers,
                json={
                    "query": query,
                    "entity_types": [
                        "mcp_server",
                        "tool",
                        "a2a_agent",
                        "skill",
                        "virtual_server",
                    ],
                    "max_results": max_results,
                },
            )
            response.raise_for_status()
            data = response.json()

        servers = data.get("servers", []) if isinstance(data, dict) else []
        tools = data.get("tools", []) if isinstance(data, dict) else []
        agents = data.get("agents", []) if isinstance(data, dict) else []
        skills = data.get("skills", []) if isinstance(data, dict) else []

        candidate_results = []
        for tool in tools:
            candidate_results.append(
                {
                    "asset_type": "tool",
                    "service_path": tool.get("server_path") or tool.get("path") or "",
                    "name": tool.get("tool_name") or tool.get("name") or "",
                    "similarity_score": tool.get("relevance_score") or tool.get("score"),
                }
            )
        for server in servers:
            server_path = server.get("path", "")
            for tool in server.get("matching_tools", []):
                candidate_results.append(
                    {
                        "asset_type": "tool",
                        "service_path": server_path,
                        "name": tool.get("tool_name", ""),
                        "similarity_score": tool.get("relevance_score"),
                    }
                )
        for agent in agents:
            candidate_results.append(
                {
                    "asset_type": "agent",
                    "service_path": agent.get("path") or agent.get("url") or "",
                    "name": agent.get("agent_name") or agent.get("name") or "",
                    "similarity_score": agent.get("relevance_score") or agent.get("score"),
                }
            )
        for skill in skills:
            candidate_results.append(
                {
                    "asset_type": "skill",
                    "service_path": skill.get("path") or skill.get("url") or "",
                    "name": skill.get("skill_name") or skill.get("name") or "",
                    "similarity_score": skill.get("relevance_score") or skill.get("score"),
                }
            )
        # Dedupe so a tool returned in both tools[] and a server's matching_tools
        # is counted once. Then split into what the caller saw vs what the limit
        # held back.
        candidate_results = _dedupe_candidates(candidate_results)
        exposed_results = candidate_results[:max_results]
        withheld_results = candidate_results[max_results:]

        total_results = len(servers) + len(tools) + len(agents) + len(skills)
        result = {
            "servers": servers,
            "tools": tools,
            "agents": agents,
            "skills": skills,
            "query": query,
            "total_results": total_results,
            "status": "success",
        }
        if include_discovery_receipt:
            result["discovery_receipt"] = _build_discovery_receipt(
                query=query,
                limit=max_results,
                exposed_results=exposed_results,
                withheld_results=withheld_results,
                status="success",
                stop_reason="results_returned" if total_results else "no_match",
            )
        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "query": query,
            "total_results": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to search registry: {e}")
        return {
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
@track_tool()
async def intelligent_tool_finder(
    query: str,
    top_n: int = 5,
    include_discovery_receipt: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    DEPRECATED: Use search_registry instead. This tool will be removed in v1.26.0.

    Search for tools using natural language semantic search.

    Args:
        query: Natural language description of what you want to do
        top_n: Number of results to return (default: 5, max: 50)
        include_discovery_receipt: Include compact eval metadata about exposed and withheld results

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
                    "entity_types": ["mcp_server", "tool", "a2a_agent", "skill", "virtual_server"],
                    "max_results": top_n,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract servers array from response
        servers = data.get("servers", []) if isinstance(data, dict) else []

        # Flatten matching_tools from all servers into ToolSearchResult objects
        result_list = []
        candidate_results = []
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
                candidate_results.append(
                    {
                        "asset_type": "tool",
                        "service_path": server_path,
                        "name": tool.get("tool_name", ""),
                        "similarity_score": tool.get("relevance_score"),
                    }
                )

        # Enforce client-side limit (safety net in case registry returns more)
        result_list = result_list[:top_n]
        result = {
            "results": result_list,
            "query": query,
            "total_results": len(result_list),
            "status": "success",
        }
        if include_discovery_receipt:
            # Dedupe before splitting so a duplicate tool isn't reported as withheld.
            candidate_results = _dedupe_candidates(candidate_results)
            result["discovery_receipt"] = _build_discovery_receipt(
                query=query,
                limit=top_n,
                exposed_results=candidate_results[:top_n],
                withheld_results=candidate_results[top_n:],
                status="success",
                stop_reason="results_returned" if result_list else "no_match",
            )
        return result

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
@track_tool()
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
        logger.info(f"Running in HTTP mode on {host}:{port} (stateless=True)")
        mcp.run(
            transport="streamable-http",
            host=host,
            port=int(port),
            stateless_http=True,
        )
    else:
        logger.info("Running in stdio mode")
        mcp.run(transport="stdio")
