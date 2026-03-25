import logging
import re
from typing import (
    Annotated,
    Literal,
)

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..audit import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..core.config import DeploymentMode, RegistryMode, settings
from ..repositories.factory import get_search_repository
from ..repositories.interfaces import SearchRepositoryBase
from ..services.agent_service import agent_service
from ..services.server_service import server_service
from ..services.virtual_server_service import get_virtual_server_service

logger = logging.getLogger(__name__)

router = APIRouter()

EntityType = Literal["mcp_server", "tool", "a2a_agent", "skill", "virtual_server"]


def get_search_repo() -> SearchRepositoryBase:
    """Dependency injection function for search repository."""
    return get_search_repository()


class MatchingToolResult(BaseModel):
    """Tool matching result with optional schema for display."""

    tool_name: str
    description: str | None = None
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    match_context: str | None = None
    inputSchema: dict | None = Field(
        default=None, description="JSON Schema for tool input parameters"
    )


class SyncMetadata(BaseModel):
    """Metadata for items synced from peer registries."""

    is_federated: bool = False
    source_peer_id: str | None = None
    synced_at: str | None = None
    original_path: str | None = None
    is_orphaned: bool = False
    orphaned_at: str | None = None
    is_read_only: bool = True


def _compute_endpoint_url(
    path: str,
    proxy_pass_url: str | None,
    mcp_endpoint: str | None,
    base_url: str | None,
) -> str | None:
    """Compute the endpoint URL for an MCP server.

    URL determination with fallback chain:
    1. mcp_endpoint (custom override) - always takes precedence
    2. proxy_pass_url (in registry-only mode)
    3. Constructed gateway URL (default/fallback in with-gateway mode)

    Args:
        path: Server path (e.g., /context7)
        proxy_pass_url: Internal backend URL
        mcp_endpoint: Custom endpoint override
        base_url: Base URL from request (e.g., https://mcpgateway.ddns.net)

    Returns:
        The computed endpoint URL, or None if not determinable
    """
    # Priority 1: Explicit mcp_endpoint override
    if mcp_endpoint:
        return mcp_endpoint

    # Priority 2: In registry-only mode, use proxy_pass_url directly
    if settings.deployment_mode == DeploymentMode.REGISTRY_ONLY:
        return proxy_pass_url

    # Priority 3: Construct gateway URL
    if base_url:
        clean_path = path.rstrip("/")
        if not clean_path.startswith("/"):
            clean_path = f"/{clean_path}"
        return f"{base_url}{clean_path}/mcp"

    # Fallback: return proxy_pass_url if nothing else works
    return proxy_pass_url


class ServerSearchResult(BaseModel):
    path: str
    server_name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    num_tools: int = 0
    is_enabled: bool = False
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_context: str | None = None
    matching_tools: list[MatchingToolResult] = Field(default_factory=list)
    sync_metadata: SyncMetadata | None = None
    # Endpoint URL for agent connectivity (computed based on deployment mode)
    endpoint_url: str | None = Field(
        default=None, description="URL for agents to connect to this MCP server"
    )
    # Raw endpoint fields (for advanced use cases)
    proxy_pass_url: str | None = Field(
        default=None, description="Base URL for the MCP server backend (internal)"
    )
    mcp_endpoint: str | None = Field(
        default=None, description="Explicit streamable-http endpoint URL (if set)"
    )
    sse_endpoint: str | None = Field(default=None, description="Explicit SSE endpoint URL (if set)")
    supported_transports: list[str] = Field(
        default_factory=list, description="Supported transport types (e.g., streamable-http, sse)"
    )
    trust_verified: str = Field(
        default="none",
        description="Trust verification status: none, verified, expired, revoked, not_found, pending",
    )


class ToolSearchResult(BaseModel):
    server_path: str
    server_name: str
    tool_name: str
    description: str | None = None
    inputSchema: dict | None = Field(default=None, description="JSON Schema for tool input")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_context: str | None = None
    # Endpoint URL for the parent MCP server
    endpoint_url: str | None = Field(
        default=None, description="URL for agents to connect to the parent MCP server"
    )


class AgentSearchResult(BaseModel):
    """Agent search result with minimal top-level fields to avoid duplication.

    Only search-specific fields are at the top level. All agent details
    (name, description, url, skills, etc.) are in the agent_card.
    """

    path: str = Field(..., description="Agent path for identification")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_context: str | None = None
    agent_card: dict = Field(..., description="Full agent card with all details")
    trust_verified: str = Field(
        default="none",
        description="Trust verification status: none, verified, expired, revoked, not_found, pending",
    )


class SkillSearchResult(BaseModel):
    path: str
    skill_name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    skill_md_url: str | None = None
    skill_md_raw_url: str | None = None
    version: str | None = None
    author: str | None = None
    visibility: str | None = None
    owner: str | None = None
    is_enabled: bool = False
    health_status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    last_checked_time: str | None = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_context: str | None = None


class VirtualServerSearchResult(BaseModel):
    path: str
    server_name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    num_tools: int = 0
    backend_count: int = 0
    backend_paths: list[str] = Field(default_factory=list)
    is_enabled: bool = False
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_context: str | None = None
    matching_tools: list[MatchingToolResult] = Field(default_factory=list)
    # Endpoint URL for agent connectivity (computed based on deployment mode)
    endpoint_url: str | None = Field(
        default=None, description="URL for agents to connect to this virtual MCP server"
    )


class SemanticSearchRequest(BaseModel):
    query: str = Field(
        default="",
        max_length=512,
        description="Natural language query. Can be empty when filtering by tags only.",
    )
    entity_types: list[EntityType] | None = Field(
        default=None, description="Optional entity filters"
    )
    tags: list[str] | None = Field(
        default=None,
        description="Exact tag filters. Only return entities that have ALL specified tags.",
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum results per entity collection"
    )


class SemanticSearchResponse(BaseModel):
    query: str
    search_mode: str = Field(
        default="hybrid", description="Search mode: 'hybrid' (semantic+lexical) or 'lexical-only'"
    )
    servers: list[ServerSearchResult] = Field(default_factory=list)
    tools: list[ToolSearchResult] = Field(default_factory=list)
    agents: list[AgentSearchResult] = Field(default_factory=list)
    skills: list[SkillSearchResult] = Field(default_factory=list)
    virtual_servers: list[VirtualServerSearchResult] = Field(default_factory=list)
    total_servers: int = 0
    total_tools: int = 0
    total_agents: int = 0
    total_skills: int = 0
    total_virtual_servers: int = 0


async def _get_tool_schema_for_virtual_server(
    vs_path: str,
    tool_name: str,
) -> dict | None:
    """Look up tool schema from backend server for a virtual server's tool.

    Args:
        vs_path: Virtual server path
        tool_name: Name of the tool to look up (can be the original name or alias)

    Returns:
        Tool inputSchema dict if found, None otherwise
    """
    try:
        vs_service = get_virtual_server_service()
        vs_config = await vs_service.get_virtual_server(vs_path)

        if not vs_config:
            return None

        # Find the tool mapping for this tool (check both tool_name and alias)
        tool_mapping = None
        for tm in vs_config.tool_mappings:
            if tm.tool_name == tool_name or tm.alias == tool_name:
                tool_mapping = tm
                break

        if not tool_mapping:
            return None

        # Get the backend server info
        backend_path = tool_mapping.backend_server_path
        if tool_mapping.backend_version:
            backend_path = f"{backend_path}:{tool_mapping.backend_version}"

        server_info = await server_service.get_server_info(backend_path)
        if not server_info:
            return None

        # Find the tool in the backend's tool list using the original tool name
        tool_list = server_info.get("tool_list", [])
        for tool in tool_list:
            if tool.get("name") == tool_mapping.tool_name:
                return tool.get("schema") or tool.get("inputSchema")

        return None
    except Exception as e:
        logger.warning(f"Failed to get tool schema for {vs_path}/{tool_name}: {e}")
        return None


async def _user_can_access_server(path: str, server_name: str, user_context: dict) -> bool:
    """Validate whether the current user can view the specified server."""
    if user_context.get("is_admin"):
        return True

    accessible_servers = user_context.get("accessible_servers") or []
    if "all" in accessible_servers:
        return True

    if not accessible_servers:
        return False

    try:
        if await server_service.user_can_access_server_path(path, accessible_servers):
            return True
    except Exception:
        # Fall through to string comparisons if server lookup failed
        logger.debug("Unable to validate server path via service for %s", path, exc_info=True)

    technical_name = path.strip("/")
    return technical_name in accessible_servers or (
        server_name and server_name in accessible_servers
    )


async def _user_can_access_agent(agent_path: str, user_context: dict) -> bool:
    """Validate user access for a given agent."""
    if user_context.get("is_admin"):
        return True

    accessible_agents = user_context.get("accessible_agents") or []
    if "all" not in accessible_agents and agent_path not in accessible_agents:
        return False

    agent_card = await agent_service.get_agent_info(agent_path)
    if not agent_card:
        return False

    if agent_card.visibility == "public":
        return True

    if agent_card.visibility == "internal":
        return agent_card.registered_by == user_context.get("username")

    if agent_card.visibility == "group-restricted":
        allowed_groups = set(agent_card.allowed_groups)
        user_groups = set(user_context.get("groups", []))
        return bool(allowed_groups & user_groups)

    return False


async def _user_can_access_skill(
    skill_path: str,
    visibility: str,
    owner: str,
    allowed_groups: list,
    user_context: dict,
) -> bool:
    """Validate user access for a given skill based on visibility.

    Args:
        skill_path: The skill path
        visibility: Skill visibility (public, private, group)
        owner: Skill owner username
        allowed_groups: Groups allowed to access the skill
        user_context: User context with username, groups, is_admin

    Returns:
        True if user can access the skill, False otherwise
    """
    if user_context.get("is_admin"):
        return True

    if visibility == "public":
        return True

    if visibility == "private":
        return owner == user_context.get("username")

    if visibility == "group":
        user_groups = set(user_context.get("groups", []))
        skill_groups = set(allowed_groups or [])
        return bool(user_groups & skill_groups)

    return False


def _compute_trust_verified(
    ans_metadata: dict | None,
) -> str:
    """Derive the trust_verified label from ANS metadata.

    Args:
        ans_metadata: ANS metadata dict from the agent card, or None.

    Returns:
        "none" when there is no ANS metadata, otherwise the ANS status
        value (verified, expired, revoked, not_found, pending).
    """
    if not ans_metadata:
        return "none"

    return ans_metadata.get("status", "none")


_HASHTAG_PATTERN = re.compile(r"#([\w-]+)")


def _parse_hashtags(
    query: str,
) -> tuple[str, list[str]]:
    """Extract #tag tokens from the query string.

    Returns:
        Tuple of (remaining_query, extracted_tags).
        Tags are lowercased for case-insensitive matching.
    """
    tags = [m.group(1).lower() for m in _HASHTAG_PATTERN.finditer(query)]
    remaining = _HASHTAG_PATTERN.sub("", query).strip()
    # Collapse multiple spaces left after removal
    remaining = re.sub(r"\s+", " ", remaining).strip()
    return remaining, tags


def _entity_has_all_tags(
    entity_tags: list[str],
    required_tags: list[str],
) -> bool:
    """Check if an entity has ALL required tags (case-insensitive)."""
    entity_tags_lower = {t.lower() for t in entity_tags}
    return all(tag in entity_tags_lower for tag in required_tags)


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    summary="Unified semantic search for MCP servers and tools",
)
async def semantic_search(
    http_request: Request,
    request: SemanticSearchRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    search_repo: SearchRepositoryBase = Depends(get_search_repo),
) -> SemanticSearchResponse:
    """
    Run a semantic search against MCP servers (and their tools) using FAISS embeddings.
    """
    # Parse #tag tokens from query for exact tag matching
    search_query, hashtag_tags = _parse_hashtags(request.query)

    # Merge hashtag-extracted tags with explicitly provided tags
    required_tags: list[str] = list(
        {t.lower() for t in (request.tags or []) + hashtag_tags}
    )

    # Set audit action for search
    set_audit_action(
        http_request, "search", "search", description=f"Semantic search: {request.query[:50]}..."
    )

    logger.info(
        "Semantic search requested by %s (entities=%s, max=%s, tags=%s)",
        user_context.get("username"),
        request.entity_types or ["mcp_server", "tool"],
        request.max_results,
        required_tags or None,
    )

    # Determine the effective text query after removing #tag tokens
    effective_query = search_query.strip()

    # Validate: must have either a text query or tags
    if not effective_query and not required_tags:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a search query, tags, or both.",
        )

    try:
        if not effective_query and required_tags:
            # Tag-only search: query DB directly by tags
            raw_results = await search_repo.search_by_tags(
                tags=required_tags,
                entity_types=request.entity_types,
                max_results=request.max_results,
            )
        else:
            # Text search (possibly combined with tags filtered after)
            raw_results = await search_repo.search(
                query=effective_query,
                entity_types=request.entity_types,
                max_results=request.max_results,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("FAISS search service unavailable: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is temporarily unavailable. Please try again later.",
        ) from exc

    # Extract base URL from request for endpoint URL computation
    # Use X-Forwarded-Proto and X-Forwarded-Host if behind proxy, otherwise use request URL
    forwarded_proto = http_request.headers.get("x-forwarded-proto", "https")
    forwarded_host = http_request.headers.get("x-forwarded-host") or http_request.headers.get(
        "host"
    )
    if forwarded_host:
        base_url = f"{forwarded_proto}://{forwarded_host}"
    else:
        base_url = str(http_request.base_url).rstrip("/")

    filtered_servers: list[ServerSearchResult] = []
    for server in raw_results.get("servers", []):
        if not await _user_can_access_server(
            server.get("path", ""),
            server.get("server_name", ""),
            user_context,
        ):
            continue

        matching_tools = [
            MatchingToolResult(
                tool_name=tool.get("tool_name", ""),
                description=tool.get("description"),
                relevance_score=tool.get("relevance_score", 0.0),
                match_context=tool.get("match_context"),
            )
            for tool in server.get("matching_tools", [])
        ]

        # Parse sync_metadata if present
        raw_sync = server.get("sync_metadata")
        sync_meta = SyncMetadata(**raw_sync) if raw_sync else None

        # Compute endpoint URL based on deployment mode
        server_path = server.get("path", "")
        server_proxy_url = server.get("proxy_pass_url")
        server_mcp_endpoint = server.get("mcp_endpoint")
        endpoint_url = _compute_endpoint_url(
            path=server_path,
            proxy_pass_url=server_proxy_url,
            mcp_endpoint=server_mcp_endpoint,
            base_url=base_url,
        )

        # Look up ANS metadata from server info for trust verification
        server_full_info = await server_service.get_server_info(server_path)
        server_ans_meta = server_full_info.get("ans_metadata") if server_full_info else None
        server_trust = _compute_trust_verified(server_ans_meta)

        filtered_servers.append(
            ServerSearchResult(
                path=server_path,
                server_name=server.get("server_name", ""),
                description=server.get("description"),
                tags=server.get("tags", []),
                num_tools=server.get("num_tools", 0),
                is_enabled=server.get("is_enabled", False),
                relevance_score=server.get("relevance_score", 0.0),
                match_context=server.get("match_context"),
                matching_tools=matching_tools,
                sync_metadata=sync_meta,
                endpoint_url=endpoint_url,
                proxy_pass_url=server_proxy_url,
                mcp_endpoint=server_mcp_endpoint,
                sse_endpoint=server.get("sse_endpoint"),
                supported_transports=server.get("supported_transports", []),
                trust_verified=server_trust,
            )
        )

    # Build a map of server_path -> endpoint_url for tool results
    server_endpoint_map: dict[str, str | None] = {
        server.path: server.endpoint_url for server in filtered_servers
    }

    filtered_tools: list[ToolSearchResult] = []
    for tool in raw_results.get("tools", []):
        server_path = tool.get("server_path", "")
        server_name = tool.get("server_name", "")
        if not await _user_can_access_server(server_path, server_name, user_context):
            continue

        # Get endpoint_url from filtered servers, or compute it if not available
        tool_endpoint_url = server_endpoint_map.get(server_path)
        if tool_endpoint_url is None:
            # Server not in filtered results, compute endpoint_url
            tool_endpoint_url = _compute_endpoint_url(
                path=server_path,
                proxy_pass_url=None,  # We don't have this info for tools
                mcp_endpoint=None,
                base_url=base_url,
            )

        filtered_tools.append(
            ToolSearchResult(
                server_path=server_path,
                server_name=server_name,
                tool_name=tool.get("tool_name", ""),
                description=tool.get("description"),
                inputSchema=tool.get("inputSchema"),
                relevance_score=tool.get("relevance_score", 0.0),
                match_context=tool.get("match_context"),
                endpoint_url=tool_endpoint_url,
            )
        )

    filtered_agents: list[AgentSearchResult] = []
    for agent in raw_results.get("agents", []):
        agent_path = agent.get("path", "")
        if not agent_path:
            continue

        if not await _user_can_access_agent(agent_path, user_context):
            continue

        agent_card_obj = await agent_service.get_agent_info(agent_path)
        agent_card_dict = (
            agent_card_obj.model_dump() if agent_card_obj else agent.get("agent_card", {})
        )

        # Ensure agent_card has the path for consistency
        if agent_card_dict and "path" not in agent_card_dict:
            agent_card_dict["path"] = agent_path

        # Compute trust verification status from ANS metadata
        ans_meta = agent_card_dict.get("ans_metadata") if agent_card_dict else None
        trust_verified = _compute_trust_verified(ans_meta)

        filtered_agents.append(
            AgentSearchResult(
                path=agent_path,
                relevance_score=agent.get("relevance_score", 0.0),
                match_context=agent.get("match_context") or agent_card_dict.get("description"),
                agent_card=agent_card_dict or {},
                trust_verified=trust_verified,
            )
        )

    filtered_skills: list[SkillSearchResult] = []
    for skill in raw_results.get("skills", []):
        skill_path = skill.get("path", "")
        if not skill_path:
            continue

        visibility = skill.get("visibility", "public")
        owner = skill.get("owner", "")
        allowed_groups = skill.get("allowed_groups", [])

        if not await _user_can_access_skill(
            skill_path, visibility, owner, allowed_groups, user_context
        ):
            continue

        filtered_skills.append(
            SkillSearchResult(
                path=skill_path,
                skill_name=skill.get("skill_name", skill_path.strip("/")),
                description=skill.get("description"),
                tags=skill.get("tags", []),
                skill_md_url=skill.get("skill_md_url"),
                skill_md_raw_url=skill.get("skill_md_raw_url"),
                version=skill.get("version"),
                author=skill.get("author"),
                visibility=visibility,
                owner=owner,
                is_enabled=skill.get("is_enabled", False),
                health_status=skill.get("health_status", "unknown"),
                last_checked_time=skill.get("last_checked_time"),
                relevance_score=skill.get("relevance_score", 0.0),
                match_context=skill.get("match_context"),
            )
        )

    # Process virtual servers
    filtered_virtual_servers: list[VirtualServerSearchResult] = []
    for vs in raw_results.get("virtual_servers", []):
        vs_path = vs.get("path", "")
        if not vs_path:
            continue

        # Virtual servers use the same access control as regular servers
        if not await _user_can_access_server(
            vs_path,
            vs.get("server_name", ""),
            user_context,
        ):
            continue

        # Build matching tools with schema lookup from backend servers
        # Only include tools that matched the search query
        matching_tools: list[MatchingToolResult] = []
        for tool in vs.get("matching_tools", []):
            tool_name = tool.get("tool_name", "")
            # Look up the tool schema from the backend server
            input_schema = await _get_tool_schema_for_virtual_server(vs_path, tool_name)
            matching_tools.append(
                MatchingToolResult(
                    tool_name=tool_name,
                    description=tool.get("description"),
                    relevance_score=tool.get("relevance_score", 0.0),
                    match_context=tool.get("match_context"),
                    inputSchema=input_schema,
                )
            )

        # Compute endpoint URL for virtual server
        vs_endpoint_url = _compute_endpoint_url(
            path=vs_path,
            proxy_pass_url=None,  # Virtual servers don't have proxy_pass_url
            mcp_endpoint=None,
            base_url=base_url,
        )

        metadata = vs.get("metadata", {})
        filtered_virtual_servers.append(
            VirtualServerSearchResult(
                path=vs_path,
                server_name=vs.get("server_name", ""),
                description=vs.get("description"),
                tags=vs.get("tags", []),
                num_tools=metadata.get("num_tools", 0),
                backend_count=metadata.get("backend_count", 0),
                backend_paths=metadata.get("backend_paths", []),
                is_enabled=vs.get("is_enabled", False),
                relevance_score=vs.get("relevance_score", 0.0),
                match_context=vs.get("match_context"),
                matching_tools=matching_tools,
                endpoint_url=vs_endpoint_url,
            )
        )

    # Filter results by required tags (from #tag syntax or explicit tags param)
    if required_tags:
        filtered_servers = [
            s for s in filtered_servers if _entity_has_all_tags(s.tags, required_tags)
        ]
        filtered_tools = [
            t for t in filtered_tools
            if _entity_has_all_tags(
                # Tools don't have tags directly; filter by parent server tags
                next(
                    (s.tags for s in filtered_servers if s.path == t.server_path),
                    [],
                ),
                required_tags,
            )
        ]
        filtered_agents = [
            a for a in filtered_agents
            if _entity_has_all_tags(
                a.agent_card.get("tags", []),
                required_tags,
            )
        ]
        filtered_skills = [
            s for s in filtered_skills if _entity_has_all_tags(s.tags, required_tags)
        ]
        filtered_virtual_servers = [
            vs for vs in filtered_virtual_servers if _entity_has_all_tags(vs.tags, required_tags)
        ]

    # Filter results based on registry mode
    # In skills-only mode, only return skills; in servers-only mode, only return servers, etc.
    mode = settings.registry_mode

    if mode == RegistryMode.SKILLS_ONLY:
        # Only skills are enabled
        filtered_servers = []
        filtered_tools = []
        filtered_agents = []
        filtered_virtual_servers = []
    elif mode == RegistryMode.MCP_SERVERS_ONLY:
        # Only servers, tools, and virtual servers are enabled
        filtered_agents = []
        filtered_skills = []
    elif mode == RegistryMode.AGENTS_ONLY:
        # Only agents are enabled
        filtered_servers = []
        filtered_tools = []
        filtered_skills = []
        filtered_virtual_servers = []
    # In FULL mode, return all results (no filtering needed)

    return SemanticSearchResponse(
        query=request.query.strip(),
        servers=filtered_servers,
        tools=filtered_tools,
        agents=filtered_agents,
        skills=filtered_skills,
        virtual_servers=filtered_virtual_servers,
        total_servers=len(filtered_servers),
        total_tools=len(filtered_tools),
        total_agents=len(filtered_agents),
        total_skills=len(filtered_skills),
        total_virtual_servers=len(filtered_virtual_servers),
    )


@router.get(
    "/tags",
    summary="Get all unique tags across all indexed entities",
)
async def get_all_tags(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    search_repo: SearchRepositoryBase = Depends(get_search_repo),
) -> dict[str, list[str]]:
    """Return a sorted list of all unique tags across servers, agents, skills, and virtual servers."""
    try:
        tags = await search_repo.get_all_tags()
        return {"tags": tags}
    except Exception as e:
        logger.error("Failed to retrieve tags: %s", e, exc_info=True)
        return {"tags": []}
