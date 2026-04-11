"""
Federation configuration API routes.

Provides endpoints to manage federation configurations.
"""

import logging
from datetime import UTC
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..audit import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..repositories.factory import get_federation_config_repository
from ..repositories.interfaces import FederationConfigRepositoryBase
from ..schemas.federation_schema import (
    AwsRegistryConfig,
    FederationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_federation_repo() -> FederationConfigRepositoryBase:
    """Get federation config repository dependency."""
    return get_federation_config_repository()


@router.get("/federation/config", tags=["federation"], summary="Get federation configuration")
async def get_federation_config(
    request: Request,
    config_id: str = "default",
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Get federation configuration by ID.

    Args:
        config_id: Configuration ID (default: "default")
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Federation configuration

    Raises:
        404: Configuration not found
    """
    # Set audit action for federation config read
    set_audit_action(
        request,
        "read",
        "federation",
        resource_id=config_id,
        description=f"Read federation config {config_id}",
    )

    logger.info(f"User {user_context['username']} retrieving federation config: {config_id}")

    config = await repo.get_config(config_id)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    return config.model_dump()


@router.post(
    "/federation/config",
    tags=["federation"],
    summary="Create or update federation configuration",
    status_code=status.HTTP_201_CREATED,
)
async def save_federation_config(
    request: Request,
    config: FederationConfig,
    config_id: str = "default",
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Create or update federation configuration.

    Args:
        config: Federation configuration to save
        config_id: Configuration ID (default: "default")
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Saved configuration

    Example:
        ```json
        {
          "anthropic": {
            "enabled": true,
            "endpoint": "https://registry.modelcontextprotocol.io",
            "sync_on_startup": false,
            "servers": [
              {"name": "io.github.jgador/websharp"},
              {"name": "modelcontextprotocol/filesystem"}
            ]
          },
          "asor": {
            "enabled": false,
            "endpoint": "",
            "auth_env_var": "ASOR_ACCESS_TOKEN",
            "sync_on_startup": false,
            "agents": []
          }
        }
        ```
    """
    # Set audit action for federation config create/update
    set_audit_action(
        request,
        "create",
        "federation",
        resource_id=config_id,
        description=f"Save federation config {config_id}",
    )

    logger.info(
        f"User {user_context['username']} saving federation config: {config_id} "
        f"(anthropic: {config.anthropic.enabled}, asor: {config.asor.enabled})"
    )

    try:
        saved_config = await repo.save_config(config, config_id)
        logger.info(f"Federation config saved successfully: {config_id}")

        # Reconcile: remove stale federated servers
        reconciliation_result = None
        try:
            from ..core.nginx_service import nginx_service
            from ..repositories.factory import get_server_repository
            from ..services.federation_reconciliation import reconcile_anthropic_servers
            from ..services.server_service import server_service

            server_repo = get_server_repository()
            reconciliation_result = await reconcile_anthropic_servers(
                config=saved_config,
                server_service=server_service,
                server_repo=server_repo,
                nginx_service=nginx_service,
                audit_username=user_context.get("username"),
            )
            if reconciliation_result.get("removed"):
                logger.info(
                    f"Reconciliation removed {reconciliation_result['removed_count']} stale servers: "
                    f"{reconciliation_result['removed']}"
                )
        except Exception as e:
            logger.error(f"Reconciliation failed (non-fatal): {e}")

        response = {
            "message": "Federation configuration saved successfully",
            "config_id": config_id,
            "config": saved_config.model_dump(),
        }
        if reconciliation_result:
            response["reconciliation"] = reconciliation_result

        return response

    except Exception as e:
        logger.error(f"Failed to save federation config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save federation config: {str(e)}",
        )


@router.put(
    "/federation/config/{config_id}",
    tags=["federation"],
    summary="Update specific federation configuration",
)
async def update_federation_config(
    request: Request,
    config_id: str,
    config: FederationConfig,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Update a specific federation configuration.

    Args:
        config_id: Configuration ID to update
        config: Updated federation configuration
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    # Set audit action for federation config update
    set_audit_action(
        request,
        "update",
        "federation",
        resource_id=config_id,
        description=f"Update federation config {config_id}",
    )

    logger.info(f"User {user_context['username']} updating federation config: {config_id}")

    try:
        saved_config = await repo.save_config(config, config_id)
        logger.info(f"Federation config updated successfully: {config_id}")

        # Reconcile: remove stale federated servers
        reconciliation_result = None
        try:
            from ..core.nginx_service import nginx_service
            from ..repositories.factory import get_server_repository
            from ..services.federation_reconciliation import reconcile_anthropic_servers
            from ..services.server_service import server_service

            server_repo = get_server_repository()
            reconciliation_result = await reconcile_anthropic_servers(
                config=saved_config,
                server_service=server_service,
                server_repo=server_repo,
                nginx_service=nginx_service,
                audit_username=user_context.get("username"),
            )
            if reconciliation_result.get("removed"):
                logger.info(
                    f"Reconciliation removed {reconciliation_result['removed_count']} stale servers: "
                    f"{reconciliation_result['removed']}"
                )
        except Exception as e:
            logger.error(f"Reconciliation failed (non-fatal): {e}")

        response = {
            "message": "Federation configuration updated successfully",
            "config_id": config_id,
            "config": saved_config.model_dump(),
        }
        if reconciliation_result:
            response["reconciliation"] = reconciliation_result

        return response

    except Exception as e:
        logger.error(f"Failed to update federation config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update federation config: {str(e)}",
        )


@router.delete(
    "/federation/config/{config_id}", tags=["federation"], summary="Delete federation configuration"
)
async def delete_federation_config(
    config_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, str]:
    """
    Delete a federation configuration.

    Args:
        config_id: Configuration ID to delete
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Deletion confirmation

    Raises:
        404: Configuration not found
    """
    logger.info(f"User {user_context['username']} deleting federation config: {config_id}")

    deleted = await repo.delete_config(config_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    logger.info(f"Federation config deleted successfully: {config_id}")
    return {
        "message": f"Federation configuration '{config_id}' deleted successfully",
        "config_id": config_id,
    }


@router.get(
    "/federation/configs", tags=["federation"], summary="List all federation configurations"
)
async def list_federation_configs(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    List all federation configurations.

    Args:
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        List of configuration summaries with id, created_at, updated_at
    """
    logger.info(f"User {user_context['username']} listing federation configs")

    configs = await repo.list_configs()

    return {"configs": configs, "total": len(configs)}


@router.post(
    "/federation/config/{config_id}/anthropic/servers",
    tags=["federation"],
    summary="Add Anthropic server to config",
)
async def add_anthropic_server(
    config_id: str,
    server_name: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Add a server to Anthropic federation configuration.

    Args:
        config_id: Configuration ID
        server_name: Server name to add (e.g., "io.github.jgador/websharp")
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    logger.info(f"User {user_context['username']} adding Anthropic server: {server_name}")

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Check if server already exists
    from ..schemas.federation_schema import AnthropicServerConfig

    for server in config.anthropic.servers:
        if server.name == server_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Server '{server_name}' already exists in configuration",
            )

    # Add new server
    config.anthropic.servers.append(AnthropicServerConfig(name=server_name))

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    return {
        "message": f"Server '{server_name}' added to Anthropic configuration",
        "config": saved_config.model_dump(),
    }


@router.delete(
    "/federation/config/{config_id}/anthropic/servers/{server_name:path}",
    tags=["federation"],
    summary="Remove Anthropic server from config",
)
async def remove_anthropic_server(
    config_id: str,
    server_name: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Remove a server from Anthropic federation configuration.

    Also removes the server from mcp_servers_default if it was
    previously synced.

    Args:
        config_id: Configuration ID
        server_name: Server name to remove (e.g., "io.github.jgador/websharp")
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration with removal details
    """
    logger.info(f"User {user_context['username']} removing Anthropic server: {server_name}")

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Find and remove server from config
    original_count = len(config.anthropic.servers)
    config.anthropic.servers = [s for s in config.anthropic.servers if s.name != server_name]

    if len(config.anthropic.servers) == original_count:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_name}' not found in configuration",
        )

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    # Remove the server from mcp_servers_default if it exists
    server_path = f"/{server_name.replace('/', '-')}"
    server_removed = False
    try:
        from ..services.server_service import server_service

        server_info = await server_service.get_server_info(server_path)
        if server_info and server_info.get("source") == "anthropic":
            server_removed = await server_service.remove_server(server_path)
            if server_removed:
                logger.info(
                    f"Removed server '{server_name}' from mcp_servers_default ({server_path})"
                )

                # Regenerate nginx config
                from ..core.nginx_service import nginx_service

                all_servers = await server_service.get_all_servers(
                    include_inactive=False,
                )
                enabled_servers = {
                    p: info for p, info in all_servers.items() if info.get("is_enabled", False)
                }
                await nginx_service.generate_config_async(enabled_servers)
    except Exception as e:
        logger.error(f"Failed to remove server from mcp_servers_default: {e}")

    return {
        "message": f"Server '{server_name}' removed from Anthropic configuration",
        "config": saved_config.model_dump(),
        "server_removed_from_registry": server_removed,
    }


@router.post(
    "/federation/config/{config_id}/asor/agents",
    tags=["federation"],
    summary="Add ASOR agent to config",
)
async def add_asor_agent(
    config_id: str,
    agent_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Add an agent to ASOR federation configuration.

    Args:
        config_id: Configuration ID
        agent_id: Agent ID to add (e.g., "aws_assistant")
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    logger.info(f"User {user_context['username']} adding ASOR agent: {agent_id}")

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Check if agent already exists
    from ..schemas.federation_schema import AsorAgentConfig

    for agent in config.asor.agents:
        if agent.id == agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent '{agent_id}' already exists in configuration",
            )

    # Add new agent
    config.asor.agents.append(AsorAgentConfig(id=agent_id))

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    return {
        "message": f"Agent '{agent_id}' added to ASOR configuration",
        "config": saved_config.model_dump(),
    }


@router.delete(
    "/federation/config/{config_id}/asor/agents/{agent_id}",
    tags=["federation"],
    summary="Remove ASOR agent from config",
)
async def remove_asor_agent(
    config_id: str,
    agent_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Remove an agent from ASOR federation configuration.

    Args:
        config_id: Configuration ID
        agent_id: Agent ID to remove
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    logger.info(f"User {user_context['username']} removing ASOR agent: {agent_id}")

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Find and remove agent
    original_count = len(config.asor.agents)
    config.asor.agents = [a for a in config.asor.agents if a.id != agent_id]

    if len(config.asor.agents) == original_count:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found in configuration",
        )

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    return {
        "message": f"Agent '{agent_id}' removed from ASOR configuration",
        "config": saved_config.model_dump(),
    }


@router.post(
    "/federation/config/{config_id}/aws_registry/registries",
    tags=["federation"],
    summary="Add AWS registry to config",
)
async def add_aws_registry(
    request: Request,
    config_id: str,
    registry_config: AwsRegistryConfig,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Add a registry to AWS Registry federation configuration.

    Args:
        config_id: Configuration ID
        registry_config: AWS registry configuration (JSON body)
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    set_audit_action(
        request,
        "create",
        "federation",
        resource_id=config_id,
        description=f"Add AWS registry {registry_config.registry_id}",
    )

    logger.info(
        f"User {user_context['username']} adding AWS registry: {registry_config.registry_id}"
    )

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Check if registry already exists
    for reg in config.aws_registry.registries:
        if reg.registry_id == registry_config.registry_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registry '{registry_config.registry_id}' already exists in configuration",
            )

    # Add new registry
    config.aws_registry.registries.append(registry_config)

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    return {
        "message": f"Registry '{registry_config.registry_id}' added to AWS Registry configuration",
        "config": saved_config.model_dump(),
    }


@router.delete(
    "/federation/config/{config_id}/aws_registry/registries/{registry_id:path}",
    tags=["federation"],
    summary="Remove AWS registry from config",
)
async def remove_aws_registry(
    request: Request,
    config_id: str,
    registry_id: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Remove a registry from AWS Registry federation configuration.

    Args:
        config_id: Configuration ID
        registry_id: Registry ID to remove (e.g., ARN)
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Updated configuration
    """
    set_audit_action(
        request,
        "delete",
        "federation",
        resource_id=config_id,
        description=f"Remove AWS registry {registry_id}",
    )

    logger.info(f"User {user_context['username']} removing AWS registry: {registry_id}")

    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    # Find and remove registry
    original_count = len(config.aws_registry.registries)
    config.aws_registry.registries = [
        r for r in config.aws_registry.registries
        if r.registry_id != registry_id
    ]

    if len(config.aws_registry.registries) == original_count:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Registry '{registry_id}' not found in configuration",
        )

    # Save updated config
    saved_config = await repo.save_config(config, config_id)

    # Deregister all entities synced from this registry
    cleanup = await _deregister_entities_from_registry(registry_id)

    total = cleanup["servers_count"] + cleanup["agents_count"] + cleanup["skills_count"]
    message = f"Registry '{registry_id}' removed from AWS Registry configuration"
    if total > 0:
        parts = []
        if cleanup["servers_count"]:
            parts.append(f"{cleanup['servers_count']} server(s)")
        if cleanup["agents_count"]:
            parts.append(f"{cleanup['agents_count']} agent(s)")
        if cleanup["skills_count"]:
            parts.append(f"{cleanup['skills_count']} skill(s)")
        message += f" and {', '.join(parts)} deregistered"

    return {
        "message": message,
        "deregistered": cleanup,
        "config": saved_config.model_dump(),
    }


async def _deregister_entities_from_registry(
    registry_id: str,
) -> dict[str, Any]:
    """
    Find and remove all servers, agents, and skills synced from a specific AWS registry.

    Matches entities where metadata.agentcore_registry_id equals the given registry_id.

    Args:
        registry_id: The AWS registry ARN to match

    Returns:
        Dict with deregistered servers, agents, skills lists and counts
    """
    servers = await _deregister_servers_from_registry(registry_id)
    agents = await _deregister_agents_from_registry(registry_id)
    skills = await _deregister_skills_from_registry(registry_id)

    return {
        "servers": servers,
        "servers_count": len(servers),
        "agents": agents,
        "agents_count": len(agents),
        "skills": skills,
        "skills_count": len(skills),
    }


async def _deregister_servers_from_registry(
    registry_id: str,
) -> list[str]:
    """
    Remove all servers synced from a specific AWS registry.

    Args:
        registry_id: The AWS registry ARN to match

    Returns:
        List of server paths that were deregistered
    """
    from ..repositories.factory import get_server_repository
    from ..services.server_service import server_service

    server_repo = get_server_repository()
    all_agentcore_servers = await server_repo.list_by_source("agentcore")

    deregistered = []
    for path, server_info in all_agentcore_servers.items():
        metadata = server_info.get("metadata", {})
        if metadata.get("agentcore_registry_id") == registry_id:
            try:
                removed = await server_service.remove_server(path)
                if removed:
                    deregistered.append(path)
                    logger.info(f"Deregistered server {path} from registry {registry_id}")
            except Exception as e:
                logger.error(f"Failed to deregister server {path}: {e}")

    logger.info(f"Deregistered {len(deregistered)} server(s) from registry {registry_id}")
    return deregistered


async def _deregister_agents_from_registry(
    registry_id: str,
) -> list[str]:
    """
    Remove all agents synced from a specific AWS registry.

    Matches agents by:
    1. metadata.agentcore_registry_id (primary)
    2. 'agentcore' tag + path starting with /agents/agentcore- (fallback for older records)

    Args:
        registry_id: The AWS registry ARN to match

    Returns:
        List of agent paths that were deregistered
    """
    from ..repositories.factory import get_agent_repository
    from ..services.agent_service import agent_service

    agent_repo = get_agent_repository()

    # Primary: query by metadata
    matching_paths = set()
    by_metadata = await agent_repo.find_with_filter(
        {"metadata.agentcore_registry_id": registry_id}
    )
    matching_paths.update(by_metadata.keys())

    # Fallback: query by tag + path pattern for older records without metadata
    by_tag = await agent_repo.find_with_filter(
        {"tags": "agentcore", "_id": {"$regex": "^/agents/agentcore-"}}
    )
    matching_paths.update(by_tag.keys())

    deregistered = []
    for path in matching_paths:
        try:
            removed = await agent_service.remove_agent(path)
            if removed:
                deregistered.append(path)
                logger.info(f"Deregistered agent {path} from registry {registry_id}")
        except Exception as e:
            logger.error(f"Failed to deregister agent {path}: {e}")

    logger.info(f"Deregistered {len(deregistered)} agent(s) from registry {registry_id}")
    return deregistered


async def _deregister_skills_from_registry(
    registry_id: str,
) -> list[str]:
    """
    Remove all skills synced from a specific AWS registry.

    Matches skills by:
    1. metadata.agentcore_registry_id (newer skills)
    2. 'agentcore' tag + path starting with /skills/agentcore- (older skills without metadata)

    Args:
        registry_id: The AWS registry ARN to match

    Returns:
        List of skill paths that were deregistered
    """
    from ..repositories.factory import get_skill_repository

    skill_repo = get_skill_repository()
    all_skills = await skill_repo.list_all()

    matching_paths = set()
    for skill in all_skills:
        meta = skill.metadata or {}
        extra = meta.extra if hasattr(meta, "extra") else {}
        meta_dict = meta if isinstance(meta, dict) else {}

        # Match by metadata.agentcore_registry_id
        if meta_dict.get("agentcore_registry_id") == registry_id:
            matching_paths.add(skill.path)
            continue
        if extra.get("agentcore_registry_id") == registry_id:
            matching_paths.add(skill.path)
            continue

        # Fallback: match by 'agentcore' tag + path pattern
        tags = skill.tags or []
        if "agentcore" in tags and str(skill.path).startswith("/skills/agentcore-"):
            matching_paths.add(skill.path)

    deregistered = []
    for path in matching_paths:
        try:
            removed = await skill_repo.delete(path)
            if removed:
                deregistered.append(path)
                logger.info(f"Deregistered skill {path} from registry {registry_id}")
        except Exception as e:
            logger.error(f"Failed to deregister skill {path}: {e}")

    logger.info(f"Deregistered {len(deregistered)} skill(s) from registry {registry_id}")
    return deregistered


@router.post("/federation/sync", tags=["federation"], summary="Trigger manual federation sync")
async def sync_federation(
    request: Request,
    config_id: str = "default",
    source: str | None = None,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
    repo: FederationConfigRepositoryBase = Depends(_get_federation_repo),
) -> dict[str, Any]:
    """
    Manually trigger federation sync to import servers/agents from configured sources.

    Args:
        config_id: Configuration ID to use for sync (default: "default")
        source: Optional source filter ("anthropic", "asor", or "aws_registry"). If None, syncs all enabled sources.
        user_context: Authenticated user context
        repo: Federation config repository

    Returns:
        Sync results with counts of synced items

    Example:
        Sync all enabled federations:
        ```bash
        POST /api/federation/sync
        ```

        Sync only Anthropic:
        ```bash
        POST /api/federation/sync?source=anthropic
        ```
    """
    # Set audit action for federation sync
    set_audit_action(
        request,
        "sync",
        "federation",
        resource_id=config_id,
        description=f"Sync federation from {source or 'all sources'}",
    )

    logger.info(f"User {user_context['username']} triggering federation sync: {config_id}")

    # Get federation config
    config = await repo.get_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federation config '{config_id}' not found",
        )

    try:
        # Import federation clients
        from ..services.federation.anthropic_client import AnthropicFederationClient
        from ..services.federation.asor_client import AsorFederationClient

        results: dict[str, Any] = {
            "anthropic": {"servers": [], "count": 0},
            "asor": {"agents": [], "count": 0},
            "aws_registry": {"servers": [], "agents": [], "skills": [], "count": 0},
        }

        # Sync Anthropic servers if enabled and requested
        if (source is None or source == "anthropic") and config.anthropic.enabled:
            logger.info("Syncing servers from Anthropic MCP Registry...")

            anthropic_client = AnthropicFederationClient(endpoint=config.anthropic.endpoint)

            servers = anthropic_client.fetch_all_servers(config.anthropic.servers)

            # Register servers via server service
            from ..services.server_service import server_service

            for server_data in servers:
                try:
                    server_path = server_data.get("path")
                    if not server_path:
                        logger.warning(
                            f"Server missing path: {server_data.get('server_name')}, skipping"
                        )
                        continue

                    # Ensure UUID id field exists for federation sync
                    if "id" not in server_data or not server_data["id"]:
                        server_data["id"] = str(uuid4())

                    # Register server
                    # server_data already includes the "path" field
                    result = await server_service.register_server(server_data)
                    success = result["success"]

                    if not success and not result.get("is_new_version"):
                        logger.warning(
                            f"Server already exists or failed to register: {server_path}"
                        )
                        # Ensure UUID exists before updating (for servers registered before UUID feature)
                        if "id" not in server_data or not server_data["id"]:
                            server_data["id"] = str(uuid4())
                        # Try updating instead
                        success = await server_service.update_server(server_path, server_data)

                    if success:
                        # Enable the server
                        await server_service.toggle_service(server_path, True)

                        server_name = server_data.get("server_name", server_path)
                        logger.info(f"Synced Anthropic server: {server_name} at {server_path}")
                        results["anthropic"]["servers"].append(server_name)
                    else:
                        logger.error(f"Failed to register or update server: {server_path}")

                except Exception as e:
                    logger.error(
                        f"Failed to sync Anthropic server {server_data.get('server_name', 'unknown')}: {e}"
                    )

            results["anthropic"]["count"] = len(results["anthropic"]["servers"])
            logger.info(f"Synced {results['anthropic']['count']} servers from Anthropic")

        # Sync ASOR agents if enabled and requested
        if (source is None or source == "asor") and config.asor.enabled:
            logger.info("Syncing agents from ASOR...")

            tenant_url = (
                config.asor.endpoint.split("/api")[0]
                if "/api" in config.asor.endpoint
                else config.asor.endpoint
            )

            asor_client = AsorFederationClient(
                endpoint=config.asor.endpoint,
                auth_env_var=config.asor.auth_env_var,
                tenant_url=tenant_url,
            )

            agents = asor_client.fetch_all_agents(config.asor.agents)

            # Register agents
            from datetime import datetime

            from ..schemas.agent_models import AgentCard
            from ..services.agent_service import agent_service

            for agent_data in agents:
                try:
                    agent_name = agent_data.get("name", "Unknown ASOR Agent")
                    agent_path = f"/{agent_name.lower().replace('_', '-')}"

                    # Extract skills
                    skills_data = agent_data.get("skills", [])
                    skills = []
                    for skill in skills_data:
                        skills.append(
                            {
                                "name": skill.get("name", ""),
                                "description": skill.get("description", ""),
                                "id": skill.get("id", ""),
                            }
                        )

                    agent_card = AgentCard(
                        protocol_version="1.0",
                        name=agent_name,
                        path=agent_path,
                        url=agent_data.get("url", ""),
                        description=agent_data.get("description", f"ASOR agent: {agent_name}"),
                        version=agent_data.get("version", "1.0.0"),
                        provider="ASOR",
                        author="ASOR",
                        license="Unknown",
                        skills=skills,
                        tags=["asor", "federated", "workday"],
                        visibility="public",
                        registered_by="asor-federation",
                        registered_at=datetime.now(UTC),
                    )

                    if agent_path not in agent_service.registered_agents:
                        await agent_service.register_agent(agent_card)
                        logger.info(f"Synced ASOR agent: {agent_name}")
                        results["asor"]["agents"].append(agent_name)

                except Exception as e:
                    logger.error(
                        f"Failed to sync ASOR agent {agent_data.get('name', 'unknown')}: {e}"
                    )

            results["asor"]["count"] = len(results["asor"]["agents"])
            logger.info(f"Synced {results['asor']['count']} agents from ASOR")

        # Sync AgentCore records if enabled and requested
        if (source is None or source == "aws_registry") and config.aws_registry.enabled:
            logger.info("Syncing from AWS Agent Registry...")

            from ..schemas.agent_models import AgentCard
            from ..schemas.skill_models import SkillCard
            from ..repositories.factory import (
                get_agent_repository,
                get_skill_repository,
            )
            from ..services.agent_service import agent_service
            from ..services.federation.agentcore_client import AgentCoreFederationClient
            from ..services.server_service import server_service
            from ..services.skill_service import get_skill_service

            agentcore_client = AgentCoreFederationClient(
                aws_region=config.aws_registry.aws_region
            )
            records = agentcore_client.fetch_all_records(
                registry_configs=config.aws_registry.registries,
                sync_timeout_seconds=config.aws_registry.sync_timeout_seconds,
                max_concurrent_fetches=config.aws_registry.max_concurrent_fetches,
            )

            # Register servers (MCP records)
            for srv in records["servers"]:
                try:
                    srv_path = srv.get("path")
                    if not srv_path:
                        continue
                    if "id" not in srv or not srv["id"]:
                        srv["id"] = str(uuid4())

                    result = await server_service.register_server(srv)
                    if not result["success"]:
                        if "id" not in srv or not srv["id"]:
                            srv["id"] = str(uuid4())
                        await server_service.update_server(srv_path, srv)

                    await server_service.toggle_service(srv_path, True)
                    results["aws_registry"]["servers"].append(srv.get("server_name", srv_path))
                except Exception as e:
                    logger.error(f"Failed to sync AgentCore server {srv.get('server_name', 'unknown')}: {e}")

            # Register agents (A2A + CUSTOM records)
            for agent_data in records["agents"]:
                try:
                    agent_path = agent_data.get("path")
                    if not agent_path:
                        continue
                    try:
                        agent_card = AgentCard(**agent_data)
                        await agent_service.register_agent(agent_card)
                    except ValueError:
                        await agent_service.update_agent(agent_path, agent_data)
                    results["aws_registry"]["agents"].append(agent_data.get("name", agent_path))
                except Exception as e:
                    logger.error(f"Failed to sync AgentCore agent {agent_data.get('name', 'unknown')}: {e}")

            # Register skills (AGENT_SKILLS records)
            skill_service = get_skill_service()
            skill_repo = get_skill_repository()
            for skill_data in records["skills"]:
                try:
                    skill_path = skill_data.get("path")
                    if not skill_path:
                        continue
                    try:
                        skill_card = SkillCard(**skill_data)
                        await skill_repo.create(skill_card)
                    except Exception as create_err:
                        logger.debug(f"Skill create failed for {skill_path}, trying update: {create_err}")
                        update_fields = {
                            k: v for k, v in skill_data.items()
                            if k not in ("path", "id", "created_at")
                        }
                        await skill_repo.update(skill_path, update_fields)
                    results["aws_registry"]["skills"].append(skill_data.get("name", skill_path))
                except Exception as e:
                    logger.error(f"Failed to sync AgentCore skill {skill_data.get('name', 'unknown')}: {e}")

            agentcore_total = (
                len(results["aws_registry"]["servers"])
                + len(results["aws_registry"]["agents"])
                + len(results["aws_registry"]["skills"])
            )
            results["aws_registry"]["count"] = agentcore_total
            logger.info(
                f"Synced from AWS Agent Registry: "
                f"{len(results['aws_registry']['servers'])} servers, "
                f"{len(results['aws_registry']['agents'])} agents, "
                f"{len(results['aws_registry']['skills'])} skills"
            )

        # Reconcile: remove stale federated servers after sync
        reconciliation_result = None
        try:
            from ..core.nginx_service import nginx_service as nginx_svc
            from ..repositories.factory import get_server_repository
            from ..services.federation_reconciliation import reconcile_anthropic_servers

            server_repo = get_server_repository()
            reconciliation_result = await reconcile_anthropic_servers(
                config=config,
                server_service=server_service,
                server_repo=server_repo,
                nginx_service=nginx_svc,
                audit_username=user_context.get("username"),
            )
            if reconciliation_result.get("removed"):
                logger.info(
                    f"Reconciliation removed {reconciliation_result['removed_count']} stale servers: "
                    f"{reconciliation_result['removed']}"
                )
        except Exception as reconcile_error:
            logger.warning(f"Reconciliation failed after sync: {reconcile_error}")

        return {
            "message": "Federation sync completed",
            "config_id": config_id,
            "results": results,
            "total_synced": (
                results["anthropic"]["count"]
                + results["asor"]["count"]
                + results["aws_registry"]["count"]
            ),
            "reconciliation": reconciliation_result,
        }

    except Exception as e:
        logger.error(f"Federation sync failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Federation sync failed: {str(e)}",
        )
