"""
System information and operational API routes.

These endpoints provide system-level information for monitoring and display.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..version import __version__


logger = logging.getLogger(__name__)
router = APIRouter()


# Global variables for server start time and stats caching
_server_start_time: Optional[datetime] = None
_stats_cache: Optional[dict] = None
_stats_cache_time: Optional[datetime] = None
STATS_CACHE_TTL_SECONDS = 30  # Cache stats for 30 seconds


def set_server_start_time(
    start_time: datetime,
) -> None:
    """Set the server start time (called from main.py lifespan)."""
    global _server_start_time
    _server_start_time = start_time
    logger.info(f"System routes: Server start time set to {start_time.isoformat()}")


def _detect_deployment_type() -> str:
    """Auto-detect deployment environment based on environment variables.

    Detection order:
    1. Kubernetes - Check for KUBERNETES_SERVICE_HOST
    2. ECS - Check for ECS_CONTAINER_METADATA_URI
    3. EC2 - Check for AWS_EXECUTION_ENV
    4. Local - Default fallback

    Returns:
        Deployment type: "Kubernetes", "ECS", "EC2", or "Local"
    """
    # Check for Kubernetes
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "Kubernetes"

    # Check for ECS
    if os.getenv("ECS_CONTAINER_METADATA_URI") or os.getenv(
        "ECS_CONTAINER_METADATA_URI_V4"
    ):
        return "ECS"

    # Check for EC2
    if os.getenv("AWS_EXECUTION_ENV") == "AWS_ECS_EC2":
        return "EC2"

    # Default to Local
    return "Local"


async def _get_registry_stats() -> dict:
    """Get current registry statistics (servers, agents, skills counts).

    Uses efficient count() methods instead of loading all resources.

    Returns:
        Dictionary with servers, agents, skills counts
    """
    try:
        # Import repositories
        from registry.repositories.factory import (
            get_server_repository,
            get_agent_repository,
            get_skill_repository,
        )

        # Get repository instances
        server_repo = get_server_repository()
        agent_repo = get_agent_repository()
        skill_repo = get_skill_repository()

        # Count resources efficiently using count() methods
        servers_count = await server_repo.count()
        agents_count = await agent_repo.count()
        skills_count = await skill_repo.count()

        return {
            "servers": servers_count,
            "agents": agents_count,
            "skills": skills_count,
        }
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        # Return zeros on error
        return {
            "servers": 0,
            "agents": 0,
            "skills": 0,
        }


async def _get_auth_status() -> dict:
    """Check authentication server health and connection status.

    Returns:
        Dictionary with provider, status, and URL information
    """
    provider = settings.auth_provider
    auth_url = settings.auth_server_url

    # Try to ping the auth server health endpoint
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try common health check endpoints
            health_endpoints = [
                f"{auth_url}/health",
                f"{auth_url}/healthcheck",
                f"{auth_url}/.well-known/openid-configuration",
            ]

            for endpoint in health_endpoints:
                try:
                    response = await client.get(endpoint)
                    if response.status_code < 500:  # 2xx, 3xx, 4xx are all "reachable"
                        return {
                            "provider": provider,
                            "status": "Healthy",
                            "url": auth_url,
                        }
                except Exception:
                    continue

            # If all endpoints failed, auth server is unhealthy
            return {
                "provider": provider,
                "status": "Unhealthy",
                "url": auth_url,
            }

    except Exception as e:
        logger.error(f"Auth server health check failed: {e}")
        return {
            "provider": provider,
            "status": "Unhealthy",
            "url": auth_url,
        }


async def _get_database_status() -> dict:
    """Check database health and connection status.

    Returns:
        Dictionary with backend, status, and host information
    """
    backend = settings.storage_backend

    # File backend has no database to check
    if backend == "file":
        return {
            "backend": "file",
            "status": "N/A",
            "host": "N/A",
        }

    # DocumentDB/MongoDB backend - check connection
    try:
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()

        # Try to ping the database (db is AsyncIOMotorDatabase, not client)
        await db.command("ping")

        # Get host information
        host_str = f"{settings.documentdb_host}:{settings.documentdb_port}"

        return {
            "backend": backend,
            "status": "Healthy",
            "host": host_str,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        host_str = f"{settings.documentdb_host}:{settings.documentdb_port}"
        return {
            "backend": backend,
            "status": "Unhealthy",
            "host": host_str,
        }


async def _get_cached_stats() -> dict:
    """Get system stats with caching to reduce load.

    Cache TTL: 30 seconds

    Returns:
        Cached or freshly computed stats dictionary
    """
    global _stats_cache, _stats_cache_time

    now = datetime.now(timezone.utc)

    # Check if cache is valid
    if (
        _stats_cache is not None
        and _stats_cache_time is not None
        and (now - _stats_cache_time).total_seconds() < STATS_CACHE_TTL_SECONDS
    ):
        return _stats_cache

    # Compute fresh stats
    registry_stats = await _get_registry_stats()
    database_status = await _get_database_status()
    auth_status = await _get_auth_status()

    # Calculate uptime
    if _server_start_time:
        uptime_seconds = int((now - _server_start_time).total_seconds())
        started_at = _server_start_time
    else:
        # Fallback if start time not set (shouldn't happen)
        uptime_seconds = 0
        started_at = now

    stats = {
        "uptime_seconds": uptime_seconds,
        "started_at": started_at.isoformat(),
        "version": __version__,
        "deployment_type": _detect_deployment_type(),
        "deployment_mode": settings.deployment_mode.value,
        "registry_stats": registry_stats,
        "database_status": database_status,
        "auth_status": auth_status,
    }

    # Update cache
    _stats_cache = stats
    _stats_cache_time = now

    return stats


@router.get("/api/version")
async def get_version():
    """Get application version.

    Returns:
        Dictionary with version string
    """
    return {"version": __version__}


@router.get("/api/stats")
async def get_system_stats():
    """Get system statistics including uptime, deployment info, and registry metrics.

    This endpoint provides operational information for monitoring and display:
    - Application uptime since last restart
    - Deployment environment and mode
    - Registry resource counts (servers, agents, skills)
    - Database health status

    Response is cached for 30 seconds to reduce load.

    Returns:
        System statistics dictionary with:
        - uptime_seconds: Time since server started
        - started_at: ISO 8601 timestamp of server start
        - version: Application version
        - deployment_type: Kubernetes/ECS/EC2/Local
        - deployment_mode: with-gateway/registry-only
        - registry_stats: Object with servers, agents, skills counts
        - database_status: Object with backend, status, host
        - auth_status: Object with provider, status, url
    """
    try:
        stats = await _get_cached_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to compute system statistics"
        )
