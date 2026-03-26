# registry/api/ans_routes.py

import logging
import time
from collections import defaultdict
from typing import (
    Annotated,
    Any,
)

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from registry.auth.csrf import verify_csrf_token_flexible
from registry.auth.dependencies import nginx_proxied_auth
from registry.audit import set_audit_action
from registry.core.config import settings
from registry.schemas.ans_models import LinkANSRequest
from registry.services.ans_service import (
    get_ans_metrics,
    get_sync_history,
    link_ans_to_agent,
    link_ans_to_server,
    sync_all_ans_status,
    unlink_ans_from_agent,
    unlink_ans_from_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()

RATE_LIMIT_MAX_REQUESTS: int = 10
RATE_LIMIT_WINDOW_SECONDS: int = 3600
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(
    username: str,
) -> None:
    """Check per-user rate limit for ANS link operations.

    Args:
        username: Authenticated user's username

    Raises:
        HTTPException 429 if rate limit exceeded
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    _rate_limit_store[username] = [
        t for t in _rate_limit_store[username] if t > window_start
    ]

    if len(_rate_limit_store[username]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} ANS link operations per hour",
        )

    _rate_limit_store[username].append(now)


def _check_ans_enabled() -> None:
    """Raise 404 if ANS integration is disabled."""
    if not settings.ans_integration_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ANS integration is not enabled",
        )


def _get_username(
    user_context: dict | None,
) -> str:
    """Extract username from user context.

    Args:
        user_context: Auth context dict

    Returns:
        Username string
    """
    if not user_context:
        return "unknown"
    return user_context.get("username", user_context.get("sub", "unknown"))


def _check_admin(
    user_context: dict | None,
) -> None:
    """Verify user has admin role/scope.

    Args:
        user_context: Auth context dict

    Raises:
        HTTPException 403 if not admin
    """
    if not user_context:
        raise HTTPException(status_code=403, detail="Admin access required")

    scopes = user_context.get("scopes", [])
    groups = user_context.get("groups", [])
    is_admin = (
        "admin" in groups
        or "ans-admin/manage" in scopes
        or any("unrestricted" in s for s in scopes)
    )
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")


def _normalize_path(path: str) -> str:
    """Normalize entity path to ensure leading slash."""
    if not path.startswith("/"):
        path = "/" + path
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    return path


# --- Agent ANS endpoints ---


@router.post("/agents/{path:path}/ans/link")
async def link_ans_to_agent_endpoint(
    request: Request,
    path: str,
    body: LinkANSRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Link an ANS Agent ID to an agent."""
    _check_ans_enabled()
    path = _normalize_path(path)
    await verify_csrf_token_flexible(request)
    username = _get_username(user_context)
    _check_rate_limit(username)
    set_audit_action(
        request, "create", "ans_link",
        resource_id=path,
        description=f"Link ANS ID to agent {path}",
    )
    result = await link_ans_to_agent(path, body.ans_agent_id, username=username)
    if not result["success"]:
        status_code = status.HTTP_400_BAD_REQUEST
        if "Not authorized" in result.get("message", ""):
            status_code = status.HTTP_403_FORBIDDEN
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


@router.get("/agents/{path:path}/ans/status")
async def get_agent_ans_status(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Get ANS verification status for an agent."""
    _check_ans_enabled()
    path = _normalize_path(path)
    from registry.repositories.factory import get_agent_repository

    repo = get_agent_repository()
    agent = await repo.get(path)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    ans_metadata = agent.ans_metadata
    if not ans_metadata:
        raise HTTPException(status_code=404, detail="No ANS link found")

    return ans_metadata


@router.delete("/agents/{path:path}/ans/link")
async def unlink_ans_from_agent_endpoint(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Remove ANS link from an agent."""
    _check_ans_enabled()
    path = _normalize_path(path)
    await verify_csrf_token_flexible(request)
    username = _get_username(user_context)
    set_audit_action(
        request, "delete", "ans_link",
        resource_id=path,
        description=f"Unlink ANS ID from agent {path}",
    )
    result = await unlink_ans_from_agent(path, username=username)
    if not result["success"]:
        status_code = 404
        if "Not authorized" in result.get("message", ""):
            status_code = 403
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


# --- Server ANS endpoints ---


@router.post("/servers/{path:path}/ans/link")
async def link_ans_to_server_endpoint(
    request: Request,
    path: str,
    body: LinkANSRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Link an ANS Agent ID to an MCP server."""
    _check_ans_enabled()
    path = _normalize_path(path)
    await verify_csrf_token_flexible(request)
    username = _get_username(user_context)
    _check_rate_limit(username)
    set_audit_action(
        request, "create", "ans_link",
        resource_id=path,
        description=f"Link ANS ID to server {path}",
    )
    result = await link_ans_to_server(path, body.ans_agent_id, username=username)
    if not result["success"]:
        status_code = status.HTTP_400_BAD_REQUEST
        if "Not authorized" in result.get("message", ""):
            status_code = status.HTTP_403_FORBIDDEN
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


@router.get("/servers/{path:path}/ans/status")
async def get_server_ans_status(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Get ANS verification status for a server."""
    _check_ans_enabled()
    path = _normalize_path(path)
    from registry.repositories.factory import get_server_repository

    repo = get_server_repository()
    server = await repo.get(path)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    ans_metadata = server.get("ans_metadata")
    if not ans_metadata:
        raise HTTPException(status_code=404, detail="No ANS link found")

    return ans_metadata


@router.delete("/servers/{path:path}/ans/link")
async def unlink_ans_from_server_endpoint(
    request: Request,
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Remove ANS link from a server."""
    _check_ans_enabled()
    path = _normalize_path(path)
    await verify_csrf_token_flexible(request)
    username = _get_username(user_context)
    set_audit_action(
        request, "delete", "ans_link",
        resource_id=path,
        description=f"Unlink ANS ID from server {path}",
    )
    result = await unlink_ans_from_server(path, username=username)
    if not result["success"]:
        status_code = 404
        if "Not authorized" in result.get("message", ""):
            status_code = 403
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


# --- Admin ANS endpoints ---


@router.post("/admin/ans/sync")
async def trigger_ans_sync(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Manually trigger ANS status sync (admin only)."""
    _check_ans_enabled()
    _check_admin(user_context)
    set_audit_action(
        request, "execute", "ans_sync",
        description="Manual ANS sync triggered",
    )
    stats = await sync_all_ans_status()
    return stats.model_dump()


@router.get("/admin/ans/metrics")
async def get_ans_metrics_endpoint(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Get ANS integration metrics (admin only)."""
    _check_ans_enabled()
    _check_admin(user_context)
    metrics = await get_ans_metrics()
    result = metrics.model_dump(mode="json")
    result["sync_history"] = get_sync_history()
    return result


@router.get("/admin/ans/health")
async def get_ans_health(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Check ANS API reachability (admin only)."""
    _check_ans_enabled()
    _check_admin(user_context)

    from registry.services.ans_client import _check_circuit_breaker

    circuit_ok = _check_circuit_breaker()
    if not circuit_ok:
        return {
            "status": "degraded",
            "message": "ANS API circuit breaker is open",
            "api_reachable": False,
        }

    try:
        headers = {
            "Authorization": f"sso-key {settings.ans_api_key}:{settings.ans_api_secret}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.ans_api_endpoint}/v1/agents?limit=1",
                headers=headers,
            )
            return {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "api_reachable": resp.status_code == 200,
                "api_status_code": resp.status_code,
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "api_reachable": False,
            "error": str(e),
        }
