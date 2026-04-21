"""Direct CRUD endpoints for M2M client registration.

These endpoints write directly to the ``idp_m2m_clients`` MongoDB collection
without calling any IdP Admin API. Operators without ``OKTA_API_TOKEN`` (or
equivalent) can register M2M ``client_id`` values and their group mappings so
the auth server can enrich M2M tokens during authorization.

Tracked by issue #851.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from prometheus_client import Counter

from registry.audit.context import set_audit_action
from registry.auth.dependencies import nginx_proxied_auth
from registry.repositories.documentdb.client import get_documentdb_client
from registry.schemas.idp_m2m_client import (
    IdPM2MClient,
    IdPM2MClientCreate,
    IdPM2MClientPatch,
    M2MClientListResponse,
)
from registry.services.m2m_management_service import (
    M2MClientConflict,
    M2MClientImmutable,
    M2MClientNotFound,
    M2MManagementService,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_RESOURCE_TYPE: str = "m2m_client"
_LIST_DEFAULT_LIMIT: int = 500
_LIST_MAX_LIMIT: int = 1000


m2m_management_requests_total = Counter(
    "m2m_management_requests_total",
    "Count of direct M2M client registration API calls",
    ["operation", "outcome"],
)


def _require_admin(
    user_context: dict | None,
    operation: str,
) -> None:
    """Enforce admin permission or raise 401/403 and increment metrics."""
    if not user_context:
        m2m_management_requests_total.labels(operation=operation, outcome="auth_error").inc()
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user_context.get("is_admin"):
        m2m_management_requests_total.labels(operation=operation, outcome="forbidden").inc()
        raise HTTPException(status_code=403, detail="Administrator permissions are required")


async def _get_service() -> M2MManagementService:
    db = await get_documentdb_client()
    return M2MManagementService(db)


@router.post(
    "/iam/m2m-clients",
    response_model=IdPM2MClient,
    status_code=status.HTTP_201_CREATED,
)
async def create_m2m_client(
    payload: IdPM2MClientCreate,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPM2MClient:
    """Register a new M2M client with its group mappings (admin only)."""
    _require_admin(user_context, operation="create")
    service = await _get_service()
    created_by = user_context.get("username") if user_context else None
    try:
        result = await service.create(payload, created_by=created_by)
    except M2MClientConflict:
        m2m_management_requests_total.labels(operation="create", outcome="conflict").inc()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"M2M client {payload.client_id} already exists",
        )
    set_audit_action(
        request,
        "create",
        _RESOURCE_TYPE,
        resource_id=payload.client_id,
        description=f"Created M2M client {payload.client_name}",
    )
    m2m_management_requests_total.labels(operation="create", outcome="success").inc()
    return result


@router.get("/iam/m2m-clients", response_model=M2MClientListResponse)
async def list_m2m_clients(
    provider: str | None = None,
    limit: int = Query(default=_LIST_DEFAULT_LIMIT, ge=1, le=_LIST_MAX_LIMIT),
    skip: int = Query(default=0, ge=0),
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> M2MClientListResponse:
    """List M2M clients (any authenticated user)."""
    if not user_context:
        m2m_management_requests_total.labels(operation="list", outcome="auth_error").inc()
        raise HTTPException(status_code=401, detail="Not authenticated")
    service = await _get_service()
    items, total = await service.list_paged(provider=provider, limit=limit, skip=skip)
    m2m_management_requests_total.labels(operation="list", outcome="success").inc()
    return M2MClientListResponse(total=total, limit=limit, skip=skip, items=items)


@router.get("/iam/m2m-clients/{client_id}", response_model=IdPM2MClient)
async def get_m2m_client(
    client_id: str,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPM2MClient:
    """Get a specific M2M client (any authenticated user)."""
    if not user_context:
        m2m_management_requests_total.labels(operation="get", outcome="auth_error").inc()
        raise HTTPException(status_code=401, detail="Not authenticated")
    service = await _get_service()
    try:
        result = await service.get(client_id)
    except M2MClientNotFound:
        m2m_management_requests_total.labels(operation="get", outcome="not_found").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"M2M client {client_id} not found",
        )
    m2m_management_requests_total.labels(operation="get", outcome="success").inc()
    return result


@router.patch("/iam/m2m-clients/{client_id}", response_model=IdPM2MClient)
async def patch_m2m_client(
    client_id: str,
    payload: IdPM2MClientPatch,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> IdPM2MClient:
    """Update fields of an existing manual M2M client (admin only)."""
    _require_admin(user_context, operation="patch")
    service = await _get_service()
    try:
        result = await service.patch(client_id, payload)
    except M2MClientNotFound:
        m2m_management_requests_total.labels(operation="patch", outcome="not_found").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"M2M client {client_id} not found",
        )
    except M2MClientImmutable:
        m2m_management_requests_total.labels(operation="patch", outcome="forbidden").inc()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"M2M client {client_id} was registered by IdP sync and "
                "cannot be modified via this API"
            ),
        )
    set_audit_action(
        request,
        "update",
        _RESOURCE_TYPE,
        resource_id=client_id,
        description=f"Updated M2M client {client_id}",
    )
    m2m_management_requests_total.labels(operation="patch", outcome="success").inc()
    return result


@router.delete("/iam/m2m-clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_m2m_client(
    client_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> None:
    """Delete a manual M2M client (admin only)."""
    _require_admin(user_context, operation="delete")
    service = await _get_service()
    try:
        await service.delete(client_id)
    except M2MClientNotFound:
        m2m_management_requests_total.labels(operation="delete", outcome="not_found").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"M2M client {client_id} not found",
        )
    except M2MClientImmutable:
        m2m_management_requests_total.labels(operation="delete", outcome="forbidden").inc()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"M2M client {client_id} was registered by IdP sync and "
                "cannot be deleted via this API"
            ),
        )
    set_audit_action(
        request,
        "delete",
        _RESOURCE_TYPE,
        resource_id=client_id,
        description=f"Deleted M2M client {client_id}",
    )
    m2m_management_requests_total.labels(operation="delete", outcome="success").inc()
