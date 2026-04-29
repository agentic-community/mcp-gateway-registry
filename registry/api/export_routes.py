"""Routes for data export audit events and admin data dumps."""

import logging
from typing import (
    Annotated,
    Any,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    Field,
)

from ..audit import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..repositories.factory import get_scope_repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["Data Export"])


class ExportAuditRequest(BaseModel):
    """Request body for recording a data export audit event."""

    export_type: str = Field(
        ...,
        description="Type of export: 'single' for one collection, 'all' for bulk ZIP",
        pattern="^(single|all)$",
    )
    collections: list[str] = Field(
        ...,
        description="List of collection IDs that were exported",
        min_length=1,
    )


def _require_admin(
    user_context: dict[str, Any] = Depends(nginx_proxied_auth),
) -> dict[str, Any]:
    """Dependency that requires admin access."""
    if not user_context.get("is_admin", False):
        logger.warning(
            f"Non-admin user '{user_context.get('username', 'unknown')}' "
            "attempted to record export audit event"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_context


@router.post("/audit-event")
async def record_export_audit_event(
    request: Request,
    body: ExportAuditRequest,
    user_context: Annotated[dict, Depends(_require_admin)],
) -> dict[str, str]:
    """Record an audit event for a data export action.

    This endpoint emits a dedicated audit event so that export activity
    is easily searchable in the audit log (operation='export', resource_type='data').
    """
    collections_str = ", ".join(body.collections)
    set_audit_action(
        request,
        "export",
        "data",
        description=f"Data export ({body.export_type}): {collections_str}",
    )
    logger.info(
        f"Data export audit event recorded: type={body.export_type}, "
        f"collections={collections_str}, user={user_context.get('username', 'unknown')}"
    )
    return {"status": "ok"}


@router.get("/scopes")
async def export_scopes(
    user_context: Annotated[dict, Depends(_require_admin)],
) -> dict[str, Any]:
    """Export all scope documents from the mcp_scopes collection.

    Returns the raw scope documents with full server_access rules,
    group_mappings, ui_permissions, and agent_access details.
    """
    scope_repo = get_scope_repository()
    collection = await scope_repo._get_collection()
    cursor = collection.find({})
    scopes = []
    async for doc in cursor:
        doc["scope_name"] = doc.pop("_id", None)
        scopes.append(doc)
    logger.info(
        f"Exported {len(scopes)} scope documents for user "
        f"'{user_context.get('username', 'unknown')}'"
    )
    return {"scopes": scopes, "total_count": len(scopes)}
