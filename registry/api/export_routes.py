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
from ..repositories.factory import (
    get_custom_entity_repository,
    get_custom_type_repository,
    get_scope_repository,
)

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


@router.get("/custom-types")
async def export_custom_types(
    user_context: Annotated[dict, Depends(_require_admin)],
) -> dict[str, Any]:
    """Export all custom entity type descriptors (the admin-defined schemas).

    Returns the raw descriptor documents from the mcp_custom_types collection.
    """
    type_repo = get_custom_type_repository()
    collection = await type_repo._get_collection()
    cursor = collection.find({})
    custom_types = []
    async for doc in cursor:
        custom_types.append(doc)
    logger.info(
        f"Exported {len(custom_types)} custom type documents for user "
        f"'{user_context.get('username', 'unknown')}'"
    )
    return {"custom_types": custom_types, "total_count": len(custom_types)}


@router.get("/custom-entities")
async def export_custom_entities(
    user_context: Annotated[dict, Depends(_require_admin)],
) -> dict[str, Any]:
    """Export all custom entity records across every custom type.

    Returns the raw record documents from the single mcp_custom_entities
    collection (records of all types, discriminated by entity_type).
    """
    entity_repo = get_custom_entity_repository()
    collection = await entity_repo._get_collection()
    cursor = collection.find({})
    custom_entities = []
    async for doc in cursor:
        custom_entities.append(doc)
    logger.info(
        f"Exported {len(custom_entities)} custom entity records for user "
        f"'{user_context.get('username', 'unknown')}'"
    )
    return {"custom_entities": custom_entities, "total_count": len(custom_entities)}
