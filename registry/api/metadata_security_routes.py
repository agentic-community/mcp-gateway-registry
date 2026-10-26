"""Admin route to rescan registrant metadata and reindex it (issue #1811).

See registry/services/metadata_rescan_service.py for what a rescan does and
why it is safe to run more than once.
"""

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from registry.audit.context import set_audit_action
from registry.auth.csrf import verify_csrf_token_flexible
from registry.auth.dependencies import nginx_proxied_auth
from registry.services.metadata_rescan_service import MetadataRescanError, rescan_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/metadata-security", tags=["Metadata Security Admin"])


class MetadataRescanRequest(BaseModel):
    """Request body for POST /api/admin/metadata-security/rescan."""

    model_config = ConfigDict(extra="forbid")

    entity_types: list[Literal["server", "agent", "skill"]] | None = Field(
        default=None,
        min_length=1,
        description="Kinds of record to process; all three when omitted",
    )
    dry_run: bool = Field(
        default=False,
        description="Count records that need a new verdict without writing anything",
    )


def _require_admin(user_context: dict) -> None:
    """Verify user has admin permissions."""
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions are required for this operation",
        )


@router.post("/rescan")
async def rescan_metadata_endpoint(
    http_request: Request,
    request: MetadataRescanRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict[str, Any]:
    """Recompute missing or outdated metadata verdicts and reindex for search.

    Idempotent and resumable: records whose verdict is current and whose search
    entry is up to date are skipped, so an interrupted run can simply be repeated.
    """
    _require_admin(user_context)
    set_audit_action(
        http_request,
        operation="rescan",
        resource_type="metadata_security",
        description="Rescan registrant metadata and reindex it for search",
        metadata={"entity_types": request.entity_types, "dry_run": request.dry_run},
    )
    try:
        result = await rescan_metadata(request.entity_types, dry_run=request.dry_run)
    except MetadataRescanError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    logger.info(f"Metadata rescan requested by {user_context.get('username')}: {result}")
    return result
