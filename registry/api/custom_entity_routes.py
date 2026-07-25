"""
API routes for custom entity RECORDS.

Generic CRUD over records of any admin-defined custom type. The ``{type}`` and
``{uuid}`` path segments are interpolated into Mongo queries (the synthetic
record path is ``/{type}/{uuid}``), so both are constrained at the signature
(NoSQL-injection guard). Record visibility is enforced in the service layer.
"""

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from pydantic import BaseModel

from ..audit.context import set_audit_action
from ..auth.asset_permissions import user_has_asset_permission
from ..auth.csrf import verify_csrf_token_flexible
from ..auth.dependencies import nginx_proxied_auth
from ..schemas.custom_entity_models import (
    CustomEntityCreate,
    CustomEntityRecord,
    CustomEntityUpdate,
)
from ..services.custom_entity_errors import (
    CustomEntityNotFoundError,
    CustomEntityValidationError,
    CustomTypeRecordCapError,
    UnknownCustomTypeError,
)
from ..services.custom_entity_scopes import (
    entity_scope,
    list_grant_allows_type,
    list_grant_record_paths,
)
from ..services.custom_entity_service import CustomEntityService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/custom", tags=["custom-entities"])

# NoSQL-injection guards: both segments compose the record path
# /{type}/{uuid} interpolated into find({"_id": path}) / find({"entity_type": type}).
TYPE_PARAM = Path(..., pattern=r"^[a-z0-9_-]+$", max_length=64)
UUID_PARAM = Path(
    ...,
    pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
)


class RatingRequest(BaseModel):
    """Body for POST /api/custom/{type}/{uuid}/rate."""

    rating: int


def _get_service() -> CustomEntityService:
    """Resolve the custom entity service singleton."""
    from ..repositories.factory import get_custom_entity_service

    return get_custom_entity_service()


def _has_type_scope(
    action: str,
    type_name: str,
    user_context: dict,
) -> bool:
    """Return True if the caller holds the per-type MUTATION scope, or is admin.

    Used for the mutation actions (create/modify/delete), which stay type-level:
    the caller must hold ``<action>_<type>_entity`` for this type (or "all").
    Admin is the catch-all bypass. Fails closed on a missing ui_permissions dict.
    The read/list gate is per-record aware and lives in ``_require_view_scope`` /
    ``user_can_list_custom_entity_type`` instead.

    Args:
        action: One of create/modify/delete.
        type_name: The custom type being accessed.
        user_context: The authenticated request context.

    Returns:
        True if access is permitted, False otherwise.
    """
    return user_has_asset_permission(
        "custom_entity", action, type_name, user_context, type_name=type_name
    )


def _require_view_scope(
    type_name: str,
    user_context: dict,
    record_path: str | None = None,
) -> None:
    """Raise 404 (hide existence) if the caller lacks list access.

    Read gate for list/get/search/rating. The ``list_<type>_entity`` grant is
    per-record aware (parity with ``list_agents``): ``"all"``/type-name open the
    whole type; a record path opens just that record.

    - On the COLLECTION list (``record_path=None``): passes if the caller can see
      ANY record of the type (whole-type OR at least one granted record), so a
      record-scoped grant does NOT 404 the type — the list then filters to the
      granted records. Holding nothing 404s (hides existence, incl. public).
    - On a SINGLE record (``record_path`` given): passes only if the grant covers
      that specific record; otherwise 404 (indistinguishable from not-found).
    """
    from ..auth.dependencies import user_can_list_custom_entity_type

    if not user_can_list_custom_entity_type(type_name, user_context, record_path):
        logger.info(
            "User %s denied list access to custom type %s (record=%s) -> 404",
            user_context.get("username"),
            type_name,
            record_path or "<collection>",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown custom type: {type_name}",
        )


def _require_mutate_scope(
    action: str,
    type_name: str,
    user_context: dict,
) -> None:
    """Raise 403 if the caller lacks the ``<action>_<type>_entity`` scope.

    Mutation gate for create/modify/delete. Unlike the read gate, existence is
    not concealed (the caller can already see the type via the list scope), so a
    non-holder gets a 403.
    """
    if not _has_type_scope(action, type_name, user_context):
        logger.warning(
            "User %s denied %s on custom type %s (no %s_%s_entity scope) -> 403",
            user_context.get("username"),
            action,
            type_name,
            action,
            type_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to {action} {type_name} records",
        )


def _list_restrict_paths(
    type_name: str,
    user_context: dict,
) -> list[str] | None:
    """Return the record-path restriction for the collection list.

    - ``None`` — whole-type access (admin, or a ``"all"``/type-name grant): the
      list is bounded only by per-record visibility.
    - ``list[str]`` — the caller holds only specific records of this type; the
      list must be restricted to those paths (intersected with the per-record
      visibility filter).

    Assumes ``_require_view_scope`` already passed, so a non-whole-type caller
    holds at least one record path here.
    """
    if user_context.get("is_admin", False):
        return None
    granted = (user_context.get("ui_permissions") or {}).get(entity_scope("list", type_name)) or []
    if list_grant_allows_type(type_name, granted):
        return None
    return list_grant_record_paths(type_name, granted)


@router.get("/{type}", summary="List records of a custom type")
async def list_custom_entities(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
) -> dict:
    """List records of a type, filtered to those the caller may see."""
    _require_view_scope(type, user_context)
    # SECURITY: restrict_paths MUST be derived from _list_restrict_paths and
    # passed to list_records. _require_view_scope passes a record-scoped caller
    # (they hold at least one record path), so a whole-type read here would leak
    # every record. _list_restrict_paths returns None only for whole-type/admin;
    # a non-admin without the scope yields [] -> empty $in -> no records.
    restrict_paths = _list_restrict_paths(type, user_context)
    service = _get_service()
    try:
        items, total = await service.list_records(
            type, skip, limit, user_context, restrict_paths=restrict_paths
        )
    except UnknownCustomTypeError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return {
        "records": [r.model_dump(mode="json") for r in items],
        "total_count": total,
        "skip": skip,
        "limit": limit,
    }


@router.get(
    "/{type}/{uuid}",
    response_model=CustomEntityRecord,
    summary="Get a custom record",
)
async def get_custom_entity(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    uuid: str = UUID_PARAM,
) -> CustomEntityRecord:
    """Get a single record by type and uuid (404 if not viewable)."""
    path = f"/{type}/{uuid}"
    _require_view_scope(type, user_context, record_path=path)
    service = _get_service()
    try:
        return await service.get_record(path, user_context)
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{type}",
    response_model=CustomEntityRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom record",
)
async def create_custom_entity(
    http_request: Request,
    body: CustomEntityCreate,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> CustomEntityRecord:
    """Create a record of the given custom type."""
    _require_mutate_scope("create", type, user_context)
    service = _get_service()
    owner = user_context.get("username")  # server-derived, never from body
    try:
        created = await service.create_record(type, body, owner=owner)
    except UnknownCustomTypeError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except CustomTypeRecordCapError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CustomEntityValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors)

    set_audit_action(
        http_request,
        "create",
        "custom_entity",
        resource_id=created.path,
        description=f"Create {type} {created.name}",
    )
    logger.info(f"Created custom record {created.path} by {owner}")
    return created


@router.put(
    "/{type}/{uuid}",
    response_model=CustomEntityRecord,
    summary="Update a custom record",
)
async def update_custom_entity(
    http_request: Request,
    body: CustomEntityUpdate,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    uuid: str = UUID_PARAM,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> CustomEntityRecord:
    """Update a record (owner or admin only; partial-update semantics)."""
    # Type-level gate first; the service still enforces per-record owner-or-admin.
    _require_mutate_scope("modify", type, user_context)
    service = _get_service()
    path = f"/{type}/{uuid}"
    try:
        updated = await service.update_record(type, path, body, user_context)
    except UnknownCustomTypeError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except CustomEntityValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors)

    set_audit_action(
        http_request,
        "update",
        "custom_entity",
        resource_id=path,
        description=f"Update {type} {updated.name}",
    )
    return updated


@router.delete(
    "/{type}/{uuid}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom record",
)
async def delete_custom_entity(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    uuid: str = UUID_PARAM,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> None:
    """Delete a record (owner or admin only)."""
    # Type-level gate first; the service still enforces per-record owner-or-admin.
    _require_mutate_scope("delete", type, user_context)
    service = _get_service()
    path = f"/{type}/{uuid}"
    try:
        await service.delete_record(type, path, user_context)
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    set_audit_action(
        http_request,
        "delete",
        "custom_entity",
        resource_id=path,
        description=f"Delete {type} record {path}",
    )


@router.post("/{type}/{uuid}/rate", summary="Rate a custom record")
async def rate_custom_entity(
    http_request: Request,
    rating_request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    uuid: str = UUID_PARAM,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Add or update the caller's 1-5 rating on a record they can view."""
    path = f"/{type}/{uuid}"
    _require_view_scope(type, user_context, record_path=path)
    service = _get_service()
    set_audit_action(
        http_request,
        "rate",
        "custom_entity",
        resource_id=path,
        description=f"Rate {type} record with {rating_request.rating}",
    )
    try:
        average = await service.update_rating(
            path, user_context["username"], rating_request.rating, user_context
        )
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {"message": "Rating added successfully", "average_rating": average}


@router.get("/{type}/{uuid}/rating", summary="Get a custom record's rating")
async def get_custom_entity_rating(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    uuid: str = UUID_PARAM,
) -> dict:
    """Return {num_stars, rating_details} for a record the caller can view."""
    path = f"/{type}/{uuid}"
    _require_view_scope(type, user_context, record_path=path)
    service = _get_service()
    try:
        return await service.get_rating(path, user_context)
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
