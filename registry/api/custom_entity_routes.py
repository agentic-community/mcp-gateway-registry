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


@router.get("/{type}", summary="List records of a custom type")
async def list_custom_entities(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    type: str = TYPE_PARAM,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
) -> dict:
    """List records of a type, filtered to those the caller may see."""
    service = _get_service()
    try:
        items, total = await service.list_records(type, skip, limit, user_context)
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
    service = _get_service()
    path = f"/{type}/{uuid}"
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
) -> CustomEntityRecord:
    """Create a record of the given custom type."""
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
) -> CustomEntityRecord:
    """Update a record (owner or admin only; partial-update semantics)."""
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
) -> None:
    """Delete a record (owner or admin only)."""
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
) -> dict:
    """Add or update the caller's 1-5 rating on a record they can view."""
    service = _get_service()
    path = f"/{type}/{uuid}"
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
    service = _get_service()
    path = f"/{type}/{uuid}"
    try:
        return await service.get_rating(path, user_context)
    except CustomEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
