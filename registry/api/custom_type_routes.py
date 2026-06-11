"""
API routes for custom entity TYPE descriptors (admin-only).

Admins define schema-driven catalog types (e.g. n8n workflows, rules) at
runtime. These routes manage the type DESCRIPTORS; record CRUD lives in
``custom_entity_routes``. The ``{name}`` path segment is interpolated into
Mongo queries, so it is constrained at the signature with ``TYPE_PARAM``
(NoSQL-injection guard).
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

from ..audit.context import set_audit_action
from ..auth.dependencies import nginx_proxied_auth
from ..schemas.custom_entity_models import CustomTypeDescriptor, CustomTypeUpdate
from ..services.custom_entity_errors import (
    CustomEntityValidationError,
    CustomTypeAlreadyExistsError,
    CustomTypeHasRecordsError,
    CustomTypeLimitError,
)
from ..services.custom_entity_service import CustomEntityService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/custom-types", tags=["custom-types"])

# NoSQL-injection guard: the {name} segment is interpolated into
# find_one({"_id": name}) / find({"entity_type": name}); constrain it here so
# a dict-shaped operator can never reach the query.
TYPE_PARAM = Path(..., pattern=r"^[a-z0-9_-]+$", max_length=64)


def _require_admin(
    user_context: dict,
) -> None:
    """Raise HTTP 403 unless the caller has registry-admin privileges."""
    is_admin = user_context.get("is_admin", False)
    groups = user_context.get("groups", [])
    scopes = user_context.get("scopes", [])
    if not (is_admin or "mcp-registry-admin" in groups or "mcp-registry-admin" in scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required to manage custom types",
        )


def _get_service() -> CustomEntityService:
    """Resolve the custom entity service singleton."""
    from ..repositories.factory import get_custom_entity_service

    return get_custom_entity_service()


@router.get("", summary="List all custom type descriptors")
async def list_custom_types(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> dict:
    """Return all defined custom type descriptors.

    Readable by any authenticated user — the descriptors describe the schema,
    not the records, so they carry no record-level visibility.
    """
    service = _get_service()
    descriptors = await service.list_types()
    return {
        "custom_types": [d.model_dump(mode="json") for d in descriptors],
        "total_count": len(descriptors),
    }


@router.get("/{name}", summary="Get a custom type descriptor")
async def get_custom_type(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    name: str = TYPE_PARAM,
) -> CustomTypeDescriptor:
    """Return a single custom type descriptor by name."""
    service = _get_service()
    descriptor = await service.get_type(name)
    if descriptor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown custom type: {name}",
        )
    return descriptor


@router.post(
    "",
    response_model=CustomTypeDescriptor,
    status_code=status.HTTP_201_CREATED,
    summary="Define a new custom type (admin)",
)
async def create_custom_type(
    http_request: Request,
    descriptor: CustomTypeDescriptor,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> CustomTypeDescriptor:
    """Define a new custom entity type. Admin only."""
    _require_admin(user_context)

    descriptor.created_by = user_context.get("username")

    set_audit_action(
        http_request,
        "create",
        "custom_type",
        resource_id=descriptor.name,
        description=f"Define custom type {descriptor.name} ({len(descriptor.fields)} fields)",
    )

    service = _get_service()
    try:
        created = await service.create_type(descriptor)
    except CustomTypeAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CustomTypeLimitError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CustomEntityValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors)

    logger.info(f"Custom type defined: {created.name} by {user_context.get('username')}")
    return created


@router.patch(
    "/{name}",
    response_model=CustomTypeDescriptor,
    summary="Update a custom type's mutable metadata (admin)",
)
async def update_custom_type(
    http_request: Request,
    updates: CustomTypeUpdate,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    name: str = TYPE_PARAM,
) -> CustomTypeDescriptor:
    """Update a custom type's display_name/description. Admin only.

    The type name and field schema are IMMUTABLE; only the human-facing
    display_name and description can change. Renaming a type or altering its
    fields is not supported (it would orphan existing records and embeddings);
    delete and recreate the type for that.
    """
    _require_admin(user_context)

    set_audit_action(
        http_request,
        "update",
        "custom_type",
        resource_id=name,
        description=f"Update custom type {name} metadata",
    )

    service = _get_service()
    updated = await service.update_type(name, updates)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown custom type: {name}",
        )

    logger.info(f"Custom type updated: {name} by {user_context.get('username')}")
    return updated


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom type (admin, cascading)",
)
async def delete_custom_type(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    name: str = TYPE_PARAM,
    force: bool = Query(
        False,
        description="Cascade-delete all records of this type. Required when records exist.",
    ),
) -> None:
    """Delete a custom type and (with force) cascade-delete its records. Admin only."""
    _require_admin(user_context)

    service = _get_service()
    try:
        count = await service.delete_type(name, force=force)
    except CustomTypeHasRecordsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    set_audit_action(
        http_request,
        "delete",
        "custom_type",
        resource_id=name,
        description=f"Delete custom type {name} (cascaded {count} records)",
    )
    logger.info(f"Custom type deleted: {name} (cascaded {count} records)")
