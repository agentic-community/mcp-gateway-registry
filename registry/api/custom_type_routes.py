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
from ..auth.csrf import verify_csrf_token_flexible
from ..auth.dependencies import nginx_proxied_auth
from ..schemas.custom_entity_models import CustomTypeDescriptor, CustomTypeUpdate
from ..services.custom_entity_errors import (
    CustomEntityValidationError,
    CustomTypeAlreadyExistsError,
    CustomTypeHasRecordsError,
    CustomTypeLimitError,
)
from ..services.custom_entity_service import CustomEntityService
from ..services.scope_service import (
    ScopeMintError,
    cleanup_custom_type_scopes,
    mint_custom_type_scopes,
    trigger_auth_server_reload,
)

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
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
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

    # Mint the per-type scope set to the admin group so admins can manage this
    # type's records immediately. FATAL: a type whose scopes could not be granted
    # is unusable. Roll back the just-created descriptor so the type-create stays
    # atomic -- otherwise the orphaned descriptor makes an identical retry fail
    # with 409 (already-exists) even though its scopes were never provisioned.
    try:
        await mint_custom_type_scopes(created.name)
    except ScopeMintError as e:
        logger.error(f"Failed to mint scopes for custom type {created.name}: {e}")
        try:
            # No records can exist yet (the type was created moments ago), so a
            # non-forced delete is sufficient and leaves the scope store untouched.
            await service.delete_type(created.name, force=False)
            logger.info(f"Rolled back custom type {created.name} after scope-mint failure")
        except Exception:
            logger.exception(
                "Failed to roll back custom type %s after scope-mint failure; "
                "the descriptor is orphaned and a retry may 409",
                created.name,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom type scope provisioning failed and the type was rolled back: {e}",
        )

    # Reload is non-fatal: the auth server self-heals on its next periodic reload.
    try:
        await trigger_auth_server_reload()
    except Exception:
        logger.warning(
            "Auth-server reload after minting scopes for %s failed (self-heals)", created.name
        )

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
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
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
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> None:
    """Delete a custom type and (with force) cascade-delete its records. Admin only."""
    _require_admin(user_context)

    service = _get_service()
    try:
        count = await service.delete_type(name, force=force)
    except CustomTypeHasRecordsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    # Scope cleanup runs AFTER record/descriptor teardown (matches the existing
    # embeddings -> records -> descriptor order). Non-fatal: a dangling scope on a
    # deleted type is harmless (no records remain), and the backfill/sweep can
    # re-run. Sweep ALL groups so a granted non-admin loses the scope too.
    try:
        await cleanup_custom_type_scopes(name)
        await trigger_auth_server_reload()
    except Exception:
        logger.warning("Scope cleanup/reload after deleting custom type %s failed", name)

    set_audit_action(
        http_request,
        "delete",
        "custom_type",
        resource_id=name,
        description=f"Delete custom type {name} (cascaded {count} records)",
    )
    logger.info(f"Custom type deleted: {name} (cascaded {count} records)")
