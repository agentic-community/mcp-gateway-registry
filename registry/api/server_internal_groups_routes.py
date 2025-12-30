from __future__ import annotations

from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    Form,
    Request,
)

from ..auth.dependencies import (
    nginx_proxied_auth,
)
from .server_group_ops import (
    _add_server_to_groups,
    _create_group,
    _delete_group,
    _list_groups,
    _remove_server_from_groups,
)
from .server_routes_common import (
    _require_admin_user_context,
)

router = APIRouter()


@router.post("/internal/add-to-groups")
async def internal_add_server_to_groups(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to add a server to specific scopes groups (requires admin auth)."""
    return await _add_server_to_groups(
        server_name=server_name,
        group_names=group_names,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
    )


@router.post("/internal/remove-from-groups")
async def internal_remove_server_from_groups(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to remove a server from specific scopes groups (requires admin auth)."""
    return await _remove_server_from_groups(
        server_name=server_name,
        group_names=group_names,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
    )


@router.post("/internal/create-group")
async def internal_create_group(
    request: Request,
    group_name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    create_in_keycloak: Annotated[bool, Form()] = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to create a new group in both Keycloak and scopes.yml (requires admin auth)."""
    return await _create_group(
        group_name=group_name,
        description=description,
        create_in_keycloak=create_in_keycloak,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
    )


@router.post("/internal/delete-group")
async def internal_delete_group(
    request: Request,
    group_name: Annotated[str, Form()],
    delete_from_keycloak: Annotated[bool, Form()] = True,
    force: Annotated[bool, Form()] = False,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to delete a group from both Keycloak and scopes.yml (requires admin auth)."""
    return await _delete_group(
        group_name=group_name,
        delete_from_keycloak=delete_from_keycloak,
        force=force,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
    )


@router.get("/internal/list-groups")
async def internal_list_groups(
    request: Request,
    include_keycloak: bool = True,
    include_scopes: bool = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Internal endpoint to list groups from Keycloak and/or scopes.yml (requires admin auth)."""
    return await _list_groups(
        include_keycloak=include_keycloak,
        include_scopes=include_scopes,
        user_context=user_context,
        require_user_context=_require_admin_user_context,
    )
