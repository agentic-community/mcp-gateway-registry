import logging
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
from .server_internal_routes import (
    internal_add_server_to_groups,
    internal_create_group,
    internal_delete_group,
    internal_list_groups,
    internal_remove_server_from_groups,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/servers/groups/add")
async def add_server_to_groups_api(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Add a service to scope groups via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/add-to-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `server_name` (required): Service name
    - `group_names` (required): Comma-separated list of group names

    **Response:**
    Returns confirmation of group assignment.
    """
    logger.info(
        "API add to groups request from user '%s' for server '%s'",
        user_context.get("username") if user_context else "unknown",
        server_name,
    )

    return await internal_add_server_to_groups(
        request,
        server_name,
        group_names,
        user_context=user_context,
    )


@router.post("/servers/groups/remove")
async def remove_server_from_groups_api(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Remove a service from scope groups via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/remove-from-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `server_name` (required): Service name
    - `group_names` (required): Comma-separated list of group names to remove

    **Response:**
    Returns confirmation of removal from groups.
    """
    logger.info(
        "API remove from groups request from user '%s' for server '%s'",
        user_context.get("username") if user_context else "unknown",
        server_name,
    )

    return await internal_remove_server_from_groups(
        request,
        server_name,
        group_names,
        user_context=user_context,
    )


@router.post("/servers/groups/create")
async def create_group_api(
    request: Request,
    group_name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    create_in_keycloak: Annotated[bool, Form()] = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Create a new scope group via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/create-group
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `group_name` (required): Name of the new group
    - `description` (optional): Group description
    - `create_in_keycloak` (optional): Whether to create in Keycloak (default: true)

    **Response:**
    Returns confirmation of group creation.
    """
    logger.info(
        "API create group request from user '%s' for group '%s'",
        user_context.get("username") if user_context else "unknown",
        group_name,
    )

    return await internal_create_group(
        request,
        group_name,
        description,
        create_in_keycloak,
        user_context=user_context,
    )


@router.post("/servers/groups/delete")
async def delete_group_api(
    request: Request,
    group_name: Annotated[str, Form()],
    delete_from_keycloak: Annotated[bool, Form()] = True,
    force: Annotated[bool, Form()] = False,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Delete a scope group via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/delete-group
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `group_name` (required): Name of the group to delete
    - `delete_from_keycloak` (optional): Whether to delete from Keycloak (default: true)
    - `force` (optional): Force deletion of system groups (default: false)

    **Response:**
    Returns confirmation of group deletion.
    """
    logger.info(
        "API delete group request from user '%s' for group '%s'",
        user_context.get("username") if user_context else "unknown",
        group_name,
    )

    return await internal_delete_group(
        request,
        group_name,
        delete_from_keycloak,
        force,
        user_context=user_context,
    )


@router.get("/servers/groups")
async def list_groups_api(
    request: Request,
    include_keycloak: bool = True,
    include_scopes: bool = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    List all scope groups via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as GET /api/internal/list-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Response:**
    Returns a list of all groups and their synchronization status.
    """
    logger.info(
        "API list groups request from user '%s'",
        user_context.get("username") if user_context else "unknown",
    )

    return await internal_list_groups(
        request,
        include_keycloak,
        include_scopes,
        user_context=user_context,
    )

