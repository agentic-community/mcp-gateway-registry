"""

Anthropic MCP Registry API endpoints.

Implements the standard MCP Registry REST API for compatibility with
Anthropic's official registry specification.

Spec: https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/api/openapi.yaml
"""

import logging
from datetime import UTC, datetime
from typing import Annotated
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth.dependencies import nginx_proxied_auth
from ..constants import REGISTRY_CONSTANTS
from ..health.service import health_service
from ..repositories.factory import get_registry_card_repository
from ..schemas.anthropic_schema import ErrorResponse, ServerList, ServerResponse
from ..schemas.registry_card import RegistryCard
from ..services.server_service import server_service
from ..services.transform_service import (
    transform_to_server_list,
    transform_to_server_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"/{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION}",
    tags=["Anthropic Registry API"],
)


@router.get(
    "/servers",
    response_model=ServerList,
    summary="List MCP servers",
    description="Returns a paginated list of all registered MCP servers that the authenticated user can access.",
)
async def list_servers(
    cursor: Annotated[str | None, Query(description="Pagination cursor")] = None,
    limit: Annotated[
        int | None, Query(description="Maximum number of items", ge=1, le=1000)
    ] = None,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> ServerList:
    """
    List all MCP servers with pagination.

    This endpoint respects user permissions - users only see servers they have access to.

    Args:
        cursor: Pagination cursor (opaque string from previous response)
        limit: Max results per page (default: 100, max: 1000)
        user_context: Authenticated user context from enhanced_auth

    Returns:
        ServerList with servers and pagination metadata
    """
    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Listing servers for user '{user_context['username']}' (cursor={cursor}, limit={limit})"
    )

    # Get servers based on user permissions (same logic as existing /servers endpoint)
    if user_context["is_admin"]:
        # Admin sees all servers
        all_servers = await server_service.get_all_servers()
        logger.debug(f"Admin user accessing all {len(all_servers)} servers")
    else:
        # Regular user sees only accessible servers
        all_servers = await server_service.get_all_servers_with_permissions(
            user_context["accessible_servers"]
        )
        logger.debug(f"User accessing {len(all_servers)} accessible servers")

    # For API, we don't need UI service filtering - accessible_servers already handles MCP server permissions
    # No additional filtering needed here - the get_all_servers_with_permissions already filtered by accessible_servers
    filtered_servers = []

    for path, server_info in all_servers.items():
        # Fetch enabled status before health check to avoid race condition (Issue #612)
        is_enabled = await server_service.is_service_enabled(path)

        # Add health status with current enabled state
        health_data = health_service._get_service_health_data(
            path,
            {**server_info, "is_enabled": is_enabled},
        )

        server_info_with_status = server_info.copy()
        server_info_with_status["health_status"] = health_data["status"]
        server_info_with_status["last_checked_iso"] = health_data["last_checked_iso"]
        server_info_with_status["is_enabled"] = is_enabled

        filtered_servers.append(server_info_with_status)

    # Transform to Anthropic format with pagination
    server_list = transform_to_server_list(filtered_servers, cursor=cursor, limit=limit or 100)

    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Returning {len(server_list.servers)} servers (hasMore={server_list.metadata.nextCursor is not None})"
    )

    return server_list


@router.get(
    "/servers/{serverName:path}/versions",
    response_model=ServerList,
    summary="List server versions",
    description="Returns all available versions for a specific MCP server.",
    responses={404: {"model": ErrorResponse, "description": "Server not found"}},
)
async def list_server_versions(
    serverName: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> ServerList:
    """
    List all versions of a specific server.

    Currently, we only maintain one version per server, so this returns a single-item list.

    Args:
        serverName: URL-encoded server name in reverse-DNS format (e.g., "io.mcpgateway%2Fexample-server")
        user_context: Authenticated user context

    Returns:
        ServerList with single version

    Raises:
        HTTPException: 404 if server not found or user lacks access
    """
    # URL-decode the server name
    decoded_name = unquote(serverName)
    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Listing versions for server '{decoded_name}' (user='{user_context['username']}')"
    )

    # Extract path from reverse-DNS name
    # Expected format: "io.mcpgateway/example-server"
    namespace = REGISTRY_CONSTANTS.ANTHROPIC_SERVER_NAMESPACE
    expected_prefix = f"{namespace}/"

    if not decoded_name.startswith(expected_prefix):
        logger.warning(f"Invalid server name format: {decoded_name}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Construct initial path for lookup
    lookup_path = "/" + decoded_name.replace(expected_prefix, "")

    # Get server info - try with and without trailing slash
    server_info = await server_service.get_server_info(lookup_path)
    if not server_info:
        # Try with trailing slash
        server_info = await server_service.get_server_info(lookup_path + "/")

    if not server_info:
        logger.warning(f"Server not found: {lookup_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Use the actual path from server_info (has correct trailing slash)
    path = server_info.get("path", lookup_path)

    # Check user permissions - use accessible_servers (MCP scopes) not accessible_services (UI scopes)
    accessible_servers = user_context.get("accessible_servers", [])
    server_name = server_info["server_name"]

    if not user_context["is_admin"]:
        # Check if user can access this server
        if server_name not in accessible_servers:
            logger.warning(
                f"User '{user_context['username']}' attempted to access unauthorized server: {server_name}"
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Fetch enabled status before health check to avoid race condition (Issue #612)
    is_enabled = await server_service.is_service_enabled(path)

    # Add health and status info using the correct path
    health_data = health_service._get_service_health_data(
        path,
        {**server_info, "is_enabled": is_enabled},
    )

    server_info_with_status = server_info.copy()
    server_info_with_status["health_status"] = health_data["status"]
    server_info_with_status["last_checked_iso"] = health_data["last_checked_iso"]
    server_info_with_status["is_enabled"] = is_enabled

    # Since we only have one version, return a list with one item
    server_list = transform_to_server_list([server_info_with_status])

    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Returning version info for {decoded_name}"
    )

    return server_list


@router.get(
    "/servers/{serverName:path}/versions/{version}",
    response_model=ServerResponse,
    summary="Get server version details",
    description="Returns detailed information about a specific version of an MCP server. Use 'latest' to get the most recent version.",
    responses={404: {"model": ErrorResponse, "description": "Server or version not found"}},
)
async def get_server_version(
    serverName: str,
    version: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> ServerResponse:
    """
    Get detailed information about a specific server version.

    Args:
        serverName: URL-encoded server name (e.g., "io.mcpgateway%2Fexample-server")
        version: Version string (e.g., "1.0.0" or "latest")
        user_context: Authenticated user context

    Returns:
        ServerResponse with full server details

    Raises:
        HTTPException: 404 if server not found or user lacks access
    """
    # URL-decode parameters
    decoded_name = unquote(serverName)
    decoded_version = unquote(version)

    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Getting server '{decoded_name}' version '{decoded_version}' (user='{user_context['username']}')"
    )

    # Extract path from reverse-DNS name
    namespace = REGISTRY_CONSTANTS.ANTHROPIC_SERVER_NAMESPACE
    expected_prefix = f"{namespace}/"

    if not decoded_name.startswith(expected_prefix):
        logger.warning(f"Invalid server name format: {decoded_name}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Construct initial path for lookup
    lookup_path = "/" + decoded_name.replace(expected_prefix, "")

    # Get server info - try with and without trailing slash
    server_info = await server_service.get_server_info(lookup_path)
    if not server_info:
        # Try with trailing slash
        server_info = await server_service.get_server_info(lookup_path + "/")

    if not server_info:
        logger.warning(f"Server not found: {lookup_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Use the actual path from server_info (has correct trailing slash)
    path = server_info.get("path", lookup_path)

    # Check user permissions - use accessible_servers (MCP scopes) not accessible_services (UI scopes)
    accessible_servers = user_context.get("accessible_servers", [])
    server_name = server_info["server_name"]

    if not user_context["is_admin"]:
        if server_name not in accessible_servers:
            logger.warning(
                f"User '{user_context['username']}' attempted to access unauthorized server: {server_name}"
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

    # Currently we only support "latest" or "1.0.0" since we don't version servers
    if decoded_version not in ["latest", "1.0.0"]:
        logger.warning(f"Unsupported version requested: {decoded_version}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {decoded_version} not found",
        )

    # Fetch enabled status before health check to avoid race condition (Issue #612)
    is_enabled = await server_service.is_service_enabled(path)

    # Add health and status info
    health_data = health_service._get_service_health_data(
        path,
        {**server_info, "is_enabled": is_enabled},
    )

    server_info_with_status = server_info.copy()
    server_info_with_status["health_status"] = health_data["status"]
    server_info_with_status["last_checked_iso"] = health_data["last_checked_iso"]
    server_info_with_status["is_enabled"] = is_enabled

    # Transform to Anthropic format
    server_response = transform_to_server_response(
        server_info_with_status, include_registry_meta=True
    )

    logger.info(
        f"{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION} API: Returning details for {decoded_name} v{decoded_version}"
    )

    return server_response


async def _auto_initialize_registry_card():
    """
    Auto-initialize registry card from config defaults if it doesn't exist.

    Returns the existing or newly created card.
    """
    repo = get_registry_card_repository()
    card = await repo.get()

    if card is None:
        # Auto-initialize from config defaults
        import random
        from uuid import uuid4

        from registry.core.config import settings
        from registry.schemas.registry_card import RegistryContact
        from registry.version import __version__

        logger.info("Registry card not found, auto-initializing from config")

        # Generate UUID for registry_id if not configured
        if settings.registry_id:
            registry_id = settings.registry_id
        else:
            registry_id = str(uuid4())
            logger.info(f"Generated UUID for registry_id: {registry_id}")

        # Generate random Docker-style registry name if using default
        if settings.registry_name != "AI Registry":
            registry_name = settings.registry_name
        else:
            adjectives = ["brave", "clever", "swift", "bright", "noble", "wise", "bold", "keen"]
            nouns = ["falcon", "dolphin", "tiger", "phoenix", "dragon", "wolf", "eagle", "lion"]
            registry_name = f"{random.choice(adjectives)}-{random.choice(nouns)}-registry"
            logger.info(f"Generated random registry name: {registry_name}")

        # Use organization name from config (defaults to "ACME Inc.")
        organization_name = settings.registry_organization_name
        logger.info(f"Using organization name: {organization_name}")

        # Get full API version from version module (e.g., "1.0.17")
        version_str = __version__
        # Remove 'v' prefix if present (e.g., "v1.0.17" -> "1.0.17")
        if version_str.startswith("v"):
            version_str = version_str[1:]
        # Remove git suffix if present (e.g., "1.0.17-6-gf5c000c3-main" -> "1.0.17")
        version_parts = version_str.split("-")[0]
        federation_api_version = version_parts
        logger.info(
            f"Using federation API version: {federation_api_version} (from app version: {__version__})"
        )

        contact = None
        if settings.registry_contact_email or settings.registry_contact_url:
            contact = RegistryContact(
                email=settings.registry_contact_email,
                url=settings.registry_contact_url,
            )

        # Build OAuth params based on auth provider
        oauth2_issuer = None
        oauth2_token_endpoint = None

        if settings.auth_provider == "okta":
            import os

            okta_domain = os.getenv("OKTA_DOMAIN")
            okta_auth_server_id = os.getenv("OKTA_AUTH_SERVER_ID", "default")
            if okta_domain:
                oauth2_issuer = f"https://{okta_domain}/oauth2/{okta_auth_server_id}"
                oauth2_token_endpoint = (
                    f"https://{okta_domain}/oauth2/{okta_auth_server_id}/v1/token"
                )
        elif settings.auth_provider == "keycloak":
            import os

            keycloak_external_url = os.getenv("KEYCLOAK_EXTERNAL_URL", "http://localhost:8080")
            keycloak_realm = os.getenv("KEYCLOAK_REALM", "mcp-gateway")
            oauth2_issuer = f"{keycloak_external_url}/realms/{keycloak_realm}"
            oauth2_token_endpoint = (
                f"{keycloak_external_url}/realms/{keycloak_realm}/protocol/openid-connect/token"
            )
        elif settings.auth_provider == "entra":
            import os

            entra_tenant_id = os.getenv("ENTRA_TENANT_ID")
            if entra_tenant_id:
                oauth2_issuer = f"https://login.microsoftonline.com/{entra_tenant_id}/v2.0"
                oauth2_token_endpoint = (
                    f"https://login.microsoftonline.com/{entra_tenant_id}/oauth2/v2.0/token"
                )
        elif settings.auth_provider == "cognito":
            import os

            cognito_user_pool_id = os.getenv("COGNITO_USER_POOL_ID")
            cognito_domain = os.getenv("COGNITO_DOMAIN")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            if cognito_user_pool_id:
                oauth2_issuer = (
                    f"https://cognito-idp.{aws_region}.amazonaws.com/{cognito_user_pool_id}"
                )
            if cognito_domain:
                oauth2_token_endpoint = (
                    f"https://{cognito_domain}.auth.{aws_region}.amazoncognito.com/oauth2/token"
                )

        from registry.schemas.registry_card import RegistryAuthConfig

        auth_config = RegistryAuthConfig(
            oauth2_issuer=oauth2_issuer,
            oauth2_token_endpoint=oauth2_token_endpoint,
        )

        card = RegistryCard(
            id=registry_id,
            name=registry_name,
            description=settings.registry_description,
            registry_url=settings.registry_url,
            organization_name=organization_name,
            federation_api_version=federation_api_version,
            federation_endpoint=f"{settings.registry_url}/api/v1/federation",
            authentication=auth_config,
            contact=contact,
        )

        # Save the auto-initialized card
        card = await repo.save(card)
        logger.info(f"Auto-initialized registry card: {card.id}")

    return card


@router.get("/card")
async def get_registry_card():
    """
    Get the Registry Card for this instance.

    Auto-initializes from config if not found.
    Public endpoint for federation discovery.
    Returns flattened contact fields for frontend compatibility.
    """
    card = await _auto_initialize_registry_card()

    # Serialize to dict and flatten contact fields for frontend
    card_dict = card.model_dump(mode="json")
    contact = card_dict.pop("contact", None)
    if contact:
        card_dict["contact_email"] = contact.get("email")
        card_dict["contact_url"] = contact.get("url")
    else:
        card_dict["contact_email"] = None
        card_dict["contact_url"] = None

    return card_dict


@router.post("/card", response_model=dict)
async def update_registry_card(
    request: dict,
    user_context: dict = Depends(nginx_proxied_auth),
):
    """
    Create or update the Registry Card.

    Requires admin role. All updates are audit logged.
    """
    # Check admin permissions
    username = user_context.get("username", "unknown")
    is_admin = user_context.get("is_admin", False)

    if not is_admin:
        logger.warning(
            "Unauthorized registry card update attempt",
            extra={"username": username, "is_admin": is_admin},
        )
        raise HTTPException(
            status_code=403,
            detail="Admin role required to update registry card",
        )

    repo = get_registry_card_repository()

    # Get existing card or create new
    existing = await repo.get()
    operation = "update" if existing else "create"

    # Handle nested contact fields from frontend (flat) to backend (nested)
    from registry.core.config import settings
    from registry.schemas.registry_card import RegistryContact

    if existing:
        # Update existing card
        card_data = existing.model_dump()
        old_values = {k: v for k, v in card_data.items() if k in request}

        if "contact_email" in request or "contact_url" in request:
            # Build contact object from flat fields
            existing_contact = card_data.get("contact") or {}
            contact_data = {
                "email": request.get("contact_email", existing_contact.get("email")),
                "url": request.get("contact_url", existing_contact.get("url")),
            }
            # Only create contact if at least one field is non-null
            if contact_data["email"] or contact_data["url"]:
                card_data["contact"] = RegistryContact(**contact_data).model_dump()
            else:
                card_data["contact"] = None

            # Remove flat fields
            request_cleaned = {
                k: v for k, v in request.items() if k not in ["contact_email", "contact_url"]
            }
            card_data.update(request_cleaned)
        else:
            card_data.update(request)

        card = RegistryCard(**card_data)
    else:
        # Create new card
        request_cleaned = request.copy()

        if "contact_email" in request or "contact_url" in request:
            # Build contact object from flat fields
            contact_data = {
                "email": request.get("contact_email"),
                "url": request.get("contact_url"),
            }
            if contact_data["email"] or contact_data["url"]:
                request_cleaned["contact"] = RegistryContact(**contact_data)

            # Remove flat fields
            request_cleaned.pop("contact_email", None)
            request_cleaned.pop("contact_url", None)

        card = RegistryCard(
            registry_id=settings.registry_id or "default",
            federation_endpoint=settings.registry_url + "/api",
            **request_cleaned,
        )
        old_values = {}

    saved = await repo.save(card)

    # Audit log
    logger.info(
        f"Registry card {operation} by admin",
        extra={
            "operation": operation,
            "username": username,
            "timestamp": datetime.now(UTC).isoformat(),
            "registry_id": str(saved.id),
            "changed_fields": list(request.keys()),
            "old_values": old_values if operation == "update" else None,
        },
    )

    # Flatten contact fields for frontend response
    saved_dict = saved.model_dump(mode="json")
    contact = saved_dict.pop("contact", None)
    if contact:
        saved_dict["contact_email"] = contact.get("email")
        saved_dict["contact_url"] = contact.get("url")
    else:
        saved_dict["contact_email"] = None
        saved_dict["contact_url"] = None

    return {
        "message": f"Registry card {operation}d successfully",
        "registry_card": saved_dict,
    }


@router.patch("/card", response_model=dict)
async def patch_registry_card(
    request: dict,
    user_context: dict = Depends(nginx_proxied_auth),
):
    """
    Partially update the Registry Card.

    Requires admin role. Only updates provided fields.
    """
    # Check admin permissions
    username = user_context.get("username", "unknown")
    is_admin = user_context.get("is_admin", False)

    if not is_admin:
        logger.warning(
            "Unauthorized registry card update attempt",
            extra={"username": username, "is_admin": is_admin},
        )
        raise HTTPException(
            status_code=403,
            detail="Admin role required to update registry card",
        )

    repo = get_registry_card_repository()

    # Get existing card
    existing = await repo.get()
    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Registry card not found. Use POST to create.",
        )

    # Partial update
    card_data = existing.model_dump()
    old_values = {k: v for k, v in card_data.items() if k in request}

    # Handle nested contact fields from frontend (flat) to backend (nested)
    from registry.schemas.registry_card import RegistryContact

    if "contact_email" in request or "contact_url" in request:
        # Build contact object from flat fields
        existing_contact = card_data.get("contact") or {}
        contact_data = {
            "email": request.get("contact_email", existing_contact.get("email")),
            "url": request.get("contact_url", existing_contact.get("url")),
        }
        # Only create contact if at least one field is non-null
        if contact_data["email"] or contact_data["url"]:
            card_data["contact"] = RegistryContact(**contact_data).model_dump()
        else:
            card_data["contact"] = None

        # Remove flat fields from request before updating
        request_cleaned = {
            k: v for k, v in request.items() if k not in ["contact_email", "contact_url"]
        }
        card_data.update(request_cleaned)
    else:
        card_data.update(request)

    card = RegistryCard(**card_data)

    saved = await repo.save(card)

    # Audit log
    logger.info(
        "Registry card partially updated by admin",
        extra={
            "operation": "patch",
            "username": username,
            "timestamp": datetime.now(UTC).isoformat(),
            "registry_id": str(saved.id),
            "changed_fields": list(request.keys()),
            "old_values": old_values,
        },
    )

    # Flatten contact fields for frontend response
    saved_dict = saved.model_dump(mode="json")
    contact = saved_dict.pop("contact", None)
    if contact:
        saved_dict["contact_email"] = contact.get("email")
        saved_dict["contact_url"] = contact.get("url")
    else:
        saved_dict["contact_email"] = None
        saved_dict["contact_url"] = None

    return {
        "message": "Registry card updated successfully",
        "registry_card": saved_dict,
    }
