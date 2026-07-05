import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..auth.oauth_metadata import (
    build_canonical_resource_url,
    build_resource_documentation_url,
    derive_supported_scopes,
    enforce_https,
)
from ..core.config import settings
from ..repositories.factory import get_registry_card_repository
from ..schemas.registry_card import RegistryCard, RegistryContact
from ..services import ard_service

logger = logging.getLogger(__name__)

router = APIRouter()


OAUTH_DISCOVERY_CACHE_HEADERS: dict[str, str] = {
    "Cache-Control": "public, max-age=300",
    "Content-Type": "application/json",
}


@router.get("/ai-catalog.json")
async def get_ai_catalog(
    request: Request,
) -> JSONResponse:
    """Public ARD Catalog Publisher manifest (Agentic Resource Discovery v1.0).

    Renders public + enabled MCP servers, A2A agents, and skills as ARD catalog
    entries. No authentication: a `.well-known` document is crawled anonymously.
    See issue #1294.
    """
    if not settings.enable_wellknown_discovery or not settings.ard_catalog_enabled:
        raise HTTPException(status_code=404, detail="ARD catalog is disabled")

    manifest = await ard_service.build_catalog(request)
    payload = manifest.model_dump(by_alias=True, exclude_none=True)
    headers = {
        "Cache-Control": f"public, max-age={settings.wellknown_cache_ttl}",
        "Content-Type": "application/json",
    }
    return JSONResponse(content=payload, headers=headers)


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

        from registry.version import __version__

        logger.info("Registry card not found, auto-initializing from config")

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
        import os

        oauth2_issuer = None
        oauth2_token_endpoint = None

        if settings.auth_provider == "okta":
            okta_domain = os.getenv("OKTA_DOMAIN")
            okta_auth_server_id = os.getenv("OKTA_AUTH_SERVER_ID", "default")
            if okta_domain:
                oauth2_issuer = f"https://{okta_domain}/oauth2/{okta_auth_server_id}"
                oauth2_token_endpoint = (
                    f"https://{okta_domain}/oauth2/{okta_auth_server_id}/v1/token"
                )
        elif settings.auth_provider == "keycloak":
            keycloak_external_url = os.getenv("KEYCLOAK_EXTERNAL_URL", "http://localhost:8080")
            keycloak_realm = os.getenv("KEYCLOAK_REALM", "mcp-gateway")
            oauth2_issuer = f"{keycloak_external_url}/realms/{keycloak_realm}"
            oauth2_token_endpoint = (
                f"{keycloak_external_url}/realms/{keycloak_realm}/protocol/openid-connect/token"
            )
        elif settings.auth_provider == "entra":
            entra_tenant_id = os.getenv("ENTRA_TENANT_ID")
            if entra_tenant_id:
                oauth2_issuer = f"https://login.microsoftonline.com/{entra_tenant_id}/v2.0"
                oauth2_token_endpoint = (
                    f"https://login.microsoftonline.com/{entra_tenant_id}/oauth2/v2.0/token"
                )
        elif settings.auth_provider == "cognito":
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

        # Don't pass id - let RegistryCard auto-generate UUID via default_factory
        # registry_id was for the old implementation, now we use auto-generated UUIDs
        card = RegistryCard(
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


@router.get("/registry-card", response_model=RegistryCard)
async def get_well_known_registry_card():
    """
    Get the Registry Card via .well-known discovery endpoint.

    This is the standard discovery endpoint for registry federation.
    Public endpoint - no authentication required.
    """
    card = await _auto_initialize_registry_card()
    return card


def _get_active_auth_provider():
    """Lazy import + factory call so route handlers don't pay the cost at import time."""
    from auth_server.providers.factory import get_auth_provider

    return get_auth_provider()


@router.get("/oauth-protected-resource")
async def get_oauth_protected_resource() -> JSONResponse:
    """
    Return the gateway's RFC 9728 Protected Resource Metadata document.

    This is the entry point for spec-compliant MCP clients (Claude Code,
    Claude.ai connectors, Cursor, etc.) to discover which authorization
    server protects this gateway and which scopes it recognizes.

    Per the MCP 2025-06-18 authorization spec, MCP servers MUST publish
    this document. The `resource` field is the canonical gateway URL, which
    is also the `resource_metadata` URL embedded in WWW-Authenticate 401
    headers (byte-for-byte match required by RFC 9728 §5.1).

    Public endpoint - no authentication required.
    """
    resource = build_canonical_resource_url(settings.registry_url)
    enforce_https(resource, https_required=settings.mcp_https_required)

    try:
        provider = _get_active_auth_provider()
    except Exception as exc:
        logger.exception("Failed to initialize auth provider for PRM document")
        raise HTTPException(
            status_code=500,
            # Public endpoint: do not reflect internal exception detail; logged above.
            detail="Auth provider not configured",
        ) from exc

    try:
        scopes_supported = await derive_supported_scopes()
        document = provider.protected_resource_metadata(
            resource=resource,
            scopes_supported=scopes_supported,
            resource_documentation=build_resource_documentation_url(),
        )
    except NotImplementedError as exc:
        logger.error(
            f"Active auth provider has not implemented authorization_server_metadata(): {exc}"
        )
        raise HTTPException(
            status_code=501,
            detail="OAuth discovery not implemented for the configured auth provider",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to build PRM document")
        raise HTTPException(
            status_code=502,
            detail="Could not build Protected Resource Metadata",
        ) from exc

    return JSONResponse(content=document, headers=OAUTH_DISCOVERY_CACHE_HEADERS)


@router.get("/oauth-authorization-server")
async def get_oauth_authorization_server() -> JSONResponse:
    """
    Return the configured IdP's RFC 8414 Authorization Server Metadata.

    This is a thin passthrough/normalization of the upstream IdP's metadata
    document, with provider-specific quirks flattened (e.g. Cognito's split
    endpoints rehomed onto the cognito-domain host).

    Public endpoint - no authentication required.
    """
    try:
        provider = _get_active_auth_provider()
    except Exception as exc:
        logger.exception("Failed to initialize auth provider for AS metadata")
        raise HTTPException(
            status_code=500,
            detail="Auth provider not configured",
        ) from exc

    try:
        document = provider.authorization_server_metadata()
    except NotImplementedError as exc:
        logger.error(
            f"Active auth provider has not implemented authorization_server_metadata(): {exc}"
        )
        raise HTTPException(
            status_code=501,
            detail="OAuth discovery not implemented for the configured auth provider",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to fetch authorization server metadata")
        raise HTTPException(
            status_code=502,
            detail="Could not fetch authorization server metadata",
        ) from exc

    return JSONResponse(content=document, headers=OAUTH_DISCOVERY_CACHE_HEADERS)
