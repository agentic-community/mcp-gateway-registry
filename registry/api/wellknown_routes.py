import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..auth.oauth_metadata import (
    build_canonical_resource_url,
    build_per_server_resource_url,
    build_resource_documentation_url,
    derive_supported_scopes,
    enforce_https,
    entra_forces_per_server_prm,
)
from ..core.config import settings
from ..repositories.factory import get_registry_card_repository
from ..schemas.registry_card import RegistryCard, RegistryContact
from ..services import ard_service
from ..services.server_service import server_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Lock to prevent TOCTOU race in registry-card auto-initialization.
_registry_card_init_lock = asyncio.Lock()


# OAuth discovery metadata (RFC 8414 authorization-server and RFC 9728
# protected-resource documents) points clients at the authorization server and
# token endpoint. It must never be served from a shared/intermediary/CDN cache:
# a poisoned cached copy would redirect every client to an attacker-controlled
# authorization server for the cache lifetime. Force "no-store" so only the
# origin ever answers these endpoints (no public/CDN caching, no revalidation
# window).
OAUTH_DISCOVERY_CACHE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store",
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

    Returns the existing or newly created card.  Uses double-checked locking
    to prevent the TOCTOU race where concurrent coroutines within
    the same worker both read None and both write.

    Note: the lock is per-worker (asyncio.Lock is not cross-process). With
    multiple uvicorn workers starting simultaneously, each may attempt one
    write. The underlying ``replace_one(..., upsert=True)`` is idempotent,
    so no data corruption occurs; the lock eliminates redundant writes
    within a single worker's event loop (the common case under load).
    """
    repo = get_registry_card_repository()
    card = await repo.get()
    if card is not None:
        return card

    # Slow path: card missing — serialize initialization under the lock.
    async with _registry_card_init_lock:
        # Re-read under lock; another coroutine may have initialized.
        card = await repo.get()
        if card is not None:
            return card
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
            registry_name = f"{random.choice(adjectives)}-{random.choice(nouns)}-registry"  # nosec B311 - cosmetic display name, not security-sensitive
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


def server_needs_per_server_prm(egress_auth_mode: str | None) -> bool:
    """Whether a server should advertise a per-server RFC 9728 PRM (vs the global one).

    The per-server PRM exists to satisfy Entra's strict resource/scope alignment
    (the ingress login must target a path-qualified App ID URI, not the bare
    origin). It applies to:

    - ``obo_exchange`` on any provider (the same-IdP exchange always logs the user
      in at the gateway against a per-server resource),
    - ``oauth_user`` ONLY on Entra. The 3LO vault's ingress leg is a gateway login
      too, but lenient IdPs (Keycloak/Cognito) accept the gateway-wide root PRM's
      bare origin + OIDC scopes, which is how Keycloak 3LO works today. We must NOT
      route those through the per-server PRM, or we'd change that working path (and
      force an exact connection-URL match they don't need). Only Entra requires it.
    - **every other server (including plain ``none``) ONLY on Entra** (issue #990).
      A plain server has no egress mode, but a spec-compliant coding assistant
      (Claude Code) still does RFC 8707 ingress OAuth against it, and Entra rejects
      the bare-origin resource the gateway-wide PRM advertises (see below). So on
      Entra, plain servers also need the per-server, connection-URL resource.

    On non-Entra IdPs, everything except the two egress modes above uses the
    gateway-wide PRM (unchanged behavior) -- lenient IdPs match the bare origin
    fine. Entra is the only provider that forces a per-server resource for plain
    servers, and the only one that incurs the per-connection-URL App ID URI
    registration cost.
    """
    if egress_auth_mode == "obo_exchange":
        return True
    if egress_auth_mode == "oauth_user":
        return entra_forces_per_server_prm(settings.auth_provider)
    # issue #990: on Entra, EVERY server -- including plain
    # (egress_auth_mode none) servers like airegistry-tools -- needs a per-server
    # PRM, because the bare-origin global PRM's resource is unmatchable to an
    # Entra App ID URI (trailing slash) at token exchange (AADSTS9010010). Only
    # the exact per-server connection URL satisfies both Claude (RFC 9728 §3.3)
    # and Entra. Requires each connection URL registered as an identifierUri.
    # Non-Entra IdPs keep the lenient origin-based global PRM (unchanged).
    # `entra_forces_per_server_prm` is the shared source of truth mirrored by the
    # auth-server's `_server_advertises_per_server_prm` (keep the two in sync).
    if entra_forces_per_server_prm(settings.auth_provider):
        return True
    return False


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


def _normalize_prm_server_path(server_path: str) -> str:
    """Reduce a path-aware PRM suffix to the registered server path.

    MCP clients doing RFC 9728 path-aware discovery request
    ``/.well-known/oauth-protected-resource/<server>/mcp`` (the connection URL's
    path). Strip a trailing ``/mcp`` transport segment and normalize to the
    leading-slash registered path (e.g. ``obo-echo/mcp`` -> ``/obo-echo``).
    """
    p = "/" + server_path.strip("/")
    if p.endswith("/mcp"):
        p = p[: -len("/mcp")]
    return p or "/"


@router.get("/oauth-protected-resource/{server_path:path}")
async def get_oauth_protected_resource_for_server(
    server_path: str,
) -> JSONResponse:
    """Per-server RFC 9728 PRM (path-aware discovery).

    Spec-compliant MCP clients (Claude Code, etc.) try the path-suffixed
    well-known URL first, derived from the per-server connection URL. We serve a
    document only for servers that need it (see ``server_needs_per_server_prm``):
    ``obo_exchange`` on any provider, ``oauth_user`` on Entra, and -- on Entra --
    **every server including plain ones** (issue #990). Everything else --
    including Keycloak/Cognito/Okta/Auth0 plain and 3LO servers, which work with
    the gateway-wide root PRM -- 404s here so the client falls back to the global
    PRM (unchanged behavior).

    The advertised ``resource`` is the **per-server connection URL** (e.g.
    ``https://gw/github/mcp``). This is the ONLY value that satisfies all three
    constraints simultaneously:
      - RFC 9728 §3.3: the client only accepts a PRM ``resource`` equal to the
        connection URL it is accessing (or the origin); a made-up shared path is
        rejected.
      - RFC 8707 canonicalization: a bare origin gets a trailing ``/`` on the
        wire, which Entra App ID URIs cannot match -- a path-qualified per-server
        URL is sent verbatim.
      - Entra: the sent ``resource`` must equal a registered App ID URI exactly.
    The trade-off (Entra only): each such server's per-server URL must be an App
    ID URI on the gateway app (operator maintains the ``identifierUris`` list;
    see GET /api/egress/obo-identifier-uris for the exact list to register).
    On Entra this applies to EVERY server (issue #990), not just the egress
    modes. The registry side is fully dynamic -- derived from the server entry,
    no per-server env config. Lenient IdPs (Keycloak/Cognito/Okta/Auth0) do not
    hit these constraints; their plain servers keep using the gateway-wide PRM
    and this per-server PRM is what makes Entra ingress login work.
    """
    normalized = _normalize_prm_server_path(server_path)
    # Server lookup is inside a guard: a repository/backend error on this
    # UNAUTHENTICATED endpoint must not surface as an unhandled 500 with a
    # traceback. Fail closed to a generic 502 (logged for operators).
    try:
        info = await server_service.get_server_info(normalized)
    except Exception:
        logger.exception("Per-server PRM: server lookup failed for %s", normalized)
        raise HTTPException(
            status_code=502, detail="Could not build Protected Resource Metadata"
        ) from None
    if not info or not server_needs_per_server_prm(info.get("egress_auth_mode")):
        # No per-server PRM for this server -> client falls back to the global PRM.
        raise HTTPException(status_code=404, detail="no per-server resource metadata")

    # Per-server connection-URL resource: the only value that satisfies the
    # client's RFC 9728 §3.3 match, RFC 8707 canonicalization, and Entra's exact
    # App ID URI match simultaneously (see the docstring).
    append_mcp = info.get("append_mcp_path") is not False
    resource = build_per_server_resource_url(
        settings.registry_url, normalized, append_mcp=append_mcp
    )
    scopes_supported = [f"{resource}/user_impersonation"]
    enforce_https(resource, https_required=settings.mcp_https_required)

    try:
        provider = _get_active_auth_provider()
        document = provider.protected_resource_metadata(
            resource=resource,
            scopes_supported=scopes_supported,
            resource_documentation=build_resource_documentation_url(),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="OAuth discovery not implemented for the configured auth provider",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to build per-server PRM document")
        raise HTTPException(
            status_code=502,
            # Do not reflect internal exception detail on unauthenticated
            # endpoint; logged above for operators.
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
