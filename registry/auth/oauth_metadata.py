"""Helpers for building the gateway's OAuth discovery documents.

Used by the registry's `.well-known/oauth-protected-resource` and
`.well-known/oauth-authorization-server` routes (RFC 9728 and RFC 8414).

The functions here are intentionally small and pure: route handlers compose
them with provider-supplied IdP metadata to assemble a final response.
"""

import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


WELLKNOWN_PRM_PATH: str = "/.well-known/oauth-protected-resource"
WELLKNOWN_AS_METADATA_PATH: str = "/.well-known/oauth-authorization-server"

# Basic, IdP-universal OIDC scopes advertised by default in the PRM. These exist
# on every supported IdP (Keycloak, Entra, Okta, Auth0, Cognito), so the IDE
# OAuth login handshake succeeds everywhere. A user's actual access is NOT
# derived from these advertised scopes - it comes from the token's `groups`
# claim, which the auth server maps to registry scopes. So advertising only the
# basics does not reduce access. Operators can override with the
# `mcp_advertised_scopes` setting (env `MCP_ADVERTISED_SCOPES`).
DEFAULT_ADVERTISED_SCOPES: list[str] = ["openid", "email", "profile", "offline_access"]


def build_canonical_resource_url(registry_url: str) -> str:
    """Return the canonical MCP resource URL for the gateway.

    The result is used as both the `resource` field in the PRM document and
    the `resource_metadata` URL embedded in `WWW-Authenticate` 401 headers.
    They MUST match byte-for-byte (acceptance criterion for issue #989).

    Args:
        registry_url: Configured public URL of this gateway, from
            settings.registry_url. May or may not have a trailing slash.

    Returns:
        URL with no trailing slash.

    Raises:
        ValueError: If registry_url is empty or has no scheme.
    """
    if not registry_url:
        raise ValueError("registry_url is empty; cannot build canonical resource URL")

    if "://" not in registry_url:
        raise ValueError(
            f"registry_url '{registry_url}' has no scheme (expected http:// or https://)"
        )

    return registry_url.rstrip("/")


def build_per_server_prm_url(
    registry_url: str,
    server_path: str,
    append_mcp: bool = True,
) -> str:
    """Return the per-server PRM URL in RFC 9728 §3.1 path-aware form.

    The well-known segment is inserted between the origin and the resource path:
    ``<origin>/.well-known/oauth-protected-resource/<server>/mcp`` -- NOT appended
    to the end of the resource. This is the URL the gateway advertises in the
    per-server 401 ``WWW-Authenticate: resource_metadata=...`` header and the one
    the per-server PRM route serves.
    """
    base = build_canonical_resource_url(registry_url)
    seg = server_path.strip("/")
    suffix = seg
    if append_mcp and not suffix.endswith("mcp"):
        suffix = f"{suffix}/mcp" if suffix else "mcp"
    return f"{base}{WELLKNOWN_PRM_PATH}/{suffix}" if suffix else f"{base}{WELLKNOWN_PRM_PATH}"


def build_per_server_resource_url(
    registry_url: str,
    server_path: str,
    append_mcp: bool = True,
) -> str:
    """Return the per-server canonical resource URL (RFC 8707) for an obo server.

    This equals the MCP client's CONNECTION URL for the server, which is the
    only per-server value spec-compliant clients (RFC 9728 §3.3) accept as the
    PRM ``resource`` and the only one Microsoft Entra can match to an App ID URI
    (Entra forbids a trailing slash, so a bare origin -- which clients
    canonicalize with a trailing ``/`` -- cannot be matched; a path-qualified
    per-server URL can).

    Args:
        registry_url: the gateway public URL (``settings.registry_url``).
        server_path: the registered server path (e.g. ``/obo-echo``).
        append_mcp: whether the connect URL carries the ``/mcp`` transport
            suffix (the server's ``append_mcp_path``). Most streamable-http
            servers do; root-endpoint servers set it false.

    Returns:
        e.g. ``https://gw.example.com/obo-echo/mcp`` (no trailing slash).
    """
    base = build_canonical_resource_url(registry_url)
    seg = server_path.strip("/")
    url = f"{base}/{seg}" if seg else base
    if append_mcp and not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


def entra_forces_per_server_prm(auth_provider: str | None) -> bool:
    """Whether the configured IdP forces a per-server PRM for EVERY server.

    Entra's gateway-wide PRM advertises the bare origin, whose resource a
    spec-compliant client canonicalizes with a trailing slash -- unmatchable to
    an Entra App ID URI (Entra forbids the trailing slash), which fails token
    exchange with AADSTS9010010. So on Entra every server (including plain ones
    with no egress mode) needs a per-server, connection-URL PRM (issue #990).
    Non-Entra IdPs accept the bare-origin gateway-wide PRM, so this is Entra-only.

    Single source of truth consulted by BOTH the registry's
    ``server_needs_per_server_prm`` (the advertise side) and the auth-server's
    ``_server_advertises_per_server_prm`` (the audience-accept side) so the two
    predicates cannot drift.
    """
    return (auth_provider or "").strip().lower() == "entra"


def enforce_https(
    resource_url: str,
    https_required: bool,
) -> None:
    """Raise if `resource_url` is http and HTTPS is required for this deployment.

    The MCP authorization spec requires HTTPS for all OAuth endpoints in
    non-local environments. We fail fast at startup rather than serve a
    PRM document advertising an http resource.

    Args:
        resource_url: The canonical resource URL.
        https_required: If True and `resource_url` is http, raise.

    Raises:
        ValueError: If https_required and the URL is not HTTPS.
    """
    if not https_required:
        return
    if resource_url.startswith("https://"):
        return
    raise ValueError(
        f"Canonical resource URL '{resource_url}' is not HTTPS; "
        "set MCP_HTTPS_REQUIRED=false for local development, or use an HTTPS registry_url"
    )


async def derive_supported_scopes() -> list[str]:
    """Build the `scopes_supported` array for the PRM document.

    When `mcp_advertised_scopes` is set, returns that explicit list (split on
    whitespace). This is the operator override for any IdP that validates the
    requested scopes against its own registered scope set.

    When unset, returns the basic IdP-universal OIDC scopes
    (`DEFAULT_ADVERTISED_SCOPES`). It deliberately does NOT advertise the
    registry's internal scope/group names from the `mcp_scopes` collection.
    Those names (e.g. `registry-admins`, `tla-consumer-*`) are authorization
    identifiers, NOT IdP OAuth scopes; advertising them makes the IDE request
    scopes the IdP has never heard of, and any scope-validating IdP (Keycloak
    without DCR, Okta, Entra, Auth0) rejects the authorization request with
    `invalid_scope`. Access is group-derived (see `map_groups_to_scopes` in the
    auth server), so omitting these names does not reduce any user's access.

    The result is stable across requests (cache-friendly): the override
    preserves operator order, and the default is a fixed list.

    Returns:
        List of scope name strings for the PRM `scopes_supported` array.
    """
    override = getattr(settings, "mcp_advertised_scopes", "") or ""
    if override.strip():
        # Preserve operator-specified order so it is byte-stable across requests.
        return [s for s in override.split() if s]

    return list(DEFAULT_ADVERTISED_SCOPES)


def build_resource_documentation_url() -> str:
    """Return the URL of the operator-facing OAuth docs page.

    Used as the `resource_documentation` field in the PRM document. Defaults
    to a docs page on the registry itself; an explicit override via
    `mcp_resource_documentation_url` (settings) takes precedence when set.
    """
    override = getattr(settings, "mcp_resource_documentation_url", None)
    if override:
        return override
    return f"{build_canonical_resource_url(settings.registry_url)}/docs/oauth"


def build_www_authenticate_header(resource_metadata_url: str) -> str:
    """Build the WWW-Authenticate header value for 401 responses.

    Per RFC 9728 §5.1 and the MCP 2025-06-18 authorization spec, the
    `resource_metadata` parameter points discovery clients at the gateway's
    Protected Resource Metadata document.

    Args:
        resource_metadata_url: Absolute URL of the PRM endpoint. Must equal
            the `resource` field returned by the PRM document, byte-for-byte.

    Returns:
        A complete header value suitable for `WWW-Authenticate`.
    """
    return f'Bearer realm="mcp", resource_metadata="{resource_metadata_url}"'


def build_resource_metadata_url(resource_url: str) -> str:
    """Return the absolute URL of the gateway's PRM endpoint.

    Args:
        resource_url: The canonical resource URL (no trailing slash).

    Returns:
        `<resource_url>/.well-known/oauth-protected-resource`.
    """
    return f"{resource_url}{WELLKNOWN_PRM_PATH}"
