"""Who may write a server record that points at the openapi-mcp container (issue #2022).

An OpenAPI MCP server owns one ordinary ``ServerInfo`` at ``/<slug>``, marked
with ``metadata.openapi_server.slug``. Two rules keep that record and the
container's identity tied to the OpenAPI record:

- A managed record must point at exactly ``<container>/s/<slug>/mcp``. Without
  this, anyone allowed to edit the record could repoint it, and the egress
  vend would send every caller's per-user keys to the new host.
- No other record may point at the container. Only the OpenAPI MCP server
  service creates records there, so the SSRF guard's built-in identity can
  never be claimed through the generic registration routes.

``check_openapi_ownership`` is called from every write in ``server_service``
that sets a target URL, so the rules hold for any route, including future ones.
"""

import logging
from typing import Any

from ..auth.openapi_mcp_sync import (
    ADMISSION_HEADER,
    SyncTokenError,
    admission_marker,
)
from ..core.config import settings
from ..utils.url_guard import (
    UrlValidationError,
    is_builtin_openapi_mcp_target,
    is_openapi_mcp_container_url,
    normalize_url_identity,
    openapi_mcp_target_url,
)

logger = logging.getLogger(__name__)

OPENAPI_METADATA_KEY: str = "openapi_server"
OPENAPI_ROUTES_HINT: str = "manage it under /api/openapi-servers"


def _expected_target(
    slug: str,
) -> str | None:
    try:
        return openapi_mcp_target_url(slug)
    except UrlValidationError:
        return None


def _same_target(
    url: str,
    expected: str,
) -> bool:
    try:
        return normalize_url_identity(url) == expected
    except UrlValidationError:
        return False


def managed_openapi_slug(
    server_info: dict[str, Any] | None,
) -> str | None:
    """The OpenAPI MCP server slug that owns this record, or None for an ordinary record."""
    metadata = (server_info or {}).get("metadata")
    if not isinstance(metadata, dict):
        return None
    entry = metadata.get(OPENAPI_METADATA_KEY)
    if not isinstance(entry, dict):
        return None
    slug = entry.get("slug")
    return slug if isinstance(slug, str) and slug else None


def points_at_own_hub(
    server_info: dict[str, Any] | None,
) -> bool:
    """True only for a managed record whose path and target are exactly its own hub's."""
    slug = managed_openapi_slug(server_info)
    url = (server_info or {}).get("proxy_pass_url")
    if slug is None or not isinstance(url, str):
        return False
    if (server_info or {}).get("path") != f"/{slug}":
        return False
    expected = _expected_target(slug)
    if expected is None:
        # The feature is off, so the exact URL cannot be derived. Records left
        # from when it was on still point at the container host, and keeping
        # that host is enough to refuse a repoint.
        return is_openapi_mcp_container_url(url)
    return _same_target(url, expected)


def admission_headers(
    entity_path: str | None,
    registered_target: str | None,
    destination_url: str | None = None,
) -> dict[str, str]:
    """The admission marker header for a registry call to an OpenAPI MCP server's hub.

    The container refuses a hub call without the marker for that slug. Only a
    record at ``/<slug>`` whose target is exactly ``<container>/s/<slug>/mcp``
    gets one, and only when the call goes to that same URL. Ownership checks
    keep every other record off the container's hostname, so a record that
    reaches the container by IP or another name never gets a marker.

    Returns an empty dict for every other call.
    """
    if not is_builtin_openapi_mcp_target("mcp_server", entity_path, registered_target):
        return {}
    destination = destination_url or registered_target
    try:
        same_hub = normalize_url_identity(destination).rstrip("/") == normalize_url_identity(
            registered_target
        ).rstrip("/")
    except (UrlValidationError, TypeError, AttributeError):
        same_hub = False
    if not same_hub:
        return {}
    slug = str(entity_path).strip("/")
    try:
        return {ADMISSION_HEADER: admission_marker(settings.openapi_mcp_sync_secret, slug)}
    except SyncTokenError:
        logger.warning(f"No admission marker for '{slug}': OPENAPI_MCP_SYNC_SECRET is not set")
        return {}


def check_openapi_ownership(
    server_info: dict[str, Any],
) -> None:
    """Refuse a record that breaks either ownership rule. Pass the merged, to-be-stored state.

    Raises:
        UrlValidationError: When a managed record points anywhere but its own
            hub, or an ordinary record points at the container.
    """
    url = server_info.get("proxy_pass_url")
    slug = managed_openapi_slug(server_info)
    if slug is not None:
        if not points_at_own_hub(server_info):
            logger.warning(f"Refused a write that repoints OpenAPI MCP server '{slug}'")
            raise UrlValidationError(
                str(url),
                f"this server belongs to OpenAPI MCP server '{slug}' and must keep its "
                f"openapi-mcp target; {OPENAPI_ROUTES_HINT}",
            )
        return
    if is_openapi_mcp_container_url(url):
        logger.warning(
            f"Refused an ordinary server record targeting the openapi-mcp container: {server_info.get('path')}"
        )
        raise UrlValidationError(
            str(url),
            f"the openapi-mcp container serves only OpenAPI MCP servers; {OPENAPI_ROUTES_HINT}",
        )


def openapi_mcp_feature_on() -> bool:
    """True when OpenAPI MCP servers are on (after the startup mode gate)."""
    from ..core.config import settings

    return getattr(settings, "openapi_mcp_enabled", False) is True


def hidden_while_feature_off(
    server_info: dict[str, Any] | None,
) -> bool:
    """True for a record an OpenAPI MCP server owns while the feature is off.

    Turning OPENAPI_MCP_ENABLED off leaves the records in place, so turning it
    back on restores every server and its users' keys. While it is off they
    have no backend, so listings, search, health checks and the gateway's
    routes skip them. This is a filter on reads; nothing stored changes.
    """
    return managed_openapi_slug(server_info) is not None and not openapi_mcp_feature_on()


def without_hidden_records(
    servers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The servers minus OpenAPI MCP records while the feature is off."""
    if openapi_mcp_feature_on():
        return servers
    return {path: info for path, info in servers.items() if managed_openapi_slug(info) is None}
