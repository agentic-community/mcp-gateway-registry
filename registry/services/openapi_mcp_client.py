"""The registry's calls to the openapi-mcp container (issue #2022).

Every call to /internal/* carries a fresh, single-use sync token. The target is
OPENAPI_MCP_INTERNAL_URL, which is operator configuration, so these calls use
the shared plain client like the other in-cluster calls; the SSRF guard is for
URLs that come from requests or stored records.
"""

import logging
from typing import Any

import httpx

from ..auth.openapi_mcp_sync import mint_sync_token
from ..common.log_redaction import redact_url
from ..core.config import settings
from ..utils.url_guard import shared_plain_async_client
from .openapi_errors import (
    OpenApiServerError,
    container_unavailable,
    unprocessable,
)

logger = logging.getLogger(__name__)


PUSH_TIMEOUT_SECONDS: float = 30.0
PREVIEW_TIMEOUT_SECONDS: float = 20.0
STATUS_TIMEOUT_SECONDS: float = 10.0
HEALTH_TIMEOUT_SECONDS: float = 5.0


def _url(
    path: str,
) -> str:
    return f"{settings.openapi_mcp_internal_url.rstrip('/')}{path}"


def _auth_headers(
    purpose: str,
) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_sync_token(settings.openapi_mcp_sync_secret, purpose)}"}


async def _request(
    method: str,
    path: str,
    purpose: str,
    timeout: float,
    json_body: Any = None,
) -> httpx.Response:
    """Send one control call. Network errors and 401/5xx become container_unavailable."""
    url = _url(path)
    try:
        response = await shared_plain_async_client().request(
            method, url, json=json_body, headers=_auth_headers(purpose), timeout=timeout
        )
    except httpx.HTTPError as e:
        logger.warning(f"openapi-mcp unreachable at {redact_url(url)} ({type(e).__name__})")
        raise container_unavailable(f"openapi-mcp did not answer {purpose}") from e
    if response.status_code == 401:
        logger.error(
            "openapi-mcp rejected the sync token; OPENAPI_MCP_SYNC_SECRET differs between registry and container"
        )
        raise container_unavailable("openapi-mcp rejected the sync token")
    if response.status_code >= 500:
        raise container_unavailable(f"openapi-mcp returned {response.status_code} for {purpose}")
    return response


async def put_server(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Push one server's full state. Returns the applied generation and per-target status."""
    response = await _request(
        "PUT", f"/internal/servers/{state['slug']}", "put_server", PUSH_TIMEOUT_SECONDS, state
    )
    if response.status_code != 200:
        raise container_unavailable(
            f"openapi-mcp refused the server state ({response.status_code})"
        )
    return response.json()


async def delete_server(
    slug: str,
) -> None:
    """Remove a hub. A missing hub counts as done."""
    response = await _request(
        "DELETE", f"/internal/servers/{slug}", "delete_server", PUSH_TIMEOUT_SECONDS
    )
    if response.status_code not in (204, 404):
        raise container_unavailable(
            f"openapi-mcp refused to delete '{slug}' ({response.status_code})"
        )


async def preview(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Ask the container which tools a spec would produce. No call reaches the REST API."""
    response = await _request("POST", "/internal/preview", "preview", PREVIEW_TIMEOUT_SECONDS, body)
    if response.status_code == 422:
        detail = response.json().get("detail", "conversion failed")
        raise unprocessable("conversion_failed", f"FastMCP could not convert the spec: {detail}")
    if response.status_code != 200:
        raise OpenApiServerError(
            502, "preview_failed", f"openapi-mcp preview returned {response.status_code}"
        )
    return response.json()


async def status() -> dict[str, Any]:
    """Every running hub's generation and per-target status."""
    response = await _request("GET", "/internal/status", "status", STATUS_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise container_unavailable(f"openapi-mcp status returned {response.status_code}")
    return response.json()


async def health() -> dict[str, Any] | None:
    """The container's /health body, or None when it does not answer."""
    try:
        response = await shared_plain_async_client().get(
            _url("/health"), timeout=HEALTH_TIMEOUT_SECONDS
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return response.json()
