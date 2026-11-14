"""An httpx2 transport that sends every backend call through the registry's SSRF guard.

FastMCP's OpenAPI provider wants an ``httpx2.AsyncClient``. The registry's guard
(``registry/utils/url_guard.py``) is an ``httpx`` transport. This adapter copies
each httpx2 request into an httpx request, hands it to ``GuardedAsyncTransport``
under ``PROXY_PROFILE``, and copies the response back, reading at most
``max_response_bytes``. It makes no trust decision of its own: validation, IP
pinning and the metadata-address block all stay in ``url_guard.py``.
"""

import logging
from dataclasses import dataclass

import httpx
import httpx2
from fastmcp.server.dependencies import get_http_request

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

logger = logging.getLogger(__name__)


class BackendResponseTooLargeError(httpx2.RequestError):
    """The backend response passed the size cap."""


class BackendUrlBlockedError(httpx2.RequestError):
    """The SSRF guard refused the backend URL."""


class BackendRequestFailedError(httpx2.RequestError):
    """The backend could not be reached (DNS, connect, TLS, timeout)."""


class MissingUserCredentialError(httpx2.RequestError):
    """A per-user target was called without the caller's key."""


# Key under which the dispatcher stores the verified per-user keys on the ASGI
# scope. FastMCP rebuilds the request from the scope inside the tool call.
USER_CREDENTIALS_SCOPE_KEY: str = "openapi_mcp_user_credentials"


@dataclass(frozen=True)
class PerUserAuth:
    """How a per-user target presents the caller's key."""

    target_name: str
    scheme: str
    header_name: str | None
    base_url: str


def _request_credentials() -> dict:
    """The verified per-user keys of the request being served, or {}."""
    try:
        request = get_http_request()
    except RuntimeError:
        return {}
    return request.scope.get(USER_CREDENTIALS_SCOPE_KEY) or {}


def _same_destination(
    a: str,
    b: str,
) -> bool:
    try:
        return url_guard.normalize_url_identity(a).rstrip("/") == url_guard.normalize_url_identity(
            b
        ).rstrip("/")
    except UrlValidationError:
        return False


def _user_auth_header(
    per_user: PerUserAuth,
    request: httpx2.Request,
) -> tuple[str, str]:
    """The (header, value) carrying the caller's key. Fails closed when there is none."""
    entry = _request_credentials().get(per_user.target_name)
    if not entry or not _same_destination(entry.get("base_url", ""), per_user.base_url):
        raise MissingUserCredentialError(
            f"No API key for '{per_user.target_name}'. Add your own key for this API in the "
            "registry (Connected Accounts, or the server's details), then call the tool again.",
            request=request,
        )
    if per_user.scheme == "bearer":
        return "Authorization", f"Bearer {entry['key']}"
    return per_user.header_name or "X-API-Key", entry["key"]


async def _read_capped(
    response: httpx.Response,
    max_bytes: int,
    request: httpx2.Request,
) -> bytes:
    """Read a streamed response body, stopping once it passes max_bytes."""
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise BackendResponseTooLargeError(
                    f"backend response is larger than {max_bytes} bytes",
                    request=request,
                )
            chunks.append(chunk)
    finally:
        await response.aclose()
    return b"".join(chunks)


def _to_httpx_request(
    request: httpx2.Request,
    body: bytes,
) -> httpx.Request:
    """Copy an httpx2 request into an httpx request with the same bytes on the wire."""
    return httpx.Request(
        method=request.method,
        url=str(request.url),
        headers=list(request.headers.raw),
        content=body,
        extensions={"timeout": request.extensions.get("timeout", {})},
    )


class GuardedHttpx2Transport(httpx2.AsyncBaseTransport):
    """Send each httpx2 request through the canonical SSRF guard."""

    def __init__(
        self,
        max_response_bytes: int,
        per_user: PerUserAuth | None = None,
    ) -> None:
        self._inner = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        self._max_response_bytes = max_response_bytes
        self._per_user = per_user

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        body = await request.aread()
        inner_request = _to_httpx_request(request, body)
        if self._per_user is not None:
            name, value = _user_auth_header(self._per_user, request)
            inner_request.headers[name] = value
        try:
            inner_response = await self._inner.handle_async_request(inner_request)
        except UrlValidationError as e:
            logger.warning(f"Backend call blocked by the SSRF guard: {e}")
            raise BackendUrlBlockedError(
                "backend URL blocked by the SSRF guard", request=request
            ) from e
        except httpx.TimeoutException as e:
            raise httpx2.TimeoutException(
                f"backend timed out ({type(e).__name__})", request=request
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestFailedError(
                f"backend request failed ({type(e).__name__})", request=request
            ) from e

        content = await _read_capped(inner_response, self._max_response_bytes, request)
        headers = [
            (k, v)
            for k, v in inner_response.headers.raw
            if k.lower() not in (b"content-encoding", b"transfer-encoding", b"content-length")
        ]
        return httpx2.Response(
            status_code=inner_response.status_code,
            headers=headers,
            content=content,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()
