"""An httpx2 transport that sends every backend call through the registry's SSRF guard.

FastMCP's OpenAPI provider wants an ``httpx2.AsyncClient``. The registry's guard
(``registry/utils/url_guard.py``) is an ``httpx`` transport. This adapter copies
each httpx2 request into an httpx request, hands it to ``GuardedAsyncTransport``
under ``PROXY_PROFILE``, and copies the response back, reading at most
``max_response_bytes``. It makes no trust decision of its own: validation, IP
pinning and the metadata-address block all stay in ``url_guard.py``.
"""

import logging

import httpx
import httpx2

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

logger = logging.getLogger(__name__)


class BackendResponseTooLargeError(httpx2.RequestError):
    """The backend response passed the size cap."""


class BackendUrlBlockedError(httpx2.RequestError):
    """The SSRF guard refused the backend URL."""


class BackendRequestFailedError(httpx2.RequestError):
    """The backend could not be reached (DNS, connect, TLS, timeout)."""


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
    ) -> None:
        self._inner = url_guard.GuardedAsyncTransport(guard_profile=url_guard.PROXY_PROFILE)
        self._max_response_bytes = max_response_bytes

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        body = await request.aread()
        inner_request = _to_httpx_request(request, body)
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
