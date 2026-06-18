"""
Custom Authorizer Service for MCP Gateway Registry.

Handles all runtime logic for the custom authorizer integration:
  - Reading and caching the AUTHORIZER_MODE from env
  - Constructing the outbound payload (with header masking)
  - Managing the singleton HTTP client (CustomAuthorizerClient)
  - Startup configuration validation

This module is imported by server.py at exactly four points:
  1. Startup  — validate_custom_authorizer_config()  inside lifespan()
  2. /validate — get_authorizer_mode()               to branch execution
  3. /validate — build_custom_auth_payload()         to construct the request
  4. /validate — get_custom_authorizer_client()      to call the webhook

Environment variables consumed:
  AUTHORIZER_MODE            native | custom | both  (default: native)
  CUSTOM_AUTHORIZER_URL      Full URL of the POST /authorize endpoint
  CUSTOM_AUTHORIZER_TIMEOUT  HTTP timeout in seconds (default: 5)
  CUSTOM_AUTHORIZER_API_KEY  Optional Bearer token sent to the authorizer
"""

from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

import httpx

from authorizer_models.custom_authorizer import (
    AuthorizerMode,
    CustomAuthContext,
    CustomAuthErrorDetail,
    CustomAuthRequest,
    CustomAuthorizerPayload,
    CustomAuthorizerResponse,
    NativeAuthResult,
)

from registry.utils.request_utils import get_client_ip

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

# Module-level singletons — reset only via _reset_globals() in tests

_authorizer_mode: Optional[AuthorizerMode] = None
_authorizer_client: Optional["CustomAuthorizerClient"] = None


_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "x-authorization",
        "set-cookie",
        "proxy-authorization",
    }
)

_MASK_VALUE: str = "***MASKED***"


def mask_sensitive_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with sensitive values replaced by a mask.

    For ``Authorization``-family headers the scheme token is preserved so
    that the header is still recognisable in logs
    (e.g. ``Bearer eyJ...***MASKED***``).  All other sensitive headers are
    fully replaced with ``***MASKED***``.

    Args:
        headers: Raw request headers dict.

    Returns:
        New dict — the original is never mutated.
    """
    masked: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADERS:
            # Preserve the scheme token for Authorization-style headers so that
            # the masked value is still recognisable in audit logs.
            lower_val = str(value).lower()
            if lower_val.startswith("bearer ") and len(value) > 7:
                # Keep up to 4 chars of the credential for traceability.
                credential = value[7:]
                preview = credential[:4] if len(credential) >= 4 else credential
                masked[key] = f"Bearer {preview}...{_MASK_VALUE}"
            else:
                masked[key] = _MASK_VALUE
        else:
            masked[key] = value
    return masked


# HTTP client

class CustomAuthorizerClient:
    """Async HTTP client that calls the external authorizer endpoint.

    Fail-closed contract: every error path (timeout, network failure, non-2xx
    status, malformed JSON) returns
    ``CustomAuthorizerResponse(authorized=False, ...)`` and never raises.

    The instance is created once via ``get_custom_authorizer_client()`` and
    reused across requests for connection-pool efficiency.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        api_key: str | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=timeout),
        )

    async def authorize(
        self, payload: CustomAuthorizerPayload
    ) -> CustomAuthorizerResponse:
        """POST *payload* to the custom authorizer and return its decision.

        Both HTTP 200 and HTTP 403 are treated as valid authorizer responses
        and parsed into ``CustomAuthorizerResponse``.  Any other status code,
        network error, timeout, or unparseable body results in a fail-closed
        ``authorized=False`` response.

        Args:
            payload: The fully-constructed authorization request.

        Returns:
            ``CustomAuthorizerResponse`` — never raises.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.post(
                self._url,
                content=payload.model_dump_json(),
                headers=headers,
            )

            # Both 200 (allow or deny) and 403 (deny) are valid responses.
            if response.status_code in (200, 403):
                try:
                    return CustomAuthorizerResponse.model_validate_json(response.content)
                except Exception as exc:
                    logger.error(
                        "Custom authorizer returned malformed response: %s", exc
                    )
                    return CustomAuthorizerResponse(
                        authorized=False,
                        error=CustomAuthErrorDetail(
                            code="MALFORMED_RESPONSE",
                            message=f"Custom authorizer returned unparseable response: {exc}",
                        ),
                    )

            # Any unexpected status code → fail-closed.
            logger.error(
                "Custom authorizer returned unexpected HTTP %d", response.status_code
            )
            return CustomAuthorizerResponse(
                authorized=False,
                error=CustomAuthErrorDetail(
                    code="UNEXPECTED_STATUS",
                    message=f"Custom authorizer returned HTTP {response.status_code}",
                ),
            )

        except httpx.TimeoutException:
            logger.error(
                "Custom authorizer timed out after %.1fs (url=%s)", self._timeout, self._url
            )
            return CustomAuthorizerResponse(
                authorized=False,
                error=CustomAuthErrorDetail(
                    code="TIMEOUT",
                    message=f"Custom authorizer timed out after {self._timeout}s",
                ),
            )
        except httpx.ConnectError as exc:
            logger.error("Custom authorizer unreachable: %s", exc)
            return CustomAuthorizerResponse(
                authorized=False,
                error=CustomAuthErrorDetail(
                    code="UNREACHABLE",
                    message=f"Custom authorizer unreachable: {exc}",
                ),
            )
        except Exception as exc:
            logger.error("Custom authorizer call failed unexpectedly: %s", exc)
            return CustomAuthorizerResponse(
                authorized=False,
                error=CustomAuthErrorDetail(
                    code="INTERNAL_ERROR",
                    message=f"Custom authorizer call failed: {exc}",
                ),
            )

    async def close(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._client.aclose()


# Public API consumed by server.py


def get_authorizer_mode() -> AuthorizerMode:
    """Read ``AUTHORIZER_MODE`` from the environment and cache the result.

    The value is parsed once and stored in ``_authorizer_mode`` for the
    lifetime of the process.  Invalid values produce a warning and fall back
    to ``NATIVE``.

    Returns:
        Parsed ``AuthorizerMode``; defaults to ``NATIVE``.
    """
    global _authorizer_mode
    if _authorizer_mode is None:
        raw = os.environ.get("AUTHORIZER_MODE", AuthorizerMode.NATIVE.value).lower().strip()
        try:
            _authorizer_mode = AuthorizerMode(raw)
        except ValueError:
            logger.warning(
                "Invalid AUTHORIZER_MODE=%r — falling back to 'native'. "
                "Valid values: native, custom, both.",
                raw,
            )
            _authorizer_mode = AuthorizerMode.NATIVE
        logger.info("Custom authorizer mode: %s", _authorizer_mode.value)
    return _authorizer_mode


def get_custom_authorizer_client() -> Optional[CustomAuthorizerClient]:
    """Return the singleton ``CustomAuthorizerClient``.

    The client is created lazily on the first call and reused thereafter.
    Returns ``None`` when ``AUTHORIZER_MODE=native`` so that callers can
    guard with a simple ``if client:`` check.

    Returns:
        ``None`` in native mode; a shared ``CustomAuthorizerClient`` otherwise.
    """
    global _authorizer_client

    if get_authorizer_mode() == AuthorizerMode.NATIVE:
        return None

    if _authorizer_client is None:
        url = os.environ.get("CUSTOM_AUTHORIZER_URL", "").strip()
        timeout = float(os.environ.get("CUSTOM_AUTHORIZER_TIMEOUT", "5"))
        api_key = os.environ.get("CUSTOM_AUTHORIZER_API_KEY", "").strip() or None

        _authorizer_client = CustomAuthorizerClient(
            url=url,
            timeout=timeout,
            api_key=api_key,
        )
        logger.info(
            "Custom authorizer client created: url=%s, timeout=%.1fs, api_key=%s",
            url,
            timeout,
            "set" if api_key else "not set",
        )

    return _authorizer_client


def build_custom_auth_payload(
    request: "Request",
    native_auth_result: Optional[NativeAuthResult],
    request_id: str,
) -> CustomAuthorizerPayload:
    """Construct the payload to POST to the custom authorizer.

    Extracts HTTP request metadata from the FastAPI ``Request`` object,
    masks sensitive headers, and wraps everything into a
    ``CustomAuthorizerPayload`` ready for serialisation.

    Args:
        request:            The FastAPI ``Request`` object from ``/validate``.
        native_auth_result: Native auth result; ``None`` in custom-only mode.
        request_id:         Unique ID for request correlation.

    Returns:
        A fully-populated ``CustomAuthorizerPayload``.
    """
    raw_headers = dict(request.headers)
    safe_headers = mask_sensitive_headers(raw_headers)

    query_params: dict[str, str] = (
        dict(request.query_params) if hasattr(request, "query_params") else {}
    )

    # nginx's auth_request module does not forward the original request body,
    # so the gateway passes it as the X-Body header instead.
    body: Optional[str] = request.headers.get("X-Body") or None

    auth_request = CustomAuthRequest(
        method=request.method,
        path=request.url.path,
        original_url=str(request.url),
        query_params=query_params,
        headers=safe_headers,
        body=body,
        client_ip=get_client_ip(request),
    )

    context = CustomAuthContext(
        request_id=request_id,
        gateway_version=os.environ.get("GATEWAY_VERSION", "1.0.0"),
    )

    return CustomAuthorizerPayload(
        request=auth_request,
        native_auth_result=native_auth_result,
        context=context,
    )


def validate_custom_authorizer_config() -> None:
    """Validate the custom authorizer configuration at startup.

    Called inside FastAPI's ``lifespan()`` context.
    Raises with a descriptive message when the configuration is inconsistent
    so that the server fails fast with a clear explanation rather than
    silently misbehaving at request time.

    Raises:
        ValueError: When ``AUTHORIZER_MODE`` is ``custom`` or ``both`` but
                    ``CUSTOM_AUTHORIZER_URL`` is empty or whitespace-only.
    """
    mode = get_authorizer_mode()

    if mode == AuthorizerMode.NATIVE:
        logger.info("Custom authorizer: disabled (AUTHORIZER_MODE=native)")
        return

    url = os.environ.get("CUSTOM_AUTHORIZER_URL", "").strip()
    if not url:
        raise ValueError(
            f"AUTHORIZER_MODE={mode.value!r} requires CUSTOM_AUTHORIZER_URL to be set. "
            "Set CUSTOM_AUTHORIZER_URL to the full URL of your custom authorizer "
            "endpoint (e.g. http://custom-authorizer:8080/authorize)."
        )

    if url.startswith("http://") and "localhost" not in url and "127.0.0.1" not in url:
        logger.warning(
            "CUSTOM_AUTHORIZER_URL uses plain HTTP (%s). "
            "HTTPS is strongly recommended for production deployments.",
            url,
        )

    timeout = float(os.environ.get("CUSTOM_AUTHORIZER_TIMEOUT", "5"))
    logger.info(
        "Custom authorizer: enabled (mode=%s, url=%s, timeout=%.1fs)",
        mode.value,
        url,
        timeout,
    )


# Test helper — never call in production code


def _reset_globals() -> None:
    """Reset module-level singletons.  FOR TESTING ONLY."""
    global _authorizer_mode, _authorizer_client
    _authorizer_mode = None
    _authorizer_client = None
