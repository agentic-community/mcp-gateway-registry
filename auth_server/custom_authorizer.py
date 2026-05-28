"""
Custom Authorizer Integration Module

This module provides stubs and interfaces for integrating external authorization
endpoints into the auth server's /validate flow. It's not yet implemented.

Modes:
  - NATIVE: Default behavior, no custom authorizer call
  - CUSTOM: Skip all native validation, only call custom authorizer
  - BOTH: Native validation first, then custom authorizer as final gate
"""

import logging
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AuthorizerMode(str, Enum):
    """Authorization mode enum."""

    NATIVE = "native"  # Default — built-in JWT/OAuth2/session only
    CUSTOM = "custom"  # External authorizer only (skip native auth)
    BOTH = "both"  # Native validation first, then external authorizer


# Pydantic models (stubs)
class CustomAuthRequest(BaseModel):
    """HTTP request details sent to custom authorizer."""

    pass


class NativeAuthResult(BaseModel):
    """Native JWT validation result."""

    pass


class CustomAuthContext(BaseModel):
    """Context metadata for custom authorizer."""

    pass


class CustomAuthorizerPayload(BaseModel):
    """Full payload sent to custom authorizer."""

    pass


class CustomAuthErrorDetail(BaseModel):
    """Error details from custom authorizer."""

    pass


class CustomAuthorizerResponse(BaseModel):
    """Response from custom authorizer (authorized true/false)."""

    authorized: bool
    metadata: dict[str, Any] | None = None
    error: CustomAuthErrorDetail | None = None


# Configuration and setup functions
def get_authorizer_mode() -> AuthorizerMode:
    """
    Read AUTHORIZER_MODE environment variable and return AuthorizerMode.

    Returns:
        AuthorizerMode enum value
        Defaults to NATIVE if not set or invalid

    Raises:
        ValueError: If invalid mode is provided (optional - can default to native)
    """
    mode_str = os.environ.get("AUTHORIZER_MODE", "native").lower()
    try:
        return AuthorizerMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid AUTHORIZER_MODE '{mode_str}', defaulting to 'native'")
        return AuthorizerMode.NATIVE


def validate_custom_authorizer_config() -> None:
    """
    Validate custom authorizer configuration at startup.

    Should be called once in the lifespan() during startup.

    Raises:
        ValueError: If configuration is invalid
            - AUTHORIZER_MODE is 'custom' or 'both' but CUSTOM_AUTHORIZER_URL is not set
    """
    mode = get_authorizer_mode()
    url = os.environ.get("CUSTOM_AUTHORIZER_URL", "").strip()

    if mode in (AuthorizerMode.CUSTOM, AuthorizerMode.BOTH) and not url:
        raise ValueError(
            f"AUTHORIZER_MODE is '{mode.value}' but CUSTOM_AUTHORIZER_URL is not set. "
            f"Must provide CUSTOM_AUTHORIZER_URL when using custom or both mode."
        )

    if mode != AuthorizerMode.NATIVE:
        timeout = int(os.environ.get("CUSTOM_AUTHORIZER_TIMEOUT", "5"))
        logger.info(
            f"Custom Authorizer Configuration Validated: "
            f"mode={mode.value}, url={url}, timeout={timeout}s"
        )


class CustomAuthorizerClient:
    """
    HTTP client for calling custom authorizer endpoint.


    """

    def __init__(
        self,
        url: str,
        timeout: int = 5,
        api_key: str | None = None,
    ):
        """
        Initialize custom authorizer client.

        Args:
            url: Custom authorizer endpoint URL
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self._url = url
        self._timeout = timeout
        self._api_key = api_key

    async def authorize(
        self,
        payload: CustomAuthorizerPayload,
    ) -> CustomAuthorizerResponse:
        """
        Call the custom authorizer endpoint.

        Args:
            payload: CustomAuthorizerPayload with request context and native auth result

        Returns:
            CustomAuthorizerResponse with authorization decision

        Raises:
            ValueError: If authorization is denied or times out
            HTTPError: If custom authorizer is unreachable

        Note:
            FAIL-CLOSED: Any error (timeout, network, 5xx) results in denial.
            No fail-open option.
        """
        # Stub - to be implemented
        logger.debug(f"[STUB] CustomAuthorizerClient.authorize() called with payload")
        return CustomAuthorizerResponse(
            authorized=True,
            metadata=None,
            error=None,
        )


def get_custom_authorizer_client() -> Optional[CustomAuthorizerClient]:
    """
    Get or create the custom authorizer client singleton.

    Returns:
        CustomAuthorizerClient if custom authorizer is configured (mode != native)
        None if native mode only

    Note:
        This function maintains a singleton instance to avoid recreating
        the HTTP client on each request.
    """
    mode = get_authorizer_mode()
    if mode == AuthorizerMode.NATIVE:
        return None

    url = os.environ.get("CUSTOM_AUTHORIZER_URL", "").strip()
    timeout = int(os.environ.get("CUSTOM_AUTHORIZER_TIMEOUT", "5"))
    api_key = os.environ.get("CUSTOM_AUTHORIZER_API_KEY", "").strip() or None

    if not url:
        raise ValueError("CUSTOM_AUTHORIZER_URL must be set for non-native mode")

    # TODO: implement proper singleton with httpx.AsyncClient
    return CustomAuthorizerClient(url=url, timeout=timeout, api_key=api_key)


def build_custom_auth_payload(
    request: Any,  # FastAPI Request object
    native_auth_result: dict[str, Any] | None,
    request_id: str,
) -> CustomAuthorizerPayload:
    """
    Build CustomAuthorizerPayload from FastAPI request and native auth result.

    Args:
        request: FastAPI Request object
        native_auth_result: Dict with native auth validation result (None in custom mode)
        request_id: Unique request ID for correlation

    Returns:
        CustomAuthorizerPayload ready to send to authorizer endpoint

    Note:
        This function:
        - Masks sensitive headers (Authorization, Cookie, etc.)
        - Extracts X-Original-URL for full URL context
        - Includes client IP from request.client
        - Includes query parameters
    """
    # Stub - to be implemented
    logger.debug(f"[STUB] build_custom_auth_payload() called for request {request_id}")
    return CustomAuthorizerPayload()


def mask_sensitive_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Mask sensitive header values before sending to custom authorizer.

    Masks headers like Authorization, Cookie, X-API-Key to protect credentials.

    Args:
        headers: Original request headers dict

    Returns:
        Dict with sensitive values masked (first 10 + last 4 chars visible, rest ***MASKED***)
    """
    return headers
