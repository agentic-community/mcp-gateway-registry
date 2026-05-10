"""
Internal service-to-service authentication using self-signed JWTs.

This module provides utilities for authenticating internal API calls
between services (e.g., mcpgw -> registry, registry -> auth-server)
using JWTs signed with the shared SECRET_KEY.
"""

import logging
import os
import time

import jwt as pyjwt
from fastapi import HTTPException, Request, status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# JWT constants (must match auth_server/server.py)
_INTERNAL_JWT_ISSUER: str = "mcp-auth-server"
_INTERNAL_JWT_AUDIENCE: str = "mcp-registry"
_INTERNAL_JWT_TTL_SECONDS: int = 60


def generate_internal_token(
    subject: str = "internal-service",
    purpose: str = "internal-api",
) -> str:
    """
    Generate a short-lived self-signed JWT for internal service-to-service auth.

    Uses the shared SECRET_KEY that both services have access to.

    Args:
        subject: Identity of the calling service
        purpose: Purpose of the request (for audit logging)

    Returns:
        Encoded JWT string

    Raises:
        ValueError: If SECRET_KEY is not configured
    """
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable not set")

    now = int(time.time())
    claims = {
        "iss": _INTERNAL_JWT_ISSUER,
        "aud": _INTERNAL_JWT_AUDIENCE,
        "sub": subject,
        "purpose": purpose,
        "token_use": "access",
        "iat": now,
        "exp": now + _INTERNAL_JWT_TTL_SECONDS,
    }
    return pyjwt.encode(claims, secret_key, algorithm="HS256")


async def validate_internal_auth(request: Request) -> str:
    """
    FastAPI dependency that validates internal service authentication.

    Accepts Bearer JWT signed with the shared ``SECRET_KEY``. Used as
    the router-level gate on ``/internal/*`` routes in both the
    registry and auth-server FastAPI apps.

    Args:
        request: The FastAPI request object

    Returns:
        Caller identity string (e.g., 'registry-service')

    Raises:
        HTTPException: 401 if authentication fails
    """
    return _validate_authorization_header(request.headers.get("Authorization"))


def _validate_authorization_header(authorization: str | None) -> str:
    """Implementation detail of :func:`validate_internal_auth`.

    Takes the raw ``Authorization`` header value so the public
    dependency can be a thin shim over ``request.headers.get(...)``.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if authorization.startswith("Bearer "):
        return _validate_bearer_token(authorization)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unsupported authentication scheme. Use Bearer token.",
    )


def _validate_bearer_token(auth_header: str) -> str:
    """Validate a Bearer JWT token signed with SECRET_KEY."""
    token = auth_header.split(" ", 1)[1]

    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        logger.error("SECRET_KEY not set, cannot validate internal JWT")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    try:
        claims = pyjwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            issuer=_INTERNAL_JWT_ISSUER,
            audience=_INTERNAL_JWT_AUDIENCE,
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
            # Internal JWT TTL is 60 seconds (see _INTERNAL_JWT_TTL_SECONDS).
            # Registry mints the token immediately before the HTTP POST and
            # both services are co-located in the same cluster, so clocks
            # are NTP-synced within milliseconds. A 5-second leeway covers
            # realistic NTP jitter without extending the replay window by
            # 50% of the TTL. Issue #998.
            leeway=5,
        )

        token_use = claims.get("token_use")
        if token_use != "access":  # nosec B105 - OAuth2 token type validation per RFC 6749, not a password
            raise ValueError(f"Invalid token_use: {token_use}")

        caller = claims.get("sub", "service")
        logger.debug(f"Internal auth via JWT for: {caller}")
        return caller

    except pyjwt.ExpiredSignatureError:
        logger.warning("Expired JWT token for internal request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except (pyjwt.InvalidTokenError, ValueError) as e:
        logger.warning(f"JWT validation failed for internal request: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
