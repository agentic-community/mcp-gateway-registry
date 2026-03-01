"""
Internal service-to-service authentication using self-signed JWTs.

This module provides utilities for authenticating internal API calls
between services (e.g., mcpgw -> registry, registry -> auth-server)
using JWTs signed with the shared SECRET_KEY instead of hardcoded
admin credentials.
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

    Accepts either:
    - Bearer JWT signed with the shared SECRET_KEY (preferred)
    - Basic Auth with ADMIN_USER/ADMIN_PASSWORD (deprecated fallback)

    Args:
        request: The FastAPI request object

    Returns:
        Caller identity string (e.g., 'registry-service' or admin username)

    Raises:
        HTTPException: If authentication fails
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth_header.startswith("Bearer "):
        return _validate_bearer_token(auth_header)

    if auth_header.startswith("Basic "):
        return _validate_basic_auth_deprecated(auth_header)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unsupported authentication scheme",
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
            leeway=30,
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


def _validate_basic_auth_deprecated(auth_header: str) -> str:
    """Validate Basic Auth credentials (deprecated, for backward compatibility)."""
    import base64

    logger.warning(
        "Internal API called with deprecated Basic Auth. "
        "Migrate to Bearer token using shared SECRET_KEY."
    )

    try:
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)
    except (IndexError, ValueError, Exception) as e:
        logger.warning(f"Failed to decode Basic Auth credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format",
            headers={"WWW-Authenticate": "Basic"},
        )

    admin_user = os.environ.get("ADMIN_USER", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_password:
        logger.error("ADMIN_PASSWORD not set and Basic Auth attempted on internal endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    if username != admin_user or password != admin_password:
        logger.warning(f"Failed admin authentication attempt from {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return username
