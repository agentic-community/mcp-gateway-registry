"""
Internal service-to-service authentication using self-signed JWTs.

This module provides utilities for authenticating internal API calls
between services (e.g., mcpgw -> registry, registry -> auth-server)
using JWTs signed with the shared SECRET_KEY.
"""

import hashlib
import hmac
import logging
import os
import time

import jwt as pyjwt
from fastapi import Header, HTTPException, Request, status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# JWT constants for internal service-to-service tokens.
#
# IMPORTANT (Security Finding 1): the internal audience MUST stay distinct from
# the user-token audience (``_USER_JWT_AUDIENCE = "mcp-registry"`` in
# auth_server/server.py). These two token classes were historically minted and
# validated against the SAME audience/key, which made a low-privilege user token
# interchangeable with a trusted internal service credential. Do NOT re-merge
# them: internal tokens use ``mcp-internal`` + a mandatory ``token_kind`` claim +
# a separate derived signing key so a user token can never satisfy the internal
# gate. The issuer is legitimately shared (the same auth server issues both).
_INTERNAL_JWT_ISSUER: str = "mcp-auth-server"
_INTERNAL_JWT_AUDIENCE: str = "mcp-internal"
_INTERNAL_JWT_TTL_SECONDS: int = 60
_INTERNAL_TOKEN_KIND: str = "internal-service"
_INTERNAL_KEY_INFO: bytes = b"mcp-internal-token-v1"


def _derive_internal_signing_key(
    secret_key: str,
) -> bytes:
    """Derive a dedicated signing key for internal service tokens.

    Uses HMAC-SHA256 over the shared ``SECRET_KEY`` with a fixed info
    string so that a user token (signed with the raw ``SECRET_KEY``) can
    never verify against the internal key, and vice versa. This is a
    deterministic derivation: both services derive the same key from the
    same ``SECRET_KEY`` with no additional configuration.

    Args:
        secret_key: The shared application secret.

    Returns:
        A 32-byte key suitable for HS256 signing.
    """
    return hmac.new(secret_key.encode(), _INTERNAL_KEY_INFO, hashlib.sha256).digest()


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
        "token_kind": _INTERNAL_TOKEN_KIND,
        "token_use": "access",
        "iat": now,
        "exp": now + _INTERNAL_JWT_TTL_SECONDS,
    }
    signing_key = _derive_internal_signing_key(secret_key)
    return pyjwt.encode(claims, signing_key, algorithm="HS256")


async def validate_internal_session_secret(
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
) -> None:
    """Gate the internal virtual-server session endpoints.

    These routes (``/api/internal/sessions/*``) are called only by the nginx
    Lua router through the ``internal;``-protected ``/_internal/sessions/``
    location, which injects the shared ``SECRET_KEY`` as the ``X-Internal-Secret``
    header. Any request arriving by another path (a direct hit on the app port,
    or the externally-reachable ``/api/`` proxy location) will not carry the
    header and is rejected.

    A static shared secret is used here rather than the 60-second internal JWT
    (``validate_internal_auth``) because the nginx/Lua layer cannot mint a
    signed JWT (no SECRET_KEY-backed signing library at that layer); the secret
    is compared in constant time to avoid leaking it via timing.

    Args:
        x_internal_secret: Value of the ``X-Internal-Secret`` request header.

    Raises:
        HTTPException: 500 if SECRET_KEY is unset; 403 if the header is missing
            or does not match.
    """
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        logger.error("SECRET_KEY not set, cannot validate internal session request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    if not x_internal_secret or not hmac.compare_digest(x_internal_secret, secret_key):
        logger.warning("Rejected internal session request with missing/invalid X-Internal-Secret")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )


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

    signing_key = _derive_internal_signing_key(secret_key)

    try:
        claims = pyjwt.decode(
            token,
            signing_key,
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

        # Defense-in-depth (Security Finding 1): even if the audience/key ever
        # collide, a token that is not explicitly an internal-service token is
        # rejected. This also explicitly denies user/resource-kind tokens.
        token_kind = claims.get("token_kind")
        if token_kind != _INTERNAL_TOKEN_KIND:
            raise ValueError(f"Not an internal service token: token_kind={token_kind}")

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
