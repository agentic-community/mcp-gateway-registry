"""Shared self-signed (user-vended) JWT verification.

This module consolidates the 6 duplicate _validate_self_signed_token methods
(one per IdP provider + the Cognito copy in server.py) into a single function.
All providers delegate to this; the Phase A asymmetric signing work will modify
only this function when adding kid-based dispatch + ES256 verification.

Security:
- Fails closed: raises ValueError on any validation failure (missing key,
  expired, wrong issuer/audience, wrong token_use).
- SECRET_KEY is REQUIRED (no fallback default — enforced at process startup
  and re-checked here as defense-in-depth).
- Never logs the token value, decoded claims payload, or group names.
  Logs only: subject (masked), group count, scope count.
"""

import logging
import os

import jwt as pyjwt

logger = logging.getLogger(__name__)

# Constants — shared across all providers. Read from env at module load
# (same pattern as the individual providers, ensuring identical values).
JWT_ISSUER: str = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE: str = os.environ.get("JWT_AUDIENCE", "mcp-registry")
AUTH_METHOD_SELF_SIGNED: str = "self_signed"


def _get_secret_key() -> str:
    """Return SECRET_KEY or raise if unset/empty.

    Defense-in-depth: process startup enforces SECRET_KEY presence, but this
    function re-checks so a race condition or dynamic env mutation cannot
    silently mint/verify with an empty key.
    """
    key = os.environ.get("SECRET_KEY", "")
    if not key or not key.strip():
        raise ValueError("SECRET_KEY is required for self-signed token validation")
    return key.strip()


def verify_self_signed_user_token(token: str) -> dict:
    """Verify a self-signed (user-vended) HS256 JWT.

    This is the consolidated verification function for tokens minted by the
    auth-server's /tokens/generate endpoint. All IdP providers call this
    instead of maintaining their own copy.

    Args:
        token: The raw JWT string to verify.

    Returns:
        Dict with validation results:
        {
            "valid": True,
            "method": "self_signed",
            "data": <full claims dict>,
            "client_id": <from claims or "user-generated">,
            "username": <sub claim>,
            "email": <email claim or "">,
            "expires_at": <exp claim>,
            "scopes": <list of scope strings>,
            "groups": <list of group strings>,
            "token_type": "user_generated",
        }

    Raises:
        ValueError: On any validation failure (expired, invalid signature,
            wrong issuer/audience, missing SECRET_KEY, wrong token_use).
    """
    secret_key = _get_secret_key()

    try:
        claims = pyjwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
            leeway=30,  # 30s leeway for clock skew across replicas
        )
    except pyjwt.ExpiredSignatureError:
        logger.warning("Self-signed token validation failed: token has expired")
        raise ValueError("Token has expired")
    except pyjwt.InvalidTokenError as e:
        logger.warning("Self-signed token validation failed: invalid token")
        raise ValueError(f"Invalid self-signed token: {e}")

    # Validate token_use claim (must be "access")
    token_use = claims.get("token_use")
    if token_use != "access":  # nosec B105 - OAuth2 token type, not a password
        raise ValueError(f"Invalid token_use: {token_use}")

    # Extract scopes (string or list)
    scopes: list[str] = []
    scope_value = claims.get("scope", "")
    if isinstance(scope_value, str):
        scopes = scope_value.split() if scope_value else []
    elif isinstance(scope_value, list):
        scopes = scope_value

    # Extract groups (string or list)
    groups: list[str] = claims.get("groups", [])
    if isinstance(groups, str):
        groups = [groups]

    # Log success with masked subject (never log full username, groups, or scopes)
    sub = str(claims.get("sub") or "")
    masked_sub = f"{sub[:4]}***" if sub else "unknown"
    logger.info(
        "Validated self-signed token for %s (groups=%d, scopes=%d)",
        masked_sub,
        len(groups),
        len(scopes),
    )

    return {
        "valid": True,
        "method": AUTH_METHOD_SELF_SIGNED,
        "data": claims,
        "client_id": claims.get("client_id", "user-generated"),
        "username": claims.get("sub", ""),
        "email": claims.get("email", ""),
        "expires_at": claims.get("exp"),
        "scopes": scopes,
        "groups": groups,
        "token_type": "user_generated",
    }
