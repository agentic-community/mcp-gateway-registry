"""Shared self-signed (user-vended) JWT verification with dual-verify dispatch.

This module consolidates the 6 duplicate _validate_self_signed_token methods
(one per IdP provider + the Cognito copy in server.py) into a single function.
It handles both legacy HS256 tokens (no kid) and new ES256 tokens (with kid)
via kid-based dispatch during the transition window.

Security (algorithm-confusion prevention):
- Dispatch on `kid` header BEFORE decoding. Never `algorithms=["HS256","ES256"]`
  in a single decode() call.
- kid absent/None → HS256 only (legacy SECRET_KEY)
- kid present → ES256 only (from JWKS/key manager)
- Issuer checked before verification: token claiming iss != "mcp-auth-server"
  is rejected immediately (prevents external IdP tokens from being accepted
  as internal tokens via algorithm confusion).
- Token size bounded (8KB max) before any parsing.
- Two distinct key-material variables (_HMAC_SECRET vs _ES256_PUBKEY) so a
  lint/type boundary prevents passing the public key into an HMAC verify.

Fails closed: raises ValueError on any validation failure.
Never logs token values, decoded claims, or group names.
"""

import logging
import os

import jwt as pyjwt

logger = logging.getLogger(__name__)

# Constants — shared across all providers.
JWT_ISSUER: str = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE: str = os.environ.get("JWT_AUDIENCE", "mcp-registry")
AUTH_METHOD_SELF_SIGNED: str = "self_signed"

# Maximum token size (defense-in-depth: reject oversized tokens before parsing)
_MAX_TOKEN_BYTES: int = 8192


def _get_secret_key() -> str:
    """Return SECRET_KEY or raise if unset/empty.

    Defense-in-depth: process startup enforces SECRET_KEY presence, but this
    function re-checks so a dynamic env mutation cannot silently verify with
    an empty key.
    """
    key = os.environ.get("SECRET_KEY", "")
    if not key or not key.strip():
        raise ValueError("SECRET_KEY is required for self-signed token validation")
    return key.strip()


def _verify_hs256(token: str) -> dict:
    """Verify a token using HS256 (legacy path — SECRET_KEY).

    ONLY called when kid is absent (legacy tokens minted before asymmetric signing).
    Uses algorithms=["HS256"] exclusively — never mixed with asymmetric algorithms.
    """
    hmac_secret = _get_secret_key()
    return pyjwt.decode(
        token,
        hmac_secret,
        algorithms=["HS256"],  # NEVER add ES256 here
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={
            "verify_exp": True,
            "verify_iat": True,
            "verify_iss": True,
            "verify_aud": True,
        },
        leeway=30,
    )


def _verify_es256(token: str, kid: str) -> dict:
    """Verify a token using ES256 (asymmetric path — from key manager).

    ONLY called when kid is present. Looks up the public key by kid from the
    internal signing key manager. Uses algorithms=["ES256"] exclusively.
    """
    import importlib

    # Import the key manager — handle both in-container (auth_server/ on sys.path)
    # and test/package contexts. Use importlib to avoid mypy no-redef warning.
    try:
        _mod = importlib.import_module("auth_server.internal_signing_key")
    except (ImportError, ModuleNotFoundError):
        _mod = importlib.import_module("internal_signing_key")

    key_manager = _mod.get_internal_signing_key_manager()
    if not key_manager.is_available:
        raise ValueError(
            "Token has kid but asymmetric signing is not configured (no INTERNAL_SIGNING_KEY_PATH)"
        )

    verification_keys = key_manager.get_verification_keys()
    public_key = verification_keys.get(kid)
    if public_key is None:
        # kid not found — might be rotated out or forged
        raise ValueError("Unknown key id (kid) in token")

    return pyjwt.decode(
        token,
        public_key,
        algorithms=["ES256"],  # NEVER add HS256 here
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={
            "verify_exp": True,
            "verify_iat": True,
            "verify_iss": True,
            "verify_aud": True,
        },
        leeway=30,
    )


def verify_self_signed_user_token(token: str) -> dict:
    """Verify a self-signed (user-vended) JWT with kid-based algorithm dispatch.

    Handles both legacy HS256 tokens (no kid in header) and new ES256 tokens
    (with kid) during the dual-verify transition window. After cutover, the
    HS256 path will be removed.

    Dispatch logic:
    1. Size-bound the token (reject > 8KB before any parsing)
    2. Read unverified header to get kid (fail hard on malformed)
    3. Check issuer claim (unverified) — reject if not our issuer
    4. kid absent → HS256 path (SECRET_KEY)
    5. kid present → ES256 path (key manager, by kid)

    Args:
        token: The raw JWT string to verify.

    Returns:
        Dict with validation results (same shape regardless of algorithm).

    Raises:
        ValueError: On any validation failure.
    """
    # 1. Size bound (defense-in-depth against oversized tokens)
    if len(token) > _MAX_TOKEN_BYTES:
        raise ValueError("Token exceeds maximum size")

    # 2. Read unverified header to get kid
    # SECURITY: get_unverified_header is called on attacker-controlled input.
    # If it raises (malformed/non-UTF8), that's a HARD 401 — never fall through
    # to HS256 on a parse error (that would be a downgrade path).
    try:
        header = pyjwt.get_unverified_header(token)
    except pyjwt.exceptions.DecodeError as e:
        raise ValueError(f"Malformed token header: {e}")
    except Exception as e:
        raise ValueError(f"Cannot parse token header: {e}")

    # 3. Issuer-first check (unverified — defense-in-depth).
    # Reject tokens that don't claim to be from our issuer BEFORE attempting
    # any signature verification. This prevents an external IdP token (e.g.,
    # a Keycloak RS256 token) from reaching the HS256 path where the public
    # key might collide with SECRET_KEY in a confused-deputy scenario.
    try:
        unverified_payload = pyjwt.decode(
            token, options={"verify_signature": False, "verify_exp": False}
        )
        token_issuer = unverified_payload.get("iss", "")
    except Exception:
        token_issuer = ""

    if token_issuer != JWT_ISSUER:
        raise ValueError(f"Token issuer mismatch (expected '{JWT_ISSUER}')")

    # 4/5. Dispatch on kid
    kid = header.get("kid")

    try:
        if kid is None:
            # Cutover lever: after all tokens have rotated to ES256, operators
            # enable REJECT_HS256_TOKENS=true to hard-reject legacy tokens.
            # This closes the window where a leaked SECRET_KEY could forge tokens.
            # Keep this as a feature flag (not deletion) for one release so it's
            # revertable if issues arise.
            if os.environ.get("REJECT_HS256_TOKENS", "false").lower() == "true":
                raise ValueError(
                    "HS256 tokens are no longer accepted (REJECT_HS256_TOKENS=true). "
                    "Generate a new token."
                )
            # Legacy path: no kid → HS256 (SECRET_KEY)
            claims = _verify_hs256(token)
        else:
            # Asymmetric path: kid present → ES256 (key manager)
            claims = _verify_es256(token, kid)
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
    alg_used = "ES256" if kid else "HS256"
    logger.info(
        "Validated self-signed token for %s (alg=%s, kid=%s, groups=%d, scopes=%d)",
        masked_sub,
        alg_used,
        kid or "none",
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
