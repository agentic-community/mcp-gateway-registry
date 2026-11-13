"""Sync tokens for registry-to-openapi-mcp-container calls.

The registry pushes OpenAPI MCP server state (including decrypted target
credentials) to the ``openapi-mcp`` container. Every call carries a fresh
60-second HS256 token that is valid once.

The signing key is derived from ``OPENAPI_MCP_SYNC_SECRET``, never from
``SECRET_KEY``, so the container never holds the registry's master key and a
compromised container cannot mint user or internal tokens.

This module imports only the standard library and ``jwt`` because the
container image copies it unchanged.
"""

import hashlib
import hmac
import logging
import threading
import time
import uuid

import jwt as pyjwt

logger = logging.getLogger(__name__)


SYNC_ISSUER: str = "mcp-registry"
SYNC_AUDIENCE: str = "openapi-mcp-sync"
SYNC_TOKEN_KIND: str = "openapi-mcp-sync"
SYNC_TOKEN_TTL_SECONDS: int = 60
SYNC_SUBJECT: str = "registry-service"
SYNC_KEY_INFO: bytes = b"openapi-mcp-sync-v1"
SYNC_ALGORITHM: str = "HS256"
# Tolerated clock difference between the registry and the container.
SYNC_LEEWAY_SECONDS: int = 5
REQUIRED_CLAIMS: tuple[str, ...] = ("iss", "aud", "sub", "exp", "iat", "jti", "token_kind")


class SyncTokenError(Exception):
    """A sync token failed verification. The message is safe to log."""


class JtiCache:
    """Remember the jti of every accepted token until it expires.

    In memory is enough because only one container replica runs. A restart
    clears the cache, but every token it held expires within 60 seconds.
    """

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _prune(
        self,
        now: float,
    ) -> None:
        expired = [jti for jti, expires_at in self._seen.items() if expires_at <= now]
        for jti in expired:
            del self._seen[jti]

    def consume(
        self,
        jti: str,
        expires_at: float,
    ) -> bool:
        """Record a jti. Returns False when it was already used."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if jti in self._seen:
                return False
            self._seen[jti] = expires_at
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)


def _derive_signing_key(
    secret: str,
) -> bytes:
    """Derive the HS256 key from the sync secret with a fixed info string."""
    return hmac.new(secret.encode(), SYNC_KEY_INFO, hashlib.sha256).digest()


def _require_secret(
    secret: str | None,
) -> str:
    """Fail closed on an unset secret. Strength is checked once at startup."""
    if not secret or not secret.strip():
        raise SyncTokenError("OPENAPI_MCP_SYNC_SECRET is not set")
    return secret.strip()


def mint_sync_token(
    secret: str,
    purpose: str,
) -> str:
    """Mint a 60-second, single-use token for one registry-to-container call.

    Args:
        secret: The value of ``OPENAPI_MCP_SYNC_SECRET``.
        purpose: What the call does (for example ``put_server``), for the
            container's logs.

    Returns:
        The encoded JWT.

    Raises:
        SyncTokenError: If the secret is unset.
    """
    signing_key = _derive_signing_key(_require_secret(secret))
    now = int(time.time())
    claims = {
        "iss": SYNC_ISSUER,
        "aud": SYNC_AUDIENCE,
        "sub": SYNC_SUBJECT,
        "purpose": purpose,
        "token_kind": SYNC_TOKEN_KIND,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + SYNC_TOKEN_TTL_SECONDS,
    }
    return pyjwt.encode(claims, signing_key, algorithm=SYNC_ALGORITHM)


def verify_sync_token(
    token: str,
    secret: str,
    seen_jtis: JtiCache,
) -> dict:
    """Verify a sync token and consume its jti.

    Checks the signature, issuer, audience, expiry, ``token_kind`` and that the
    lifetime is no longer than ``SYNC_TOKEN_TTL_SECONDS``, then rejects a jti
    that was already used.

    Args:
        token: The bearer token from the request.
        secret: The value of ``OPENAPI_MCP_SYNC_SECRET``.
        seen_jtis: The cache of consumed jtis.

    Returns:
        The verified claims.

    Raises:
        SyncTokenError: On any failure. The message names the reason and
            never contains the token.
    """
    if not token:
        raise SyncTokenError("missing sync token")
    signing_key = _derive_signing_key(_require_secret(secret))
    try:
        claims = pyjwt.decode(
            token,
            signing_key,
            algorithms=[SYNC_ALGORITHM],
            audience=SYNC_AUDIENCE,
            issuer=SYNC_ISSUER,
            leeway=SYNC_LEEWAY_SECONDS,
            options={"require": list(REQUIRED_CLAIMS)},
        )
    except pyjwt.PyJWTError as e:
        raise SyncTokenError(f"sync token rejected: {type(e).__name__}") from e

    if claims.get("token_kind") != SYNC_TOKEN_KIND:
        raise SyncTokenError("sync token rejected: wrong token_kind")
    lifetime = int(claims["exp"]) - int(claims["iat"])
    if lifetime > SYNC_TOKEN_TTL_SECONDS:
        raise SyncTokenError("sync token rejected: lifetime too long")
    jti = claims.get("jti")
    if not isinstance(jti, str) or not jti:
        raise SyncTokenError("sync token rejected: invalid jti")
    expires_at = float(claims["exp"]) + SYNC_LEEWAY_SECONDS
    if not seen_jtis.consume(jti, expires_at):
        raise SyncTokenError("sync token rejected: jti already used")
    return claims
