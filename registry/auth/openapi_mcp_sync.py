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


# ---------------------------------------------------------------------------
# Per-user credential envelope (per-user REST credentials).
#
# The registry's egress vend packs one user's keys for one OpenAPI MCP server
# into a signed envelope. The auth server's mcp-proxy hop forwards it as an
# opaque header value and never holds the signing key; the container verifies
# it and reads each target's key for this one request.
# ---------------------------------------------------------------------------

ENVELOPE_AUDIENCE: str = "openapi-mcp-user-credentials"
ENVELOPE_TOKEN_KIND: str = "openapi-mcp-user-credentials"
ENVELOPE_KEY_INFO: bytes = b"openapi-mcp-user-cred-v1"
ENVELOPE_TTL_SECONDS: int = 60
ENVELOPE_HEADER: str = "X-OpenAPI-MCP-User-Credentials"
ENVELOPE_REQUIRED_CLAIMS: tuple[str, ...] = (
    "iss",
    "aud",
    "exp",
    "iat",
    "jti",
    "token_kind",
    "server",
    "credentials",
)


def _envelope_key(
    secret: str,
) -> bytes:
    return hmac.new(_require_secret(secret).encode(), ENVELOPE_KEY_INFO, hashlib.sha256).digest()


def mint_credential_envelope(
    secret: str,
    server_slug: str,
    credentials: dict[str, dict[str, str]],
) -> str:
    """Sign one user's per-target keys for one server.

    Args:
        secret: The value of ``OPENAPI_MCP_SYNC_SECRET``.
        server_slug: The OpenAPI MCP server the keys are for.
        credentials: ``{target_name: {"key": ..., "base_url": ...}}``. May be
            empty, so a user with no keys still reaches the server's shared
            targets.

    Returns:
        The encoded JWT. It is signed, not encrypted: it travels only on the
        internal hops, like the registry's state push.
    """
    now = int(time.time())
    claims = {
        "iss": SYNC_ISSUER,
        "aud": ENVELOPE_AUDIENCE,
        "token_kind": ENVELOPE_TOKEN_KIND,
        "server": server_slug,
        "credentials": credentials,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + ENVELOPE_TTL_SECONDS,
    }
    return pyjwt.encode(claims, _envelope_key(secret), algorithm=SYNC_ALGORITHM)


def _check_envelope_credentials(
    credentials: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(credentials, dict):
        raise SyncTokenError("credential envelope rejected: credentials is not an object")
    for name, entry in credentials.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SyncTokenError("credential envelope rejected: malformed entry")
        if not isinstance(entry.get("key"), str) or not isinstance(entry.get("base_url"), str):
            raise SyncTokenError("credential envelope rejected: entry needs key and base_url")
    return credentials


def verify_credential_envelope(
    token: str,
    secret: str,
    server_slug: str,
    seen_jtis: JtiCache,
) -> dict[str, dict[str, str]]:
    """Verify an envelope for this server and return its per-target keys.

    Raises:
        SyncTokenError: On a bad signature, wrong audience or kind, expiry, a
            lifetime over 60 seconds, a reused jti, a different server, or a
            malformed credentials claim. The message never contains a key.
    """
    if not token:
        raise SyncTokenError("missing credential envelope")
    try:
        claims = pyjwt.decode(
            token,
            _envelope_key(secret),
            algorithms=[SYNC_ALGORITHM],
            audience=ENVELOPE_AUDIENCE,
            issuer=SYNC_ISSUER,
            leeway=SYNC_LEEWAY_SECONDS,
            options={"require": list(ENVELOPE_REQUIRED_CLAIMS)},
        )
    except pyjwt.PyJWTError as e:
        raise SyncTokenError(f"credential envelope rejected: {type(e).__name__}") from e
    if claims.get("token_kind") != ENVELOPE_TOKEN_KIND:
        raise SyncTokenError("credential envelope rejected: wrong token_kind")
    if int(claims["exp"]) - int(claims["iat"]) > ENVELOPE_TTL_SECONDS:
        raise SyncTokenError("credential envelope rejected: lifetime too long")
    if claims.get("server") != server_slug:
        raise SyncTokenError("credential envelope rejected: issued for another server")
    jti = claims.get("jti")
    if not isinstance(jti, str) or not jti:
        raise SyncTokenError("credential envelope rejected: invalid jti")
    if not seen_jtis.consume(f"env:{jti}", float(claims["exp"]) + SYNC_LEEWAY_SECONDS):
        raise SyncTokenError("credential envelope rejected: jti already used")
    return _check_envelope_credentials(claims.get("credentials"))
