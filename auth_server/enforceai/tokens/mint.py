from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.tokens.claims import (
    DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
    GatewayTokenClaims,
    datetime_to_jwt_timestamp,
    validate_gateway_token_claims,
)


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def mint_gateway_token(
    *,
    keyring: GatewayKeyring,
    issuer: str,
    user_id: str,
    agent_id: str,
    scopes: list[str],
    issued_at: Optional[datetime] = None,
    expires_at: Optional[datetime] = None,
    ttl_seconds: Optional[int] = None,
    jti: Optional[str] = None,
    max_lifetime_seconds: int = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
) -> str:
    """Mint an RS256 gateway token with a `kid` header.

    Args:
        keyring: Loaded gateway keyring with signing private key and active kid.
        issuer: Token `iss` claim (gateway identifier).
        user_id: Canonical user id in `<iss>|<sub>` format (stored in `sub`).
        agent_id: Agent UUIDv4 string.
        scopes: Token scopes; must be non-empty strings.
        issued_at: Optional deterministic issued-at for tests.
        expires_at: Optional explicit expiry (mutually exclusive with `ttl_seconds`).
        ttl_seconds: Optional TTL in seconds (mutually exclusive with `expires_at`).
        jti: Optional deterministic token id for tests (defaults to UUIDv4).
        max_lifetime_seconds: Maximum allowed `exp - iat`.

    Returns:
        Signed JWT string.

    Raises:
        ValueError: On invalid inputs or temporal rules violations.
    """
    if not issuer.strip():
        raise ValueError("issuer must be a non-empty string")
    if not scopes:
        raise ValueError("scopes must be a non-empty list")

    if expires_at is not None and ttl_seconds is not None:
        raise ValueError("Provide only one of expires_at or ttl_seconds")

    effective_issued_at = _ensure_aware_utc(issued_at or datetime.now(timezone.utc)).replace(
        microsecond=0
    )

    if ttl_seconds is not None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        effective_expires_at = effective_issued_at + timedelta(seconds=ttl_seconds)
    elif expires_at is not None:
        effective_expires_at = _ensure_aware_utc(expires_at).replace(microsecond=0)
    else:
        effective_expires_at = effective_issued_at + timedelta(hours=1)

    effective_jti = jti or str(uuid.uuid4())

    claims = GatewayTokenClaims(
        iss=issuer,
        sub=user_id,
        agent_id=agent_id,
        scopes=scopes,
        iat=datetime_to_jwt_timestamp(effective_issued_at),
        exp=datetime_to_jwt_timestamp(effective_expires_at),
        jti=effective_jti,
    )
    validate_gateway_token_claims(
        claims,
        now=effective_issued_at,
        max_lifetime_seconds=max_lifetime_seconds,
    )

    token = jwt.encode(
        payload=claims.model_dump(),
        key=keyring.signing_private_key,
        algorithm="RS256",
        headers={"kid": keyring.active_kid},
    )
    if not isinstance(token, str):
        raise TypeError("JWT encode returned non-string token")

    return token

