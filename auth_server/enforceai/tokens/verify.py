from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import jwt

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    UnauthorizedError,
)
from auth_server.enforceai.tokens.claims import (
    DEFAULT_CLOCK_SKEW_SECONDS,
    DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
    GatewayTokenClaims,
    validate_gateway_token_claims,
)


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def verify_gateway_token(
    token: str,
    *,
    keyring: GatewayKeyring,
    now: Optional[datetime] = None,
    expected_issuer: Optional[str] = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    max_lifetime_seconds: int = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
) -> GatewayTokenClaims:
    """Verify an RS256 gateway token locally and return validated claims.

    This function verifies the signature using the public key selected by JWT
    header `kid`, then validates required claims and time rules with leeway.

    Raises:
        UnauthorizedError: Missing/invalid token or claims.
        DependencyUnavailableError: Internal keyring/system failure.
    """
    if not token.strip():
        raise UnauthorizedError("Missing gateway token")

    try:
        header = jwt.get_unverified_header(token)
    except Exception as exc:  # noqa: BLE001 - map to 401
        raise UnauthorizedError("Invalid gateway token header") from exc

    kid = header.get("kid")
    if not isinstance(kid, str) or not kid.strip():
        raise UnauthorizedError("Missing gateway token kid")

    alg = header.get("alg")
    if alg != "RS256":
        raise UnauthorizedError("Unsupported gateway token algorithm")

    try:
        public_key = keyring.get_public_key(kid=kid)
    except Exception as exc:  # noqa: BLE001 - treat as dependency failure
        raise DependencyUnavailableError("Keyring unavailable") from exc

    if public_key is None:
        raise UnauthorizedError("Unknown gateway token kid")

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256"],
            options={
                "verify_aud": False,
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
            },
        )
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError("Invalid gateway token signature") from exc

    try:
        claims = GatewayTokenClaims.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - map to 401
        raise UnauthorizedError("Invalid gateway token claims") from exc

    if expected_issuer is not None and claims.iss != expected_issuer:
        raise UnauthorizedError("Gateway token issuer mismatch")

    effective_now = _ensure_aware_utc(now or datetime.now(timezone.utc)).replace(
        microsecond=0
    )
    try:
        validate_gateway_token_claims(
            claims,
            now=effective_now,
            clock_skew_seconds=clock_skew_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
        )
    except ValueError as exc:
        raise UnauthorizedError("Invalid gateway token claims") from exc
    return claims
