from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .._validation import (
    _normalize_non_empty_str_list,
    _validate_user_id,
    _validate_uuid4,
)

DEFAULT_CLOCK_SKEW_SECONDS: int = 60
DEFAULT_MAX_TOKEN_LIFETIME_SECONDS: int = 365 * 24 * 60 * 60


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def datetime_to_jwt_timestamp(
    value: datetime,
) -> int:
    """Convert a datetime to an integer JWT timestamp (seconds since epoch)."""
    value = _ensure_aware_utc(value).replace(microsecond=0)
    return int(value.timestamp())


def jwt_timestamp_to_datetime(
    value: int,
) -> datetime:
    """Convert an integer JWT timestamp to a UTC datetime."""
    return datetime.fromtimestamp(
        value,
        tz=timezone.utc,
    ).replace(microsecond=0)


class GatewayTokenClaims(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    iss: str = Field(..., min_length=1)
    sub: str = Field(
        ...,
        min_length=1,
        description="Canonical user_id in '<iss>|<sub>' format",
    )
    agent_id: str = Field(..., min_length=1)
    scopes: list[str] = Field(default_factory=list)

    iat: int = Field(..., ge=0, description="Issued-at (JWT timestamp, seconds)")
    exp: int = Field(..., ge=0, description="Expires-at (JWT timestamp, seconds)")
    jti: str = Field(..., min_length=1)

    @field_validator("sub")
    @classmethod
    def _sub_is_user_id(
        cls,
        value: str,
    ) -> str:
        return _validate_user_id(
            value,
            label="sub",
            prefix="sub must be a canonical user_id",
        )

    @field_validator("agent_id")
    @classmethod
    def _agent_id_is_uuid4(
        cls,
        value: str,
    ) -> str:
        return _validate_uuid4(value, label="agent_id")

    @field_validator("scopes")
    @classmethod
    def _scopes_are_non_empty_strings(
        cls,
        value: list[str],
    ) -> list[str]:
        return _normalize_non_empty_str_list(value, label="scopes")

    @property
    def issued_at(self) -> datetime:
        return jwt_timestamp_to_datetime(self.iat)

    @property
    def expires_at(self) -> datetime:
        return jwt_timestamp_to_datetime(self.exp)


def validate_gateway_token_claims(
    claims: GatewayTokenClaims,
    *,
    now: datetime,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    max_lifetime_seconds: int = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
) -> None:
    """Validate temporal safety rules for gateway tokens.

    Args:
        claims: Parsed gateway token claims.
        now: Current time to validate against.
        clock_skew_seconds: Allowed clock skew leeway for `iat` and `exp`.
        max_lifetime_seconds: Maximum acceptable token lifetime (`exp - iat`).

    Raises:
        ValueError: If claims violate temporal safety rules.
    """
    if clock_skew_seconds < 0:
        raise ValueError("clock_skew_seconds must be non-negative")
    if max_lifetime_seconds <= 0:
        raise ValueError("max_lifetime_seconds must be positive")

    effective_now = _ensure_aware_utc(now).replace(microsecond=0)
    leeway = timedelta(seconds=clock_skew_seconds)

    issued_at = claims.issued_at
    expires_at = claims.expires_at

    if claims.exp <= claims.iat:
        raise ValueError("exp must be greater than iat")

    lifetime_seconds = claims.exp - claims.iat
    if lifetime_seconds > max_lifetime_seconds:
        raise ValueError("Token lifetime exceeds maximum allowed duration")

    if issued_at > effective_now + leeway:
        raise ValueError("iat is too far in the future")

    if expires_at <= effective_now - leeway:
        raise ValueError("Token is expired")


def validate_optional_gateway_token_claims(
    claims: GatewayTokenClaims,
    *,
    now: Optional[datetime] = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    max_lifetime_seconds: int = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
) -> None:
    """Validate gateway claims using an optional `now` (defaults to UTC now)."""
    effective_now = now or datetime.now(timezone.utc)
    validate_gateway_token_claims(
        claims,
        now=effective_now,
        clock_skew_seconds=clock_skew_seconds,
        max_lifetime_seconds=max_lifetime_seconds,
    )
