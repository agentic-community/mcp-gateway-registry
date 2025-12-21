from __future__ import annotations

import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any, Optional

import jwt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .._validation import (
    _normalize_non_empty_str_list,
    _validate_user_id,
)
from ..errors import (
    UnauthorizedError,
)

ENFORCEAI_UI_SESSION_TOKEN_ISSUER: str = "enforceai-ui"
ENFORCEAI_UI_SESSION_TOKEN_AUDIENCE: str = "enforceai-management"

DEFAULT_UI_SESSION_TOKEN_TTL_SECONDS: int = 5 * 60
DEFAULT_UI_SESSION_TOKEN_CLOCK_SKEW_SECONDS: int = 60
DEFAULT_UI_SESSION_TOKEN_MAX_LIFETIME_SECONDS: int = 60 * 60


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_to_jwt_timestamp(
    value: datetime,
) -> int:
    value = _ensure_aware_utc(value).replace(microsecond=0)
    return int(value.timestamp())


def _jwt_timestamp_to_datetime(
    value: int,
) -> datetime:
    return datetime.fromtimestamp(
        value,
        tz=timezone.utc,
    ).replace(microsecond=0)


class EnforceAIUISessionTokenClaims(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    iss: str = Field(..., min_length=1)
    aud: list[str] = Field(default_factory=list)
    sub: str = Field(..., min_length=1)
    sid: str = Field(..., min_length=1)
    groups: list[str] = Field(default_factory=list)
    iat: int = Field(..., ge=0)
    exp: int = Field(..., ge=0)
    jti: str = Field(..., min_length=1)

    @field_validator("aud", mode="before")
    @classmethod
    def _normalize_aud(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("aud")
    @classmethod
    def _validate_aud(
        cls,
        value: list[str],
    ) -> list[str]:
        return _normalize_non_empty_str_list(value, label="aud")

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

    @field_validator("sid")
    @classmethod
    def _sid_non_empty(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("sid must not be empty")
        return stripped

    @field_validator("groups")
    @classmethod
    def _groups_are_non_empty_strings(
        cls,
        value: list[str],
    ) -> list[str]:
        return _normalize_non_empty_str_list(value, label="groups")

    @property
    def issued_at(self) -> datetime:
        return _jwt_timestamp_to_datetime(self.iat)

    @property
    def expires_at(self) -> datetime:
        return _jwt_timestamp_to_datetime(self.exp)


def mint_enforceai_ui_session_token(
    *,
    secret_key: str,
    user_id: str,
    session_id: str,
    groups: list[str],
    issuer: str = ENFORCEAI_UI_SESSION_TOKEN_ISSUER,
    audience: str = ENFORCEAI_UI_SESSION_TOKEN_AUDIENCE,
    ttl_seconds: int = DEFAULT_UI_SESSION_TOKEN_TTL_SECONDS,
    issued_at: Optional[datetime] = None,
    jti: Optional[str] = None,
) -> tuple[str, datetime]:
    if not secret_key.strip():
        raise ValueError("secret_key must be a non-empty string")
    if not issuer.strip():
        raise ValueError("issuer must be a non-empty string")
    if not audience.strip():
        raise ValueError("audience must be a non-empty string")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    effective_issued_at = _ensure_aware_utc(issued_at or datetime.now(timezone.utc)).replace(
        microsecond=0
    )
    expires_at = effective_issued_at + timedelta(seconds=ttl_seconds)
    token_jti = jti or str(uuid.uuid4())

    claims = EnforceAIUISessionTokenClaims(
        iss=issuer,
        aud=[audience],
        sub=user_id,
        sid=session_id,
        groups=groups,
        iat=_datetime_to_jwt_timestamp(effective_issued_at),
        exp=_datetime_to_jwt_timestamp(expires_at),
        jti=token_jti,
    )

    token = jwt.encode(
        payload=claims.model_dump(),
        key=secret_key,
        algorithm="HS256",
        headers={"typ": "enforceai-ui-session"},
    )
    if not isinstance(token, str):
        raise TypeError("JWT encode returned non-string token")

    return token, expires_at


def verify_enforceai_ui_session_token(
    token: str,
    *,
    secret_key: str,
    now: Optional[datetime] = None,
    expected_issuer: str = ENFORCEAI_UI_SESSION_TOKEN_ISSUER,
    expected_audience: str = ENFORCEAI_UI_SESSION_TOKEN_AUDIENCE,
    clock_skew_seconds: int = DEFAULT_UI_SESSION_TOKEN_CLOCK_SKEW_SECONDS,
    max_lifetime_seconds: int = DEFAULT_UI_SESSION_TOKEN_MAX_LIFETIME_SECONDS,
) -> EnforceAIUISessionTokenClaims:
    if not token.strip():
        raise UnauthorizedError("Missing token")
    if not secret_key.strip():
        raise UnauthorizedError("UI token secret unavailable")
    if clock_skew_seconds < 0:
        raise ValueError("clock_skew_seconds must be non-negative")
    if max_lifetime_seconds <= 0:
        raise ValueError("max_lifetime_seconds must be positive")

    try:
        payload = jwt.decode(
            token,
            key=secret_key,
            algorithms=["HS256"],
            options={
                "verify_aud": False,
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
            },
        )
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError("Invalid token signature") from exc

    try:
        claims = EnforceAIUISessionTokenClaims.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise UnauthorizedError("Invalid token claims") from exc

    if claims.iss != expected_issuer:
        raise UnauthorizedError("Token issuer mismatch")

    if expected_audience not in set(claims.aud):
        raise UnauthorizedError("Token audience mismatch")

    effective_now = _ensure_aware_utc(now or datetime.now(timezone.utc)).replace(microsecond=0)
    leeway = timedelta(seconds=clock_skew_seconds)

    if claims.exp <= claims.iat:
        raise UnauthorizedError("Invalid token claims")

    lifetime_seconds = claims.exp - claims.iat
    if lifetime_seconds > max_lifetime_seconds:
        raise UnauthorizedError("Invalid token claims")

    if claims.issued_at > effective_now + leeway:
        raise UnauthorizedError("Invalid token claims")

    if claims.expires_at <= effective_now - leeway:
        raise UnauthorizedError("Token expired")

    return claims

