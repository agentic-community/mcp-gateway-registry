from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from ._validation import (
    USER_ID_SEPARATOR,
    _normalize_non_empty_str_list,
    _validate_user_id,
    _validate_uuid4,
)

Provider = Literal[
    "oidc",
    "gateway-token",
    "api-key",
]


def build_user_id(
    issuer: str,
    subject: str,
) -> str:
    if USER_ID_SEPARATOR in issuer or USER_ID_SEPARATOR in subject:
        raise ValueError("issuer and subject must not contain '|'")
    return _validate_user_id(f"{issuer}{USER_ID_SEPARATOR}{subject}")


def parse_user_id(
    user_id: str,
) -> tuple[str, str]:
    validated = _validate_user_id(user_id)
    issuer, subject = validated.split(USER_ID_SEPARATOR, maxsplit=1)
    return issuer, subject


class IdentityContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    user_id: str = Field(
        ...,
        description="OIDC issuer-namespaced subject in '<iss>|<sub>' format",
    )
    agent_id: str = Field(
        ...,
        description="Agent UUIDv4 string (canonical identifier)",
    )
    provider: Provider
    scopes: list[str]
    user_roles: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None

    @field_validator("user_id")
    @classmethod
    def _user_id_is_canonical(
        cls,
        value: str,
    ) -> str:
        return _validate_user_id(value)

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
