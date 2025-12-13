from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

USER_ID_SEPARATOR: str = "|"

Provider = Literal[
    "oidc",
    "gateway-token",
    "api-key",
]


def _validate_user_id(
    user_id: str,
) -> str:
    parts = user_id.split(USER_ID_SEPARATOR)
    if len(parts) != 2:
        raise ValueError("user_id must be in '<iss>|<sub>' format")
    issuer, subject = parts
    if not issuer or not subject:
        raise ValueError("user_id must be in '<iss>|<sub>' format")
    return user_id


def _validate_agent_id(
    agent_id: str,
) -> str:
    try:
        parsed = uuid.UUID(agent_id)
    except ValueError as exc:
        raise ValueError("agent_id must be a UUIDv4 string") from exc

    if parsed.version != 4:
        raise ValueError("agent_id must be a UUIDv4 string")

    return agent_id


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
        return _validate_agent_id(value)

    @field_validator("scopes")
    @classmethod
    def _scopes_are_non_empty_strings(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("scopes must not contain empty strings")
            normalized.append(stripped)
        return normalized

