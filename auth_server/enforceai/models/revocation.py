from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

USER_ID_SEPARATOR: str = "|"


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


class TokenRevocationRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    jti: str = Field(..., min_length=1)
    user_id: str
    agent_id: str
    revoked_at: datetime
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None

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

