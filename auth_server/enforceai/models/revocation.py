from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .._validation import (
    _validate_user_id,
    _validate_uuid4,
)


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
        return _validate_uuid4(value, label="agent_id")
