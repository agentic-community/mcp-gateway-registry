from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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


class AuditEventRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    event_id: int
    occurred_at: datetime
    user_id: str
    agent_id: str
    action: str = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1)
    request_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None

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
