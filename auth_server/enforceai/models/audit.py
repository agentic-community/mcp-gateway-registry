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


# Default time window for audit queries (60 minutes in seconds)
DEFAULT_AUDIT_WINDOW_SECONDS: int = 60 * 60

# Default and maximum page sizes
DEFAULT_AUDIT_PAGE_SIZE: int = 100
MAX_AUDIT_PAGE_SIZE: int = 500


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


class AuditEventsQueryResult(BaseModel):
    """Result of a paginated audit events query."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    items: list[AuditEventRecord]
    next_cursor: Optional[str] = None
    server_time: datetime
