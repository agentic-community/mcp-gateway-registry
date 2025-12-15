from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)

from .._validation import (
    _validate_user_id,
)


SessionAuthMethod = Literal["oidc", "password"]


class SessionRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    session_id: str
    user_id: str
    auth_method: SessionAuthMethod

    created_at: datetime
    expires_at: datetime
    last_seen_at: datetime
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def _session_id_non_empty(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id must not be empty")
        return stripped

    @field_validator("user_id")
    @classmethod
    def _user_id_is_canonical(
        cls,
        value: str,
    ) -> str:
        return _validate_user_id(value)

