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
    _normalize_optional_non_empty_str_list,
)

class ApiKeySummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    key_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)

    scopes: Optional[list[str]] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    created_at: datetime
    last_used_at: Optional[datetime] = None

    @field_validator("scopes")
    @classmethod
    def _scopes_are_non_empty_strings(
        cls,
        value: Optional[list[str]],
    ) -> Optional[list[str]]:
        return _normalize_optional_non_empty_str_list(value, label="scopes")
