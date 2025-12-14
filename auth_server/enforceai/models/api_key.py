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
    _validate_user_id,
    _validate_uuid4,
)


class ApiKeyRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    key_id: str = Field(..., min_length=1)
    secret_hash: str = Field(
        ...,
        min_length=1,
        description="Hashed-at-rest verifier (opaque at this stage)",
    )
    user_id: str
    agent_id: str

    scopes: Optional[list[str]] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    created_at: datetime
    last_used_at: Optional[datetime] = None

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
        value: Optional[list[str]],
    ) -> Optional[list[str]]:
        return _normalize_optional_non_empty_str_list(value, label="scopes")
