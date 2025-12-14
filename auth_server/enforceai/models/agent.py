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
    _normalize_non_empty_str_list,
    _normalize_optional_non_empty_str_list,
    _validate_user_id,
    _validate_uuid4,
)


class AgentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    user_id: str
    agent_id: str

    scopes: list[str] = Field(default_factory=list)
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    revoked_at: Optional[datetime] = None
    tokens_valid_after: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime

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

    @field_validator("allowed_tools")
    @classmethod
    def _allowed_tools_are_non_empty_strings(
        cls,
        value: Optional[list[str]],
    ) -> Optional[list[str]]:
        return _normalize_optional_non_empty_str_list(value, label="allowed_tools")
