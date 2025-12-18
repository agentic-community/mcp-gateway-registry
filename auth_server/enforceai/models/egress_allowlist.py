from __future__ import annotations

from datetime import datetime
from typing import (
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

EgressAllowlistEntryKind = Literal[
    "hostname",
    "domain-suffix",
    "ip-cidr",
]


class EgressAllowlistEntryRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    entry_id: int = Field(..., ge=1)
    kind: EgressAllowlistEntryKind
    value: str = Field(..., min_length=1)
    comment: Optional[str] = None

    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("value")
    @classmethod
    def _normalize_value(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be a non-empty string")
        return stripped

    @field_validator("comment")
    @classmethod
    def _normalize_comment(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

