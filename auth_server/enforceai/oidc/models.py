from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class OIDCValidatedToken(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    issuer: str = Field(
        ...,
        min_length=1,
    )
    subject: str = Field(
        ...,
        min_length=1,
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description='Canonical user_id derived as "<iss>|<sub>"',
    )

    audiences: list[str] = Field(
        default_factory=list,
        description="Normalized token audiences (aud claim)",
    )
    scopes: list[str] = Field(
        default_factory=list,
        description="Normalized scopes (for authorization inputs later)",
    )
    roles: list[str] = Field(
        default_factory=list,
        description="Normalized roles/groups for audit only",
    )

    claims: dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal verified claim subset for audit/debug (no raw token)",
    )

