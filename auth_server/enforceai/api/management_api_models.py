from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: list[str]
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, object]] = None


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, object]] = None


class CreateApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: Optional[list[str]] = None
    expires_at: Optional[datetime] = None


class CreateApiKeyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    secret: str
    api_key_value: str


class MintTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: list[str]
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    expires_at: Optional[datetime] = None


class MintTokenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str


class RevokeTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: Optional[str] = None
    jti: Optional[str] = None
    gateway_token: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self) -> "RevokeTokenRequest":
        if self.gateway_token is not None:
            if self.agent_id is not None or self.jti is not None:
                raise ValueError("Provide either gateway_token or (agent_id and jti), not both")
            if not self.gateway_token.strip():
                raise ValueError("gateway_token must be a non-empty string")
            return self

        if self.agent_id is None or self.jti is None:
            raise ValueError("Provide either gateway_token or both agent_id and jti")
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not self.jti.strip():
            raise ValueError("jti must be a non-empty string")
        return self

