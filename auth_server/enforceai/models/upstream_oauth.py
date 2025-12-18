from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .upstream_auth import (
    UpstreamCredentialBinding,
)

UpstreamOAuthCredentialType = Literal[
    "oauth2",
    "oidc",
    "provider-oauth",
]


class UpstreamOAuthStateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str = Field(..., min_length=1)
    server_path: str = Field(..., min_length=1)
    credential_type: UpstreamOAuthCredentialType
    credential_binding: UpstreamCredentialBinding
    user_id: str = Field(..., min_length=1)
    agent_id: Optional[str] = None
    provider: str = Field(..., min_length=1)
    redirect_uri: str = Field(..., min_length=1)
    created_at: datetime
    expires_at: datetime

    @field_validator("server_path")
    @classmethod
    def _server_path_is_canonical(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped.startswith("/"):
            raise ValueError("server_path must start with '/'")
        return stripped

    @field_validator("provider")
    @classmethod
    def _normalize_provider(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _validate_binding(self) -> "UpstreamOAuthStateRecord":
        if self.credential_binding == "user":
            if self.agent_id is not None:
                raise ValueError("agent_id must be null for user binding")
            return self

        if self.credential_binding == "user+agent":
            if self.agent_id is None:
                raise ValueError("agent_id is required for user+agent binding")
            return self

        raise ValueError("OAuth state only supports user and user+agent bindings")


class UpstreamOAuthStateSecret(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(
        ...,
        repr=False,
        description="Decrypted OAuth state payload (PKCE verifier, nonce).",
    )


class UpstreamOAuthStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_path: str
    credential_type: UpstreamOAuthCredentialType
    credential_binding: UpstreamCredentialBinding = Field(default="user")
    agent_id: Optional[str] = None
    provider: str
    scopes: Optional[list[str]] = None

    @field_validator("server_path")
    @classmethod
    def _normalize_server_path(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("server_path must be a non-empty string")
        if not stripped.startswith("/"):
            stripped = f"/{stripped}"
        return stripped.rstrip("/") or "/"

    @field_validator("provider")
    @classmethod
    def _normalize_provider(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _validate_binding(self) -> "UpstreamOAuthStartRequest":
        if self.credential_binding not in {"user", "user+agent"}:
            raise ValueError("credential_binding must be user or user+agent for OAuth flows")

        if self.credential_binding == "user" and self.agent_id is not None:
            raise ValueError("agent_id must be omitted for user binding")
        if self.credential_binding == "user+agent" and self.agent_id is None:
            raise ValueError("agent_id is required for user+agent binding")
        return self


class UpstreamOAuthStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_url: str
    state_id: str
    expires_at: datetime


class UpstreamOAuthCallbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_id: str
    server_path: str
    provider: str


class UpstreamOAuthDisconnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_path: str
    credential_type: UpstreamOAuthCredentialType
    credential_binding: UpstreamCredentialBinding = Field(default="user")
    agent_id: Optional[str] = None
    provider: str

    @field_validator("server_path")
    @classmethod
    def _normalize_server_path(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("server_path must be a non-empty string")
        if not stripped.startswith("/"):
            stripped = f"/{stripped}"
        return stripped.rstrip("/") or "/"

    @field_validator("provider")
    @classmethod
    def _normalize_provider(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _validate_binding(self) -> "UpstreamOAuthDisconnectRequest":
        if self.credential_binding not in {"user", "user+agent"}:
            raise ValueError("credential_binding must be user or user+agent for OAuth flows")

        if self.credential_binding == "user" and self.agent_id is not None:
            raise ValueError("agent_id must be omitted for user binding")
        if self.credential_binding == "user+agent" and self.agent_id is None:
            raise ValueError("agent_id is required for user+agent binding")
        return self


class UpstreamOAuthDisconnectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revoked_count: int

