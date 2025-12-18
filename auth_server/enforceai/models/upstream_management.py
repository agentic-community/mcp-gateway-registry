from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .upstream_auth import (
    UpstreamAuthType,
    UpstreamCredentialBinding,
)
from .upstream_credentials import (
    UpstreamCredentialRecord,
)


class UpstreamCredentialCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential_type: UpstreamAuthType
    credential_binding: UpstreamCredentialBinding
    agent_id: Optional[str] = None
    provider: Optional[str] = None
    scopes: Optional[list[str]] = None
    token_type: Optional[str] = None
    expires_at: Optional[datetime] = None
    secret_payload: Optional[dict[str, Any]] = Field(
        default=None,
        repr=False,
        description="Secret payload to encrypt-at-rest; returned only on create.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "UpstreamCredentialCreateRequest":
        if self.credential_type in {"none"}:
            raise ValueError("credential_type=none is not valid for stored upstream credentials")
        if self.credential_type in {"mtls"}:
            raise ValueError("credential_type=mtls is not supported yet")

        if self.credential_type == "header-trust" and self.secret_payload is not None:
            raise ValueError("header-trust credentials must not include secret_payload")

        if self.credential_type == "provider-oauth" and (self.provider is None or not self.provider.strip()):
            raise ValueError("provider is required for provider-oauth credentials")

        if self.credential_type == "api-key" and self.provider is not None:
            raise ValueError("provider must be omitted for api-key credentials")

        if self.credential_type == "jwt" and self.provider is not None:
            raise ValueError("provider must be omitted for jwt credentials")

        if self.credential_binding in {"agent", "user+agent"} and self.agent_id is None:
            raise ValueError("agent_id is required for agent and user+agent bindings")
        if self.credential_binding in {"service", "user"} and self.agent_id is not None:
            raise ValueError("agent_id must be omitted for service and user bindings")

        return self


class UpstreamCredentialCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credential: UpstreamCredentialRecord
    secret_payload: Optional[dict[str, Any]] = Field(
        default=None,
        repr=False,
        description="Returned only at creation time; never returned by list endpoints.",
    )


class UpstreamCredentialRevokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = None


class UpstreamServerSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_path: str
    active_credential_count: int

