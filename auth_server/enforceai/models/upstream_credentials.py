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
    field_validator,
    model_validator,
)

from .._validation import (
    _normalize_optional_non_empty_str_list,
    _validate_user_id,
    _validate_uuid4,
)
from .upstream_auth import (
    UpstreamAuthType,
    UpstreamCredentialBinding,
)


class UpstreamCredentialRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    credential_id: str = Field(..., min_length=1)
    server_path: str = Field(..., min_length=1)
    credential_type: UpstreamAuthType
    credential_binding: UpstreamCredentialBinding

    user_id: Optional[str] = None
    agent_id: Optional[str] = None

    provider: Optional[str] = None
    scopes: Optional[list[str]] = None
    token_type: Optional[str] = None

    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime

    @field_validator("server_path")
    @classmethod
    def _server_path_is_canonical(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("server_path must be a non-empty string")
        if not stripped.startswith("/"):
            raise ValueError("server_path must start with '/'")
        return stripped

    @field_validator("credential_type")
    @classmethod
    def _credential_type_is_supported_for_storage(
        cls,
        value: str,
    ) -> str:
        if value == "none":
            raise ValueError("credential_type=none is not a valid stored credential type")
        return value

    @field_validator("user_id")
    @classmethod
    def _user_id_is_canonical(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        return _validate_user_id(value)

    @field_validator("agent_id")
    @classmethod
    def _agent_id_is_uuid4(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        return _validate_uuid4(value, label="agent_id")

    @field_validator("provider")
    @classmethod
    def _normalize_provider(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("scopes")
    @classmethod
    def _scopes_are_non_empty_strings(
        cls,
        value: Optional[list[str]],
    ) -> Optional[list[str]]:
        return _normalize_optional_non_empty_str_list(value, label="scopes")

    @field_validator("token_type")
    @classmethod
    def _normalize_token_type(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_binding_fields(
        self,
    ) -> "UpstreamCredentialRecord":
        if self.credential_binding == "service":
            if self.user_id is not None or self.agent_id is not None:
                raise ValueError("service binding requires user_id and agent_id to be null")
            return self

        if self.credential_binding == "user":
            if self.user_id is None:
                raise ValueError("user binding requires user_id")
            if self.agent_id is not None:
                raise ValueError("user binding requires agent_id to be null")
            return self

        if self.credential_binding == "agent":
            if self.agent_id is None:
                raise ValueError("agent binding requires agent_id")
            if self.user_id is not None:
                raise ValueError("agent binding requires user_id to be null")
            return self

        if self.credential_binding == "user+agent":
            if self.user_id is None or self.agent_id is None:
                raise ValueError("user+agent binding requires both user_id and agent_id")
            return self

        raise ValueError(f"Unknown credential_binding: {self.credential_binding}")


class UpstreamCredentialSecret(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    credential_id: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(
        ...,
        repr=False,
        description="Decrypted secret payload; never log or return from list endpoints.",
    )

