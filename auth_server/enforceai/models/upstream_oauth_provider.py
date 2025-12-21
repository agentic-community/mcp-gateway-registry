from __future__ import annotations

from datetime import datetime
import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_LOCALHOST_HOSTNAMES = {"localhost", "127.0.0.1"}


def _validate_provider_id(
    value: str,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("provider_id must be a non-empty string")
    if stripped != value:
        raise ValueError("provider_id must not include leading/trailing whitespace")
    if not _PROVIDER_ID_RE.fullmatch(stripped):
        raise ValueError(
            "provider_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
        )
    return stripped


def _validate_endpoint_url(
    value: str,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("endpoint must be a non-empty string")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("endpoint must be an https:// or http:// URL")
    if not parsed.netloc:
        raise ValueError("endpoint must include a hostname")
    if parsed.scheme == "http" and parsed.hostname not in _LOCALHOST_HOSTNAMES:
        raise ValueError("http:// endpoints are only allowed for localhost development")
    return stripped


class UpstreamOAuthProviderRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    provider_id: str = Field(..., min_length=1)
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    default_scopes: list[str] = Field(default_factory=list)
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("provider_id")
    @classmethod
    def _normalize_provider_id(
        cls,
        value: str,
    ) -> str:
        return _validate_provider_id(value)

    @field_validator("authorization_endpoint", "token_endpoint")
    @classmethod
    def _normalize_endpoints(
        cls,
        value: str,
    ) -> str:
        return _validate_endpoint_url(value)

    @field_validator("client_id")
    @classmethod
    def _normalize_client_id(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_id must be a non-empty string")
        return stripped

    @field_validator("default_scopes")
    @classmethod
    def _normalize_scopes(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        return sorted(set(normalized))

    @field_validator("extra_authorize_params")
    @classmethod
    def _normalize_authorize_params(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = raw_key.strip()
            val = raw_value.strip()
            if not key or not val:
                continue
            normalized[key] = val
        return normalized


class UpstreamOAuthProviderCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(..., min_length=1)
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    client_secret: str = Field(..., min_length=1, repr=False)
    default_scopes: list[str] = Field(default_factory=list)
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)

    @field_validator("provider_id")
    @classmethod
    def _normalize_provider_id(
        cls,
        value: str,
    ) -> str:
        return _validate_provider_id(value)

    @field_validator("authorization_endpoint", "token_endpoint")
    @classmethod
    def _normalize_endpoints(
        cls,
        value: str,
    ) -> str:
        return _validate_endpoint_url(value)

    @field_validator("client_id")
    @classmethod
    def _normalize_client_id(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_id must be a non-empty string")
        return stripped

    @field_validator("client_secret")
    @classmethod
    def _normalize_client_secret(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_secret must be a non-empty string")
        return stripped

    @field_validator("default_scopes")
    @classmethod
    def _normalize_scopes(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        return sorted(set(normalized))

    @field_validator("extra_authorize_params")
    @classmethod
    def _normalize_authorize_params(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = raw_key.strip()
            val = raw_value.strip()
            if not key or not val:
                continue
            normalized[key] = val
        return normalized


class UpstreamOAuthProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_endpoint: Optional[str] = None
    token_endpoint: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = Field(default=None, repr=False)
    default_scopes: Optional[list[str]] = None
    extra_authorize_params: Optional[dict[str, str]] = None

    @field_validator("authorization_endpoint", "token_endpoint")
    @classmethod
    def _normalize_endpoints(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        return _validate_endpoint_url(value)

    @field_validator("client_id")
    @classmethod
    def _normalize_client_id(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_id must be a non-empty string")
        return stripped

    @field_validator("client_secret")
    @classmethod
    def _normalize_client_secret(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_secret must be a non-empty string")
        return stripped

    @field_validator("default_scopes")
    @classmethod
    def _normalize_scopes(
        cls,
        value: Optional[list[str]],
    ) -> Optional[list[str]]:
        if value is None:
            return None
        normalized = [item.strip() for item in value if item.strip()]
        return sorted(set(normalized))

    @field_validator("extra_authorize_params")
    @classmethod
    def _normalize_authorize_params(
        cls,
        value: Optional[dict[str, str]],
    ) -> Optional[dict[str, str]]:
        if value is None:
            return None
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = raw_key.strip()
            val = raw_value.strip()
            if not key or not val:
                continue
            normalized[key] = val
        return normalized

    @model_validator(mode="after")
    def _validate_has_updates(
        self,
    ) -> "UpstreamOAuthProviderUpdate":
        if (
            self.authorization_endpoint is None
            and self.token_endpoint is None
            and self.client_id is None
            and self.client_secret is None
            and self.default_scopes is None
            and self.extra_authorize_params is None
        ):
            raise ValueError("At least one field must be provided for update")
        return self


class UpstreamOAuthProviderPublic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: UpstreamOAuthProviderRecord
    secret_present: bool
