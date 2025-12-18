from __future__ import annotations

import json
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

UpstreamAuthType = Literal[
    "none",
    "api-key",
    "oauth2",
    "oidc",
    "provider-oauth",
    "jwt",
    "mtls",
    "header-trust",
]

UpstreamCredentialBinding = Literal[
    "service",
    "user",
    "agent",
    "user+agent",
]

UpstreamInjectionKind = Literal[
    "header",
]


def _coerce_legacy_auth_type(
    value: Optional[str],
) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return normalized.replace("_", "-")


def _parse_headers_list(
    raw: object,
) -> list[dict[str, str]]:
    if raw is None:
        return []

    parsed: object = raw
    if isinstance(raw, str):
        parsed = json.loads(raw)

    if not isinstance(parsed, list):
        raise ValueError("headers must be a JSON list")

    headers: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("headers must be a list of objects")
        normalized: dict[str, str] = {}
        for key, value in item.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("headers must be a list of string->string objects")
            stripped_key = key.strip()
            if not stripped_key:
                raise ValueError("headers must not contain empty header names")
            normalized[stripped_key] = value
        headers.append(normalized)
    return headers


class UpstreamAuthInjection(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    kind: UpstreamInjectionKind = Field(
        default="header",
        description="Where/how to inject upstream auth for this server.",
    )
    header_name: str = Field(
        ...,
        min_length=1,
        description="Header name to inject into upstream requests (e.g., Authorization).",
    )
    scheme: Optional[str] = Field(
        default=None,
        description="Optional auth scheme/prefix (e.g., Bearer).",
    )

    @field_validator("header_name")
    @classmethod
    def _normalize_header_name(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("header_name must be a non-empty string")
        if any(ch in stripped for ch in ("\r", "\n")):
            raise ValueError("header_name must not contain newline characters")
        return stripped

    @field_validator("scheme")
    @classmethod
    def _normalize_scheme(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if any(ch in stripped for ch in ("\r", "\n")):
            raise ValueError("scheme must not contain newline characters")
        return stripped


class UpstreamAuthConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    type: UpstreamAuthType = Field(
        ...,
        description="Upstream authentication type required by the MCP server.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional upstream provider identifier (e.g., github, google).",
    )
    credential_binding: UpstreamCredentialBinding = Field(
        default="service",
        description="How upstream credentials are scoped for lookup.",
    )
    injection: Optional[UpstreamAuthInjection] = Field(
        default=None,
        description="Upstream injection behavior (header-based in Phase 1).",
    )

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

    @model_validator(mode="after")
    def _validate_phase_1_constraints(
        self,
    ) -> "UpstreamAuthConfig":
        if self.type == "mtls":
            raise ValueError("upstream_auth.type=mtls is not supported yet")
        return self


def normalize_upstream_auth(
    *,
    upstream_auth: Optional[object] = None,
    auth_type: Optional[str] = None,
    auth_provider: Optional[str] = None,
    headers: Optional[object] = None,
) -> UpstreamAuthConfig:
    """Normalize legacy server auth fields into a canonical upstream_auth config.

    Args:
        upstream_auth: Optional upstream_auth object or JSON string. If provided,
            it is validated and returned.
        auth_type: Legacy auth_type string (e.g., "api_key", "oauth", "none").
        auth_provider: Legacy auth_provider string.
        headers: Legacy headers list (typically a JSON string) used to infer injection.

    Returns:
        Normalized upstream_auth config.

    Raises:
        ValueError: If inputs are invalid or unsupported.
    """
    if upstream_auth is not None:
        raw = upstream_auth
        if isinstance(upstream_auth, str):
            raw = json.loads(upstream_auth)
        if not isinstance(raw, dict):
            raise ValueError("upstream_auth must be a JSON object")
        return UpstreamAuthConfig.model_validate(raw)

    coerced_type = _coerce_legacy_auth_type(auth_type) or "none"

    inferred_injection: Optional[UpstreamAuthInjection] = None
    headers_list = _parse_headers_list(headers)
    if headers_list:
        header_name = next(iter(headers_list[0].keys()))
        scheme: Optional[str] = None
        value = headers_list[0].get(header_name, "")
        if header_name.lower() == "authorization" and isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value.lower().startswith("bearer "):
                scheme = "Bearer"
        inferred_injection = UpstreamAuthInjection(
            header_name=header_name,
            scheme=scheme,
        )

    if coerced_type in {"none"}:
        return UpstreamAuthConfig(
            type="none",
            provider=None,
            credential_binding="service",
            injection=None,
        )

    if coerced_type in {"api-key"}:
        return UpstreamAuthConfig(
            type="api-key",
            provider=None,
            credential_binding="service",
            injection=inferred_injection
            or UpstreamAuthInjection(
                header_name="X-API-Key",
                scheme=None,
            ),
        )

    if coerced_type in {"oauth", "oauth2"}:
        return UpstreamAuthConfig(
            type="oauth2",
            provider=auth_provider,
            credential_binding="user",
            injection=inferred_injection
            or UpstreamAuthInjection(
                header_name="Authorization",
                scheme="Bearer",
            ),
        )

    if coerced_type in {"oidc"}:
        return UpstreamAuthConfig(
            type="oidc",
            provider=auth_provider,
            credential_binding="user",
            injection=inferred_injection
            or UpstreamAuthInjection(
                header_name="Authorization",
                scheme="Bearer",
            ),
        )

    if coerced_type in {"provider-oauth"}:
        return UpstreamAuthConfig(
            type="provider-oauth",
            provider=auth_provider,
            credential_binding="user",
            injection=inferred_injection
            or UpstreamAuthInjection(
                header_name="Authorization",
                scheme="Bearer",
            ),
        )

    if coerced_type in {"jwt"}:
        return UpstreamAuthConfig(
            type="jwt",
            provider=None,
            credential_binding="service",
            injection=inferred_injection
            or UpstreamAuthInjection(
                header_name="Authorization",
                scheme="Bearer",
            ),
        )

    if coerced_type in {"header-trust"}:
        return UpstreamAuthConfig(
            type="header-trust",
            provider=None,
            credential_binding="service",
            injection=None,
        )

    raise ValueError(f"Unsupported auth_type: {auth_type}")

