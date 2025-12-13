from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

DEFAULT_OIDC_SCOPE_CLAIMS: list[str] = [
    "scp",
    "scope",
    "permissions",
]
DEFAULT_OIDC_ROLE_CLAIMS: list[str] = [
    "roles",
    "groups",
    "permissions",
]
DEFAULT_JWKS_CACHE_TTL_SECONDS: int = 300
DEFAULT_OIDC_CLOCK_SKEW_SECONDS: int = 60


def _parse_json_mapping(
    env_var_name: str,
    raw: str,
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {env_var_name}: {exc.msg} (pos {exc.pos})"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{env_var_name} must be a JSON object (map)")

    return parsed


class OIDCIssuerConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    jwks_uri: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices(
            "jwks_uri",
            "jwks_url",
        ),
        description="Issuer JWKS URI (e.g., https://issuer/.well-known/jwks.json)",
    )
    audiences: list[str] = Field(
        ...,
        validation_alias=AliasChoices(
            "audiences",
            "audience",
        ),
        description="Allowed JWT audiences for this issuer",
    )
    algorithms: list[str] = Field(
        default_factory=lambda: ["RS256"],
        description="Accepted JWT algorithms for this issuer",
    )
    scope_claims: list[str] = Field(
        default_factory=lambda: list(DEFAULT_OIDC_SCOPE_CLAIMS),
        description="Claim precedence for scopes for this issuer",
    )
    role_claims: list[str] = Field(
        default_factory=lambda: list(DEFAULT_OIDC_ROLE_CLAIMS),
        description="Claim precedence for roles/groups (audit-only) for this issuer",
    )
    jwks_cache_ttl_seconds: int = Field(
        default=DEFAULT_JWKS_CACHE_TTL_SECONDS,
        ge=1,
        description="JWKS cache TTL in seconds (in-memory cache)",
    )
    clock_skew_seconds: int = Field(
        default=DEFAULT_OIDC_CLOCK_SKEW_SECONDS,
        ge=0,
        description="Clock skew tolerance in seconds for exp/iat validation",
    )

    @field_validator("jwks_uri")
    @classmethod
    def _validate_jwks_uri(
        cls,
        value: str,
    ) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("jwks_uri must be a non-empty string")

        parsed = urlparse(stripped)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError("jwks_uri must be an https:// or http:// URL")

        if not parsed.netloc:
            raise ValueError("jwks_uri must include a hostname")

        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("jwks_uri http:// is only allowed for localhost development")

        return stripped

    @field_validator("audiences", mode="before")
    @classmethod
    def _normalize_audiences(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("audiences")
    @classmethod
    def _validate_audiences(
        cls,
        value: list[str],
    ) -> list[str]:
        if not value:
            raise ValueError("audiences must contain at least one audience")

        normalized: list[str] = []
        for audience in value:
            stripped = audience.strip()
            if stripped:
                normalized.append(stripped)

        if not normalized:
            raise ValueError("audiences must contain at least one non-empty audience")

        return sorted(set(normalized))

    @field_validator("algorithms", mode="before")
    @classmethod
    def _normalize_algorithms(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("algorithms")
    @classmethod
    def _validate_algorithms(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized = [algorithm.strip() for algorithm in value if algorithm.strip()]
        if not normalized:
            raise ValueError("algorithms must contain at least one algorithm")
        return sorted(set(normalized))

    @field_validator("scope_claims", "role_claims", mode="before")
    @classmethod
    def _normalize_claim_lists(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("scope_claims", "role_claims")
    @classmethod
    def _validate_claim_lists(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("claim list must contain at least one entry")
        return normalized


class EnforceAISettings(BaseSettings):
    """Validated EnforceAI configuration sourced from environment variables.

    This module must not perform network operations or side effects; it only
    parses and validates configuration so later phases can rely on it.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
    )

    oidc_issuers: dict[str, OIDCIssuerConfig] = Field(
        ...,
        validation_alias="OIDC_ISSUERS",
    )
    db_path: Path = Field(
        ...,
        validation_alias="ENFORCEAI_DB_PATH",
    )

    gateway_private_key_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH",
            "GATEWAY_PRIVATE_KEY_PATH",
        ),
    )
    gateway_public_keys_dir: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR",
            "GATEWAY_PUBLIC_KEYS_DIR",
        ),
    )
    gateway_active_kid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_GATEWAY_ACTIVE_KID",
            "GATEWAY_ACTIVE_KID",
        ),
    )

    audit_retention_days: int = Field(
        default=30,
        ge=0,
        validation_alias="ENFORCEAI_AUDIT_RETENTION_DAYS",
    )
    audit_max_db_bytes: int = Field(
        default=500_000_000,
        ge=0,
        validation_alias="ENFORCEAI_AUDIT_MAX_DB_BYTES",
    )

    @field_validator("oidc_issuers", mode="before")
    @classmethod
    def _parse_oidc_issuers(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return _parse_json_mapping(
                "OIDC_ISSUERS",
                value,
            )
        return value

    @model_validator(mode="after")
    def _validate_settings(self) -> "EnforceAISettings":
        if not self.oidc_issuers:
            raise ValueError("OIDC_ISSUERS must contain at least one issuer")

        for issuer in self.oidc_issuers.keys():
            stripped = issuer.strip()
            if not stripped:
                raise ValueError("OIDC_ISSUERS keys must be non-empty strings")
            if stripped != issuer:
                raise ValueError("OIDC_ISSUERS keys must not contain surrounding whitespace")

        if (
            self.gateway_private_key_path is not None
            or self.gateway_public_keys_dir is not None
            or self.gateway_active_kid is not None
        ):
            missing: list[str] = []
            if self.gateway_private_key_path is None:
                missing.append("ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH")
            if self.gateway_public_keys_dir is None:
                missing.append("ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR")
            if self.gateway_active_kid is None:
                missing.append("GATEWAY_ACTIVE_KID")

            if missing:
                missing_display = ", ".join(missing)
                raise ValueError(
                    "Gateway token key configuration incomplete; "
                    f"missing {missing_display}"
                )

        return self
