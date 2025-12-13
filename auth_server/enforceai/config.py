from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
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

    jwks_url: str = Field(
        ...,
        min_length=1,
        description="Issuer JWKS URL (e.g., https://issuer/.well-known/jwks.json)",
    )
    audience: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional audience claim requirement for this issuer",
    )
    algorithms: list[str] = Field(
        default_factory=lambda: ["RS256"],
        description="Accepted JWT algorithms for this issuer",
    )


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
