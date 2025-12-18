from __future__ import annotations

import json
import os as _os
import sys
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import urlparse
from typing import (
    Any,
    Literal,
    Optional,
)

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

from .secrets.upstream_kek import (
    load_upstream_kek,
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
DEFAULT_UPSTREAM_OAUTH_STATE_TTL_SECONDS: int = 10 * 60
DEFAULT_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS: int = 60

AuthProviderMode = Literal[
    "oidc",
    "api-key",
    "gateway-token",
    "mixed",
]


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


class UpstreamOAuthClientSecretRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["env", "file"]
    env_var: Optional[str] = None
    path: Optional[Path] = None

    @field_validator("env_var")
    @classmethod
    def _normalize_env_var(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_ref(self) -> "UpstreamOAuthClientSecretRef":
        if self.kind == "env":
            if self.env_var is None:
                raise ValueError("env client_secret_ref requires env_var")
            if self.path is not None:
                raise ValueError("env client_secret_ref must not include path")
            return self

        if self.kind == "file":
            if self.path is None:
                raise ValueError("file client_secret_ref requires path")
            if self.env_var is not None:
                raise ValueError("file client_secret_ref must not include env_var")
            return self

        raise ValueError("Unsupported client_secret_ref.kind")


class UpstreamOAuthProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    client_secret_ref: UpstreamOAuthClientSecretRef
    default_scopes: list[str] = Field(default_factory=list)
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)

    @field_validator("authorization_endpoint", "token_endpoint")
    @classmethod
    def _validate_endpoint_url(
        cls,
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
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("http:// endpoints are only allowed for localhost development")
        return stripped

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


class EnforceAISettings(BaseSettings):
    """Validated EnforceAI configuration sourced from environment variables.

    This module must not perform network operations or side effects; it only
    parses and validates configuration so later phases can rely on it.
    """

    model_config = SettingsConfigDict(
        env_file=None if "pytest" in sys.modules else ".env",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
    )

    auth_provider: AuthProviderMode = Field(
        default="oidc",
        validation_alias=AliasChoices(
            "ENFORCEAI_AUTH_PROVIDER",
            "AUTH_PROVIDER",
        ),
    )
    oidc_issuers: dict[str, OIDCIssuerConfig] = Field(
        default_factory=dict,
        validation_alias="OIDC_ISSUERS",
    )
    db_path: Path = Field(
        ...,
        validation_alias="ENFORCEAI_DB_PATH",
    )

    upstream_oauth_providers: dict[str, "UpstreamOAuthProviderConfig"] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS",
            "UPSTREAM_OAUTH_PROVIDERS",
        ),
        description="Upstream OAuth provider config for gateway-terminated upstream auth.",
    )
    upstream_oauth_state_ttl_seconds: int = Field(
        default=DEFAULT_UPSTREAM_OAUTH_STATE_TTL_SECONDS,
        ge=30,
        validation_alias="ENFORCEAI_UPSTREAM_OAUTH_STATE_TTL_SECONDS",
    )
    upstream_oauth_refresh_skew_seconds: int = Field(
        default=DEFAULT_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS,
        ge=0,
        validation_alias="ENFORCEAI_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS",
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

    gateway_issuer: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_GATEWAY_ISSUER",
            "GATEWAY_ISSUER",
        ),
    )

    api_key_pepper_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_API_KEY_PEPPER_PATH",
            "API_KEY_PEPPER_PATH",
        ),
    )

    upstream_kek_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_UPSTREAM_KEK_PATH",
            "UPSTREAM_KEK_PATH",
        ),
        description="Path to a hex-encoded 32-byte KEK used for upstream secret encryption-at-rest.",
    )

    scopes_catalog_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ENFORCEAI_SCOPES_CATALOG_PATH",
            "SCOPES_CATALOG_PATH",
        ),
        description="Path to scopes.yml catalog used for FGAC enforcement",
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

    @field_validator("auth_provider", mode="before")
    @classmethod
    def _normalize_auth_provider(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"keycloak", "cognito"}:
                return "oidc"
            return normalized
        return value

    @field_validator("oidc_issuers", mode="before")
    @classmethod
    def _parse_oidc_issuers(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            return _parse_json_mapping(
                "OIDC_ISSUERS",
                value,
            )
        return value

    @field_validator("upstream_oauth_providers", mode="before")
    @classmethod
    def _parse_upstream_oauth_providers(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            return _parse_json_mapping(
                "UPSTREAM_OAUTH_PROVIDERS",
                value,
            )
        return value

    @field_validator("gateway_issuer")
    @classmethod
    def _normalize_gateway_issuer(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Gateway issuer must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _validate_settings(self) -> "EnforceAISettings":
        if self.auth_provider == "oidc" and not self.oidc_issuers:
            raise ValueError(
                "OIDC_ISSUERS must contain at least one issuer "
                "when AUTH_PROVIDER is oidc"
            )

        for issuer in self.oidc_issuers.keys():
            stripped = issuer.strip()
            if not stripped:
                raise ValueError("OIDC_ISSUERS keys must be non-empty strings")
            if stripped != issuer:
                raise ValueError("OIDC_ISSUERS keys must not contain surrounding whitespace")

        gateway_key_config_present = (
            self.gateway_private_key_path is not None
            or self.gateway_public_keys_dir is not None
            or self.gateway_active_kid is not None
        )
        if self.auth_provider in {"gateway-token", "mixed"} or gateway_key_config_present:
            missing: list[str] = []
            if self.gateway_private_key_path is None:
                missing.append("ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH")
            if self.gateway_public_keys_dir is None:
                missing.append("ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR")
            if self.gateway_active_kid is None:
                missing.append("GATEWAY_ACTIVE_KID")
            if self.gateway_issuer is None:
                missing.append("ENFORCEAI_GATEWAY_ISSUER")

            if missing:
                missing_display = ", ".join(missing)
                raise ValueError(
                    "Gateway token configuration incomplete; "
                    f"missing {missing_display}"
                )

        if self.gateway_issuer is not None:
            if not self.gateway_issuer:
                raise ValueError("Gateway issuer must be a non-empty string")

        if self.auth_provider in {"api-key", "mixed"} and self.api_key_pepper_path is None:
            raise ValueError(
                "ENFORCEAI_API_KEY_PEPPER_PATH is required when AUTH_PROVIDER is api-key or mixed"
            )

        if self.api_key_pepper_path is not None:
            if not self.api_key_pepper_path.exists():
                raise ValueError("API key pepper file does not exist")
            if not self.api_key_pepper_path.is_file():
                raise ValueError("API key pepper path must be a file")
            try:
                pepper_bytes = self.api_key_pepper_path.read_bytes()
            except OSError as exc:
                raise ValueError("API key pepper file is not readable") from exc
            if not pepper_bytes.strip():
                raise ValueError("API key pepper file is empty")

        if self.scopes_catalog_path is not None:
            if not self.scopes_catalog_path.exists():
                raise ValueError("Scopes catalog file does not exist")
            if not self.scopes_catalog_path.is_file():
                raise ValueError("Scopes catalog path must be a file")
            try:
                catalog_bytes = self.scopes_catalog_path.read_bytes()
            except OSError as exc:
                raise ValueError("Scopes catalog file is not readable") from exc
            if not catalog_bytes.strip():
                raise ValueError("Scopes catalog file is empty")

        if self.upstream_kek_path is not None:
            if not self.upstream_kek_path.exists():
                raise ValueError("Upstream KEK file does not exist")
            if not self.upstream_kek_path.is_file():
                raise ValueError("Upstream KEK path must be a file")
            try:
                load_upstream_kek(self.upstream_kek_path)
            except ValueError as exc:
                raise ValueError("Invalid upstream KEK file") from exc

        for provider_id, provider in self.upstream_oauth_providers.items():
            normalized = provider_id.strip()
            if not normalized:
                raise ValueError("UPSTREAM_OAUTH_PROVIDERS keys must be non-empty strings")
            if normalized != provider_id:
                raise ValueError("UPSTREAM_OAUTH_PROVIDERS keys must not include whitespace")

            if provider.client_secret_ref.kind == "env":
                env_name = provider.client_secret_ref.env_var
                if env_name is None:
                    raise ValueError(
                        f"UPSTREAM_OAUTH_PROVIDERS.{provider_id}.client_secret_ref.env_var is required"
                    )
                secret_value = _os.environ.get(env_name)
                if secret_value is None or not secret_value.strip():
                    raise ValueError(
                        f"Missing upstream OAuth client secret env var for provider '{provider_id}': {env_name}"
                    )
            elif provider.client_secret_ref.kind == "file":
                secret_path = provider.client_secret_ref.path
                if secret_path is None:
                    raise ValueError(
                        f"UPSTREAM_OAUTH_PROVIDERS.{provider_id}.client_secret_ref.path is required"
                    )
                if not secret_path.exists() or not secret_path.is_file():
                    raise ValueError(
                        f"Upstream OAuth client secret file missing for provider '{provider_id}': {secret_path}"
                    )
                try:
                    secret_bytes = secret_path.read_bytes()
                except OSError as exc:
                    raise ValueError(
                        f"Upstream OAuth client secret file is not readable for provider '{provider_id}'"
                    ) from exc
                if not secret_bytes.strip():
                    raise ValueError(
                        f"Upstream OAuth client secret file is empty for provider '{provider_id}'"
                    )
            else:
                raise ValueError(
                    f"Unsupported upstream OAuth client_secret_ref.kind for provider '{provider_id}'"
                )

        return self
