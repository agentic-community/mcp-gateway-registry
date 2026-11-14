"""Models for OpenAPI MCP servers (issue #2022).

Three groups:
- Requests an admin sends (``*Request``, ``OpenApiTargetAuthInput``). The
  credential is a write-only ``SecretStr``.
- Stored records (``OpenApiServer``, ``OpenApiTarget``, ``StoredTargetAuth``).
  The credential is Fernet-encrypted.
- Responses (``*Response``). They never carry a credential, only
  ``credential_set``.
"""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)

from ..utils.header_validation import validate_header_name

OpenApiAuthScheme = Literal["none", "api_key_header", "api_key_query", "bearer", "basic"]
# shared: one admin-entered credential for every caller. per_user: each user
# submits their own key (my-credential routes), stored in the egress vault.
CredentialMode = Literal["shared", "per_user"]
PER_USER_SCHEMES: frozenset[str] = frozenset({"bearer", "api_key_header"})
PER_USER_PROVIDER_PREFIX: str = "openapi-"
OpenApiVisibility = Literal["public", "private", "group-restricted"]
SyncStatus = Literal["applied", "pending"]

SLUG_PATTERN: str = r"^[a-z0-9]+(-[a-z0-9]+)*$"
TARGET_NAME_PATTERN: str = r"^[a-z][a-z0-9]{1,23}$"
QUERY_PARAM_PATTERN: str = r"^[A-Za-z0-9_.\-]{1,128}$"
MAX_OVERRIDE_ENTRIES: int = 500
MAX_OVERRIDE_KEY_CHARS: int = 256
MAX_TOOL_NAME_OVERRIDE_CHARS: int = 64
MAX_DESCRIPTION_OVERRIDE_CHARS: int = 2000
FORBIDDEN_AUTH_HEADERS: frozenset[str] = frozenset(
    {"host", "cookie", "content-length", "transfer-encoding", "authorization", "connection"}
)


def _check_per_user_fields(
    auth: "OpenApiTargetAuthInput",
) -> None:
    """A per-user target names the header shape; each user brings the key."""
    if auth.scheme not in PER_USER_SCHEMES:
        raise ValueError(
            f"credential_mode 'per_user' supports {sorted(PER_USER_SCHEMES)}, not '{auth.scheme}'"
        )
    if auth.credential is not None:
        raise ValueError(
            "credential_mode 'per_user' takes no credential: each user submits their own key"
        )
    if auth.scheme == "api_key_header" and not auth.header_name:
        raise ValueError("auth scheme 'api_key_header' needs: header_name")
    if auth.query_param or auth.username:
        raise ValueError("credential_mode 'per_user' does not use query_param or username")


def _check_scheme_fields(
    auth: "OpenApiTargetAuthInput",
) -> None:
    """Each scheme needs exactly the fields it uses."""
    if auth.credential_mode == "per_user":
        _check_per_user_fields(auth)
        return
    required = {
        "none": set(),
        "api_key_header": {"header_name", "credential"},
        "api_key_query": {"query_param", "credential"},
        "bearer": {"credential"},
        "basic": {"username", "credential"},
    }[auth.scheme]
    present = {
        name
        for name in ("header_name", "query_param", "username", "credential")
        if getattr(auth, name)
    }
    missing = required - present
    extra = present - required
    if missing:
        raise ValueError(f"auth scheme '{auth.scheme}' needs: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"auth scheme '{auth.scheme}' does not use: {', '.join(sorted(extra))}")


class OpenApiTargetAuthInput(BaseModel):
    """Auth for one REST target, as the admin sends it. The credential is write-only."""

    scheme: OpenApiAuthScheme = "none"
    credential_mode: CredentialMode = "shared"
    header_name: str | None = Field(default=None, max_length=128)
    query_param: str | None = Field(default=None, pattern=QUERY_PARAM_PATTERN)
    username: str | None = Field(default=None, max_length=256)
    credential: SecretStr | None = Field(default=None)

    @field_validator("header_name")
    @classmethod
    def _valid_header_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        validate_header_name(value, context="auth.header_name")
        if value.lower() in FORBIDDEN_AUTH_HEADERS:
            raise ValueError(
                f"header '{value}' cannot carry an API key; use the bearer scheme for Authorization"
            )
        return value

    @field_validator("credential")
    @classmethod
    def _non_empty_credential(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret.strip():
            return None
        if any(ch in secret for ch in "\r\n\x00"):
            raise ValueError("credential must not contain line breaks or NUL")
        return value

    @model_validator(mode="after")
    def _scheme_fields(self) -> "OpenApiTargetAuthInput":
        _check_scheme_fields(self)
        return self


def _check_override_map(
    value: dict[str, str],
    max_value_chars: int,
) -> dict[str, str]:
    if len(value) > MAX_OVERRIDE_ENTRIES:
        raise ValueError(f"at most {MAX_OVERRIDE_ENTRIES} entries")
    for key, text in value.items():
        if not key or len(key) > MAX_OVERRIDE_KEY_CHARS:
            raise ValueError(f"keys must be 1 to {MAX_OVERRIDE_KEY_CHARS} characters")
        if not text or len(text) > max_value_chars:
            raise ValueError(f"values must be 1 to {max_value_chars} characters")
    return value


class _TargetToolOptions(BaseModel):
    """Fields that decide which tools a target produces and what they are called."""

    tool_name_overrides: dict[str, str] = Field(default_factory=dict)
    description_overrides: dict[str, str] = Field(default_factory=dict)
    include_tags: list[str] = Field(default_factory=list, max_length=100)
    include_operations: list[str] = Field(default_factory=list, max_length=500)
    keep_output_schemas: bool = False

    @field_validator("tool_name_overrides")
    @classmethod
    def _tool_names(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        return _check_override_map(value, MAX_TOOL_NAME_OVERRIDE_CHARS)

    @field_validator("description_overrides")
    @classmethod
    def _descriptions(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        return _check_override_map(value, MAX_DESCRIPTION_OVERRIDE_CHARS)


class OpenApiTargetCreateRequest(_TargetToolOptions):
    """Body of POST /api/openapi-servers/{slug}/targets."""

    name: str = Field(..., pattern=TARGET_NAME_PATTERN)
    description: str = Field(default="", max_length=1000)
    spec_url: HttpUrl
    base_url: HttpUrl
    auth: OpenApiTargetAuthInput = Field(default_factory=OpenApiTargetAuthInput)

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "petstore",
                "description": "Swagger Petstore sample API",
                "spec_url": "https://petstore3.swagger.io/api/v3/openapi.json",
                "base_url": "https://petstore3.swagger.io/api/v3",
                "auth": {
                    "scheme": "api_key_header",
                    "header_name": "api_key",
                    "credential": "YOUR_PETSTORE_KEY",
                },
                "include_tags": ["pet"],
                "include_operations": ["getInventory"],
            }
        }
    }


class OpenApiTargetUpdateRequest(BaseModel):
    """Body of PATCH /api/openapi-servers/{slug}/targets/{name}. A missing auth keeps the stored one."""

    description: str | None = Field(default=None, max_length=1000)
    base_url: HttpUrl | None = None
    auth: OpenApiTargetAuthInput | None = None
    tool_name_overrides: dict[str, str] | None = None
    description_overrides: dict[str, str] | None = None
    include_tags: list[str] | None = Field(default=None, max_length=100)
    include_operations: list[str] | None = Field(default=None, max_length=500)
    keep_output_schemas: bool | None = None
    refresh_spec: bool = False

    @field_validator("tool_name_overrides")
    @classmethod
    def _tool_names(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        return None if value is None else _check_override_map(value, MAX_TOOL_NAME_OVERRIDE_CHARS)

    @field_validator("description_overrides")
    @classmethod
    def _descriptions(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        return None if value is None else _check_override_map(value, MAX_DESCRIPTION_OVERRIDE_CHARS)


class OpenApiServerCreateRequest(BaseModel):
    """Body of POST /api/openapi-servers. The name becomes the path."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    visibility: OpenApiVisibility = "private"
    allowed_groups: list[str] = Field(default_factory=list, max_length=50)

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Weather APIs",
                "description": "Forecast and air quality from Open-Meteo",
                "tags": ["weather"],
                "visibility": "private",
            }
        }
    }

    @model_validator(mode="after")
    def _groups_for_restricted(self) -> "OpenApiServerCreateRequest":
        if self.visibility == "group-restricted" and not self.allowed_groups:
            raise ValueError("group-restricted visibility requires at least one allowed_group")
        return self


class OpenApiServerUpdateRequest(BaseModel):
    """Body of PATCH /api/openapi-servers/{slug}. The name and path cannot change."""

    description: str | None = Field(default=None, max_length=1000)
    tags: list[str] | None = Field(default=None, max_length=20)
    visibility: OpenApiVisibility | None = None
    allowed_groups: list[str] | None = Field(default=None, max_length=50)


class StoredTargetAuth(BaseModel):
    """Auth as stored. The credential is Fernet-encrypted (credential_encryption.py)."""

    scheme: OpenApiAuthScheme = "none"
    credential_mode: CredentialMode = "shared"
    header_name: str | None = None
    query_param: str | None = None
    username: str | None = None
    credential_encrypted: str | None = None
    credential_updated_at: datetime | None = None


class OpenApiTarget(_TargetToolOptions):
    """One REST target as stored inside its server document."""

    name: str
    description: str = ""
    spec_url: str
    spec_sha256: str
    spec_title: str = ""
    spec_version: str = ""
    openapi_version: str = ""
    base_url: str
    auth: StoredTargetAuth
    tool_names: list[str] = Field(default_factory=list)
    created_by: str
    created_at: datetime
    updated_at: datetime


class OpenApiServer(BaseModel):
    """One OpenAPI MCP server, stored in openapi_servers_<ns> with _id = slug."""

    slug: str = Field(..., pattern=SLUG_PATTERN)
    name: str
    description: str = ""
    path: str
    tags: list[str] = Field(default_factory=list)
    visibility: OpenApiVisibility = "private"
    allowed_groups: list[str] = Field(default_factory=list)
    generation: int = 0
    targets: list[OpenApiTarget] = Field(default_factory=list)
    created_by: str
    created_at: datetime
    updated_at: datetime


class StoredSpec(BaseModel):
    """One spec, stored in openapi_specs_<ns> with _id = SHA-256 of its bytes."""

    sha256: str
    content: str
    size_bytes: int
    content_type: str = ""
    fetched_at: datetime


class TargetAuthResponse(BaseModel):
    """Auth as returned. No credential, only whether one is set."""

    scheme: OpenApiAuthScheme
    credential_mode: CredentialMode = "shared"
    header_name: str | None = None
    query_param: str | None = None
    username: str | None = None
    credential_set: bool = False


class OpenApiTargetResponse(BaseModel):
    """One target as returned. spec_url and base_url are None for non-admins."""

    name: str
    description: str
    spec_url: str | None = None
    base_url: str | None = None
    spec_title: str
    spec_version: str
    openapi_version: str
    auth: TargetAuthResponse
    tool_names: list[str]
    tool_name_overrides: dict[str, str]
    description_overrides: dict[str, str]
    include_tags: list[str]
    include_operations: list[str]
    keep_output_schemas: bool
    status: Literal["ok", "error", "pending"] = "pending"
    error: str | None = None
    updated_at: datetime


class OpenApiServerResponse(BaseModel):
    """One server as returned."""

    slug: str
    name: str
    description: str
    path: str
    mcp_url: str
    tags: list[str]
    visibility: OpenApiVisibility
    allowed_groups: list[str]
    generation: int
    sync_status: SyncStatus | None = None
    targets: list[OpenApiTargetResponse]
    created_at: datetime
    updated_at: datetime


class AddTargetResponse(OpenApiTargetResponse):
    """Response to adding or updating a target."""

    warnings: list[str] = Field(default_factory=list)
    findings: list[dict] = Field(default_factory=list)
    generation: int
    sync_status: SyncStatus


class MyCredentialRequest(BaseModel):
    """Body of PUT .../targets/{name}/my-credential: the caller's own key for a per-user target."""

    secret: SecretStr
    ttl_value: int = Field(..., ge=1)
    ttl_unit: Literal["minutes", "hours", "days"]
    sub: str | None = Field(
        default=None, description="Admin only: store the key for this user instead"
    )
    auth_method: str | None = Field(
        default=None, description="Admin only, with sub: the user's ingress auth method"
    )

    @field_validator("secret")
    @classmethod
    def _usable_secret(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        secret = value.get_secret_value()
        if not secret.strip() or any(ch in secret for ch in "\r\n\x00"):
            raise ValueError("secret must be non-empty and contain no line breaks or NUL")
        return value


class MyCredentialStatus(BaseModel):
    """Whether the caller has a key for a per-user target. Never the key itself."""

    target: str
    configured: bool
    expires_at: str | None = None
