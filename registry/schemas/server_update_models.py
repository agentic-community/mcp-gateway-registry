"""Pydantic models for server metadata updates (PUT and PATCH)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .agent_models import AgentProvider

# Fields that callers must not mutate via PUT or PATCH.
# Server-managed (timestamps, health) or identity anchors.
SERVER_REGISTRANT_ONLY_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "path",
        "registered_by",
        "registered_at",
        "updated_at",
        "is_enabled",
        "is_active",
        "version",
        "deployment",
        "local_runtime",
        "health_status",
        "last_health_check",
        "auth_credential_encrypted",
        "custom_headers_encrypted",
        "sync_metadata",
        # Auth/credential fields, owned by the dedicated
        # PATCH /api/servers/{path}/auth-credential endpoint.
        # Rejected on this endpoint to keep credential mutation
        # behind a single, narrowly-scoped surface.
        "auth_scheme",
        "auth_credential",
        "auth_header_name",
        "custom_headers",
        "custom_header_names",
    }
)


_MAX_METADATA_BYTES: int = 64 * 1024
_MAX_TAGS: int = 50
_MAX_TAG_LEN: int = 64
_MAX_DESCRIPTION_LEN: int = 4096
_MAX_SERVER_NAME_LEN: int = 256


def _validate_tag_list(v: list[str] | str | None) -> list[str] | None:
    """Normalise CSV strings to lists, enforce count and per-tag length caps."""
    if v is None:
        return None
    if isinstance(v, str):
        v = [t.strip() for t in v.split(",") if t.strip()]
    if len(v) > _MAX_TAGS:
        raise ValueError(f"tag list must contain at most {_MAX_TAGS} entries")
    for tag in v:
        if len(tag) > _MAX_TAG_LEN:
            raise ValueError(f"each tag must be at most {_MAX_TAG_LEN} chars")
    return v


def _validate_metadata_size(v: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reject metadata blobs that serialise to more than _MAX_METADATA_BYTES."""
    if v is None:
        return None
    import json

    encoded = json.dumps(v, default=str).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError(
            f"metadata must serialise to at most {_MAX_METADATA_BYTES} bytes (got {len(encoded)})"
        )
    return v


class ServerUpdateRequest(BaseModel):
    """Full-replacement body for PUT /api/servers/{path}.

    Only mutable metadata fields are accepted. Identity anchors and
    server-managed fields must not appear; supplying them is a 422.

    Size caps (from LLD Should-fix #3):
        - server_name <= 256 chars
        - description <= 4096 chars
        - tags <= 50 entries, each <= 64 chars
        - external_tags <= 50 entries, each <= 64 chars
        - metadata <= 64 KB serialised JSON
        These caps protect downstream code (DocumentDB document size,
        embedding text budget, UI rendering). NOTE: POST /register
        does not currently enforce these caps; a follow-up issue
        will add them there for consistency.

    Visibility & access control:
        `visibility` and `allowed_groups` are deliberately mutable. The
        owner of a server can change these at any time (same authority
        they have at registration time). The handler enforces an
        ownership check before this body is applied, so non-owners
        cannot mutate them via this endpoint. See the LLD Threat
        Model section for the full reasoning.

    Auth/credential fields:
        `auth_scheme`, `auth_credential`, `auth_header_name`,
        `custom_headers`, and `custom_header_names` are intentionally
        absent. Use PATCH /api/servers/{path}/auth-credential instead.

    Local-deployment fields:
        `deployment` and `local_runtime` are in
        SERVER_REGISTRANT_ONLY_FIELDS and rejected with 422. To change
        a local server's launch recipe today, re-register with
        overwrite=true. A dedicated endpoint is planned for the future.
    """

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(..., min_length=1, max_length=_MAX_SERVER_NAME_LEN)
    description: str = Field(..., min_length=1, max_length=_MAX_DESCRIPTION_LEN)

    # Routing: remote-deployment only; server's existing deployment is
    # preserved by the handler. PUT cannot flip deployment type.
    proxy_pass_url: str | None = None
    mcp_endpoint: str | None = None
    sse_endpoint: str | None = None
    transport: str | None = None
    supported_transports: list[str] | None = None
    headers: list[dict[str, Any]] | None = None

    # Per-server Connect-config overrides (token-less IDE OAuth login + /mcp path).
    oauth_client_id: str | None = None
    append_mcp_path: bool | None = None

    # NOTE: Credential-shaped fields (auth_scheme, auth_credential,
    # auth_header_name, custom_headers) are intentionally absent.
    # They are owned by PATCH /api/servers/{path}/auth-credential.
    # Supplying them on this endpoint is rejected by Pydantic
    # extra="forbid" (returns 422).
    auth_provider: str | None = None  # provider name only; non-secret

    # Metadata
    tags: list[str] | str = Field(default_factory=list)
    license: str = "N/A"
    num_tools: int | None = None
    tool_list: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    visibility: str = "public"
    allowed_groups: list[str] | None = None
    status: str | None = None

    # Provider/lineage
    provider: AgentProvider | None = None  # structured; persisted via .model_dump()
    source_created_at: str | None = None
    source_updated_at: str | None = None
    external_tags: list[str] | None = None

    @field_validator("tags", mode="after")
    @classmethod
    def _cap_tags(
        cls,
        v: list[str] | str,
    ) -> list[str]:
        return _validate_tag_list(v) or []

    @field_validator("external_tags", mode="after")
    @classmethod
    def _cap_external_tags(
        cls,
        v: list[str] | None,
    ) -> list[str] | None:
        return _validate_tag_list(v)

    @field_validator("metadata", mode="after")
    @classmethod
    def _cap_metadata(
        cls,
        v: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return _validate_metadata_size(v)


class ServerCardPatch(BaseModel):
    """RFC 7396 JSON Merge Patch body for PATCH /api/servers/{path}.

    Every field is optional. Fields explicitly supplied override the
    stored value; fields absent are preserved. SERVER_REGISTRANT_ONLY_FIELDS
    are rejected with 422 if supplied.

    Visibility & access control:
        `visibility` and `allowed_groups` are mutable. The handler's
        ownership check ensures only the server's owner (or an admin)
        can apply this patch, so this is not a privilege-escalation
        path. See the LLD Threat Model section.

        Self-lockout: a non-admin owner can narrow `allowed_groups` in
        a way that removes their own access. This is not prevented;
        the audit trail makes it recoverable by an admin.

    Auth/credential fields:
        `auth_scheme`, `auth_credential`, `auth_header_name`,
        `custom_headers`, `custom_header_names` are intentionally
        absent and rejected by extra="forbid" with 422. Use
        PATCH /api/servers/{path}/auth-credential to mutate auth.

    Local-deployment fields:
        `deployment` and `local_runtime` are not patchable here. To
        change a local server's launch recipe today, re-register with
        overwrite=true. See LLD Non-Goals for rationale.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    server_name: str | None = Field(default=None, min_length=1, max_length=_MAX_SERVER_NAME_LEN)
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_LEN)
    proxy_pass_url: str | None = None
    mcp_endpoint: str | None = None
    sse_endpoint: str | None = None
    transport: str | None = None
    supported_transports: list[str] | None = None
    headers: list[dict[str, Any]] | None = None
    # Per-server Connect-config overrides (token-less IDE OAuth login + /mcp path).
    oauth_client_id: str | None = None
    append_mcp_path: bool | None = None
    # NOTE: Credential-shaped fields (auth_scheme, auth_credential,
    # auth_header_name, custom_headers, custom_header_names) are
    # intentionally absent and rejected by extra="forbid" with 422.
    # Use PATCH /api/servers/{path}/auth-credential to mutate auth.
    auth_provider: str | None = None  # provider name only; non-secret
    tags: list[str] | str | None = None
    license: str | None = None
    num_tools: int | None = None
    tool_list: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    visibility: str | None = None
    allowed_groups: list[str] | None = None
    status: str | None = None
    provider: AgentProvider | None = None  # structured; persisted via .model_dump()
    source_created_at: str | None = None
    source_updated_at: str | None = None
    external_tags: list[str] | None = None

    @model_validator(mode="after")
    def _reject_registrant_only(self) -> "ServerCardPatch":
        supplied = set(self.model_dump(exclude_unset=True, by_alias=False).keys())
        bad = SERVER_REGISTRANT_ONLY_FIELDS & supplied
        if bad:
            raise ValueError(f"Field(s) {sorted(bad)} are read-only and cannot be patched")
        return self

    @field_validator("tags", mode="after")
    @classmethod
    def _cap_tags(
        cls,
        v: list[str] | str | None,
    ) -> list[str] | None:
        return _validate_tag_list(v)

    @field_validator("external_tags", mode="after")
    @classmethod
    def _cap_external_tags(
        cls,
        v: list[str] | str | None,
    ) -> list[str] | None:
        return _validate_tag_list(v)

    @field_validator("metadata", mode="after")
    @classmethod
    def _cap_metadata(
        cls,
        v: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return _validate_metadata_size(v)
