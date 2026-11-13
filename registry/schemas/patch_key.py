"""Per-user long-lived API key ("patch key") schemas for MongoDB storage.

A patch key is a non-expiring Bearer credential that a logged-in (console
session) user mints for machine clients. The plaintext key is shown exactly
once in the mint response; the database stores only a SHA-256 hash plus
metadata. The auth server's /validate resolves an ``wgk-`` prefixed Bearer
token by hash and continues the request as the owning user (same
groups/scope chain the user gets with a JWT).

The stored snapshot of the owner's groups is taken at mint time: revoking the
owner's session does not revoke the key (use the revoke endpoint), but scope
mappings are resolved per request so scope changes take effect immediately.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# Key lifecycle states. Revocation is one-way: there is no transition back to
# active (the API exposes no un-revoke).
PATCH_KEY_STATUS_ACTIVE: str = "active"
PATCH_KEY_STATUS_REVOKED: str = "revoked"

_KEY_NAME_PATTERN_MAX: int = 128


class PatchKeyCreate(BaseModel):
    """Request body for POST /api/patch-keys."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=_KEY_NAME_PATTERN_MAX,
        description="Human-readable label for the key (e.g. 'ci-runner')",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class PatchKeyInfo(BaseModel):
    """Metadata view of a patch key. Never contains the plaintext or its hash."""

    key_id: str = Field(..., description="Server-generated opaque key identifier")
    name: str = Field(..., description="Human-readable label chosen at mint time")
    key_prefix: str = Field(
        ...,
        description=(
            "First characters of the plaintext key (e.g. 'wgk-ab12cd34'), safe "
            "for display/correlation; not sufficient to reconstruct the key"
        ),
    )
    username: str = Field(..., description="Owning user (login username / OIDC sub)")
    email: str | None = Field(None, description="Owning user's email captured at mint time")
    provider: str | None = Field(
        None,
        description=(
            "Auth method / IdP the owner's console session used at mint time "
            "(audit labeling only; not used for authorization)"
        ),
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Snapshot of the owner's groups taken at mint time",
    )
    status: str = Field(
        ..., description=f"'{PATCH_KEY_STATUS_ACTIVE}' or '{PATCH_KEY_STATUS_REVOKED}'"
    )
    created_at: datetime = Field(..., description="When the key was minted")
    last_used_at: datetime | None = Field(
        None, description="When the key last authenticated a request (None if never used)"
    )
    revoked_at: datetime | None = Field(
        None, description="When the key was revoked (None while active)"
    )


class PatchKeyCreated(BaseModel):
    """Response for POST /api/patch-keys.

    ``key`` is the plaintext Bearer value. It is returned exactly once and is
    not recoverable afterwards; the database stores only its SHA-256 hash.
    """

    key: str = Field(..., description="Plaintext API key (shown once, store it now)")
    info: PatchKeyInfo = Field(..., description="Persisted key metadata")


class PatchKeyListResponse(BaseModel):
    """Response envelope for GET /api/patch-keys (caller's keys only)."""

    total: int = Field(..., description="Total number of the caller's keys")
    items: list[PatchKeyInfo] = Field(default_factory=list)
