"""
Data models for admin-defined custom catalog entity types.

A custom entity type is described at runtime by an admin via a
``CustomTypeDescriptor`` (a name + a list of typed field descriptors).
Records of that type are stored as ``CustomEntityRecord`` documents: a
uniform envelope (name/description/visibility/owner/tags/...) plus a
per-type ``attributes`` bag whose shape is validated against the
descriptor at write time.

These types are catalog-only — never proxied, executed, or health-checked.
"""

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# --- Resource bounds (no record cap by default, so bound the shape instead) ---
MAX_FIELDS_PER_TYPE: int = 100  # cap descriptor field count
MAX_ENUM_VALUES: int = 100  # cap enum option count
MAX_STRING_LEN: int = 1_000  # single-line string value
MAX_TEXT_LEN: int = 50_000  # textarea value; keeps doc well under DocumentDB 16MB
MAX_ARRAY_ITEMS: int = 200  # array<string> item count
# Admin-authored descriptor text (labels, display names, descriptions, enum
# options). Bounded so a single descriptor can't bloat the cached snapshot that
# every replica holds in memory.
MAX_LABEL_LEN: int = 200  # field label / enum option / display name
MAX_DESCRIPTION_LEN: int = 2_000  # type description

# Envelope keys a custom field MUST NOT shadow (promoted/managed by
# CustomEntityRecord), plus Mongo-interpreted keys that would collide at
# the storage layer.
RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "path",
        "entity_type",
        "visibility",
        "allowed_groups",
        "owner",
        "tags",
        "is_enabled",
        "created_at",
        "updated_at",
        "attributes",
        "_id",
        "_identity_url_normalized",
    }
)

# Type names reserved by the three default entity types and related concepts.
RESERVED_TYPE_NAMES: frozenset[str] = frozenset(
    {"mcp_server", "a2a_agent", "skill", "virtual_server", "tool"}
)

# URL-safe type name pattern (path prefix + entity_type discriminator).
TYPE_NAME_PATTERN: str = r"^[a-z0-9_-]+$"


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class CustomFieldType(StrEnum):
    """Allowed datatypes for a custom-type field (v1: scalars + scalar arrays)."""

    STRING = "string"  # single-line text input
    TEXT = "text"  # textarea (rendered as PLAIN TEXT in v1)
    NUMBER = "number"  # int + float
    BOOL = "bool"  # checkbox
    ENUM = "enum"  # select; requires enum_values
    DATE = "date"  # date-only, ISO-8601 calendar date "YYYY-MM-DD"
    ARRAY_STRING = "array<string>"  # tag/chip input


class CustomFieldDescriptor(BaseModel):
    """One field in a custom type's schema."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Field key stored inside the record's attributes bag",
    )
    label: str | None = Field(
        default=None,
        max_length=MAX_LABEL_LEN,
        description="Optional display label; omit to fall back to humanized name",
    )
    datatype: CustomFieldType
    enum_values: list[str] | None = Field(
        default=None,
        max_length=MAX_ENUM_VALUES,
        description="Allowed values; REQUIRED iff datatype == enum",
    )
    required: bool = False
    semantic: bool = Field(
        default=False,
        description="Include this field's value in the search embedding text",
    )
    show_in_list: bool = Field(
        default=False,
        description="Render this field on the list/card view",
    )

    @field_validator("name")
    @classmethod
    def _name_is_attribute_safe(
        cls,
        v: str,
    ) -> str:
        """Reject names that aren't attribute-safe or collide with envelope keys."""
        if not v.replace("_", "").isalnum():
            raise ValueError("field name must be alphanumeric/underscore")
        if v in RESERVED_FIELD_NAMES:
            raise ValueError(f"field name '{v}' is reserved (collides with an envelope key)")
        return v

    @model_validator(mode="after")
    def _enum_values_consistency(self) -> "CustomFieldDescriptor":
        """Enforce that enum_values is present iff datatype is enum, and bound item length."""
        if self.datatype == CustomFieldType.ENUM and not self.enum_values:
            raise ValueError(f"field '{self.name}': enum datatype requires non-empty enum_values")
        if self.datatype != CustomFieldType.ENUM and self.enum_values:
            raise ValueError(f"field '{self.name}': enum_values only valid for enum datatype")
        # Per-item length cap (the field's max_length only bounds the list count).
        for opt in self.enum_values or []:
            if len(opt) > MAX_LABEL_LEN:
                raise ValueError(
                    f"field '{self.name}': enum value exceeds {MAX_LABEL_LEN} characters"
                )
        return self


class CustomTypeDescriptor(BaseModel):
    """Admin-authored schema for a custom entity type. Stored in mcp_custom_types.

    ``extra="ignore"``: the type repo splats raw Mongo docs
    (``CustomTypeDescriptor(**doc)``) which carry an ``_id`` key, so the
    model must tolerate the extra key.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=TYPE_NAME_PATTERN,
        description="URL-safe, IMMUTABLE type name; used as path prefix and entity_type",
    )
    display_name: str | None = Field(
        default=None,
        max_length=MAX_LABEL_LEN,
        description="Optional human label for the tab",
    )
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    fields: list[CustomFieldDescriptor] = Field(
        ...,
        min_length=1,
        max_length=MAX_FIELDS_PER_TYPE,
    )
    schema_version: int = Field(
        default=1, description="Descriptor schema version for future compat"
    )
    created_by: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("name")
    @classmethod
    def _name_not_reserved(
        cls,
        v: str,
    ) -> str:
        """Prevent collision with the three default entity types and reserved words."""
        if v in RESERVED_TYPE_NAMES:
            raise ValueError(f"'{v}' is a reserved entity type name")
        return v

    @field_validator("fields")
    @classmethod
    def _unique_field_names(
        cls,
        v: list[CustomFieldDescriptor],
    ) -> list[CustomFieldDescriptor]:
        """Reject descriptors with duplicate field names."""
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names in descriptor")
        return v


class CustomTypeUpdate(BaseModel):
    """Partial update for a custom type's MUTABLE metadata only.

    The type ``name`` (the URL-safe identifier, path prefix, and entity_type
    discriminator) and the ``fields`` schema are IMMUTABLE in v1 -- changing
    them would orphan existing records and embeddings. Only the human-facing
    ``display_name`` and ``description`` can be edited. A field left None is
    not touched (no-op); an explicit value (including empty string) overwrites.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)


class CustomEntityRecord(BaseModel):
    """A record of a custom type. Stored in mcp_custom_entities (envelope + attributes).

    ``extra="ignore"`` so a raw Mongo doc (which carries an ``_id`` key) can be
    splatted directly via ``CustomEntityRecord(**doc)`` without raising. The
    repository ALSO pops ``_id`` before construction; both are kept for defense
    in depth.
    """

    model_config = ConfigDict(extra="ignore")

    # --- envelope (uniform across all custom types) ---
    path: str = Field(default="", description="Synthetic /{type}/{uuid}; becomes _id")
    entity_type: str = Field(..., description="The custom type name (discriminator)")
    name: str = Field(..., min_length=1, description="Display name (promoted out of attributes)")
    description: str | None = Field(
        default=None,
        max_length=MAX_TEXT_LEN,
        description="Optional free-text blurb; promoted to envelope",
    )
    visibility: str = Field(default="private")
    allowed_groups: list[str] = Field(default_factory=list)
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    is_enabled: bool = True
    num_stars: float = Field(default=0.0, ge=0.0, le=5.0, description="Average user rating")
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-user ratings ([{user, rating}], rotating buffer)",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # --- payload (per-type user-defined fields, validated against descriptor) ---
    attributes: dict[str, Any] = Field(default_factory=dict)

    def assign_path(self) -> None:
        """Generate the synthetic /{type}/{uuid} path if not already set."""
        if not self.path:
            self.path = f"/{self.entity_type}/{uuid4()}"


class CustomEntityCreate(BaseModel):
    """Client payload for POST /api/custom/{type}. No owner/path/entity_type."""

    name: str = Field(..., min_length=1, max_length=MAX_STRING_LEN)
    description: str | None = Field(default=None, max_length=MAX_TEXT_LEN)
    visibility: str = Field(default="private")
    allowed_groups: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, max_length=MAX_ARRAY_ITEMS)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(
        cls,
        v: str,
    ) -> str:
        """Normalize/validate against the canonical visibility set (single source)."""
        from registry.utils.visibility import validate_visibility

        return validate_visibility(v)

    @model_validator(mode="after")
    def _group_restricted_needs_groups(self) -> "CustomEntityCreate":
        """Reject group-restricted visibility with no allowed_groups (undefined access)."""
        if self.visibility == "group-restricted" and not self.allowed_groups:
            raise ValueError("group-restricted visibility requires at least one allowed_group")
        return self


class CustomEntityUpdate(BaseModel):
    """Client payload for PUT /api/custom/{type}/{uuid}.

    Only these envelope fields are mutable; everything else is server-managed.
    All fields optional (None = leave unchanged). The group-restricted invariant
    is checked against the MERGED state in the service layer, because a PUT may
    set only visibility or only allowed_groups, not both.
    """

    name: str | None = Field(default=None, min_length=1, max_length=MAX_STRING_LEN)
    description: str | None = Field(default=None, max_length=MAX_TEXT_LEN)
    visibility: str | None = None
    allowed_groups: list[str] | None = None
    tags: list[str] | None = Field(default=None, max_length=MAX_ARRAY_ITEMS)
    attributes: dict[str, Any] | None = None

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(
        cls,
        v: str | None,
    ) -> str | None:
        """Normalize/validate visibility when present; None leaves it unchanged."""
        if v is None:
            return None
        from registry.utils.visibility import validate_visibility

        return validate_visibility(v)
