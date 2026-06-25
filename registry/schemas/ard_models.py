"""Pydantic models for the Agentic Resource Discovery (ARD) catalog.

These mirror the ARD ai-catalog.schema.json (spec v1.0) exactly. Field names use
the spec's camelCase via aliases, and we serialize with ``by_alias=True,
exclude_none=True`` so optional empty fields are omitted. This matters for two
schema constraints:

1. ``host`` uses ``additionalProperties: false`` (only the named keys allowed).
2. Each entry must carry exactly one of ``url`` / ``data`` (oneOf).

See issue #1294 and the spec reference under
``.scratchpad/issue-1294/ard-spec-ref/``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArdTrustManifest(BaseModel):
    """Minimal Phase 1 trust manifest (identity only, no signature)."""

    model_config = ConfigDict(populate_by_name=True)

    identity: str = Field(
        ...,
        description="Publisher FQDN or DID (e.g. https://registry.example.com)",
    )
    identity_type: str = Field(
        default="https",
        alias="identityType",
        description="One of: spiffe | did | https | other",
    )


class ArdCatalogEntry(BaseModel):
    """A single ARD catalog entry.

    Phase 1 always populates ``url`` (never ``data``) so the oneOf constraint is
    satisfied by construction.
    """

    model_config = ConfigDict(populate_by_name=True)

    identifier: str = Field(
        ...,
        description="URN of the form urn:air:<publisher-fqdn>:<namespace>:<name>",
    )
    display_name: str = Field(
        ...,
        alias="displayName",
        description="Human-friendly display name",
    )
    type: str = Field(
        ...,
        description="IANA media type for the artifact wrapper",
    )
    url: str | None = Field(
        default=None,
        description="HTTP URL to retrieve the full artifact record",
    )
    description: str | None = Field(
        default=None,
        description="Brief natural-language description of the capability",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags or keywords for basic categorization and filtering",
    )
    capabilities: list[str] | None = Field(
        default=None,
        description="Explicit indexing tags representing tools, skills, or functions",
    )
    representative_queries: list[str] | None = Field(
        default=None,
        alias="representativeQueries",
        description="Representative natural-language search queries (2-5 items)",
    )
    version: str | None = Field(
        default=None,
        description="Semantic version of the capability",
    )
    updated_at: str | None = Field(
        default=None,
        alias="updatedAt",
        description="ISO 8601 timestamp of the last modification",
    )


class ArdHost(BaseModel):
    """Catalog publisher info. Schema sets additionalProperties:false."""

    model_config = ConfigDict(populate_by_name=True)

    display_name: str = Field(
        ...,
        alias="displayName",
        description="Human-readable name of the host",
    )
    identifier: str | None = Field(
        default=None,
        description="Verifiable identifier of the host (e.g. did:web:domain.com)",
    )
    documentation_url: str | None = Field(
        default=None,
        alias="documentationUrl",
        description="URL to publisher documentation",
    )
    trust_manifest: ArdTrustManifest | None = Field(
        default=None,
        alias="trustManifest",
        description="Minimal trust envelope for the publisher",
    )


class AICatalogManifest(BaseModel):
    """Root ai-catalog.json document."""

    model_config = ConfigDict(populate_by_name=True)

    spec_version: str = Field(
        default="1.0",
        alias="specVersion",
        description="Version of the ai-catalog specification",
    )
    host: ArdHost
    entries: list[ArdCatalogEntry] = Field(default_factory=list)


# =============================================================================
# Phase 2: ARD Registry adapter (POST /search, GET /agents) — issue #1295
# Request models use extra="forbid" to honor the ARD additionalProperties:false.
# =============================================================================


class ArdQuery(BaseModel):
    """ARD SearchRequest.query."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural-language query (required).",
    )
    filter: dict[str, str | list[str]] | None = Field(
        default=None,
        description=(
            "Dot-path field -> string|array filter. Values OR within a key, "
            "AND across keys. Supported keys: 'type'/'entity_type' and 'tags'."
        ),
    )


class ArdSearchRequest(BaseModel):
    """ARD SearchRequest. additionalProperties: false."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: ArdQuery
    federation: Literal["auto", "referrals", "none"] = Field(
        default="auto",
        description="Federation mode. Phase 2: own-index results for all modes.",
    )
    page_size: int = Field(
        default=10,
        ge=1,
        le=100,
        alias="pageSize",
        description="Maximum results in the page.",
    )
    page_token: str | None = Field(
        default=None,
        alias="pageToken",
        description="Opaque pagination cursor from a previous response.",
    )


class ArdSearchResult(ArdCatalogEntry):
    """A full catalogEntry plus search annotations (ARD search result item)."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Relevance 0-100 (rescaled from the internal 0-1 relevance).",
    )
    source: str = Field(
        ...,
        description="URI of the search endpoint that produced this result.",
    )


class ArdReferral(BaseModel):
    """Pointer to another registry (application/ai-registry+json). Empty in Phase 2."""

    model_config = ConfigDict(populate_by_name=True)

    identifier: str
    type: str = "application/ai-registry+json"
    url: str


class ArdSearchResponse(BaseModel):
    """ARD SearchResponse."""

    model_config = ConfigDict(populate_by_name=True)

    results: list[ArdSearchResult] = Field(default_factory=list)
    referrals: list[ArdReferral] = Field(default_factory=list)
    page_token: str | None = Field(default=None, alias="pageToken")


class ArdListResponse(BaseModel):
    """ARD ListResponse for GET /agents (browse)."""

    model_config = ConfigDict(populate_by_name=True)

    items: list[ArdCatalogEntry] = Field(default_factory=list)
    total: int = 0
    page_token: str | None = Field(default=None, alias="pageToken")


class ArdError(BaseModel):
    """ARD error envelope ({errorCode, message})."""

    model_config = ConfigDict(populate_by_name=True)

    error_code: str = Field(..., alias="errorCode")
    message: str
