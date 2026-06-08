"""Pydantic request/response models for the duplicate-check API.

Shared by the per-entity ``/check-duplicates`` endpoints and by
``DuplicateCheckService``. Each entity type has its own
``*DuplicateCheckRequest`` so the OpenAPI surface mirrors the existing
``/register`` routes; all three endpoints return the same
``DuplicateCheckResult`` shape.

The service runs two checks: an *exact-match* check on the entity's
identity URL and a *similarity* check via the existing semantic
search pipeline. Both are advisory — the endpoint never returns 4xx
and the ``/register`` handlers never invoke the service. The
frontend chooses whether to surface results based on the
``DEDUP_REGISTRATION_HINT_ENABLED`` flag exposed via ``/api/config``.
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field

EntityType = Literal["mcp_server", "a2a_agent", "skill"]


class ExistingEntity(BaseModel):
    """Lightweight projection of an existing registry entity.

    Used for both the ``collision_with`` field (the entity that blocks
    registration on an exact-URL match) and the ``advisory_matches``
    list (similar entities surfaced by the similarity check).
    """

    entity_type: EntityType
    path: str
    name: str
    owner: str | None = None
    registered_at: str | None = None
    relevance_score: float | None = Field(
        default=None,
        description=(
            "Score from the semantic search backend; populated only for "
            "entries surfaced by the similarity check. Exact-URL "
            "collisions leave this as None."
        ),
    )
    match_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation of why this entity was surfaced "
            "(e.g. 'exact URL match', 'similar name and description')."
        ),
    )


class DuplicateCheckResult(BaseModel):
    """Response shape for ``/check-duplicates`` endpoints and the
    return type of ``DuplicateCheckService.check()``.

    Both the URL match and the similarity matches are advisory. The
    HTTP route layer always returns 200 with this envelope — there is
    no 4xx path. Frontend behavior is gated by
    ``DEDUP_REGISTRATION_HINT_ENABLED``, not by anything in this
    response.

    Both checks span all three entity types:
    - ``collision_with``: each repository is queried with its own
      normalization rule and the union is returned. A server
      registration that collides with an existing agent at the same
      URL surfaces here, tagged with ``entity_type="a2a_agent"``.
    - ``advisory_matches``: similarity hits from any of
      {server, agent, skill} can appear together. Each entry carries
      its own ``entity_type``.
    """

    collision_with: list[ExistingEntity] = Field(
        default_factory=list,
        description=(
            "Existing entities (across all entity types) whose "
            "identity URL matches the proposed registration. Each "
            "repository is queried with its own normalization rule; "
            "a hit in any repo populates this list with that repo's "
            "entity_type. Empty when no URL collision is found. "
            "Owner/path are redacted to blanks for callers who cannot "
            "view the colliding entity per the registry's visibility "
            "rules — the fact of a collision is still exposed, but no "
            "ownership details leak."
        ),
    )
    advisory_matches: list[ExistingEntity] = Field(
        default_factory=list,
        description=(
            "Best-effort list of semantically similar existing "
            "entities (across all entity types), filtered by "
            "visibility and capped by ``dedup_max_suggestions``. "
            "Distinct from ``collision_with``: these are similarity-"
            "based candidates, not exact-URL hits. Both lists can be "
            "populated simultaneously. The cap is global across "
            "entity types, not per-type — a registering server may "
            "see only similar skills if those skills score higher. "
            "The frontend should group results by entity_type when "
            "rendering."
        ),
    )
    threshold: float = Field(
        ...,
        description=(
            "Minimum semantic-search score required for a similarity-"
            "based advisory match (echoes settings)."
        ),
    )
    similarity_search_available: bool = Field(
        ...,
        description=(
            "True when the semantic search backend was reachable. "
            "False indicates the similarity check was skipped due to "
            "embedder unavailability; the exact-match check still ran. "
            "Frontends can use this to surface a degraded-state hint."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_collision(self) -> bool:
        """Convenience flag: True iff ``collision_with`` is non-empty.

        Pure derived state; the frontend can render the URL-match
        modal section based on this single field instead of
        inspecting list length.
        """
        return len(self.collision_with) > 0


class _DuplicateCheckRequestBase(BaseModel):
    """Common request fields shared by all three per-entity endpoints."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(
        default=None,
        max_length=10_000,
        description=(
            "Optional description used as part of the similarity "
            "search query. The combined name + description text is "
            "truncated to ~500 characters before being passed to the "
            "embedding backend; longer values are accepted but only "
            "the leading portion contributes to the similarity match."
        ),
    )
    self_path: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "When re-registering an existing entity, set this to the "
            "caller's own path so it is excluded from collision and "
            "advisory results."
        ),
    )


class ServerDuplicateCheckRequest(_DuplicateCheckRequestBase):
    """Request body for ``POST /api/servers/check-duplicates``."""

    proxy_pass_url: str | None = Field(
        default=None,
        max_length=2_048,
        description=(
            "The server's identity URL. The exact-match check uses "
            "the canonical (scheme-collapsed) form to detect URL "
            "collisions. Optional: when omitted, the exact-match "
            "check is skipped and only similarity-based advisory "
            "matches are returned."
        ),
    )


class AgentDuplicateCheckRequest(_DuplicateCheckRequestBase):
    """Request body for ``POST /api/agents/check-duplicates``."""

    url: str | None = Field(
        default=None,
        max_length=2_048,
        description=(
            "The agent endpoint URL (from the AgentCard). The "
            "exact-match check uses the canonical form to detect URL "
            "collisions."
        ),
    )


class SkillDuplicateCheckRequest(_DuplicateCheckRequestBase):
    """Request body for ``POST /api/skills/check-duplicates``."""

    skill_md_url: str | None = Field(
        default=None,
        max_length=2_048,
        description=(
            "The skill's identity URL — the link to the SKILL.md "
            "file in the source repository. The exact-match check "
            "uses the canonical GitHub form to detect URL collisions."
        ),
    )
