"""Shared duplicate-check service for registration deduplication.

Powers the three per-entity ``/check-duplicates`` endpoints (servers,
agents, skills). Each call runs two checks and returns both result
sets in a single envelope:

1. **Exact-match check**: the entity's *identity URL*
   (``proxy_pass_url`` / agent ``url`` / ``skill_md_url``) is
   normalized per each target repository's rule and looked up in all
   three repositories. With the sidecar field + sparse index, each
   lookup is a single indexed ``$eq`` query — microseconds — so
   scanning all three repos is essentially free. A hit in any repo
   populates ``collision_with`` so the user is alerted even when the
   colliding entity is a different type (e.g. a server registration
   would surface an existing agent at the same URL).
2. **Similarity check**: a query against the semantic search pipeline
   without an entity-type filter (so the advisory list can still
   surface cross-type name/description matches, which are far more
   common than cross-type URL matches). Results are filtered by
   visibility, threshold, and self-exclusion, then returned as
   ``advisory_matches`` with each entry carrying its own
   ``entity_type``.

The service is always available — it is not gated by a feature flag.
The ``DEDUP_REGISTRATION_HINT_ENABLED`` setting controls only whether
the *frontend* pre-flights ``/check-duplicates`` before submission;
external callers (CLI, CI, federation) hit the endpoint directly
regardless.

The ``/register`` endpoints do not call this service. Both result sets
are hints, not enforcement gates; users can register a duplicate URL
if they choose to.
"""

import logging

from ..core.config import settings
from ..repositories.factory import (
    get_agent_repository,
    get_server_repository,
    get_skill_repository,
)
from ..repositories.interfaces import (
    AgentRepositoryBase,
    ServerRepositoryBase,
    SkillRepositoryBase,
)
from ..schemas.duplicate_check_models import (
    DuplicateCheckResult,
    EntityType,
    ExistingEntity,
)
from ..utils.url_normalize import (
    ENTITY_TYPE_AGENT,
    ENTITY_TYPE_SERVER,
    ENTITY_TYPE_SKILL,
    normalize_identity_url,
)
from .semantic_search_service import SemanticSearchService
from .visibility import (
    user_can_access_agent_from_doc,
    user_can_access_server,
    user_can_access_skill,
)

logger = logging.getLogger(__name__)

# The similarity check fetches more candidates than the user-visible
# cap to leave headroom for visibility filtering and self-exclusion to
# drop some results without shrinking the advisory list below the
# configured cap. 10x is calibrated for multi-tenant deployments
# where a non-admin caller may only have access to ~10% of the
# registry — at max_suggestions=3 that's 30 candidates fetched, ~3
# expected to be visible. The cost is one search call returning more
# rows; the candidates that don't make the cut are dropped before
# they cross the API boundary.
_SIMILARITY_OVERFETCH_FACTOR: int = 10

# Cap on the text passed to the semantic search backend per query.
# The description field is unbounded in the registration form, and a
# multi-paragraph description would otherwise be embedded verbatim,
# wasting tokens and risking provider-side query-length errors. The
# first ~500 chars of name+description carry the bulk of the
# distinguishing signal for similarity.
_QUERY_TEXT_CHAR_CAP: int = 500


class DuplicateCheckService:
    """Cross-entity duplicate detection for entity registration.

    Holds references to all three entity repositories and the semantic
    search service so a single :meth:`check` call can scan across
    types. Stateless apart from its dependencies.
    """

    def __init__(self) -> None:
        self._settings = settings
        self._semantic_search_service: SemanticSearchService = SemanticSearchService()
        # One repository per entity type — keyed so the cross-entity URL
        # check can iterate without a chain of if/elif branches.
        self._repositories: dict[
            EntityType,
            ServerRepositoryBase | AgentRepositoryBase | SkillRepositoryBase,
        ] = {
            ENTITY_TYPE_SERVER: get_server_repository(),
            ENTITY_TYPE_AGENT: get_agent_repository(),
            ENTITY_TYPE_SKILL: get_skill_repository(),
        }

    async def check(
        self,
        name: str,
        description: str | None,
        identity_url: str | None,
        self_path: str | None,
        user_context: dict,
    ) -> DuplicateCheckResult:
        """Run both duplicate checks for one registration request.

        Both checks run unconditionally — neither short-circuits the
        other. The result envelope contains URL matches in
        ``collision_with`` (any entity type whose repo holds a
        match) and similarity matches in ``advisory_matches`` (any
        entity type); the frontend chooses what to display.

        Args:
            name: The proposed entity name.
            description: The proposed description (used as part of the
                similarity search query). May be empty/None.
            identity_url: The entity's identity URL. May be None — in
                that case the exact-match check is skipped and only
                similarity matches are returned.
            self_path: When re-registering, the caller's own path so
                it is excluded from both result sets.
            user_context: The caller's auth context, as produced by
                the ``nginx_proxied_auth`` dependency. Used for
                visibility filtering and for redacting
                ``collision_with`` entries for unauthorized callers.

        Returns:
            A :class:`DuplicateCheckResult`. The route layer wraps
            this in a 200 envelope; there is no 4xx path.
        """
        threshold = self._settings.dedup_score_threshold

        collisions = await self._find_exact_match_collisions(
            identity_url=identity_url,
            self_path=self_path,
            user_context=user_context,
        )
        # Exclude the caller's own path and any URL-match path from the
        # similarity list so an exact-URL hit does not also appear as
        # an advisory entry.
        excluded_paths = {entity.path for entity in collisions if entity.path}
        if self_path:
            excluded_paths.add(self_path)
        advisory_matches, similarity_search_available = await self._fetch_similarity_advisory(
            name=name,
            description=description,
            excluded_paths=excluded_paths,
            user_context=user_context,
        )
        return DuplicateCheckResult(
            collision_with=collisions,
            advisory_matches=advisory_matches,
            threshold=threshold,
            similarity_search_available=similarity_search_available,
        )

    async def _find_exact_match_collisions(
        self,
        identity_url: str | None,
        self_path: str | None,
        user_context: dict,
    ) -> list[ExistingEntity]:
        """Look up entities (any type) with a matching identity URL.

        All three repositories are scanned. Each one normalizes the
        incoming URL with **its own** rule (HTTP-style for
        server/agent repos, GitHub-style for skill repo) before the
        lookup, so cross-type comparisons are apples-to-apples with
        whatever each repo stored at write time. Each lookup is one
        indexed ``$eq`` against the sidecar field — total cost per
        check is three indexed queries.

        Self-exclusion has two paths: callers can pass ``self_path``
        explicitly (the UI does this for re-registration), and the
        service also auto-excludes when the matching document was
        registered by the same user. The owner-based exclusion makes
        the API robust for external callers (CLI, federation, CI) who
        may not track their own paths — re-registering their own
        entry shouldn't surface as a collision against itself.

        Visibility-redacted entries (caller cannot view the colliding
        entity) have blank ``path``/``name``/``owner`` but still appear
        in the list so the frontend knows a collision exists.
        """
        if not identity_url:
            return []

        collisions: list[ExistingEntity] = []
        for candidate_entity_type, repository in self._repositories.items():
            normalized = normalize_identity_url(identity_url, candidate_entity_type)
            if normalized is None:
                continue
            try:
                match = await repository.find_by_identity_url(normalized)
            except Exception:
                logger.exception(
                    "find_by_identity_url failed on %s repository; treating as no match.",
                    candidate_entity_type,
                )
                continue
            if match is None:
                continue
            match_path = match.get("path")
            if self_path is not None and match_path == self_path:
                continue
            collisions.append(
                await self._build_collision_entity(
                    entity_type=candidate_entity_type,
                    document=match,
                    user_context=user_context,
                )
            )
        return collisions

    @staticmethod
    def _caller_owns_match(
        entity_type: EntityType,
        document: dict,
        user_context: dict,
    ) -> bool:
        """True iff the matching document was registered by the caller.

        Compares the caller's username (from auth context) against the
        ownership field appropriate for the entity type. Servers and
        agents use ``registered_by``; skills use ``owner``. Returns
        False when either side is missing — we'd rather show an
        unnecessary collision than silently hide one.
        """
        username = str(user_context.get("username") or "")
        if not username:
            return False
        if entity_type == ENTITY_TYPE_SKILL:
            owner = str(document.get("owner") or "")
            return bool(owner) and owner == username
        if entity_type == ENTITY_TYPE_AGENT:
            agent_card = document.get("agent_card")
            if isinstance(agent_card, dict):
                registered_by = str(agent_card.get("registered_by") or "")
                if registered_by:
                    return registered_by == username
        registered_by = str(document.get("registered_by") or "")
        return bool(registered_by) and registered_by == username

    async def _build_collision_entity(
        self,
        entity_type: EntityType,
        document: dict,
        user_context: dict,
    ) -> ExistingEntity:
        """Project a repo doc into an ``ExistingEntity`` for collision_with.

        Applies visibility-aware redaction: callers who cannot view the
        colliding entity get a stub entry (blank fields) so the
        existence is exposed but ownership leaks nothing.
        """
        match_path = str(document.get("path") or "")
        match_name = self._extract_entity_name(entity_type, document)
        match_owner = self._extract_owner(entity_type, document)
        match_registered_at = self._extract_registered_at(document)

        can_view = await self._caller_can_view(entity_type, document, user_context)
        if not can_view:
            return ExistingEntity(
                entity_type=entity_type,
                path="",
                name="",
                owner=None,
                registered_at=None,
                relevance_score=None,
                match_reason="exact URL match",
            )
        return ExistingEntity(
            entity_type=entity_type,
            path=match_path,
            name=match_name,
            owner=match_owner,
            registered_at=match_registered_at,
            relevance_score=None,
            match_reason="exact URL match",
        )

    async def _fetch_similarity_advisory(
        self,
        name: str,
        description: str | None,
        excluded_paths: set[str],
        user_context: dict,
    ) -> tuple[list[ExistingEntity], bool]:
        """Query the search pipeline for similar entities across all types.

        Returns ``(matches, available)``. ``available`` is False when
        the embedding backend was unreachable. Matches are filtered to
        the configured similarity threshold and the caller's
        visibility scope, then capped to ``dedup_max_suggestions``
        across all entity types (the cap is global, not per-type).
        """
        threshold = self._settings.dedup_score_threshold
        max_suggestions = self._settings.dedup_max_suggestions
        query = self._build_query_text(name, description)
        if not query:
            return [], True

        try:
            raw_results = await self._semantic_search_service.search(
                query=query,
                entity_types=None,  # cross-entity advisory
                max_results=max_suggestions * _SIMILARITY_OVERFETCH_FACTOR,
            )
        except RuntimeError as exc:
            logger.warning(
                "Semantic search unavailable for duplicate check (%s); "
                "skipping similarity-based advisory matches.",
                exc,
            )
            return [], False
        except Exception:
            logger.exception(
                "Unexpected error from search backend during duplicate check; "
                "treating as unavailable."
            )
            return [], False

        candidates = self._flatten_search_results(raw_results)
        candidates.sort(key=lambda c: float(c[1].get("relevance_score") or 0.0), reverse=True)

        advisory: list[ExistingEntity] = []
        for entity_type, candidate in candidates:
            score = float(candidate.get("relevance_score") or 0.0)
            if score < threshold:
                continue
            candidate_path = str(candidate.get("path") or "")
            if candidate_path in excluded_paths:
                continue
            if not await self._caller_can_view(entity_type, candidate, user_context):
                continue
            advisory.append(
                ExistingEntity(
                    entity_type=entity_type,
                    path=candidate_path,
                    name=self._extract_entity_name(entity_type, candidate),
                    owner=self._extract_owner(entity_type, candidate),
                    registered_at=self._extract_registered_at(candidate),
                    relevance_score=score,
                    match_reason="similar name and description",
                )
            )
            if len(advisory) >= max_suggestions:
                break
        return advisory, True

    @staticmethod
    def _build_query_text(
        name: str,
        description: str | None,
    ) -> str:
        """Compose the similarity search query from name and description.

        Truncates to ``_QUERY_TEXT_CHAR_CAP`` to keep the embedded
        text bounded — the registration form's description field is
        unbounded but only the first few hundred characters typically
        carry meaningful similarity signal.
        """
        parts: list[str] = []
        if name:
            parts.append(name.strip())
        if description:
            parts.append(description.strip())
        joined = " ".join(part for part in parts if part)
        if len(joined) > _QUERY_TEXT_CHAR_CAP:
            return joined[:_QUERY_TEXT_CHAR_CAP]
        return joined

    @staticmethod
    def _flatten_search_results(
        raw_results: dict[str, list[dict]],
    ) -> list[tuple[EntityType, dict]]:
        """Flatten the categorized search results into (type, hit) pairs.

        ``SemanticSearchService.search`` returns a dict keyed by
        entity plural ("servers", "agents", "skills", ...). We only
        surface the three entity types the dedup service knows about;
        "tools" and "virtual_servers" are ignored.
        """
        flattened: list[tuple[EntityType, dict]] = []
        for hit in raw_results.get("servers") or []:
            flattened.append((ENTITY_TYPE_SERVER, hit))
        for hit in raw_results.get("agents") or []:
            flattened.append((ENTITY_TYPE_AGENT, hit))
        for hit in raw_results.get("skills") or []:
            flattened.append((ENTITY_TYPE_SKILL, hit))
        return flattened

    @staticmethod
    def _extract_entity_name(
        entity_type: EntityType,
        document: dict,
    ) -> str:
        """Pull the human-readable name out of a repo doc or search hit."""
        if entity_type == ENTITY_TYPE_SERVER:
            return str(document.get("server_name") or document.get("name") or "")
        if entity_type == ENTITY_TYPE_SKILL:
            return str(document.get("skill_name") or document.get("name") or "")
        # a2a_agent: search hits expose name on agent_card; repo dump has top-level name.
        agent_card = document.get("agent_card")
        if isinstance(agent_card, dict) and agent_card.get("name"):
            return str(agent_card["name"])
        return str(document.get("name") or "")

    @staticmethod
    def _extract_owner(
        entity_type: EntityType,
        document: dict,
    ) -> str | None:
        """Pull the owner / registered_by field out of a repo doc.

        Each entity type uses exactly one field for "who registered
        this": servers and agents store ``registered_by``, skills store
        ``owner``. No cross-field fallback — if the canonical field is
        missing the modal shows no owner rather than guessing.

        Agent search hits nest the field under ``agent_card``; agent
        repo dumps put it at the top level. Both shapes are handled.
        """
        if entity_type == ENTITY_TYPE_SERVER:
            value = document.get("registered_by")
            return str(value) if value else None
        if entity_type == ENTITY_TYPE_AGENT:
            agent_card = document.get("agent_card")
            if isinstance(agent_card, dict):
                value = agent_card.get("registered_by")
                if value:
                    return str(value)
            value = document.get("registered_by")
            return str(value) if value else None
        if entity_type == ENTITY_TYPE_SKILL:
            value = document.get("owner")
            return str(value) if value else None
        return None

    @staticmethod
    def _extract_registered_at(
        document: dict,
    ) -> str | None:
        """Pull the registration timestamp; tolerates string or datetime."""
        for key in ("registered_at", "created_at"):
            value = document.get(key)
            if value:
                return str(value)
        return None

    async def _caller_can_view(
        self,
        entity_type: EntityType,
        document: dict,
        user_context: dict,
    ) -> bool:
        """Check whether the caller is allowed to see this entity.

        Delegates to :mod:`registry.services.visibility`. For agents
        we use ``user_can_access_agent_from_doc`` (sync, reads
        visibility/registered_by/allowed_groups from the candidate
        dict) instead of the path-based variant — the dedup flow
        already has the data in hand, so refetching would be N+1.
        """
        path = str(document.get("path") or "")
        if entity_type == ENTITY_TYPE_SERVER:
            server_name = str(document.get("server_name") or document.get("name") or "")
            return await user_can_access_server(path, server_name, user_context)
        if entity_type == ENTITY_TYPE_AGENT:
            return user_can_access_agent_from_doc(document, user_context)
        if entity_type == ENTITY_TYPE_SKILL:
            # Missing visibility is treated as "private" rather than
            # defaulted to "public": for the dedup advisory list we
            # would rather hide a legacy/externally-inserted skill
            # from non-admins than risk exposing one. This matches
            # the agent path in user_can_access_agent_from_doc, which
            # also falls through to False when visibility is empty.
            visibility = str(document.get("visibility") or "private")
            owner = str(document.get("owner") or document.get("registered_by") or "")
            allowed_groups = list(document.get("allowed_groups") or [])
            return await user_can_access_skill(
                skill_path=path,
                visibility=visibility,
                owner=owner,
                allowed_groups=allowed_groups,
                user_context=user_context,
            )
        return False


# Module-level singleton. The service is stateless apart from its
# resolved dependencies, so a single shared instance avoids the cost
# of re-running the four factory calls on every request. Tests reset
# this via the helper below.
_singleton: DuplicateCheckService | None = None


def get_duplicate_check_service() -> DuplicateCheckService:
    """Return the shared ``DuplicateCheckService`` instance.

    Constructed lazily on first call. Routes use this rather than
    instantiating the service directly so per-request construction
    cost (4 factory calls + a SemanticSearchService) is paid once.

    Thread-safety note: the lazy ``if _singleton is None`` check is
    not protected by a lock. This is intentional and safe for the
    deployment model: the registry runs as single-process uvicorn
    workers driving an asyncio event loop, and ``__init__`` is
    fully synchronous (no ``await``), so two coroutines on the same
    loop cannot interleave inside the constructor. pytest-xdist
    forks processes, so cross-test interference is also impossible.
    A lock would only matter if the project ever switched to a
    threaded runtime — at which point the entire factory pattern
    would need revisiting, not just this function.
    """
    global _singleton
    if _singleton is None:
        _singleton = DuplicateCheckService()
    return _singleton


def reset_duplicate_check_service() -> None:
    """Reset the singleton. Tests call this in fixtures so each test
    builds a fresh service against the current monkey-patched factories.
    """
    global _singleton
    _singleton = None
