"""Thin adapter over ``SearchRepositoryBase.search`` for callers that
want a service layer rather than the raw repository.

The ``/api/search/semantic`` route currently calls the repository
directly; this adapter exists so other code (notably
``DuplicateCheckService``) can depend on a service abstraction without
pulling the repository factories into its constructor.
"""

import logging

from ..repositories.factory import get_search_repository
from ..repositories.interfaces import SearchRepositoryBase

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """Wraps the search repository's ``search()`` method.

    Stateless. Resolves its ``SearchRepositoryBase`` via the factory
    at construction; tests patch the factory at this module's level
    rather than passing a repo in.
    """

    def __init__(self) -> None:
        self._search_repository: SearchRepositoryBase = get_search_repository()

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict]]:
        """Run a semantic search and return categorized results.

        Args mirror ``SearchRepositoryBase.search`` so callers can swap
        in the repository directly during tests if they prefer.

        Returns:
            Dict keyed by entity plural ("servers", "tools", "agents",
            "skills", "virtual_servers"). Each value is a list of hits
            with at least ``path`` and ``relevance_score``.

        Raises:
            RuntimeError: When the embedding backend is unavailable.
                Callers should catch this and surface a degraded
                response rather than propagating the 500.
        """
        return await self._search_repository.search(
            query=query,
            entity_types=entity_types,
            max_results=max_results,
            include_draft=include_draft,
            include_deprecated=include_deprecated,
            include_disabled=include_disabled,
        )
