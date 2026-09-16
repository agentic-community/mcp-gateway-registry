"""Pre-filter and post-filter must not drift too far apart.

The registry runs three backends and they filter at different points:

- MongoDB CE and Atlas take ``_client_side_search()``, which filters inside the
  ``find()`` and then ranks the survivors exactly. Pre-filter.
- DocumentDB runs ``$search`` and filters afterwards, because ``$search`` has no
  filter option. Post-filter.

Post-filtering loses recall whenever the filter correlates with the similarity
ranking, and nothing enforced a bound on that loss. Issue #1751 measured the
consequence: on a corpus where 8 of 396 documents were visible skills, a narrow
``entity_types=["skill"]`` query returned 8 results on CE and 0 to 5 on
DocumentDB.

These tests pin the arithmetic rather than the engine. They model both orders of
operation over a seeded corpus, so they run on MongoDB CE in CI with no
DocumentDB. They do not cover HNSW approximation, which needs a real DocumentDB
target and is a separate tier.

The relationship they encode: recall from post-filtering tracks how much of the
corpus ``k`` retrieves, and how dense the wanted documents are within it. A
low-density filter needs a large ``k``, which is why ``k`` over-requests.
"""

import pytest

from registry.core.config import settings

# A seeded corpus stands in for a registry: three entity types, a similarity
# ordering, and an eligibility flag. Similarity is the list position, which lets
# the tests reason about recall without embeddings.
CORPUS_SIZE = 400


def _seeded_corpus(
    skill_share: float,
    visible_share: float,
    skills_rank_late: bool = False,
) -> list[dict]:
    """Build a corpus ordered best-similarity-first.

    Args:
        skill_share: Fraction of documents that are skills.
        visible_share: Fraction of documents that pass the status filter.
        skills_rank_late: When True, push skills to the back of the similarity
            order, modelling a query that semantically favours another type.

    Returns:
        Documents in similarity order, each with entity_type and visible.
    """
    docs = []
    for i in range(CORPUS_SIZE):
        is_skill = (i % int(1 / skill_share)) == 0 if skill_share > 0 else False
        docs.append(
            {
                "path": f"/doc-{i}",
                "entity_type": "skill" if is_skill else "mcp_server",
                "visible": (i % int(1 / visible_share)) == 0 if visible_share > 0 else False,
            }
        )
    if skills_rank_late:
        docs.sort(key=lambda d: d["entity_type"] == "skill")
    return docs


def _pre_filter(corpus: list[dict], etype: str, limit: int) -> list[str]:
    """CE behaviour: filter, then rank, then truncate."""
    return [d["path"] for d in corpus if d["entity_type"] == etype and d["visible"]][:limit]


def _post_filter(corpus: list[dict], etype: str, limit: int, k: int) -> list[str]:
    """DocumentDB behaviour: rank, truncate to k, then filter."""
    return [d["path"] for d in corpus[:k] if d["entity_type"] == etype and d["visible"]][:limit]


class TestPostFilterLosesRecallWhenKIsSmall:
    """The regression that issue #1751 found, reduced to arithmetic."""

    def test_small_k_starves_a_low_density_filter(self) -> None:
        """8 wanted documents in 400 means a 30-window expects under one."""
        corpus = _seeded_corpus(skill_share=0.25, visible_share=0.02)
        wanted = _pre_filter(corpus, "skill", 10)
        got = _post_filter(corpus, "skill", 10, k=30)

        assert len(wanted) >= 2, "seed should leave something to find"
        assert len(got) < len(wanted), "small k must under-return, which is the bug"

    def test_k_covering_the_corpus_reaches_parity(self) -> None:
        """Retrieve everything and post-filtering equals pre-filtering."""
        corpus = _seeded_corpus(skill_share=0.25, visible_share=0.02)
        wanted = _pre_filter(corpus, "skill", 10)
        got = _post_filter(corpus, "skill", 10, k=CORPUS_SIZE)

        assert got == wanted

    @pytest.mark.parametrize("k", [30, 100, 200, 400])
    def test_recall_never_decreases_as_k_grows(self, k) -> None:
        """Monotonicity: a larger window cannot lose a document it already had."""
        corpus = _seeded_corpus(skill_share=0.25, visible_share=0.02)
        wanted = set(_pre_filter(corpus, "skill", 10))
        smaller = set(_post_filter(corpus, "skill", 10, k=max(1, k // 2)))
        larger = set(_post_filter(corpus, "skill", 10, k=k))

        assert len(larger & wanted) >= len(smaller & wanted)

    def test_high_eligibility_needs_far_less_over_request(self) -> None:
        """Eligibility density dominates. This is why the ratio is worth logging."""
        sparse = _seeded_corpus(skill_share=0.25, visible_share=0.02)
        dense = _seeded_corpus(skill_share=0.25, visible_share=1.0)

        sparse_recall = len(
            set(_post_filter(sparse, "skill", 10, k=100)) & set(_pre_filter(sparse, "skill", 10))
        )
        dense_recall = len(
            set(_post_filter(dense, "skill", 10, k=100)) & set(_pre_filter(dense, "skill", 10))
        )

        assert dense_recall > sparse_recall


class TestCorrelationIsTheResidual:
    """Density predicts the average case, not the tail."""

    def test_a_query_favouring_another_type_starves_even_a_dense_filter(self) -> None:
        """Skills can be 25% of the corpus and still miss the window.

        This is the residual that fixing eligibility cannot remove, and the
        reason DocumentDB stays best-effort on narrow queries.
        """
        corpus = _seeded_corpus(skill_share=0.25, visible_share=1.0, skills_rank_late=True)
        wanted = _pre_filter(corpus, "skill", 10)
        got = _post_filter(corpus, "skill", 10, k=30)

        assert len(wanted) == 10
        assert len(got) == 0, "skills ranked last cannot appear in a top-30 window"


class TestConfiguredOverRequestIsSane:
    """The shipped defaults have to satisfy their own arithmetic."""

    def test_over_request_multiplier_exceeds_one(self) -> None:
        assert settings.vector_search_overrequest > 1

    def test_ef_search_can_hold_the_largest_k_the_api_allows(self) -> None:
        """max_results caps at 50, so k tops out at 50 * multiplier."""
        largest_k = 50 * settings.vector_search_overrequest
        assert settings.vector_search_ef_search >= min(largest_k, 1000)

    def test_ef_search_stays_within_the_documentdb_ceiling(self) -> None:
        """DocumentDB rejects efSearch above 1000."""
        assert 1 <= settings.vector_search_ef_search <= 1000
