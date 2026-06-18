"""DocumentDB-based repository for hybrid search (text + vector)."""

import logging
import math
import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ...core.config import embedding_config, settings
from ...schemas.agent_models import AgentCard
from ...utils.metadata import flatten_metadata_to_text
from ...utils.vector import cosine_similarity
from ..interfaces import SearchRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


# Stopwords to filter out when tokenizing queries for keyword matching
_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "as",
    "into",
    "through",
    "from",
    "what",
    "when",
    "where",
    "who",
    "which",
    "how",
    "why",
    "get",
    "set",
    "put",
}


def _tokenize_query(query: str) -> list[str]:
    """Tokenize a query string into meaningful keywords.

    Splits on non-word characters, filters stopwords and short tokens.

    Args:
        query: The search query string

    Returns:
        List of lowercase tokens suitable for keyword matching
    """
    tokens = [
        token.lower()
        for token in re.split(r"\W+", query)
        if token and len(token) > 2 and token.lower() not in _STOPWORDS
    ]
    return tokens


def _tokens_match_text(
    tokens: list[str],
    text: str,
) -> bool:
    """Check if any token matches within the given text.

    Args:
        tokens: List of query tokens
        text: Text to search within

    Returns:
        True if any token is found in the text
    """
    if not tokens or not text:
        return False
    text_lower = text.lower()
    return any(token in text_lower for token in tokens)


# Maximum possible text_boost sum for lexical scoring normalization
# path(5.0) + name(3.0) + description(2.0) + tag(1.5) + metadata(1.0) + tool(1.0) = 13.5
MAX_LEXICAL_BOOST: float = 13.5

# RRF sensitivity constant (industry standard: k=60)
# Higher k smooths differences between ranks; lower k amplifies top positions.
RRF_K: int = 60

# Maximum fraction of max_results any single entity type can claim
# when other entity types have results competing for slots.
# 0.6 means no type gets more than 60% of total unless no competition.
SOFT_CAP_RATIO: float = 0.6


def _tool_extraction_limit(
    max_results: int,
) -> int:
    """Calculate the maximum number of tools to extract from server matching_tools.

    Uses the soft cap ratio but never goes below 3 for backward compatibility.

    Args:
        max_results: The max_results parameter from the search request.

    Returns:
        Maximum number of tools to extract.
    """
    return max(3, math.ceil(max_results * SOFT_CAP_RATIO))


def _format_custom_result(
    doc: dict[str, Any],
    relevance_score: float,
) -> dict[str, Any]:
    """Format a custom-entity search hit into a generic result envelope.

    Custom types have no fixed schema, so the result carries the entity_type
    discriminator and the common envelope fields; the schema-driven UI renders
    attributes from the type descriptor.
    """
    return {
        "entity_type": doc.get("entity_type"),
        "path": doc.get("path"),
        "name": doc.get("name"),
        "description": doc.get("description"),
        "tags": doc.get("tags", []),
        "visibility": doc.get("visibility", "private"),
        "owner": doc.get("owner"),
        "allowed_groups": doc.get("allowed_groups", []),
        "is_enabled": doc.get("is_enabled", False),
        "relevance_score": relevance_score,
        "match_context": doc.get("description"),
    }


def _distribute_results(
    scored_results: list[tuple[dict, float]],
    max_results: int,
) -> list[tuple[dict, float]]:
    """Select top results with competitive soft caps per entity type.

    Picks the top max_results items by relevance_score. A soft cap prevents
    any single entity type from taking more than 60% of slots -- but the cap
    is only enforced when other entity types have results waiting below in
    the ranking. If no other types remain, the cap is lifted.

    Uses a two-pass approach:
    1. First pass: pick items respecting soft caps
    2. Backfill pass: if we haven't reached max_results, fill remaining
       slots from skipped items (highest score first)

    Args:
        scored_results: List of (doc, relevance_score) tuples, sorted by
            relevance_score descending.
        max_results: Maximum number of results to return.

    Returns:
        Filtered list of (doc, relevance_score) tuples, length <= max_results.
    """
    if not scored_results or max_results <= 0:
        return []

    soft_cap = max(1, math.ceil(max_results * SOFT_CAP_RATIO))
    type_counts: dict[str, int] = {}
    selected: list[tuple[dict, float]] = []
    skipped: list[tuple[dict, float]] = []

    # Pre-compute which entity types exist at each position onward.
    # remaining_types[i] = set of entity types present in scored_results[i:]
    total = len(scored_results)
    remaining_types: list[set[str]] = [set() for _ in range(total + 1)]
    for i in range(total - 1, -1, -1):
        entity_type = scored_results[i][0].get("entity_type", "")
        remaining_types[i] = remaining_types[i + 1] | {entity_type}

    # Pass 1: pick items respecting soft caps
    for i, (doc, score) in enumerate(scored_results):
        if len(selected) >= max_results:
            break

        entity_type = doc.get("entity_type", "")
        current_count = type_counts.get(entity_type, 0)

        if current_count >= soft_cap:
            # Check if other types still have results after this position
            types_after = remaining_types[i + 1] - {entity_type}
            if types_after:
                skipped.append((doc, score))
                continue  # Other types waiting -- enforce cap
            # No competition -- allow this type to fill remaining slots

        selected.append((doc, score))
        type_counts[entity_type] = current_count + 1

    # Pass 2: backfill from skipped items if we haven't reached max_results
    # Skipped items are already in descending score order
    for doc, score in skipped:
        if len(selected) >= max_results:
            break
        selected.append((doc, score))

    logger.debug(
        "Search distribution: max_results=%d, soft_cap=%d, selected=%d, per_type=%s",
        max_results,
        soft_cap,
        len(selected),
        dict(type_counts),
    )

    return selected


def _reciprocal_rank_fusion(
    vector_ranked: list[dict],
    keyword_ranked: list[dict],
    k: int = RRF_K,
) -> list[tuple[dict, float]]:
    """Combine vector and keyword ranked lists using Reciprocal Rank Fusion.

    RRF produces a single merged ranking from two independently-ranked lists
    without needing score normalization. The formula per document is:

        rrf_score = sum over lists: 1 / (k + rank)

    where rank starts at 1 for the top result in each list.

    This is the industry standard approach used by Elasticsearch, OpenSearch,
    MongoDB Atlas, and Azure AI Search (all with k=60).

    Args:
        vector_ranked: Documents sorted by vector similarity (best first).
            Documents missing from this list (no embedding) simply receive
            no vector contribution.
        keyword_ranked: Documents sorted by text_boost (best first).
        k: Sensitivity constant. Default 60 (industry standard).
            Higher k reduces the influence of top positions.

    Returns:
        List of (doc, rrf_score) tuples sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank_zero, doc in enumerate(vector_ranked):
        doc_id = doc.get("_id") or doc.get("path", f"__vec_{rank_zero}")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_zero + 1)
        doc_map[doc_id] = doc

    for rank_zero, doc in enumerate(keyword_ranked):
        doc_id = doc.get("_id") or doc.get("path", f"__kw_{rank_zero}")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_zero + 1)
        doc_map[doc_id] = doc

    results = [(doc_map[doc_id], score) for doc_id, score in scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


SCORE_DISPLAY_FLOOR: float = 0.10


def _normalize_scores(
    scored_results: list[tuple[dict, float]],
    max_results: int = 10,
) -> list[tuple[dict, float]]:
    """Normalize scores to [0, 1] and drop results below the display floor.

    Maps the highest score to 1.0 and scales others proportionally.
    Results whose normalized score falls below SCORE_DISPLAY_FLOOR (10%)
    are excluded, unless that would leave fewer results than max_results.

    For a single result, returns 1.0. For empty input, returns empty.

    Args:
        scored_results: List of (doc, score) tuples (any score range).
        max_results: Don't drop results if we'd have fewer than this.

    Returns:
        Filtered list with scores in [0, 1.0].
    """
    if not scored_results:
        return []
    if len(scored_results) == 1:
        return [(scored_results[0][0], 1.0)]

    max_score = scored_results[0][1]
    min_score = scored_results[-1][1]

    if max_score == min_score:
        return [(doc, 1.0) for doc, _ in scored_results]

    normalized = []
    for doc, score in scored_results:
        norm = (score - min_score) / (max_score - min_score)
        display_score = max(0.0, round(norm, 4))
        normalized.append((doc, display_score))

    above_floor = [item for item in normalized if item[1] >= SCORE_DISPLAY_FLOOR]

    if len(above_floor) >= max_results:
        return above_floor

    return normalized

    return normalized


def _score_tool_relevance(
    tool_name: str,
    tool_description: str,
    query_tokens: list[str],
) -> float:
    """Score a tool independently based on keyword match strength.

    Returns a score between 0.0 and 1.0 based on how well the tool's
    name and description match the query tokens.

    Args:
        tool_name: The tool's name
        tool_description: The tool's description
        query_tokens: Tokenized query

    Returns:
        Score from 0.0 to 1.0
    """
    if not query_tokens:
        return 0.0

    score = 0.0
    name_lower = tool_name.lower()
    desc_lower = tool_description.lower()

    matched_tokens = 0
    for token in query_tokens:
        if token in name_lower:
            score += 0.5
            matched_tokens += 1
        elif token in desc_lower:
            score += 0.3
            matched_tokens += 1

    if not matched_tokens:
        return 0.0

    token_coverage = matched_tokens / len(query_tokens)
    score = min(1.0, score) * 0.7 + token_coverage * 0.3

    return round(min(1.0, score), 4)


def _build_status_filter(
    include_draft: bool = False,
    include_deprecated: bool = False,
    include_disabled: bool = False,
) -> dict:
    """Build MongoDB $match filter to exclude statuses and disabled entities.

    By default, draft, deprecated, and disabled assets are excluded from search.
    Existing documents without a status field are treated as active.
    Existing documents without is_enabled field are treated as enabled.

    Applied consistently across servers, agents, and skills.

    Args:
        include_draft: If True, include draft assets in results
        include_deprecated: If True, include deprecated assets in results
        include_disabled: If True, include disabled assets in results

    Returns:
        MongoDB filter dict (empty dict if no filtering needed)
    """
    conditions: list[dict] = []

    # Status filtering
    excluded_statuses = []
    if not include_draft:
        excluded_statuses.append("draft")
    if not include_deprecated:
        excluded_statuses.append("deprecated")

    if excluded_statuses:
        # Exclude listed statuses; documents missing the field are treated as active
        conditions.append(
            {
                "$or": [
                    {"status": {"$nin": excluded_statuses}},
                    {"status": {"$exists": False}},
                ]
            }
        )

    # Enabled filtering
    if not include_disabled:
        # Exclude disabled entities; documents missing is_enabled are treated as enabled
        conditions.append(
            {
                "$or": [
                    {"is_enabled": True},
                    {"is_enabled": {"$exists": False}},
                ]
            }
        )

    if not conditions:
        return {}

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def _build_keyword_match_filter(
    token_regex: str,
    entity_types: list[str] | None = None,
) -> dict:
    """Build the $match filter for keyword matching across document fields.

    Args:
        token_regex: Regex pattern combining query tokens with OR
        entity_types: Optional list of entity types to filter

    Returns:
        MongoDB $match filter dict
    """
    match_filter: dict[str, Any] = {
        "$or": [
            {"name": {"$regex": token_regex, "$options": "i"}},
            {"path": {"$regex": token_regex, "$options": "i"}},
            {"description": {"$regex": token_regex, "$options": "i"}},
            {"tags": {"$regex": token_regex, "$options": "i"}},
            {"tools.name": {"$regex": token_regex, "$options": "i"}},
            {"tools.description": {"$regex": token_regex, "$options": "i"}},
            {"metadata_text": {"$regex": token_regex, "$options": "i"}},
        ]
    }
    if entity_types:
        match_filter["entity_type"] = {"$in": entity_types}
    return match_filter


def _build_text_boost_stage(
    token_regex: str,
) -> dict:
    """Build the $addFields stage for text boost calculation.

    Computes text_boost by matching query tokens against document fields:
    path (+5.0), name (+3.0), description (+2.0), tags (+1.5), metadata (+1.0), tools (+1.0).

    Args:
        token_regex: Regex pattern combining query tokens with OR

    Returns:
        MongoDB $addFields pipeline stage dict
    """
    return {
        "$addFields": {
            "text_boost": {
                "$add": [
                    # Path match: +5.0
                    {
                        "$cond": [
                            {
                                "$regexMatch": {
                                    "input": {"$ifNull": ["$path", ""]},
                                    "regex": token_regex,
                                    "options": "i",
                                }
                            },
                            5.0,
                            0.0,
                        ]
                    },
                    # Name match: +3.0
                    {
                        "$cond": [
                            {
                                "$regexMatch": {
                                    "input": {"$ifNull": ["$name", ""]},
                                    "regex": token_regex,
                                    "options": "i",
                                }
                            },
                            3.0,
                            0.0,
                        ]
                    },
                    # Description match: +2.0
                    {
                        "$cond": [
                            {
                                "$regexMatch": {
                                    "input": {"$ifNull": ["$description", ""]},
                                    "regex": token_regex,
                                    "options": "i",
                                }
                            },
                            2.0,
                            0.0,
                        ]
                    },
                    # Tags match: +1.5 if any tag matches
                    {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$size": {
                                            "$filter": {
                                                "input": {"$ifNull": ["$tags", []]},
                                                "as": "tag",
                                                "cond": {
                                                    "$regexMatch": {
                                                        "input": "$$tag",
                                                        "regex": token_regex,
                                                        "options": "i",
                                                    }
                                                },
                                            }
                                        }
                                    },
                                    0,
                                ]
                            },
                            1.5,
                            0.0,
                        ]
                    },
                    # Metadata match: +1.0
                    {
                        "$cond": [
                            {
                                "$regexMatch": {
                                    "input": {"$ifNull": ["$metadata_text", ""]},
                                    "regex": token_regex,
                                    "options": "i",
                                }
                            },
                            1.0,
                            0.0,
                        ]
                    },
                    # Tools match: +1.0 per matching tool
                    {
                        "$size": {
                            "$filter": {
                                "input": {"$ifNull": ["$tools", []]},
                                "as": "tool",
                                "cond": {
                                    "$or": [
                                        {
                                            "$regexMatch": {
                                                "input": {"$ifNull": ["$$tool.name", ""]},
                                                "regex": token_regex,
                                                "options": "i",
                                            }
                                        },
                                        {
                                            "$regexMatch": {
                                                "input": {"$ifNull": ["$$tool.description", ""]},
                                                "regex": token_regex,
                                                "options": "i",
                                            }
                                        },
                                    ]
                                },
                            }
                        }
                    },
                ]
            },
            # Track matching tools for display
            "matching_tools": {
                "$map": {
                    "input": {
                        "$filter": {
                            "input": {"$ifNull": ["$tools", []]},
                            "as": "tool",
                            "cond": {
                                "$or": [
                                    {
                                        "$regexMatch": {
                                            "input": {"$ifNull": ["$$tool.name", ""]},
                                            "regex": token_regex,
                                            "options": "i",
                                        }
                                    },
                                    {
                                        "$regexMatch": {
                                            "input": {"$ifNull": ["$$tool.description", ""]},
                                            "regex": token_regex,
                                            "options": "i",
                                        }
                                    },
                                ]
                            },
                        }
                    },
                    "as": "tool",
                    "in": {
                        "tool_name": "$$tool.name",
                        "description": {"$ifNull": ["$$tool.description", ""]},
                        "relevance_score": 1.0,
                        "match_context": {
                            "$cond": [
                                {"$ne": ["$$tool.description", None]},
                                "$$tool.description",
                                {"$concat": ["Tool: ", "$$tool.name"]},
                            ]
                        },
                    },
                }
            },
        }
    }


class DocumentDBSearchRepository(SearchRepositoryBase):
    """DocumentDB implementation with hybrid search (text + vector)."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name(
            f"mcp_embeddings_{settings.embeddings_model_dimensions}"
        )
        self._embedding_model = None
        self._embedding_unavailable: bool = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from ...embeddings import create_embeddings_client

            self._embedding_model = create_embeddings_client(
                provider=settings.embeddings_provider,
                model_name=settings.embeddings_model_name,
                model_dir=settings.embeddings_model_dir,
                api_key=settings.embeddings_api_key,
                api_base=settings.embeddings_api_base,
                aws_region=settings.embeddings_aws_region,
                embedding_dimension=settings.embeddings_model_dimensions,
            )
        return self._embedding_model

    async def _embed_texts(
        self,
        texts: list[str],
        *,
        context: str,
        latch_unavailable: bool = True,
    ) -> list[list[float]] | None:
        """Encode texts using the lazy-loaded embedding model.

        Single funnel for every embedding call in this repository so the
        embedding-invariant (query and corpus must use the same model)
        holds structurally — there is exactly one encoder, used by both
        index_* and search/dedup paths.

        Args:
            texts: strings to encode.
            context: short human-readable label included in the warning
                log line on failure (e.g. ``"server 'github'"``,
                ``"hybrid search query"``). Helps operators trace which
                call path tripped the _embedding_unavailable latch.
            latch_unavailable: when True (default), an exception from
                ``model.encode()`` flips ``_embedding_unavailable`` so
                subsequent calls short-circuit. ``index_*`` callers pass
                False because they want to keep trying for later docs;
                ``search`` and ``dedup`` callers want the latch.

        Returns:
            A list of vectors (one per input) on success, or ``None`` on
            failure (exception or latch already set).
        """
        if latch_unavailable and self._embedding_unavailable:
            return None
        try:
            model = await self._get_embedding_model()
            vectors = model.encode(texts)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.warning(
                "Embedding model unavailable for %s: %s",
                context,
                exc,
            )
            if latch_unavailable:
                self._embedding_unavailable = True
            return None

    async def initialize(self) -> None:
        """Initialize the search service and create vector index."""
        logger.info(f"Initializing DocumentDB hybrid search on collection: {self._collection_name}")
        collection = await self._get_collection()

        try:
            indexes = await collection.list_indexes().to_list(length=100)
            index_names = [idx["name"] for idx in indexes]

            if "embedding_vector_idx" not in index_names:
                try:
                    logger.info("Creating HNSW vector index for embeddings...")
                    await collection.create_index(
                        [("embedding", "vector")],
                        name="embedding_vector_idx",
                        vectorOptions={
                            "type": "hnsw",
                            "similarity": "cosine",
                            "dimensions": settings.embeddings_model_dimensions,
                            "m": 16,
                            "efConstruction": 128,
                        },
                    )
                    logger.info("Created HNSW vector index")
                except Exception as vector_error:
                    # Check if this is a MongoDB CE error (vectorOptions not supported)
                    if "vectorOptions" in str(
                        vector_error
                    ) or "not valid for an index specification" in str(vector_error):
                        logger.warning(
                            "Vector indexes not supported (MongoDB CE detected). "
                            "Creating regular index on embedding field."
                        )
                        # Create a regular index on the embedding field for faster retrieval
                        await collection.create_index(
                            [("embedding", 1)], name="embedding_vector_idx"
                        )
                        logger.info("Created regular embedding index")
                    else:
                        # Re-raise if it's a different error
                        raise vector_error
            else:
                logger.info("Vector index already exists")

            if "path_idx" not in index_names:
                await collection.create_index([("path", 1)], name="path_idx", unique=True)
                logger.info("Created path index")

        except Exception as e:
            logger.error(f"Failed to initialize search indexes: {e}", exc_info=True)

    async def index_server(
        self,
        path: str,
        server_info: dict[str, Any],
        is_enabled: bool = False,
        skip_if_unchanged: bool = False,
    ) -> None:
        """Index a server for search.

        Args:
            skip_if_unchanged: When True (used during startup), skip embedding
                generation if the existing document has the same content hash.
                This avoids redundant embedding API calls on every boot.
        """
        import hashlib

        collection = await self._get_collection()

        text_parts = [
            server_info.get("server_name", ""),
            server_info.get("description", ""),
        ]

        tags = server_info.get("tags", [])
        if tags:
            text_parts.append("Tags: " + ", ".join(tags))

        for tool in server_info.get("tool_list", []):
            text_parts.append(tool.get("name", ""))
            text_parts.append(tool.get("description", ""))

        # Include custom metadata key-value pairs in embedding text
        metadata = server_info.get("metadata", {})
        if isinstance(metadata, dict) and metadata:
            for key, value in metadata.items():
                text_parts.append(f"{key}: {value}")

        text_for_embedding = " ".join(filter(None, text_parts))
        content_hash = hashlib.sha256(text_for_embedding.encode()).hexdigest()[:16]

        # Skip re-embedding if content hasn't changed (startup optimization)
        if skip_if_unchanged:
            existing = await collection.find_one(
                {"_id": path},
                {"content_hash": 1, "is_enabled": 1, "embedding": 1},
            )
            if (
                existing
                and existing.get("content_hash") == content_hash
                and existing.get("embedding")
            ):
                # Content unchanged and embedding exists; just update is_enabled if needed
                if existing.get("is_enabled") != is_enabled:
                    await collection.update_one({"_id": path}, {"$set": {"is_enabled": is_enabled}})
                return

        # Flatten metadata into a searchable text field for keyword matching
        metadata_text = flatten_metadata_to_text(metadata)

        # latch_unavailable=False: indexing failures should not poison
        # the latch for later docs in the same batch.
        vectors = await self._embed_texts(
            [text_for_embedding],
            context=f"server {server_info.get('server_name', path)!r}",
            latch_unavailable=False,
        )
        embedding = vectors[0] if vectors else []

        doc = {
            "_id": path,
            "entity_type": "mcp_server",
            "path": path,
            "name": server_info.get("server_name", ""),
            "description": server_info.get("description", ""),
            "tags": server_info.get("tags", []),
            "metadata_text": metadata_text,
            "is_enabled": is_enabled,
            "status": server_info.get("status", "active"),
            "text_for_embedding": text_for_embedding,
            "content_hash": content_hash,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "tools": [
                {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    # Support both "inputSchema" (MCP standard) and "schema" (legacy)
                    "inputSchema": t.get("inputSchema") or t.get("schema", {}),
                }
                for t in server_info.get("tool_list", [])
            ],
            "metadata": server_info,
            "indexed_at": server_info.get("updated_at", server_info.get("registered_at")),
        }

        try:
            await collection.replace_one({"_id": path}, doc, upsert=True)
            logger.info(f"Indexed server '{server_info.get('server_name')}' for search")
        except Exception as e:
            logger.error(f"Failed to index server in search: {e}", exc_info=True)

    async def index_agent(
        self,
        path: str,
        agent_card: AgentCard,
        is_enabled: bool = False,
        skip_if_unchanged: bool = False,
    ) -> None:
        """Index an agent for search."""
        import hashlib

        collection = await self._get_collection()

        text_parts = [
            agent_card.name,
            agent_card.description or "",
        ]

        tags = agent_card.tags or []
        if tags:
            text_parts.append("Tags: " + ", ".join(tags))

        # Include capability keys (feature flags like "streaming")
        if agent_card.capabilities:
            text_parts.append("Capabilities: " + ", ".join(agent_card.capabilities))

        # Include skill names and descriptions for better semantic search
        if agent_card.skills:
            for skill in agent_card.skills:
                text_parts.append(skill.name)
                if skill.description:
                    text_parts.append(skill.description)

        text_for_embedding = " ".join(filter(None, text_parts))
        content_hash = hashlib.sha256(text_for_embedding.encode()).hexdigest()[:16]

        if skip_if_unchanged:
            existing = await collection.find_one(
                {"_id": path},
                {"content_hash": 1, "is_enabled": 1, "embedding": 1},
            )
            if (
                existing
                and existing.get("content_hash") == content_hash
                and existing.get("embedding")
            ):
                if existing.get("is_enabled") != is_enabled:
                    await collection.update_one({"_id": path}, {"$set": {"is_enabled": is_enabled}})
                return

        vectors = await self._embed_texts(
            [text_for_embedding],
            context=f"agent {agent_card.name!r}",
            latch_unavailable=False,
        )
        embedding = vectors[0] if vectors else []

        # Flatten agent metadata for keyword search
        agent_metadata = getattr(agent_card, "metadata", None) or {}
        agent_metadata_text = flatten_metadata_to_text(agent_metadata)

        doc = {
            "_id": path,
            "entity_type": "a2a_agent",
            "path": path,
            "name": agent_card.name,
            "description": agent_card.description or "",
            "tags": agent_card.tags or [],
            "metadata_text": agent_metadata_text,
            "is_enabled": is_enabled,
            "status": getattr(agent_card, "status", "active"),
            "text_for_embedding": text_for_embedding,
            "content_hash": content_hash,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "capabilities": agent_card.capabilities or [],
            "metadata": agent_card.model_dump(mode="json"),
            "indexed_at": agent_card.updated_at or agent_card.registered_at,
        }

        try:
            await collection.replace_one({"_id": path}, doc, upsert=True)
            logger.info(f"Indexed agent '{agent_card.name}' for search")
        except Exception as e:
            logger.error(f"Failed to index agent in search: {e}", exc_info=True)

    async def index_skill(
        self,
        path: str,
        skill: Any,
        is_enabled: bool = False,
        skip_if_unchanged: bool = False,
    ) -> None:
        """Index a skill for semantic search.

        Args:
            path: Skill path (e.g., /skills/pdf-processing)
            skill: SkillCard object
            is_enabled: Whether skill is enabled
            skip_if_unchanged: Skip embedding if content hash matches existing doc
        """
        import hashlib

        collection = await self._get_collection()

        # Compose text for embedding
        text_parts = [
            skill.name,
            skill.description,
        ]

        if skill.tags:
            text_parts.append(f"Tags: {', '.join(skill.tags)}")

        if skill.compatibility:
            text_parts.append(f"Compatibility: {skill.compatibility}")

        if skill.target_agents:
            text_parts.append(f"For: {', '.join(skill.target_agents)}")

        if skill.metadata and skill.metadata.author:
            text_parts.append(f"Author: {skill.metadata.author}")

        if skill.metadata and skill.metadata.extra:
            extra_text = flatten_metadata_to_text(skill.metadata.extra)
            if extra_text:
                text_parts.append(extra_text)

        text_for_embedding = " ".join(filter(None, text_parts))
        content_hash = hashlib.sha256(text_for_embedding.encode()).hexdigest()[:16]

        if skip_if_unchanged:
            existing = await collection.find_one(
                {"_id": path},
                {"content_hash": 1, "is_enabled": 1, "embedding": 1},
            )
            if (
                existing
                and existing.get("content_hash") == content_hash
                and existing.get("embedding")
            ):
                if existing.get("is_enabled") != is_enabled:
                    await collection.update_one({"_id": path}, {"$set": {"is_enabled": is_enabled}})
                return

        vectors = await self._embed_texts(
            [text_for_embedding],
            context=f"skill {skill.name!r}",
            latch_unavailable=False,
        )
        embedding = vectors[0] if vectors else []

        # Handle visibility enum
        visibility_value = skill.visibility
        if hasattr(visibility_value, "value"):
            visibility_value = visibility_value.value

        # Flatten skill metadata for keyword search
        skill_metadata_parts = []
        if skill.metadata and skill.metadata.author:
            skill_metadata_parts.append(f"author {skill.metadata.author}")
        if skill.metadata and skill.metadata.version:
            skill_metadata_parts.append(f"version {skill.metadata.version}")
        if skill.metadata and skill.metadata.extra:
            extra_text = flatten_metadata_to_text(skill.metadata.extra)
            if extra_text:
                skill_metadata_parts.append(extra_text)
        if skill.registry_name:
            skill_metadata_parts.append(f"registry {skill.registry_name}")
        skill_metadata_text = " ".join(skill_metadata_parts)

        # Build search document
        search_doc = {
            "_id": path,
            "entity_type": "skill",
            "path": path,
            "name": skill.name,
            "description": skill.description,
            "tags": skill.tags or [],
            "metadata_text": skill_metadata_text,
            "is_enabled": is_enabled,
            "visibility": visibility_value,
            "allowed_groups": skill.allowed_groups or [],
            "owner": skill.owner,
            "health_status": skill.health_status,
            "last_checked_time": skill.last_checked_time.isoformat()
            if skill.last_checked_time
            else None,
            "status": getattr(skill, "status", "active"),
            "text_for_embedding": text_for_embedding,
            "content_hash": content_hash,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "metadata": {
                "skill_md_url": str(skill.skill_md_url),
                "skill_md_raw_url": str(skill.skill_md_raw_url) if skill.skill_md_raw_url else None,
                "author": skill.metadata.author if skill.metadata else None,
                "version": skill.metadata.version if skill.metadata else None,
                "compatibility": skill.compatibility,
                "target_agents": skill.target_agents or [],
                "registry_name": skill.registry_name,
            },
            "indexed_at": skill.updated_at or skill.created_at,
        }

        # Upsert to search collection
        try:
            await collection.replace_one({"_id": path}, search_doc, upsert=True)
            logger.info(f"Indexed skill for search: {path}")
        except Exception as e:
            logger.error(f"Failed to index skill in search: {e}", exc_info=True)

    async def index_custom_entity(
        self,
        record: Any,
        descriptor: Any,
    ) -> None:
        """Index a custom entity record for semantic search.

        Mirrors index_skill, but assembles the embedding text from descriptor
        fields flagged ``semantic`` and routes the remaining fields into
        ``metadata_text`` for keyword-search boosting.

        Args:
            record: CustomEntityRecord to index.
            descriptor: CustomTypeDescriptor describing the record's type.
        """
        import hashlib

        collection = await self._get_collection()

        # Semantic fields -> embedding text (vector search).
        text_parts = [record.name]
        if record.description:
            text_parts.append(record.description)
        for f in descriptor.fields:
            if f.semantic and f.name in record.attributes:
                val = record.attributes[f.name]
                text_parts.append(", ".join(val) if isinstance(val, list) else str(val))
        if record.tags:
            text_parts.append("Tags: " + ", ".join(record.tags))
        text_for_embedding = " ".join(p for p in text_parts if p)
        content_hash = hashlib.sha256(text_for_embedding.encode()).hexdigest()[:16]

        # Non-semantic fields -> metadata_text (keyword-search boost).
        metadata_parts: list[str] = []
        for f in descriptor.fields:
            if not f.semantic and f.name in record.attributes:
                val = record.attributes[f.name]
                if isinstance(val, list):
                    metadata_parts.append(f"{f.name}: {', '.join(str(x) for x in val)}")
                elif isinstance(val, bool):
                    pass  # booleans don't help keyword search
                elif val is not None:
                    metadata_parts.append(f"{f.name}: {val}")
        metadata_text = " | ".join(metadata_parts) if metadata_parts else ""

        vectors = await self._embed_texts(
            [text_for_embedding],
            context=f"custom {record.entity_type}",
            latch_unavailable=False,
        )
        embedding = vectors[0] if vectors else []

        search_doc = {
            "_id": record.path,
            "entity_type": record.entity_type,
            "path": record.path,
            "name": record.name,
            "description": record.description or "",
            "tags": record.tags or [],
            "metadata_text": metadata_text,
            "is_enabled": record.is_enabled,
            "visibility": record.visibility,
            "allowed_groups": record.allowed_groups or [],
            "owner": record.owner,
            "text_for_embedding": text_for_embedding,
            "content_hash": content_hash,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "indexed_at": record.updated_at or record.created_at,
        }

        try:
            await collection.replace_one({"_id": record.path}, search_doc, upsert=True)
            logger.info(f"Indexed custom entity for search: {record.path}")
        except Exception as e:
            logger.error(f"Failed to index custom entity in search: {e}", exc_info=True)

    async def delete_custom_entity_index(
        self,
        path: str,
    ) -> None:
        """Remove a single custom record's embedding (called on record DELETE)."""
        collection = await self._get_collection()
        try:
            result = await collection.delete_one({"_id": path})
            if result.deleted_count > 0:
                logger.info(f"Removed custom entity '{path}' from search index")
        except Exception as e:
            logger.error(f"Failed to remove custom entity from search index: {e}", exc_info=True)

    async def delete_custom_entity_index_by_type(
        self,
        entity_type: str,
    ) -> int:
        """Bulk-remove all embeddings for a type (first step of delete-type cascade)."""
        collection = await self._get_collection()
        try:
            result = await collection.delete_many({"entity_type": entity_type})
            logger.info(
                f"Removed {result.deleted_count} custom entity embeddings of type {entity_type}"
            )
            return result.deleted_count
        except Exception as e:
            logger.error(
                f"Failed to bulk-remove custom entity embeddings for type {entity_type}: {e}",
                exc_info=True,
            )
            return 0

    async def index_virtual_server(
        self,
        path: str,
        virtual_server: Any,
        is_enabled: bool = False,
    ) -> None:
        """Index a virtual server for semantic search.

        Args:
            path: Virtual server path (e.g., /virtual/dev-essentials)
            virtual_server: VirtualServerConfig object
            is_enabled: Whether virtual server is enabled
        """
        # Lazy import to avoid circular dependency
        from ...services.server_service import server_service

        collection = await self._get_collection()

        # Get backend server paths for metadata
        backend_paths = list(
            {mapping.backend_server_path for mapping in virtual_server.tool_mappings}
        )

        # Fetch tool descriptions from backend servers
        # Build a map: backend_path -> {tool_name -> description}
        backend_tool_descriptions: dict[str, dict[str, str]] = {}
        for backend_path in backend_paths:
            try:
                server_info = await server_service.get_server_info(backend_path)
                if server_info:
                    tool_list = server_info.get("tool_list", [])
                    backend_tool_descriptions[backend_path] = {
                        tool.get("name", ""): tool.get("description", "") for tool in tool_list
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch tools from backend {backend_path}: {e}")
                backend_tool_descriptions[backend_path] = {}

        # Compose text for embedding
        text_parts = [
            virtual_server.server_name,
            virtual_server.description or "",
        ]

        # Add tags
        if virtual_server.tags:
            text_parts.append(f"Tags: {', '.join(virtual_server.tags)}")

        # Build tools array and collect text for embedding
        tools = []
        tool_names = []
        for mapping in virtual_server.tool_mappings:
            display_name = mapping.alias or mapping.tool_name
            tool_names.append(display_name)

            # Use description_override if set, otherwise get from backend
            if mapping.description_override:
                description = mapping.description_override
            else:
                backend_tools = backend_tool_descriptions.get(mapping.backend_server_path, {})
                description = backend_tools.get(mapping.tool_name, "")

            # Add description to embedding text
            if description:
                text_parts.append(description)

            tools.append(
                {
                    "name": display_name,
                    "description": description,
                    "backend_server": mapping.backend_server_path,
                }
            )

        if tool_names:
            text_parts.append(f"Tools: {', '.join(tool_names)}")

        text_for_embedding = " ".join(filter(None, text_parts))

        vectors = await self._embed_texts(
            [text_for_embedding],
            context=f"virtual server {virtual_server.server_name!r}",
            latch_unavailable=False,
        )
        embedding = vectors[0] if vectors else []

        # Flatten virtual server metadata for keyword search
        vs_metadata_parts = []
        if virtual_server.created_by:
            vs_metadata_parts.append(f"created_by {virtual_server.created_by}")
        vs_metadata_text = " ".join(vs_metadata_parts)

        # Build search document
        search_doc = {
            "_id": path,
            "entity_type": "virtual_server",
            "path": path,
            "name": virtual_server.server_name,
            "description": virtual_server.description or "",
            "tags": virtual_server.tags or [],
            "metadata_text": vs_metadata_text,
            "is_enabled": is_enabled,
            "text_for_embedding": text_for_embedding,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "tools": tools,
            "metadata": {
                "server_name": virtual_server.server_name,
                "num_tools": len(virtual_server.tool_mappings),
                "backend_count": len(backend_paths),
                "backend_paths": backend_paths,
                "required_scopes": virtual_server.required_scopes,
                "supported_transports": virtual_server.supported_transports,
                "created_by": virtual_server.created_by,
            },
            "indexed_at": virtual_server.updated_at or virtual_server.created_at,
        }

        # Upsert to search collection
        try:
            await collection.replace_one({"_id": path}, search_doc, upsert=True)
            logger.info(f"Indexed virtual server for search: {path}")
        except Exception as e:
            logger.error(f"Failed to index virtual server in search: {e}", exc_info=True)

    async def search_by_tags(
        self,
        tags: list[str],
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search entities by exact tag match using a direct DB query."""
        collection = await self._get_collection()

        # Build a case-insensitive match for ALL tags
        tag_conditions: list[dict[str, Any]] = [
            {"tags": {"$regex": f"^{re.escape(tag)}$", "$options": "i"}} for tag in tags
        ]

        # Add lifecycle status and enabled filter
        status_filter = _build_status_filter(
            include_draft=include_draft,
            include_deprecated=include_deprecated,
            include_disabled=include_disabled,
        )
        if status_filter:
            tag_conditions.append(status_filter)

        query_filter: dict[str, Any] = {"$and": tag_conditions}
        if entity_types:
            query_filter["entity_type"] = {"$in": entity_types}

        cursor = collection.find(query_filter).limit(max_results * 5)
        results = await cursor.to_list(length=max_results * 5)

        logger.info(
            "Tag-only search for %s returned %d documents",
            tags,
            len(results),
        )

        # Format into grouped results using the lexical formatter
        # Assign relevance 1.0 since these are exact tag matches
        for doc in results:
            doc["text_boost"] = MAX_LEXICAL_BOOST
            doc["matching_tools"] = []
        return self._format_lexical_results(results, max_results)

    async def get_all_tags(self) -> list[str]:
        """Return a sorted list of all unique tags across all indexed entities."""
        collection = await self._get_collection()
        try:
            pipeline: list[dict[str, Any]] = [
                {"$match": {"tags": {"$exists": True, "$ne": []}}},
                {"$unwind": "$tags"},
                {"$group": {"_id": {"$toLower": "$tags"}, "original": {"$first": "$tags"}}},
                {"$sort": {"_id": 1}},
            ]
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=500)
            return [doc["original"] for doc in results]
        except Exception as e:
            logger.error("Failed to retrieve tags: %s", e, exc_info=True)
            return []

    async def remove_entity(
        self,
        path: str,
    ) -> bool:
        """Remove entity from search index.

        Returns:
            True if removal succeeded (including idempotent not-found).
            False if removal failed.
        """
        collection = await self._get_collection()

        try:
            result = await collection.delete_one({"_id": path})
            if result.deleted_count > 0:
                logger.info(f"Removed entity '{path}' from search index")
            else:
                logger.warning(f"Entity '{path}' not found in search index")
            return True
        except Exception as e:
            logger.error(f"Failed to remove entity from search index: {e}", exc_info=True)
            return False

    async def find_missing_embeddings(self) -> dict[str, Any]:
        """Find documents in source collections that have no embedding indexed.

        Compares _id values across mcp_servers, mcp_agents, and agent_skills
        collections against the embeddings collection.

        Returns:
            Dictionary with missing list, counts, and summary.
        """
        db = await get_documentdb_client()

        source_collections = [
            (get_collection_name("mcp_servers"), "mcp_server"),
            (get_collection_name("mcp_agents"), "a2a_agent"),
            (get_collection_name("agent_skills"), "skill"),
        ]

        embeddings_collection = await self._get_collection()
        indexed_cursor = embeddings_collection.find({}, {"_id": 1})
        indexed_docs = await indexed_cursor.to_list(length=None)
        indexed_ids = {doc["_id"] for doc in indexed_docs}

        missing = []
        total_source = 0

        for col_name, entity_type in source_collections:
            collection = db[col_name]
            cursor = collection.find(
                {},
                {"_id": 1, "server_name": 1, "name": 1, "is_enabled": 1},
            )
            source_docs = await cursor.to_list(length=None)
            total_source += len(source_docs)

            for doc in source_docs:
                doc_id = doc["_id"]
                if doc_id not in indexed_ids:
                    name = doc.get("server_name") or doc.get("name") or doc_id
                    missing.append(
                        {
                            "path": doc_id,
                            "entity_type": entity_type,
                            "name": name,
                            "is_enabled": doc.get("is_enabled", True),
                        }
                    )

        missing.sort(key=lambda x: (x["entity_type"], x["path"]))

        return {
            "missing": missing,
            "total_missing": len(missing),
            "total_indexed": len(indexed_ids),
            "total_source": total_source,
        }

    async def find_stale_embeddings(self) -> dict[str, Any]:
        """Find embedding-index documents with no matching source registry record.

        Inverse of :meth:`find_missing_embeddings` — detects orphaned vectors
        left behind when delete-time cleanup failed (issue #1145).

        NOTE: This performs an unbounded full scan of the embeddings collection
        and all source collections. For very large registries, add pagination
        (follow-up) before running this in production on a hot path.

        Returns:
            Dictionary with stale list, counts, and summary.
        """
        db = await get_documentdb_client()

        source_collections = [
            (get_collection_name("mcp_servers"), "mcp_server"),
            (get_collection_name("mcp_agents"), "a2a_agent"),
            (get_collection_name("agent_skills"), "skill"),
            (get_collection_name("virtual_servers"), "virtual_server"),
        ]

        source_ids: set[str] = set()
        total_source = 0

        for col_name, _entity_type in source_collections:
            collection = db[col_name]
            cursor = collection.find({}, {"_id": 1})
            source_docs = await cursor.to_list(length=None)
            total_source += len(source_docs)
            source_ids.update(doc["_id"] for doc in source_docs)

        embeddings_collection = await self._get_collection()
        indexed_cursor = embeddings_collection.find(
            {},
            {"_id": 1, "entity_type": 1, "name": 1, "is_enabled": 1},
        )
        indexed_docs = await indexed_cursor.to_list(length=None)

        stale = []
        for doc in indexed_docs:
            doc_id = doc["_id"]
            if doc_id not in source_ids:
                stale.append({
                    "path": doc_id,
                    "entity_type": doc.get("entity_type", "unknown"),
                    "name": doc.get("name") or doc_id,
                    "is_enabled": doc.get("is_enabled", True),
                })

        stale.sort(key=lambda x: (x["entity_type"], x["path"]))

        return {
            "stale": stale,
            "total_stale": len(stale),
            "total_indexed": len(indexed_docs),
            "total_source": total_source,
        }

    async def remove_stale_embeddings(
        self,
        paths: list[str],
    ) -> dict[str, Any]:
        """Remove orphaned embedding documents by path.

        Deletes inline (not via ``remove_entity``) so the caller can see whether
        each path was actually removed or was a no-op. ``remove_entity`` returns
        True for a not-found path on purpose (delete flows must proceed when no
        embedding exists), which would otherwise mask a typo'd or already-clean
        path here as ``removed``. Distinguishing the two gives admins honest
        feedback during manual cleanup.

        Per-path ``status`` is one of:
          - ``removed``   - a stale embedding existed and was deleted
          - ``not_found`` - nothing to delete at this path (no-op)
          - ``failed``    - the delete raised an error

        Args:
            paths: Embedding index paths to remove (max 100).

        Returns:
            Dictionary with removed/not_found/failed counts and per-path details.
        """
        collection = await self._get_collection()
        details: list[dict[str, Any]] = []

        for path in paths:
            try:
                result = await collection.delete_one({"_id": path})
                if result.deleted_count > 0:
                    logger.info("Removed stale embedding '%s' from search index", path)
                    details.append({"path": path, "status": "removed", "error": None})
                else:
                    logger.warning("Stale embedding '%s' not found (no-op)", path)
                    details.append({"path": path, "status": "not_found", "error": None})
            except Exception as e:
                logger.error("Failed to remove stale embedding '%s': %s", path, e)
                details.append({
                    "path": path,
                    "status": "failed",
                    "error": "Failed to remove stale embedding",
                })

        removed_count = sum(1 for d in details if d["status"] == "removed")
        not_found_count = sum(1 for d in details if d["status"] == "not_found")
        failed_count = sum(1 for d in details if d["status"] == "failed")

        logger.info(
            "Stale embedding cleanup: %d removed, %d not_found, %d failed out of %d paths",
            removed_count,
            not_found_count,
            failed_count,
            len(paths),
        )

        return {
            "removed": removed_count,
            "not_found": not_found_count,
            "failed": failed_count,
            "total": len(paths),
            "details": details,
        }

    async def reindex_paths(
        self,
        paths: list[str],
    ) -> dict[str, Any]:
        """Re-index specific documents by reading from source and generating embeddings.

        For each path, finds the source document in the appropriate collection
        and calls the corresponding index method.

        Args:
            paths: List of document paths to re-index (max 100).

        Returns:
            Dictionary with success/failed counts and per-path details.
        """
        db = await get_documentdb_client()

        servers_col = db[get_collection_name("mcp_servers")]
        agents_col = db[get_collection_name("mcp_agents")]
        skills_col = db[get_collection_name("agent_skills")]

        details = []

        for path in paths:
            try:
                server_doc = await servers_col.find_one({"_id": path})
                if server_doc:
                    await self.index_server(
                        path,
                        server_doc,
                        is_enabled=server_doc.get("is_enabled", True),
                    )
                    details.append(
                        {
                            "path": path,
                            "entity_type": "mcp_server",
                            "status": "success",
                        }
                    )
                    continue

                agent_doc = await agents_col.find_one({"_id": path})
                if agent_doc:
                    agent_card = AgentCard(**agent_doc)
                    await self.index_agent(
                        path,
                        agent_card,
                        is_enabled=agent_doc.get("is_enabled", True),
                    )
                    details.append(
                        {
                            "path": path,
                            "entity_type": "a2a_agent",
                            "status": "success",
                        }
                    )
                    continue

                skill_doc = await skills_col.find_one({"_id": path})
                if skill_doc:
                    from ...schemas.skill_models import SkillCard

                    skill_card = SkillCard(**skill_doc)
                    await self.index_skill(
                        path,
                        skill_card,
                        is_enabled=skill_doc.get("is_enabled", True),
                    )
                    details.append(
                        {
                            "path": path,
                            "entity_type": "skill",
                            "status": "success",
                        }
                    )
                    continue

                details.append(
                    {
                        "path": path,
                        "entity_type": "unknown",
                        "status": "failed",
                        "error": "Not found in any source collection",
                    }
                )

            except Exception as e:
                logger.error(f"Failed to reindex '{path}': {e}", exc_info=True)
                details.append(
                    {
                        "path": path,
                        "entity_type": "unknown",
                        "status": "failed",
                        "error": "Failed to re-index document due to an internal error",
                    }
                )

        success_count = sum(1 for d in details if d["status"] == "success")
        failed_count = sum(1 for d in details if d["status"] == "failed")

        logger.info(
            "Reindex completed: %d success, %d failed out of %d paths",
            success_count,
            failed_count,
            len(paths),
        )

        return {
            "success": success_count,
            "failed": failed_count,
            "total": len(paths),
            "details": details,
        }

    async def _client_side_search(
        self,
        query: str,
        query_embedding: list[float],
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fallback search using client-side cosine similarity for MongoDB CE.

        This method is used when MongoDB doesn't support native vector search.
        It fetches all embeddings from the database and computes similarity locally.
        """
        collection = await self._get_collection()

        try:
            # Build query filter
            query_filter = {}
            if entity_types:
                query_filter["entity_type"] = {"$in": entity_types}

            # Apply lifecycle status and enabled filter
            status_filter = _build_status_filter(
                include_draft=include_draft,
                include_deprecated=include_deprecated,
                include_disabled=include_disabled,
            )
            if status_filter:
                query_filter.update(status_filter)

            # Fetch all embeddings from MongoDB
            cursor = collection.find(
                query_filter,
                {
                    "_id": 1,
                    "path": 1,
                    "entity_type": 1,
                    "name": 1,
                    "description": 1,
                    "tags": 1,
                    "tools": 1,
                    "metadata": 1,
                    "metadata_text": 1,
                    "is_enabled": 1,
                    "status": 1,
                    "visibility": 1,
                    "owner": 1,
                    "allowed_groups": 1,
                    "health_status": 1,
                    "last_checked_time": 1,
                    "embedding": 1,
                },
            )

            all_docs = await cursor.to_list(length=None)
            docs_with_embeddings = sum(1 for d in all_docs if d.get("embedding"))
            logger.info(
                "Client-side search: Retrieved %d documents (%d with embeddings)",
                len(all_docs),
                docs_with_embeddings,
            )

            # Tokenize query for keyword matching
            query_tokens = _tokenize_query(query)
            logger.debug(f"Client-side search tokens: {query_tokens}")

            # Score each document on BOTH axes independently for RRF
            vector_scored: list[tuple[dict, float]] = []
            keyword_scored: list[tuple[dict, float]] = []

            for doc in all_docs:
                # --- Vector score (only for docs with embeddings) ---
                embedding = doc.get("embedding", [])
                if embedding:
                    vector_score = cosine_similarity(query_embedding, embedding)
                    vector_scored.append((doc, vector_score))

                # --- Keyword text_boost (all docs participate) ---
                text_boost = 0.0
                name = doc.get("name", "")
                description = doc.get("description", "")
                tags = doc.get("tags", [])
                tools = doc.get("tools", [])
                matching_tools = []

                path = doc.get("path", "")
                server_name_matched = False
                if path and _tokens_match_text(query_tokens, path):
                    text_boost += 5.0
                    server_name_matched = True
                if name and _tokens_match_text(query_tokens, name):
                    text_boost += 3.0
                    server_name_matched = True
                if description and _tokens_match_text(query_tokens, description):
                    text_boost += 2.0
                if tags and any(_tokens_match_text(query_tokens, tag) for tag in tags):
                    text_boost += 1.5

                metadata_text = doc.get("metadata_text", "")
                if metadata_text and _tokens_match_text(query_tokens, metadata_text):
                    text_boost += 1.0

                for tool in tools:
                    tool_name = tool.get("name", "")
                    tool_desc = tool.get("description") or ""
                    tool_score = _score_tool_relevance(tool_name, tool_desc, query_tokens)

                    if tool_score > 0:
                        text_boost += 1.0
                        matching_tools.append(
                            {
                                "tool_name": tool_name,
                                "description": tool_desc,
                                "relevance_score": tool_score,
                                "match_context": tool_desc or f"Tool: {tool_name}",
                            }
                        )

                doc["_matching_tools"] = matching_tools
                doc["text_boost"] = text_boost

                if text_boost > 0:
                    keyword_scored.append((doc, text_boost))

            # Sort each list independently (best first)
            vector_scored.sort(key=lambda x: x[1], reverse=True)
            keyword_scored.sort(key=lambda x: x[1], reverse=True)

            if settings.search_fusion_method == "rrf":
                logger.info(
                    "RRF inputs: %d vector-ranked docs, %d keyword-ranked docs",
                    len(vector_scored),
                    len(keyword_scored),
                )
                vector_ranked_docs = [doc for doc, _ in vector_scored]
                keyword_ranked_docs = [doc for doc, _ in keyword_scored]
                scored_tuples = _reciprocal_rank_fusion(vector_ranked_docs, keyword_ranked_docs)
                for doc, rrf_score in scored_tuples[:10]:
                    logger.info(
                        "RRF score for '%s' (type=%s): %.6f, text_boost=%.1f",
                        doc.get("name"),
                        doc.get("entity_type"),
                        rrf_score,
                        doc.get("text_boost", 0.0),
                    )
            else:
                scored_tuples = []
                for doc, vector_score in vector_scored:
                    text_boost = doc.get("text_boost", 0.0)
                    normalized_vector_score = (vector_score + 1.0) / 2.0
                    text_boost_contribution = text_boost * 0.1
                    relevance_score = max(
                        0.0, min(1.0, normalized_vector_score + text_boost_contribution)
                    )
                    scored_tuples.append((doc, relevance_score))
                for doc, text_boost in keyword_scored:
                    doc_id = doc.get("_id") or doc.get("path")
                    if not any((d.get("_id") or d.get("path")) == doc_id for d, _ in scored_tuples):
                        relevance_score = min(1.0, text_boost * 0.1)
                        scored_tuples.append((doc, relevance_score))
                scored_tuples.sort(key=lambda x: x[1], reverse=True)

            selected = _distribute_results(scored_tuples, max_results)

            if settings.search_fusion_method == "rrf":
                selected = _normalize_scores(selected, max_results)

            # Format results to match the API contract
            grouped_results: dict[str, list[dict[str, Any]]] = {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }

            tool_count = 0
            tool_limit = _tool_extraction_limit(max_results)

            for doc, relevance_score in selected:
                entity_type = doc.get("entity_type")

                if entity_type == "mcp_server":
                    matching_tools = doc.get("_matching_tools", [])
                    server_metadata = doc.get("metadata", {})

                    result_entry = {
                        "entity_type": "mcp_server",
                        "path": doc.get("path"),
                        "server_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "num_tools": server_metadata.get("num_tools", 0),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "matching_tools": matching_tools,
                        "proxy_pass_url": server_metadata.get("proxy_pass_url"),
                        "mcp_endpoint": server_metadata.get("mcp_endpoint"),
                        "sse_endpoint": server_metadata.get("sse_endpoint"),
                        "supported_transports": server_metadata.get("supported_transports", []),
                        "deployment": server_metadata.get("deployment", "remote"),
                        "local_runtime": server_metadata.get("local_runtime"),
                    }
                    grouped_results["servers"].append(result_entry)

                    # Also add matching tools to the top-level tools array
                    original_tools = doc.get("tools", [])
                    tool_schema_map = {
                        t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                    }

                    server_path = doc.get("path", "")
                    server_name = doc.get("name", "")
                    for tool in matching_tools:
                        if tool_count >= tool_limit:
                            break
                        tool_name = tool.get("tool_name", "")
                        grouped_results["tools"].append(
                            {
                                "entity_type": "tool",
                                "server_path": server_path,
                                "server_name": server_name,
                                "tool_name": tool_name,
                                "description": tool.get("description", ""),
                                "inputSchema": tool_schema_map.get(tool_name, {}),
                                "relevance_score": tool.get("relevance_score", 0.0),
                                "match_context": tool.get("match_context", ""),
                            }
                        )
                        tool_count += 1

                elif entity_type == "a2a_agent":
                    metadata = doc.get("metadata", {})
                    result_entry = {
                        "entity_type": "a2a_agent",
                        "path": doc.get("path"),
                        "agent_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "skills": metadata.get("skills", []),
                        "visibility": metadata.get("visibility", "public"),
                        "trust_level": metadata.get("trust_level"),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "agent_card": metadata.get("agent_card", {}),
                    }
                    grouped_results["agents"].append(result_entry)

                elif entity_type == "mcp_tool":
                    result_entry = {
                        "entity_type": "mcp_tool",
                        "path": doc.get("path"),
                        "tool_name": doc.get("name"),
                        "description": doc.get("description"),
                        "inputSchema": doc.get("inputSchema", {}),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                    }
                    grouped_results["tools"].append(result_entry)

                elif entity_type == "skill":
                    metadata = doc.get("metadata", {})
                    result_entry = {
                        "entity_type": "skill",
                        "path": doc.get("path"),
                        "skill_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "skill_md_url": metadata.get("skill_md_url"),
                        "version": metadata.get("version"),
                        "author": metadata.get("author"),
                        "visibility": doc.get("visibility", "public"),
                        "owner": doc.get("owner"),
                        "is_enabled": doc.get("is_enabled", False),
                        "status": doc.get("status", "active"),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                    }
                    grouped_results["skills"].append(result_entry)

                elif entity_type == "virtual_server":
                    metadata = doc.get("metadata", {})
                    matching_tools = doc.get("_matching_tools", [])
                    result_entry = {
                        "entity_type": "virtual_server",
                        "path": doc.get("path"),
                        "server_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "num_tools": metadata.get("num_tools", 0),
                        "backend_count": metadata.get("backend_count", 0),
                        "backend_paths": metadata.get("backend_paths", []),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "matching_tools": matching_tools,
                    }
                    grouped_results["virtual_servers"].append(result_entry)

                else:
                    grouped_results["custom"].append(_format_custom_result(doc, relevance_score))

            logger.info(
                "Client-side search returned "
                "%d servers, %d tools, %d agents, %d skills, "
                "%d virtual_servers, %d custom from %d total documents (max_results=%d)",
                len(grouped_results["servers"]),
                len(grouped_results["tools"]),
                len(grouped_results["agents"]),
                len(grouped_results["skills"]),
                len(grouped_results["virtual_servers"]),
                len(grouped_results["custom"]),
                len(all_docs),
                max_results,
            )

            return grouped_results

        except Exception as e:
            logger.error(f"Failed to perform client-side search: {e}", exc_info=True)
            return {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }

    async def _lexical_only_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fallback search using keyword matching only (no embeddings).

        Used when the embedding model fails to load. Scores results purely
        by keyword matches against name, path, description, tags, and tools.

        Args:
            query: The search query string
            entity_types: Optional list of entity types to filter
            max_results: Maximum number of results to return
            include_draft: If True, include draft assets in results
            include_deprecated: If True, include deprecated assets in results
            include_disabled: If True, include disabled assets in results

        Returns:
            Grouped search results dict with servers, tools, agents lists
        """
        collection = await self._get_collection()
        query_tokens = _tokenize_query(query)

        if not query_tokens:
            logger.info("Lexical search: no valid tokens from query '%s'", query)
            return {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }

        escaped_tokens = [re.escape(token) for token in query_tokens]
        token_regex = "|".join(escaped_tokens)

        keyword_match_filter = _build_keyword_match_filter(
            token_regex=token_regex,
            entity_types=entity_types,
        )

        text_boost_stage = _build_text_boost_stage(token_regex)

        pipeline: list[dict[str, Any]] = [
            {"$match": keyword_match_filter},
        ]

        # Apply lifecycle status and enabled filter
        status_filter = _build_status_filter(
            include_draft=include_draft,
            include_deprecated=include_deprecated,
            include_disabled=include_disabled,
        )
        if status_filter:
            pipeline.append({"$match": status_filter})

        pipeline.extend(
            [
                text_boost_stage,
                {"$sort": {"text_boost": -1}},
                {"$limit": max(max_results * 3, 50)},
            ]
        )

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=max(max_results * 3, 50))

        grouped_results = self._format_lexical_results(results, max_results)

        logger.info(
            "Lexical-only search for '%s' returned %d servers, %d tools, %d agents",
            query,
            len(grouped_results["servers"]),
            len(grouped_results["tools"]),
            len(grouped_results["agents"]),
        )

        return grouped_results

    def _format_lexical_results(
        self,
        results: list[dict],
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Format lexical search results into grouped response.

        Uses fixed-denominator normalization for relevance scoring.
        Applies global ranking with competitive soft caps via _distribute_results().

        Args:
            results: Raw MongoDB documents with text_boost field
            max_results: Maximum number of results to return

        Returns:
            Grouped search results dict with servers, tools, agents lists
        """
        # Score results and sort by relevance before distributing
        scored_tuples: list[tuple[dict, float]] = []
        for doc in results:
            text_boost = doc.get("text_boost", 0.0)
            relevance_score = min(1.0, text_boost / MAX_LEXICAL_BOOST)
            scored_tuples.append((doc, relevance_score))

        scored_tuples.sort(key=lambda x: x[1], reverse=True)
        selected = _distribute_results(scored_tuples, max_results)

        # Group selected results by entity type
        grouped_results: dict[str, list[dict[str, Any]]] = {
            "servers": [],
            "tools": [],
            "agents": [],
            "skills": [],
            "virtual_servers": [],
            "custom": [],
        }
        tool_count = 0
        tool_limit = _tool_extraction_limit(max_results)

        for doc, relevance_score in selected:
            entity_type = doc.get("entity_type")

            if entity_type == "mcp_server":
                matching_tools = doc.get("matching_tools", [])
                server_metadata = doc.get("metadata", {})
                result_entry = {
                    "entity_type": "mcp_server",
                    "path": doc.get("path"),
                    "server_name": doc.get("name"),
                    "description": doc.get("description"),
                    "tags": doc.get("tags", []),
                    "num_tools": server_metadata.get("num_tools", 0),
                    "is_enabled": doc.get("is_enabled", False),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                    "matching_tools": matching_tools,
                    "proxy_pass_url": server_metadata.get("proxy_pass_url"),
                    "mcp_endpoint": server_metadata.get("mcp_endpoint"),
                    "sse_endpoint": server_metadata.get("sse_endpoint"),
                    "supported_transports": server_metadata.get("supported_transports", []),
                }
                grouped_results["servers"].append(result_entry)

                # Add matching tools to top-level tools array
                original_tools = doc.get("tools", [])
                tool_schema_map = {
                    t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                }
                server_path = doc.get("path", "")
                server_name = doc.get("name", "")
                for tool in matching_tools:
                    if tool_count >= tool_limit:
                        break
                    tool_name = tool.get("tool_name", "")
                    grouped_results["tools"].append(
                        {
                            "entity_type": "tool",
                            "server_path": server_path,
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "description": tool.get("description", ""),
                            "inputSchema": tool_schema_map.get(tool_name, {}),
                            "relevance_score": tool.get("relevance_score", 0.0),
                            "match_context": tool.get("match_context", ""),
                        }
                    )
                    tool_count += 1

            elif entity_type == "a2a_agent":
                metadata = doc.get("metadata", {})
                result_entry = {
                    "entity_type": "a2a_agent",
                    "path": doc.get("path"),
                    "agent_name": doc.get("name"),
                    "description": doc.get("description"),
                    "tags": doc.get("tags", []),
                    "skills": metadata.get("skills", []),
                    "visibility": metadata.get("visibility", "public"),
                    "trust_level": metadata.get("trust_level"),
                    "is_enabled": doc.get("is_enabled", False),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                    "agent_card": metadata.get("agent_card", {}),
                }
                grouped_results["agents"].append(result_entry)

            elif entity_type == "mcp_tool":
                result_entry = {
                    "entity_type": "mcp_tool",
                    "path": doc.get("path"),
                    "tool_name": doc.get("name"),
                    "description": doc.get("description"),
                    "inputSchema": doc.get("inputSchema", {}),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                }
                grouped_results["tools"].append(result_entry)

            elif entity_type == "skill":
                metadata = doc.get("metadata", {})
                result_entry = {
                    "entity_type": "skill",
                    "path": doc.get("path"),
                    "skill_name": doc.get("name"),
                    "description": doc.get("description"),
                    "tags": doc.get("tags", []),
                    "skill_md_url": metadata.get("skill_md_url"),
                    "version": metadata.get("version"),
                    "author": metadata.get("author"),
                    "visibility": doc.get("visibility", "public"),
                    "owner": doc.get("owner"),
                    "is_enabled": doc.get("is_enabled", False),
                    "status": doc.get("status", "active"),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                }
                grouped_results["skills"].append(result_entry)

            elif entity_type == "virtual_server":
                metadata = doc.get("metadata", {})
                matching_tools = doc.get("matching_tools", [])
                result_entry = {
                    "entity_type": "virtual_server",
                    "path": doc.get("path"),
                    "server_name": doc.get("name"),
                    "description": doc.get("description"),
                    "tags": doc.get("tags", []),
                    "num_tools": metadata.get("num_tools", 0),
                    "backend_count": metadata.get("backend_count", 0),
                    "backend_paths": metadata.get("backend_paths", []),
                    "is_enabled": doc.get("is_enabled", False),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                    "matching_tools": matching_tools,
                }
                grouped_results["virtual_servers"].append(result_entry)

            else:
                grouped_results["custom"].append(_format_custom_result(doc, relevance_score))

        return grouped_results

    async def _default_search_scope(self) -> list[str]:
        """Default entity types for unscoped search.

        Built at query time so newly defined custom types appear in the
        "search everything" results. Custom type names come from the shared
        in-memory descriptor cache (never a per-query Mongo read); a stale
        cache only delays a brand-new type by the cache TTL.
        """
        scope = ["mcp_server", "a2a_agent", "skill", "virtual_server"]
        if not settings.custom_entity_types_enabled:
            return scope
        try:
            from ..factory import get_custom_type_repository

            descriptors = await get_custom_type_repository().cache.list_descriptors()
            scope.extend(d.name for d in descriptors)
        except Exception as e:
            logger.warning(f"Could not append custom types to search scope: {e}")
        return scope

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perform hybrid search (text + vector).

        Note: DocumentDB vector search returns results sorted by similarity
        but does NOT support $meta operators for score retrieval.
        We apply text-based boosting as a secondary ranking factor.
        """
        collection = await self._get_collection()

        try:
            # Try to get embedding; fall back to lexical-only search if
            # unavailable. _embed_texts() honors and updates the
            # _embedding_unavailable latch for us.
            vectors = await self._embed_texts([query], context="hybrid search query")
            query_embedding = vectors[0] if vectors else None

            if query_embedding is None:
                return await self._lexical_only_search(
                    query,
                    entity_types,
                    max_results,
                    include_draft=include_draft,
                    include_deprecated=include_deprecated,
                    include_disabled=include_disabled,
                )

            # DocumentDB vector search returns results sorted by similarity.
            # Run separate vector search per entity type to ensure each type
            # gets fair representation in the candidate pool (prevents servers
            # from crowding out agents/skills in large registries).
            ef_search = settings.vector_search_ef_search
            k_per_type = max(max_results * 2, 30)

            status_filter = _build_status_filter(
                include_draft=include_draft,
                include_deprecated=include_deprecated,
                include_disabled=include_disabled,
            )

            # Tokenize query and create regex pattern for matching any token
            query_tokens = _tokenize_query(query)
            escaped_tokens = [re.escape(token) for token in query_tokens]
            token_regex = "|".join(escaped_tokens) if escaped_tokens else query
            logger.info(
                "Hybrid search tokens for '%s': %s (regex: %s)",
                query,
                query_tokens,
                token_regex,
            )

            text_boost_stage = _build_text_boost_stage(token_regex)

            default_scope = await self._default_search_scope()
            search_types = [t for t in (entity_types or default_scope) if t != "tool"]

            results = []
            result_ids: set[str] = set()

            for search_type in search_types:
                pipeline: list[dict[str, Any]] = [
                    {
                        "$search": {
                            "vectorSearch": {
                                "vector": query_embedding,
                                "path": "embedding",
                                "similarity": "cosine",
                                "k": k_per_type,
                                "efSearch": ef_search,
                            }
                        }
                    },
                    {"$match": {"entity_type": search_type}},
                ]
                if status_filter:
                    pipeline.append({"$match": status_filter})
                pipeline.append(text_boost_stage)
                pipeline.append({"$sort": {"text_boost": -1}})
                pipeline.append({"$limit": k_per_type})

                cursor = collection.aggregate(pipeline)
                type_results = await cursor.to_list(length=k_per_type)

                for doc in type_results:
                    doc_id = doc.get("_id")
                    if doc_id not in result_ids:
                        results.append(doc)
                        result_ids.add(doc_id)

            logger.info(
                "Per-type vector search for '%s': %d total candidates "
                "(k_per_type=%d, efSearch=%d, types=%s)",
                query,
                len(results),
                k_per_type,
                ef_search,
                search_types,
            )

            # NOTE: DocumentDB does not support $unionWith, so we run a separate
            # keyword query and merge results in Python code after the main pipeline.
            keyword_match_filter = _build_keyword_match_filter(
                token_regex=token_regex,
                entity_types=entity_types,
            )

            keyword_cursor = collection.find(keyword_match_filter).limit(max(max_results, 10))
            keyword_results = await keyword_cursor.to_list(length=max(max_results, 10))

            logger.info(
                "Keyword search for '%s' found %d candidates",
                query,
                len(keyword_results),
            )
            for i, kw_doc in enumerate(keyword_results):
                already_in = kw_doc.get("_id") in {doc.get("_id") for doc in results}
                logger.info(
                    "  Keyword candidate [%d]: name='%s', type=%s, path='%s', already_in_vector=%s",
                    i,
                    kw_doc.get("name"),
                    kw_doc.get("entity_type"),
                    kw_doc.get("path"),
                    already_in,
                )

            # Merge keyword results with vector results, avoiding duplicates
            # Calculate text_boost and matching_tools for keyword results since they
            # didn't go through the aggregation pipeline
            keyword_added_count = 0
            for kw_doc in keyword_results:
                if kw_doc.get("_id") not in result_ids:
                    # Calculate text_boost for keyword-matched docs
                    # Use same weights as pipeline: path(+5), name(+3),
                    # description(+2), tags(+1.5), tools(+1 each)
                    kw_text_boost = 0.0
                    doc_name = (kw_doc.get("name") or "").lower()
                    doc_path = (kw_doc.get("path") or "").lower()
                    doc_desc = (kw_doc.get("description") or "").lower()
                    doc_tags = [(t or "").lower() for t in kw_doc.get("tags", [])]

                    for token in query_tokens:
                        token_lower = token.lower()
                        if token_lower in doc_path:
                            kw_text_boost += 5.0  # Path match
                        if token_lower in doc_name:
                            kw_text_boost += 3.0  # Name match
                        if token_lower in doc_desc:
                            kw_text_boost += 2.0  # Description match
                        if any(token_lower in tag for tag in doc_tags):
                            kw_text_boost += 1.5  # Tags match

                    # Calculate matching_tools for keyword-matched docs
                    tools = kw_doc.get("tools", [])
                    matching_tools = []
                    for tool in tools:
                        t_name = tool.get("name") or ""
                        t_desc = tool.get("description") or ""
                        tool_score = _score_tool_relevance(t_name, t_desc, query_tokens)
                        if tool_score > 0:
                            kw_text_boost += 1.0
                            matching_tools.append(
                                {
                                    "tool_name": t_name,
                                    "description": t_desc,
                                    "relevance_score": tool_score,
                                    "match_context": t_desc or f"Tool: {t_name}",
                                }
                            )

                    kw_doc["text_boost"] = kw_text_boost
                    kw_doc["matching_tools"] = matching_tools

                    results.append(kw_doc)
                    result_ids.add(kw_doc.get("_id"))
                    keyword_added_count += 1
                    logger.info(
                        "Keyword merge added '%s' (type=%s, text_boost=%.1f)",
                        kw_doc.get("name"),
                        kw_doc.get("entity_type"),
                        kw_text_boost,
                    )

            logger.info(
                "After keyword merge: %d total results (%d added from keyword search)",
                len(results),
                keyword_added_count,
            )

            # Combine vector and keyword signals using configured fusion method
            vector_ranked_docs = list(results)
            keyword_ranked_docs = sorted(
                [doc for doc in results if doc.get("text_boost", 0.0) > 0],
                key=lambda d: d.get("text_boost", 0.0),
                reverse=True,
            )

            if settings.search_fusion_method == "rrf":
                scored_results = _reciprocal_rank_fusion(vector_ranked_docs, keyword_ranked_docs)
                for doc, rrf_score in scored_results[:10]:
                    logger.info(
                        "RRF score for '%s' (type=%s): %.6f, text_boost=%.1f",
                        doc.get("name"),
                        doc.get("entity_type"),
                        rrf_score,
                        doc.get("text_boost", 0.0),
                    )
            else:
                scored_results = []
                for doc in results:
                    doc_embedding = doc.get("embedding", [])
                    vector_score = cosine_similarity(query_embedding, doc_embedding)
                    text_boost = doc.get("text_boost", 0.0)
                    normalized_vector_score = (vector_score + 1.0) / 2.0
                    text_boost_contribution = text_boost * 0.1
                    relevance_score = max(
                        0.0, min(1.0, normalized_vector_score + text_boost_contribution)
                    )
                    scored_results.append((doc, relevance_score))
                scored_results.sort(key=lambda x: x[1], reverse=True)

            selected_results = _distribute_results(scored_results, max_results)

            if settings.search_fusion_method == "rrf":
                selected_results = _normalize_scores(selected_results, max_results)

            # Group selected results by entity type for the response
            grouped_results: dict[str, list[dict[str, Any]]] = {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }
            tool_count = 0
            tool_limit = _tool_extraction_limit(max_results)

            for doc, relevance_score in selected_results:
                entity_type = doc.get("entity_type")

                if entity_type == "mcp_server":
                    matching_tools = doc.get("matching_tools", [])
                    server_metadata = doc.get("metadata", {})
                    result_entry = {
                        "entity_type": "mcp_server",
                        "path": doc.get("path"),
                        "server_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "num_tools": server_metadata.get("num_tools", 0),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "matching_tools": matching_tools,
                        "sync_metadata": server_metadata.get("sync_metadata"),
                        "proxy_pass_url": server_metadata.get("proxy_pass_url"),
                        "mcp_endpoint": server_metadata.get("mcp_endpoint"),
                        "sse_endpoint": server_metadata.get("sse_endpoint"),
                        "supported_transports": server_metadata.get("supported_transports", []),
                        "deployment": server_metadata.get("deployment", "remote"),
                        "local_runtime": server_metadata.get("local_runtime"),
                    }
                    grouped_results["servers"].append(result_entry)

                    # Also add matching tools to the top-level tools array
                    original_tools = doc.get("tools", [])
                    tool_schema_map = {
                        t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                    }

                    server_path = doc.get("path", "")
                    server_name = doc.get("name", "")
                    for tool in matching_tools:
                        if tool_count >= tool_limit:
                            break
                        tool_name = tool.get("tool_name", "")
                        grouped_results["tools"].append(
                            {
                                "entity_type": "tool",
                                "server_path": server_path,
                                "server_name": server_name,
                                "tool_name": tool_name,
                                "description": tool.get("description", ""),
                                "inputSchema": tool_schema_map.get(tool_name, {}),
                                "relevance_score": tool.get("relevance_score", 0.0),
                                "match_context": tool.get("match_context", ""),
                            }
                        )
                        tool_count += 1

                elif entity_type == "a2a_agent":
                    metadata = doc.get("metadata", {})
                    result_entry = {
                        "entity_type": "a2a_agent",
                        "path": doc.get("path"),
                        "agent_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "skills": metadata.get("skills", []),
                        "visibility": metadata.get("visibility", "public"),
                        "trust_level": metadata.get("trust_level"),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "agent_card": metadata.get("agent_card", {}),
                        "sync_metadata": metadata.get("sync_metadata"),
                    }
                    grouped_results["agents"].append(result_entry)

                elif entity_type == "mcp_tool":
                    result_entry = {
                        "entity_type": "mcp_tool",
                        "path": doc.get("path"),
                        "tool_name": doc.get("name"),
                        "description": doc.get("description"),
                        "inputSchema": doc.get("inputSchema", {}),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                    }
                    grouped_results["tools"].append(result_entry)

                elif entity_type == "skill":
                    metadata = doc.get("metadata", {})
                    result_entry = {
                        "entity_type": "skill",
                        "path": doc.get("path"),
                        "skill_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "skill_md_url": metadata.get("skill_md_url"),
                        "version": metadata.get("version"),
                        "author": metadata.get("author"),
                        "visibility": doc.get("visibility", "public"),
                        "owner": doc.get("owner"),
                        "is_enabled": doc.get("is_enabled", False),
                        "status": doc.get("status", "active"),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                    }
                    grouped_results["skills"].append(result_entry)

                elif entity_type == "virtual_server":
                    metadata = doc.get("metadata", {})
                    matching_tools = doc.get("matching_tools", [])
                    result_entry = {
                        "entity_type": "virtual_server",
                        "path": doc.get("path"),
                        "server_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "num_tools": metadata.get("num_tools", 0),
                        "backend_count": metadata.get("backend_count", 0),
                        "backend_paths": metadata.get("backend_paths", []),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "matching_tools": matching_tools,
                    }
                    grouped_results["virtual_servers"].append(result_entry)

                else:
                    # Custom entity types (admin-defined). Generic envelope —
                    # the schema-driven UI renders attributes from the descriptor.
                    grouped_results["custom"].append(_format_custom_result(doc, relevance_score))

            # Sort each group by relevance_score (descending) to ensure highest matches
            # appear first. This is needed because the DB sorts by text_boost only,
            # but relevance_score combines both vector similarity and text boost.
            grouped_results["servers"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            grouped_results["tools"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            grouped_results["agents"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            grouped_results["skills"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            grouped_results["virtual_servers"].sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            grouped_results["custom"].sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            logger.info(
                "Hybrid search for '%s' returned "
                "%d servers, %d tools, %d agents, %d skills, "
                "%d virtual_servers, %d custom (max_results=%d)",
                query,
                len(grouped_results["servers"]),
                len(grouped_results["tools"]),
                len(grouped_results["agents"]),
                len(grouped_results["skills"]),
                len(grouped_results["virtual_servers"]),
                len(grouped_results["custom"]),
                max_results,
            )

            return grouped_results

        except Exception as e:
            # Check if this is MongoDB CE without vector search support
            from pymongo.errors import OperationFailure

            if isinstance(e, OperationFailure) and (e.code == 31082 or "vectorSearch" in str(e)):
                # MongoDB CE doesn't support $vectorSearch - fall back to client-side search
                logger.warning(
                    "Vector search not supported (MongoDB CE detected). "
                    "Falling back to client-side cosine similarity search."
                )
                return await self._client_side_search(
                    query,
                    query_embedding,
                    entity_types,
                    max_results,
                    include_draft=include_draft,
                    include_deprecated=include_deprecated,
                    include_disabled=include_disabled,
                )
            elif "vectorSearch" in str(e) or "$search" in str(e):
                # General vector search not supported - fall back to client-side search
                logger.warning(
                    "Vector search not supported by this MongoDB instance. "
                    "Falling back to client-side cosine similarity search."
                )
                return await self._client_side_search(
                    query,
                    query_embedding,
                    entity_types,
                    max_results,
                    include_draft=include_draft,
                    include_deprecated=include_deprecated,
                    include_disabled=include_disabled,
                )

            logger.error(f"Failed to perform hybrid search: {e}", exc_info=True)
            return {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
                "custom": [],
            }
