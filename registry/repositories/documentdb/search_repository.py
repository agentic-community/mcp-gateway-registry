"""DocumentDB-based repository for hybrid search (text + vector)."""

import logging
import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ...core.config import embedding_config, settings
from ...schemas.agent_models import AgentCard
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


def _flatten_metadata_to_text(metadata: dict[str, Any]) -> str:
    """Flatten a metadata dict into a searchable text string.

    Handles nested lists and dicts by joining their string values.
    Example: {"team": "myteam", "langs": ["python", "go"]}
    becomes: "team myteam langs python go"
    """
    if not isinstance(metadata, dict) or not metadata:
        return ""
    parts = []
    for key, value in metadata.items():
        parts.append(str(key))
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(v) for v in value.values())
        else:
            parts.append(str(value))
    return " ".join(parts)


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
    match_filter = {
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
    ) -> None:
        """Index a server for search."""
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

        # Flatten metadata into a searchable text field for keyword matching
        metadata_text = _flatten_metadata_to_text(metadata)

        try:
            model = await self._get_embedding_model()
            embedding = model.encode([text_for_embedding])[0].tolist()
        except Exception as e:
            logger.warning(
                "Embedding model unavailable, indexing '%s' without embeddings: %s",
                server_info.get("server_name", path),
                e,
            )
            embedding = []

        doc = {
            "_id": path,
            "entity_type": "mcp_server",
            "path": path,
            "name": server_info.get("server_name", ""),
            "description": server_info.get("description", ""),
            "tags": server_info.get("tags", []),
            "metadata_text": metadata_text,
            "is_enabled": is_enabled,
            "text_for_embedding": text_for_embedding,
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
    ) -> None:
        """Index an agent for search."""
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

        try:
            model = await self._get_embedding_model()
            embedding = model.encode([text_for_embedding])[0].tolist()
        except Exception as e:
            logger.warning(
                "Embedding model unavailable, indexing agent '%s' without embeddings: %s",
                agent_card.name,
                e,
            )
            embedding = []

        # Flatten agent metadata for keyword search
        agent_metadata = getattr(agent_card, "metadata", None) or {}
        agent_metadata_text = _flatten_metadata_to_text(agent_metadata)

        doc = {
            "_id": path,
            "entity_type": "a2a_agent",
            "path": path,
            "name": agent_card.name,
            "description": agent_card.description or "",
            "tags": agent_card.tags or [],
            "metadata_text": agent_metadata_text,
            "is_enabled": is_enabled,
            "text_for_embedding": text_for_embedding,
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
    ) -> None:
        """Index a skill for semantic search.

        Args:
            path: Skill path (e.g., /skills/pdf-processing)
            skill: SkillCard object
            is_enabled: Whether skill is enabled
        """
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
            extra_text = _flatten_metadata_to_text(skill.metadata.extra)
            if extra_text:
                text_parts.append(extra_text)

        text_for_embedding = " ".join(filter(None, text_parts))

        # Generate embedding
        try:
            model = await self._get_embedding_model()
            embedding = model.encode([text_for_embedding])[0].tolist()
        except Exception as e:
            logger.warning(
                "Embedding model unavailable, indexing skill '%s' without embeddings: %s",
                skill.name,
                e,
            )
            embedding = []

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
            extra_text = _flatten_metadata_to_text(skill.metadata.extra)
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
            "text_for_embedding": text_for_embedding,
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

        # Generate embedding
        try:
            model = await self._get_embedding_model()
            embedding = model.encode([text_for_embedding])[0].tolist()
        except Exception as e:
            logger.warning(
                "Embedding model unavailable, indexing virtual server '%s' without embeddings: %s",
                virtual_server.server_name,
                e,
            )
            embedding = []

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

    def _calculate_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Returns a value between 0 and 1, where 1 is identical.
        """
        import math

        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def search_by_tags(
        self,
        tags: list[str],
        entity_types: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search entities by exact tag match using a direct DB query."""
        collection = await self._get_collection()

        # Build a case-insensitive match for ALL tags
        tag_conditions = [
            {"tags": {"$regex": f"^{re.escape(tag)}$", "$options": "i"}} for tag in tags
        ]
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
        return self._format_lexical_results(results)

    async def get_all_tags(self) -> list[str]:
        """Return a sorted list of all unique tags across all indexed entities."""
        collection = await self._get_collection()
        try:
            pipeline = [
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
    ) -> None:
        """Remove entity from search index."""
        collection = await self._get_collection()

        try:
            result = await collection.delete_one({"_id": path})
            if result.deleted_count > 0:
                logger.info(f"Removed entity '{path}' from search index")
            else:
                logger.warning(f"Entity '{path}' not found in search index")
        except Exception as e:
            logger.error(f"Failed to remove entity from search index: {e}", exc_info=True)

    async def _client_side_search(
        self,
        query: str,
        query_embedding: list[float],
        entity_types: list[str] | None = None,
        max_results: int = 10,
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
                    "embedding": 1,
                },
            )

            all_docs = await cursor.to_list(length=None)
            logger.info(f"Client-side search: Retrieved {len(all_docs)} documents with embeddings")

            # Tokenize query for keyword matching
            query_tokens = _tokenize_query(query)
            logger.debug(f"Client-side search tokens: {query_tokens}")

            # Calculate cosine similarity for each document
            scored_docs = []
            for doc in all_docs:
                embedding = doc.get("embedding", [])
                if not embedding:
                    vector_score = 0.0
                else:
                    vector_score = self._calculate_cosine_similarity(query_embedding, embedding)

                # Add text-based boost using tokenized matching
                text_boost = 0.0
                name = doc.get("name", "")
                description = doc.get("description", "")
                tags = doc.get("tags", [])
                tools = doc.get("tools", [])
                matching_tools = []

                # Token-based matching for text boost
                # Check path match first (highest priority - user explicitly named the server)
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
                # Check if any token matches any tag
                if tags and any(_tokens_match_text(query_tokens, tag) for tag in tags):
                    text_boost += 1.5

                # Check metadata_text match
                metadata_text = doc.get("metadata_text", "")
                if metadata_text and _tokens_match_text(query_tokens, metadata_text):
                    text_boost += 1.0

                # Check if any token matches any tool name or description
                for tool in tools:
                    tool_name = tool.get("name", "")
                    tool_desc = tool.get("description") or ""
                    tool_matched = _tokens_match_text(
                        query_tokens, tool_name
                    ) or _tokens_match_text(query_tokens, tool_desc)

                    if tool_matched:
                        text_boost += 1.0
                        matching_tools.append(
                            {
                                "tool_name": tool_name,
                                "description": tool_desc,
                                "relevance_score": 1.0,
                                "match_context": tool_desc or f"Tool: {tool_name}",
                            }
                        )
                    elif server_name_matched:
                        # If server name/path matched, include all tools with base score
                        matching_tools.append(
                            {
                                "tool_name": tool_name,
                                "description": tool_desc,
                                "relevance_score": 0.8,
                                "match_context": tool_desc or f"Tool: {tool_name}",
                            }
                        )

                # Store matching tools for later use
                doc["_matching_tools"] = matching_tools

                # Hybrid score: vector score + normalized text boost
                # Normalize vector_score to [0, 1] range (cosine can be [-1, 1])
                normalized_vector_score = (vector_score + 1.0) / 2.0
                # Text boost multiplier: 0.1 (same as DocumentDB search path)
                # Path match (5.0) adds +0.50, Name match (3.0) adds +0.30
                text_boost_contribution = text_boost * 0.1
                relevance_score = normalized_vector_score + text_boost_contribution
                relevance_score = max(0.0, min(1.0, relevance_score))

                logger.info(
                    "Score for '%s' (type=%s): vector=%.4f, "
                    "normalized_vector=%.4f, text_boost=%.1f, "
                    "boost_contrib=%.4f, final=%.4f",
                    doc.get("name"),
                    doc.get("entity_type"),
                    vector_score,
                    normalized_vector_score,
                    text_boost,
                    text_boost_contribution,
                    relevance_score,
                )

                scored_docs.append(
                    {
                        "doc": doc,
                        "relevance_score": relevance_score,
                        "vector_score": vector_score,
                        "text_boost": text_boost,
                    }
                )

            # Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Separate by entity type and take top 3 of each
            servers = []
            agents = []
            tools = []
            skills = []
            virtual_servers = []

            for item in scored_docs:
                doc = item["doc"]
                entity_type = doc.get("entity_type")

                if entity_type == "mcp_server" and len(servers) < 3:
                    servers.append(item)
                elif entity_type == "a2a_agent" and len(agents) < 3:
                    agents.append(item)
                elif entity_type == "mcp_tool" and len(tools) < 3:
                    tools.append(item)
                elif entity_type == "skill" and len(skills) < 3:
                    skills.append(item)
                elif entity_type == "virtual_server" and len(virtual_servers) < 3:
                    virtual_servers.append(item)

            # Format results to match the API contract
            grouped_results = {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
            }

            tool_count = 0
            for item in servers:
                doc = item["doc"]
                relevance_score = item["relevance_score"]
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
                }
                grouped_results["servers"].append(result_entry)

                # Also add matching tools to the top-level tools array
                # Build a lookup map from tool name to inputSchema from original tools
                original_tools = doc.get("tools", [])
                tool_schema_map = {
                    t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                }

                server_path = doc.get("path", "")
                server_name = doc.get("name", "")
                for tool in matching_tools:
                    if tool_count >= 3:
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
                            "relevance_score": tool.get("relevance_score", relevance_score),
                            "match_context": tool.get("match_context", ""),
                        }
                    )
                    tool_count += 1

            for item in agents:
                doc = item["doc"]
                relevance_score = item["relevance_score"]
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

            for item in tools:
                doc = item["doc"]
                relevance_score = item["relevance_score"]

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

            for item in skills:
                doc = item["doc"]
                relevance_score = item["relevance_score"]
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
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                }
                grouped_results["skills"].append(result_entry)

            for item in virtual_servers:
                doc = item["doc"]
                relevance_score = item["relevance_score"]
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

            logger.info(
                f"Client-side search returned "
                f"{len(grouped_results['servers'])} servers, "
                f"{len(grouped_results['tools'])} tools, "
                f"{len(grouped_results['agents'])} agents, "
                f"{len(grouped_results['skills'])} skills, "
                f"{len(grouped_results['virtual_servers'])} virtual_servers "
                f"from {len(all_docs)} total documents (top 3 per type)"
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
            }

    async def _lexical_only_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fallback search using keyword matching only (no embeddings).

        Used when the embedding model fails to load. Scores results purely
        by keyword matches against name, path, description, tags, and tools.

        Args:
            query: The search query string
            entity_types: Optional list of entity types to filter
            max_results: Maximum number of results to return

        Returns:
            Grouped search results dict with servers, tools, agents lists
        """
        collection = await self._get_collection()
        query_tokens = _tokenize_query(query)

        if not query_tokens:
            logger.info("Lexical search: no valid tokens from query '%s'", query)
            return {"servers": [], "tools": [], "agents": [], "skills": []}

        escaped_tokens = [re.escape(token) for token in query_tokens]
        token_regex = "|".join(escaped_tokens)

        keyword_match_filter = _build_keyword_match_filter(
            token_regex=token_regex,
            entity_types=entity_types,
        )

        text_boost_stage = _build_text_boost_stage(token_regex)

        pipeline = [
            {"$match": keyword_match_filter},
            text_boost_stage,
            {"$sort": {"text_boost": -1}},
            {"$limit": max_results},
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=max_results)

        grouped_results = self._format_lexical_results(results)

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
    ) -> dict[str, list[dict[str, Any]]]:
        """Format lexical search results into grouped response.

        Uses fixed-denominator normalization for relevance scoring.

        Args:
            results: Raw MongoDB documents with text_boost field

        Returns:
            Grouped search results dict with servers, tools, agents lists
        """
        grouped_results = {
            "servers": [],
            "tools": [],
            "agents": [],
            "skills": [],
            "virtual_servers": [],
        }
        server_count = 0
        agent_count = 0
        tool_count = 0
        skill_count = 0
        virtual_server_count = 0

        for doc in results:
            entity_type = doc.get("entity_type")
            text_boost = doc.get("text_boost", 0.0)
            relevance_score = min(1.0, text_boost / MAX_LEXICAL_BOOST)

            if entity_type == "mcp_server" and server_count < 3:
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
                server_count += 1

                # Add matching tools to top-level tools array
                original_tools = doc.get("tools", [])
                tool_schema_map = {
                    t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                }
                server_path = doc.get("path", "")
                server_name = doc.get("name", "")
                for tool in matching_tools:
                    if tool_count >= 3:
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
                            "relevance_score": tool.get("relevance_score", relevance_score),
                            "match_context": tool.get("match_context", ""),
                        }
                    )
                    tool_count += 1

            elif entity_type == "a2a_agent" and agent_count < 3:
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
                agent_count += 1

            elif entity_type == "mcp_tool" and tool_count < 3:
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
                tool_count += 1

            elif entity_type == "skill" and skill_count < 3:
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
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                }
                grouped_results["skills"].append(result_entry)
                skill_count += 1

            elif entity_type == "virtual_server" and virtual_server_count < 3:
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
                virtual_server_count += 1

        return grouped_results

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perform hybrid search (text + vector).

        Note: DocumentDB vector search returns results sorted by similarity
        but does NOT support $meta operators for score retrieval.
        We apply text-based boosting as a secondary ranking factor.
        """
        collection = await self._get_collection()

        try:
            # Try to get embedding; fall back to lexical-only search if unavailable
            query_embedding = None
            if not self._embedding_unavailable:
                try:
                    model = await self._get_embedding_model()
                    query_embedding = model.encode([query])[0].tolist()
                except Exception as embed_error:
                    logger.warning(
                        "Embedding model unavailable, falling back to lexical-only search: %s",
                        embed_error,
                    )
                    self._embedding_unavailable = True

            if query_embedding is None:
                return await self._lexical_only_search(query, entity_types, max_results)

            # DocumentDB vector search returns results sorted by similarity
            # We get more results than needed to allow for text-based re-ranking
            ef_search = settings.vector_search_ef_search
            k_value = max(max_results * 3, 50)  # At least 50 to avoid missing docs
            pipeline = [
                {
                    "$search": {
                        "vectorSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "similarity": "cosine",
                            "k": k_value,
                            "efSearch": ef_search,
                        }
                    }
                }
            ]
            logger.info(
                "Vector search pipeline: k=%d, efSearch=%d",
                k_value,
                ef_search,
            )

            # Apply entity type filter if specified
            if entity_types:
                pipeline.append({"$match": {"entity_type": {"$in": entity_types}}})

            # Tokenize query and create regex pattern for matching any token
            query_tokens = _tokenize_query(query)
            # Create regex that matches any token (e.g., "current|time|timezone")
            # Escape special regex characters in tokens for safety
            escaped_tokens = [re.escape(token) for token in query_tokens]
            token_regex = "|".join(escaped_tokens) if escaped_tokens else query
            logger.info(
                "Hybrid search tokens for '%s': %s (regex: %s)",
                query,
                query_tokens,
                token_regex,
            )

            # NOTE: DocumentDB does not support $unionWith, so we run a separate
            # keyword query and merge results in Python code after the main pipeline.
            # Reuse shared helper for consistent matching across all fields
            keyword_match_filter = _build_keyword_match_filter(
                token_regex=token_regex,
                entity_types=entity_types,
            )

            # Add text-based scoring for re-ranking using shared helper
            # Scores: path (+5.0), name (+3.0), description (+2.0),
            # tags (+1.5), tools (+1.0 per match)
            text_boost_stage = _build_text_boost_stage(token_regex)
            pipeline.append(text_boost_stage)

            # Sort by text boost (descending), keeping vector search order as secondary
            pipeline.append({"$sort": {"text_boost": -1}})

            # Limit to requested number of results
            pipeline.append({"$limit": max_results})

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=max_results)

            # Log vector search results for diagnosis
            logger.info(
                "Vector search for '%s' returned %d documents (k=%d, efSearch=%d)",
                query,
                len(results),
                k_value,
                ef_search,
            )
            for i, doc in enumerate(results):
                logger.info(
                    "  Vector result [%d]: name='%s', type=%s, text_boost=%.1f, path='%s'",
                    i,
                    doc.get("name"),
                    doc.get("entity_type"),
                    doc.get("text_boost", 0.0),
                    doc.get("path"),
                )

            # DocumentDB doesn't support $unionWith, so we run a separate keyword
            # query to find documents that match by name/path/description/tags/tools
            # but may not appear in vector search results
            keyword_cursor = collection.find(keyword_match_filter).limit(5)
            keyword_results = await keyword_cursor.to_list(length=5)

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
            result_ids = {doc.get("_id") for doc in results}
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
                        tool_name = (tool.get("name") or "").lower()
                        tool_desc = (tool.get("description") or "").lower()
                        # Check if any token matches tool name or description
                        tool_matches = any(
                            token.lower() in tool_name or token.lower() in tool_desc
                            for token in query_tokens
                        )
                        if tool_matches:
                            kw_text_boost += 1.0  # Tool match
                            matching_tools.append(
                                {
                                    "tool_name": tool.get("name", ""),
                                    "description": tool.get("description", ""),
                                    "relevance_score": 1.0,
                                    "match_context": tool.get("description")
                                    or f"Tool: {tool.get('name', '')}",
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

            # Calculate hybrid scores for ALL results before grouping
            # This ensures we log every document's score for diagnosis
            scored_results = []
            for doc in results:
                entity_type = doc.get("entity_type")
                doc_embedding = doc.get("embedding", [])
                vector_score = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                text_boost = doc.get("text_boost", 0.0)

                # Normalize vector_score from [-1, 1] to [0, 1]
                normalized_vector_score = (vector_score + 1.0) / 2.0

                # Text boost multiplier: 0.1 gives significant weight to keyword matches
                # Name match (3.0) adds +0.30, Description (2.0) adds +0.20
                text_boost_contribution = text_boost * 0.1
                relevance_score = normalized_vector_score + text_boost_contribution
                relevance_score = max(0.0, min(1.0, relevance_score))

                logger.info(
                    "Score for '%s' (type=%s): vector=%.4f, "
                    "normalized_vector=%.4f, text_boost=%.1f, "
                    "boost_contrib=%.4f, final=%.4f",
                    doc.get("name"),
                    entity_type,
                    vector_score,
                    normalized_vector_score,
                    text_boost,
                    text_boost_contribution,
                    relevance_score,
                )

                scored_results.append((doc, relevance_score))

            # Sort by hybrid score descending so top-3 picks the best
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Group results: top 3 per entity type
            grouped_results = {
                "servers": [],
                "tools": [],
                "agents": [],
                "skills": [],
                "virtual_servers": [],
            }
            server_count = 0
            agent_count = 0
            tool_count = 0
            skill_count = 0
            virtual_server_count = 0

            for doc, relevance_score in scored_results:
                entity_type = doc.get("entity_type")

                # Skip if we already have 3 of this type
                if entity_type == "mcp_server" and server_count >= 3:
                    continue
                elif entity_type == "a2a_agent" and agent_count >= 3:
                    continue
                elif entity_type == "mcp_tool" and tool_count >= 3:
                    continue
                elif entity_type == "skill" and skill_count >= 3:
                    continue
                elif entity_type == "virtual_server" and virtual_server_count >= 3:
                    continue

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
                    }
                    grouped_results["servers"].append(result_entry)
                    server_count += 1

                    # Also add matching tools to the top-level tools array
                    # Build a lookup map from tool name to inputSchema from original tools
                    original_tools = doc.get("tools", [])
                    tool_schema_map = {
                        t.get("name", ""): t.get("inputSchema", {}) for t in original_tools
                    }

                    server_path = doc.get("path", "")
                    server_name = doc.get("name", "")
                    for tool in matching_tools:
                        if tool_count >= 3:
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
                                "relevance_score": tool.get("relevance_score", relevance_score),
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
                    agent_count += 1

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
                    tool_count += 1

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
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                    }
                    grouped_results["skills"].append(result_entry)
                    skill_count += 1

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
                    virtual_server_count += 1

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

            logger.info(
                f"Hybrid search for '{query}' returned "
                f"{len(grouped_results['servers'])} servers, "
                f"{len(grouped_results['tools'])} tools, "
                f"{len(grouped_results['agents'])} agents, "
                f"{len(grouped_results['skills'])} skills, "
                f"{len(grouped_results['virtual_servers'])} virtual_servers (top 3 per type)"
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
                    query, query_embedding, entity_types, max_results
                )
            elif "vectorSearch" in str(e) or "$search" in str(e):
                # General vector search not supported - fall back to client-side search
                logger.warning(
                    "Vector search not supported by this MongoDB instance. "
                    "Falling back to client-side cosine similarity search."
                )
                return await self._client_side_search(
                    query, query_embedding, entity_types, max_results
                )

            logger.error(f"Failed to perform hybrid search: {e}", exc_info=True)
            return {"servers": [], "tools": [], "agents": [], "skills": []}
