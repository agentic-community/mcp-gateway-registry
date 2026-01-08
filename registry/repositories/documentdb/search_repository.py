"""DocumentDB-based repository for hybrid search (text + vector)."""

import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from ...core.config import embedding_config, settings
from ...schemas.agent_models import AgentCard
from ..interfaces import SearchRepositoryBase
from .client import get_collection_name, get_documentdb_client


logger = logging.getLogger(__name__)


class DocumentDBSearchRepository(SearchRepositoryBase):
    """DocumentDB implementation with hybrid search (text + vector)."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = get_collection_name(
            f"mcp_embeddings_{settings.embeddings_model_dimensions}"
        )
        self._embedding_model = None


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
        logger.info(
            f"Initializing DocumentDB hybrid search on collection: {self._collection_name}"
        )
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
                            "efConstruction": 128
                        }
                    )
                    logger.info("Created HNSW vector index")
                except Exception as vector_error:
                    # Check if this is a MongoDB CE error (vectorOptions not supported)
                    if "vectorOptions" in str(vector_error) or "not valid for an index specification" in str(vector_error):
                        logger.warning(
                            "Vector indexes not supported (MongoDB CE detected). "
                            "Creating regular index on embedding field."
                        )
                        # Create a regular index on the embedding field for faster retrieval
                        await collection.create_index(
                            [("embedding", 1)],
                            name="embedding_vector_idx"
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
        server_info: Dict[str, Any],
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

        text_for_embedding = " ".join(filter(None, text_parts))

        model = await self._get_embedding_model()
        embedding = model.encode([text_for_embedding])[0].tolist()

        doc = {
            "_id": path,
            "entity_type": "mcp_server",
            "path": path,
            "name": server_info.get("server_name", ""),
            "description": server_info.get("description", ""),
            "tags": server_info.get("tags", []),
            "is_enabled": is_enabled,
            "text_for_embedding": text_for_embedding,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "tools": [
                {"name": t.get("name"), "description": t.get("description")}
                for t in server_info.get("tool_list", [])
            ],
            "metadata": server_info,
            "indexed_at": server_info.get("updated_at", server_info.get("registered_at"))
        }

        try:
            await collection.replace_one(
                {"_id": path},
                doc,
                upsert=True
            )
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

        if agent_card.capabilities:
            text_parts.append("Capabilities: " + ", ".join(agent_card.capabilities))

        text_for_embedding = " ".join(filter(None, text_parts))

        model = await self._get_embedding_model()
        embedding = model.encode([text_for_embedding])[0].tolist()

        doc = {
            "_id": path,
            "entity_type": "a2a_agent",
            "path": path,
            "name": agent_card.name,
            "description": agent_card.description or "",
            "tags": agent_card.tags or [],
            "is_enabled": is_enabled,
            "text_for_embedding": text_for_embedding,
            "embedding": embedding,
            "embedding_metadata": embedding_config.get_embedding_metadata(),
            "capabilities": agent_card.capabilities or [],
            "metadata": agent_card.model_dump(mode="json"),
            "indexed_at": agent_card.updated_at or agent_card.registered_at
        }

        try:
            await collection.replace_one(
                {"_id": path},
                doc,
                upsert=True
            )
            logger.info(f"Indexed agent '{agent_card.name}' for search")
        except Exception as e:
            logger.error(f"Failed to index agent in search: {e}", exc_info=True)


    def _calculate_cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Returns a value between 0 and 1, where 1 is identical.
        """
        import math

        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


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
        query_embedding: List[float],
        entity_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
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
            cursor = collection.find(query_filter, {
                "_id": 1,
                "path": 1,
                "entity_type": 1,
                "name": 1,
                "description": 1,
                "tags": 1,
                "tools": 1,
                "metadata": 1,
                "is_enabled": 1,
                "embedding": 1
            })

            all_docs = await cursor.to_list(length=None)
            logger.info(f"Client-side search: Retrieved {len(all_docs)} documents with embeddings")

            # Calculate cosine similarity for each document
            scored_docs = []
            for doc in all_docs:
                embedding = doc.get("embedding", [])
                if not embedding:
                    continue

                # Calculate vector similarity
                vector_score = self._calculate_cosine_similarity(query_embedding, embedding)

                # Add text-based boost (same logic as server-side search)
                text_boost = 0.0
                name = doc.get("name", "")
                description = doc.get("description", "")
                tags = doc.get("tags", [])
                tools = doc.get("tools", [])
                query_lower = query.lower()
                matching_tools = []

                # Case-insensitive substring matching for text boost
                if name and query_lower in name.lower():
                    text_boost += 3.0
                if description and query_lower in description.lower():
                    text_boost += 2.0
                # Check if query matches any tag
                if tags and any(query_lower in tag.lower() for tag in tags):
                    text_boost += 1.5
                # Check if query matches any tool name or description
                for tool in tools:
                    tool_name = tool.get("name", "")
                    tool_desc = tool.get("description") or ""
                    if (tool_name and query_lower in tool_name.lower()) or \
                       (tool_desc and query_lower in tool_desc.lower()):
                        text_boost += 1.0
                        # Store full tool object for frontend
                        matching_tools.append({
                            "tool_name": tool_name,
                            "description": tool_desc,
                            "relevance_score": 1.0,  # Tool matched, full score
                            "match_context": tool_desc or f"Tool: {tool_name}"
                        })

                # Store matching tools for later use
                doc["_matching_tools"] = matching_tools

                # Hybrid score: vector score + normalized text boost
                # Normalize vector_score to [0, 1] range (cosine can be [-1, 1])
                normalized_vector_score = (vector_score + 1.0) / 2.0
                relevance_score = normalized_vector_score + (text_boost * 0.03)
                relevance_score = max(0.0, min(1.0, relevance_score))

                scored_docs.append({
                    "doc": doc,
                    "relevance_score": relevance_score,
                    "vector_score": vector_score,
                    "text_boost": text_boost
                })

            # Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Separate by entity type and take top 3 of each
            servers = []
            agents = []
            tools = []

            for item in scored_docs:
                doc = item["doc"]
                entity_type = doc.get("entity_type")

                if entity_type == "mcp_server" and len(servers) < 3:
                    servers.append(item)
                elif entity_type == "a2a_agent" and len(agents) < 3:
                    agents.append(item)
                elif entity_type == "mcp_tool" and len(tools) < 3:
                    tools.append(item)

            # Format results to match the API contract
            grouped_results = {"servers": [], "tools": [], "agents": []}

            for item in servers:
                doc = item["doc"]
                relevance_score = item["relevance_score"]

                result_entry = {
                    "entity_type": "mcp_server",
                    "path": doc.get("path"),
                    "server_name": doc.get("name"),
                    "description": doc.get("description"),
                    "tags": doc.get("tags", []),
                    "num_tools": doc.get("metadata", {}).get("num_tools", 0),
                    "is_enabled": doc.get("is_enabled", False),
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description"),
                    "matching_tools": doc.get("_matching_tools", [])
                }
                grouped_results["servers"].append(result_entry)

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
                    "agent_card": metadata.get("agent_card", {})
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
                    "relevance_score": relevance_score,
                    "match_context": doc.get("description")
                }
                grouped_results["tools"].append(result_entry)

            logger.info(
                f"Client-side search returned "
                f"{len(grouped_results['servers'])} servers, "
                f"{len(grouped_results['tools'])} tools, "
                f"{len(grouped_results['agents'])} agents "
                f"from {len(all_docs)} total documents (top 3 per type)"
            )

            return grouped_results

        except Exception as e:
            logger.error(f"Failed to perform client-side search: {e}", exc_info=True)
            return {"servers": [], "tools": [], "agents": []}


    async def search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform hybrid search (text + vector).

        Note: DocumentDB vector search returns results sorted by similarity
        but does NOT support $meta operators for score retrieval.
        We apply text-based boosting as a secondary ranking factor.
        """
        collection = await self._get_collection()

        try:
            model = await self._get_embedding_model()
            query_embedding = model.encode([query])[0].tolist()

            # DocumentDB vector search returns results sorted by similarity
            # We get more results than needed to allow for text-based re-ranking
            pipeline = [
                {
                    "$search": {
                        "vectorSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "similarity": "cosine",
                            "k": max_results * 3  # Get 3x results for re-ranking
                        }
                    }
                }
            ]

            # Apply entity type filter if specified
            if entity_types:
                pipeline.append({"$match": {"entity_type": {"$in": entity_types}}})

            # Add text-based scoring for re-ranking
            # Higher scores for matches in name (3.0), description (2.0), tags (1.5), tools (1.0 per match)
            pipeline.append({
                "$addFields": {
                    "text_boost": {
                        "$add": [
                            # Name match: 3.0
                            {
                                "$cond": [
                                    {
                                        "$regexMatch": {
                                            "input": {"$ifNull": ["$name", ""]},
                                            "regex": query,
                                            "options": "i"
                                        }
                                    },
                                    3.0,
                                    0.0
                                ]
                            },
                            # Description match: 2.0
                            {
                                "$cond": [
                                    {
                                        "$regexMatch": {
                                            "input": {"$ifNull": ["$description", ""]},
                                            "regex": query,
                                            "options": "i"
                                        }
                                    },
                                    2.0,
                                    0.0
                                ]
                            },
                            # Tags match: 1.5 if any tag matches
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
                                                                "regex": query,
                                                                "options": "i"
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            0
                                        ]
                                    },
                                    1.5,
                                    0.0
                                ]
                            },
                            # Tools match: 1.0 per matching tool (check name and description)
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
                                                        "regex": query,
                                                        "options": "i"
                                                    }
                                                },
                                                {
                                                    "$regexMatch": {
                                                        "input": {"$ifNull": ["$$tool.description", ""]},
                                                        "regex": query,
                                                        "options": "i"
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            }
                        ]
                    },
                    # Also track matching tools for display
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
                                                    "regex": query,
                                                    "options": "i"
                                                }
                                            },
                                            {
                                                "$regexMatch": {
                                                    "input": {"$ifNull": ["$$tool.description", ""]},
                                                    "regex": query,
                                                    "options": "i"
                                                }
                                            }
                                        ]
                                    }
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
                                        {"$concat": ["Tool: ", "$$tool.name"]}
                                    ]
                                }
                            }
                        }
                    }
                }
            })

            # Sort by text boost (descending), keeping vector search order as secondary
            pipeline.append({"$sort": {"text_boost": -1}})

            # Limit to requested number of results
            pipeline.append({"$limit": max_results})

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=max_results)

            # Return results with keys matching the API contract (same as FAISS service)
            # Calculate cosine similarity scores manually since DocumentDB doesn't expose them
            # Limit to top 3 per entity type
            grouped_results = {"servers": [], "tools": [], "agents": []}
            server_count = 0
            agent_count = 0
            tool_count = 0

            for doc in results:
                entity_type = doc.get("entity_type")

                # Skip if we already have 3 of this type
                if entity_type == "mcp_server" and server_count >= 3:
                    continue
                elif entity_type == "a2a_agent" and agent_count >= 3:
                    continue
                elif entity_type == "mcp_tool" and tool_count >= 3:
                    continue

                # Calculate actual cosine similarity from embeddings
                doc_embedding = doc.get("embedding", [])
                vector_score = self._calculate_cosine_similarity(query_embedding, doc_embedding)

                # Get text boost (already calculated in pipeline)
                text_boost = doc.get("text_boost", 0.0)

                # Hybrid score: Keep DocumentDB's vector ranking but show the actual similarity
                # Text boost adds a small bonus if there are keyword matches
                # Vector score is 0-1, text_boost can be 0-6.5 (3+2+1.5 max), so normalize text_boost to 0-0.195 range
                # Normalize vector_score to [0, 1] range (cosine can be [-1, 1])
                normalized_vector_score = (vector_score + 1.0) / 2.0
                relevance_score = normalized_vector_score + (text_boost * 0.03)
                relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to [0, 1]

                if entity_type == "mcp_server":
                    result_entry = {
                        "entity_type": "mcp_server",
                        "path": doc.get("path"),
                        "server_name": doc.get("name"),
                        "description": doc.get("description"),
                        "tags": doc.get("tags", []),
                        "num_tools": doc.get("metadata", {}).get("num_tools", 0),
                        "is_enabled": doc.get("is_enabled", False),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description"),
                        "matching_tools": doc.get("matching_tools", [])
                    }
                    grouped_results["servers"].append(result_entry)
                    server_count += 1

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
                        "agent_card": metadata.get("agent_card", {})
                    }
                    grouped_results["agents"].append(result_entry)
                    agent_count += 1

                elif entity_type == "mcp_tool":
                    result_entry = {
                        "entity_type": "mcp_tool",
                        "path": doc.get("path"),
                        "tool_name": doc.get("name"),
                        "description": doc.get("description"),
                        "relevance_score": relevance_score,
                        "match_context": doc.get("description")
                    }
                    grouped_results["tools"].append(result_entry)
                    tool_count += 1

            logger.info(
                f"Hybrid search for '{query}' returned "
                f"{len(grouped_results['servers'])} servers, "
                f"{len(grouped_results['tools'])} tools, "
                f"{len(grouped_results['agents'])} agents (top 3 per type)"
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
            return {"servers": [], "tools": [], "agents": []}
