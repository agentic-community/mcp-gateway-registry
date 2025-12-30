from __future__ import annotations

import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)

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


def _distance_to_relevance(
    distance: float,
) -> float:
    """Convert FAISS Inner Product distance to cosine similarity score (0-1)."""
    try:
        dist = float(distance)

        if dist < 0:
            similarity = -dist
        else:
            similarity = 1.0 - dist

        clamped_similarity = max(0.0, min(1.0, similarity))

        logger.info(
            "IP-to-similarity conversion: faiss_distance=%.4f, similarity=%.4f, clamped=%.4f, percentage=%.1f%%",
            distance,
            similarity,
            clamped_similarity,
            clamped_similarity * 100.0,
        )

        return clamped_similarity
    except Exception as exc:
        logger.error(
            "Error in _distance_to_relevance: faiss_distance=%s, exception=%s",
            distance,
            str(exc),
            exc_info=True,
        )
        return 0.0


def _normalize_embedding(
    embedding: np.ndarray,
) -> np.ndarray:
    """Normalize embedding vector to unit length for cosine similarity."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        logger.warning("Zero-norm embedding detected, returning as-is")
        return embedding
    return embedding / norm


def _calculate_keyword_boost(
    query: str,
    server_info: Dict[str, Any],
) -> float:
    """Calculate keyword match boost for hybrid search."""
    query_lower = query.lower()
    query_tokens = {
        token
        for token in re.split(r"\W+", query_lower)
        if token and len(token) > 2 and token not in _STOPWORDS
    }

    if not query_tokens:
        return 1.0

    boost = 1.0
    boost_reasons: List[str] = []

    server_name = server_info.get("server_name", "").lower()
    if any(token in server_name for token in query_tokens):
        boost += 0.5
        boost_reasons.append(f"name({server_name}):+0.5")

    tools = server_info.get("tool_list") or []
    tool_matches = 0
    matching_tool_names = []
    for tool in tools:
        tool_name = tool.get("name", "").lower()
        if any(token in tool_name for token in query_tokens):
            tool_matches += 1
            matching_tool_names.append(tool_name)

    tool_boost = min(0.6, tool_matches * 0.3)
    if tool_boost > 0:
        boost += tool_boost
        boost_reasons.append(
            f"tools({','.join(matching_tool_names[:2])}):+{tool_boost:.1f}"
        )

    tags = server_info.get("tags", [])
    tag_matches = sum(
        1 for tag in tags if any(token in tag.lower() for token in query_tokens)
    )
    tag_boost = min(0.4, tag_matches * 0.2)
    if tag_boost > 0:
        boost += tag_boost
        boost_reasons.append(f"tags:{tag_matches}:+{tag_boost:.1f}")

    description = server_info.get("description", "").lower()
    if description:
        desc_matches = sum(1 for token in query_tokens if token in description)
        match_ratio = desc_matches / len(query_tokens)
        desc_boost = match_ratio * 0.2
        if desc_boost > 0.01:
            boost += desc_boost
            boost_reasons.append(f"desc:{desc_matches}/{len(query_tokens)}:+{desc_boost:.2f}")

    if boost_reasons:
        logger.info("  Keyword boost breakdown: %s", " | ".join(boost_reasons))

    return min(2.0, boost)


def _extract_matching_tools(
    query: str,
    server_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract tool matches using simple keyword overlap."""
    tools = server_info.get("tool_list") or []
    if not tools:
        return []

    tokens = [
        token
        for token in re.split(r"\W+", query.lower())
        if token and len(token) > 2 and token not in _STOPWORDS
    ]
    if not tokens:
        return []

    matches: List[Tuple[float, Dict[str, Any]]] = []
    for tool in tools:
        tool_name = tool.get("name", "")
        parsed_description = tool.get("parsed_description", {}) or {}
        tool_desc = (
            parsed_description.get("main")
            or tool.get("description")
            or parsed_description.get("summary")
            or ""
        )
        tool_args = parsed_description.get("args") or ""

        tool_name = tool_name or ""
        tool_desc = tool_desc or ""
        tool_args = tool_args or ""

        searchable_text = f"{tool_name} {tool_desc} {tool_args}".lower()
        if not searchable_text.strip():
            continue

        tool_name_lower = tool_name.lower()
        name_matches = sum(1 for token in tokens if token in tool_name_lower)
        desc_matches = sum(
            1
            for token in tokens
            if token in tool_desc.lower() or token in tool_args.lower()
        )

        weighted_matches = (name_matches * 2.0) + desc_matches
        max_possible_score = len(tokens) * 2.0

        if weighted_matches == 0:
            continue

        coverage = min(1.0, weighted_matches / max_possible_score)
        matches.append(
            (
                coverage,
                {
                    "tool_name": tool_name,
                    "description": tool_desc,
                    "match_context": (tool_desc or tool_args or "")[:180],
                    "schema": tool.get("schema", {}),
                    "raw_score": coverage,
                },
            )
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    return [match for _, match in matches]
