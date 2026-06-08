"""Vector math helpers shared across the registry.

Cosine similarity is used by the DocumentDB hybrid search re-ranker
(``SearchRepository``). Kept here so any future caller imports from a
single canonical implementation.
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors.

    Returns 0.0 for any of: empty vectors, length mismatch, or a vector
    with zero magnitude. Otherwise, returns a value in ``[-1.0, 1.0]``;
    in practice both inputs are non-negative-ish (sentence-transformer
    or LiteLLM embeddings) and results land in roughly ``[0.0, 1.0]``.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
