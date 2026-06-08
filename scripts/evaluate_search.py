"""Evaluate search scoring methods against ground truth using NDCG@10.

Self-contained evaluation harness that loads the unified dataset (with
embeddings), runs both RRF and legacy scoring offline, and measures
quality using Normalized Discounted Cumulative Gain (NDCG@10).

No server required. Uses the same scoring logic as the production code.

Usage:
    uv run python scripts/evaluate_search.py

    # Show per-query details
    uv run python scripts/evaluate_search.py --verbose

    # Only run one method
    uv run python scripts/evaluate_search.py --method rrf
    uv run python scripts/evaluate_search.py --method legacy
"""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "tests/fixtures/search_dataset/unified_dataset.json"
GROUND_TRUTH_PATH = Path(__file__).parent.parent / "tests/fixtures/search_dataset/ground_truth.json"

RRF_K: int = 60

# Stopwords (same as production)
_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "as", "into", "through", "from", "what",
    "when", "where", "who", "which", "how", "why", "get", "set", "put",
}


# =============================================================================
# SCORING PRIMITIVES (copied from search_repository.py for offline use)
# =============================================================================


def _tokenize_query(
    query: str,
) -> list[str]:
    """Tokenize a query string into meaningful keywords."""
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
    """Check if any token matches within the given text."""
    if not tokens or not text:
        return False
    text_lower = text.lower()
    return any(token in text_lower for token in tokens)


def _cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_text_boost(
    doc: dict,
    query_tokens: list[str],
) -> float:
    """Compute text_boost for a document (same logic as production)."""
    text_boost = 0.0

    path = doc.get("path", "")
    if path and _tokens_match_text(query_tokens, path):
        text_boost += 5.0
    name = doc.get("name", "")
    if name and _tokens_match_text(query_tokens, name):
        text_boost += 3.0
    description = doc.get("description", "")
    if description and _tokens_match_text(query_tokens, description):
        text_boost += 2.0
    tags = doc.get("tags", [])
    if tags and any(_tokens_match_text(query_tokens, tag) for tag in tags):
        text_boost += 1.5
    metadata_text = doc.get("metadata_text", "")
    if metadata_text and _tokens_match_text(query_tokens, metadata_text):
        text_boost += 1.0

    tools = doc.get("tools", [])
    for tool in tools:
        tool_name = tool.get("name", "")
        tool_desc = tool.get("description") or ""
        if _tokens_match_text(query_tokens, tool_name) or _tokens_match_text(
            query_tokens, tool_desc
        ):
            text_boost += 1.0

    return text_boost


_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model (same as production)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(
            "Model loaded. Dimension: %d",
            _embedding_model.get_sentence_embedding_dimension(),
        )
    return _embedding_model


def _embed_query(
    query: str,
    docs: list[dict],
) -> list[float] | None:
    """Encode query using the same embedding model as production."""
    model = _get_embedding_model()
    embedding = model.encode([query])[0]
    return embedding.tolist()


# =============================================================================
# SCORING METHODS
# =============================================================================


def _score_rrf(
    docs: list[dict],
    query_embedding: list[float] | None,
    query_tokens: list[str],
) -> list[tuple[dict, float]]:
    """Score documents using Reciprocal Rank Fusion."""
    vector_scored: list[tuple[dict, float]] = []
    keyword_scored: list[tuple[dict, float]] = []

    for doc in docs:
        embedding = doc.get("embedding", [])
        if embedding and query_embedding:
            score = _cosine_similarity(query_embedding, embedding)
            vector_scored.append((doc, score))

        text_boost = _compute_text_boost(doc, query_tokens)
        if text_boost > 0:
            keyword_scored.append((doc, text_boost))

    vector_scored.sort(key=lambda x: x[1], reverse=True)
    keyword_scored.sort(key=lambda x: x[1], reverse=True)

    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank_zero, (doc, _) in enumerate(vector_scored):
        doc_id = doc.get("_id", "")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank_zero + 1)
        doc_map[doc_id] = doc

    for rank_zero, (doc, _) in enumerate(keyword_scored):
        doc_id = doc.get("_id", "")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank_zero + 1)
        doc_map[doc_id] = doc

    results = [(doc_map[doc_id], score) for doc_id, score in scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _score_legacy(
    docs: list[dict],
    query_embedding: list[float] | None,
    query_tokens: list[str],
) -> list[tuple[dict, float]]:
    """Score documents using the legacy additive formula."""
    scored: list[tuple[dict, float]] = []

    for doc in docs:
        embedding = doc.get("embedding", [])
        if embedding and query_embedding:
            vector_score = _cosine_similarity(query_embedding, embedding)
        else:
            vector_score = 0.0

        text_boost = _compute_text_boost(doc, query_tokens)

        normalized_vector_score = (vector_score + 1.0) / 2.0
        text_boost_contribution = text_boost * 0.1
        relevance_score = max(0.0, min(1.0, normalized_vector_score + text_boost_contribution))

        if relevance_score > 0.5:
            scored.append((doc, relevance_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# =============================================================================
# EVALUATION METRICS
# =============================================================================


def _dcg_at_k(
    relevance_grades: list[int],
    k: int = 10,
) -> float:
    """Compute Discounted Cumulative Gain at position k."""
    dcg = 0.0
    for i, grade in enumerate(relevance_grades[:k]):
        dcg += (2**grade - 1) / math.log2(i + 2)
    return dcg


def _ndcg_at_k(
    relevance_grades: list[int],
    ideal_grades: list[int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at position k."""
    dcg = _dcg_at_k(relevance_grades, k)
    idcg = _dcg_at_k(sorted(ideal_grades, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _evaluate_query(
    ranked_results: list[tuple[dict, float]],
    expected: list[dict],
    k: int = 10,
) -> dict:
    """Evaluate a single query's results against ground truth."""
    expected_map = {e["path"]: e["grade"] for e in expected}
    ideal_grades = sorted([e["grade"] for e in expected], reverse=True)

    result_paths = [doc.get("_id", "") for doc, _ in ranked_results[:k]]
    relevance_grades = [expected_map.get(path, 0) for path in result_paths]

    ndcg = _ndcg_at_k(relevance_grades, ideal_grades, k)

    recall_at_k = sum(
        1 for path in result_paths if path in expected_map
    ) / len(expected_map) if expected_map else 0.0

    first_relevant_rank = None
    for i, path in enumerate(result_paths):
        if path in expected_map:
            first_relevant_rank = i + 1
            break

    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "ndcg@10": ndcg,
        "recall@10": recall_at_k,
        "mrr": mrr,
        "found_in_top10": [p for p in result_paths if p in expected_map],
        "missing": [p for p in expected_map if p not in result_paths],
        "top5_results": [
            {"path": doc.get("_id"), "name": doc.get("name"), "score": round(score, 6)}
            for doc, score in ranked_results[:5]
        ],
    }


# =============================================================================
# MAIN
# =============================================================================


def _run_evaluation(
    docs: list[dict],
    ground_truth: list[dict],
    method: str,
    verbose: bool = False,
) -> dict:
    """Run a full evaluation of one scoring method."""
    score_fn = _score_rrf if method == "rrf" else _score_legacy

    metrics = []
    start_time = time.time()

    for gt in ground_truth:
        query = gt["query"]
        expected = gt["expected"]
        query_tokens = _tokenize_query(query)
        query_embedding = _embed_query(query, docs)

        ranked = score_fn(docs, query_embedding, query_tokens)
        eval_result = _evaluate_query(ranked, expected)

        metrics.append({
            "query": query,
            "description": gt.get("description", ""),
            **eval_result,
        })

        if verbose:
            found_str = ", ".join(eval_result["found_in_top10"][:3]) or "(none)"
            print(
                f"  [{method:6}] '{query:35}' "
                f"NDCG={eval_result['ndcg@10']:.3f} "
                f"Recall={eval_result['recall@10']:.2f} "
                f"Found: {found_str}"
            )

    elapsed = time.time() - start_time

    avg_ndcg = sum(m["ndcg@10"] for m in metrics) / len(metrics)
    avg_recall = sum(m["recall@10"] for m in metrics) / len(metrics)
    avg_mrr = sum(m["mrr"] for m in metrics) / len(metrics)
    perfect_queries = sum(1 for m in metrics if m["ndcg@10"] == 1.0)
    zero_queries = sum(1 for m in metrics if m["ndcg@10"] == 0.0)

    # Score saturation check
    all_scores = []
    for gt in ground_truth:
        query_tokens = _tokenize_query(gt["query"])
        query_embedding = _embed_query(gt["query"], docs)
        ranked = score_fn(docs, query_embedding, query_tokens)
        all_scores.extend([s for _, s in ranked[:10]])

    saturated = sum(1 for s in all_scores if s >= 0.999)
    unique_scores = len(set(round(s, 6) for s in all_scores))

    return {
        "method": method,
        "avg_ndcg@10": round(avg_ndcg, 4),
        "avg_recall@10": round(avg_recall, 4),
        "avg_mrr": round(avg_mrr, 4),
        "perfect_queries": perfect_queries,
        "zero_queries": zero_queries,
        "total_queries": len(ground_truth),
        "elapsed_seconds": round(elapsed, 2),
        "score_health": {
            "unique_scores_in_top10": unique_scores,
            "saturated_at_1.0": saturated,
            "total_scored": len(all_scores),
        },
        "per_query": metrics,
    }


def main():
    """Run evaluation and print summary."""
    parser = argparse.ArgumentParser(
        description="Evaluate search scoring methods against ground truth",
    )
    parser.add_argument(
        "--method",
        choices=["rrf", "legacy", "both"],
        default="both",
        help="Scoring method to evaluate (default: both)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save detailed results to JSON file",
    )

    args = parser.parse_args()

    logger.info(f"Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH) as f:
        docs = json.load(f)
    logger.info(f"Loaded {len(docs)} documents")

    logger.info(f"Loading ground truth from {GROUND_TRUTH_PATH}")
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)
    logger.info(f"Loaded {len(ground_truth)} queries with ground truth")

    methods = ["rrf", "legacy"] if args.method == "both" else [args.method]
    results = {}

    for method in methods:
        logger.info(f"\nEvaluating: {method.upper()}")
        result = _run_evaluation(docs, ground_truth, method, args.verbose)
        results[method] = result

    # Print summary
    print(f"\n{'='*70}")
    print(f"SEARCH SCORING EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Dataset: {len(docs)} documents | Queries: {len(ground_truth)}")
    print(f"{'='*70}")

    header = f"{'Metric':<30}"
    for method in methods:
        header += f"  {method.upper():>12}"
    print(header)
    print("-" * 70)

    metric_rows = [
        ("NDCG@10 (avg)", "avg_ndcg@10"),
        ("Recall@10 (avg)", "avg_recall@10"),
        ("MRR (avg)", "avg_mrr"),
        ("Perfect queries (NDCG=1.0)", "perfect_queries"),
        ("Zero-hit queries (NDCG=0.0)", "zero_queries"),
        ("Elapsed (seconds)", "elapsed_seconds"),
    ]

    for label, key in metric_rows:
        row = f"{label:<30}"
        for method in methods:
            val = results[method][key]
            if isinstance(val, float):
                row += f"  {val:>12.4f}"
            else:
                row += f"  {val:>12}"
        print(row)

    print()
    print("Score Health:")
    for method in methods:
        sh = results[method]["score_health"]
        print(
            f"  {method.upper():>8}: "
            f"{sh['unique_scores_in_top10']} unique scores, "
            f"{sh['saturated_at_1.0']} saturated at 1.0 "
            f"(out of {sh['total_scored']} top-10 results)"
        )

    if len(methods) == 2:
        print()
        ndcg_diff = results["rrf"]["avg_ndcg@10"] - results["legacy"]["avg_ndcg@10"]
        direction = "BETTER" if ndcg_diff > 0 else "WORSE" if ndcg_diff < 0 else "SAME"
        print(f"RRF vs Legacy NDCG@10 difference: {ndcg_diff:+.4f} ({direction})")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
