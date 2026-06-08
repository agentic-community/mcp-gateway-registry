"""Benchmark search quality against a deployed AI Registry.

Runs the ground truth query set (100 queries) against a live deployment's
/api/search/semantic endpoint and captures results for quality assessment
and before/after comparison.

Usage:
    # Test against your registry (provide URL and token file)
    uv run python scripts/benchmark_search.py \
        --url https://your-registry.example.com \
        --token-file .token \
        --output results.json

    # Compare two runs (e.g., before and after upgrade)
    uv run python scripts/benchmark_search.py \
        --compare results_before.json results_after.json

    # Use inline token instead of file
    uv run python scripts/benchmark_search.py \
        --url http://localhost \
        --token "eyJhbG..." \
        --output results.json
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).parent.parent / "tests/fixtures/search_dataset/ground_truth.json"


def _extract_token(
    raw: str,
) -> str:
    """Extract the access token from various formats.

    Supports:
    - Plain JWT string (eyJhbG...)
    - "Bearer eyJhbG..." prefix
    - Full JSON response from the registry's "Get JWT Token" button
    """
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            token = (
                data.get("tokens", {}).get("access_token")
                or data.get("token_data", {}).get("access_token")
                or data.get("access_token")
            )
            if token:
                return token
        except json.JSONDecodeError:
            pass

    if raw.lower().startswith("bearer "):
        return raw[7:].strip()

    return raw


def _load_queries(
    queries_file: Path,
) -> list[dict]:
    """Load benchmark queries from JSON file."""
    with open(queries_file) as f:
        return json.load(f)


def _run_search(
    base_url: str,
    query: str,
    token: str | None = None,
) -> dict:
    """Execute a semantic search query against the deployment."""
    url = f"{base_url}/api/search/semantic"
    payload = {
        "query": query,
        "entity_types": ["mcp_server", "tool", "a2a_agent", "skill", "virtual_server"],
        "max_results": 20,
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _extract_summary(
    response: dict,
) -> dict:
    """Extract a comparable summary from search response."""
    summary = {"servers": [], "tools": [], "agents": [], "skills": []}

    for server in response.get("servers", []):
        summary["servers"].append({
            "name": server.get("server_name"),
            "path": server.get("path"),
            "score": server.get("relevance_score"),
        })

    for tool in response.get("tools", []):
        summary["tools"].append({
            "name": tool.get("tool_name"),
            "server": tool.get("server_name"),
            "score": tool.get("relevance_score"),
        })

    for agent in response.get("agents", []):
        agent_card = agent.get("agent_card", {})
        summary["agents"].append({
            "name": agent_card.get("name") or agent.get("agent_name"),
            "path": agent.get("path"),
            "score": agent.get("relevance_score"),
        })

    for skill in response.get("skills", []):
        summary["skills"].append({
            "name": skill.get("skill_name"),
            "path": skill.get("path"),
            "score": skill.get("relevance_score"),
        })

    return summary


def _fetch_all_assets(
    base_url: str,
    token: str | None = None,
) -> dict:
    """Fetch all servers, agents, and skills from the registry."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    assets = {"servers": [], "agents": [], "skills": []}

    for endpoint, key in [
        ("/api/servers", "servers"),
        ("/api/agents", "agents"),
        ("/api/skills", "skills"),
    ]:
        try:
            response = requests.get(
                f"{base_url}{endpoint}",
                headers=headers,
                params={"limit": 2000},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                assets[key] = data.get(key, [])
            elif isinstance(data, list):
                assets[key] = data
        except Exception as e:
            logger.warning(f"Could not fetch {key}: {e}")

    return assets


def _generate_queries_from_assets(
    assets: dict,
) -> list[dict]:
    """Generate ground truth queries from registry assets.

    Strategies:
    1. Exact name queries (should find the asset by name)
    2. Tag-based queries (should find assets with those tags)
    3. Description-derived queries (key phrases from descriptions)
    4. Multi-word queries combining name + domain terms
    """
    ground_truth = []
    seen_queries = set()

    def _add_query(query, category, description, expected):
        q_lower = query.lower().strip()
        if not q_lower or q_lower in seen_queries or len(q_lower) < 3:
            return
        seen_queries.add(q_lower)
        ground_truth.append({
            "query": query,
            "category": category,
            "description": description,
            "expected": expected,
        })

    # Strategy 1: Server names as queries
    for server in assets.get("servers", []):
        path = server.get("path", "")
        name = server.get("server_name") or server.get("name", "")
        if not name or not path:
            continue
        tags = server.get("tags", [])
        if "stress-test" in tags or "security-pending" in tags:
            continue

        _add_query(
            name,
            "exact-name",
            f"Server name: {name}",
            [{"path": path, "grade": 3, "reason": f"Exact name match: {name}"}],
        )

    # Strategy 2: Agent names as queries
    for agent in assets.get("agents", []):
        path = agent.get("path", "")
        name = agent.get("name", "")
        if not name or not path:
            continue
        tags = agent.get("tags", [])
        if "stress-test" in tags or "security-pending" in tags:
            continue

        _add_query(
            name,
            "exact-name",
            f"Agent name: {name}",
            [{"path": path, "grade": 3, "reason": f"Exact name match: {name}"}],
        )

    # Strategy 3: Skill names as queries
    for skill in assets.get("skills", []):
        path = skill.get("path", "")
        name = skill.get("name", "")
        if not name or not path:
            continue
        tags = skill.get("tags", [])
        if "stress-test" in tags:
            continue

        _add_query(
            name,
            "exact-name",
            f"Skill name: {name}",
            [{"path": path, "grade": 3, "reason": f"Exact name match: {name}"}],
        )

    # Strategy 4: Tag-based queries (group assets by shared tags)
    tag_to_assets: dict[str, list[dict]] = {}
    for asset_type, key in [("mcp_server", "servers"), ("a2a_agent", "agents"), ("skill", "skills")]:
        for asset in assets.get(key, []):
            path = asset.get("path", "")
            name = asset.get("server_name") or asset.get("name", "")
            tags = asset.get("tags", [])
            if "stress-test" in tags or "security-pending" in tags:
                continue
            for tag in tags:
                if tag in ("stress-test", "security-pending", "security-pending-local"):
                    continue
                if tag not in tag_to_assets:
                    tag_to_assets[tag] = []
                tag_to_assets[tag].append({
                    "path": path,
                    "name": name,
                    "entity_type": asset_type,
                })

    for tag, tag_assets in tag_to_assets.items():
        if len(tag) < 3 or len(tag_assets) > 20:
            continue
        expected = [
            {"path": a["path"], "grade": 2, "reason": f"Has tag: {tag}"}
            for a in tag_assets[:5]
        ]
        _add_query(
            tag,
            "tag-based",
            f"Tag query: assets tagged with '{tag}'",
            expected,
        )

    # Strategy 5: Description keywords (first 3 meaningful words)
    for asset_type, key in [("mcp_server", "servers"), ("a2a_agent", "agents"), ("skill", "skills")]:
        for asset in assets.get(key, []):
            path = asset.get("path", "")
            name = asset.get("server_name") or asset.get("name", "")
            desc = asset.get("description", "")
            tags = asset.get("tags", [])
            if "stress-test" in tags or "security-pending" in tags:
                continue
            if not desc or len(desc) < 20:
                continue

            import re
            words = [
                w.lower() for w in re.split(r"\W+", desc)
                if len(w) > 3 and w.lower() not in (
                    "this", "that", "with", "from", "your", "have", "will",
                    "been", "they", "their", "about", "would", "could", "should",
                    "into", "also", "tool", "server", "agent", "skill",
                )
            ][:4]
            if len(words) >= 2:
                query = " ".join(words[:3])
                _add_query(
                    query,
                    "description-derived",
                    f"Keywords from {name} description",
                    [{"path": path, "grade": 3, "reason": f"Description contains these terms"}],
                )

    # Cap at 100 queries, balanced across categories
    if len(ground_truth) > 100:
        from collections import Counter
        cat_counts = Counter(q["category"] for q in ground_truth)
        max_per_cat = max(10, 100 // len(cat_counts))
        filtered = []
        cat_used: dict[str, int] = {}
        for q in ground_truth:
            cat = q["category"]
            if cat_used.get(cat, 0) < max_per_cat:
                filtered.append(q)
                cat_used[cat] = cat_used.get(cat, 0) + 1
        ground_truth = filtered[:100]

    return ground_truth


def _generate_ground_truth(
    base_url: str,
    token: str | None = None,
) -> None:
    """Generate ground truth file from a live registry's assets."""
    logger.info(f"Fetching assets from {base_url}")
    assets = _fetch_all_assets(base_url, token)

    server_count = len(assets.get("servers", []))
    agent_count = len(assets.get("agents", []))
    skill_count = len(assets.get("skills", []))
    logger.info(
        f"Found {server_count} servers, {agent_count} agents, {skill_count} skills"
    )

    if server_count + agent_count + skill_count == 0:
        logger.error("No assets found. Check URL and token.")
        raise SystemExit(1)

    ground_truth = _generate_queries_from_assets(assets)
    logger.info(f"Generated {len(ground_truth)} queries")

    output_dir = Path(__file__).parent.parent / "tests/fixtures/search_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated_ground_truth.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    from collections import Counter
    cats = Counter(q["category"] for q in ground_truth)
    logger.info(f"Saved to {output_path}")
    logger.info(f"Categories: {dict(cats)}")
    print(f"\nGround truth generated: {output_path}")
    print(f"  {len(ground_truth)} queries across {len(cats)} categories")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")
    print(f"\nTo run the benchmark with your generated ground truth:")
    print(f"  uv run python scripts/benchmark_search.py --url {base_url} --token-file .token --queries {output_path}")


def _fetch_registry_stats(
    base_url: str,
    token: str | None = None,
) -> dict:
    """Fetch registry stats (version, counts, backend)."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get(
            f"{base_url}/api/stats", headers=headers, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch registry stats: {e}")
        return {}


def _run_benchmark(
    base_url: str,
    queries: list[dict],
    token: str | None = None,
) -> dict:
    """Run all benchmark queries and collect results."""
    stats = _fetch_registry_stats(base_url, token)
    results = {
        "url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "registry": {
            "version": stats.get("version", "unknown"),
            "servers": stats.get("registry_stats", {}).get("servers", 0),
            "agents": stats.get("registry_stats", {}).get("agents", 0),
            "skills": stats.get("registry_stats", {}).get("skills", 0),
            "database_backend": stats.get("database_status", {}).get("backend", "unknown"),
            "deployment_mode": stats.get("deployment_mode", "unknown"),
        },
        "queries": [],
    }

    for q in queries:
        query_text = q["query"]
        logger.info(f"Running query: '{query_text}'")

        start = time.time()
        try:
            response = _run_search(base_url, query_text, token)
            elapsed = time.time() - start

            summary = _extract_summary(response)
            results["queries"].append({
                "query": query_text,
                "description": q.get("description", ""),
                "elapsed_ms": round(elapsed * 1000, 1),
                "total_results": sum(len(v) for v in summary.values()),
                "results": summary,
            })
            logger.info(
                f"  -> {summary['servers'].__len__()} servers, "
                f"{summary['tools'].__len__()} tools, "
                f"{summary['agents'].__len__()} agents, "
                f"{summary['skills'].__len__()} skills "
                f"({elapsed*1000:.0f}ms)"
            )
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            results["queries"].append({
                "query": query_text,
                "description": q.get("description", ""),
                "error": str(e),
            })

    return results


def _compare_results(
    file_a: Path,
    file_b: Path,
) -> None:
    """Print side-by-side comparison of two benchmark runs."""
    with open(file_a) as f:
        results_a = json.load(f)
    with open(file_b) as f:
        results_b = json.load(f)

    print(f"\n{'='*80}")
    print(f"COMPARISON: {file_a.name} vs {file_b.name}")
    print(f"{'='*80}")
    print(f"  A: {results_a['url']} ({results_a['timestamp']})")
    print(f"  B: {results_b['url']} ({results_b['timestamp']})")
    print()

    queries_a = {q["query"]: q for q in results_a["queries"]}
    queries_b = {q["query"]: q for q in results_b["queries"]}

    all_queries = list(queries_a.keys() | queries_b.keys())
    all_queries.sort()

    for query in all_queries:
        qa = queries_a.get(query, {})
        qb = queries_b.get(query, {})

        print(f"\n--- Query: '{query}' ---")

        if qa.get("error") or qb.get("error"):
            print(f"  A: ERROR {qa.get('error', 'N/A')}")
            print(f"  B: ERROR {qb.get('error', 'N/A')}")
            continue

        ra = qa.get("results", {})
        rb = qb.get("results", {})

        for entity_type in ["servers", "tools", "agents", "skills"]:
            items_a = ra.get(entity_type, [])
            items_b = rb.get(entity_type, [])

            if not items_a and not items_b:
                continue

            print(f"\n  {entity_type.upper()}:")
            max_len = max(len(items_a), len(items_b))

            for i in range(max_len):
                a_item = items_a[i] if i < len(items_a) else None
                b_item = items_b[i] if i < len(items_b) else None

                a_str = (
                    f"{a_item['name']} ({a_item['score']:.4f})"
                    if a_item
                    else "(none)"
                )
                b_str = (
                    f"{b_item['name']} ({b_item['score']:.4f})"
                    if b_item
                    else "(none)"
                )

                changed = ""
                if a_item and b_item:
                    if a_item["name"] != b_item["name"]:
                        changed = " [RERANKED]"
                    elif abs(a_item["score"] - b_item["score"]) > 0.001:
                        changed = " [score changed]"

                print(f"    #{i+1}: {a_str:40} | {b_str}{changed}")

        # Score saturation check
        all_scores_a = [
            item["score"]
            for items in ra.values()
            for item in items
            if item.get("score") is not None
        ]
        all_scores_b = [
            item["score"]
            for items in rb.values()
            for item in items
            if item.get("score") is not None
        ]

        saturated_a = sum(1 for s in all_scores_a if s >= 0.999)
        saturated_b = sum(1 for s in all_scores_b if s >= 0.999)
        unique_a = len(set(round(s, 4) for s in all_scores_a))
        unique_b = len(set(round(s, 4) for s in all_scores_b))

        print(f"\n  Score health: A={unique_a} unique scores ({saturated_a} saturated at 1.0)"
              f" | B={unique_b} unique scores ({saturated_b} saturated at 1.0)")


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


def _evaluate_query_results(
    results: dict,
    expected: list[dict],
) -> dict:
    """Evaluate a single query's live results against ground truth."""
    expected_map = {e["path"]: e["grade"] for e in expected}
    ideal_grades = sorted([e["grade"] for e in expected], reverse=True)

    result_paths = []
    for entity_type in ["servers", "tools", "agents", "skills"]:
        for item in results.get(entity_type, []):
            path = item.get("path") or item.get("server_path", "")
            if path and path not in result_paths:
                result_paths.append(path)

    relevance_grades = [expected_map.get(path, 0) for path in result_paths[:10]]
    ndcg = _ndcg_at_k(relevance_grades, ideal_grades, 10)

    recall = (
        sum(1 for path in result_paths[:10] if path in expected_map)
        / len(expected_map)
        if expected_map
        else 0.0
    )

    first_relevant_rank = None
    for i, path in enumerate(result_paths[:10]):
        if path in expected_map:
            first_relevant_rank = i + 1
            break
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "ndcg@10": ndcg,
        "recall@10": recall,
        "mrr": mrr,
        "found": [p for p in result_paths[:10] if p in expected_map],
        "missing": [p for p in expected_map if p not in result_paths[:10]],
    }


def _generate_report(
    results_file: Path,
) -> None:
    """Generate a markdown report from benchmark results with NDCG metrics."""
    with open(results_file) as f:
        data = json.load(f)

    gt_path = Path(__file__).parent.parent / "tests/fixtures/search_dataset/ground_truth.json"
    ground_truth = {}
    if gt_path.exists():
        with open(gt_path) as f:
            for q in json.load(f):
                ground_truth[q["query"]] = q

    queries = data.get("queries", [])
    successful = [q for q in queries if not q.get("error")]
    failed = [q for q in queries if q.get("error")]

    all_scores = []
    for q in successful:
        results = q.get("results", {})
        for entity_type in ["servers", "tools", "agents", "skills"]:
            for item in results.get(entity_type, []):
                if item.get("score") is not None:
                    all_scores.append(item["score"])

    saturated = sum(1 for s in all_scores if s >= 0.999)
    unique_scores = len(set(round(s, 4) for s in all_scores))
    avg_latency = (
        sum(q.get("elapsed_ms", 0) for q in successful) / len(successful)
        if successful
        else 0
    )

    query_metrics = []
    for q in successful:
        gt = ground_truth.get(q["query"])
        if gt and gt.get("expected"):
            metrics = _evaluate_query_results(q.get("results", {}), gt["expected"])
            metrics["query"] = q["query"]
            metrics["category"] = gt.get("category", "")
            query_metrics.append(metrics)

    avg_ndcg = (
        sum(m["ndcg@10"] for m in query_metrics) / len(query_metrics)
        if query_metrics
        else 0.0
    )
    avg_recall = (
        sum(m["recall@10"] for m in query_metrics) / len(query_metrics)
        if query_metrics
        else 0.0
    )
    avg_mrr = (
        sum(m["mrr"] for m in query_metrics) / len(query_metrics)
        if query_metrics
        else 0.0
    )
    perfect = sum(1 for m in query_metrics if m["ndcg@10"] == 1.0)
    zero_hit = sum(1 for m in query_metrics if m["ndcg@10"] == 0.0)

    registry = data.get("registry", {})

    report_path = results_file.with_suffix(".md")
    with open(report_path, "w") as f:
        f.write("# Search Benchmark Report\n\n")
        f.write(f"- **Target:** {data.get('url', 'unknown')}\n")
        f.write(f"- **Timestamp:** {data.get('timestamp', 'unknown')}\n")
        f.write(f"- **Version:** {registry.get('version', 'unknown')}\n")
        f.write(f"- **Database:** {registry.get('database_backend', 'unknown')}\n")
        f.write(f"- **Registry contents:** {registry.get('servers', 0)} servers, "
                f"{registry.get('agents', 0)} agents, {registry.get('skills', 0)} skills\n")
        f.write(f"- **Queries:** {len(queries)} total, {len(successful)} succeeded, {len(failed)} failed\n")
        f.write(f"- **Avg latency:** {avg_latency:.0f}ms per query\n\n")

        f.write("## Quality Metrics (against ground truth)\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| NDCG@10 (avg) | {avg_ndcg:.4f} |\n")
        f.write(f"| MRR (avg) | {avg_mrr:.4f} |\n")
        f.write(f"| Recall@10 (avg) | {avg_recall:.4f} |\n")
        f.write(f"| Perfect queries (NDCG=1.0) | {perfect} / {len(query_metrics)} |\n")
        f.write(f"| Zero-hit queries (NDCG=0.0) | {zero_hit} / {len(query_metrics)} |\n")
        f.write(f"| Evaluated queries | {len(query_metrics)} (queries with ground truth expectations) |\n")
        skipped = len(successful) - len(query_metrics)
        if skipped > 0:
            f.write(f"| Skipped from eval | {skipped} (no-answer queries with empty expected results) |\n")
        if failed:
            f.write(f"| Failed queries | {len(failed)} (API rejected, e.g. empty query) |\n")
        f.write("\n")

        f.write("## Score Health\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total scores in results | {len(all_scores)} |\n")
        f.write(f"| Unique score values | {unique_scores} |\n")
        f.write(f"| Saturated at 1.0 | {saturated} ({saturated*100//max(len(all_scores),1)}%) |\n")
        if all_scores:
            f.write(f"| Score range | {min(all_scores):.4f} to {max(all_scores):.4f} |\n\n")

        if query_metrics:
            cat_metrics: dict[str, list] = {}
            for m in query_metrics:
                cat = m.get("category", "unknown")
                if cat not in cat_metrics:
                    cat_metrics[cat] = []
                cat_metrics[cat].append(m)

            f.write("## Quality by Category\n\n")
            f.write("| Category | Queries | NDCG@10 | MRR | Recall@10 |\n")
            f.write("|----------|---------|---------|-----|----------|\n")
            for cat, metrics in sorted(cat_metrics.items()):
                n = len(metrics)
                cat_ndcg = sum(m["ndcg@10"] for m in metrics) / n
                cat_mrr = sum(m["mrr"] for m in metrics) / n
                cat_recall = sum(m["recall@10"] for m in metrics) / n
                f.write(f"| {cat} | {n} | {cat_ndcg:.3f} | {cat_mrr:.3f} | {cat_recall:.3f} |\n")
            f.write("\n")

        f.write("## Results by Query\n\n")
        for q in successful:
            results = q.get("results", {})
            total = q.get("total_results", 0)
            f.write(f"### \"{q['query']}\"\n\n")
            if q.get("description"):
                f.write(f"*{q['description']}*\n\n")

            gt = ground_truth.get(q["query"])
            if gt and gt.get("expected"):
                metrics = _evaluate_query_results(results, gt["expected"])
                f.write(
                    f"NDCG@10={metrics['ndcg@10']:.3f} | "
                    f"MRR={metrics['mrr']:.3f} | "
                    f"Recall={metrics['recall@10']:.2f}"
                )
                if metrics["found"]:
                    f.write(f" | Found: {', '.join(metrics['found'][:3])}")
                if metrics["missing"]:
                    f.write(f" | Missing: {', '.join(metrics['missing'][:2])}")
                f.write("\n\n")

            f.write(f"Latency: {q.get('elapsed_ms', 0):.0f}ms | Results: {total} total\n\n")

            for entity_type in ["servers", "tools", "agents", "skills"]:
                items = results.get(entity_type, [])
                if not items:
                    continue
                f.write(f"**{entity_type.title()}:**\n\n")
                f.write("| # | Name | Score |\n")
                f.write("|---|------|-------|\n")
                for i, item in enumerate(items[:5]):
                    name = item.get("name", "unknown")
                    score = item.get("score", 0)
                    f.write(f"| {i+1} | {name} | {score:.4f} |\n")
                f.write("\n")

            f.write("---\n\n")

        if failed:
            f.write("## Failed Queries\n\n")
            f.write("| Query | Error |\n")
            f.write("|-------|-------|\n")
            for q in failed:
                f.write(f"| {q['query']} | {q.get('error', 'unknown')[:80]} |\n")

    print(f"Report saved to: {report_path}")


def main():
    """Parse args and run benchmark or comparison."""
    parser = argparse.ArgumentParser(
        description="Benchmark search scoring for before/after comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Test against your deployed registry
    uv run python scripts/benchmark_search.py \\
        --url https://registry.example.com --token-file .token --output results.json

    # Compare before/after upgrade
    uv run python scripts/benchmark_search.py \\
        --compare results_before.json results_after.json
""",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Base URL of the deployment to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Bearer token for authentication (inline)",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default=None,
        help="Path to file containing bearer token (e.g. .token)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(QUERIES_FILE),
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two benchmark result files",
    )
    parser.add_argument(
        "--report",
        type=str,
        metavar="RESULTS_FILE",
        help="Generate a markdown report from a results JSON file",
    )
    parser.add_argument(
        "--generate-ground-truth",
        action="store_true",
        help="Generate ground truth from your registry's assets (requires --url and --token-file)",
    )

    args = parser.parse_args()

    if args.report:
        _generate_report(Path(args.report))
        return

    if args.compare:
        _compare_results(Path(args.compare[0]), Path(args.compare[1]))
        return

    if not args.url:
        parser.error("--url is required when not using --compare or --report")

    if args.generate_ground_truth:
        token = args.token
        if not token and args.token_file:
            token_path = Path(args.token_file)
            if not token_path.exists():
                parser.error(f"Token file not found: {token_path}")
            raw = token_path.read_text().strip()
            token = _extract_token(raw)
        _generate_ground_truth(args.url, token)
        return

    if not args.output:
        output_dir = Path(__file__).parent.parent / "tests/fixtures/search_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / "benchmark_results.json")

    token = args.token
    if not token and args.token_file:
        token_path = Path(args.token_file)
        if not token_path.exists():
            parser.error(f"Token file not found: {token_path}")
        raw = token_path.read_text().strip()
        token = _extract_token(raw)

    queries = _load_queries(Path(args.queries))
    logger.info(f"Loaded {len(queries)} benchmark queries")
    logger.info(f"Target: {args.url}")

    results = _run_benchmark(args.url, queries, token)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    _generate_report(output_path)


if __name__ == "__main__":
    main()
