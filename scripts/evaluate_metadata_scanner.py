"""Evaluate the registrant-metadata YARA scanner against a labelled corpus (#1811).

Runs ``registry.utils.metadata_security.scan_metadata`` (the function every
registry write uses) over each record in the corpus and reports:

- record-level true/false positives and negatives, precision, recall and
  false-positive rate, overall and per category;
- for detected malicious records, whether the flagged keys exactly match the
  keys labelled malicious;
- per-record scan latency (the rules are compiled once before timing, and that
  compile time is reported separately) and total evaluation time;
- an ablation that matches the same rules against the compact JSON alone,
  without the plain-text copy of keys and values the scanner appends.

Records marked ``post_hoc`` in the corpus were added after the first run to
test one mechanism; they are reported separately and kept out of the
headline metrics.

The corpus labels were assigned from each record's intent, not from scanner
output. No server, database or network is needed.

Usage:
    uv run python scripts/evaluate_metadata_scanner.py
    uv run python scripts/evaluate_metadata_scanner.py --verbose
    uv run python scripts/evaluate_metadata_scanner.py --output results.json --repeat 50
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS: Path = REPO_ROOT / "tests" / "fixtures" / "metadata_security" / "corpus.json"
DEFAULT_REPEAT: int = 20

sys.path.insert(0, str(REPO_ROOT))

from registry.utils import metadata_security  # noqa: E402


def _ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _confusion(
    rows: list[dict[str, Any]],
    predicted_key: str,
) -> dict[str, Any]:
    """Record-level confusion matrix and rates for one prediction column."""
    tp = sum(1 for r in rows if r["label"] == "malicious" and r[predicted_key])
    fn = sum(1 for r in rows if r["label"] == "malicious" and not r[predicted_key])
    fp = sum(1 for r in rows if r["label"] == "benign" and r[predicted_key])
    tn = sum(1 for r in rows if r["label"] == "benign" and not r[predicted_key])
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
    }


def _json_only_flagged(
    record: dict[str, Any],
) -> bool:
    """Ablation: match the same compiled rules against the compact JSON alone."""
    document = metadata_security.canonical_metadata(record["metadata"], record["entity_type"])
    if not document:
        return False
    serialized = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    rules = metadata_security._load_rules().rules
    return bool(rules.match(data=serialized.encode("utf-8")))


def _time_scan(
    record: dict[str, Any],
    repeat: int,
) -> tuple[Any, list[float]]:
    """Scan one record ``repeat`` times; return the verdict and each run's ms."""
    timings = []
    verdict = None
    for _ in range(repeat):
        started = time.perf_counter()
        verdict = metadata_security.scan_metadata(record["metadata"], record["entity_type"])
        timings.append((time.perf_counter() - started) * 1000)
    return verdict, timings


def _evaluate_record(
    record: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    verdict, timings = _time_scan(record, repeat)
    expected = sorted(tuple(p) for p in record.get("malicious_keys", []))
    flagged = sorted(tuple(p) for p in verdict.flagged_keys)
    return {
        "id": record["id"],
        "label": record["label"],
        "post_hoc": bool(record.get("post_hoc")),
        "category": record["category"],
        "status": verdict.status,
        "flagged": verdict.status == "flagged",
        "json_only_flagged": _json_only_flagged(record),
        "rules": [finding.rule for finding in verdict.findings],
        "flagged_keys": [list(p) for p in flagged],
        "keys_exact": flagged == expected if record["label"] == "malicious" else None,
        "median_ms": round(statistics.median(timings), 4),
        "timings_ms": timings,
    }


def _latency_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_timings = sorted(t for r in rows for t in r["timings_ms"])
    per_record = sorted(r["median_ms"] for r in rows)
    p95_index = max(0, int(round(0.95 * len(all_timings))) - 1)
    return {
        "per_record_median_ms_min": round(per_record[0], 4),
        "per_record_median_ms_median": round(statistics.median(per_record), 4),
        "per_record_median_ms_max": round(per_record[-1], 4),
        "all_scans_p95_ms": round(all_timings[p95_index], 4),
        "scans_timed": len(all_timings),
    }


def _per_category(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = result.setdefault(row["category"], {"records": 0, "flagged": 0})
        bucket["records"] += 1
        bucket["flagged"] += int(row["flagged"])
    return result


def _build_report(
    rows: list[dict[str, Any]],
    compile_ms: float,
    total_seconds: float,
    ruleset: str,
) -> dict[str, Any]:
    headline = [r for r in rows if not r["post_hoc"]]
    detected = [r for r in headline if r["label"] == "malicious" and r["flagged"]]
    post_hoc = [
        {"id": r["id"], "flagged": r["flagged"], "json_only_flagged": r["json_only_flagged"]}
        for r in rows
        if r["post_hoc"]
    ]
    return {
        "ruleset": ruleset,
        "records": len(headline),
        "malicious": sum(1 for r in headline if r["label"] == "malicious"),
        "benign": sum(1 for r in headline if r["label"] == "benign"),
        "scanner": _confusion(headline, "flagged"),
        "ablation_json_only": _confusion(headline, "json_only_flagged"),
        "post_hoc_records": post_hoc,
        "key_attribution_exact": f"{sum(1 for r in detected if r['keys_exact'])}/{len(detected)}",
        "per_category": _per_category(headline),
        "latency": _latency_summary(rows),
        "rule_compile_ms": round(compile_ms, 2),
        "total_evaluation_seconds": round(total_seconds, 3),
        "false_negatives": [
            r["id"] for r in headline if r["label"] == "malicious" and not r["flagged"]
        ],
        "false_positives": [r["id"] for r in headline if r["label"] == "benign" and r["flagged"]],
    }


def _print_rows(
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        logger.info(
            f"{row['id']:6} {row['label']:9} {row['status']:8} rules={row['rules']} "
            f"keys={row['flagged_keys']} json_only={row['json_only_flagged']} "
            f"median={row['median_ms']}ms"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/evaluate_metadata_scanner.py\n"
            "  uv run python scripts/evaluate_metadata_scanner.py --verbose --repeat 50\n"
            "  METADATA_EVAL_CORPUS=my.json uv run python scripts/evaluate_metadata_scanner.py"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.getenv("METADATA_EVAL_CORPUS", str(DEFAULT_CORPUS))),
        help="Labelled corpus JSON (env METADATA_EVAL_CORPUS)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"Timed scans per record (default {DEFAULT_REPEAT})",
    )
    parser.add_argument("--output", type=Path, help="Write the full report as JSON here")
    parser.add_argument("--verbose", action="store_true", help="Print every record's result")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Load the corpus, run the evaluation, print and optionally save the report."""
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    records = corpus["records"]
    logger.info(f"Evaluating {len(records)} records from {args.corpus} (repeat={args.repeat})")
    started = time.perf_counter()
    metadata_security._load_rules.cache_clear()
    compile_started = time.perf_counter()
    ruleset = metadata_security._load_rules().ruleset
    compile_ms = (time.perf_counter() - compile_started) * 1000
    rows = [_evaluate_record(record, args.repeat) for record in records]
    report = _build_report(rows, compile_ms, time.perf_counter() - started, ruleset)
    if args.verbose:
        _print_rows(rows)
    logger.info(f"Report:\n{json.dumps(report, indent=2)}")
    if args.output:
        full = {**report, "rows": [{k: v for k, v in r.items() if k != "timings_ms"} for r in rows]}
        args.output.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
