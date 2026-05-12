"""Shared helpers for stress test data generators.

Exposes:
- GeneratorResult: summary printed by each generator
- write_payload, unique_suffix, ensure_output_dir, build_argparser helpers
- run_generator: shared main loop used by the per-entity scripts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tests.stress.config import data_dir_for, default_cache_dir
from tests.stress.constants import STRESS_SUFFIX_TEMPLATE, TARGET_SIZES, EntityType

logger = logging.getLogger(__name__)


class GeneratorResult(BaseModel):
    """Outcome of a single generator run, printed as JSON at the end."""

    entity_type: EntityType
    target_count: int
    actual_count: int
    source_records: int
    augmented_records: int
    elapsed_seconds: float
    output_dir: str
    errors: list[str] = Field(default_factory=list)


def unique_suffix(index: int) -> str:
    """Return the canonical augmented-record suffix for the given index."""
    return STRESS_SUFFIX_TEMPLATE.format(index=index)


def ensure_output_dir(path: Path, force: bool) -> None:
    """Create the output dir, optionally clearing existing JSON files first."""
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for old in path.glob("*.json"):
            old.unlink()


def write_payload(
    output_dir: Path,
    payload: dict[str, Any],
    filename: str,
) -> Path:
    """Write a single payload JSON. Overwrites on re-run (idempotent)."""
    out_path = output_dir / filename
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def safe_filename(seed: str, index: int) -> str:
    """Build a deterministic, filesystem-safe filename from a seed and index."""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]  # nosec B324 - non-cryptographic
    return f"{index:05d}-{digest}.json"


def build_argparser(
    description: str,
) -> argparse.ArgumentParser:
    """Return the standard generator CLI parser."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        choices=TARGET_SIZES,
        help=f"Target number of payloads to generate (one of {TARGET_SIZES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: tests/stress/data/<entity>/<count>/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for upstream API responses (default: tests/stress/data/.cache/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing JSON files in the output dir before writing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )


def resolve_dirs(
    args: argparse.Namespace,
    entity_type: EntityType,
) -> tuple[Path, Path]:
    """Return (output_dir, cache_dir) given parsed CLI args."""
    output_dir = args.output_dir or data_dir_for(entity_type, args.count)
    cache_dir = args.cache_dir or default_cache_dir() / entity_type
    return output_dir, cache_dir


def run_generator(
    entity_type: EntityType,
    description: str,
    fetch_records: Callable[[Path], list[dict[str, Any]]],
    build_payload: Callable[[dict[str, Any], int | None], dict[str, Any]],
    validate_payload: Callable[[dict[str, Any]], None],
    payload_seed: Callable[[dict[str, Any]], str],
) -> int:
    """Shared generator entry point.

    The per-entity scripts wire up these four callables and call this from main().
    """
    parser = build_argparser(description)
    args = parser.parse_args()
    configure_logging(args.debug)

    output_dir, cache_dir = resolve_dirs(args, entity_type)
    ensure_output_dir(output_dir, force=args.force)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generator starting: entity_type=%s target_count=%d output_dir=%s cache_dir=%s",
        entity_type,
        args.count,
        output_dir,
        cache_dir,
    )

    start = time.time()
    errors: list[str] = []

    try:
        source_records = fetch_records(cache_dir)
    except Exception as exc:
        logger.exception("Upstream fetch failed")
        _emit_result(
            GeneratorResult(
                entity_type=entity_type,
                target_count=args.count,
                actual_count=0,
                source_records=0,
                augmented_records=0,
                elapsed_seconds=time.time() - start,
                output_dir=str(output_dir),
                errors=[f"fetch_failed: {exc}"],
            )
        )
        return 1

    source_count = len(source_records)
    logger.info("Upstream returned %d unique records", source_count)
    if source_count == 0:
        _emit_result(
            GeneratorResult(
                entity_type=entity_type,
                target_count=args.count,
                actual_count=0,
                source_records=0,
                augmented_records=0,
                elapsed_seconds=time.time() - start,
                output_dir=str(output_dir),
                errors=["no_source_records"],
            )
        )
        return 1

    selected = _select_records(source_records, args.count)
    augmented_records = max(0, args.count - source_count)

    actual = 0
    for global_index, (record, suffix_index) in enumerate(selected):
        try:
            payload = build_payload(record, suffix_index)
            validate_payload(payload)
        except Exception as exc:
            err = f"payload_invalid index={global_index} seed={payload_seed(record)}: {exc}"
            logger.error(err)
            errors.append(err)
            continue

        filename = safe_filename(payload_seed(record), global_index)
        write_payload(output_dir, payload, filename)
        actual += 1

    elapsed = time.time() - start
    result = GeneratorResult(
        entity_type=entity_type,
        target_count=args.count,
        actual_count=actual,
        source_records=source_count,
        augmented_records=augmented_records,
        elapsed_seconds=elapsed,
        output_dir=str(output_dir),
        errors=errors,
    )
    _emit_result(result)
    return 0 if not errors and actual == args.count else 1


def _select_records(
    source: list[dict[str, Any]],
    target_count: int,
) -> list[tuple[dict[str, Any], int | None]]:
    """Return (record, suffix_index) tuples up to target_count.

    suffix_index is None for the first appearance of a source record, and a
    monotonically increasing integer for augmented (repeated) copies.
    """
    selected: list[tuple[dict[str, Any], int | None]] = []
    source_count = len(source)

    take = min(source_count, target_count)
    for i in range(take):
        selected.append((source[i], None))

    if target_count <= source_count:
        return selected

    aug_index = 0
    while len(selected) < target_count:
        record = source[aug_index % source_count]
        selected.append((record, aug_index + 1))
        aug_index += 1

    return selected


def _emit_result(result: GeneratorResult) -> None:
    """Print the GeneratorResult as a JSON object to stdout."""
    print(json.dumps(result.model_dump(), indent=2, default=str))


def cache_read_json(cache_path: Path) -> Any | None:
    """Return parsed JSON from cache_path or None if not present."""
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Cache file is not valid JSON, ignoring: %s", cache_path)
        return None


def cache_write_json(cache_path: Path, payload: Any) -> None:
    """Persist payload to cache_path."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, default=str))


def ensure_project_on_path() -> None:
    """Add the project root to sys.path so `cli.` / `api.` imports resolve.

    Generators run as standalone scripts (`python tests/stress/...`) so they
    need this fallback when not invoked via `python -m`.
    """
    project_root_path = Path(__file__).resolve().parents[3]
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))


def collect_unique(
    records: Iterable[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """De-duplicate records by the key function, preserving first occurrence."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = key_fn(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out
