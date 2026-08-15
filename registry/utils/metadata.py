"""Shared metadata utilities for keyword search and field projection."""

import logging
from typing import Any

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# --- Projection constants (security/abuse-prevention bounds) ---
MAX_METADATA_PATHS: int = 20
MAX_PATH_DEPTH: int = 5
MAX_SEGMENT_LENGTH: int = 64


def parse_metadata_fields(
    raw: str | None,
) -> list[str] | None:
    """Parse and validate a comma-separated metadata_fields string.

    Args:
        raw: The raw query parameter value, or None.

    Returns:
        A list of validated dot-paths, or None if raw is None/empty.

    Raises:
        ValueError: If validation fails (caller should map to HTTP 422).
    """
    if not raw or not raw.strip():
        return None

    paths = [p.strip() for p in raw.split(",") if p.strip()]
    if not paths:
        return None

    if len(paths) > MAX_METADATA_PATHS:
        raise ValueError(f"Too many paths (max {MAX_METADATA_PATHS})")

    validated: list[str] = []
    for path in paths:
        segments = path.split(".")
        if len(segments) > MAX_PATH_DEPTH:
            raise ValueError(f"Path too deep (max {MAX_PATH_DEPTH} levels): '{path}'")
        for segment in segments:
            if not segment:
                raise ValueError(f"Invalid path segment (must not contain empty parts): '{path}'")
            if segment.startswith("$"):
                raise ValueError(f"Invalid path segment (must not start with '$'): '{path}'")
            if len(segment) > MAX_SEGMENT_LENGTH:
                raise ValueError(
                    f"Path segment too long (max {MAX_SEGMENT_LENGTH} chars): '{segment}'"
                )
        validated.append(path)

    return validated


def normalize_metadata_paths(
    paths: list[str],
) -> list[str]:
    """Deduplicate paths where one is an ancestor of another.

    If both 'config' and 'config.region' are requested, keep only 'config'
    (the ancestor already includes all descendants).

    Args:
        paths: Validated dot-path list.

    Returns:
        Deduplicated list with descendants removed.
    """
    sorted_paths = sorted(paths, key=lambda p: p.count("."))
    result: list[str] = []
    for path in sorted_paths:
        is_descendant = False
        for accepted in result:
            if path == accepted or path.startswith(accepted + "."):
                is_descendant = True
                break
        if not is_descendant:
            result.append(path)
    return result


def project_metadata(
    metadata: dict[str, Any] | None,
    paths: list[str] | None,
) -> dict[str, Any] | None:
    """Project metadata to only the requested dot-notation paths.

    This is the canonical (oracle) implementation of projection semantics.
    The DB-level $set stage MUST produce the same result for any valid input.

    Args:
        metadata: The full metadata dict, or None.
        paths: Normalized list of dot-paths to keep, or None (= no projection).

    Returns:
        Projected metadata dict, or None if input metadata is None.
        Returns {} if paths is a non-None empty list.
    """
    if paths is None:
        return metadata
    if metadata is None:
        return None
    if not paths:
        return {}

    result: dict[str, Any] = {}
    for path in paths:
        segments = path.split(".")
        # Navigate into the source metadata
        source: Any = metadata
        value_found = True
        for segment in segments:
            if not isinstance(source, dict) or segment not in source:
                value_found = False
                break
            source = source[segment]

        if not value_found:
            continue

        # Build nested result structure
        target = result
        for i, segment in enumerate(segments[:-1]):
            if segment not in target:
                target[segment] = {}
            elif not isinstance(target[segment], dict):
                # Path conflict: a scalar was already placed at this position
                # by a shallower path. normalize_metadata_paths should prevent
                # this, but handle gracefully.
                value_found = False
                break
            target = target[segment]

        if value_found:
            target[segments[-1]] = source

    return result


def build_metadata_set_stage(
    paths: list[str],
) -> dict[str, Any]:
    """Build a MongoDB $set aggregation stage that rebuilds the metadata subdocument.

    The generated stage replaces the entire 'metadata' field with a new object
    containing only the requested paths, using $ifNull to gracefully handle
    missing paths (they produce $$REMOVE so the key is absent from output).

    Args:
        paths: Normalized list of dot-paths.

    Returns:
        A dict suitable as a {"$set": ...} pipeline stage.

    Example:
        paths = ["owner", "config.region"]
        -> {"$set": {"metadata": {"owner": ..., "config": {"region": ...}}}}
    """
    if not paths:
        return {"$set": {"metadata": {}}}

    def _build_nested(
        grouped_paths: dict[str, list[str]],
        prefix: str,
    ) -> dict[str, Any]:
        """Recursively build the $set expression tree."""
        obj: dict[str, Any] = {}
        for key, sub_paths in grouped_paths.items():
            full_path = f"{prefix}.{key}" if prefix else key
            if not sub_paths:
                # Leaf: reference the field directly with $$REMOVE for missing
                obj[key] = {"$ifNull": [f"$metadata.{full_path}", "$$REMOVE"]}
            else:
                # Branch: recurse into nested structure
                nested_groups: dict[str, list[str]] = {}
                for sp in sub_paths:
                    parts = sp.split(".", 1)
                    nested_key = parts[0]
                    remainder = parts[1] if len(parts) > 1 else ""
                    if nested_key not in nested_groups:
                        nested_groups[nested_key] = []
                    if remainder:
                        nested_groups[nested_key].append(remainder)
                obj[key] = _build_nested(nested_groups, full_path)
        return obj

    # Group paths by their first segment
    grouped: dict[str, list[str]] = {}
    for path in paths:
        parts = path.split(".", 1)
        key = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        if key not in grouped:
            grouped[key] = []
        if remainder:
            grouped[key].append(remainder)

    metadata_obj = _build_nested(grouped, "")
    return {"$set": {"metadata": metadata_obj}}


def parse_and_validate_metadata_fields(
    metadata_fields: str | list[str] | None,
) -> list[str] | None:
    """Parse, validate, and normalize metadata_fields — raising HTTP 422 on error.

    Convenience wrapper for route handlers that combines parse + normalize
    and maps ValueError to HTTPException(422).

    Accepts both a comma-separated string and a list of strings (from repeated
    query params), or a mix of both (e.g. ['owner,config', 'limits.rps']).

    Args:
        metadata_fields: Raw query parameter value(s), or None.

    Returns:
        Normalized list of dot-paths, or None if no projection requested.

    Raises:
        HTTPException(422): On invalid input.
    """
    # Normalize list input to a single comma-separated string
    if isinstance(metadata_fields, list):
        metadata_fields = ",".join(metadata_fields)

    try:
        parsed = parse_metadata_fields(metadata_fields)
        if parsed:
            parsed = normalize_metadata_paths(parsed)
        return parsed
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid metadata_fields: {e}",
        ) from e


def flatten_metadata_to_text(metadata: dict[str, Any]) -> str:
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
