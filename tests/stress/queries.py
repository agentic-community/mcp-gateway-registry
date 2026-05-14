"""Curated query set for Phase 2 semantic-search measurement.

Loads and validates `queries.json` (or any caller-supplied path). The query
file is a JSON array of objects matching `Query` below.

`expected_entity_types` values must match the registry's `EntityType` enum
(see `registry/api/search_routes.py`): `mcp_server`, `tool`, `a2a_agent`,
`skill`, `virtual_server`. The LLD's example uses `agent` but the real
enum value is `a2a_agent`; we use the enum values here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["mcp_server", "tool", "a2a_agent", "skill", "virtual_server"]


class Query(BaseModel):
    """A single curated semantic-search query."""

    id: str = Field(..., min_length=1, description="Stable identifier, e.g. 'server-01'.")
    query: str = Field(..., min_length=1, max_length=512, description="Search text.")
    expected_entity_types: list[EntityType] = Field(
        ...,
        min_length=1,
        description="Entity types the query should reasonably match.",
    )


def load_queries(path: Path) -> list[Query]:
    """Load and validate the curated query set.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file is empty, not a list, or fails validation
    """
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Queries file must be a JSON array, got {type(raw).__name__}: {path}")
    if not raw:
        raise ValueError(f"Queries file is empty: {path}")

    queries = [Query.model_validate(item) for item in raw]

    seen: set[str] = set()
    for q in queries:
        if q.id in seen:
            raise ValueError(f"Duplicate query id in {path}: {q.id}")
        seen.add(q.id)

    return queries


def default_queries_path() -> Path:
    """Path to the bundled `queries.json` next to this module."""
    return Path(__file__).resolve().parent / "queries.json"
