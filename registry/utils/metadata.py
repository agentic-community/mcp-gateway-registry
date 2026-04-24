"""Shared metadata utilities for keyword search."""

from typing import Any


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
