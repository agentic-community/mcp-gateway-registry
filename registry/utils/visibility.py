"""Shared visibility normalization utilities.

Provides a single source of truth for valid visibility values and
normalizes backward-compatible aliases (e.g. "internal" -> "private").
"""

# Canonical visibility values used across agents, servers, and skills
VALID_VISIBILITY_VALUES: list[str] = ["public", "private", "group-restricted"]

# Aliases that are silently normalized to canonical values
_VISIBILITY_ALIASES: dict[str, str] = {
    "internal": "private",
    "group": "group-restricted",
}


def _normalize_visibility(
    value: str,
) -> str:
    """Normalize a visibility value to its canonical form.

    Accepts backward-compatible aliases:
    - "internal" -> "private"
    - "group" -> "group-restricted"

    Case-insensitive: input is lowercased before normalization.

    Args:
        value: The visibility value to normalize.

    Returns:
        The canonical visibility value (lowercased).
    """
    lowered = value.lower()
    return _VISIBILITY_ALIASES.get(lowered, lowered)


def validate_visibility(
    value: str,
) -> str:
    """Normalize and validate a visibility value.

    Args:
        value: The visibility value to normalize and validate.

    Returns:
        The canonical visibility value.

    Raises:
        ValueError: If the value is not a valid visibility after normalization.
    """
    normalized = _normalize_visibility(value)
    if normalized not in VALID_VISIBILITY_VALUES:
        raise ValueError(f"Visibility must be one of: {', '.join(VALID_VISIBILITY_VALUES)}")
    return normalized
