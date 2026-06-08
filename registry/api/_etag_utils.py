"""
Shared weak-ETag helpers for resources that expose optimistic concurrency
via If-Match headers.

These helpers are timestamp-based (epoch milliseconds derived from an
``updated_at`` field, with optional fallback to a ``registered_at`` /
``created_at`` timestamp) so they are reusable across resource types
(agents, servers, etc.) without coupling to a specific Pydantic model.

Weak validators (``W/"..."``) are used because JSON serialization is not
byte-stable, but the ``updated_at`` timestamp reliably changes on every
persisted mutation.
"""

import re
from datetime import datetime

from fastapi import HTTPException

_WEAK_ETAG_PATTERN = re.compile(r'W/"(\d+)"')


def weak_etag_for_timestamp(
    updated_at: datetime | None,
    registered_at_fallback: datetime | None = None,
) -> str:
    """Build a weak ETag from a timestamp (epoch milliseconds).

    Args:
        updated_at: Primary timestamp to derive the ETag from.
        registered_at_fallback: Fallback timestamp if updated_at is None.

    Returns:
        Weak ETag string of the form ``W/"<epoch_ms>"``. Returns
        ``W/"0"`` if both timestamps are None.
    """
    ts = updated_at or registered_at_fallback
    epoch_ms = int(ts.timestamp() * 1000) if ts else 0
    return f'W/"{epoch_ms}"'


def parse_if_match(if_match: str | None) -> int | None:
    """Parse a weak ETag of the form ``W/"<epoch_ms>"`` into its int value.

    Args:
        if_match: Raw If-Match header value.

    Returns:
        The epoch-ms integer, or None if if_match is None.

    Raises:
        HTTPException: 400 on malformed input, including the strong-ETag
            form. Strong form is explicitly rejected rather than silently
            ignored so clients that think they set a precondition see the
            error rather than getting last-write-wins.
    """
    if if_match is None:
        return None
    s = if_match.strip()
    if s.startswith('"') and s.endswith('"'):
        raise HTTPException(
            status_code=400,
            detail='Strong ETag not supported; use weak form W/"<epoch_ms>"',
        )
    m = _WEAK_ETAG_PATTERN.fullmatch(s)
    if not m:
        raise HTTPException(status_code=400, detail="Malformed If-Match header")
    return int(m.group(1))


def updated_ms(
    updated_at: datetime | None,
    registered_at_fallback: datetime | None = None,
) -> int:
    """Return epoch-ms of the timestamp (or fallback, else 0).

    Args:
        updated_at: Primary timestamp.
        registered_at_fallback: Fallback timestamp if updated_at is None.

    Returns:
        Epoch milliseconds, or 0 if both timestamps are None.
    """
    ts = updated_at or registered_at_fallback
    return int(ts.timestamp() * 1000) if ts else 0
