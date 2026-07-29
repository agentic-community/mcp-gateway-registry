"""Server-controlled audit request identifiers.

The unique key of every audit record is the composite ``(request_id, log_type)``
enforced by a MongoDB unique index. That key MUST be server-controlled: a client
that can choose ``request_id`` can pre-seed a collision and have a later,
malicious action of the same ``log_type`` silently dropped by the unique-index
dedup, leaving no audit trail.

This module centralises two rules so no call site reinvents them:

- ``new_audit_request_id`` always mints a fresh server-side UUID for the durable
  key. Call sites must never derive the key from a client header.
- ``sanitize_correlation_id`` bounds and character-validates any client-supplied
  correlation value (``X-Request-ID`` / ``X-Correlation-ID``). The value is kept
  ONLY as a non-key correlation field for cross-record stitching; it is never
  trusted as the key. Anything malformed, over-length, or empty is dropped
  (returns ``None``) — fail closed rather than store raw, unbounded attacker
  input that later ends up in warm storage and audit exports.
"""

import re
import uuid

# A correlation id is a caller-chosen opaque token used only to stitch related
# records together for humans. Accept the common formats real proxies emit
# (nginx ``$request_id`` = 32 hex chars, UUIDs, ULID/trace-style ids) while
# rejecting anything that could carry an injection payload or bloat the record.
# Deliberately restrictive: hex, dashes, and a small punctuation set are enough
# for every standard request/trace identifier.
#
# Matched with ``re.fullmatch`` (not ``$``) so the anchor is explicit and
# strip-independent: ``$`` matches just before a trailing newline, so a value
# like ``"abc\n"`` would slip past a ``$``-anchored ``match``. ``fullmatch``
# requires the WHOLE string to match, rejecting any embedded/trailing newline or
# control character regardless of the (retained) ``.strip()``.
_CORRELATION_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]+")

# Cap the stored length so an attacker cannot bloat records or logs. 200 chars
# comfortably fits UUIDs, nginx request ids, and W3C traceparent values.
_MAX_CORRELATION_ID_LEN = 200


def new_audit_request_id() -> str:
    """Mint a fresh server-controlled request id for an audit record key.

    Returns:
        A new random UUID4 string. This is the ONLY value that may be used as
        the durable ``request_id`` key of an audit record; never substitute a
        client-supplied header.
    """
    return str(uuid.uuid4())


def sanitize_correlation_id(
    raw: str | None,
) -> str | None:
    """Validate and bound a client-supplied correlation identifier.

    The returned value is safe to store as a NON-key correlation field only.
    It must never be used as the audit record's unique key.

    Args:
        raw: The raw client header value (e.g. ``X-Request-ID`` or
            ``X-Correlation-ID``), or ``None`` when absent.

    Returns:
        The trimmed value when it is a plausible, bounded correlation token;
        ``None`` when it is missing, empty, over-length, or contains characters
        outside the allowed set. Failing closed to ``None`` means a malformed or
        hostile value is simply not recorded rather than stored raw.
    """
    if not raw:
        return None

    candidate = raw.strip()
    if not candidate:
        return None

    if len(candidate) > _MAX_CORRELATION_ID_LEN:
        return None

    if not _CORRELATION_ID_PATTERN.fullmatch(candidate):
        return None

    return candidate
