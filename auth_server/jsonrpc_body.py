"""Strict parsing for the JSON-RPC request bodies the gateway authorizes.

The auth server reads ``method`` and ``params.name`` from a request body to
decide whether the caller may make that call, then forwards the same bytes to
the MCP backend. Both sides must read the same values from those bytes.

Python's ``json.loads`` matches object keys exactly and keeps the last of two
identical keys. Some backend JSON decoders match keys without regard to case,
so to them ``"Name"`` and ``"name"`` are the same member. A body that holds two
spellings of one member can therefore mean one call to the gateway and another
to the backend.

``parse_jsonrpc_body`` refuses any body that is ambiguous in this way, so every
body the gateway authorizes has a single reading:

* No JSON object, at any depth, may repeat a key exactly.
* In the JSON-RPC envelope and in ``params``, no two keys may be equal after
  Unicode case folding (``str.casefold``), and a key that folds to a member
  name the gateway reads must be spelled exactly as that member.

``str.casefold`` folds a superset of what the case-insensitive decoders fold,
including the non-ASCII forms U+017F (long s) and U+212A (Kelvin sign). The
check may refuse a few bodies a lenient decoder would also accept, but it never
lets one through that a lenient decoder reads differently.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Envelope members the gateway reads when it authorizes a request.
ENVELOPE_MEMBER_NAMES: frozenset[str] = frozenset(
    {"jsonrpc", "id", "method", "params", "tool", "name"}
)

# ``params`` members the gateway reads or strips, or that select what a backend
# runs. ``requestState`` and ``inputResponses`` carry the gateway's own consent
# round trip (see ``_strip_gateway_consent_retry`` in server.py).
PARAMS_MEMBER_NAMES: frozenset[str] = frozenset(
    {
        "name",
        "arguments",
        "uri",
        "_meta",
        "tool",
        "method",
        "cursor",
        "requestState",
        "inputResponses",
    }
)

# Longest key text included in a log line; keys are caller input.
MAX_LOGGED_KEY_LENGTH: int = 64


class AmbiguousJsonRpcBodyError(ValueError):
    """Raised when a request body could be read two different ways."""


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a dict from decoded pairs, refusing an exactly repeated key.

    Used as the ``object_pairs_hook`` of ``json.loads``, so it runs for every
    object in the document, nested ones included.

    Args:
        pairs: The object's members in document order.

    Returns:
        The members as a dict.

    Raises:
        AmbiguousJsonRpcBodyError: When a key appears more than once.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AmbiguousJsonRpcBodyError(
                f"duplicate JSON object key {key[:MAX_LOGGED_KEY_LENGTH]!r}"
            )
        result[key] = value
    return result


def _find_ambiguous_key(
    members: dict[str, Any],
    known_names: frozenset[str],
) -> str | None:
    """Return the first key that a case-insensitive decoder could misread.

    Args:
        members: One decoded JSON object, already free of exact duplicates.
        known_names: The member names the gateway reads, spelled exactly.

    Returns:
        The offending key, or None when every key has a single reading.
    """
    known_folded = {name.casefold() for name in known_names}
    seen_folded: set[str] = set()
    for key in members:
        folded = key.casefold()
        if folded in seen_folded:
            return key
        seen_folded.add(folded)
        if folded in known_folded and key not in known_names:
            return key
    return None


def _check_message(
    message: Any,
) -> None:
    """Check one JSON-RPC message's envelope and params for ambiguous keys.

    Args:
        message: One decoded message. Values that are not objects are left to
            the caller, which already refuses them.

    Raises:
        AmbiguousJsonRpcBodyError: When the envelope or params is ambiguous.
    """
    if not isinstance(message, dict):
        return

    envelope_key = _find_ambiguous_key(message, ENVELOPE_MEMBER_NAMES)
    if envelope_key is not None:
        raise AmbiguousJsonRpcBodyError(
            f"ambiguous JSON-RPC envelope key {envelope_key[:MAX_LOGGED_KEY_LENGTH]!r}"
        )

    params = message.get("params")
    if not isinstance(params, dict):
        return
    params_key = _find_ambiguous_key(params, PARAMS_MEMBER_NAMES)
    if params_key is not None:
        raise AmbiguousJsonRpcBodyError(
            f"ambiguous JSON-RPC params key {params_key[:MAX_LOGGED_KEY_LENGTH]!r}"
        )


def parse_jsonrpc_body(
    body: str | bytes,
) -> Any:
    """Decode a JSON-RPC request body, refusing any body with two readings.

    Use this, never a bare ``json.loads``, wherever the gateway reads a request
    body to make an authorization decision.

    Args:
        body: The request body as text, or as UTF-8 bytes.

    Returns:
        The decoded JSON value. It is usually a dict, but the caller must still
        check the type, because a valid body may be any JSON value.

    Raises:
        UnicodeDecodeError: When ``body`` is bytes that are not UTF-8.
        json.JSONDecodeError: When the body is not valid JSON.
        AmbiguousJsonRpcBodyError: When a key is repeated, or the envelope or
            params holds a key a case-insensitive decoder would read
            differently.

    Example:
        >>> parse_jsonrpc_body('{"jsonrpc":"2.0","id":1,"method":"ping"}')
        {'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}
    """
    text = body.decode("utf-8") if isinstance(body, bytes) else body
    payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)

    if isinstance(payload, list):
        for message in payload:
            _check_message(message)
    else:
        _check_message(payload)
    return payload
