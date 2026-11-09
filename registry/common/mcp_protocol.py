"""MCP 2026-07-28 protocol constants and result shaping, defined once.

The auth server, the metrics middleware and the health check read the
protocol version, the ``_meta`` keys and the 2026-07-28 error codes from here.
The Lua router keeps its own copy of the same values at the top of
``docker/lua/virtual_router.lua``, because nginx Lua cannot import Python.

Spec: https://modelcontextprotocol.io/specification/2026-07-28/changelog
"""

from typing import Any

MODERN_PROTOCOL_VERSION: str = "2026-07-28"

# Keys a 2026-07-28 request carries in params._meta.
META_PROTOCOL_VERSION_KEY: str = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO_KEY: str = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES_KEY: str = "io.modelcontextprotocol/clientCapabilities"

# JSON-RPC error codes 2026-07-28 defines (the -32020 to -32099 range is
# reserved for the MCP specification).
HEADER_MISMATCH: int = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY: int = -32021
UNSUPPORTED_PROTOCOL_VERSION: int = -32022
MODERN_ERROR_CODES: frozenset[int] = frozenset(
    {HEADER_MISMATCH, MISSING_REQUIRED_CLIENT_CAPABILITY, UNSUPPORTED_PROTOCOL_VERSION}
)

# Methods whose 2026-07-28 result must carry ttlMs and cacheScope (CacheableResult).
CACHEABLE_RESULT_METHODS: frozenset[str] = frozenset(
    {"tools/list", "prompts/list", "resources/list", "resources/templates/list", "resources/read"}
)

# ttlMs for a result the gateway builds itself and that should not be reused,
# such as a pre-consent list that changes the moment the user connects.
NO_REUSE_TTL_MS: int = 0


def request_meta(
    payload: object,
) -> dict[str, Any]:
    """Return params._meta of a JSON-RPC request, or an empty dict."""
    if not isinstance(payload, dict):
        return {}
    params = payload.get("params")
    meta = params.get("_meta") if isinstance(params, dict) else None
    return meta if isinstance(meta, dict) else {}


def is_modern_version(
    version: object,
) -> bool:
    """Whether a protocol version string names the 2026 revision or later."""
    return isinstance(version, str) and version >= "2026-"


def is_modern_request(
    payload: object,
) -> bool:
    """Whether a JSON-RPC request carries 2026-07-28 metadata in params._meta."""
    return is_modern_version(request_meta(payload).get(META_PROTOCOL_VERSION_KEY))


def shape_result(
    result: dict[str, Any],
    method: str | None,
    payload: object,
    ttl_ms: int = NO_REUSE_TTL_MS,
) -> dict[str, Any]:
    """Add the fields a 2026-07-28 result must carry to a result the gateway built.

    For a 2026-07-28 request this adds ``resultType: "complete"``, and for the
    cacheable methods ``cacheScope: "private"`` (the gateway's answers are per
    user) and ``ttlMs``. A field the result already has is kept. A request
    without 2026-07-28 metadata gets the result unchanged.

    Args:
        result: The result object (the value of the JSON-RPC "result" key).
        method: The JSON-RPC method the result answers.
        payload: The JSON-RPC request, to tell which protocol it speaks.
        ttl_ms: ttlMs for a cacheable result that lacks one.

    Returns:
        The shaped result: a new dict when fields were added, else ``result``.
    """
    if not is_modern_request(payload):
        return result
    shaped = dict(result)
    shaped.setdefault("resultType", "complete")
    if method in CACHEABLE_RESULT_METHODS:
        shaped.setdefault("cacheScope", "private")
        shaped.setdefault("ttlMs", ttl_ms)
    return shaped
