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

# JSON-RPC code the gateway uses when it refuses a request (issue #2006). It
# sits in the implementation-defined server-error range (-32000 to -32099),
# below the -32020 to -32099 block the MCP specification reserves.
GATEWAY_ACCESS_DENIED: int = -32003

# Reasons the gateway gives in error.data.reason when it refuses a request.
DENIED_NO_SCOPES: str = "no_scopes"
DENIED_BY_SCOPES: str = "not_permitted"
DENIED_ORIGIN: str = "origin_refused"

# Longest tool name copied from the request into a refusal message.
MAX_DENIED_NAME_LENGTH: int = 128


def _denied_message(
    reason: str,
    method: str,
    tool_name: str | None,
    server_path: str,
) -> str:
    """Write the one-line refusal message for access_denied_error()."""
    if reason == DENIED_ORIGIN:
        return (
            f"Access denied: {method} on {server_path} was signed in with a browser "
            "session cookie but did not come from the gateway's own origin. "
            "MCP clients should send a bearer token instead."
        )
    if reason == DENIED_NO_SCOPES:
        return f"Access denied: your account has no scopes that grant access to {server_path}."
    if tool_name:
        return (
            f'Access denied: {method} for tool "{tool_name}" on {server_path} '
            "is not permitted for your account."
        )
    return f"Access denied: {method} on {server_path} is not permitted for your account."


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


def access_denied_error(
    payload: object,
    reason: str,
    method: str,
    tool_name: str | None,
    server_path: str,
) -> dict[str, Any] | None:
    """Build the JSON-RPC error the gateway returns when it refuses a request.

    The message names the refused method, the tool for ``tools/call``, and the
    server, and ``error.data`` carries the same facts for a client to read. It
    never names scopes, groups or token claims.

    Args:
        payload: The JSON-RPC request the gateway refused.
        reason: DENIED_NO_SCOPES, DENIED_BY_SCOPES or DENIED_ORIGIN.
        method: The JSON-RPC method of the request.
        tool_name: The tool name for tools/call, else None.
        server_path: The registered server path, such as "/github/".

    Returns:
        The JSON-RPC error response, or None when the payload is not a single
        JSON-RPC request with an id (a notification gets no response body).
    """
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "id" not in payload:
        return None
    request_id = payload.get("id")
    if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
        request_id = None
    tool = str(tool_name)[:MAX_DENIED_NAME_LENGTH] if tool_name else None
    data: dict[str, Any] = {"reason": reason, "method": method, "server": server_path}
    if tool:
        data["tool"] = tool
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": GATEWAY_ACCESS_DENIED,
            "message": _denied_message(reason, method, tool, server_path),
            "data": data,
        },
    }
