"""
Pydantic models for audit log records.

This module defines the structured data models for audit events,
including credential masking validators to ensure sensitive data
is never logged in plain text.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def mask_credential(value: str) -> str:
    """
    Mask a credential, emitting no part of its value.

    Audit records persist to a store that may be read more widely than the
    request path, so a credential value must never survive there -- not even a
    suffix. The last characters of a bearer token, session cookie, or API key
    are real key-space (and for short tokens a large fraction of it), so nothing
    from the value is emitted; the fixed marker records only that a credential
    was present. The credential TYPE (session vs bearer) is captured separately
    on the audit Identity, so this loses no diagnostic value.

    Args:
        value: The credential string to mask

    Returns:
        A fixed ``"***"`` marker regardless of the value's length or content.
    """
    if not value:
        return "***"
    return "***"


# Set of sensitive query parameter keys that should be masked by exact match.
# NOTE: Exact-match alone is fragile — a new parameter name (e.g.
# ``auth_credential``, ``client_secret``) silently escapes masking until it is
# added here. The masker below therefore ALSO applies substring matching against
# SENSITIVE_QUERY_PARAM_SUBSTRINGS so future variants fail closed (masked by
# default) rather than fail open (logged in plaintext).
SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "token",
        "password",
        "key",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "auth_credential",
    }
)


# Substrings that, when present anywhere in a query parameter name (case
# insensitive), mark its value as sensitive and force masking. This is the
# fail-closed layer: any current or future parameter whose name contains one of
# these tokens is masked without needing an exact-match entry above.
#
# This list is query-parameter oriented and intentionally DIFFERS from the
# header substring lists (registry.common.log_redaction.SENSITIVE_HEADER_SUBSTRINGS
# and auth_server._SENSITIVE_HEADER_SUBSTRINGS): header-only markers like
# ``cookie``/``jwt``/``bearer``/``session`` appear in header names, not query
# keys, so they are omitted here. ``key`` is deliberately broad -- it also masks
# benign params such as ``sort_key`` or ``partition_key``. That over-redaction
# is intentional (fail closed): do NOT narrow or remove ``key`` to un-mask a
# benign param, or a future ``*_key`` credential would leak in plaintext.
SENSITIVE_QUERY_PARAM_SUBSTRINGS: tuple[str, ...] = (
    "token",
    "password",
    "passwd",
    "secret",
    "credential",
    "auth",
    "apikey",
    "api_key",
    "key",
    "pwd",
)


def _is_sensitive_query_param(name: str) -> bool:
    """Return True if a query parameter name should have its value masked.

    Matching is case-insensitive and combines an exact-match allowlist with
    substring matching so that variant names (``auth_credential``,
    ``client_secret``, ``x_api_key``) are masked by default. Masking is
    fail-closed: when in doubt, mask.

    Args:
        name: The query parameter key.

    Returns:
        True when the value for this key must be masked before logging.
    """
    lowered = name.lower()
    if lowered in SENSITIVE_QUERY_PARAMS:
        return True
    return any(marker in lowered for marker in SENSITIVE_QUERY_PARAM_SUBSTRINGS)


class Identity(BaseModel):
    """
    Identity information for the user making the request.

    Captures authentication context including username, auth method,
    provider, groups, scopes, and credential hints (masked).
    """

    username: str = Field(description="Username or identifier of the requester")
    auth_method: str = Field(description="Authentication method: oauth2, jwt_bearer, anonymous")
    provider: str | None = Field(
        default=None,
        description="Identity provider: cognito, entra_id, keycloak, okta, auth0, pingfederate",
    )
    groups: list[str] = Field(default_factory=list, description="Groups the user belongs to")
    scopes: list[str] = Field(default_factory=list, description="OAuth scopes granted to the user")
    is_admin: bool = Field(default=False, description="Whether the user has admin privileges")
    credential_type: str = Field(
        description="Type of credential: session_cookie, bearer_token, none"
    )
    credential_hint: str | None = Field(
        default=None,
        description="Fixed marker recording that a credential was present; "
        "emits no part of the credential value",
    )

    @field_validator("credential_hint", mode="before")
    @classmethod
    def mask_credential_hint(cls, v: str | None) -> str | None:
        """Mask the credential hint to protect sensitive data."""
        if v:
            return mask_credential(v)
        return v


class Request(BaseModel):
    """
    HTTP request information captured for audit logging.

    Includes method, path, query parameters (with sensitive values masked),
    client IP, and other request metadata.
    """

    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE, etc.")
    path: str = Field(description="Request path")
    query_params: dict[str, Any] = Field(
        default_factory=dict, description="Query parameters (sensitive values masked)"
    )
    client_ip: str = Field(description="Client IP address")
    forwarded_for: str | None = Field(default=None, description="X-Forwarded-For header value")
    user_agent: str | None = Field(default=None, description="User-Agent header value")
    content_length: int | None = Field(
        default=None, description="Content-Length of the request body"
    )

    @field_validator("query_params", mode="before")
    @classmethod
    def mask_sensitive_params(cls, v: dict[str, Any] | None) -> dict[str, Any]:
        """Mask sensitive query parameter values."""
        if not v:
            return {}
        return {
            k: mask_credential(str(val)) if _is_sensitive_query_param(k) else val
            for k, val in v.items()
        }


class Response(BaseModel):
    """
    HTTP response information captured for audit logging.
    """

    status_code: int = Field(description="HTTP status code")
    duration_ms: float = Field(description="Request duration in milliseconds")
    content_length: int | None = Field(
        default=None, description="Content-Length of the response body"
    )


class Action(BaseModel):
    """
    Business-level action information set by route handlers.

    Provides semantic context about what operation was performed
    on what resource.
    """

    operation: str = Field(
        description="Operation type: create, read, update, delete, list, toggle, rate, login, logout, search"
    )
    resource_type: str = Field(
        description="Resource type: server, agent, auth, federation, health, search"
    )
    resource_id: str | None = Field(
        default=None, description="Identifier of the resource being acted upon"
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the action"
    )
    idp_skip_reason: str | None = Field(
        default=None,
        description=(
            "When an IdP admin call was intentionally skipped, the reason. "
            "One of: 'local_only' (is_idp_managed=False), "
            "'forbidden' (IdP 403), 'not_found' (IdP 404)."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional structured dimensions recorded alongside the action "
            "(e.g., {'had_if_match': True}). Used for later analysis."
        ),
    )


class Authorization(BaseModel):
    """
    Authorization decision information for the request.
    """

    decision: str = Field(description="Authorization decision: ALLOW, DENY, NOT_REQUIRED")
    required_permission: str | None = Field(
        default=None, description="Permission required for the action"
    )
    evaluated_scopes: list[str] = Field(
        default_factory=list, description="Scopes that were evaluated for authorization"
    )


class RegistryApiAccessRecord(BaseModel):
    """
    Complete audit record for a Registry API access event.

    This is the primary audit log record type for Phase 1,
    capturing all relevant information about an API request
    for compliance and security review.
    """

    timestamp: datetime = Field(description="When the event occurred (UTC)")
    log_type: str = Field(default="registry_api_access", description="Type of audit log record")
    version: str = Field(default="1.0", description="Schema version for this record type")
    request_id: str = Field(description="Unique identifier for this request")
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracing across services"
    )
    identity: Identity = Field(description="Identity of the requester")
    request: Request = Field(description="HTTP request details")
    response: Response = Field(description="HTTP response details")
    action: Action | None = Field(default=None, description="Business-level action context")
    authorization: Authorization | None = Field(
        default=None, description="Authorization decision details"
    )


# =============================================================================
# MCP Server Access Log Models (Phase 4)
# =============================================================================


class MCPServer(BaseModel):
    """
    MCP server information for audit logging.

    Captures details about the target MCP server being accessed
    through the gateway proxy.
    """

    name: str = Field(description="Name of the MCP server")
    path: str = Field(description="Path/route to the MCP server")
    version: str | None = Field(default=None, description="Version of the MCP server")
    proxy_target: str = Field(description="Target URL the request is proxied to")


class MCPRequest(BaseModel):
    """
    MCP protocol request information for audit logging.

    Captures JSON-RPC method details including tool invocations
    and resource access requests.
    """

    method: str = Field(description="JSON-RPC method name (e.g., tools/call, resources/read)")
    tool_name: str | None = Field(
        default=None, description="Name of the tool being called (for tools/call method)"
    )
    resource_uri: str | None = Field(
        default=None, description="URI of the resource being accessed (for resources/read method)"
    )
    mcp_session_id: str | None = Field(default=None, description="MCP session identifier")
    transport: str = Field(
        default="streamable-http", description="Transport protocol: streamable-http, sse, stdio"
    )
    jsonrpc_id: str | None = Field(default=None, description="JSON-RPC request ID")


class MCPResponse(BaseModel):
    """
    MCP protocol response information for audit logging.

    Captures the outcome of an MCP request including success/error
    status and timing information.
    """

    status: str = Field(description="Response status: success, error, timeout")
    duration_ms: float = Field(description="Request duration in milliseconds")
    error_code: int | None = Field(
        default=None, description="JSON-RPC error code (if status is error)"
    )
    error_message: str | None = Field(
        default=None, description="Error message (if status is error)"
    )


class MCPServerAccessRecord(BaseModel):
    """
    Complete audit record for an MCP server access event.

    This is the audit log record type for Phase 4,
    capturing all relevant information about an MCP protocol
    request proxied through the gateway for compliance and
    security review.
    """

    timestamp: datetime = Field(description="When the event occurred (UTC)")
    log_type: str = Field(default="mcp_server_access", description="Type of audit log record")
    version: str = Field(default="1.0", description="Schema version for this record type")
    request_id: str = Field(description="Unique identifier for this request")
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracing across services"
    )
    identity: Identity = Field(description="Identity of the requester")
    mcp_server: MCPServer = Field(description="Target MCP server details")
    mcp_request: MCPRequest = Field(description="MCP protocol request details")
    mcp_response: MCPResponse = Field(description="MCP protocol response details")
    request: Request | None = Field(
        default=None, description="HTTP request details (client_ip, forwarded_for, user_agent)"
    )


# =============================================================================
# Token Mint Audit Record
# =============================================================================


class TokenMintAuditRecord(BaseModel):
    """Audit record emitted at the token-signing point in the auth server.

    Captures every mint (self-signed and M2M, success and failure) so reviewers
    can answer "which servers were scoped to a token, for whom, and did it succeed".
    No raw token material is ever stored; the username is pre-hashed by the caller.

    Note on ``username_hash``: the caller hashes with SHA-256 and keeps only the
    first 8 hex chars (32 bits, ``user_<8hex>``). This is deliberate for privacy,
    but it is a low-entropy hash: distinct usernames can collide once the audit
    population grows into the tens of thousands (birthday bound ~2^16). Treat the
    hash as a privacy-preserving grouping key for analytics, not a unique user id.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC time the mint was attempted",
    )
    log_type: str = Field(
        default="token_mint",
        description="Discriminator for the audit store",
    )
    version: str = Field(default="1.0", description="Schema version")
    request_id: str = Field(
        ...,
        description="Fresh UUID for this mint; part of the (request_id, log_type) key",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Cross-service trace id propagated from the registry",
    )

    # Who
    username_hash: str = Field(
        ...,
        description="SHA-256 (8 hex) hash of the requesting user; never the raw name",
    )
    auth_method: str = Field(
        ...,
        description="oauth2, network-trusted, etc.",
    )
    provider: str | None = Field(
        default=None,
        description="cognito, keycloak, entra, etc.",
    )
    internal_caller: str = Field(
        ...,
        description="Identity of the internal service that called /internal/tokens",
    )

    # What was minted
    token_kind: str = Field(
        ...,
        description="'user' (unrestricted within scopes) or 'resource' (bound)",
    )
    resource_type: str | None = Field(
        default=None,
        description="For resource-bound tokens: server, agent, peer-registry, etc.",
    )
    resource_id: str | None = Field(
        default=None,
        description="For resource-bound tokens: the resource id, e.g. 'fininfo'",
    )
    token_path: str = Field(
        ...,
        description="'self_signed' or 'm2m' (which signing path produced the token)",
    )
    requested_scopes: list[str] = Field(
        default_factory=list,
        description="Scopes requested for the token",
    )
    expires_in_seconds: int | None = Field(
        default=None,
        description="Token lifetime in seconds (None on failure before computed)",
    )

    # Outcome
    outcome: str = Field(
        ...,
        description="'success' or 'failure'",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Short reason when outcome='failure' (rate_limited, provider_error, ...)",
    )
