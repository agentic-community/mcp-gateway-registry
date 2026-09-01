"""Pydantic models for the per-user egress credential vault.

These models are the data contract for the SecretStore, the OAuth engine, and
the egress vend path. The vault is the single source of truth for token state;
there is no companion app-DB metadata table.

Security notes:
- ``StoredToken`` is the secret payload held by the SecretStore. It is NEVER
  returned to API consumers; the non-secret view is ``EgressConnection``.
- The vault key is ``(auth_method, user_id, provider, server_path)``:
  ``auth_method`` discriminates per-user identities (e.g. ``oauth2``) from
  non-per-user callers (``network-trusted``/``federation-static``) so an
  operator-chosen static-key name cannot collide with a real IdP username.
"""

from enum import Enum

from pydantic import BaseModel, Field


class EgressAuthMode(str, Enum):
    """How the gateway authenticates to the upstream MCP server on egress."""

    NONE = "none"  # no egress auth
    OPERATOR_CREDENTIAL = "operator_credential"  # shared registered backend credential
    OAUTH_USER = "oauth_user"  # per-user 3LO token from the vault
    OBO_EXCHANGE = "obo_exchange"  # same-IdP OBO token exchange; stateless, no vault
    PAT = "pat"  # per-user static PAT/API-key from the vault


EGRESS_MODES_REQUIRING_OAUTH_CONFIG: frozenset[EgressAuthMode] = frozenset(
    {
        EgressAuthMode.OAUTH_USER,
        EgressAuthMode.OBO_EXCHANGE,
        EgressAuthMode.PAT,
    }
)


def parse_egress_auth_mode(
    value: str,
) -> EgressAuthMode:
    """Parse an egress mode with a stable, actionable validation error.

    Args:
        value: The raw ``egress_auth_mode`` value.

    Returns:
        The matching ``EgressAuthMode`` member.

    Raises:
        ValueError: If ``value`` is not a recognized egress mode; the message
            lists the accepted values.
    """
    try:
        return EgressAuthMode(value)
    except ValueError as exc:
        expected = ", ".join(repr(mode.value) for mode in EgressAuthMode)
        raise ValueError(f"invalid egress_auth_mode {value!r}; expected {expected}") from exc


def validate_operator_credential_backend(
    auth_scheme: str,
    encrypted_credential: str | None,
) -> None:
    """Validate the Backend Authentication fields used by operator mode.

    Args:
        auth_scheme: The server's Backend Authentication scheme.
        encrypted_credential: The stored encrypted credential, if any.

    Raises:
        ValueError: If the scheme is not ``bearer``/``api_key`` or no
            encrypted credential is stored.
    """
    if auth_scheme not in ("bearer", "api_key"):
        raise ValueError(
            "egress_auth_mode='operator_credential' requires Backend "
            "Authentication scheme 'bearer' or 'api_key'"
        )
    if not encrypted_credential:
        raise ValueError("operator_credential requires a stored Backend Authentication credential")


class TokenEndpointAuthStyle(str, Enum):
    """Where the client_secret goes when calling the provider token endpoint.

    ``NONE`` is RFC 7591 ``token_endpoint_auth_method=none``: a PUBLIC client
    with no client_secret at all (e.g. a client minted by an MCP resource
    server's Dynamic Client Registration endpoint, such as Datadog's MCP).
    Only ``client_id`` is sent; the PKCE verifier is the proof of possession,
    so ``use_pkce`` stays mandatory for these providers.
    """

    POST_BODY = "post_body"  # client_id/client_secret in the form body
    BASIC_HEADER = "basic_header"  # HTTP Basic auth header
    NONE = "none"  # public client: client_id only, no secret (PKCE required)


class OAuthProviderConfig(BaseModel):
    """Static, per-provider OAuth wiring.

    Built-ins are code rows in ``providers.PROVIDER_REGISTRY``; custom OIDC
    providers are assembled at runtime from operator-supplied URLs. The
    operator-supplied ``client_id``/``client_secret``/``scopes`` are NOT here --
    those live on the server entry (encrypted) and are passed to the engine.
    """

    name: str = Field(..., description="Provider key, e.g. 'github', 'google', 'custom'")
    display_name: str
    authorize_url: str
    token_url: str
    scope_separator: str = " "
    token_endpoint_auth_style: TokenEndpointAuthStyle = TokenEndpointAuthStyle.POST_BODY
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)
    use_pkce: bool = True
    resource: str | None = Field(
        default=None,
        description="RFC 8707 resource indicator. When set, it is sent as the "
        "``resource`` parameter on BOTH the authorize request AND the "
        "token/refresh requests, binding the issued token to one protected "
        "resource. Required by resource servers that mint per-resource tokens -- "
        "e.g. Atlassian's Rovo MCP, whose Authorization Server rejects a code "
        "minted without it ('Invalid context provided') and whose MCP endpoint "
        "rejects a token whose audience is the generic REST API instead of the "
        "MCP resource. None (default) keeps built-in providers unchanged.",
    )
    token_response_parser: str | None = Field(
        default=None,
        description="Name of a registered parser for non-JSON token responses (e.g. 'github_form').",
    )
    is_builtin: bool = True


class StoredToken(BaseModel):
    """The secret payload held by the SecretStore (never returned to API consumers).

    This is the ONLY place token state lives. There is no companion app-DB row:
    the vault is the single source of truth, addressed by deterministic
    namespacing. Lifecycle metadata (expiry, status, timestamps) travels INSIDE
    this payload so a single vault read returns everything needed to decide
    vend-vs-refresh.
    """

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_at: str | None = Field(default=None, description="ISO8601 access-token expiry.")
    scopes: list[str] = Field(default_factory=list)
    status: str = Field(default="active", description="active | refresh_failed")
    client_id: str | None = Field(
        default=None,
        description="OAuth client_id this token was minted under; checked on vend so a "
        "rotated provider client_id forces re-consent instead of vending a stale token.",
    )
    bound_upstreams: list[str] = Field(
        default_factory=list,
        description="Upstream base URLs (scheme://host[:port]) registered for the "
        "server when this credential was written; the vend requires the request's "
        "destination to be a member, so repointing proxy_pass_url forces re-consent "
        "instead of shipping the credential to a new destination.",
    )
    bound_token_url: str | None = Field(
        default=None,
        description="OAuth token endpoint this credential was minted against; "
        "checked on vend so repointing a custom_token_url forces re-consent instead "
        "of POSTing the refresh_token + client_secret to a new endpoint. None for "
        "pat / legacy entries (skips the check).",
    )
    created_at: str | None = None
    last_refreshed_at: str | None = None


class EgressConnection(BaseModel):
    """Non-secret view of a connection, returned by GET /api/egress-auth/connections.

    Derived at read time from a vault list + per-entry read (status/expiry/scopes
    only); the access/refresh tokens are stripped. Not persisted on its own.
    """

    provider: str
    server_path: str
    scopes: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    status: str = "active"
    last_refreshed_at: str | None = None


class OAuthState(BaseModel):
    """Signed+encrypted, short-lived state for the consent round-trip.

    ``auth_method`` is REQUIRED: the callback is reached via a browser redirect
    and must namespace the stored token to the SAME vault bucket the consent was
    initiated under. It cannot trust the auth context of the callback request
    (the user may have logged out / re-logged-in as a different identity between
    initiate and redirect), so the binding lives in the signed state.

    The whole state (in particular ``pkce_verifier``) MUST be AEAD-encrypted, not
    merely signed, before round-tripping through the provider -- a signed-only
    ``state`` exposes the PKCE verifier in the URL/Referer/logs and defeats PKCE.

    ``client_id`` is bound for audit/iss-checks only; it is NOT a vault-key
    segment (interactive users have an unstable client_id).
    """

    user_id: str
    auth_method: str
    client_id: str = ""
    provider: str
    server_path: str
    session_id: str = ""
    pkce_verifier: str | None = None
    nonce: str
    issued_at: str
