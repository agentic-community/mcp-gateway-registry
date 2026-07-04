"""Built-in OAuth provider config table + resolver.

Adding a provider is a config row here, not a new module.

The operator supplies ``client_id``/``client_secret``/``scopes`` at server
registration time (stored on the server entry, encrypted); they are NOT in this
table. A ``custom`` provider is assembled at runtime from the server's
``EgressOAuthConfig.custom_*`` fields.
"""

from registry.egress_auth.schemas import OAuthProviderConfig, TokenEndpointAuthStyle

# Refresh-token enablement differs per provider and is operator-facing config,
# not code: Google access_type=offline (+ prompt=consent), Microsoft/Atlassian
# offline_access scope, GitHub user-to-server expiration app setting, Slack
# token rotation. The extra_authorize_params below cover the query-param half.
PROVIDER_REGISTRY: dict[str, OAuthProviderConfig] = {
    "github": OAuthProviderConfig(
        name="github",
        display_name="GitHub",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",  # nosec B106 - public OAuth token endpoint URL, not a credential
        # GitHub returns the token form-encoded unless Accept: application/json;
        # the engine sends that header, but the parser hook is the safety net.
        token_response_parser="github_form",
        extra_authorize_params={},
    ),
    "google": OAuthProviderConfig(
        name="google",
        display_name="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",  # nosec B106 - public OAuth token endpoint URL, not a credential
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
    ),
    "atlassian": OAuthProviderConfig(
        name="atlassian",
        display_name="Atlassian",
        authorize_url="https://auth.atlassian.com/authorize",
        token_url="https://auth.atlassian.com/oauth/token",  # nosec B106 - public OAuth token endpoint URL, not a credential
        extra_authorize_params={"audience": "api.atlassian.com", "prompt": "consent"},
    ),
    "microsoft": OAuthProviderConfig(
        name="microsoft",
        display_name="Microsoft",
        authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",  # nosec B106 - public OAuth token endpoint URL, not a credential
        extra_authorize_params={"prompt": "consent"},  # offline_access goes in scopes
    ),
    "slack": OAuthProviderConfig(
        name="slack",
        display_name="Slack",
        # mcp.slack.com is an OAuth resource server that requires a USER token
        # (xoxp-), not a bot token (xoxb-). Its published AS metadata
        # (https://mcp.slack.com/.well-known/oauth-authorization-server) points at
        # Slack's dedicated user-token endpoints below -- NOT the classic
        # oauth/v2/authorize + oauth.v2.access pair, which mints a bot token the
        # MCP server rejects with 401. The user endpoints take a plain ``scope``
        # (no ``user_scope`` indirection) and support PKCE S256 + client_secret_post.
        authorize_url="https://slack.com/oauth/v2_user/authorize",
        token_url="https://slack.com/api/oauth.v2.user.access",  # nosec B106 - public OAuth token endpoint URL, not a credential
        # The user-token endpoint returns the token at the top level, but classic
        # oauth.v2.access nests it under authed_user; the parser handles both
        # (it falls back to the top-level access_token when authed_user is absent).
        token_response_parser="slack_nested",
    ),
}


def list_provider_names() -> list[str]:
    """Built-in provider keys plus the 'custom' OIDC option (for the UI dropdown)."""
    return sorted(PROVIDER_REGISTRY.keys()) + ["custom"]


def resolve_provider(egress_oauth: dict) -> OAuthProviderConfig:
    """Resolve a provider config from a server's ``egress_oauth`` sub-model dict.

    Args:
        egress_oauth: The ``EgressOAuthConfig`` as a dict (``.model_dump()`` or
            the persisted mapping). Must carry ``provider``; ``custom`` requires
            ``custom_authorize_url`` and ``custom_token_url``.

    Returns:
        The matching built-in ``OAuthProviderConfig`` or an assembled custom one.

    Raises:
        ValueError: unknown provider, or custom provider missing required URLs.
    """
    name = egress_oauth.get("provider")
    if not name:
        raise ValueError("egress_oauth.provider is required")

    if name == "custom":
        authorize_url = egress_oauth.get("custom_authorize_url")
        token_url = egress_oauth.get("custom_token_url")
        if not authorize_url or not token_url:
            raise ValueError(
                "Custom OIDC provider requires custom_authorize_url and custom_token_url."
            )
        return OAuthProviderConfig(
            name="custom",
            display_name="Custom OIDC",
            is_builtin=False,
            authorize_url=authorize_url,
            token_url=token_url,
            scope_separator=egress_oauth.get("custom_scope_separator") or " ",
            token_endpoint_auth_style=TokenEndpointAuthStyle(
                egress_oauth.get("custom_token_auth_style") or "post_body"
            ),
            use_pkce=True,
        )

    cfg = PROVIDER_REGISTRY.get(name)
    if cfg is None:
        valid = ", ".join(list_provider_names())
        raise ValueError(f"Unknown egress provider {name!r}. Valid providers: {valid}")
    return cfg
