"""Provider registry + resolver tests."""

import pytest

from registry.egress_auth.providers import (
    PROVIDER_REGISTRY,
    list_provider_names,
    resolve_provider,
)
from registry.egress_auth.schemas import OAuthProviderConfig, TokenEndpointAuthStyle


@pytest.mark.unit
class TestProviderRegistry:
    @pytest.mark.parametrize(
        "name", ["github", "google", "atlassian", "atlassian-rovo", "microsoft", "slack"]
    )
    def test_builtin_present_and_wellformed(self, name):
        cfg = PROVIDER_REGISTRY[name]
        assert cfg.name == name
        assert cfg.authorize_url.startswith("https://")
        assert cfg.token_url.startswith("https://")

    def test_list_includes_custom(self):
        names = list_provider_names()
        assert "custom" in names
        assert "github" in names

    def test_github_has_form_parser(self):
        assert PROVIDER_REGISTRY["github"].token_response_parser == "github_form"

    def test_slack_has_nested_parser(self):
        assert PROVIDER_REGISTRY["slack"].token_response_parser == "slack_nested"

    def test_slack_uses_user_token_endpoints(self):
        # mcp.slack.com requires a USER token (xoxp-), not a bot token (xoxb-).
        # Its published AS metadata points at the v2_user endpoints; the classic
        # oauth/v2/authorize + oauth.v2.access pair mints a bot token the MCP
        # server rejects with 401. Pin the user-token endpoints so we don't
        # regress to the bot-token flow.
        slack = PROVIDER_REGISTRY["slack"]
        assert slack.authorize_url == "https://slack.com/oauth/v2_user/authorize"
        assert slack.token_url == "https://slack.com/api/oauth.v2.user.access"

    def test_google_offline_params(self):
        params = PROVIDER_REGISTRY["google"].extra_authorize_params
        assert params.get("access_type") == "offline"
        assert params.get("prompt") == "consent"

    def test_atlassian_is_the_classic_3lo_provider(self):
        # Existing deployments configured ``atlassian`` with a classic 3LO app
        # (confidential client, api.atlassian.com audience). Adding the Rovo MCP
        # provider must not change this row, or their refresh and consent break.
        assert PROVIDER_REGISTRY["atlassian"] == OAuthProviderConfig(
            name="atlassian",
            display_name="Atlassian",
            authorize_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",  # nosec B106 - public OAuth token endpoint URL, not a credential
            extra_authorize_params={"audience": "api.atlassian.com", "prompt": "consent"},
        )


@pytest.mark.unit
class TestAtlassianRovoProvider:
    """Atlassian Rovo MCP v2: a DCR-registered public PKCE client."""

    def test_is_a_dcr_none_style_client(self):
        rovo = PROVIDER_REGISTRY["atlassian-rovo"]
        assert rovo.requires_dcr is True
        assert rovo.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
        assert rovo.use_pkce is True

    def test_uses_atlassian_endpoints(self):
        rovo = PROVIDER_REGISTRY["atlassian-rovo"]
        assert rovo.authorize_url == "https://auth.atlassian.com/authorize"
        assert rovo.token_url == "https://auth.atlassian.com/oauth/token"

    def test_discovers_from_the_v2_protected_resource_metadata(self):
        rovo = PROVIDER_REGISTRY["atlassian-rovo"]
        assert rovo.protected_resource_metadata_url == (
            "https://mcp.atlassian.com/.well-known/oauth-protected-resource/v2/mcp"
        )
        assert rovo.registration_url is None

    def test_sends_prompt_but_no_audience_or_resource(self):
        # The classic api.atlassian.com audience does not apply to MCP tokens,
        # and the authorization server rejects an RFC 8707 resource.
        rovo = PROVIDER_REGISTRY["atlassian-rovo"]
        assert rovo.extra_authorize_params == {"prompt": "consent"}
        assert rovo.resource is None

    def test_default_scopes_are_the_v2_agent_interface_set(self):
        assert PROVIDER_REGISTRY["atlassian-rovo"].default_scopes == [
            "read:me",
            "read:account",
            "offline_access",
            "read:jira:agent-interface",
            "write:jira:agent-interface",
            "search:jira:agent-interface",
        ]

    def test_requires_read_account(self):
        # read:account is mandatory and its absence fails only after the user
        # submits consent. It must be in required_scopes (always unioned in), not
        # only in default_scopes, which an explicit operator list overrides.
        assert PROVIDER_REGISTRY["atlassian-rovo"].required_scopes == ["read:account"]

    def test_other_providers_have_no_default_or_required_scopes(self):
        for name, cfg in PROVIDER_REGISTRY.items():
            if name != "atlassian-rovo":
                assert cfg.default_scopes == [], f"{name} unexpectedly has default_scopes"
                assert cfg.required_scopes == [], f"{name} unexpectedly has required_scopes"

    def test_only_atlassian_rovo_uses_dcr(self):
        # Every other built-in stays a static operator client.
        for name, cfg in PROVIDER_REGISTRY.items():
            if name != "atlassian-rovo":
                assert cfg.requires_dcr is False, name


@pytest.mark.unit
class TestResolveProvider:
    def test_resolve_builtin(self):
        cfg = resolve_provider({"provider": "github"})
        assert cfg.name == "github" and cfg.is_builtin

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown egress provider"):
            resolve_provider({"provider": "bogus"})

    def test_missing_provider_raises(self):
        with pytest.raises(ValueError, match="provider is required"):
            resolve_provider({})

    def test_custom_requires_urls(self):
        with pytest.raises(ValueError, match="custom_authorize_url and custom_token_url"):
            resolve_provider({"provider": "custom"})

    def test_custom_assembled(self):
        cfg = resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://idp.example/authorize",
                "custom_token_url": "https://idp.example/token",
                "custom_scope_separator": ",",
                "custom_token_auth_style": "basic_header",
            }
        )
        assert cfg.name == "custom" and not cfg.is_builtin
        assert cfg.authorize_url == "https://idp.example/authorize"
        assert cfg.scope_separator == ","
        assert cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.BASIC_HEADER
        assert cfg.use_pkce is True

    def test_custom_public_client_none_style(self):
        # token_endpoint_auth_method=none (RFC 7591 public client, e.g. a
        # DCR-minted Datadog MCP client): the style resolves and PKCE stays on.
        cfg = resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://app.datadoghq.com/oauth2/v1/authorize",
                "custom_token_url": "https://app.datadoghq.com/api/v2/oauth2/token",
                "custom_token_auth_style": "none",
            }
        )
        assert cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
        assert cfg.use_pkce is True

    def test_custom_invalid_auth_style_raises(self):
        with pytest.raises(ValueError):
            resolve_provider(
                {
                    "provider": "custom",
                    "custom_authorize_url": "https://idp/auth",
                    "custom_token_url": "https://idp/token",
                    "custom_token_auth_style": "bogus_style",
                }
            )

    def test_builtin_providers_are_confidential(self):
        # A built-in with a static operator app must never resolve to the
        # public-client style. Only a DCR provider registers its own public client.
        for name, cfg in PROVIDER_REGISTRY.items():
            if not cfg.requires_dcr:
                assert cfg.token_endpoint_auth_style != TokenEndpointAuthStyle.NONE, name

    def test_custom_resource_threaded(self):
        # RFC 8707 resource indicator is carried onto the resolved config.
        cfg = resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://auth.atlassian.com/authorize",
                "custom_token_url": "https://auth.atlassian.com/oauth/token",
                "custom_resource": "https://mcp.atlassian.com/v1/mcp/authv2",
            }
        )
        assert cfg.resource == "https://mcp.atlassian.com/v1/mcp/authv2"

    def test_custom_resource_absent_is_none(self):
        cfg = resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://idp/auth",
                "custom_token_url": "https://idp/token",
            }
        )
        assert cfg.resource is None

    def test_builtin_has_no_resource(self):
        # Built-ins never carry a resource indicator (keeps their flow unchanged).
        for name, cfg in PROVIDER_REGISTRY.items():
            assert cfg.resource is None, name
