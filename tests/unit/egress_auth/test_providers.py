"""Provider registry + resolver tests."""

import pytest

from registry.egress_auth.providers import (
    PROVIDER_REGISTRY,
    list_provider_names,
    resolve_provider,
)
from registry.egress_auth.schemas import TokenEndpointAuthStyle


@pytest.mark.unit
class TestProviderRegistry:
    @pytest.mark.parametrize("name", ["github", "google", "atlassian", "microsoft", "slack"])
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

    def test_atlassian_is_authv2_dcr_none_style(self):
        # Atlassian Rovo authv2: the gateway DCR-registers its own public PKCE
        # client (NONE style; a static classic-3LO app is rejected at the MCP tool
        # layer). Pin requires_dcr + NONE style + the PRM discovery URL.
        atl = PROVIDER_REGISTRY["atlassian"]
        assert atl.requires_dcr is True
        assert atl.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
        assert atl.use_pkce is True
        assert atl.protected_resource_metadata_url == (
            "https://mcp.atlassian.com/.well-known/oauth-protected-resource/v1/mcp/authv2"
        )

    def test_atlassian_drops_classic_audience(self):
        # audience=api.atlassian.com is the classic-3LO REST audience; authv2
        # rejects it. It must not be sent on the authorize request.
        assert "audience" not in PROVIDER_REGISTRY["atlassian"].extra_authorize_params

    def test_atlassian_default_scopes_are_the_verified_minimal_set(self):
        # Verified end to end against live authv2: this set mints a token and real
        # data tools (getVisibleJiraProjects, getJiraIssue) return 200. Confluence
        # scopes are deliberately excluded -- add them per-server when needed.
        atl = PROVIDER_REGISTRY["atlassian"]
        assert atl.default_scopes == [
            "read:me",
            "read:account",
            "offline_access",
            "read:jira-work",
            "write:jira-work",
        ]

    def test_atlassian_requires_read_account(self):
        # read:account is mandatory for authv2 and its absence fails the authorize
        # request only AFTER consent submission, with an opaque invalid_request.
        # It must be in required_scopes (always unioned in) and not merely in
        # default_scopes, which an explicit operator scope list would override.
        atl = PROVIDER_REGISTRY["atlassian"]
        assert atl.required_scopes == ["read:account"]

    def test_non_atlassian_providers_have_no_default_or_required_scopes(self):
        for name, cfg in PROVIDER_REGISTRY.items():
            if name != "atlassian":
                assert cfg.default_scopes == [], f"{name} unexpectedly has default_scopes"
                assert cfg.required_scopes == [], f"{name} unexpectedly has required_scopes"

    def test_only_dcr_providers_opt_in(self):
        # Every other built-in stays a static confidential client (no DCR).
        for name, cfg in PROVIDER_REGISTRY.items():
            if name != "atlassian":
                assert cfg.requires_dcr is False


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
        # Non-DCR built-ins are always confidential (no NONE style).
        # DCR providers (atlassian) are explicitly allowed to use NONE style.
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
        assert PROVIDER_REGISTRY["atlassian"].resource is None
