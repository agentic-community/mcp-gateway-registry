"""Test-local dynamic-client-registration provider for the egress-auth unit tests."""

from registry.egress_auth.schemas import OAuthProviderConfig, TokenEndpointAuthStyle

DCR_TEST_PROVIDER = "dcr-test"
DCR_TEST_PRM_URL = "https://mcp.example.com/.well-known/oauth-protected-resource/v2/mcp"
DCR_TEST_RESOURCE = "https://mcp.example.com/v2/mcp"
DCR_TEST_ISSUER = "https://auth.example.com/TENANT"
DCR_TEST_DEFAULT_SCOPES = ["read:me", "read:account", "offline_access", "read:work"]


def make_dcr_test_config(
    **overrides,
) -> OAuthProviderConfig:
    """A requires_dcr provider shaped like a public-client MCP authorization server.

    Test-local so the DCR machinery is exercised without depending on any
    built-in provider row.
    """
    fields = {
        "name": DCR_TEST_PROVIDER,
        "display_name": "DCR test provider",
        "authorize_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/oauth/token",
        "token_endpoint_auth_style": TokenEndpointAuthStyle.NONE,
        "requires_dcr": True,
        "protected_resource_metadata_url": DCR_TEST_PRM_URL,
        "default_scopes": list(DCR_TEST_DEFAULT_SCOPES),
        "required_scopes": ["read:account"],
    }
    fields.update(overrides)
    return OAuthProviderConfig(**fields)
