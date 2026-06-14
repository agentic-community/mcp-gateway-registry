"""Tests for the OAuth discovery helpers in registry/auth/oauth_metadata.py.

These cover the pure helper functions used by the gateway's RFC 9728 PRM and
RFC 8414 AS-metadata routes (issue #989).
"""

from unittest.mock import patch

import pytest

from registry.auth.oauth_metadata import (
    DEFAULT_ADVERTISED_SCOPES,
    WELLKNOWN_PRM_PATH,
    build_canonical_resource_url,
    build_resource_documentation_url,
    build_resource_metadata_url,
    build_www_authenticate_header,
    derive_supported_scopes,
    enforce_https,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]


class TestBuildCanonicalResourceUrl:
    """build_canonical_resource_url() normalizes the gateway's public URL."""

    def test_strips_trailing_slash(self):
        assert build_canonical_resource_url("https://gw.example.com/") == "https://gw.example.com"

    def test_no_change_when_no_trailing_slash(self):
        assert build_canonical_resource_url("https://gw.example.com") == "https://gw.example.com"

    def test_preserves_path_segments(self):
        assert (
            build_canonical_resource_url("https://gw.example.com/registry/")
            == "https://gw.example.com/registry"
        )

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            build_canonical_resource_url("")

    def test_raises_on_missing_scheme(self):
        with pytest.raises(ValueError, match="no scheme"):
            build_canonical_resource_url("gw.example.com")


class TestEnforceHttps:
    """enforce_https() guards against http resource URLs in production."""

    def test_passes_with_https(self):
        enforce_https("https://gw.example.com", https_required=True)

    def test_passes_with_http_when_not_required(self):
        enforce_https("http://localhost:8000", https_required=False)

    def test_raises_with_http_when_required(self):
        with pytest.raises(ValueError, match="not HTTPS"):
            enforce_https("http://gw.example.com", https_required=True)


class TestBuildResourceMetadataUrl:
    """build_resource_metadata_url() must match the PRM `resource` byte-for-byte."""

    def test_appends_well_known_path(self):
        prm_url = build_resource_metadata_url("https://gw.example.com")
        assert prm_url == f"https://gw.example.com{WELLKNOWN_PRM_PATH}"

    def test_no_double_slash(self):
        prm_url = build_resource_metadata_url("https://gw.example.com")
        # Three "://" pairs, then exactly one slash before .well-known
        assert "//.well-known" not in prm_url


class TestBuildWWWAuthenticateHeader:
    """RFC 9728 §5.1 header format."""

    def test_includes_realm_and_resource_metadata(self):
        header = build_www_authenticate_header(
            "https://gw.example.com/.well-known/oauth-protected-resource"
        )
        assert header.startswith("Bearer ")
        assert 'realm="mcp"' in header
        assert (
            'resource_metadata="https://gw.example.com/.well-known/oauth-protected-resource"'
            in header
        )

    def test_byte_for_byte_match_with_prm_endpoint(self):
        """The header's resource_metadata MUST equal the PRM endpoint URL byte-for-byte."""
        resource = "https://gw.example.com"
        prm_url = build_resource_metadata_url(resource)
        header = build_www_authenticate_header(prm_url)

        assert prm_url in header


class TestDeriveSupportedScopes:
    """derive_supported_scopes() builds the PRM `scopes_supported` array.

    Regression guard for the bug where the no-override path advertised every
    `mcp_scopes` `_id` (registry group/scope names) as OAuth scopes, which any
    scope-validating IdP rejects with `invalid_scope`. The default must now be
    the basic IdP-universal OIDC scopes only.
    """

    @pytest.mark.asyncio
    async def test_default_returns_basic_oidc_scopes(self):
        """With no override, only the basic IdP-universal OIDC scopes are
        advertised - NOT the registry's internal scope/group names."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = ""
            scopes = await derive_supported_scopes()

        assert scopes == DEFAULT_ADVERTISED_SCOPES
        assert scopes == ["openid", "email", "profile", "offline_access"]

    @pytest.mark.asyncio
    async def test_default_never_includes_registry_group_names(self):
        """The default must not leak internal authorization/group names (the
        exact bug: names like `registry-admins` are not IdP OAuth scopes)."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = ""
            scopes = await derive_supported_scopes()

        for leaked in ("registry-admins", "mcp-registry-admin", "tla-consumer-empty"):
            assert leaked not in scopes


class TestAdvertisedScopesOverride:
    """`mcp_advertised_scopes` setting overrides the default advertised scopes.

    Operators set this when their IdP exposes/validates a specific scope set."""

    @pytest.mark.asyncio
    async def test_override_replaces_default_scopes(self):
        """When the override is set, it is used verbatim."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = "openid profile email offline_access"
            scopes = await derive_supported_scopes()

        assert scopes == ["openid", "profile", "email", "offline_access"]

    @pytest.mark.asyncio
    async def test_override_preserves_caller_order(self):
        """Operators may want a specific order; the override doesn't sort."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = "zeta alpha mu"
            scopes = await derive_supported_scopes()

        assert scopes == ["zeta", "alpha", "mu"]

    @pytest.mark.asyncio
    async def test_empty_override_falls_back_to_default(self):
        """Empty string is not an override; the basic default is used."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = ""
            scopes = await derive_supported_scopes()

        assert scopes == DEFAULT_ADVERTISED_SCOPES

    @pytest.mark.asyncio
    async def test_whitespace_only_override_falls_back_to_default(self):
        """Whitespace-only override is treated as unset."""
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.mcp_advertised_scopes = "   "
            scopes = await derive_supported_scopes()

        assert scopes == DEFAULT_ADVERTISED_SCOPES


class TestBuildResourceDocumentationUrl:
    """build_resource_documentation_url() defaults to <registry>/docs/oauth or honors override."""

    def test_default_uses_registry_url(self):
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.registry_url = "https://gw.example.com"
            mock_settings.mcp_resource_documentation_url = None

            url = build_resource_documentation_url()

        assert url == "https://gw.example.com/docs/oauth"

    def test_override_takes_precedence(self):
        with patch("registry.auth.oauth_metadata.settings") as mock_settings:
            mock_settings.registry_url = "https://gw.example.com"
            mock_settings.mcp_resource_documentation_url = "https://docs.example.com/mcp/oauth"

            url = build_resource_documentation_url()

        assert url == "https://docs.example.com/mcp/oauth"
