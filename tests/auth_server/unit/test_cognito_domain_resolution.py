"""
Unit tests for Cognito hosted-UI domain resolution.

The browser login URLs come from auth_server/oauth2_providers.yml and the
token endpoints from CognitoProvider. Both must resolve COGNITO_DOMAIN the same
way, so an empty or "auto" value cannot build https://.auth... or
https://auto.auth... on one path while the other derives a working prefix.

URLs are compared by parsed hostname, not by substring, to satisfy CodeQL's
py/incomplete-url-substring-sanitization rule.
"""

from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.auth]

POOL_ID: str = "us-east-1_KmP5A3La3"
DERIVED_PREFIX: str = "us-east-1kmp5a3la3"
DERIVED_HOST: str = f"{DERIVED_PREFIX}.auth.us-east-1.amazoncognito.com"
PROVIDERS_YML: Path = Path(__file__).resolve().parents[3] / "auth_server" / "oauth2_providers.yml"


def _hostname_of(
    url: str,
) -> str:
    """Return the host part of an https URL."""
    parsed = urlsplit(url)
    assert parsed.scheme == "https", f"Expected https URL, got: {url}"
    return parsed.hostname or ""


class TestResolveCognitoDomainPrefix:
    """Tests for resolve_cognito_domain_prefix."""

    @pytest.mark.parametrize("configured", [None, "", "  ", "auto", "AUTO"])
    def test_derives_prefix_when_unset_empty_or_auto(self, configured):
        from auth_server.providers.cognito import resolve_cognito_domain_prefix

        assert resolve_cognito_domain_prefix(configured, POOL_ID) == DERIVED_PREFIX

    def test_returns_configured_prefix_unchanged(self):
        from auth_server.providers.cognito import resolve_cognito_domain_prefix

        assert resolve_cognito_domain_prefix("mcp-gw-123", POOL_ID) == "mcp-gw-123"

    def test_returns_empty_when_nothing_to_derive_from(self):
        from auth_server.providers.cognito import resolve_cognito_domain_prefix

        assert resolve_cognito_domain_prefix("", "") == ""


class TestCognitoProviderDomain:
    """CognitoProvider endpoints use the resolved prefix."""

    @pytest.mark.parametrize("configured", [None, "", "auto"])
    def test_auto_and_empty_derive_from_pool_id(self, configured):
        from auth_server.providers.cognito import CognitoProvider

        provider = CognitoProvider(
            user_pool_id=POOL_ID,
            client_id="c",
            client_secret="s",
            region="us-east-1",
            domain=configured,
        )

        assert _hostname_of(provider.token_url) == DERIVED_HOST
        assert _hostname_of(provider.auth_url) == DERIVED_HOST


class TestOauthProvidersYmlCognitoUrls:
    """The browser login URLs in oauth2_providers.yml resolve the same way."""

    @pytest.mark.parametrize(
        ("configured", "expected_host"),
        [
            ("", DERIVED_HOST),
            ("auto", DERIVED_HOST),
            ("my-pool", "my-pool.auth.us-east-1.amazoncognito.com"),
        ],
    )
    def test_login_urls_match_the_provider(self, monkeypatch, configured, expected_host):
        from auth_server.server import substitute_env_vars

        monkeypatch.setenv("COGNITO_DOMAIN", configured)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", POOL_ID)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        cognito = yaml.safe_load(PROVIDERS_YML.read_text())["providers"]["cognito"]

        for field in ("auth_url", "token_url", "user_info_url", "logout_url"):
            assert _hostname_of(substitute_env_vars(cognito[field])) == expected_host, field

    def test_login_urls_derive_when_variable_is_unset(self, monkeypatch):
        from auth_server.server import substitute_env_vars

        monkeypatch.delenv("COGNITO_DOMAIN", raising=False)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", POOL_ID)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        cognito = yaml.safe_load(PROVIDERS_YML.read_text())["providers"]["cognito"]

        assert _hostname_of(substitute_env_vars(cognito["auth_url"])) == DERIVED_HOST
