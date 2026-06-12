"""Unit tests for the ide_oauth_client_id surfaced via /api/config.

Validates that the pre-registered public OAuth client_id (used to drive the
token-less, login-button Connect config in the frontend) is exposed on the
public config endpoint and the admin config view, defaulting to None.
"""

from registry.api.config_routes import (
    CONFIG_GROUPS,
    _export_as_env,
    get_config,
)
from registry.core.config import settings


class TestIdeOAuthClientIdConfig:
    """ide_oauth_client_id exposure tests."""

    async def test_get_config_returns_client_id_when_set(self, monkeypatch):
        """When set, /api/config advertises the client_id verbatim."""
        monkeypatch.setattr(settings, "ide_oauth_client_id", "mcp-gateway")

        result = await get_config()

        assert result["ide_oauth_client_id"] == "mcp-gateway"

    async def test_get_config_returns_none_when_unset(self, monkeypatch):
        """Empty (default) is surfaced as None so the frontend keeps legacy tokens."""
        monkeypatch.setattr(settings, "ide_oauth_client_id", "")

        result = await get_config()

        assert result["ide_oauth_client_id"] is None

    def test_field_present_in_auth_config_group(self):
        """The admin config view lists the field under the Authentication group."""
        auth_fields = {f[0] for f in CONFIG_GROUPS["auth"]["fields"]}
        assert "ide_oauth_client_id" in auth_fields

    def test_field_is_not_masked_in_export(self, monkeypatch):
        """A public client_id is non-sensitive, so exports show its value."""
        monkeypatch.setattr(settings, "ide_oauth_client_id", "mcp-gateway")

        output = _export_as_env(include_sensitive=False)

        assert "IDE_OAUTH_CLIENT_ID=mcp-gateway" in output
