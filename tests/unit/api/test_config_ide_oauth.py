"""Unit tests for the IDE OAuth client_id Connect-config feature.

Covers the registry-wide default setting (ide_oauth_client_id), its exposure in
the admin config view/export, and the per-server model fields (oauth_client_id,
append_mcp_path) that drive the token-less, login-button Connect config.
"""

from registry.api.config_routes import (
    CONFIG_GROUPS,
    _export_as_env,
)
from registry.core.config import settings
from registry.core.schemas import ServerInfo


class TestGlobalIdeOAuthSetting:
    """Registry-wide default (IDE_OAUTH_CLIENT_ID)."""

    def test_setting_exists_with_empty_default(self):
        """The global default exists and is empty unless configured."""
        assert hasattr(settings, "ide_oauth_client_id")

    def test_field_present_in_auth_config_group(self):
        """The admin config view lists the field under the Authentication group."""
        auth_fields = {f[0] for f in CONFIG_GROUPS["auth"]["fields"]}
        assert "ide_oauth_client_id" in auth_fields

    def test_field_not_masked_in_export(self, monkeypatch):
        """A public client_id is non-sensitive, so exports show its value."""
        monkeypatch.setattr(settings, "ide_oauth_client_id", "mcp-gateway")

        output = _export_as_env(include_sensitive=False)

        assert "IDE_OAUTH_CLIENT_ID=mcp-gateway" in output


class TestPerServerConnectFields:
    """Per-server overrides on the ServerInfo model."""

    def test_oauth_client_id_field_defaults_none(self):
        assert "oauth_client_id" in ServerInfo.model_fields
        assert ServerInfo.model_fields["oauth_client_id"].default is None

    def test_append_mcp_path_field_defaults_none(self):
        assert "append_mcp_path" in ServerInfo.model_fields
        assert ServerInfo.model_fields["append_mcp_path"].default is None

    def test_fields_round_trip(self):
        """Values survive validation (not silently dropped)."""
        server = ServerInfo(
            server_name="aws-knowledge",
            path="/aws-knowledge",
            proxy_pass_url="https://knowledge-mcp.example.com",
            oauth_client_id="mcp-gateway",
            append_mcp_path=False,
        )

        assert server.oauth_client_id == "mcp-gateway"
        assert server.append_mcp_path is False
