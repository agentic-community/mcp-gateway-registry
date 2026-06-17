"""Unit tests for the IDE OAuth client_id Connect-config feature.

Covers the registry-wide default setting (ide_oauth_client_id), its exposure in
the admin config view/export, and the per-server model fields (oauth_client_id,
append_mcp_path) that drive the token-less, login-button Connect config.
"""

from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

from registry.api import server_routes
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


class TestUpdateModelsAcceptConnectFields:
    """PUT/PATCH bodies accept the per-server connect-config overrides.

    Without these fields on the update models (which are extra="forbid"),
    a PUT/PATCH carrying oauth_client_id / append_mcp_path would 422 — i.e.
    there would be no API write path for the values the connect-config GET
    reads back.
    """

    def test_server_update_request_accepts_fields(self):
        from registry.schemas.server_update_models import ServerUpdateRequest

        body = ServerUpdateRequest(
            server_name="AWS Knowledge",
            description="AWS docs MCP",
            oauth_client_id="mcp-gateway",
            append_mcp_path=False,
        )

        assert body.oauth_client_id == "mcp-gateway"
        assert body.append_mcp_path is False

    def test_server_card_patch_accepts_fields(self):
        from registry.schemas.server_update_models import ServerCardPatch

        patch_body = ServerCardPatch(oauth_client_id="mcp-gateway", append_mcp_path=False)
        dumped = patch_body.model_dump(exclude_unset=True)

        assert dumped["oauth_client_id"] == "mcp-gateway"
        assert dumped["append_mcp_path"] is False


class TestConnectConfigResolution:
    """connect-config endpoint resolves the effective oauth_client_id.

    Covers the per-server || global-default fallback chain plus the
    append_mcp_path pass-through.
    """

    @staticmethod
    async def _call(server_info: dict, global_default: str):
        """Invoke the endpoint directly with an admin context (skips ACL)."""
        with (
            patch.object(
                server_routes.server_service,
                "get_server_info",
                AsyncMock(return_value=server_info),
            ),
            patch.object(server_routes, "set_audit_action", MagicMock()),
            patch.object(settings, "ide_oauth_client_id", global_default),
        ):
            return await server_routes.get_server_connect_config(
                request=MagicMock(),
                service_path="aws-knowledge",
                user_context={"is_admin": True},
                _csrf=None,
            )

    async def test_per_server_client_id_wins_over_global_default(self):
        result = await self._call(
            {
                "server_name": "AWS Knowledge",
                "custom_headers_encrypted": [],
                "oauth_client_id": "kagent-public",
                "append_mcp_path": False,
            },
            global_default="global-default-client",
        )

        assert result["oauth_client_id"] == "kagent-public"
        assert result["append_mcp_path"] is False

    async def test_falls_back_to_global_default_when_unset(self):
        result = await self._call(
            {"server_name": "AWS Knowledge", "custom_headers_encrypted": []},
            global_default="global-default-client",
        )

        assert result["oauth_client_id"] == "global-default-client"
        # Absent per-server override → None (auto-detect downstream).
        assert result["append_mcp_path"] is None

    async def test_none_when_neither_set(self):
        result = await self._call(
            {"server_name": "AWS Knowledge", "custom_headers_encrypted": []},
            global_default="",
        )

        assert result["oauth_client_id"] is None

    async def test_append_mcp_path_true_round_trips_from_stored_doc(self):
        """A stored append_mcp_path=True passes through connect-config unchanged."""
        result = await self._call(
            {
                "server_name": "AWS Knowledge",
                "custom_headers_encrypted": [],
                "append_mcp_path": True,
            },
            global_default="",
        )

        assert result["append_mcp_path"] is True

    async def test_append_mcp_path_false_round_trips_from_stored_doc(self):
        """A stored append_mcp_path=False passes through (distinct from None)."""
        result = await self._call(
            {
                "server_name": "AWS Knowledge",
                "custom_headers_encrypted": [],
                "append_mcp_path": False,
            },
            global_default="",
        )

        # Must be exactly False, not None - the endpoint must preserve the
        # stored boolean so the frontend can strip /mcp for root-endpoint servers.
        assert result["append_mcp_path"] is False
