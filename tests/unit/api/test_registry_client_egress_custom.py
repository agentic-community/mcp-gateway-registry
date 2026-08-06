"""Unit tests for the custom-OIDC fields on egress-auth configuration.

``POST /api/servers/{path}/egress-auth`` accepts five ``custom_*`` fields, and the
server's ``resolve_provider`` REQUIRES two of them when ``provider="custom"``
(rejecting the config with "custom requires custom_authorize_url and
custom_token_url"). The frontend server-edit modal already exposes the URLs, but
``RegistryClient.configure_egress_auth`` and the ``egress-configure`` CLI command
did not forward any of them -- so ``--provider custom`` was accepted by argparse
and then always failed server-side.

These lock in that the fields reach the request body, and that the CLI parses the
corresponding flags, so the CLI/UI parity gap cannot silently reopen.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.registry_client import RegistryClient

# registry_management.py lives under api/ and imports sibling modules by name.
_API_DIR = Path(__file__).resolve().parents[3] / "api"
sys.path.insert(0, str(_API_DIR))

SALESFORCE_AUTHORIZE = "https://example.my.salesforce.com/services/oauth2/authorize"
# nosec B105 - test URL, not a credential
SALESFORCE_TOKEN = "https://example.my.salesforce.com/services/oauth2/token"


@pytest.fixture
def client() -> RegistryClient:
    """A RegistryClient pointed at a dummy URL with a dummy token."""
    return RegistryClient(registry_url="http://localhost", token="dummy-token-1234567890")


def _sent_body(mock_req: MagicMock) -> dict:
    """The JSON body passed to the mocked _make_request."""
    return mock_req.call_args.kwargs["data"]


class TestConfigureEgressAuthCustomFields:
    """Tests for configure_egress_auth custom-OIDC passthrough."""

    def test_custom_urls_reach_the_request_body(
        self,
        client: RegistryClient,
    ) -> None:
        """The two REQUIRED custom URLs are forwarded for provider=custom.

        Without these the server rejects every provider=custom config, which is
        the bug this guards.
        """
        response = MagicMock()
        response.json.return_value = {"path": "/sf", "egress_provider": "custom"}

        with patch.object(client, "_make_request", return_value=response) as mock_req:
            client.configure_egress_auth(
                server_path="/sf",
                mode="oauth_user",
                provider="custom",
                client_id="cid",
                client_secret="csec",  # nosec B106 - dummy test value
                scopes=["mcp_api", "refresh_token"],
                custom_authorize_url=SALESFORCE_AUTHORIZE,
                custom_token_url=SALESFORCE_TOKEN,
            )

        body = _sent_body(mock_req)
        assert body["custom_authorize_url"] == SALESFORCE_AUTHORIZE
        assert body["custom_token_url"] == SALESFORCE_TOKEN
        # The pre-existing fields must still be sent.
        assert body["egress_auth_mode"] == "oauth_user"
        assert body["egress_provider"] == "custom"
        assert body["scopes"] == ["mcp_api", "refresh_token"]

    def test_all_five_custom_fields_are_forwarded(
        self,
        client: RegistryClient,
    ) -> None:
        """Every custom_* field the API accepts is passed through."""
        response = MagicMock()
        response.json.return_value = {}

        with patch.object(client, "_make_request", return_value=response) as mock_req:
            client.configure_egress_auth(
                server_path="/sf",
                mode="oauth_user",
                provider="custom",
                custom_authorize_url=SALESFORCE_AUTHORIZE,
                custom_token_url=SALESFORCE_TOKEN,
                custom_scope_separator=",",
                custom_token_auth_style="basic_header",
                custom_resource="https://api.example.com/mcp",
            )

        body = _sent_body(mock_req)
        assert body["custom_scope_separator"] == ","
        assert body["custom_token_auth_style"] == "basic_header"
        assert body["custom_resource"] == "https://api.example.com/mcp"

    def test_custom_fields_omitted_when_not_supplied(
        self,
        client: RegistryClient,
    ) -> None:
        """A built-in provider config carries no custom_* keys.

        The fields follow the existing ``is not None`` convention, so omitting
        them must not send nulls that could overwrite stored values on edit.
        """
        response = MagicMock()
        response.json.return_value = {}

        with patch.object(client, "_make_request", return_value=response) as mock_req:
            client.configure_egress_auth(
                server_path="/github",
                mode="oauth_user",
                provider="github",
                client_id="cid",
            )

        body = _sent_body(mock_req)
        assert not [k for k in body if k.startswith("custom_")]

    def test_endpoint_and_method_unchanged(
        self,
        client: RegistryClient,
    ) -> None:
        """The custom fields do not alter the request target."""
        response = MagicMock()
        response.json.return_value = {}

        with patch.object(client, "_make_request", return_value=response) as mock_req:
            client.configure_egress_auth(
                server_path="/sf",
                mode="oauth_user",
                provider="custom",
                custom_authorize_url=SALESFORCE_AUTHORIZE,
                custom_token_url=SALESFORCE_TOKEN,
            )

        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["endpoint"] == "/api/servers/sf/egress-auth"


class TestEgressConfigureCommandWiring:
    """Tests that cmd_egress_configure forwards the custom-OIDC args.

    The parser is built inside ``main()``, so these exercise the command handler
    directly with a SimpleNamespace (the convention used by
    test_registry_management_m2m_secret.py). That covers the wiring this change
    adds: args -> client kwargs.
    """

    @staticmethod
    def _args(**overrides) -> SimpleNamespace:
        """Args as argparse would produce them, with custom_* defaulting to None."""
        base = {
            "path": "/sf",
            "mode": "oauth_user",
            "provider": "custom",
            "client_id": "cid",
            "client_secret": "csec",
            "scopes": "mcp_api,refresh_token",
            "target_audience": None,
            "custom_authorize_url": None,
            "custom_token_url": None,
            "custom_scope_separator": None,
            "custom_token_auth_style": None,
            "custom_resource": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_command_forwards_custom_args_to_client(self) -> None:
        """Every custom_* arg reaches configure_egress_auth as a kwarg."""
        import registry_management

        mock_client = MagicMock()
        mock_client.configure_egress_auth.return_value = {"path": "/sf"}
        args = self._args(
            custom_authorize_url=SALESFORCE_AUTHORIZE,
            custom_token_url=SALESFORCE_TOKEN,
            custom_scope_separator=",",
            custom_token_auth_style="basic_header",
            custom_resource="https://api.example.com/mcp",
        )

        with patch.object(registry_management, "_create_client", return_value=mock_client):
            rc = registry_management.cmd_egress_configure(args)

        assert rc == 0
        kwargs = mock_client.configure_egress_auth.call_args.kwargs
        assert kwargs["custom_authorize_url"] == SALESFORCE_AUTHORIZE
        assert kwargs["custom_token_url"] == SALESFORCE_TOKEN
        assert kwargs["custom_scope_separator"] == ","
        assert kwargs["custom_token_auth_style"] == "basic_header"
        assert kwargs["custom_resource"] == "https://api.example.com/mcp"
        # Pre-existing args must still be forwarded.
        assert kwargs["mode"] == "oauth_user"
        assert kwargs["provider"] == "custom"
        assert kwargs["scopes"] == ["mcp_api", "refresh_token"]

    def test_omitted_custom_args_forward_as_none(self) -> None:
        """Unset flags pass None, which the client then skips entirely."""
        import registry_management

        mock_client = MagicMock()
        mock_client.configure_egress_auth.return_value = {}

        with patch.object(registry_management, "_create_client", return_value=mock_client):
            rc = registry_management.cmd_egress_configure(self._args(provider="github"))

        assert rc == 0
        kwargs = mock_client.configure_egress_auth.call_args.kwargs
        assert all(kwargs[k] is None for k in kwargs if k.startswith("custom_"))
