"""Tests that ``register --config`` carries ``append_mcp_path`` to the registry.

Issue #1603: ``POST /api/servers/register`` accepts ``append_mcp_path`` as a form
field (``registry/api/server_routes.py``), but the CLI never sent it, so setting
it in a config file was silently dropped. The registry then appended ``/mcp`` to
upstreams that serve JSON-RPC at their root and 404 on ``/mcp``, and the failure
surfaced as an upstream 404 rather than anything naming the ignored setting.

The value is only meaningful when it is ``False``, which is also the value most
easily lost: ``register_service`` serializes with ``exclude_none=True``, so the
field has to be absent when unset but survive when explicitly false.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.registry_client import InternalServiceRegistration, RegistryClient

# registry_management.py lives under api/ and imports sibling modules by name.
_API_DIR = Path(__file__).resolve().parents[3] / "api"
sys.path.insert(0, str(_API_DIR))

import registry_management  # noqa: E402


def _register_from_config(tmp_path, config: dict) -> InternalServiceRegistration:
    """Run cmd_register against a config file and return the sent registration."""
    config_file = tmp_path / "server-config.json"
    config_file.write_text(registry_management.json.dumps(config))

    mock_client = MagicMock()
    mock_client.register_service.return_value = SimpleNamespace(
        path=config.get("path"), message="ok"
    )

    args = SimpleNamespace(config=str(config_file), overwrite=False)
    with patch.object(registry_management, "_create_client", return_value=mock_client):
        assert registry_management.cmd_register(args) == 0

    return mock_client.register_service.call_args[0][0]


def _base_config(**extra) -> dict:
    config = {
        "path": "/salesforce-headless-360",
        "server_name": "Salesforce Headless 360",
        "description": "Hosted MCP endpoint that serves JSON-RPC at its root.",
        "proxy_pass_url": "https://example.invalid/platform/headless-360",
    }
    config.update(extra)
    return config


def test_register_forwards_append_mcp_path_false(tmp_path):
    """``"append_mcp_path": false`` in the config reaches the registration."""
    registration = _register_from_config(tmp_path, _base_config(append_mcp_path=False))

    assert registration.append_mcp_path is False


def test_register_forwards_append_mcp_path_true(tmp_path):
    """An explicit true is carried through as well, not just the false case."""
    registration = _register_from_config(tmp_path, _base_config(append_mcp_path=True))

    assert registration.append_mcp_path is True


def test_register_leaves_append_mcp_path_unset_when_absent(tmp_path):
    """Omitting it stays unset, so the registry default is not overridden."""
    registration = _register_from_config(tmp_path, _base_config())

    assert registration.append_mcp_path is None


@pytest.fixture
def client() -> RegistryClient:
    return RegistryClient(registry_url="http://localhost", token="dummy-token-1234567890")


class TestAppendMcpPathOnTheWire:
    """The field has to survive ``model_dump(exclude_none=True)``."""

    def _sent_form_data(self, client: RegistryClient, **extra) -> dict:
        response = MagicMock()
        response.json.return_value = {"path": "/svc", "name": "svc", "message": "ok"}

        with patch.object(client, "_make_request", return_value=response) as request:
            client.register_service(InternalServiceRegistration(path="/svc", name="svc", **extra))

        return request.call_args.kwargs["data"]

    def test_false_is_sent(self, client: RegistryClient) -> None:
        """False is not None, so it must appear in the form payload."""
        assert self._sent_form_data(client, append_mcp_path=False)["append_mcp_path"] is False

    def test_absent_when_unset(self, client: RegistryClient) -> None:
        """Unset means absent, letting the registry apply its own default."""
        assert "append_mcp_path" not in self._sent_form_data(client)
