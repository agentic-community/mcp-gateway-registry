"""The public origin nginx uses for the MCP Apps connector URL (PR #2083)."""

import pytest

from registry.core import nginx_service
from registry.core.config import Settings


@pytest.mark.unit
class TestMcpConnectorOrigin:
    """_mcp_connector_origin returns an origin only for a configured public URL."""

    def test_unset_registry_url_gives_no_connector(self, monkeypatch):
        default = Settings.model_fields["registry_url"].default
        monkeypatch.setattr(nginx_service.settings, "registry_url", default)
        assert nginx_service._mcp_connector_origin() == ""

    @pytest.mark.parametrize(
        ("registry_url", "origin"),
        [
            ("https://gateway.example.com", "https://gateway.example.com"),
            ("https://gateway.example.com/registry/", "https://gateway.example.com"),
            ("http://localhost", "http://localhost"),
            ("http://gw.example:8080", "http://gw.example:8080"),
        ],
    )
    def test_configured_url_gives_its_origin(self, monkeypatch, registry_url, origin):
        monkeypatch.setattr(nginx_service.settings, "registry_url", registry_url)
        assert nginx_service._mcp_connector_origin() == origin

    @pytest.mark.parametrize("registry_url", ["", "gateway.example.com", "ftp://gw.example"])
    def test_non_url_gives_no_connector(self, monkeypatch, registry_url):
        monkeypatch.setattr(nginx_service.settings, "registry_url", registry_url)
        assert nginx_service._mcp_connector_origin() == ""
