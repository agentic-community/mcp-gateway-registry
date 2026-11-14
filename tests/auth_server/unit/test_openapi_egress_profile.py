"""The auth server's credential-injected hop reaches the openapi-mcp container only by exact identity (issue #2022)."""

from types import SimpleNamespace
from unittest.mock import patch


class TestAuthServerHopProfile:
    def test_container_identity_selects_the_openapi_profile(self):
        from auth_server.server import _egress_upstream_profile

        from registry.utils import url_guard

        settings = SimpleNamespace(
            openapi_mcp_enabled=True,
            openapi_mcp_internal_url="http://openapi-mcp:9876",
            ssrf_allowed_hosts="",
            ssrf_allowed_cidrs="",
            gateway_proxy_allow_private_targets=False,
        )
        with patch.object(url_guard, "settings", settings):
            chosen = _egress_upstream_profile(
                "weather-apis", "http://openapi-mcp:9876/s/weather-apis/mcp", "DEFAULT"
            )
            other = _egress_upstream_profile(
                "weather-apis", "https://mcp.example.com/mcp", "DEFAULT"
            )

        assert chosen is url_guard.BUILTIN_OPENAPI_MCP_PROFILE
        assert other == "DEFAULT"


def test_mcp_proxy_route_still_points_at_mcp_proxy():
    """The helper sits above the route decorator, so the route keeps its handler."""
    from auth_server.server import app

    endpoints = {r.path: r.endpoint.__name__ for r in app.routes if hasattr(r, "endpoint")}

    assert endpoints["/mcp-proxy/{server_name:path}"] == "mcp_proxy"
