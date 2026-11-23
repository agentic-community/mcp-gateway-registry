"""mcp_proxy sends every upstream hop through the SSRF-guarded pooled client.

The ordinary (no injected credential) hop used a bare httpx client, so a backend
hostname that rebound after registration could reach a private or metadata
address. The hop now uses the profile the registry's health check uses for the
same registration. These tests drive the real handler with the client stubbed,
like test_mcp_apps_domain.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi.testclient import TestClient  # noqa: E402

import auth_server.server as server  # noqa: E402
from registry.exceptions import UrlValidationError  # noqa: E402
from registry.utils.url_guard import (  # noqa: E402
    BUILTIN_AIREGISTRY_TOOLS_PROFILE,
    PROXY_PROFILE,
    proxy_profile_for_mcp_hop,
)
from tests.auth_server.unit.test_mcp_apps_domain import (  # noqa: E402
    SERVER_NAME,
    UPSTREAM_URL,
    _patch_scope_repo_allow_all,
    _upstream,
)

pytestmark = pytest.mark.unit

POOLED_CLIENT_FACTORY: str = "registry.utils.url_guard.shared_guarded_async_client"
BUILTIN_SERVER: str = "airegistry-tools"
BUILTIN_TARGET: str = "http://mcpgw-server:8003/"
OK_BODY: bytes = b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'


def _token_headers(
    server_name: str,
    upstream_url: str,
    virtual: bool = False,
) -> dict[str, str]:
    """Mint the X-Internal-Token nginx's /validate hop would forward."""
    from auth_server.internal_request_token import mint_mcp_proxy_token

    token = mint_mcp_proxy_token(
        subject="test-user",
        scopes=["admin:all"],
        server_name=server_name,
        upstream_url=upstream_url,
        virtual_backend=virtual,
    )
    return {
        "X-Internal-Token": token,
        "X-Registered-Server-Path": "/" + server_name,
        "X-Registered-Route-Mode": "virtual" if virtual else "direct",
    }


def _post_tools_list(
    server_name: str = SERVER_NAME,
    upstream_url: str = UPSTREAM_URL,
    route_suffix: str = "/mcp",
    virtual: bool = False,
):
    with (
        _patch_scope_repo_allow_all(),
        patch.object(server, "_read_mcp_filter_enabled", return_value=False),
        patch.object(server, "_get_blocked_tools", new=AsyncMock(return_value=set())),
    ):
        return TestClient(server.app).post(
            f"/mcp-proxy/{server_name}{route_suffix}",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers=_token_headers(server_name, upstream_url, virtual),
        )


def _client_streaming(
    upstream: MagicMock,
) -> MagicMock:
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=upstream)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)
    client.aclose = AsyncMock()
    return client


def test_ordinary_hop_uses_the_health_check_profile():
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client) as factory:
        response = _post_tools_list()

    assert response.status_code == 200
    assert factory.call_args.kwargs["profile"] is PROXY_PROFILE


def test_virtual_route_to_an_ordinary_backend_is_pinned_to_the_signed_endpoint():
    """A non-egress virtual backend now reaches the hop, and connects to the signed URL."""
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client) as factory:
        response = _post_tools_list(route_suffix="/", virtual=True)

    assert response.status_code == 200
    assert factory.call_args.kwargs["profile"] is PROXY_PROFILE
    assert client.stream.call_args.args[1].rstrip("/") == UPSTREAM_URL


def test_virtual_route_to_the_builtin_uses_the_builtin_profile():
    """A virtual route signs the built-in's endpoint, not its base target."""
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client) as factory:
        response = _post_tools_list(
            BUILTIN_SERVER, "http://mcpgw-server:8003/mcp", route_suffix="/", virtual=True
        )

    assert response.status_code == 200
    assert factory.call_args.kwargs["profile"] is BUILTIN_AIREGISTRY_TOOLS_PROFILE


def test_builtin_registry_tools_hop_uses_the_builtin_profile():
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client) as factory:
        response = _post_tools_list(BUILTIN_SERVER, BUILTIN_TARGET)

    assert response.status_code == 200
    assert factory.call_args.kwargs["profile"] is BUILTIN_AIREGISTRY_TOOLS_PROFILE


def test_builtin_path_with_a_non_builtin_outbound_url_is_refused_before_connecting():
    """A route suffix outside the built-in's exact endpoints never reaches mcpgw."""
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client) as factory:
        response = _post_tools_list(BUILTIN_SERVER, BUILTIN_TARGET, route_suffix="/sse")

    assert response.status_code == 502
    assert "blocked by egress policy" in response.json()["detail"]
    factory.assert_not_called()


def test_the_pooled_client_is_not_closed_after_a_request():
    client = _client_streaming(_upstream(OK_BODY))

    with patch(POOLED_CLIENT_FACTORY, return_value=client):
        _post_tools_list()

    client.aclose.assert_not_called()


def test_a_blocked_upstream_is_refused_with_502():
    client = MagicMock()
    client.stream = MagicMock(side_effect=UrlValidationError("http://x", "blocked"))

    with patch(POOLED_CLIENT_FACTORY, return_value=client):
        response = _post_tools_list()

    assert response.status_code == 502
    assert "blocked by egress policy" in response.json()["detail"]


@pytest.mark.parametrize(
    ("server_path", "signed", "outbound", "expected"),
    [
        pytest.param(
            BUILTIN_SERVER,
            BUILTIN_TARGET,
            "http://mcpgw-server:8003/mcp",
            BUILTIN_AIREGISTRY_TOOLS_PROFILE,
            id="builtin-direct-route",
        ),
        pytest.param(
            BUILTIN_SERVER,
            "http://mcpgw-server:8003/mcp",
            "http://mcpgw-server:8003/mcp",
            BUILTIN_AIREGISTRY_TOOLS_PROFILE,
            id="builtin-virtual-route-signs-the-endpoint",
        ),
        pytest.param(
            "airegistry-tools-copy",
            "http://mcpgw-server:8003/mcp",
            "http://mcpgw-server:8003/mcp",
            PROXY_PROFILE,
            id="lookalike-path-gets-the-ordinary-profile",
        ),
        pytest.param(
            BUILTIN_SERVER,
            "https://elsewhere.example.com/mcp",
            "https://elsewhere.example.com/mcp",
            PROXY_PROFILE,
            id="repointed-builtin-path-gets-the-ordinary-profile",
        ),
    ],
)
def test_hop_profile_matches_the_registration(server_path, signed, outbound, expected):
    assert proxy_profile_for_mcp_hop(server_path, signed, outbound) is expected


def test_the_ordinary_profile_refuses_the_reserved_builtin_host():
    """A lookalike that signs the built-in host gets PROXY_PROFILE, which refuses it."""
    from registry.utils.url_guard import validate_url

    profile = proxy_profile_for_mcp_hop(
        "airegistry-tools-copy",
        "http://mcpgw-server:8003/mcp",
        "http://mcpgw-server:8003/mcp",
    )

    with pytest.raises(UrlValidationError):
        validate_url("http://mcpgw-server:8003/mcp", profile=profile, resolve=False)


def test_virtual_hop_does_not_relay_routing_headers_to_the_backend():
    """The hop's route headers and a client-sent virtual URL never reach a backend."""
    client = _client_streaming(_upstream(OK_BODY))
    headers = {
        **_token_headers(SERVER_NAME, UPSTREAM_URL, virtual=True),
        "X-Virtual-Original-URL": "https://gateway.example.com/virtual/dev?x=1",
        "X-Resolved-Version": "v2",
    }

    with (
        patch(POOLED_CLIENT_FACTORY, return_value=client),
        _patch_scope_repo_allow_all(),
        patch.object(server, "_read_mcp_filter_enabled", return_value=False),
        patch.object(server, "_get_blocked_tools", new=AsyncMock(return_value=set())),
    ):
        response = TestClient(server.app).post(
            f"/mcp-proxy/{SERVER_NAME}/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers=headers,
        )

    sent = {k.lower() for k in client.stream.call_args.kwargs["headers"]}
    assert response.status_code == 200
    assert sent.isdisjoint(
        {
            "x-registered-server-path",
            "x-registered-route-mode",
            "x-virtual-original-url",
            "x-resolved-version",
            "x-internal-token",
        }
    )
