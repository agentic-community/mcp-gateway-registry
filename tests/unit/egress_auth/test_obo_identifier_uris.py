"""Tests for GET /api/egress/obo-identifier-uris.

The endpoint lists the per-server resource URLs the gateway advertises in a
per-server PRM. With PER_SERVER_PRM_ALL_PROVIDERS on, every server on every
provider has one, but only an ENABLED server answers its PRM route (a disabled
one has no nginx location), so the endpoint must list the same set the route
serves. With the setting off, the historical gating is unchanged.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes
import registry.api.wellknown_routes as wellknown_routes

REGISTRY_URL = "https://gw.example.com"


def _servers():
    """Servers covering every egress mode, enabled and disabled."""
    return {
        "/plain": {"egress_auth_mode": "none", "is_enabled": True},
        "/plain-off": {"egress_auth_mode": "none", "is_enabled": False},
        "/three-legged": {"egress_auth_mode": "oauth_user", "is_enabled": True},
        "/obo": {"egress_auth_mode": "obo_exchange", "is_enabled": True},
        "/obo-off": {"egress_auth_mode": "obo_exchange", "is_enabled": False},
        "/bare": {
            "egress_auth_mode": "none",
            "is_enabled": True,
            "append_mcp_path": False,
        },
    }


class _StubServerService:
    def __init__(self, servers):
        self._servers = servers

    async def get_all_servers(self, include_inactive=False):
        return self._servers

    async def get_server_info(self, path):
        return self._servers.get(path)


class _StubVirtualServerService:
    def __init__(self, virtual_servers):
        self._virtual_servers = virtual_servers

    async def list_virtual_servers(self):
        return self._virtual_servers


def _virtual(path, backend_paths, is_enabled=True):
    return SimpleNamespace(path=path, backend_paths=backend_paths, is_enabled=is_enabled)


@pytest.fixture
def get_uris(monkeypatch):
    """Factory: call the endpoint with a chosen toggle, provider and data."""

    def _call(
        *,
        per_server_all,
        auth_provider="keycloak",
        servers=None,
        virtual_servers=None,
        is_admin=True,
    ):
        monkeypatch.setattr(routes.settings, "per_server_prm_all_providers", per_server_all)
        monkeypatch.setattr(routes.settings, "auth_provider", auth_provider)
        monkeypatch.setattr(routes.settings, "registry_url", REGISTRY_URL)
        stub = _StubServerService(servers if servers is not None else {})
        monkeypatch.setattr(routes, "server_service", stub)
        # The virtual-server gate resolves its backing servers through the
        # wellknown module's own server_service reference.
        monkeypatch.setattr(wellknown_routes, "server_service", stub)
        monkeypatch.setattr(
            "registry.services.virtual_server_service.get_virtual_server_service",
            lambda: _StubVirtualServerService(virtual_servers or []),
        )
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: {"is_admin": is_admin}
        return TestClient(app).get("/egress/obo-identifier-uris")

    return _call


@pytest.mark.unit
class TestOboIdentifierUrisToggleOff:
    def test_lists_only_obo_exchange_on_a_lenient_provider(self, get_uris):
        """With the setting off (the default) the historical gating applies:
        obo_exchange servers only, disabled ones included as before."""
        resp = get_uris(per_server_all=False, servers=_servers())
        assert resp.status_code == 200
        assert resp.json() == {
            "identifier_uris": [f"{REGISTRY_URL}/obo-off/mcp", f"{REGISTRY_URL}/obo/mcp"],
            "count": 2,
        }

    def test_entra_lists_every_server_including_disabled(self, get_uris):
        """Unchanged Entra behavior: every registered server, enabled or not."""
        resp = get_uris(per_server_all=False, auth_provider="entra", servers=_servers())
        assert resp.json()["count"] == len(_servers())


@pytest.mark.unit
class TestOboIdentifierUrisToggleOn:
    def test_lists_every_enabled_server_on_a_lenient_provider(self, get_uris):
        resp = get_uris(per_server_all=True, servers=_servers())
        assert resp.json()["identifier_uris"] == [
            f"{REGISTRY_URL}/bare",
            f"{REGISTRY_URL}/obo/mcp",
            f"{REGISTRY_URL}/plain/mcp",
            f"{REGISTRY_URL}/three-legged/mcp",
        ]

    def test_omits_disabled_servers_because_their_prm_route_404s(self, get_uris):
        """A disabled server has no nginx location, so its per-server PRM route
        404s and no App ID URI is needed for it, whatever its egress mode."""
        uris = get_uris(per_server_all=True, servers=_servers()).json()["identifier_uris"]
        assert f"{REGISTRY_URL}/plain-off/mcp" not in uris
        assert f"{REGISTRY_URL}/obo-off/mcp" not in uris

    def test_omits_a_server_document_without_is_enabled(self, get_uris):
        """Fail closed: a document with no is_enabled flag counts as disabled,
        the same default the PRM route applies."""
        servers = {"/legacy": {"egress_auth_mode": "none"}}
        assert get_uris(per_server_all=True, servers=servers).json()["count"] == 0

    def test_honours_append_mcp_path_false(self, get_uris):
        uris = get_uris(per_server_all=True, servers=_servers()).json()["identifier_uris"]
        assert f"{REGISTRY_URL}/bare" in uris
        assert f"{REGISTRY_URL}/bare/mcp" not in uris

    def test_entra_omits_disabled_servers_too(self, get_uris):
        """With the setting on, the toggle gate decides first, so the enabled
        requirement covers Entra servers as well; the PRM route does the same."""
        resp = get_uris(per_server_all=True, auth_provider="entra", servers=_servers())
        assert f"{REGISTRY_URL}/plain-off/mcp" not in resp.json()["identifier_uris"]

    def test_lists_enabled_virtual_servers_through_their_backing_servers(self, get_uris):
        resp = get_uris(
            per_server_all=True,
            servers={"/plain": {"egress_auth_mode": "none", "is_enabled": True}},
            virtual_servers=[
                _virtual("/virtual/live", ["/plain"]),
                _virtual("/virtual/off", ["/plain"], is_enabled=False),
            ],
        )
        uris = resp.json()["identifier_uris"]
        assert f"{REGISTRY_URL}/virtual/live/mcp" in uris
        assert f"{REGISTRY_URL}/virtual/off/mcp" not in uris


@pytest.mark.unit
class TestOboIdentifierUrisAccess:
    def test_non_admin_is_refused(self, get_uris):
        resp = get_uris(per_server_all=True, servers=_servers(), is_admin=False)
        assert resp.status_code == 403
