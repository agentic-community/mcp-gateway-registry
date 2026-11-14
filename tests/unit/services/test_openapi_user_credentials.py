"""Tests for per-user credentials on OpenAPI MCP server targets (issue #2022)."""

from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest

from registry.auth.openapi_mcp_sync import (
    JtiCache,
    verify_credential_envelope,
)
from registry.schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiTarget,
    OpenApiTargetAuthInput,
    StoredTargetAuth,
)
from registry.services import openapi_server_service as service
from registry.services import (
    openapi_spec_checks,
    openapi_user_credentials,
)
from registry.services.openapi_errors import OpenApiServerError

SYNC_SECRET: str = "openapi-sync-0123456789abcdefghijklmnopqrstuv"
PRINCIPAL = SimpleNamespace(auth_method="oauth2", user_id="alice-sub")


def _target(
    name: str,
    mode: str,
    base_url: str = "https://api.example.com/v1/",
) -> OpenApiTarget:
    now = openapi_spec_checks.now_utc()
    return OpenApiTarget(
        name=name,
        spec_url="https://api.example.com/spec.json",
        spec_sha256="a" * 64,
        base_url=base_url,
        auth=StoredTargetAuth(scheme="bearer", credential_mode=mode),
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


def _server(
    targets: list[OpenApiTarget],
) -> OpenApiServer:
    now = openapi_spec_checks.now_utc()
    return OpenApiServer(
        slug="weather-apis",
        name="Weather APIs",
        path="/weather-apis",
        targets=targets,
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


class FakeVault:
    """Stands in for the egress auth service's PAT methods."""

    def __init__(self) -> None:
        self.tokens: dict[tuple, object] = {}

    async def set_pat(self, auth_method, user_id, provider, server_path, token):
        self.tokens[(auth_method, user_id, provider, server_path)] = token

    async def get_pat_status(self, auth_method, user_id, provider, server_path):
        return self.tokens.get((auth_method, user_id, provider, server_path))

    async def delete_pat(self, auth_method, user_id, provider, server_path):
        self.tokens.pop((auth_method, user_id, provider, server_path), None)

    async def get_pat(self, auth_method, user_id, provider, server_path, requested_upstream):
        token = self.tokens.get((auth_method, user_id, provider, server_path))
        if token is None or requested_upstream not in token.bound_upstreams:
            return None
        return token.access_token


@pytest.fixture
def vault(monkeypatch):
    fake = FakeVault()
    monkeypatch.setattr(openapi_user_credentials, "_egress_service", lambda: fake)
    monkeypatch.setattr(openapi_user_credentials.settings, "openapi_mcp_sync_secret", SYNC_SECRET)
    return fake


def _open(
    envelope: str,
) -> dict:
    return verify_credential_envelope(envelope, SYNC_SECRET, "weather-apis", JtiCache())


class TestStoreAndEnvelope:
    async def test_key_is_bound_to_the_canonical_base_url(self, vault):
        server = _server([_target("pets", "per_user")])

        token = await openapi_user_credentials.store_key(
            server, server.targets[0], PRINCIPAL, "k1", 3600
        )

        assert token.bound_upstreams == ["https://api.example.com/v1"]
        assert ("oauth2", "alice-sub", "openapi-pets", "/weather-apis") in vault.tokens

    async def test_envelope_carries_only_per_user_keys(self, vault):
        server = _server([_target("pets", "per_user"), _target("shared", "shared")])
        await openapi_user_credentials.store_key(server, server.targets[0], PRINCIPAL, "k1", 3600)

        envelope = await openapi_user_credentials.build_envelope(server, "oauth2", "alice-sub")

        assert _open(envelope) == {"pets": {"key": "k1", "base_url": "https://api.example.com/v1"}}

    async def test_other_users_key_is_not_used(self, vault):
        server = _server([_target("pets", "per_user")])
        await openapi_user_credentials.store_key(server, server.targets[0], PRINCIPAL, "k1", 3600)

        envelope = await openapi_user_credentials.build_envelope(server, "oauth2", "bob-sub")

        assert _open(envelope) == {}

    async def test_repointed_target_drops_the_key(self, vault):
        server = _server([_target("pets", "per_user")])
        await openapi_user_credentials.store_key(server, server.targets[0], PRINCIPAL, "k1", 3600)
        repointed = _server([_target("pets", "per_user", base_url="https://other.example.com")])

        envelope = await openapi_user_credentials.build_envelope(repointed, "oauth2", "alice-sub")

        assert _open(envelope) == {}

    async def test_shared_target_refuses_a_user_key(self, vault):
        server = _server([_target("shared", "shared")])

        with pytest.raises(OpenApiServerError) as info:
            await openapi_user_credentials.store_key(
                server, server.targets[0], PRINCIPAL, "k1", 3600
            )

        assert info.value.code == "not_per_user"


class TestServiceRouting:
    def test_per_user_target_routes_through_the_egress_hop(self):
        fields = service._egress_fields(_server([_target("pets", "per_user")]))

        assert fields == {"egress_auth_mode": "pat", "egress_oauth": {"provider": "openapi-mcp"}}

    def test_shared_only_server_has_no_egress(self):
        assert (
            service._egress_fields(_server([_target("pets", "shared")]))["egress_auth_mode"]
            == "none"
        )

    def test_per_user_needs_egress_auth_enabled(self, monkeypatch):
        monkeypatch.setattr(service.settings, "egress_auth_enabled", False)
        auth = OpenApiTargetAuthInput(scheme="bearer", credential_mode="per_user")

        with pytest.raises(OpenApiServerError) as info:
            service._require_egress_for(auth)

        assert info.value.code == "egress_auth_disabled"

    async def test_sync_updates_serverinfo_and_flushes_nginx(self, monkeypatch):
        update = AsyncMock(return_value=True)
        flush = AsyncMock()
        monkeypatch.setattr(
            service.server_service,
            "get_server_info",
            AsyncMock(return_value={"egress_auth_mode": "none"}),
        )
        monkeypatch.setattr(service.server_service, "update_server", update)
        monkeypatch.setattr(service.server_lifecycle, "flush_nginx", flush)
        monkeypatch.setattr(
            service, "openapi_mcp_target_url", lambda slug: f"http://openapi-mcp:9876/s/{slug}/mcp"
        )

        await service._sync_server_info(_server([_target("pets", "per_user")]))

        assert update.call_args.args[1]["egress_auth_mode"] == "pat"
        flush.assert_awaited_once()


class TestVendHook:
    async def test_vend_returns_the_envelope_header(self, vault, monkeypatch):
        from registry.api import egress_auth_routes

        server = _server([_target("pets", "per_user")])
        await openapi_user_credentials.store_key(server, server.targets[0], PRINCIPAL, "k1", 3600)
        repo = SimpleNamespace(get=AsyncMock(return_value=server))
        with patch("registry.repositories.factory.get_openapi_server_repository", lambda: repo):
            response = await egress_auth_routes._openapi_envelope_response(
                {"path": "/weather-apis", "metadata": {"openapi_server": {"slug": "weather-apis"}}},
                "oauth2",
                "alice-sub",
            )

        assert response.pat_header_name == "X-OpenAPI-MCP-User-Credentials"
        assert response.pat_value_prefix == ""
        assert _open(response.access_token)["pets"]["key"] == "k1"

    async def test_vend_refuses_a_path_mismatch(self, vault):
        from registry.api import egress_auth_routes

        repo = SimpleNamespace(get=AsyncMock(return_value=_server([])))
        with patch("registry.repositories.factory.get_openapi_server_repository", lambda: repo):
            response = await egress_auth_routes._openapi_envelope_response(
                {"path": "/someone-else", "metadata": {"openapi_server": {"slug": "weather-apis"}}},
                "oauth2",
                "alice-sub",
            )

        assert response.access_token is None
