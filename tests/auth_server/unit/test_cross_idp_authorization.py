"""Exercise gateway and registry authorization after provider token validation."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_PROVIDERS = ("keycloak", "cognito", "entra", "okta", "auth0", "pingfederate")
_READ_SCOPE = "reports/read"
_ADMIN_SCOPE = "mcp-registry-admin"
_ADMIN_PATH = "/api/registry-management/telemetry/info"
_MARKER = "cross-idp-nginx-marker"


def _verified_result(provider, *, groups, scopes):
    # PingFederate returns raw JWT claims in data, without a provider field.
    return {
        "valid": True,
        "method": provider,
        "username": "alice",
        "client_id": "gateway-client",
        "groups": list(groups),
        "scopes": list(scopes),
        "data": {"sub": "alice", "scope": " ".join(scopes)},
    }


@pytest.fixture
def authorization(auth_env_vars, mock_scope_repository_with_data, mock_scopes_config, monkeypatch):
    import group_filter
    import mongodb_groups_enrichment
    import rate_limiting_config

    from auth_server import server
    from registry.api.registry_management_routes import router

    mock_scopes_config.clear()
    mock_scopes_config.update(
        {
            "group_mappings": {
                "report-readers": [_READ_SCOPE],
                "registry-admins": [_ADMIN_SCOPE],
            },
            _READ_SCOPE: [
                {
                    "server": "reports",
                    "methods": ["initialize", "tools/list", "tools/call"],
                    "tools": ["read_report"],
                }
            ],
            _ADMIN_SCOPE: [{"server": "*", "methods": ["*"], "tools": ["*"]}],
        }
    )
    repo = mock_scope_repository_with_data
    repo.get_server_scopes_bulk.side_effect = lambda scopes: {
        scope: mock_scopes_config.get(scope, []) for scope in scopes
    }
    repo.get_all_group_mappings.return_value = {
        _READ_SCOPE: ["report-readers"],
        _ADMIN_SCOPE: ["registry-admins"],
    }
    repo.get_all_mapped_group_names.return_value = {"report-readers", "registry-admins"}
    ui_scopes = {
        _READ_SCOPE: {"list_service": ["reports"]},
        _ADMIN_SCOPE: {"register_service": ["all"]},
    }
    repo.get_ui_scopes_bulk.side_effect = lambda scopes: {
        scope: ui_scopes.get(scope, {}) for scope in scopes
    }

    provider = MagicMock(spec=["validate_token", "recover_bearer_groups", "is_delegated_user"])
    provider.recover_bearer_groups = AsyncMock(return_value=None)
    provider.is_delegated_user.return_value = False
    machine_records = SimpleNamespace(find_one=AsyncMock(return_value=None))
    user_records = SimpleNamespace(find_one=AsyncMock(return_value=None))
    monkeypatch.setattr(
        mongodb_groups_enrichment,
        "_get_mongodb",
        AsyncMock(
            return_value={"idp_m2m_clients": machine_records, "idp_user_groups": user_records}
        ),
    )
    monkeypatch.setattr(server, "get_auth_provider", lambda: provider)
    monkeypatch.setattr(server, "get_scope_repository", lambda: repo)
    monkeypatch.setattr("registry.repositories.factory.get_scope_repository", lambda: repo)
    monkeypatch.setattr(server, "_get_blocked_tools", AsyncMock(return_value=set()))
    monkeypatch.setattr(server, "get_mcp_logger", lambda: None)
    monkeypatch.setattr(server, "SCOPES_CONFIG", {})
    monkeypatch.setattr(server, "_STATIC_TOKEN_MAP", {})
    monkeypatch.setattr(server, "REGISTRY_ROOT_PATH", "")
    monkeypatch.setattr(server, "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", ["pingfederate"])
    monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", _MARKER)
    monkeypatch.setenv("REGISTRY_URL", "https://gateway.example")
    monkeypatch.setattr(rate_limiting_config, "RATE_LIMITING_ENABLED", False)
    monkeypatch.setattr(group_filter, "ALLOWED_IDP_GROUPS", [])
    monkeypatch.setattr(group_filter, "_mapped_group_names_cache", None)
    monkeypatch.setattr(group_filter, "_mapped_group_names_inflight", None)
    heartbeat = AsyncMock(return_value={"server_count": 2})
    monkeypatch.setattr("registry.core.telemetry._build_heartbeat_payload", heartbeat)

    registry_app = FastAPI()
    registry_app.include_router(router, prefix="/api")
    # Do not start either application's external-service lifespan.
    yield SimpleNamespace(
        gateway=TestClient(server.app),
        registry=TestClient(registry_app),
        provider=provider,
        machine_records=machine_records,
        user_records=user_records,
        heartbeat=heartbeat,
    )


def _call_tool(authorization, *, server="reports", tool="read_report", headers=None):
    return authorization.gateway.get(
        "/validate",
        headers={
            "Authorization": "Bearer provider-verified-token",
            "X-Original-URL": f"https://gateway.example/{server}/mcp",
            "X-Body": json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool}}
            ),
            **(headers or {}),
        },
    )


def _call_admin(authorization):
    validated = authorization.gateway.get(
        "/validate",
        headers={
            "Authorization": "Bearer provider-verified-token",
            "X-Original-URL": f"https://gateway.example{_ADMIN_PATH}",
            "X-Registry-Api-Auth": "1",
            "X-Validate-Source-Secret": _MARKER,
        },
    )
    assert validated.status_code == 200, validated.text
    return authorization.registry.get(
        _ADMIN_PATH,
        headers={
            "X-Internal-Token-Registry": validated.headers["X-Internal-Token-Registry"],
            # The registry must derive permissions from signed groups, even if
            # a request advertises admin scopes in an unsigned header.
            "X-Scopes": _ADMIN_SCOPE,
        },
    )


def _assert_denied(response):
    assert response.status_code == 403, response.text
    assert "X-Scopes" not in response.headers
    assert "X-Internal-Token-Registry" not in response.headers


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("advertised_scopes", [["openid", "profile"], ["openid", _ADMIN_SCOPE]])
def test_groups_restrict_server_tool_and_admin_access(
    authorization, provider_name, advertised_scopes
):
    authorization.provider.validate_token.return_value = _verified_result(
        provider_name, groups=["report-readers"], scopes=advertised_scopes
    )

    allowed = _call_tool(authorization)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["scopes"] == [_READ_SCOPE]
    assert allowed.headers["X-Scopes"] == _READ_SCOPE
    _assert_denied(_call_tool(authorization, tool="delete_report"))
    _assert_denied(_call_tool(authorization, server="payroll"))
    _assert_denied(_call_admin(authorization))
    authorization.heartbeat.assert_not_awaited()


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_unmapped_groups_cannot_fall_back_to_token_admin_scope(authorization, provider_name):
    authorization.provider.validate_token.return_value = _verified_result(
        provider_name, groups=["unmapped-team"], scopes=[_ADMIN_SCOPE]
    )

    _assert_denied(_call_tool(authorization))
    _assert_denied(_call_admin(authorization))
    authorization.heartbeat.assert_not_awaited()


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_mapped_admin_groups_authorize_gateway_and_registry(authorization, provider_name):
    authorization.provider.validate_token.return_value = _verified_result(
        provider_name, groups=["registry-admins"], scopes=["openid"]
    )

    gateway = _call_tool(authorization, server="payroll", tool="delete_report")
    assert gateway.status_code == 200, gateway.text
    assert gateway.json()["scopes"] == [_ADMIN_SCOPE]
    registry = _call_admin(authorization)
    assert registry.status_code == 200, registry.text
    assert registry.json() == {"server_count": 2}
    authorization.heartbeat.assert_awaited_once()


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_scope_only_tokens_keep_their_server_tool_grant(authorization, monkeypatch, provider_name):
    result = _verified_result(provider_name, groups=[], scopes=[_READ_SCOPE])
    if provider_name == "entra":
        # Entra machine tokens use roles. Scope-only tokens identify users;
        # operators can retain direct scope grants with Graph recovery off.
        monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "false")
        result["data"].update({"tid": "tenant", "oid": "alice-object", "scp": _READ_SCOPE})
        authorization.provider.is_delegated_user.return_value = True
        authorization.provider.recover_bearer_groups.return_value = []
    else:
        result["data"].update({"grant_type": "client_credentials"})
    authorization.provider.validate_token.return_value = result

    allowed = _call_tool(authorization)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["groups"] == []
    assert allowed.json()["scopes"] == [_READ_SCOPE]
    _assert_denied(_call_tool(authorization, tool="delete_report"))
    _assert_denied(_call_tool(authorization, server="payroll"))
    _assert_denied(_call_admin(authorization))


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("enabled", [True, False], ids=["active", "disabled"])
def test_machine_group_records_control_gateway_and_admin_grants(
    authorization, provider_name, enabled
):
    result = _verified_result(provider_name, groups=[], scopes=[])
    result["data"] = {"sub": "machine-client", "grant_type": "client_credentials"}
    authorization.provider.validate_token.return_value = result
    # Return the document even for the disabled case to exercise the real
    # enrichment enabled check, rather than implementing it in the mock.
    authorization.machine_records.find_one.return_value = {
        "client_id": "gateway-client",
        "groups": ["registry-admins"],
        "enabled": enabled,
    }

    gateway = _call_tool(authorization, server="payroll", tool="delete_report")
    registry = _call_admin(authorization)
    if provider_name == "keycloak":
        _assert_denied(gateway)
        _assert_denied(registry)
        authorization.machine_records.find_one.assert_not_awaited()
    elif enabled:
        assert gateway.status_code == 200, gateway.text
        assert gateway.json()["groups"] == ["registry-admins"]
        assert gateway.json()["scopes"] == [_ADMIN_SCOPE]
        assert registry.status_code == 200, registry.text
    else:
        _assert_denied(gateway)
        _assert_denied(registry)
        authorization.heartbeat.assert_not_awaited()


@pytest.mark.parametrize("transport", ["bearer", "session"])
@pytest.mark.parametrize("enabled", [True, False], ids=["active", "disabled"])
def test_pingfederate_user_database_groups_authorize_tools(
    authorization, monkeypatch, transport, enabled
):
    from auth_server import server

    result = _verified_result("pingfederate", groups=[], scopes=["openid", "profile"])
    headers = {}
    if transport == "session":
        result.update(method="session_cookie", scopes=[])
        result["data"] = {"provider": "pingfederate", "username": "alice"}
        monkeypatch.setattr(
            server, "validate_session_cookie", AsyncMock(side_effect=lambda _: deepcopy(result))
        )
        headers["Cookie"] = f"{server.settings.session_cookie_name}=verified-session"
        headers["Origin"] = "https://gateway.example"
        headers["X-Original-Method"] = "POST"
    else:
        authorization.provider.validate_token.side_effect = lambda *args, **kwargs: deepcopy(result)
    authorization.user_records.find_one.return_value = {
        "username": "alice",
        "groups": ["report-readers"],
        "enabled": enabled,
    }

    allowed = _call_tool(authorization, headers=headers)
    if enabled:
        assert allowed.status_code == 200, allowed.text
        assert allowed.json()["groups"] == ["report-readers"]
        assert allowed.json()["scopes"] == [_READ_SCOPE]
    else:
        _assert_denied(allowed)
    _assert_denied(_call_tool(authorization, tool="delete_report", headers=headers))
    _assert_denied(_call_tool(authorization, server="payroll", headers=headers))
    authorization.user_records.find_one.assert_awaited_with(
        {"username": "alice", "enabled": {"$ne": False}}
    )


@pytest.mark.parametrize("recovered_groups", [[], ["unmapped-team"]], ids=["empty", "unmapped"])
def test_entra_authoritative_empty_groups_cannot_reuse_delegated_scopes(
    authorization, monkeypatch, recovered_groups
):
    from auth_server import server

    monkeypatch.setenv("ENTRA_GROUP_RECOVERY_ENABLED", "true")
    monkeypatch.setattr(server, "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", ["entra"])
    result = _verified_result("entra", groups=[], scopes=[_ADMIN_SCOPE])
    result["data"].update({"tid": "tenant", "oid": "alice-object", "scp": _ADMIN_SCOPE})
    authorization.provider.validate_token.return_value = result
    authorization.provider.is_delegated_user.return_value = True
    authorization.provider.recover_bearer_groups.return_value = recovered_groups
    authorization.machine_records.find_one.return_value = {
        "groups": ["registry-admins"],
        "enabled": True,
    }
    authorization.user_records.find_one.return_value = {
        "groups": ["registry-admins"],
        "enabled": True,
    }

    _assert_denied(_call_tool(authorization))
    _assert_denied(_call_admin(authorization))
    authorization.machine_records.find_one.assert_not_awaited()
    authorization.user_records.find_one.assert_not_awaited()


@pytest.mark.parametrize("group_path", ["/team/registry-admins", "//registry-admins"])
def test_keycloak_admin_named_subgroups_never_authorize_admin(authorization, group_path):
    from auth_server.providers.keycloak import _extract_keycloak_groups

    groups = _extract_keycloak_groups(
        {"mcpgw_group_paths": [group_path, "/report-readers"]}, source="test verified claims"
    )
    authorization.provider.validate_token.return_value = _verified_result(
        "keycloak", groups=groups, scopes=[_ADMIN_SCOPE]
    )

    allowed = _call_tool(authorization)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["scopes"] == [_READ_SCOPE]
    _assert_denied(_call_tool(authorization, server="payroll", tool="delete_report"))
    _assert_denied(_call_admin(authorization))
    authorization.heartbeat.assert_not_awaited()
