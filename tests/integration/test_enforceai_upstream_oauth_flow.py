"""
Phase 5 integration tests: upstream OAuth connect flow + proxy-time refresh/injection.

These tests run fully offline by using an in-process stub OAuth provider wired via httpx
ASGITransport (no real network).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import Optional
from urllib.parse import (
    parse_qs,
    urlparse,
)

import httpx
import pytest
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)
from fastapi.testclient import TestClient

import auth_server.server as auth_server_module
from auth_server.enforceai.auth import dependency as enforceai_dependency
from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
    load_gateway_keyring_cached,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
)
from auth_server.enforceai.secrets.upstream_kek import (
    load_upstream_kek,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)
from auth_server.enforceai.upstream.oauth_client import (
    OAuthTokenClient as RealOAuthTokenClient,
)
from gateway_csrf import (
    mint_csrf_token,
)
from gateway_session import (
    build_session_cookie_payload,
)


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    clear_scope_catalog_cache()
    load_gateway_keyring_cached.cache_clear()
    auth_server_module._load_enforceai_runtime.cache_clear()


def _write_scope_catalog(
    *,
    path: Path,
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            "group_mappings: {}",
            "scope-good:",
            "  - server: fininfo",
            "    methods: [tools/list, tools/call]",
            "    tools: [good_tool]",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _headers_for_validate(
    *,
    token: str,
    extra: dict[str, str],
) -> dict[str, str]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "tools/list",
        "params": {},
    }
    headers = {
        "X-Original-URL": "http://localhost/fininfo/",
        "X-Body": json.dumps(payload),
        "X-Gateway-Token": token,
        "X-EnforceAI-Server-Path": "/fininfo",
        "X-EnforceAI-Upstream-Credential-Binding": "user",
        "X-EnforceAI-Upstream-Header-Name": "Authorization",
        "X-EnforceAI-Upstream-Scheme": "Bearer",
    }
    headers.update(extra)
    return headers


def _make_cookie_client(
    *,
    session_id: str,
    user_id: str,
    groups: list[str],
    email: str,
) -> TestClient:
    cookie_payload = build_session_cookie_payload(
        username="cookie-user-1",
        email=email,
        name=None,
        groups=groups,
        provider="keycloak",
        legacy_auth_method="oauth2",
        max_age_seconds=28800,
        session_id=session_id,
        user_id=user_id,
    )
    cookie_value = auth_server_module.signer.dumps(cookie_payload)

    client = TestClient(auth_server_module.app)
    client.cookies.set("mcp_gateway_session", cookie_value)
    return client


def _build_stub_oauth_provider() -> FastAPI:
    app = FastAPI()
    issued_codes: dict[str, str] = {}

    expected_basic_auth: Optional[tuple[str, str]] = None

    def _require_basic_auth(
        request: Request,
    ) -> Optional[JSONResponse]:
        if expected_basic_auth is None:
            return None

        raw = request.headers.get("authorization") or ""
        if not raw.lower().startswith("basic "):
            return JSONResponse(status_code=401, content={"error": "invalid_client"})

        try:
            import base64

            decoded = base64.b64decode(raw.split(" ", 1)[1]).decode("utf-8")
        except Exception:
            return JSONResponse(status_code=401, content={"error": "invalid_client"})

        client_id, sep, client_secret = decoded.partition(":")
        if sep != ":":
            return JSONResponse(status_code=401, content={"error": "invalid_client"})

        expected_client_id, expected_client_secret = expected_basic_auth
        if client_id != expected_client_id or client_secret != expected_client_secret:
            return JSONResponse(status_code=401, content={"error": "invalid_client"})

        return None

    @app.get("/authorize")
    async def authorize(
        request: Request,
    ):
        state = request.query_params.get("state") or ""
        redirect_uri = request.query_params.get("redirect_uri") or ""
        code = f"code-{uuid.uuid4()}"
        issued_codes[code] = "rt-1"
        return RedirectResponse(url=f"{redirect_uri}?code={code}&state={state}", status_code=302)

    @app.post("/token")
    async def token(
        request: Request,
    ) -> JSONResponse:
        denied = _require_basic_auth(request)
        if denied is not None:
            return denied

        body = (await request.body()).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(body).items()}

        grant_type = form.get("grant_type")
        if grant_type == "authorization_code":
            code = form.get("code") or ""
            if code not in issued_codes:
                return JSONResponse(status_code=400, content={"error": "invalid_grant"})
            return JSONResponse(
                status_code=200,
                content={
                    "access_token": "access-1",
                    "refresh_token": issued_codes[code],
                    "token_type": "Bearer",
                    "expires_in": 1,
                    "scope": "scope-a scope-b",
                },
            )

        if grant_type == "refresh_token":
            refresh_token = form.get("refresh_token") or ""
            if refresh_token not in {"rt-1", "rt-2"}:
                return JSONResponse(status_code=400, content={"error": "invalid_grant"})
            return JSONResponse(
                status_code=200,
                content={
                    "access_token": "access-2",
                    "refresh_token": "rt-2",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "scope-a scope-b",
                },
            )

        return JSONResponse(status_code=400, content={"error": "unsupported_grant_type"})

    def _enable_basic_auth(
        *,
        client_id: str,
        client_secret: str,
    ) -> None:
        nonlocal expected_basic_auth
        expected_basic_auth = (client_id, client_secret)

    setattr(app.state, "enable_basic_auth", _enable_basic_auth)

    return app


def _build_stub_upstream_server() -> FastAPI:
    app = FastAPI()

    @app.post("/mcp")
    async def mcp(
        request: Request,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "authorization": request.headers.get("authorization") or "",
                "mcp_principal": request.headers.get("x-mcp-principal") or "",
                "mcp_auth_type": request.headers.get("x-mcp-auth-type") or "",
                "mcp_scopes": request.headers.get("x-mcp-scopes") or "",
                "mcp_provider": request.headers.get("x-mcp-provider") or "",
            },
        )

    return app


def _proxy_through_validate_and_call_upstream(
    *,
    auth_server_client: TestClient,
    upstream_client: TestClient,
    validate_headers: dict[str, str],
) -> dict[str, str]:
    validate_response = auth_server_client.get(
        "/validate",
        headers=validate_headers,
    )
    assert validate_response.status_code == 200

    upstream_headers: dict[str, str] = {
        "Authorization": validate_response.headers.get("X-EnforceAI-Upstream-Authorization") or "",
        "X-MCP-Principal": validate_response.headers.get("X-MCP-Principal") or "",
        "X-MCP-Auth-Type": validate_response.headers.get("X-MCP-Auth-Type") or "",
        "X-MCP-Scopes": validate_response.headers.get("X-MCP-Scopes") or "",
        "X-MCP-Provider": validate_response.headers.get("X-MCP-Provider") or "",
    }

    upstream_response = upstream_client.post(
        "/mcp",
        headers=upstream_headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert upstream_response.status_code == 200
    payload = upstream_response.json()
    assert isinstance(payload, dict)
    return {str(key): str(value) for key, value in payload.items()}


@pytest.mark.integration
class TestEnforceAIUpstreamOAuthFlow:
    def test_connect_store_inject_and_refresh(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        oauth_app = _build_stub_oauth_provider()

        monkeypatch.setattr(
            enforceai_dependency,
            "OAuthTokenClient",
            lambda: RealOAuthTokenClient(
                transport=httpx.ASGITransport(app=oauth_app),
            ),
        )

        upstream_app = _build_stub_upstream_server()
        upstream_client = TestClient(upstream_app)

        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        upstream_kek_path = tmp_path / "upstream_kek"
        upstream_kek_path.write_text("aa" * 32)
        upstream_kek = load_upstream_kek(upstream_kek_path)

        registry_servers_dir = tmp_path / "registry_servers"
        registry_servers_dir.mkdir(parents=True, exist_ok=True)
        (registry_servers_dir / "fininfo.json").write_text(
            json.dumps(
                {
                    "server_name": "fininfo",
                    "path": "/fininfo/",
                    "proxy_pass_url": "http://fininfo-server:8000/",
                    "upstream_auth": {
                        "mode": "gateway-managed",
                        "type": "oauth2",
                        "provider": "github",
                        "credential_binding": "user",
                        "injection": {
                            "kind": "header",
                            "header_name": "Authorization",
                            "scheme": "Bearer",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (registry_servers_dir / "noauth.json").write_text(
            json.dumps(
                {
                    "server_name": "noauth",
                    "path": "/noauth/",
                    "proxy_pass_url": "http://noauth-server:8000/",
                    "upstream_auth": {
                        "mode": "none",
                        "type": "none",
                        "credential_binding": "service",
                    },
                }
            ),
            encoding="utf-8",
        )

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=upstream_kek)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        session_id = str(uuid.uuid4())
        stores.session_store.create_session(
            session_id=session_id,
            user_id=user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_UPSTREAM_KEK_PATH": str(upstream_kek_path),
                "ENFORCEAI_REGISTRY_SERVERS_DIR": str(registry_servers_dir),
                "ENFORCEAI_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS": "0",
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
                "ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS": json.dumps(
                    {
                        "github": {
                            "authorization_endpoint": "http://localhost/authorize",
                            "token_endpoint": "http://localhost/token",
                            "client_id": "client-id",
                            "client_secret_ref": {
                                "kind": "env",
                                "env_var": "TEST_UPSTREAM_OAUTH_SECRET",
                            },
                            "default_scopes": ["scope-a", "scope-b"],
                        }
                    }
                ),
                "TEST_UPSTREAM_OAUTH_SECRET": "client-secret",
            }
        )
        _reset_enforcement_caches()

        client = _make_cookie_client(
            session_id=session_id,
            user_id=user_id,
            groups=[],
            email="cookie-user-1@example.com",
        )
        csrf_token = mint_csrf_token(
            secret_key=auth_server_module.SECRET_KEY,
            session_id=session_id,
        )

        non_oauth_start = client.post(
            "/enforceai/upstream/servers/noauth/oauth/start",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "credential_type": "oauth2",
                "credential_binding": "user",
                "provider": "github",
                "ui_return_url": "/credentials/upstream/oauth/callback",
            },
        )
        assert non_oauth_start.status_code == 400
        assert non_oauth_start.json()["detail"] == "Server does not require upstream OAuth"

        start = client.post(
            "/enforceai/upstream/servers/fininfo/oauth/start",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "credential_type": "oauth2",
                "credential_binding": "user",
                "provider": "github",
                "ui_return_url": "/credentials/upstream/oauth/callback",
            },
        )
        assert start.status_code == 200
        auth_url = start.json()["authorization_url"]

        parsed = urlparse(auth_url)
        provider_client = TestClient(oauth_app)
        auth_resp = provider_client.get(
            f"{parsed.path}?{parsed.query}",
            follow_redirects=False,
        )
        assert auth_resp.status_code in {302, 307}
        redirect_location = auth_resp.headers["location"]

        redirect = urlparse(redirect_location)
        callback_resp = client.get(
            f"{redirect.path}?{redirect.query}",
            follow_redirects=False,
        )
        assert callback_resp.status_code in {302, 307}
        callback_location = callback_resp.headers.get("location") or ""
        assert callback_location.startswith("/credentials/upstream/oauth/callback")
        assert "access-1" not in callback_location
        assert "rt-1" not in callback_location

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
            ttl_seconds=3600,
            jti="jti-upstream-oauth-1",
        )

        first_proxy = _proxy_through_validate_and_call_upstream(
            auth_server_client=client,
            upstream_client=upstream_client,
            validate_headers=_headers_for_validate(
                token=token,
                extra={
                    "X-EnforceAI-Upstream-Auth-Type": "oauth2",
                    "X-EnforceAI-Upstream-Provider": "github",
                },
            ),
        )
        assert first_proxy["authorization"] == "Bearer access-1"

        # Token expires quickly in the stub; the next request should refresh.
        time.sleep(2)

        second_proxy = _proxy_through_validate_and_call_upstream(
            auth_server_client=client,
            upstream_client=upstream_client,
            validate_headers=_headers_for_validate(
                token=token,
                extra={
                    "X-EnforceAI-Upstream-Auth-Type": "oauth2",
                    "X-EnforceAI-Upstream-Provider": "github",
                },
            ),
        )
        assert second_proxy["authorization"] == "Bearer access-2"

        before_disconnect = client.get("/enforceai/upstream/servers/fininfo/credentials")
        assert before_disconnect.status_code == 200
        assert any(
            record.get("credential_type") == "oauth2" and record.get("provider") == "github"
            for record in before_disconnect.json()
        )

    def test_connect_store_inject_and_refresh_prefers_db_provider_registry_over_env(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        oauth_app = _build_stub_oauth_provider()
        oauth_app.state.enable_basic_auth(client_id="client-id", client_secret="client-secret")

        monkeypatch.setattr(
            enforceai_dependency,
            "OAuthTokenClient",
            lambda: RealOAuthTokenClient(
                transport=httpx.ASGITransport(app=oauth_app),
            ),
        )

        upstream_app = _build_stub_upstream_server()
        upstream_client = TestClient(upstream_app)

        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        upstream_kek_path = tmp_path / "upstream_kek"
        upstream_kek_path.write_text("aa" * 32)
        upstream_kek = load_upstream_kek(upstream_kek_path)

        registry_servers_dir = tmp_path / "registry_servers"
        registry_servers_dir.mkdir(parents=True, exist_ok=True)
        (registry_servers_dir / "fininfo.json").write_text(
            json.dumps(
                {
                    "server_name": "fininfo",
                    "path": "/fininfo/",
                    "proxy_pass_url": "http://fininfo-server:8000/",
                    "upstream_auth": {
                        "mode": "gateway-managed",
                        "type": "oauth2",
                        "provider": "github",
                        "credential_binding": "user",
                        "injection": {
                            "kind": "header",
                            "header_name": "Authorization",
                            "scheme": "Bearer",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=upstream_kek)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-2"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        session_id = str(uuid.uuid4())
        stores.session_store.create_session(
            session_id=session_id,
            user_id=user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )

        admin_user_id = "https://issuer.example|admin-1"
        admin_session_id = str(uuid.uuid4())
        stores.session_store.create_session(
            session_id=admin_session_id,
            user_id=admin_user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_UPSTREAM_KEK_PATH": str(upstream_kek_path),
                "ENFORCEAI_REGISTRY_SERVERS_DIR": str(registry_servers_dir),
                "ENFORCEAI_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS": "0",
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
                # Provide an env-based provider with a wrong secret; runtime must prefer DB.
                "ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS": json.dumps(
                    {
                        "github": {
                            "authorization_endpoint": "http://localhost/authorize",
                            "token_endpoint": "http://localhost/token",
                            "client_id": "client-id",
                            "client_secret_ref": {
                                "kind": "env",
                                "env_var": "WRONG_UPSTREAM_OAUTH_SECRET",
                            },
                            "default_scopes": ["scope-a", "scope-b"],
                        }
                    }
                ),
                "WRONG_UPSTREAM_OAUTH_SECRET": "wrong-secret",
            }
        )
        _reset_enforcement_caches()

        admin_client = _make_cookie_client(
            session_id=admin_session_id,
            user_id=admin_user_id,
            groups=["enforceai-admin"],
            email="cookie-admin-1@example.com",
        )
        admin_csrf = mint_csrf_token(
            secret_key=auth_server_module.SECRET_KEY,
            session_id=admin_session_id,
        )

        created_provider = admin_client.post(
            "/enforceai/admin/upstream-oauth-providers",
            headers={"X-CSRF-Token": admin_csrf},
            json={
                "provider_id": "github",
                "authorization_endpoint": "http://localhost/authorize",
                "token_endpoint": "http://localhost/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "default_scopes": ["scope-a", "scope-b"],
            },
        )
        assert created_provider.status_code == 200

        client = _make_cookie_client(
            session_id=session_id,
            user_id=user_id,
            groups=[],
            email="cookie-user-2@example.com",
        )
        csrf_token = mint_csrf_token(
            secret_key=auth_server_module.SECRET_KEY,
            session_id=session_id,
        )

        start = client.post(
            "/enforceai/upstream/servers/fininfo/oauth/start",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "credential_type": "oauth2",
                "credential_binding": "user",
                "provider": "github",
                "ui_return_url": "/credentials/upstream/oauth/callback",
            },
        )
        assert start.status_code == 200
        auth_url = start.json()["authorization_url"]

        parsed = urlparse(auth_url)
        provider_client = TestClient(oauth_app)
        auth_resp = provider_client.get(
            f"{parsed.path}?{parsed.query}",
            follow_redirects=False,
        )
        assert auth_resp.status_code in {302, 307}
        redirect_location = auth_resp.headers["location"]

        redirect = urlparse(redirect_location)
        callback_resp = client.get(
            f"{redirect.path}?{redirect.query}",
            follow_redirects=False,
        )
        assert callback_resp.status_code in {302, 307}
        callback_location = callback_resp.headers.get("location") or ""
        assert callback_location.startswith("/credentials/upstream/oauth/callback")

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
            ttl_seconds=3600,
            jti="jti-upstream-oauth-2",
        )

        first_proxy = _proxy_through_validate_and_call_upstream(
            auth_server_client=client,
            upstream_client=upstream_client,
            validate_headers=_headers_for_validate(
                token=token,
                extra={
                    "X-EnforceAI-Upstream-Auth-Type": "oauth2",
                    "X-EnforceAI-Upstream-Provider": "github",
                },
            ),
        )
        assert first_proxy["authorization"] == "Bearer access-1"

        time.sleep(2)

        second_proxy = _proxy_through_validate_and_call_upstream(
            auth_server_client=client,
            upstream_client=upstream_client,
            validate_headers=_headers_for_validate(
                token=token,
                extra={
                    "X-EnforceAI-Upstream-Auth-Type": "oauth2",
                    "X-EnforceAI-Upstream-Provider": "github",
                },
            ),
        )
        assert second_proxy["authorization"] == "Bearer access-2"

        disconnect = client.post(
            "/enforceai/upstream/servers/fininfo/oauth/disconnect",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "credential_type": "oauth2",
                "credential_binding": "user",
                "provider": "github",
            },
        )
        assert disconnect.status_code == 200
        assert disconnect.json()["revoked_count"] >= 1

        after_disconnect = client.get("/enforceai/upstream/servers/fininfo/credentials")
        assert after_disconnect.status_code == 200
        assert not any(
            record.get("credential_type") == "oauth2" and record.get("provider") == "github"
            for record in after_disconnect.json()
        )
