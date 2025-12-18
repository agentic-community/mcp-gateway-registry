"""
Unit tests for upstream OAuth refresh behavior in request-time resolver (Phase 5).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from urllib.parse import parse_qs

import httpx
import pytest

from auth_server.enforceai.config import (
    UpstreamOAuthProviderConfig,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.identity import (
    IdentityContext,
)
from auth_server.enforceai.models.upstream_auth import (
    UpstreamAuthConfig,
    UpstreamAuthInjection,
)
from auth_server.enforceai.upstream.oauth_client import (
    OAuthTokenClient,
)
from auth_server.enforceai.upstream.resolver import (
    UpstreamInjectionError,
    resolve_upstream_injection,
)


@pytest.mark.unit
class TestUpstreamOAuthRefresh:
    def test_refreshes_expired_oauth2_credential(
        self,
        enforceai_sqlite_db_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_UPSTREAM_OAUTH_SECRET", "client-secret")

        provider = UpstreamOAuthProviderConfig(
            authorization_endpoint="http://localhost/authorize",
            token_endpoint="http://localhost/token",
            client_id="client-id",
            client_secret_ref={"kind": "env", "env_var": "TEST_UPSTREAM_OAUTH_SECRET"},
            default_scopes=[],
        )

        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            form = {k: v[0] for k, v in parse_qs(request.content.decode("utf-8")).items()}
            assert form["grant_type"] == "refresh_token"
            assert form["refresh_token"] == "rt-1"
            return httpx.Response(
                200,
                json={
                    "access_token": "access-new",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rt-2",
                    "scope": "scope-a scope-b",
                },
            )

        token_client = OAuthTokenClient(transport=httpx.MockTransport(handler))

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=b"\x99" * 32)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        expired_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=5)
        created = stores.upstream_credential_store.create_credential(
            server_path="/fininfo",
            credential_type="oauth2",
            credential_binding="user",
            user_id=user_id,
            provider="github",
            expires_at=expired_at,
            secret_payload={"access_token": "access-old", "refresh_token": "rt-1"},
        )

        identity = IdentityContext(
            user_id=user_id,
            agent_id=agent_id,
            provider="gateway-token",
            scopes=["scope-good"],
        )

        upstream_auth = UpstreamAuthConfig(
            type="oauth2",
            provider="github",
            credential_binding="user",
            injection=UpstreamAuthInjection(header_name="Authorization", scheme="Bearer"),
        )

        result = asyncio.run(
            resolve_upstream_injection(
                server_path="/fininfo",
                upstream_auth=upstream_auth,
                identity=identity,
                stores=stores,
                oauth_providers={"github": provider},
                oauth_token_client=token_client,
                oauth_refresh_skew_seconds=60,
            )
        )
        assert result.upstream_authorization == "Bearer access-new"

        updated = stores.upstream_credential_store.get_credential_by_id(
            credential_id=created.credential_id
        )
        assert updated is not None
        assert updated.expires_at is not None
        assert updated.expires_at > datetime.now(timezone.utc).replace(microsecond=0)

        secret = stores.upstream_credential_store.get_credential_secret(
            credential_id=created.credential_id
        )
        assert secret is not None
        assert secret.payload["access_token"] == "access-new"
        assert secret.payload["refresh_token"] == "rt-2"

    def test_refresh_failure_maps_to_424(
        self,
        enforceai_sqlite_db_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_UPSTREAM_OAUTH_SECRET", "client-secret")

        provider = UpstreamOAuthProviderConfig(
            authorization_endpoint="http://localhost/authorize",
            token_endpoint="http://localhost/token",
            client_id="client-id",
            client_secret_ref={"kind": "env", "env_var": "TEST_UPSTREAM_OAUTH_SECRET"},
            default_scopes=[],
        )

        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            _ = request
            return httpx.Response(400, json={"error": "invalid_grant"})

        token_client = OAuthTokenClient(transport=httpx.MockTransport(handler))

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=b"\x88" * 32)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        expired_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=5)
        stores.upstream_credential_store.create_credential(
            server_path="/fininfo",
            credential_type="oauth2",
            credential_binding="user",
            user_id=user_id,
            provider="github",
            expires_at=expired_at,
            secret_payload={"access_token": "access-old", "refresh_token": "rt-1"},
        )

        identity = IdentityContext(
            user_id=user_id,
            agent_id=agent_id,
            provider="gateway-token",
            scopes=["scope-good"],
        )

        upstream_auth = UpstreamAuthConfig(
            type="oauth2",
            provider="github",
            credential_binding="user",
            injection=UpstreamAuthInjection(header_name="Authorization", scheme="Bearer"),
        )

        with pytest.raises(UpstreamInjectionError) as exc_info:
            asyncio.run(
                resolve_upstream_injection(
                    server_path="/fininfo",
                    upstream_auth=upstream_auth,
                    identity=identity,
                    stores=stores,
                    oauth_providers={"github": provider},
                    oauth_token_client=token_client,
                    oauth_refresh_skew_seconds=60,
                )
            )

        assert exc_info.value.status_code == 424
        assert exc_info.value.error_code == "UPSTREAM_OAUTH_REFRESH_FAILED"

