"""
Unit tests for the EnforceAI FastAPI dependency wiring (Stage 5.3).
"""

from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest
from fastapi import (
    Depends,
    FastAPI,
    Request,
)
from fastapi.testclient import TestClient

from auth_server.enforceai.auth.dependency import (
    EnforceAIRequestContext,
    get_enforceai_request_context,
    get_identity_resolver,
    get_scope_catalog,
)
from auth_server.enforceai.auth.resolver import (
    IdentityResolver,
)
from auth_server.enforceai.fgac.models import (
    ScopeCatalog,
)
from auth_server.enforceai.models.agent import (
    AgentRecord,
)
from auth_server.enforceai.oidc.models import (
    OIDCValidatedToken,
)
from auth_server.enforceai.providers.oidc import (
    OidcProvider,
)


class _FakeOidcVerifier:
    async def verify_bearer_token(
        self,
        token: str,
    ) -> OIDCValidatedToken:
        return OIDCValidatedToken(
            issuer="https://issuer.example",
            subject="sub",
            user_id="https://issuer.example|sub",
            audiences=["mcp-registry"],
            scopes=["scope-1"],
            roles=[],
            claims={"iss": "https://issuer.example", "sub": "sub"},
        )


class _AgentStoreRaises:
    def get_agent_by_id(
        self,
        *,
        agent_id: str,
    ) -> AgentRecord:
        raise RuntimeError("boom")


class _AgentStoreHappy:
    def __init__(
        self,
        *,
        agent: AgentRecord,
    ) -> None:
        self._agent = agent

    def get_agent_by_id(
        self,
        *,
        agent_id: str,
    ) -> AgentRecord:
        if agent_id != self._agent.agent_id:
            return None
        return self._agent


def _catalog() -> ScopeCatalog:
    return ScopeCatalog(
        path=Path("in-memory"),
        ui_scopes={},
        group_mappings={},
        scopes={},
    )


def _app_with_overrides(
    *,
    resolver: IdentityResolver,
    catalog: ScopeCatalog,
) -> FastAPI:
    app = FastAPI()

    @app.get("/protected")
    async def _protected(
        request: Request,
        ctx: EnforceAIRequestContext = Depends(get_enforceai_request_context),
    ) -> dict:
        return {
            "user_id": ctx.identity.user_id,
            "state_has_identity": hasattr(request.state, "enforceai_identity"),
            "state_has_catalog": hasattr(request.state, "enforceai_scope_catalog"),
        }

    app.dependency_overrides[get_identity_resolver] = lambda: resolver
    app.dependency_overrides[get_scope_catalog] = lambda: catalog
    return app


@pytest.mark.unit
class TestEnforceAIDependencyWiring:
    def test_missing_credentials_returns_401(self) -> None:
        verifier = _FakeOidcVerifier()
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=_AgentStoreRaises(),
        )
        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )
        app = _app_with_overrides(
            resolver=resolver,
            catalog=_catalog(),
        )

        client = TestClient(app)
        response = client.get("/protected")

        assert response.status_code == 401

    def test_missing_x_agent_id_returns_403(self) -> None:
        verifier = _FakeOidcVerifier()
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=_AgentStoreRaises(),
        )
        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )
        app = _app_with_overrides(
            resolver=resolver,
            catalog=_catalog(),
        )

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer fake"},
        )

        assert response.status_code == 403

    def test_dependency_failure_returns_503(self) -> None:
        verifier = _FakeOidcVerifier()
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=_AgentStoreRaises(),
        )
        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )
        app = _app_with_overrides(
            resolver=resolver,
            catalog=_catalog(),
        )

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={
                "Authorization": "Bearer fake",
                "X-Agent-Id": str(uuid.uuid4()),
            },
        )

        assert response.status_code == 503

    def test_attaches_context_to_request_state_on_success(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        agent_id = str(uuid.uuid4())

        agent = AgentRecord(
            user_id="https://issuer.example|sub",
            agent_id=agent_id,
            scopes=["scope-1"],
            allowed_tools=None,
            alias=None,
            metadata=None,
            revoked_at=None,
            tokens_valid_after=None,
            created_at=now,
            updated_at=now,
        )

        verifier = _FakeOidcVerifier()
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=_AgentStoreHappy(agent=agent),
        )
        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )
        app = _app_with_overrides(
            resolver=resolver,
            catalog=_catalog(),
        )

        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={
                "Authorization": "Bearer fake",
                "X-Agent-Id": agent_id,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["state_has_identity"] is True
        assert payload["state_has_catalog"] is True

