"""
Unit tests for Stage 5.5 tool execution enforcement (tools/call) + audit hook.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import auth_server.server as auth_server_module
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
)
from auth_server.enforceai.fgac.models import (
    MethodPolicy,
    ScopeCatalog,
    ScopeDefinition,
    ServerPermission,
    ToolPolicy,
)
from auth_server.enforceai.identity import (
    IdentityContext,
)


class _FakeResolver:
    def __init__(
        self,
        *,
        identity: IdentityContext,
    ) -> None:
        self._identity = identity

    async def resolve_identity(
        self,
        *,
        headers: dict[str, str],
    ) -> IdentityContext:
        return self._identity


class _RecordingAuditStore:
    def __init__(
        self,
        *,
        raise_on_append: bool = False,
    ) -> None:
        self.events: list[dict[str, object]] = []
        self._raise_on_append = raise_on_append

    def append_event(
        self,
        *,
        occurred_at,
        user_id: str,
        agent_id: str,
        action: str,
        outcome: str,
        request_id: str | None = None,
        details: dict[str, object] | None = None,
    ):
        if self._raise_on_append:
            raise RuntimeError("audit down")

        self.events.append(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "action": action,
                "outcome": outcome,
                "request_id": request_id,
                "details": details or {},
            }
        )


@dataclass(frozen=True)
class _FakeStores:
    audit_store: _RecordingAuditStore


def _catalog_allows(
    *,
    tool_name: str,
) -> ScopeCatalog:
    scope = ScopeDefinition(
        name="scope-1",
        server_permissions=(
            ServerPermission(
                server="mcpgw",
                methods=MethodPolicy(
                    all_methods=False,
                    methods=frozenset({"tools/list", "tools/call"}),
                ),
                tools=ToolPolicy(
                    all_tools=False,
                    tools=frozenset({tool_name}),
                ),
            ),
        ),
        agent_permissions=tuple(),
    )
    return ScopeCatalog(
        path=Path("in-memory"),
        ui_scopes={},
        group_mappings={},
        scopes={"scope-1": scope},
    )


def _headers_for(
    *,
    method: str,
    server_name: str = "mcpgw",
    tool_name: str | None = None,
) -> dict[str, str]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": method,
        "params": {},
    }
    if tool_name is not None:
        payload["params"] = {"name": tool_name, "arguments": {}}

    return {
        "X-Original-URL": f"http://localhost/{server_name}/",
        "X-Body": json.dumps(payload),
    }


@pytest.mark.unit
class TestToolsCallEnforcement:
    def test_allows_authorized_tool_call_and_audits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENFORCEAI_DB_PATH", "/tmp/enforceai.db")

        identity = IdentityContext(
            user_id="https://issuer.example|sub",
            agent_id=str(uuid.uuid4()),
            provider="oidc",
            scopes=["scope-1"],
            metadata=None,
        )
        resolver = _FakeResolver(identity=identity)
        audit_store = _RecordingAuditStore()

        monkeypatch.setattr(
            auth_server_module,
            "get_identity_resolver",
            lambda: resolver,
        )
        monkeypatch.setattr(
            auth_server_module,
            "load_scope_catalog",
            lambda: _catalog_allows(tool_name="good_tool"),
        )
        monkeypatch.setattr(
            auth_server_module,
            "get_enforceai_stores",
            lambda: _FakeStores(audit_store=audit_store),
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(method="tools/call", tool_name="good_tool"),
        )

        assert response.status_code == 200
        assert any(
            event["action"] == "tools/call" and event["outcome"] == "allow"
            for event in audit_store.events
        )

    def test_denies_unauthorized_tool_call_and_audits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENFORCEAI_DB_PATH", "/tmp/enforceai.db")

        identity = IdentityContext(
            user_id="https://issuer.example|sub",
            agent_id=str(uuid.uuid4()),
            provider="oidc",
            scopes=["scope-1"],
            metadata=None,
        )
        resolver = _FakeResolver(identity=identity)
        audit_store = _RecordingAuditStore()

        monkeypatch.setattr(
            auth_server_module,
            "get_identity_resolver",
            lambda: resolver,
        )
        monkeypatch.setattr(
            auth_server_module,
            "load_scope_catalog",
            lambda: _catalog_allows(tool_name="good_tool"),
        )
        monkeypatch.setattr(
            auth_server_module,
            "get_enforceai_stores",
            lambda: _FakeStores(audit_store=audit_store),
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(method="tools/call", tool_name="bad_tool"),
        )

        assert response.status_code == 403
        assert any(
            event["action"] == "tools/call" and event["outcome"] == "deny"
            for event in audit_store.events
        )

    def test_missing_tool_name_is_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENFORCEAI_DB_PATH", "/tmp/enforceai.db")

        identity = IdentityContext(
            user_id="https://issuer.example|sub",
            agent_id=str(uuid.uuid4()),
            provider="oidc",
            scopes=["scope-1"],
            metadata=None,
        )
        resolver = _FakeResolver(identity=identity)
        audit_store = _RecordingAuditStore()

        monkeypatch.setattr(
            auth_server_module,
            "get_identity_resolver",
            lambda: resolver,
        )
        monkeypatch.setattr(
            auth_server_module,
            "load_scope_catalog",
            lambda: _catalog_allows(tool_name="good_tool"),
        )
        monkeypatch.setattr(
            auth_server_module,
            "get_enforceai_stores",
            lambda: _FakeStores(audit_store=audit_store),
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(method="tools/call", tool_name=None),
        )

        assert response.status_code == 403
        assert any(
            event["details"].get("reason") == "missing_tool_name"
            for event in audit_store.events
        )

    def test_dependency_failure_returns_503(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENFORCEAI_DB_PATH", "/tmp/enforceai.db")

        identity = IdentityContext(
            user_id="https://issuer.example|sub",
            agent_id=str(uuid.uuid4()),
            provider="oidc",
            scopes=["scope-1"],
            metadata=None,
        )
        resolver = _FakeResolver(identity=identity)

        monkeypatch.setattr(
            auth_server_module,
            "get_identity_resolver",
            lambda: resolver,
        )

        def _boom():
            raise DependencyUnavailableError("catalog down")

        monkeypatch.setattr(
            auth_server_module,
            "load_scope_catalog",
            _boom,
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(method="tools/call", tool_name="good_tool"),
        )

        assert response.status_code == 503

    def test_audit_failure_does_not_flip_allow(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENFORCEAI_DB_PATH", "/tmp/enforceai.db")

        identity = IdentityContext(
            user_id="https://issuer.example|sub",
            agent_id=str(uuid.uuid4()),
            provider="oidc",
            scopes=["scope-1"],
            metadata=None,
        )
        resolver = _FakeResolver(identity=identity)
        audit_store = _RecordingAuditStore(raise_on_append=True)

        monkeypatch.setattr(
            auth_server_module,
            "get_identity_resolver",
            lambda: resolver,
        )
        monkeypatch.setattr(
            auth_server_module,
            "load_scope_catalog",
            lambda: _catalog_allows(tool_name="good_tool"),
        )
        monkeypatch.setattr(
            auth_server_module,
            "get_enforceai_stores",
            lambda: _FakeStores(audit_store=audit_store),
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(method="tools/call", tool_name="good_tool"),
        )

        assert response.status_code == 200
