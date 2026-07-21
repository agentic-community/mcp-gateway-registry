"""Authz + resolution tests for POST /internal/generic-upstream-headers.

Drives each security branch with the dependencies stubbed:
- validate_internal_auth overridden (caller already authenticated).
- verify_generic_proxy_token monkeypatched to return controlled claims
  (its own decode/verify is covered by the proxied_token tests).
- the proxyable repo + resolve_proxy_target + decrypt_custom_headers stubbed.

Security properties asserted:
- vends decrypted headers on the happy path;
- body must agree with the signed claims (403 on mismatch);
- upstream cross-check: a token upstream not matching the entity's registered
  target is refused (403), so a forged X-Resolved-Generic-Upstream cannot pull
  an entity's credentials toward an attacker host;
- clean misses (entity gone / not proxyable) return an EMPTY header set;
- missing token -> 401.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


class _StubRepo:
    def __init__(self, entity):
        self._entity = entity
        self.queried_paths: list[str] = []

    async def get(self, path):
        self.queried_paths.append(path)
        return self._entity


def _entity(**over):
    """A proxyable entity model stand-in (exposes model_dump like a pydantic model)."""
    doc = {
        "is_proxied": True,
        "proxy_target_url": "https://llm.example/",
        "custom_headers_encrypted": [{"name": "Authorization", "value_encrypted": "enc"}],
    }
    doc.update(over)
    return SimpleNamespace(model_dump=lambda: dict(doc))


def _claims(**over):
    base = {
        "entity_type": "custom-llm",
        "server": "custom-llm/chat",
        "upstream_url": "https://llm.example/",
        "has_upstream_auth": True,
    }
    base.update(over)
    return base


@pytest.fixture
def make_client(monkeypatch):
    def _build(claims, entity, *, target="https://llm.example/", decrypted=None):
        monkeypatch.setattr(routes, "verify_generic_proxy_token", lambda tok: claims)
        repo = _StubRepo(entity)
        monkeypatch.setattr(routes, "_proxyable_repo_for", lambda et: repo)
        monkeypatch.setattr(routes, "resolve_proxy_target", lambda et, doc: target)
        monkeypatch.setattr(
            routes,
            "decrypt_custom_headers",
            lambda enc: (
                decrypted
                if decrypted is not None
                else [{"name": "Authorization", "value": "Bearer sk-x"}]
            ),
        )
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.validate_internal_auth] = lambda: "auth-server"
        client = TestClient(app)
        client._repo = repo
        return client

    return _build


def _post(client, entity_type="custom-llm", registered_path="custom-llm/chat", token="gtok"):
    return client.post(
        "/internal/generic-upstream-headers",
        json={"entity_type": entity_type, "registered_path": registered_path},
        headers={"X-Internal-Token-Generic": token},
    )


class TestVend:
    def test_happy_path_vends_decrypted_headers(self, make_client):
        client = make_client(_claims(), _entity())
        resp = _post(client)
        assert resp.status_code == 200
        assert resp.json()["headers"] == {"Authorization": "Bearer sk-x"}

    def test_missing_token_401(self, make_client):
        client = make_client(_claims(), _entity())
        resp = client.post(
            "/internal/generic-upstream-headers",
            json={"entity_type": "custom-llm", "registered_path": "custom-llm/chat"},
        )
        assert resp.status_code == 401

    def test_body_token_mismatch_403(self, make_client):
        client = make_client(_claims(entity_type="skill"), _entity())
        # Body says custom-llm, token says skill -> refuse.
        resp = _post(client, entity_type="custom-llm")
        assert resp.status_code == 403

    def test_upstream_mismatch_refused_403(self, make_client):
        # Token pins an upstream that is NOT the entity's registered target.
        client = make_client(
            _claims(upstream_url="https://attacker.example/"),
            _entity(),
            target="https://llm.example/",
        )
        resp = _post(client)
        assert resp.status_code == 403

    def test_entity_gone_returns_empty(self, make_client):
        client = make_client(_claims(), None)
        resp = _post(client)
        assert resp.status_code == 200
        assert resp.json()["headers"] == {}

    def test_not_proxyable_returns_empty(self, make_client):
        # resolve_proxy_target returns None (disabled / federated / targetless).
        client = make_client(_claims(), _entity(), target=None)
        resp = _post(client)
        assert resp.status_code == 200
        assert resp.json()["headers"] == {}


class TestOverridableNamesVended:
    """The vend surfaces the caller passthrough allowlist and backstops it against
    the never-forward gateway-cred denylist."""

    def test_overridable_names_returned(self, make_client):
        client = make_client(
            _claims(),
            _entity(custom_header_overridable_names=["X-Tenant", "Authorization"]),
        )
        resp = _post(client)
        assert resp.status_code == 200
        assert sorted(resp.json()["overridable_names"]) == ["Authorization", "X-Tenant"]

    def test_empty_when_none_registered(self, make_client):
        client = make_client(_claims(), _entity())
        resp = _post(client)
        assert resp.json()["overridable_names"] == []

    def test_reserved_names_backstopped_out(self, make_client):
        # A bypass-written doc lists a gateway-cred name as overridable; the vend
        # must drop everything reserved EXCEPT Authorization.
        client = make_client(
            _claims(),
            _entity(
                custom_header_overridable_names=[
                    "X-Tenant",
                    "Cookie",
                    "X-Authorization",
                    "Authorization",
                    "Host",
                ]
            ),
        )
        resp = _post(client)
        assert sorted(resp.json()["overridable_names"]) == ["Authorization", "X-Tenant"]

    def test_default_values_backstopped_against_internal_names(self, make_client):
        # A bypass-written doc stores an internal-header name as an operator
        # DEFAULT (not just overridable). The vend must drop it so the hop can
        # never inject a gateway-internal header toward the backend.
        client = make_client(
            _claims(),
            _entity(),
            decrypted=[
                {"name": "X-Api-Key", "value": "sk-ok"},
                {"name": "X-Internal-Token-Generic", "value": "sneaky"},
                {"name": "X-User", "value": "spoofed"},
            ],
        )
        resp = _post(client)
        assert resp.json()["headers"] == {"X-Api-Key": "sk-ok"}
