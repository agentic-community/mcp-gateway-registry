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
        "custom_header_names": ["Authorization"],
        "custom_header_overridable_names": ["Authorization"],
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
    def _build(
        claims,
        entity,
        *,
        target="https://llm.example/",
        decrypted=None,
        decrypt_error=False,
    ):
        monkeypatch.setattr(routes, "verify_generic_proxy_token", lambda tok: claims)
        repo = _StubRepo(entity)
        monkeypatch.setattr(routes, "_proxyable_repo_for", lambda et: repo)
        monkeypatch.setattr(routes, "resolve_proxy_target", lambda et, doc: target)

        def _decrypt(_encrypted, *, strict=False):
            assert strict is True
            if decrypt_error:
                raise ValueError("bad ciphertext")
            return (
                decrypted
                if decrypted is not None
                else [{"name": "Authorization", "value": "Bearer sk-x"}]
            )

        monkeypatch.setattr(routes, "decrypt_custom_headers", _decrypt)
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

    def test_same_host_path_repoint_refused_403(self, make_client):
        client = make_client(
            _claims(upstream_url="https://llm.example/v1"),
            _entity(),
            target="https://llm.example/v2",
        )
        assert _post(client).status_code == 403

    def test_same_path_query_repoint_refused_403(self, make_client):
        client = make_client(
            _claims(upstream_url="https://llm.example/v1?tenant=a"),
            _entity(),
            target="https://llm.example/v1?tenant=b",
        )
        assert _post(client).status_code == 403

    def test_equivalent_normalized_full_target_vends(self, make_client):
        client = make_client(
            _claims(upstream_url="HTTPS://LLM.Example:443/v1?tenant=a"),
            _entity(),
            target="https://llm.example/v1?tenant=a",
        )
        assert _post(client).status_code == 200

    def test_entity_gone_fails_closed(self, make_client):
        client = make_client(_claims(), None)
        resp = _post(client)
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "entity",
        [
            _entity(is_enabled=False),
            _entity(proxy_target_url=None),
            _entity(proxy_disabled_reason="dns refresh failed"),
        ],
    )
    def test_inactive_or_targetless_entity_fails_closed(self, make_client, entity):
        client = make_client(_claims(), entity, target=None)
        resp = _post(client)
        assert resp.status_code == 409

    def test_same_origin_different_path_is_refused(self, make_client):
        client = make_client(
            _claims(upstream_url="https://llm.example/v2"),
            _entity(proxy_target_url="https://llm.example/v1"),
            target="https://llm.example/v1",
        )
        resp = _post(client)
        assert resp.status_code == 403

    def test_equivalent_normalized_target_is_accepted(self, make_client):
        client = make_client(
            _claims(upstream_url="https://LLM.EXAMPLE:443/v1"),
            _entity(proxy_target_url="https://llm.example/v1"),
            target="https://llm.example/v1",
        )
        resp = _post(client)
        assert resp.status_code == 200

    def test_same_path_different_query_is_refused(self, make_client):
        client = make_client(
            _claims(upstream_url="https://llm.example/v1?tenant=attacker"),
            _entity(proxy_target_url="https://llm.example/v1?tenant=registered"),
            target="https://llm.example/v1?tenant=registered",
        )
        resp = _post(client)
        assert resp.status_code == 403

    def test_decryption_failure_fails_closed(self, make_client):
        client = make_client(_claims(), _entity(), decrypt_error=True)
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credentials are unavailable"

    def test_missing_expected_fixed_header_fails_closed(self, make_client):
        client = make_client(
            _claims(),
            _entity(
                custom_header_names=["X-Api-Key"],
                custom_header_overridable_names=[],
                custom_headers_encrypted=[],
            ),
            decrypted=[],
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credentials are incomplete"

    def test_caller_only_slot_does_not_require_stored_value(self, make_client):
        client = make_client(
            _claims(),
            _entity(
                custom_header_names=["X-Caller-Token"],
                custom_header_overridable_names=["X-Caller-Token"],
                custom_headers_encrypted=[],
            ),
            decrypted=[],
        )
        resp = _post(client)
        assert resp.status_code == 200
        assert resp.json() == {"headers": {}, "overridable_names": ["X-Caller-Token"]}

    @pytest.mark.parametrize(
        ("entity", "decrypted"),
        [
            (
                _entity(
                    proxy_target_url="http://llm.example/",
                    custom_header_names=["X-Api-Key"],
                    custom_header_overridable_names=[],
                ),
                [{"name": "X-Api-Key", "value": "secret"}],
            ),
            (
                _entity(
                    proxy_target_url="http://llm.example/",
                    custom_headers_encrypted=[],
                    custom_header_names=["X-Caller-Token"],
                    custom_header_overridable_names=["X-Caller-Token"],
                ),
                [],
            ),
        ],
    )
    def test_credential_headers_refused_for_http_target(self, make_client, entity, decrypted):
        client = make_client(
            _claims(upstream_url="http://llm.example/"),
            entity,
            target="http://llm.example/",
            decrypted=decrypted,
        )
        resp = _post(client)
        assert resp.status_code == 403
        assert resp.json()["detail"] == "upstream credential headers require an HTTPS target"


class TestOverridableNamesVended:
    """The vend surfaces the caller passthrough allowlist and backstops it against
    the never-forward gateway-cred denylist."""

    def test_overridable_names_returned(self, make_client):
        client = make_client(
            _claims(),
            _entity(
                custom_header_names=["Authorization", "X-Tenant"],
                custom_header_overridable_names=["X-Tenant", "Authorization"],
            ),
        )
        resp = _post(client)
        assert resp.status_code == 200
        assert sorted(resp.json()["overridable_names"]) == ["Authorization", "X-Tenant"]

    def test_empty_when_none_registered(self, make_client):
        client = make_client(
            _claims(),
            _entity(
                custom_headers_encrypted=[],
                custom_header_names=[],
                custom_header_overridable_names=[],
            ),
            decrypted=[],
        )
        resp = _post(client)
        assert resp.json()["overridable_names"] == []

    def test_invalid_overridable_name_metadata_fails_closed(self, make_client):
        client = make_client(
            _claims(),
            _entity(custom_header_overridable_names=["X-Bad\rInjected"]),
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"

    def test_reserved_names_backstopped_out(self, make_client):
        # A bypass-written doc lists gateway-managed names as overridable; the
        # vend drops those names, while consistent safe registrations survive.
        client = make_client(
            _claims(),
            _entity(
                custom_header_names=["Authorization", "X-Tenant"],
                custom_header_overridable_names=[
                    "X-Tenant",
                    "Cookie",
                    "X-Authorization",
                    "Authorization",
                    "Host",
                ],
            ),
        )
        resp = _post(client)
        assert sorted(resp.json()["overridable_names"]) == ["Authorization", "X-Tenant"]

    def test_default_values_backstopped_against_internal_names(self, make_client):
        # A bypass-written doc stores internal-header values alongside a valid
        # registered default. Internal names are dropped before consistency checks.
        client = make_client(
            _claims(),
            _entity(custom_header_names=["Authorization", "X-Api-Key"]),
            decrypted=[
                {"name": "X-Api-Key", "value": "sk-ok"},
                {"name": "X-Internal-Token-Generic", "value": "sneaky"},
                {"name": "X-User", "value": "spoofed"},
            ],
        )
        resp = _post(client)
        assert resp.json()["headers"] == {"X-Api-Key": "sk-ok"}

    def test_unregistered_decrypted_header_fails_closed(self, make_client):
        client = make_client(
            _claims(),
            _entity(),
            decrypted=[
                {"name": "Authorization", "value": "Bearer expected"},
                {"name": "X-Unregistered-Secret", "value": "must-not-vend"},
            ],
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"

    def test_duplicate_decrypted_header_names_fail_closed(self, make_client):
        client = make_client(
            _claims(),
            _entity(
                custom_header_names=["Authorization", "X-Api-Key"],
                custom_header_overridable_names=["Authorization"],
            ),
            decrypted=[
                {"name": "Authorization", "value": "Bearer expected"},
                {"name": "X-Api-Key", "value": "first-secret"},
                {"name": "x-api-key", "value": "second-secret"},
            ],
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"

    def test_overridable_name_must_be_registered(self, make_client):
        client = make_client(
            _claims(),
            _entity(custom_header_overridable_names=["Authorization", "X-Unregistered"]),
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"


class TestUrlHelpers:
    """The private URL helpers that define the upstream comparison surface."""

    def test_base_url_lowercases_scheme_and_netloc_and_drops_path(self):
        assert routes._base_url("HTTP://Host:8080/path?q=1") == "http://host:8080"

    def test_base_url_bare_origin(self):
        assert routes._base_url("https://LLM.Example") == "https://llm.example"

    def test_registered_upstreams_unions_proxy_pass_and_versions(self):
        # versions carries a mix of a plain dict and an attribute-style object;
        # both contribute their base URL, unioned with the top-level proxy_pass_url.
        version_obj = SimpleNamespace(proxy_pass_url="HTTPS://V2.Example:443/api")
        server = {
            "proxy_pass_url": "http://Primary:9000/mcp",
            "versions": [
                {"proxy_pass_url": "https://V1.Example/path"},
                version_obj,
                {"proxy_pass_url": None},
                SimpleNamespace(proxy_pass_url=None),
            ],
        }
        assert routes._registered_upstreams(server) == {
            "http://primary:9000",
            "https://v1.example",
            "https://v2.example:443",
        }

    def test_registered_upstreams_empty_when_no_upstreams(self):
        assert routes._registered_upstreams({}) == set()
        assert routes._registered_upstreams({"versions": []}) == set()


class TestProxyableRepoFor:
    """Each entity-type token routes to the repository that owns it."""

    def test_known_types_route_to_dedicated_repos(self, monkeypatch):
        agent_repo = object()
        skill_repo = object()
        server_repo = object()
        custom_repo = object()
        monkeypatch.setattr(routes, "get_agent_repository", lambda: agent_repo)
        monkeypatch.setattr(routes, "get_skill_repository", lambda: skill_repo)
        monkeypatch.setattr(routes, "get_server_repository", lambda: server_repo)
        monkeypatch.setattr(routes, "get_custom_entity_repository", lambda: custom_repo)

        assert routes._proxyable_repo_for("a2a_agent") is agent_repo
        assert routes._proxyable_repo_for("skill") is skill_repo
        assert routes._proxyable_repo_for("mcp_server") is server_repo

    def test_unknown_type_routes_to_custom_entity_repo(self, monkeypatch):
        custom_repo = object()
        monkeypatch.setattr(routes, "get_custom_entity_repository", lambda: custom_repo)
        assert routes._proxyable_repo_for("my_custom_type") is custom_repo


class TestTargetIdentityErrors:
    """Target cross-check failure modes distinct from the plain != mismatch."""

    def test_normalize_identity_raising_refuses_403(self, make_client, monkeypatch):
        # A target that normalize_url_identity cannot canonicalize fails closed at
        # 403, never leaking headers for an unverifiable upstream.
        client = make_client(_claims(), _entity())

        def _boom(_url):
            raise ValueError("unparseable upstream")

        monkeypatch.setattr(routes, "normalize_url_identity", _boom)
        resp = _post(client)
        assert resp.status_code == 403
        assert resp.json()["detail"] == "upstream not registered for this entity"

    def test_non_list_metadata_fails_closed_500(self, make_client):
        # A bypass-written doc stores a string where a name list is required; the
        # isinstance guard fails closed before any decryption.
        client = make_client(
            _claims(),
            _entity(custom_header_names="Authorization"),
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"

    def test_non_list_overridable_metadata_fails_closed_500(self, make_client):
        client = make_client(
            _claims(),
            _entity(custom_header_overridable_names="Authorization"),
        )
        resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "stored upstream credential metadata is invalid"
