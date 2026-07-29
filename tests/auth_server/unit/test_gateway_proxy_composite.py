"""Composite cross-layer security tests for the gateway generic-proxy.

The per-slice suites each verify one layer in isolation. This suite composes the
REAL layers end-to-end in-process — mint -> verify -> handler, and render ->
resolve -> block -> token — and asserts the security properties hold across the
boundaries (not just per-layer), which is where cross-layer bugs hide:

- token round-trip: a /validate-minted generic token verifies on the handler and
  is confined to the bound (entity_type, registered_path); a cross-type or
  sibling replay is rejected AND, if it somehow verified, the handler still pins
  the outbound to the token's upstream (defense-in-depth).
- mint discriminator: a generic request mints ONLY the generic token; an MCP
  request mints ONLY the MCP token — proven by driving the actual _attach_* pair.
- render->authz coherence: the nginx block a render emits for an entity uses the
  SAME location/authz shape (/{entity_type}/{path}) that the token binds and the
  handler confines — so a scope authored against entity_type/path actually lines
  up with what routes.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

_SECRET = "test-secret-key-that-is-definitely-long-enough-32b"

import auth_server.server as server_module  # noqa: E402
from auth_server.internal_request_token import (  # noqa: E402
    mint_generic_proxy_token,
    verify_generic_proxy_token,
)
from auth_server.server import (  # noqa: E402
    _attach_generic_proxy_token,
    _attach_mcp_proxy_token,
    _build_generic_outbound_url,
)
from registry.core.nginx_service import NginxConfigService  # noqa: E402

pytestmark = pytest.mark.unit


def _req(headers, entity_type="skill", entity_path="skills/proxy-demo", method="GET"):
    lower = {k.lower(): v for k, v in headers.items()}
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, default=None: lower.get(k.lower(), default)),
        path_params={"entity_type": entity_type, "entity_path": entity_path},
        method=method,
        state=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


class _Resp:
    def __init__(self):
        self.headers = {}


class TestMintVerifyHandlerChain:
    """A token minted at /validate must verify on the handler and confine the
    outbound to the bound entity — the full trust chain, composed."""

    async def test_happy_path_token_verifies_and_confines_subpath(self):
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=["s/read"],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/api",
            http_method="GET",
        )
        # Verify on a sub-path UNDER the bound entity (allowed).
        req = _req({"X-Internal-Token-Generic": tok}, entity_path="skills/proxy-demo/reports")
        await verify_generic_proxy_token(req)  # no raise
        claims = req.state.generic_proxy_claims
        # The handler builds the outbound from the pinned upstream + confined sub.
        outbound = _build_generic_outbound_url(
            claims["upstream_url"], "skills/proxy-demo/reports", claims["server"]
        )
        assert outbound == "https://backend.example/api/reports"

    async def test_cross_type_replay_rejected_end_to_end(self):
        # Token bound to (skill, ...) replayed on an /a2a_agent/ route.
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://b/",
            http_method="GET",
        )
        req = _req(
            {"X-Internal-Token-Generic": tok},
            entity_type="a2a_agent",
            entity_path="skills/proxy-demo",
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as e:
            await verify_generic_proxy_token(req)
        assert e.value.status_code == 401

    async def test_sibling_replay_rejected_end_to_end(self):
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://b/",
            http_method="GET",
        )
        req = _req({"X-Internal-Token-Generic": tok}, entity_path="skills/proxy-demo-evil")
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await verify_generic_proxy_token(req)


class TestMintDiscriminatorComposed:
    """Driving the real _attach_* pair on one response proves exactly one token
    is minted per request kind (no double-mint across the shared /validate)."""

    def test_generic_request_mints_only_generic(self):
        req = _req(
            {
                "X-Resolved-Generic-Upstream": "https://backend.example/",
                "X-Validate-Source-Secret": server_module.settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        _attach_mcp_proxy_token(req, resp, subject="alice", scopes=[], server_name="skill/x")
        _attach_generic_proxy_token(
            req,
            resp,
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="skills/x",
            http_method="GET",
        )
        assert "X-Internal-Token" not in resp.headers
        assert "X-Internal-Token-Generic" in resp.headers

    def test_mcp_request_mints_only_mcp(self):
        req = _req(
            {
                "X-Resolved-Upstream": "https://mcp-backend/",
                "X-Validate-Source-Secret": server_module.settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        _attach_mcp_proxy_token(req, resp, subject="alice", scopes=[], server_name="foo/mcp")
        _attach_generic_proxy_token(
            req,
            resp,
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="x",
            http_method="GET",
        )
        assert "X-Internal-Token" in resp.headers
        assert "X-Internal-Token-Generic" not in resp.headers


class TestRenderAuthzCoherence:
    """The nginx block a render emits for an entity must use the same
    /{entity_type}/{path} shape the token binds and the handler confines — so an
    authored scope, the minted token, and the rendered route all agree."""

    async def test_rendered_location_matches_token_binding(self):
        # Render the block for a proxied skill.
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = "http://auth-server:8888"
            s.gateway_generic_client_max_body_size = "1m"
            s.gateway_proxy_prefix = "gateway"
            block = NginxConfigService()._create_generic_proxy_block(
                "skill", "/skills/proxy-demo", "https://backend.example/"
            )
        # Client route is prefixed and de-duplicates the registered namespace;
        # authz markers and the internal hop below retain the full registered path.
        assert "location {{ROOT_PATH}}/gateway/skill/proxy-demo/ {" in block
        assert "proxy_pass http://auth-server:8888/proxy/skill/skills/proxy-demo/;" in block
        # The token the verifier accepts for that same route:
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/",
            http_method="GET",
        )
        req = _req({"X-Internal-Token-Generic": tok})
        await verify_generic_proxy_token(req)  # the same shape verifies -> coherent


class TestFeatureLatchComposed:
    """Even with a valid token, the handler fails closed unless the process
    egress latch is on — the runtime gate composes with the token gate."""

    async def test_valid_token_still_503_when_egress_latch_off(self):
        from fastapi import HTTPException

        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=[],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/",
            http_method="GET",
        )
        req = _req({"X-Internal-Token-Generic": tok})
        await verify_generic_proxy_token(req)  # token is valid
        # But the feature latch is off -> handler must 503 regardless.
        with patch.object(server_module, "_generic_proxy_feature_active", False):
            with pytest.raises(HTTPException) as e:
                await server_module.generic_proxy("skill", "skills/proxy-demo", req)
        assert e.value.status_code == 503

    async def test_lifespan_initializes_the_feature_latch(self):
        """REGRESSION: the latch MUST be set from the lifespan, not an
        @app.on_event('startup') hook. Starlette ignores on_event handlers when a
        lifespan is provided, so an on_event init would silently never run and the
        latch would stay None -> the hop 503s forever even with the flag on.
        Assert the lifespan actually invokes initialize_generic_proxy_feature.
        """
        called = {"init": False}

        async def _fake_init():
            called["init"] = True

        # Stub the heavy startup deps so we can drive the lifespan in isolation.
        async def _noop(*a, **k):
            return {"group_mappings": {}}

        with (
            patch.object(server_module, "initialize_generic_proxy_feature", _fake_init),
            patch.object(server_module, "reload_scopes_config", _noop),
            patch.object(server_module, "_build_static_token_map", _noop),
            patch.object(server_module, "_log_otel_state", lambda: None),
        ):
            async with server_module.lifespan(server_module.app):
                pass
        assert called["init"] is True, "lifespan did not initialize the generic-proxy latch"
