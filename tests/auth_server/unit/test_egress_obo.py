"""Unit tests for the OBO token-exchange engine (auth_server/egress_obo.py).

Covers:
- Entra jwt-bearer request body shape (grant_type, assertion, scope, on_behalf_of).
- .default scope synthesis vs explicit scopes.
- IdP error-code -> typed exception mapping.
- Keycloak path raises (Phase 4 stub).
- No caching: two calls hit the token endpoint twice.
- Missing gateway credentials -> config error.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from auth_server import egress_obo
from auth_server.egress_obo import (
    OboConfigError,
    OboConsentRequired,
    OboExchangeError,
    OboReauthRequired,
    OboUnsupportedIdpError,
    obo_exchange,
)


class _FakeEntraProvider:
    """Minimal stand-in for EntraIdProvider (class name carries the 'entra' kind)."""

    def __init__(self):
        self.client_id = "gw-client"
        self.client_secret = "gw-secret"
        self.token_url = "https://login.microsoftonline.com/tenant/oauth2/v2.0/token"


class _FakeKeycloakProvider:
    def __init__(self):
        self.client_id = "gw-client"
        self.client_secret = "gw-secret"
        self.token_url = "https://kc.example/realms/r/protocol/openid-connect/token"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _patch_post(monkeypatch, response, capture: dict):
    """Patch the SSRF-guarded client so .post records its args and returns `response`.

    obo_exchange routes the IdP token POST through
    ``registry.utils.url_guard.guarded_async_client`` (imported lazily inside the
    function), so we patch it at its source module. `capture` is filled with
    {"url":..., "data":..., "calls": n}.
    """
    capture["calls"] = 0

    async def _post(url, data=None, **kwargs):
        capture["calls"] += 1
        capture["url"] = url
        capture["data"] = data
        return response

    @asynccontextmanager
    async def _fake_client(*args, **kwargs):
        client = MagicMock()
        client.post = AsyncMock(side_effect=_post)
        yield client

    monkeypatch.setattr("registry.utils.url_guard.guarded_async_client", _fake_client)


@pytest.mark.unit
class TestEntraExchangeBody:
    @pytest.mark.asyncio
    async def test_jwt_bearer_body_shape(self, monkeypatch):
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "obo-tok"}), cap)

        token = await obo_exchange(
            _FakeEntraProvider(),
            subject_token="ingress-jwt",
            target_audience="api://outlook-mcp-server",
            scopes=[],
        )

        assert token == "obo-tok"
        body = cap["data"]
        assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
        assert body["assertion"] == "ingress-jwt"
        assert body["client_id"] == "gw-client"
        assert body["client_secret"] == "gw-secret"
        assert body["requested_token_use"] == "on_behalf_of"
        # No explicit scopes -> synthesize <target>/.default
        assert body["scope"] == "api://outlook-mcp-server/.default"

    @pytest.mark.asyncio
    async def test_explicit_scopes_passed_verbatim(self, monkeypatch):
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "t"}), cap)

        await obo_exchange(
            _FakeEntraProvider(),
            subject_token="j",
            target_audience="api://srv",
            scopes=["api://srv/Mail.Read", "api://srv/Files.Read"],
        )
        assert cap["data"]["scope"] == "api://srv/Mail.Read api://srv/Files.Read"

    @pytest.mark.asyncio
    async def test_no_cache_two_calls_hit_endpoint_twice(self, monkeypatch):
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "t"}), cap)
        p = _FakeEntraProvider()
        await obo_exchange(p, subject_token="j", target_audience="api://srv")
        await obo_exchange(p, subject_token="j", target_audience="api://srv")
        assert cap["calls"] == 2


@pytest.mark.unit
class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_invalid_grant_maps_to_reauth(self, monkeypatch):
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(400, {"error": "invalid_grant", "error_description": "expired"}),
            cap,
        )
        with pytest.raises(OboReauthRequired, match="expired"):
            await obo_exchange(_FakeEntraProvider(), subject_token="j", target_audience="api://srv")

    @pytest.mark.asyncio
    async def test_interaction_required_maps_to_consent(self, monkeypatch):
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(400, {"error": "interaction_required", "error_description": "consent"}),
            cap,
        )
        with pytest.raises(OboConsentRequired):
            await obo_exchange(_FakeEntraProvider(), subject_token="j", target_audience="api://srv")

    @pytest.mark.asyncio
    async def test_invalid_client_maps_to_config(self, monkeypatch):
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(401, {"error": "invalid_client"}),
            cap,
        )
        with pytest.raises(OboConfigError):
            await obo_exchange(_FakeEntraProvider(), subject_token="j", target_audience="api://srv")


@pytest.mark.unit
class TestUnsupportedAndConfig:
    @pytest.mark.asyncio
    async def test_keycloak_raises_not_implemented(self, monkeypatch):
        # Keycloak path is a Phase 4 stub; it must raise cleanly, not silently pass.
        with pytest.raises(OboUnsupportedIdpError, match="Keycloak"):
            await obo_exchange(
                _FakeKeycloakProvider(), subject_token="j", target_audience="srv-client"
            )

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_unsupported(self, monkeypatch):
        class _Cognito:
            client_id = "x"
            client_secret = "y"
            token_url = "https://z/token"

        with pytest.raises(OboUnsupportedIdpError):
            await obo_exchange(_Cognito(), subject_token="j", target_audience="a")

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_config(self, monkeypatch):
        class _NoCreds:
            client_id = ""
            client_secret = ""
            token_url = ""

        with pytest.raises(OboConfigError):
            await obo_exchange(_NoCreds(), subject_token="j", target_audience="a")


@pytest.mark.unit
class TestSsrfGuard:
    """The IdP token POST is routed through the SSRF-guarded client (#1396 parity)."""

    @pytest.mark.asyncio
    async def test_guard_rejection_maps_to_exchange_error_without_leaking(self, monkeypatch):
        """If the guarded client rejects the token endpoint (private/metadata IP,
        bad scheme, DNS rebind), obo_exchange fails closed with OboExchangeError and
        never sends the assertion/client_secret."""
        from registry.exceptions import UrlValidationError

        def _blocking_client(*args, **kwargs):
            # The guard validates the target when the context manager is created
            # (before any bytes leave), so raising here models a rejected endpoint.
            raise UrlValidationError("https://169.254.169.254/token", "resolves to metadata IP")

        monkeypatch.setattr("registry.utils.url_guard.guarded_async_client", _blocking_client)

        with pytest.raises(OboExchangeError, match="SSRF guard"):
            await obo_exchange(_FakeEntraProvider(), subject_token="j", target_audience="api://srv")

    @pytest.mark.asyncio
    async def test_success_path_uses_guarded_client(self, monkeypatch):
        """The happy path flows through the guarded client (proves the POST is
        actually routed through it, not a raw httpx.AsyncClient)."""
        capture: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "ok"}), capture)
        # Make a raw httpx.AsyncClient blow up so a regression to the unguarded
        # path would fail loudly rather than silently pass.
        monkeypatch.setattr(
            egress_obo.httpx,
            "AsyncClient",
            MagicMock(side_effect=AssertionError("must use guarded client")),
        )
        token = await obo_exchange(
            _FakeEntraProvider(), subject_token="j", target_audience="api://srv"
        )
        assert token == "ok"
        assert capture["calls"] == 1


class TestOboFailureReason:
    """The audit failure_reason mapping used when an OBO mint is emitted to the
    token-mint audit stream (auth_server.server._obo_failure_reason). Groups the
    typed exception hierarchy into stable, low-cardinality reason codes so audit
    consumers can bucket by failure class without parsing free-text detail."""

    @pytest.mark.parametrize(
        "exc_name, expected",
        [
            ("OboReauthRequired", "reauth_required"),
            ("OboConsentRequired", "consent_required"),
            ("OboConfigError", "config_error"),
            ("OboUnsupportedIdpError", "unsupported_idp"),
            ("OboExchangeError", "exchange_failed"),
        ],
    )
    def test_typed_exception_maps_to_reason(self, exc_name, expected):
        # Construct the exception from the SAME module object server.py imports
        # (bare `from egress_obo import ...`), which conftest resolves as a
        # distinct module from `auth_server.egress_obo`. Using the class off
        # auth_server.server guarantees isinstance identity matches.
        import auth_server.server as server

        exc_cls = getattr(server, exc_name)
        assert server._obo_failure_reason(exc_cls("x")) == expected
