"""Unit tests for the OBO token-exchange engine (auth_server/egress_obo.py).

Covers:
- Entra jwt-bearer request body shape (grant_type, assertion, scope, on_behalf_of).
- .default scope synthesis vs explicit scopes.
- Keycloak RFC 8693 request body shape (subject_token, audience, client auth).
- IdP error-code -> typed exception mapping.
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


class _FakeNonJsonResponse:
    """A response whose body is not JSON — what a misbehaving proxy in front of
    the IdP returns. The error path already tolerates this; the success path
    must too."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.text = "<html>gateway timeout</html>"

    def json(self):
        raise ValueError("not json")


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
        with pytest.raises(OboReauthRequired, match="rejected the user assertion"):
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

    @pytest.mark.asyncio
    async def test_entra_access_denied_stays_generic(self, monkeypatch):
        """access_denied must NOT be reclassified on the Entra path.

        The Keycloak remediation hint would be wrong there (Entra returns this
        for denied consent and for conditional-access blocks — a different fix,
        and for CA not operator configuration at all), and moving an already
        released code path out of the exchange_failed audit bucket would break
        any alerting keyed on it. This test exists to keep that boundary.
        """
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(403, {"error": "access_denied"}), cap)
        with pytest.raises(OboExchangeError) as excinfo:
            await obo_exchange(_FakeEntraProvider(), subject_token="j", target_audience="api://srv")
        assert not isinstance(excinfo.value, OboConfigError)
        assert "Keycloak" not in str(excinfo.value)


@pytest.mark.unit
class TestKeycloakExchangeBody:
    @pytest.mark.asyncio
    async def test_token_exchange_body_shape(self, monkeypatch):
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "obo-tok"}), cap)

        token = await obo_exchange(
            _FakeKeycloakProvider(),
            subject_token="ingress-jwt",
            target_audience="finance-mcp-server",
            scopes=[],
        )

        assert token == "obo-tok"
        # The credential and the user's JWT must go to the CONFIGURED endpoint and
        # nowhere else; without this a regression that redirected the POST would
        # pass every other assertion in this file.
        assert cap["url"] == _FakeKeycloakProvider().token_url
        body = cap["data"]
        assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
        assert body["subject_token"] == "ingress-jwt"
        assert body["subject_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
        # Pinned, not left to the server default: legacy Keycloak defaults to
        # refresh_token and would mint a credential this code discards.
        assert body["requested_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
        # Keycloak takes the bare target client id as audience, never an https URL
        # and never Entra's assertion/requested_token_use convention.
        assert body["audience"] == "finance-mcp-server"
        assert body["client_id"] == "gw-client"
        assert body["client_secret"] == "gw-secret"
        assert "assertion" not in body
        assert "requested_token_use" not in body
        # No explicit scopes -> omit the field entirely so Keycloak applies
        # the target client's defaults.
        assert "scope" not in body

    @pytest.mark.asyncio
    async def test_explicit_scopes_joined_into_scope(self, monkeypatch):
        """Scopes are space-joined into `scope` when explicitly requested.

        The inputs deliberately mirror what registration actually accepts:
        ServerInfo._validate_egress_auth binds every scope's resource prefix to
        target_audience, so bare OIDC names like "profile" are rejected before
        they could ever reach this builder. Asserting on those would be green
        and meaningless.
        """
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "t"}), cap)

        await obo_exchange(
            _FakeKeycloakProvider(),
            subject_token="j",
            target_audience="srv-client",
            scopes=["srv-client/read", "srv-client/write"],
        )
        assert cap["data"]["scope"] == "srv-client/read srv-client/write"

    @pytest.mark.asyncio
    async def test_invalid_token_maps_to_reauth(self, monkeypatch):
        """An expired subject_token must surface as re-auth, not a generic failure.

        Keycloak never answers invalid_grant for token-exchange — legacy
        exchange reports an unusable subject_token as invalid_token — so
        mapping only invalid_grant would leave the single most likely runtime
        failure (the ingress JWT expiring mid-flight) in the generic bucket.
        """
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(400, {"error": "invalid_token", "error_description": "Invalid token"}),
            cap,
        )
        with pytest.raises(OboReauthRequired):
            await obo_exchange(
                _FakeKeycloakProvider(), subject_token="j", target_audience="srv-client"
            )

    @pytest.mark.asyncio
    async def test_invalid_request_stays_generic(self, monkeypatch):
        """invalid_request must NOT be classified — the code is overloaded.

        Keycloak standard exchange answers it for an expired subject_token, for
        the client's exchange toggle being off, and for an unplaceable audience.
        Two are operator config and one is user re-auth; guessing config would
        stop the caller retrying with a fresh token. error_description is the
        only discriminator and it is logged, not classified on.
        """
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(400, {"error": "invalid_request", "error_description": "Invalid token"}),
            cap,
        )
        with pytest.raises(OboExchangeError) as excinfo:
            await obo_exchange(
                _FakeKeycloakProvider(), subject_token="j", target_audience="srv-client"
            )
        assert not isinstance(excinfo.value, OboConfigError | OboReauthRequired)

    @pytest.mark.asyncio
    async def test_unsupported_grant_type_maps_to_config(self, monkeypatch):
        """Keycloak without the token-exchange feature enabled."""
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(400, {"error": "unsupported_grant_type"}), cap)
        with pytest.raises(OboConfigError, match="token-exchange grant"):
            await obo_exchange(
                _FakeKeycloakProvider(), subject_token="j", target_audience="srv-client"
            )

    @pytest.mark.asyncio
    async def test_access_denied_maps_to_config(self, monkeypatch):
        """Keycloak answers access_denied when the target client has not granted
        the token-exchange permission — the common first-run failure. It must
        surface as an actionable configuration error, not a generic exchange
        failure."""
        cap: dict = {}
        _patch_post(
            monkeypatch,
            _FakeResponse(
                403,
                {"error": "access_denied", "error_description": "Client not allowed to exchange"},
            ),
            cap,
        )
        with pytest.raises(OboConfigError, match="token-exchange permission"):
            await obo_exchange(
                _FakeKeycloakProvider(), subject_token="j", target_audience="srv-client"
            )


@pytest.mark.unit
class TestUnsupportedAndConfig:
    @pytest.mark.asyncio
    async def test_blank_target_audience_is_config_error(self, monkeypatch):
        """No credential leaves the process without a target to exchange for."""
        cap: dict = {}
        _patch_post(monkeypatch, _FakeResponse(200, {"access_token": "t"}), cap)
        with pytest.raises(OboConfigError, match="target_audience"):
            await obo_exchange(_FakeKeycloakProvider(), subject_token="j", target_audience="   ")
        assert cap["calls"] == 0

    @pytest.mark.asyncio
    async def test_non_json_success_body_is_typed_error(self, monkeypatch):
        """A 200 with a broken body must stay inside the OboExchangeError family.

        Otherwise the ValueError escapes the caller's `except OboExchangeError`
        and the user gets a 500 instead of a JSON-RPC failure.
        """
        cap: dict = {}
        _patch_post(monkeypatch, _FakeNonJsonResponse(200), cap)
        with pytest.raises(OboExchangeError, match="non-JSON"):
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

        with pytest.raises(OboExchangeError, match="security policy"):
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

    @pytest.mark.asyncio
    async def test_uses_dedicated_empty_allowlist_profile(self, monkeypatch):
        profile_capture: dict = {}

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def strict_client(*, profile, timeout):
            profile_capture["profile"] = profile
            client = MagicMock()
            client.post = AsyncMock(return_value=_FakeResponse(200, {"access_token": "ok"}))
            yield client

        monkeypatch.setattr("registry.utils.url_guard.guarded_async_client", strict_client)
        token = await obo_exchange(
            _FakeEntraProvider(), subject_token="j", target_audience="api://srv"
        )
        assert token == "ok"
        profile = profile_capture["profile"]
        assert profile is egress_obo.CREDENTIALED_OAUTH_PROFILE
        assert profile.allowlist_factory().hosts == frozenset()

    @pytest.mark.asyncio
    async def test_http_token_url_fails_before_client_is_opened(self, monkeypatch):
        provider = _FakeEntraProvider()
        provider.token_url = "http://93.184.216.34/token"
        client = MagicMock(side_effect=AssertionError("client must not open"))
        monkeypatch.setattr("registry.utils.url_guard.guarded_async_client", client)
        with pytest.raises(OboExchangeError, match="security policy"):
            await obo_exchange(provider, subject_token="assertion", target_audience="api://srv")
        client.assert_not_called()


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
