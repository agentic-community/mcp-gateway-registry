"""Tests for the A2A agent JWT authentication middleware.

These tests fail against an unauthenticated agent (any request succeeds) and pass
against the guarded agent (only valid, correctly-issued tokens succeed; health
probes stay open; misconfiguration fails closed).
"""

import importlib.util
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from starlette.testclient import TestClient

KEYCLOAK_URL = "https://keycloak.example.com"
REALM = "mcp-gateway"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"


def _load_auth_module(agent_name):
    path = Path(__file__).parents[3] / "agents" / "a2a" / "src" / agent_name / "auth_middleware.py"
    spec = importlib.util.spec_from_file_location(f"sample_auth_{agent_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_travel_auth = _load_auth_module("travel-assistant-agent")
AuthConfigurationError = _travel_auth.AuthConfigurationError
JWTAuthMiddleware = _travel_auth.JWTAuthMiddleware
install_agent_auth = _travel_auth.install_agent_auth


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate an RSA keypair for signing test tokens."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return private_pem, public_key


def _mint_token(private_pem, issuer=ISSUER, expired=False, audience=None):
    """Mint an RS256 JWT with the given issuer/expiry/audience."""
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": "test-caller",
        "iat": now,
        "exp": now - 60 if expired else now + 300,
    }
    if audience is not None:
        claims["aud"] = audience
    return jwt.encode(claims, private_pem, algorithm="RS256")


def _build_app(rsa_keypair, monkeypatch, audience=None, auth_disabled=False, presence_only=False):
    """Build a minimal FastAPI app guarded by the middleware.

    The JWKS client is monkeypatched to return the test public key so no network
    call is made.
    """
    _, public_key = rsa_keypair
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"status": "healthy"}

    @app.post("/api/book")
    def book():
        return {"booked": True}

    app.add_middleware(
        JWTAuthMiddleware,
        keycloak_url=KEYCLOAK_URL,
        keycloak_external_url=KEYCLOAK_URL,
        realm=REALM,
        audience=audience,
        auth_disabled=auth_disabled,
        presence_only=presence_only,
    )

    if not auth_disabled and not presence_only:

        class _FakeSigningKey:
            key = public_key

        def _fake_get_key(self, _token):
            return _FakeSigningKey()

        monkeypatch.setattr(
            jwt.PyJWKClient,
            "get_signing_key_from_jwt",
            _fake_get_key,
        )

    return app


def test_unauthenticated_request_is_rejected(rsa_keypair, monkeypatch):
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    response = client.post("/api/book")
    assert response.status_code == 401


def test_missing_bearer_prefix_is_rejected(rsa_keypair, monkeypatch):
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    response = client.post("/api/book", headers={"Authorization": "Token abc"})
    assert response.status_code == 401


def test_valid_token_is_accepted(rsa_keypair, monkeypatch):
    private_pem, _ = rsa_keypair
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    token = _mint_token(private_pem)
    response = client.post("/api/book", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"booked": True}


def test_token_with_wrong_issuer_is_rejected(rsa_keypair, monkeypatch):
    private_pem, _ = rsa_keypair
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    token = _mint_token(private_pem, issuer="https://attacker.example.com/realms/evil")
    response = client.post("/api/book", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_expired_token_is_rejected(rsa_keypair, monkeypatch):
    private_pem, _ = rsa_keypair
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    token = _mint_token(private_pem, expired=True)
    response = client.post("/api/book", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_health_path_is_public(rsa_keypair, monkeypatch):
    app = _build_app(rsa_keypair, monkeypatch)
    client = TestClient(app)
    response = client.get("/ping")
    assert response.status_code == 200


def test_audience_is_enforced_when_configured(rsa_keypair, monkeypatch):
    private_pem, _ = rsa_keypair
    app = _build_app(rsa_keypair, monkeypatch, audience="travel-assistant")
    client = TestClient(app)
    # Wrong audience -> reject.
    wrong = _mint_token(private_pem, audience="some-other-agent")
    assert client.post("/api/book", headers={"Authorization": f"Bearer {wrong}"}).status_code == 401
    # Correct audience -> accept.
    right = _mint_token(private_pem, audience="travel-assistant")
    assert client.post("/api/book", headers={"Authorization": f"Bearer {right}"}).status_code == 200


def test_fails_closed_when_jwks_unavailable(rsa_keypair, monkeypatch):
    """If the JWKS client failed to initialize, protected paths are denied (503)."""
    private_pem, _ = rsa_keypair
    app = _build_app(rsa_keypair, monkeypatch)

    # Simulate a JWKS client that could not be built (fail-closed path): the
    # validation call raises AuthConfigurationError, which must produce a 503
    # denial rather than falling open.

    def _raise(self, _token):
        raise AuthConfigurationError("no jwks")

    monkeypatch.setattr(
        jwt.PyJWKClient,
        "get_signing_key_from_jwt",
        _raise,
    )
    client = TestClient(app)
    token = _mint_token(private_pem)
    response = client.post("/api/book", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 503


def test_auth_disabled_bypass_allows_requests(rsa_keypair, monkeypatch):
    """The explicit opt-out bypasses auth (documented, logged sandbox mode)."""
    app = _build_app(rsa_keypair, monkeypatch, auth_disabled=True)
    client = TestClient(app)
    response = client.post("/api/book")
    assert response.status_code == 200


def test_refuses_to_start_when_auth_disabled_and_bound_to_all_interfaces(monkeypatch):
    """Disabling auth while binding to 0.0.0.0 must refuse to start."""
    monkeypatch.setenv("AGENT_AUTH_DISABLED", "true")
    app = FastAPI()
    with pytest.raises(AuthConfigurationError):
        install_agent_auth(
            app,
            keycloak_url=KEYCLOAK_URL,
            keycloak_external_url=KEYCLOAK_URL,
            realm=REALM,
            bind_host="0.0.0.0",  # nosec B104 - test asserts this is rejected
        )


def test_auth_disabled_on_loopback_is_permitted(monkeypatch):
    """Disabling auth on loopback is allowed (trusted local sandbox)."""
    monkeypatch.setenv("AGENT_AUTH_DISABLED", "true")
    app = FastAPI()
    app.add_api_route("/api/book", lambda: {"booked": True}, methods=["POST"])
    install_agent_auth(
        app,
        keycloak_url=KEYCLOAK_URL,
        keycloak_external_url=KEYCLOAK_URL,
        realm=REALM,
        bind_host="127.0.0.1",
    )
    with TestClient(app) as client:
        assert client.post("/api/book").json() == {"booked": True}


def test_presence_only_accepts_any_nonempty_bearer(rsa_keypair, monkeypatch):
    """Presence-only mode accepts an unverified bearer (test/demo passthrough)."""
    app = _build_app(rsa_keypair, monkeypatch, presence_only=True)
    client = TestClient(app)
    response = client.post("/api/book", headers={"Authorization": "Bearer a2a-demo-anything"})
    assert response.status_code == 200


def test_presence_only_still_rejects_missing_bearer(rsa_keypair, monkeypatch):
    """Presence-only still requires a bearer to be present (not fully open)."""
    app = _build_app(rsa_keypair, monkeypatch, presence_only=True)
    client = TestClient(app)
    assert client.post("/api/book").status_code == 401
    assert client.post("/api/book", headers={"Authorization": "Bearer "}).status_code == 401


def test_presence_only_on_all_interfaces_refused_without_optin(monkeypatch):
    """Presence-only is effectively unauthenticated, so binding to 0.0.0.0 is
    refused unless the operator explicitly opts in."""
    monkeypatch.setenv("AGENT_AUTH_PRESENCE_ONLY", "true")
    monkeypatch.delenv("AGENT_AUTH_ALLOW_EXPOSED_PRESENCE_ONLY", raising=False)
    app = FastAPI()
    with pytest.raises(AuthConfigurationError):
        install_agent_auth(
            app,
            keycloak_url=KEYCLOAK_URL,
            keycloak_external_url=KEYCLOAK_URL,
            realm=REALM,
            bind_host="0.0.0.0",  # nosec B104 - test asserts this is rejected
        )


def test_presence_only_on_all_interfaces_permitted_with_optin(monkeypatch):
    """Explicit opt-in allows presence-only on 0.0.0.0 (throwaway gateway test)."""
    monkeypatch.setenv("AGENT_AUTH_PRESENCE_ONLY", "true")
    monkeypatch.setenv("AGENT_AUTH_ALLOW_EXPOSED_PRESENCE_ONLY", "true")
    app = FastAPI()
    app.add_api_route("/api/book", lambda: {"booked": True}, methods=["POST"])
    install_agent_auth(
        app,
        keycloak_url=KEYCLOAK_URL,
        keycloak_external_url=KEYCLOAK_URL,
        realm=REALM,
        bind_host="0.0.0.0",  # nosec B104 - test asserts opt-in permits this
    )
    with TestClient(app) as client:
        assert client.post("/api/book").status_code == 401
        assert client.post(
            "/api/book",
            headers={"Authorization": "Bearer passthrough-test"},
        ).json() == {"booked": True}


def test_presence_only_on_loopback_is_permitted(monkeypatch):
    """Presence-only on loopback needs no opt-in (not network-exposed)."""
    monkeypatch.setenv("AGENT_AUTH_PRESENCE_ONLY", "true")
    monkeypatch.delenv("AGENT_AUTH_ALLOW_EXPOSED_PRESENCE_ONLY", raising=False)
    app = FastAPI()
    app.add_api_route("/api/book", lambda: {"booked": True}, methods=["POST"])
    install_agent_auth(
        app,
        keycloak_url=KEYCLOAK_URL,
        keycloak_external_url=KEYCLOAK_URL,
        realm=REALM,
        bind_host="127.0.0.1",
    )
    with TestClient(app) as client:
        assert client.post("/api/book").status_code == 401
        assert client.post(
            "/api/book",
            headers={"Authorization": "Bearer passthrough-test"},
        ).json() == {"booked": True}


@pytest.fixture(params=["travel-assistant-agent", "flight-booking-agent"])
def auth_module(request):
    return _load_auth_module(request.param)


@pytest.mark.parametrize("external_url", [KEYCLOAK_URL, "", "   "])
def test_public_issuer_with_internal_signing_keys(
    auth_module,
    rsa_keypair,
    monkeypatch,
    external_url,
):
    private_pem, public_key = rsa_keypair
    internal_url = "http://keycloak:8080"
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(public_key, as_dict=True)
    jwk.update({"kid": "sample-key", "use": "sig", "alg": "RS256"})

    def fetch_keys(client):
        assert client.uri == f"{internal_url}/realms/{REALM}/protocol/openid-connect/certs"
        return {"keys": [jwk]}

    monkeypatch.setattr(auth_module.PyJWKClient, "fetch_data", fetch_keys)
    app = FastAPI()

    @app.post("/api/book")
    def book():
        return {"booked": True}

    @app.get("/ping")
    def ping():
        return {"status": "healthy"}

    app.add_middleware(
        auth_module.JWTAuthMiddleware,
        keycloak_url=internal_url,
        keycloak_external_url=external_url,
        realm=REALM,
    )
    with TestClient(app) as client:
        for issuer, valid_status in (
            (ISSUER, 200),
            (f"{internal_url}/realms/{REALM}", 401),
        ):
            token = jwt.encode(
                {"iss": issuer, "exp": int(time.time()) + 300},
                private_pem,
                algorithm="RS256",
                headers={"kid": "sample-key"},
            )
            response = client.post("/api/book", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == (valid_status if external_url.strip() else 503)
            if response.status_code == 200:
                assert response.json() == {"booked": True}
        assert client.get("/ping").status_code == 200
