"""Unit tests for the registry-side registry-UI internal-token verifier.

Covers the verify half of the /validate-minted ``mcp-registry-ui`` token: the
auth-server mints it (see tests/unit/auth/test_internal_request_token.py); the
registry verifies it here and reads identity from the verified claims, ignoring
the forgeable inbound headers.
"""

import os
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from registry.auth.proxied_token import (
    _api_auth_request_enabled,
    verify_registry_ui_token,
)

_SECRET = "test-secret-key-for-testing-only"
_ISSUER = "mcp-auth-server"
_AUDIENCE = "mcp-registry-ui"


def _make_token(
    *,
    secret: str = _SECRET,
    issuer: str = _ISSUER,
    audience: str = _AUDIENCE,
    sub: str = "alice",
    token_use: str = "mcp-registry-ui",
    session_id: str = "sess-1",
    groups: list[str] | None = None,
    auth_method: str = "keycloak",
    client_id: str = "ui",
    iat_offset: int = 0,
    exp_offset: int = 30,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": sub,
        "scopes": [],
        "session_id": session_id,
        "groups": groups or [],
        "auth_method": auth_method,
        "client_id": client_id,
        "token_use": token_use,
        "iat": now + iat_offset,
        "exp": now + exp_offset,
    }
    return pyjwt.encode(claims, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


class TestVerifyRegistryUiToken:
    def test_valid_token_returns_claims(self) -> None:
        token = _make_token(sub="alice", session_id="sess-1", groups=["g1"])
        claims = verify_registry_ui_token(token)
        assert claims["sub"] == "alice"
        assert claims["session_id"] == "sess-1"
        assert claims["groups"] == ["g1"]
        assert claims["auth_method"] == "keycloak"
        assert claims["client_id"] == "ui"

    def test_garbage_token_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token("not-a-jwt")
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self) -> None:
        # exp well in the past, beyond the 5s leeway.
        token = _make_token(iat_offset=-120, exp_offset=-60)
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_future_iat_rejected(self) -> None:
        # iat far in the future, beyond leeway.
        token = _make_token(iat_offset=120, exp_offset=180)
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_audience_rejected(self) -> None:
        # An mcp-proxy token must not verify as registry-ui.
        token = _make_token(audience="mcp-proxy")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_service_audience_rejected(self) -> None:
        # The mcp-registry service-to-service audience must not verify here either.
        token = _make_token(audience="mcp-registry")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_issuer_rejected(self) -> None:
        token = _make_token(issuer="someone-else")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_token_use_rejected(self) -> None:
        token = _make_token(token_use="access")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_tampered_signature_rejected(self) -> None:
        # Signed with a different key.
        token = _make_token(secret="a-different-secret-key-entirely")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_missing_secret_raises_500(self) -> None:
        token = _make_token()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_registry_ui_token(token)
            assert exc.value.status_code == 500


class TestVerifyRegistryUiTokenES256:
    """The registry MUST accept ES256 internal tokens (kid present), verified
    against auth-server's published internal JWKS.

    Regression guard for the login loop: auth-server switched the minter to
    ES256 (Phase A) but the registry verifier was HS256-only, so it rejected
    every real token ("alg not allowed") -> /api/auth/me 401 -> login loop.
    See project_phasea_registry_verifier_gap.
    """

    @staticmethod
    def _es256_keypair():
        from cryptography.hazmat.primitives.asymmetric import ec

        priv = ec.generate_private_key(ec.SECP256R1())
        return priv, priv.public_key()

    @staticmethod
    def _make_es256_token(priv, *, kid="es256-1", audience=_AUDIENCE, token_use="mcp-registry-ui"):
        now = int(time.time())
        claims = {
            "iss": _ISSUER,
            "aud": audience,
            "sub": "alice",
            "session_id": "sess-1",
            "groups": ["g1"],
            "auth_method": "keycloak",
            "client_id": "ui",
            "token_use": token_use,
            "iat": now,
            "exp": now + 30,
        }
        return pyjwt.encode(claims, priv, algorithm="ES256", headers={"kid": kid})

    def test_es256_token_verifies_via_jwks(self) -> None:
        priv, pub = self._es256_keypair()
        token = self._make_es256_token(priv, kid="es256-1")

        # Intent: the kid is looked up in the internal JWKS and the token is
        # verified with the matching public key — NO SECRET_KEY involved.
        with patch(
            "registry.auth.internal_jwks.get_internal_verification_key",
            return_value=pub,
        ) as mock_get:
            claims = verify_registry_ui_token(token)
        assert claims["sub"] == "alice"
        assert claims["groups"] == ["g1"]
        mock_get.assert_called_once_with("es256-1")

    def test_es256_unknown_kid_rejected(self) -> None:
        priv, _pub = self._es256_keypair()
        token = self._make_es256_token(priv, kid="es256-rotated-away")

        # JWKS has no such kid (fetch returns None) -> fail closed.
        with patch(
            "registry.auth.internal_jwks.get_internal_verification_key",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc:
                verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_es256_wrong_key_rejected(self) -> None:
        priv, _pub = self._es256_keypair()
        _other_priv, other_pub = self._es256_keypair()
        token = self._make_es256_token(priv, kid="es256-1")

        # JWKS returns a DIFFERENT public key for the kid -> signature fails.
        with patch(
            "registry.auth.internal_jwks.get_internal_verification_key",
            return_value=other_pub,
        ):
            with pytest.raises(HTTPException) as exc:
                verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_es256_does_not_need_secret_key(self) -> None:
        # A pure-ES256 deployment may not set SECRET_KEY at all; verification
        # must still work (the whole point of isolating minting to a keypair).
        priv, pub = self._es256_keypair()
        token = self._make_es256_token(priv, kid="es256-1")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "registry.auth.internal_jwks.get_internal_verification_key",
                return_value=pub,
            ),
        ):
            claims = verify_registry_ui_token(token)
        assert claims["sub"] == "alice"

    def test_hs256_rejected_when_cutover_flag_set(self) -> None:
        # After the cutover, a legacy HS256 (no-kid) token is hard-rejected.
        token = _make_token()  # HS256, no kid
        with patch.dict(os.environ, {"REJECT_HS256_TOKENS": "true"}, clear=False):
            with pytest.raises(HTTPException) as exc:
                verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_oversized_token_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token("x" * 9000)
        assert exc.value.status_code == 401


class TestApiAuthRequestEnabled:
    def test_default_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _api_auth_request_enabled() is True

    @pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On"])
    def test_disabled_values(self, val: str) -> None:
        with patch.dict(os.environ, {"NGINX_DISABLE_API_AUTH_REQUEST": val}, clear=False):
            assert _api_auth_request_enabled() is False

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", ""])
    def test_enabled_values(self, val: str) -> None:
        with patch.dict(os.environ, {"NGINX_DISABLE_API_AUTH_REQUEST": val}, clear=False):
            assert _api_auth_request_enabled() is True
