"""Unit tests for the registry-side internal JWKS fetcher.

The registry fetches auth-server's ES256 public keys to verify internal hop
tokens. This asserts the intent: fetch once, cache within TTL, serve
last-known-good on failure, pick up a rotated kid, and fail closed when no key
is obtainable. See project_phasea_registry_verifier_gap.
"""

import base64
import importlib
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives.asymmetric import ec


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwk_for(kid: str) -> tuple[dict, ec.EllipticCurvePublicKey]:
    priv = ec.generate_private_key(ec.SECP256R1())
    nums = priv.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64u(nums.x.to_bytes(32, "big")),
        "y": _b64u(nums.y.to_bytes(32, "big")),
        "kid": kid,
        "use": "sig",
        "alg": "ES256",
    }
    return jwk, priv.public_key()


def _fresh_module():
    """Reimport the module so its process-global cache starts empty per test."""
    import registry.auth.internal_jwks as mod

    return importlib.reload(mod)


def _mock_client(jwks: dict) -> MagicMock:
    """A context-manager httpx.Client stub whose .get() returns the given JWKS."""
    resp = MagicMock()
    resp.json.return_value = jwks
    resp.raise_for_status.return_value = None
    client = MagicMock()
    client.get.return_value = resp
    cm = MagicMock()
    cm.__enter__.return_value = client
    cm.__exit__.return_value = False
    return cm, client


class TestInternalJwksFetcher:
    def test_returns_key_for_known_kid(self) -> None:
        mod = _fresh_module()
        jwk, _pub = _jwk_for("es256-1")
        cm, _client = _mock_client({"keys": [jwk]})
        with patch.object(mod.httpx, "Client", return_value=cm):
            key = mod.get_internal_verification_key("es256-1")
        assert key is not None

    def test_unknown_kid_returns_none(self) -> None:
        mod = _fresh_module()
        jwk, _pub = _jwk_for("es256-1")
        cm, _client = _mock_client({"keys": [jwk]})
        with patch.object(mod.httpx, "Client", return_value=cm):
            assert mod.get_internal_verification_key("nope") is None

    def test_cache_hit_avoids_second_fetch(self) -> None:
        mod = _fresh_module()
        jwk, _pub = _jwk_for("es256-1")
        cm, client = _mock_client({"keys": [jwk]})
        with patch.object(mod.httpx, "Client", return_value=cm):
            mod.get_internal_verification_key("es256-1")
            mod.get_internal_verification_key("es256-1")
        # Second call served from cache within TTL -> only one network fetch.
        assert client.get.call_count == 1

    def test_last_known_good_served_on_fetch_failure(self) -> None:
        mod = _fresh_module()
        jwk, _pub = _jwk_for("es256-1")
        cm_ok, _client = _mock_client({"keys": [jwk]})
        # Prime the cache.
        with patch.object(mod.httpx, "Client", return_value=cm_ok):
            assert mod.get_internal_verification_key("es256-1") is not None
        # Now force TTL expiry and make the fetch raise -> serve cached.
        with (
            patch.object(mod.settings, "internal_jwks_cache_ttl_seconds", 0),
            patch.object(mod.httpx, "Client", side_effect=RuntimeError("network down")),
        ):
            key = mod.get_internal_verification_key("es256-1")
        assert key is not None

    def test_rotated_kid_picked_up_after_refresh(self) -> None:
        mod = _fresh_module()
        jwk1, _ = _jwk_for("es256-1")
        jwk2, _ = _jwk_for("es256-2")
        # First fetch has only kid es256-1.
        cm1, _c1 = _mock_client({"keys": [jwk1]})
        with patch.object(mod.httpx, "Client", return_value=cm1):
            assert mod.get_internal_verification_key("es256-1") is not None
        # A request for the rotated kid forces a refresh; now JWKS has both.
        cm2, _c2 = _mock_client({"keys": [jwk1, jwk2]})
        with (
            patch.object(mod.settings, "internal_jwks_cache_ttl_seconds", 0),
            patch.object(mod.httpx, "Client", return_value=cm2),
        ):
            assert mod.get_internal_verification_key("es256-2") is not None

    def test_empty_jwks_does_not_clobber_good_cache(self) -> None:
        mod = _fresh_module()
        jwk, _pub = _jwk_for("es256-1")
        cm_ok, _c = _mock_client({"keys": [jwk]})
        with patch.object(mod.httpx, "Client", return_value=cm_ok):
            assert mod.get_internal_verification_key("es256-1") is not None
        # A later fetch returns an empty JWKS (rotation slip) -> keep old key.
        cm_empty, _c2 = _mock_client({"keys": []})
        with (
            patch.object(mod.settings, "internal_jwks_cache_ttl_seconds", 0),
            patch.object(mod.httpx, "Client", return_value=cm_empty),
        ):
            assert mod.get_internal_verification_key("es256-1") is not None
