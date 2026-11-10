"""Tests for the JWKS cache shared across provider instances (issue #2012).

``get_auth_provider()`` builds a new provider per request, so the JWKS cache
must live on the class, keyed by JWKS URL, or every token validation refetches
the keys. A token whose ``kid`` is not cached triggers one rate-limited refetch
so an IdP key rotation is picked up without waiting for the TTL.
"""

from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from auth_server.providers import base
from auth_server.providers.entra import EntraIdProvider
from auth_server.providers.factory import get_auth_provider
from auth_server.providers.pingfederate import PingFederateProvider

REQUESTS_GET = "auth_server.providers.entra.requests.get"


def _public_jwk(kid: str) -> dict:
    """Build an RSA public JWK with the given kid."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk["kid"] = kid
    return jwk


def _jwks_response(*kids: str) -> MagicMock:
    """Build a requests response stub carrying a JWKS with the given kids."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"keys": [_public_jwk(kid) for kid in kids]}
    return response


def _entra(tenant_id: str = "tenant-a") -> EntraIdProvider:
    return EntraIdProvider(tenant_id=tenant_id, client_id="client", client_secret="secret")


@pytest.fixture
def entra_env(monkeypatch):
    monkeypatch.setenv("AUTH_PROVIDER", "entra")
    monkeypatch.setenv("ENTRA_TENANT_ID", "tenant-a")
    monkeypatch.setenv("ENTRA_CLIENT_ID", "client")
    monkeypatch.setenv("ENTRA_CLIENT_SECRET", "secret")


class TestSharedJwksCache:
    """The JWKS is cached once per URL, not once per provider instance."""

    def test_factory_built_providers_share_one_jwks_fetch(self, entra_env):
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")) as mock_get:
            get_auth_provider().get_jwks()
            get_auth_provider().get_jwks()

        assert mock_get.call_count == 1

    def test_different_tenants_get_separate_entries(self):
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")) as mock_get:
            _entra("tenant-a").get_jwks()
            _entra("tenant-b").get_jwks()

        assert mock_get.call_count == 2

    def test_new_instance_sees_cached_key_without_fetch(self):
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")):
            _entra().get_jwks()

        with patch(REQUESTS_GET) as mock_get:
            key = _entra()._find_jwks_signing_key("k1")

        assert key is not None
        mock_get.assert_not_called()

    def test_pingfederate_cache_read_skips_discovery(self):
        provider = PingFederateProvider(
            base_url="https://ping.example.com",
            client_id="client",
            client_secret="secret",
        )

        with patch.object(provider, "_get_openid_configuration") as mock_discovery:
            provider._jwks_cache = {"keys": []}
            assert provider._jwks_cache == {"keys": []}

        mock_discovery.assert_not_called()


class TestKidMissRefresh:
    """An unknown kid refetches the JWKS once, rate-limited per URL."""

    def test_rotated_key_is_found_after_one_refetch(self):
        with patch(REQUESTS_GET, return_value=_jwks_response("old")):
            _entra().get_jwks()

        with patch(REQUESTS_GET, return_value=_jwks_response("old", "new")) as mock_get:
            key = _entra()._find_jwks_signing_key("new")

        assert key is not None
        assert mock_get.call_count == 1

    def test_second_miss_within_window_does_not_refetch(self):
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")):
            _entra().get_jwks()
            _entra()._find_jwks_signing_key("bogus-1")

        with patch(REQUESTS_GET, return_value=_jwks_response("k1")) as mock_get:
            key = _entra()._find_jwks_signing_key("bogus-2")

        assert key is None
        mock_get.assert_not_called()

    def test_miss_refetches_again_after_window(self):
        start = 1_000_000.0
        later = start + base.JWKS_KID_MISS_REFRESH_INTERVAL_SECONDS + 1
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")):
            with patch("time.time", return_value=start):
                _entra().get_jwks()
                _entra()._find_jwks_signing_key("bogus-1")

        with patch(REQUESTS_GET, return_value=_jwks_response("k1")) as mock_get:
            with patch("time.time", return_value=later):
                _entra()._find_jwks_signing_key("bogus-2")

        assert mock_get.call_count == 1

    def test_failed_refetch_keeps_cache_fresh(self):
        with patch(REQUESTS_GET, return_value=_jwks_response("k1")):
            _entra().get_jwks()

        with patch(REQUESTS_GET, side_effect=ConnectionError("down")):
            missing = _entra()._find_jwks_signing_key("bogus")

        with patch(REQUESTS_GET) as mock_get:
            found = _entra()._find_jwks_signing_key("k1")

        assert (missing, found is not None) == (None, True)
        mock_get.assert_not_called()

    def test_validate_token_rejects_unknown_kid(self):
        token = jwt.encode({"sub": "u"}, "x" * 32, algorithm="HS256", headers={"kid": "bogus"})

        with patch(REQUESTS_GET, return_value=_jwks_response("k1")):
            with pytest.raises(ValueError, match="No matching key found for kid"):
                _entra().validate_token(token)
