"""Unit tests for the shared, hardened JWKS cache (auth_server.providers.jwks_cache)."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from auth_server.providers import jwks_cache
from auth_server.providers.jwks_cache import JwksCache

JWKS_URL = "https://idp.example.com/.well-known/jwks.json"
KNOWN_KID = "known-kid-1"


def _rsa_jwk(kid: str) -> dict:
    """A real RSA public JWK so PyJWK can materialise a signing key."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return jwk


@pytest.fixture
def jwks_doc():
    return {"keys": [_rsa_jwk(KNOWN_KID)]}


@pytest.fixture
def mock_response(jwks_doc):
    resp = MagicMock()
    resp.json.return_value = jwks_doc
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # Keep the failure-path retry loop from sleeping in tests.
    monkeypatch.setattr(jwks_cache, "_RETRY_SLEEP_SECONDS", 0)


class TestHealthyKnownKidPath:
    """The known-kid path must behave exactly like the old per-provider code."""

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_fetches_once_then_caches(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        cache = JwksCache(provider_name="test")

        key1 = cache.get_signing_key(JWKS_URL, KNOWN_KID)
        key2 = cache.get_signing_key(JWKS_URL, KNOWN_KID)

        assert key1 is not None
        assert key2 is not None
        # Warm cache within TTL => exactly one upstream fetch for both lookups.
        assert mock_get.call_count == 1

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_get_jwks_returns_document(self, mock_get, mock_response, jwks_doc):
        mock_get.return_value = mock_response
        cache = JwksCache(provider_name="test")

        assert cache.get_jwks(JWKS_URL) == jwks_doc
        assert cache.get_jwks(JWKS_URL) == jwks_doc
        assert mock_get.call_count == 1


class TestUnknownKidHardening:
    """Unknown kid => single coalesced refetch, then negative-cached."""

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_single_coalesced_refetch_across_concurrent_callers(self, mock_get, mock_response):
        # A refetch is deliberately slow so all N threads pile onto the lock.
        def _slow_get(*_args, **_kwargs):
            time.sleep(0.05)
            return mock_response

        mock_get.side_effect = _slow_get
        cache = JwksCache(provider_name="test")

        # Warm the cache with a known-kid lookup, then zero the counter so we
        # only measure refetches provoked by the unknown kid.
        cache.get_signing_key(JWKS_URL, KNOWN_KID)
        mock_get.reset_mock()
        mock_get.side_effect = _slow_get

        n = 12
        barrier = threading.Barrier(n)
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def worker():
            barrier.wait()
            try:
                cache.get_signing_key(JWKS_URL, "totally-unknown-kid")
            except ValueError as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every caller was rejected...
        assert len(errors) == n
        assert all("totally-unknown-kid" in str(e) for e in errors)
        # ...but they COALESCED into exactly one upstream refetch.
        assert mock_get.call_count == 1

        # The unknown kid is now negative-cached: another lookup rejects
        # WITHOUT any further upstream traffic.
        with pytest.raises(ValueError):
            cache.get_signing_key(JWKS_URL, "totally-unknown-kid")
        assert mock_get.call_count == 1

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_unknown_kid_resolved_after_key_rotation(self, mock_get, jwks_doc):
        # First JWKS lacks the rotated kid; the one refetch returns it.
        rotated_kid = "rotated-kid-2"
        rotated_doc = {"keys": [jwks_doc["keys"][0], _rsa_jwk(rotated_kid)]}

        first = MagicMock()
        first.json.return_value = jwks_doc
        first.raise_for_status.return_value = None
        second = MagicMock()
        second.json.return_value = rotated_doc
        second.raise_for_status.return_value = None
        mock_get.side_effect = [first, second]

        cache = JwksCache(provider_name="test")
        cache.get_signing_key(JWKS_URL, KNOWN_KID)  # warm (fetch #1)

        key = cache.get_signing_key(JWKS_URL, rotated_kid)  # unknown => refetch (#2)
        assert key is not None
        assert mock_get.call_count == 2
        # A freshly-fetched real key must NOT be negative-cached.
        assert rotated_kid not in cache._entry(JWKS_URL).unknown_kids


class TestStaleFallback:
    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_serves_stale_within_window_on_fetch_failure(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        cache = JwksCache(provider_name="test")

        first = cache.get_jwks(JWKS_URL)  # populate

        # Expire the TTL and make every subsequent fetch fail.
        cache._entry(JWKS_URL).fetched_at -= jwks_cache.DEFAULT_TTL_SECONDS + 1
        mock_get.side_effect = Exception("network down")
        mock_get.return_value = None

        served = cache.get_jwks(JWKS_URL)
        assert served == first  # stale-but-good served within the bounded window

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_raises_when_no_cache_and_fetch_fails(self, mock_get):
        mock_get.side_effect = Exception("network down")
        cache = JwksCache(provider_name="test")

        with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
            cache.get_jwks(JWKS_URL)

    @patch("auth_server.providers.jwks_cache.requests.get")
    def test_stale_beyond_window_raises(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        cache = JwksCache(provider_name="test")
        cache.get_jwks(JWKS_URL)

        # Age past the bounded staleness window.
        cache._entry(JWKS_URL).fetched_at -= jwks_cache.DEFAULT_STALE_TTL_SECONDS + 1
        mock_get.side_effect = Exception("network down")
        mock_get.return_value = None

        with pytest.raises(ValueError, match="Cannot retrieve JWKS"):
            cache.get_jwks(JWKS_URL)
