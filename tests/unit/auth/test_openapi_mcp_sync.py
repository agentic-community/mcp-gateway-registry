"""Tests for the registry-to-openapi-mcp sync token."""

import time

import jwt as pyjwt
import pytest

from registry.auth.openapi_mcp_sync import (
    SYNC_AUDIENCE,
    SYNC_ISSUER,
    SYNC_TOKEN_KIND,
    JtiCache,
    SyncTokenError,
    _derive_signing_key,
    mint_sync_token,
    verify_sync_token,
)

SECRET: str = "test-sync-secret-0123456789abcdef-0123456789"
OTHER_SECRET: str = "other-sync-secret-0123456789abcdef-012345678"


def _encode(
    claims: dict,
    secret: str = SECRET,
) -> str:
    return pyjwt.encode(claims, _derive_signing_key(secret), algorithm="HS256")


def _valid_claims() -> dict:
    now = int(time.time())
    return {
        "iss": SYNC_ISSUER,
        "aud": SYNC_AUDIENCE,
        "sub": "registry-service",
        "token_kind": SYNC_TOKEN_KIND,
        "jti": "abc123",
        "iat": now,
        "exp": now + 60,
    }


class TestSyncToken:
    """Mint and verify behavior."""

    def test_round_trip_returns_claims(self):
        token = mint_sync_token(SECRET, "put_server")

        claims = verify_sync_token(token, SECRET, JtiCache())

        assert claims["purpose"] == "put_server"

    def test_reused_jti_rejected(self):
        token = mint_sync_token(SECRET, "put_server")
        cache = JtiCache()
        verify_sync_token(token, SECRET, cache)

        with pytest.raises(SyncTokenError, match="already used"):
            verify_sync_token(token, SECRET, cache)

    def test_wrong_secret_rejected(self):
        token = mint_sync_token(OTHER_SECRET, "put_server")

        with pytest.raises(SyncTokenError, match="InvalidSignatureError"):
            verify_sync_token(token, SECRET, JtiCache())

    def test_raw_secret_signature_rejected(self):
        """A token signed with the underived secret must not verify."""
        token = pyjwt.encode(_valid_claims(), SECRET, algorithm="HS256")

        with pytest.raises(SyncTokenError):
            verify_sync_token(token, SECRET, JtiCache())

    def test_expired_rejected(self):
        claims = _valid_claims()
        claims["iat"] -= 200
        claims["exp"] -= 200

        with pytest.raises(SyncTokenError, match="ExpiredSignatureError"):
            verify_sync_token(_encode(claims), SECRET, JtiCache())

    def test_wrong_audience_rejected(self):
        claims = _valid_claims()
        claims["aud"] = "mcp-internal"

        with pytest.raises(SyncTokenError, match="InvalidAudienceError"):
            verify_sync_token(_encode(claims), SECRET, JtiCache())

    def test_wrong_token_kind_rejected(self):
        claims = _valid_claims()
        claims["token_kind"] = "internal-service"

        with pytest.raises(SyncTokenError, match="token_kind"):
            verify_sync_token(_encode(claims), SECRET, JtiCache())

    def test_long_lifetime_rejected(self):
        claims = _valid_claims()
        claims["exp"] = claims["iat"] + 3600

        with pytest.raises(SyncTokenError, match="lifetime"):
            verify_sync_token(_encode(claims), SECRET, JtiCache())

    def test_missing_jti_rejected(self):
        claims = _valid_claims()
        del claims["jti"]

        with pytest.raises(SyncTokenError, match="MissingRequiredClaimError"):
            verify_sync_token(_encode(claims), SECRET, JtiCache())

    def test_none_algorithm_rejected(self):
        token = pyjwt.encode(_valid_claims(), key=None, algorithm="none")

        with pytest.raises(SyncTokenError):
            verify_sync_token(token, SECRET, JtiCache())

    def test_empty_secret_fails_closed_on_mint(self):
        with pytest.raises(SyncTokenError, match="not set"):
            mint_sync_token("  ", "put_server")

    def test_empty_secret_fails_closed_on_verify(self):
        token = mint_sync_token(SECRET, "put_server")

        with pytest.raises(SyncTokenError, match="not set"):
            verify_sync_token(token, "", JtiCache())

    def test_empty_token_rejected(self):
        with pytest.raises(SyncTokenError, match="missing"):
            verify_sync_token("", SECRET, JtiCache())


class TestJtiCache:
    """Expiry of consumed jtis."""

    def test_expired_entries_pruned(self):
        cache = JtiCache()
        cache.consume("old", time.time() - 1)

        assert cache.consume("new", time.time() + 60)
        assert len(cache) == 1


class TestCredentialEnvelope:
    """The per-user credential envelope the vend mints and the container verifies."""

    CREDS: dict = {
        "petstore": {"key": "user-key-1", "base_url": "https://petstore.example.com/api"}
    }

    def test_round_trip(self):
        from registry.auth.openapi_mcp_sync import (
            mint_credential_envelope,
            verify_credential_envelope,
        )

        token = mint_credential_envelope(SECRET, "weather-apis", self.CREDS)

        assert verify_credential_envelope(token, SECRET, "weather-apis", JtiCache()) == self.CREDS

    def test_empty_credentials_allowed(self):
        from registry.auth.openapi_mcp_sync import (
            mint_credential_envelope,
            verify_credential_envelope,
        )

        token = mint_credential_envelope(SECRET, "weather-apis", {})

        assert verify_credential_envelope(token, SECRET, "weather-apis", JtiCache()) == {}

    def test_other_server_rejected(self):
        from registry.auth.openapi_mcp_sync import (
            mint_credential_envelope,
            verify_credential_envelope,
        )

        token = mint_credential_envelope(SECRET, "weather-apis", self.CREDS)

        with pytest.raises(SyncTokenError, match="another server"):
            verify_credential_envelope(token, SECRET, "other", JtiCache())

    def test_replay_rejected(self):
        from registry.auth.openapi_mcp_sync import (
            mint_credential_envelope,
            verify_credential_envelope,
        )

        token = mint_credential_envelope(SECRET, "weather-apis", self.CREDS)
        cache = JtiCache()
        verify_credential_envelope(token, SECRET, "weather-apis", cache)

        with pytest.raises(SyncTokenError, match="already used"):
            verify_credential_envelope(token, SECRET, "weather-apis", cache)

    def test_sync_token_is_not_an_envelope(self):
        from registry.auth.openapi_mcp_sync import verify_credential_envelope

        token = mint_sync_token(SECRET, "put_server")

        with pytest.raises(SyncTokenError):
            verify_credential_envelope(token, SECRET, "weather-apis", JtiCache())

    def test_wrong_secret_rejected(self):
        from registry.auth.openapi_mcp_sync import (
            mint_credential_envelope,
            verify_credential_envelope,
        )

        token = mint_credential_envelope(OTHER_SECRET, "weather-apis", self.CREDS)

        with pytest.raises(SyncTokenError, match="InvalidSignatureError"):
            verify_credential_envelope(token, SECRET, "weather-apis", JtiCache())

    def test_malformed_entry_rejected(self):
        from registry.auth.openapi_mcp_sync import (
            ENVELOPE_AUDIENCE,
            ENVELOPE_TOKEN_KIND,
            _envelope_key,
            verify_credential_envelope,
        )

        now = int(time.time())
        claims = {
            "iss": SYNC_ISSUER,
            "aud": ENVELOPE_AUDIENCE,
            "token_kind": ENVELOPE_TOKEN_KIND,
            "server": "weather-apis",
            "credentials": {"petstore": {"key": 5}},
            "jti": "x",
            "iat": now,
            "exp": now + 60,
        }
        token = pyjwt.encode(claims, _envelope_key(SECRET), algorithm="HS256")

        with pytest.raises(SyncTokenError, match="key and base_url"):
            verify_credential_envelope(token, SECRET, "weather-apis", JtiCache())
