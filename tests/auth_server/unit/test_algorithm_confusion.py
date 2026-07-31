"""Algorithm-confusion test suite for self-signed token verification.

Tests the security INTENT of the dual-verify dispatch: an attacker who
knows the public key (it's published in the JWKS) must not be able to
trick the verifier into accepting a forged token by exploiting algorithm
confusion between HS256 and ES256.

These tests validate the system boundary (verify_self_signed_user_token)
against real attack scenarios, not just that PyJWT's internals work.
"""

import os
import time

import jwt as pyjwt
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-algorithm-confusion-tests!")
os.environ.setdefault("INTERNAL_SIGNING_KEY_GENERATE", "true")

from auth_server.self_signed_token import (
    JWT_AUDIENCE,
    JWT_ISSUER,
    verify_self_signed_user_token,
)


@pytest.fixture(autouse=True)
def _reset_key_manager():
    """Reset the key manager singleton so each test class gets a fresh instance.

    Must reset both possible module paths (auth_server.internal_signing_key
    and internal_signing_key) to ensure the verify function and the test
    use the same singleton.
    """
    import sys

    os.environ["INTERNAL_SIGNING_KEY_GENERATE"] = "true"

    # Reset the singleton in whichever module path is loaded
    for mod_name in ["auth_server.internal_signing_key", "internal_signing_key"]:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "_key_manager"):
            mod._key_manager = None

    yield

    for mod_name in ["auth_server.internal_signing_key", "internal_signing_key"]:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "_key_manager"):
            mod._key_manager = None


def _valid_claims(**overrides) -> dict:
    """Build a valid claim set for testing."""
    base = {
        "sub": "testuser@example.com",
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "token_use": "access",
        "scope": "openid",
        "groups": ["test-group"],
        "client_id": "user-generated",
    }
    base.update(overrides)
    return base


def _get_key_manager():
    """Get the key manager via the same import path verify_self_signed_user_token uses."""
    try:
        from auth_server.internal_signing_key import get_internal_signing_key_manager
    except ImportError:
        from internal_signing_key import get_internal_signing_key_manager
    return get_internal_signing_key_manager()


class TestAlgorithmConfusionAttacks:
    """An attacker who has the ES256 public key (from JWKS) must not be
    able to forge tokens by using the public key as an HS256 secret."""

    def test_public_key_used_as_hmac_secret_is_rejected(self):
        """Scenario: Classic RS256/ES256→HS256 confusion attack.

        The attacker obtains the public key from /.well-known/internal-jwks.json,
        then signs a token using that public key bytes as the HS256 HMAC secret.
        We construct this manually because PyJWT 2+ blocks it at encode time.

        Expected: Rejected. The HS256 path uses SECRET_KEY (not the public key),
        so the attacker's HMAC won't match."""
        import base64
        import hashlib
        import hmac
        import json

        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        key_manager = _get_key_manager()
        pub_bytes = (
            key_manager.get_signing_key()
            .public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        )

        # Manually construct HS256 token signed with public key bytes (no kid)
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(json.dumps(_valid_claims()).encode()).rstrip(b"=")
        signing_input = header + b"." + payload
        sig = base64.urlsafe_b64encode(
            hmac.new(pub_bytes, signing_input, hashlib.sha256).digest()
        ).rstrip(b"=")
        forged_token = f"{header.decode()}.{payload.decode()}.{sig.decode()}"

        with pytest.raises(ValueError, match="Invalid self-signed token"):
            verify_self_signed_user_token(forged_token)

    def test_public_key_as_hmac_with_kid_also_rejected(self):
        """Scenario: Attacker includes a kid hoping to route to ES256 path,
        but signs with HS256 using the public key as secret.

        Expected: Rejected. ES256 path uses algorithms=["ES256"] only."""
        import base64
        import hashlib
        import hmac
        import json

        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        key_manager = _get_key_manager()
        kid = key_manager.get_signing_kid()
        pub_bytes = (
            key_manager.get_signing_key()
            .public_key()
            .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        )

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT", "kid": kid}).encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(json.dumps(_valid_claims()).encode()).rstrip(b"=")
        signing_input = header + b"." + payload
        sig = base64.urlsafe_b64encode(
            hmac.new(pub_bytes, signing_input, hashlib.sha256).digest()
        ).rstrip(b"=")
        forged_token = f"{header.decode()}.{payload.decode()}.{sig.decode()}"

        with pytest.raises(ValueError, match="Invalid self-signed token"):
            verify_self_signed_user_token(forged_token)


class TestExternalIdpTokenConfusion:
    """A user with a valid external IdP token (e.g., Keycloak RS256) must
    not be able to present it as an internal self-signed token."""

    def test_external_idp_token_claiming_our_issuer_rejected(self):
        """Scenario: Attacker obtains a valid token from an external IdP
        but crafts it to claim iss='mcp-auth-server'. If the verifier
        doesn't check issuer BEFORE key selection, it might try to verify
        with the wrong key and leak timing information or pass.

        Expected: Rejected at issuer check. Our verify function checks iss
        in the unverified payload before attempting signature verification."""
        # Generate a random RS256 key (simulating an external IdP)
        from cryptography.hazmat.primitives.asymmetric import rsa

        external_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Token signed by external IdP but claiming our issuer
        external_token = pyjwt.encode(
            _valid_claims(iss=JWT_ISSUER),
            external_key,
            algorithm="RS256",
            headers={"kid": "external-kid-123"},
        )

        # Should be rejected — kid not found in our key manager
        with pytest.raises(ValueError):
            verify_self_signed_user_token(external_token)

    def test_token_with_wrong_issuer_rejected_before_signature_check(self):
        """Scenario: Token has a valid structure but claims a different issuer
        (e.g., 'https://login.microsoftonline.com/tenant/v2.0').

        Expected: Rejected at the issuer-first check — we never even attempt
        signature verification. This prevents timing attacks that could reveal
        whether a key exists for a given kid."""
        external_token = pyjwt.encode(
            _valid_claims(iss="https://login.microsoftonline.com/tenant-id/v2.0"),
            "doesnt-matter-what-key",
            algorithm="HS256",
        )

        with pytest.raises(ValueError, match="issuer mismatch"):
            verify_self_signed_user_token(external_token)


class TestMalformedTokenHandling:
    """Malformed or oversized tokens must be hard-rejected without
    falling through to any verification path."""

    def test_oversized_token_rejected_before_parsing(self):
        """Scenario: Attacker sends a 1MB token to exhaust memory or
        trigger pathological parsing behavior.

        Expected: Rejected immediately on size check, before any JWT parsing."""
        huge_token = "a" * 10000  # > 8KB limit

        with pytest.raises(ValueError, match="exceeds maximum size"):
            verify_self_signed_user_token(huge_token)

    def test_non_jwt_garbage_rejected(self):
        """Scenario: Attacker sends random bytes that aren't a JWT.

        Expected: Hard reject on header parse failure. Must NOT fall through
        to HS256 (which would be a downgrade path)."""
        with pytest.raises(ValueError, match="[Mm]alformed|[Cc]annot parse"):
            verify_self_signed_user_token("not.a.valid.jwt.at.all")

    def test_empty_token_rejected(self):
        """Scenario: Empty/missing token reaches the verify function.

        Expected: Rejected (malformed header or size check)."""
        with pytest.raises(ValueError):
            verify_self_signed_user_token("")

    def test_token_with_none_algorithm_rejected(self):
        """Scenario: Classic 'alg:none' attack — token with no signature.

        Expected: Rejected. PyJWT rejects alg:none by default, but our
        dispatch adds defense-in-depth: even if PyJWT had a bug, neither
        path accepts 'none' in its algorithms list."""
        # Manually construct an alg:none token
        import base64
        import json

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(json.dumps(_valid_claims()).encode()).rstrip(b"=")
        none_token = f"{header.decode()}.{payload.decode()}."

        with pytest.raises(ValueError):
            verify_self_signed_user_token(none_token)


class TestKidDispatchRouting:
    """Verify that the kid-based dispatch correctly routes tokens to
    the right verification path."""

    def test_no_kid_routes_to_hs256(self):
        """A token without kid verifies via SECRET_KEY (HS256 legacy)."""
        secret = os.environ["SECRET_KEY"]
        token = pyjwt.encode(_valid_claims(), secret, algorithm="HS256")

        result = verify_self_signed_user_token(token)
        assert result["valid"] is True
        assert result["username"] == "testuser@example.com"

    def test_valid_kid_routes_to_es256(self):
        """A token with a known kid verifies via the key manager (ES256)."""

        key_manager = _get_key_manager()
        private_key = key_manager.get_signing_key()
        kid = key_manager.get_signing_kid()

        token = pyjwt.encode(
            _valid_claims(),
            private_key,
            algorithm="ES256",
            headers={"kid": kid},
        )

        result = verify_self_signed_user_token(token)
        assert result["valid"] is True
        assert result["username"] == "testuser@example.com"

    def test_unknown_kid_rejected(self):
        """A token with a kid not in our key manager is rejected.

        This prevents tokens signed by a revoked or rotated-out key from
        being accepted after the retention window expires."""

        key_manager = _get_key_manager()
        private_key = key_manager.get_signing_key()

        # Sign with our key but claim a different kid
        token = pyjwt.encode(
            _valid_claims(),
            private_key,
            algorithm="ES256",
            headers={"kid": "revoked-old-kid-99"},
        )

        with pytest.raises(ValueError, match="Unknown key id"):
            verify_self_signed_user_token(token)

    def test_wrong_token_use_rejected_regardless_of_algorithm(self):
        """A token with token_use != 'access' is rejected even if the
        signature is valid. Prevents id_tokens or refresh_tokens from
        being accepted as access tokens."""
        secret = os.environ["SECRET_KEY"]
        token = pyjwt.encode(
            _valid_claims(token_use="id_token"),
            secret,
            algorithm="HS256",
        )

        with pytest.raises(ValueError, match="Invalid token_use"):
            verify_self_signed_user_token(token)
