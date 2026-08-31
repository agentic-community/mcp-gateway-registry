"""Rotation and kid-assignment tests for InternalSigningKeyManager.

These exercise the key manager directly (not through the token verifier) so a
regression in kid derivation or rotation/retention is caught at the source. The
kid is the RFC 7638 JWK thumbprint of the public key: stable per key, identical
across processes/replicas, and never reused for a different key — which is what
makes online rotation and multi-replica overlap safe.
"""

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import auth_server.internal_signing_key as isk
from auth_server.internal_signing_key import (
    InternalSigningKeyManager,
    _jwk_thumbprint,
)


def _write_p256_key(path) -> ec.EllipticCurvePrivateKey:
    """Generate a fresh P-256 key, write it PEM-encoded to ``path``, return it."""
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(path, "wb") as f:
        f.write(pem)
    return key


@pytest.fixture
def key_path(tmp_path, monkeypatch):
    """A manager configured to load from a temp PEM file (no auto-generate)."""
    monkeypatch.delenv("INTERNAL_SIGNING_KEY_GENERATE", raising=False)
    path = tmp_path / "key.pem"
    _write_p256_key(path)
    monkeypatch.setenv("INTERNAL_SIGNING_KEY_PATH", str(path))
    return str(path)


class TestKidDerivation:
    def test_kid_is_rfc7638_thumbprint(self, key_path):
        mgr = InternalSigningKeyManager()
        material = mgr.get_signing_material()
        assert material is not None
        private_key, kid = material
        assert kid == _jwk_thumbprint(private_key.public_key())
        assert kid.startswith("es256-")

    def test_kid_stable_across_replicas(self, key_path):
        """Two independent managers over the same file derive the same kid."""
        a = InternalSigningKeyManager()
        b = InternalSigningKeyManager()
        assert a.get_signing_kid() == b.get_signing_kid()

    def test_reloading_same_key_is_noop(self, key_path):
        mgr = InternalSigningKeyManager()
        kid_before = mgr.get_signing_kid()
        mgr._load_key_from_file(key_path)  # same bytes → must not add a key
        assert len(mgr._keys) == 1
        assert mgr.get_signing_kid() == kid_before


class TestRotation:
    def test_rotation_assigns_distinct_kid_and_retains_old(self, key_path):
        mgr = InternalSigningKeyManager()
        old_kid = mgr.get_signing_kid()

        new_key = _write_p256_key(key_path)  # rotate the file in place
        mgr._load_key_from_file(key_path)

        new_kid = _jwk_thumbprint(new_key.public_key())
        # Signing uses the newest key; both keys remain verifiable (overlap).
        assert mgr.get_signing_kid() == new_kid
        assert new_kid != old_kid
        verification = mgr.get_verification_keys()
        assert set(verification) == {old_kid, new_kid}

    def test_kid_never_reused_after_expiry(self, key_path, monkeypatch):
        """The count-based scheme reused es256-2 after a key expired; the
        thumbprint scheme guarantees a rotated-in key never collides with an
        expired one, so an in-flight token signed by the survivor still 200s."""
        mgr = InternalSigningKeyManager()
        first_kid = mgr.get_signing_kid()

        second_key = _write_p256_key(key_path)
        mgr._load_key_from_file(key_path)
        second_kid = _jwk_thumbprint(second_key.public_key())

        # Force the first key past retention and expire it out of the JWKS.
        monkeypatch.setattr(isk, "_KEY_RETENTION_SECONDS", 0)
        mgr._keys[0].loaded_at -= 10_000
        assert set(mgr.get_verification_keys()) == {second_kid}

        # Rotate in a third key; its kid must not collide with the expired one.
        third_key = _write_p256_key(key_path)
        mgr._load_key_from_file(key_path)
        third_kid = _jwk_thumbprint(third_key.public_key())
        assert third_kid not in {first_kid, second_kid}
        assert mgr.get_signing_kid() == third_kid

    def test_newest_key_never_expired(self, key_path, monkeypatch):
        """Even with zero retention and a single stale key, the sole/newest key
        is never dropped (no window with an empty JWKS)."""
        mgr = InternalSigningKeyManager()
        monkeypatch.setattr(isk, "_KEY_RETENTION_SECONDS", 0)
        mgr._keys[-1].loaded_at -= 10_000
        assert len(mgr.get_verification_keys()) == 1


class TestSigningMaterial:
    def test_get_signing_material_matches_newest(self, key_path):
        mgr = InternalSigningKeyManager()
        material = mgr.get_signing_material()
        assert material is not None
        private_key, kid = material
        assert kid == mgr.get_signing_kid()
        expected = mgr.get_verification_keys()[kid].public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        got = private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        assert got == expected

    def test_no_material_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("INTERNAL_SIGNING_KEY_PATH", raising=False)
        monkeypatch.delenv("INTERNAL_SIGNING_KEY_GENERATE", raising=False)
        mgr = InternalSigningKeyManager()
        assert mgr.get_signing_material() is None
        assert not mgr.is_available


class TestRetentionDefault:
    def test_retention_tracks_max_token_ttl(self, monkeypatch):
        monkeypatch.setenv("MCP_TOKEN_MAX_TTL_HOURS", "168")
        assert isk._default_retention_seconds() == (168 + 1) * 3600

    def test_retention_clamped_to_absolute_ceiling(self, monkeypatch):
        monkeypatch.setenv("MCP_TOKEN_MAX_TTL_HOURS", "9999")
        assert isk._default_retention_seconds() == (168 + 1) * 3600

    def test_retention_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MCP_TOKEN_MAX_TTL_HOURS", raising=False)
        assert isk._default_retention_seconds() == (24 + 1) * 3600
