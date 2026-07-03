"""AEAD OAuth-state codec tests: roundtrip, confidentiality, tamper, wrong-key."""

import base64

import pytest

from registry.egress_auth import state_codec
from registry.egress_auth.schemas import OAuthState


def _state(**over) -> OAuthState:
    base = {
        "user_id": "alice",
        "auth_method": "oauth2",
        "client_id": "Iv1.x",
        "provider": "github",
        "server_path": "/github-mcp",
        "session_id": "sess-1",
        "pkce_verifier": "verifier-secret-value",
        "nonce": "nonce-abc",
        "issued_at": "2026-06-19T00:00:00+00:00",
    }
    base.update(over)
    return OAuthState(**base)


@pytest.fixture(autouse=True)
def _reset_cipher():
    state_codec.reset_cipher_for_tests()
    yield
    state_codec.reset_cipher_for_tests()


@pytest.mark.unit
class TestStateCodec:
    def test_roundtrip(self):
        blob = state_codec.encode_state(_state())
        out = state_codec.decode_state(blob)
        assert out.user_id == "alice"
        assert out.pkce_verifier == "verifier-secret-value"
        assert out.nonce == "nonce-abc"

    def test_verifier_not_plaintext_in_blob(self):
        # The whole point of AEAD: the PKCE verifier must NOT be readable in the
        # state value (which travels in the URL/Referer/logs).
        blob = state_codec.encode_state(_state(pkce_verifier="SUPERSECRETVERIFIER"))
        assert "SUPERSECRETVERIFIER" not in blob
        raw = base64.urlsafe_b64decode(blob.encode())
        assert b"SUPERSECRETVERIFIER" not in raw

    def test_tamper_is_rejected(self):
        blob = state_codec.encode_state(_state())
        # flip a byte in the ciphertext region
        raw = bytearray(base64.urlsafe_b64decode(blob.encode()))
        raw[-1] ^= 0x01
        tampered = base64.urlsafe_b64encode(bytes(raw)).decode()
        with pytest.raises(state_codec.InvalidState):
            state_codec.decode_state(tampered)

    def test_garbage_rejected(self):
        with pytest.raises(state_codec.InvalidState):
            state_codec.decode_state("not-valid-base64-@@@")
        with pytest.raises(state_codec.InvalidState):
            state_codec.decode_state("c2hvcnQ=")  # decodes but too short

    def test_wrong_key_rejected(self, monkeypatch):
        blob = state_codec.encode_state(_state())
        # Re-key: a different SECRET_KEY must not decrypt the old blob.
        monkeypatch.setenv("SECRET_KEY", "a-totally-different-secret-key-value-32b")
        state_codec.reset_cipher_for_tests()
        with pytest.raises(state_codec.InvalidState):
            state_codec.decode_state(blob)
