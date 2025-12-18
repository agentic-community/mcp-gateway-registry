"""
Unit tests for upstream secret encryption envelope (Phase 1).
"""

from __future__ import annotations

import re

import pytest

from auth_server.enforceai.crypto.upstream_secrets import (
    EncryptedSecretEnvelope,
    build_aad_for_upstream_credential,
    decrypt_secret_payload,
    encrypt_secret_payload,
)
from auth_server.enforceai.models.upstream_credentials import (
    UpstreamCredentialSecret,
)


@pytest.mark.unit
class TestUpstreamSecretEnvelope:
    def test_encrypt_decrypt_roundtrip(
        self,
    ) -> None:
        key = b"\x11" * 32
        aad = build_aad_for_upstream_credential(
            credential_id="cred-1",
            server_path="/fininfo",
            credential_type="api-key",
        )
        payload = {"api_key": "super-secret"}

        envelope = encrypt_secret_payload(
            key=key,
            aad=aad,
            payload=payload,
        )
        assert isinstance(envelope, EncryptedSecretEnvelope)
        assert envelope.nonce != b""
        assert envelope.ciphertext != b""

        decrypted = decrypt_secret_payload(
            key=key,
            aad=aad,
            envelope=envelope,
        )
        assert decrypted == payload

    def test_secret_model_repr_redacts_payload(
        self,
    ) -> None:
        secret = UpstreamCredentialSecret(
            credential_id="cred-1",
            payload={"access_token": "tok-abc"},
        )
        rendered = repr(secret)
        assert "tok-abc" not in rendered
        assert re.search(r"payload=", rendered) is None

