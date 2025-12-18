from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import (
    AESGCM,
)

SECRET_ENVELOPE_VERSION: int = 1
AESGCM_NONCE_BYTES: int = 12


def _json_dumps_bytes(
    payload: dict[str, Any],
) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _json_loads_dict(
    raw: bytes,
) -> dict[str, Any]:
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Decrypted secret payload must be a JSON object")
    return parsed


def _validate_aesgcm_key(
    key: bytes,
) -> None:
    if len(key) not in {16, 24, 32}:
        raise ValueError("AESGCM key must be 16, 24, or 32 bytes")


@dataclass(frozen=True)
class EncryptedSecretEnvelope:
    version: int
    nonce: bytes
    ciphertext: bytes


def build_aad_for_upstream_credential(
    *,
    credential_id: str,
    server_path: str,
    credential_type: str,
) -> bytes:
    return f"upstream-credential|{credential_id}|{server_path}|{credential_type}".encode(
        "utf-8"
    )


def build_aad_for_upstream_oauth_state(
    *,
    state_id: str,
    server_path: str,
    credential_type: str,
) -> bytes:
    return f"upstream-oauth-state|{state_id}|{server_path}|{credential_type}".encode("utf-8")


def encrypt_secret_payload(
    *,
    key: bytes,
    aad: bytes,
    payload: dict[str, Any],
) -> EncryptedSecretEnvelope:
    _validate_aesgcm_key(key)

    nonce = secrets.token_bytes(AESGCM_NONCE_BYTES)
    aead = AESGCM(key)
    plaintext = _json_dumps_bytes(payload)
    ciphertext = aead.encrypt(
        nonce,
        plaintext,
        aad,
    )
    return EncryptedSecretEnvelope(
        version=SECRET_ENVELOPE_VERSION,
        nonce=nonce,
        ciphertext=ciphertext,
    )


def decrypt_secret_payload(
    *,
    key: bytes,
    aad: bytes,
    envelope: EncryptedSecretEnvelope,
) -> dict[str, Any]:
    _validate_aesgcm_key(key)
    if envelope.version != SECRET_ENVELOPE_VERSION:
        raise ValueError(f"Unsupported secret envelope version: {envelope.version}")

    aead = AESGCM(key)
    plaintext = aead.decrypt(
        envelope.nonce,
        envelope.ciphertext,
        aad,
    )
    return _json_loads_dict(plaintext)
