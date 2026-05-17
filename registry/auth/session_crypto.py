"""Shared crypto helpers for the OAuth session store.

The auth-server (writer) and registry (reader) both encrypt and decrypt
OAuth `id_token` blobs at rest. They MUST agree on:
  - the SECRET_KEY-derived AES-GCM key (HKDF info string)
  - the wire format (12-byte nonce || ciphertext)

Keeping the helpers in one module ensures the constants cannot drift across
processes — a mismatch would silently fail every decrypt and break SSO logout.

Sensitivity model:
  - Username, email, name, groups are stored in plaintext (already client-
    visible in the previous signed-cookie payload).
  - `id_token` is the only true bearer credential and is encrypted with
    AES-GCM. Rotating SECRET_KEY invalidates stored id_tokens — the same
    blast radius as cookie rotation today.

Key rotation:
  The AES-GCM cipher is cached in a process-wide singleton on first use
  (see ``_aesgcm`` below). Rotating ``SECRET_KEY`` therefore requires a
  process restart, not a SIGHUP-style config reload — a hot reload would
  keep using the old key for the lifetime of the cached cipher. Operators
  who rotate ``SECRET_KEY`` must restart all auth_server and registry
  replicas to pick up the new key.
"""

import os
import secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HKDF_INFO: bytes = b"mcp-gateway-session-id-token-encryption"
NONCE_BYTES: int = 12

_aesgcm: AESGCM | None = None


def _derive_token_encryption_key() -> bytes:
    """Derive the AES-GCM key for id_token encryption from SECRET_KEY via HKDF.

    SECRET_KEY is required at startup; rotating it invalidates stored
    id_tokens (acceptable — same blast radius as cookie rotation today).
    """
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        raise RuntimeError("SECRET_KEY required for session token encryption")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=HKDF_INFO,
    )
    return hkdf.derive(secret.encode("utf-8"))


def get_aesgcm() -> AESGCM:
    """Lazy-init the AES-GCM cipher singleton."""
    global _aesgcm
    if _aesgcm is None:
        _aesgcm = AESGCM(_derive_token_encryption_key())
    return _aesgcm


def encrypt_id_token(token: str) -> bytes:
    """Encrypt an id_token. Returns nonce || ciphertext."""
    nonce = secrets.token_bytes(NONCE_BYTES)
    ct = get_aesgcm().encrypt(nonce, token.encode("utf-8"), associated_data=None)
    return nonce + ct


def decrypt_id_token(blob: bytes) -> str:
    """Decrypt an id_token blob produced by encrypt_id_token."""
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    return get_aesgcm().decrypt(nonce, ct, associated_data=None).decode("utf-8")
