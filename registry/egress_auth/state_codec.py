"""AEAD-encrypted OAuth ``state`` codec for the egress consent round-trip.

The ``OAuthState`` (in particular ``pkce_verifier``) MUST be encrypted, not
merely signed, before it round-trips through the provider as the ``state`` query
param: a signed-only state is base64 *plaintext* and exposes the PKCE verifier
in the URL bar / Referer / provider logs / browser history, which defeats PKCE.
We AES-GCM-encrypt with a key derived from ``SECRET_KEY`` via
HKDF with a purpose-specific ``info`` -- mirroring ``registry.auth.session_crypto``
but with a distinct ``info`` so the egress-state key is cryptographically
independent of the session-id-token key and the PBKDF2 credential key.

Wire format: ``urlsafe_b64( nonce(12) || ciphertext )`` -- URL-safe so it can be
a ``state`` query parameter verbatim. The plaintext is the JSON of ``OAuthState``.

TTL and single-use (replay) enforcement live in the service layer (the codec
only guarantees confidentiality + integrity + tamper-detection); the service
checks ``issued_at`` against the TTL and records the ``nonce`` in a replay guard.
"""

import base64
import json
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from registry.egress_auth.schemas import OAuthState

# Distinct from session_crypto's HKDF_INFO so a leak of one key does not expose
# the other.
HKDF_INFO: bytes = b"mcp-gateway-egress-oauth-state-v1"
NONCE_BYTES: int = 12

_aesgcm: AESGCM | None = None


class InvalidState(Exception):
    """Raised when a state blob fails to decode, decrypt, or verify."""


def _derive_state_key() -> bytes:
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        raise RuntimeError("SECRET_KEY required for egress OAuth state encryption")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HKDF_INFO)
    return hkdf.derive(secret.encode("utf-8"))


def _get_aesgcm() -> AESGCM:
    global _aesgcm
    if _aesgcm is None:
        _aesgcm = AESGCM(_derive_state_key())
    return _aesgcm


def reset_cipher_for_tests() -> None:
    """Drop the cached cipher so a test that changes SECRET_KEY re-derives."""
    global _aesgcm
    _aesgcm = None


def encode_state(state: OAuthState) -> str:
    """Serialize, AES-GCM-encrypt, and URL-safe-base64 an OAuthState."""
    plaintext = state.model_dump_json().encode("utf-8")
    nonce = os.urandom(NONCE_BYTES)
    ct = _get_aesgcm().encrypt(nonce, plaintext, associated_data=None)
    return base64.urlsafe_b64encode(nonce + ct).decode("ascii")


def decode_state(blob: str) -> OAuthState:
    """Decode/decrypt/verify a state blob back into an OAuthState.

    Raises ``InvalidState`` on any malformed input, bad tag (tamper), or wrong
    key -- never leaks why. TTL/replay are enforced by the caller.
    """
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
    except Exception as exc:
        raise InvalidState("state is not valid base64") from exc
    if len(raw) <= NONCE_BYTES:
        raise InvalidState("state too short")
    nonce, ct = raw[:NONCE_BYTES], raw[NONCE_BYTES:]
    try:
        plaintext = _get_aesgcm().decrypt(nonce, ct, associated_data=None)
    except Exception as exc:
        raise InvalidState("state failed authentication") from exc
    try:
        return OAuthState(**json.loads(plaintext))
    except Exception as exc:
        raise InvalidState("state payload malformed") from exc
