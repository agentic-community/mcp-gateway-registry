"""Application-layer AEAD codec for per-user egress credentials.

This is the single, backend-independent crypto layer that sits between the
egress ``SecretStore`` implementations (Secrets Manager / OpenBao) and their
storage backends. It turns a plaintext :class:`StoredToken` into the versioned
encrypted envelope that is actually persisted, and back again -- so the vault
contains ciphertext rather than usable PAT / OAuth access / OAuth refresh
tokens.

Design (see issue #1665):

- **Root key.** A dedicated deployment-level secret,
  ``EGRESS_CREDENTIAL_ENCRYPTION_KEY``, separate from ``SECRET_KEY`` for
  cryptographic domain separation and an independent rotation lifecycle. It must
  live OUTSIDE the secret-store trust boundary this feature protects.
- **Per-user key derivation.** The root key is never used directly. A
  cryptographically independent AES-256 key is derived per canonical principal
  with HKDF-SHA256, using the SAME canonical identity that names the principal's
  vault namespace (``keys.user_principal(auth_method, user_id)``) so the same
  user deterministically gets the same key across replicas and restarts.
- **Address binding.** The AEAD Associated Data binds the ciphertext to its full
  logical vault address ``(auth_method, user_id, provider, server_path)`` so a
  ciphertext copied to a different user/server/provider fails authentication.
- **Versioned envelope.** The stored representation is an explicitly versioned
  dict (``_encrypted``/``version``/``algorithm``/``key_id``/``nonce``/
  ``ciphertext``) so future algorithm changes and key rotation are unambiguous.
- **Fail closed.** An encrypted envelope encountered while encryption is
  disabled, an unknown version/algorithm/key_id, a bad tag, or a wrong key all
  raise -- credentials are NEVER returned or re-persisted as plaintext on a
  crypto failure. Error messages never contain plaintext or key material.

Backward compatibility:

- When the key is unset the codec is a passthrough: it writes and reads the
  legacy plaintext ``StoredToken.model_dump()`` representation unchanged.
- When the key is set, reads still recognize legacy plaintext entries (so an
  in-place upgrade keeps working). :meth:`needs_migration` lets a store detect a
  legacy entry it just read so it can re-persist it encrypted ("read repair").
"""

import base64
import json
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreError

# Versioned constants for the stored envelope. Bumping the wire format or the
# HKDF domain would change these; readers validate them so an unrecognized
# envelope fails closed instead of being mis-decrypted.
_ENVELOPE_VERSION: int = 1
_ALGORITHM: str = "AES-256-GCM"
_KEY_ID: str = "v1"
_NONCE_BYTES: int = 12

# HKDF domain separator for the per-user key. Distinct from the session-id-token
# and OAuth-state HKDF ``info`` strings so a leak of one derived key never
# exposes another. The canonical principal identity is appended per derivation.
_HKDF_INFO_PREFIX: bytes = b"mcp-gateway-egress-credential-v1|"

# AEAD associated-data domain separator prepended to the encoded credential
# address. Encoded segments are base64url ASCII, so this joins unambiguously.
_AAD_PREFIX: bytes = b"mcp-gateway-egress-credential-aad-v1|"


def _is_encrypted_envelope(document: dict) -> bool:
    """True if ``document`` is one of our versioned encrypted envelopes."""
    return isinstance(document, dict) and document.get("_encrypted") is True


class CredentialCodec:
    """Encode/decode :class:`StoredToken` to/from the persisted representation.

    A single instance is shared by a backend store (built by the factory).
    Deriving a per-user key is cheap but not free, so derived ciphers are cached
    per canonical principal for the process lifetime; the cache holds only
    AES-GCM objects, never plaintext credentials.
    """

    def __init__(self, root_key: bytes | None, require_encrypted: bool = False) -> None:
        # ``None`` => encryption disabled (legacy plaintext behavior). A non-None
        # value is the raw HKDF input keying material (already length-validated
        # by config).
        self._root_key = root_key
        # Terminal "strict" state: once an operator is confident migration has
        # completed, this rejects any legacy plaintext entry on read so a
        # write-capable attacker on the backend cannot downgrade an envelope to
        # plaintext (or inject a plaintext token) and have it accepted. Only
        # meaningful when encryption is enabled.
        self._require_encrypted = require_encrypted and root_key is not None
        self._cipher_cache: dict[str, AESGCM] = {}

    @property
    def enabled(self) -> bool:
        return self._root_key is not None

    @property
    def require_encrypted(self) -> bool:
        return self._require_encrypted

    # -- key derivation ------------------------------------------------------ #

    def _cipher_for_principal(self, auth_method: str, user_id: str) -> AESGCM:
        if self._root_key is None:  # callers guard via ``enabled``; fail closed regardless
            raise SecretStoreError("Egress credential encryption key is not configured")
        principal = keys.user_principal(auth_method, user_id)
        cached = self._cipher_cache.get(principal)
        if cached is not None:
            return cached
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=_HKDF_INFO_PREFIX + principal.encode("ascii"),
        )
        cipher = AESGCM(hkdf.derive(self._root_key))
        self._cipher_cache[principal] = cipher
        return cipher

    @staticmethod
    def _aad(
        version: int,
        algorithm: str,
        key_id: str,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> bytes:
        """Build the AEAD associated data.

        Binds BOTH the envelope metadata (``version``/``algorithm``/``key_id``)
        and the full logical vault address. Binding the metadata means that once
        multiple keys exist (rotation), an attacker cannot flip ``key_id`` to
        steer decryption to a different key without breaking the tag -- a
        downgrade/key-confusion guard that is free to add now. Binding the
        address makes a ciphertext copied to another user/provider/server fail
        authentication.
        """
        return _AAD_PREFIX + (
            f"{version}|{algorithm}|{key_id}|"
            f"{keys.encode_segment(auth_method)}|{keys.encode_segment(user_id)}|"
            f"{keys.encode_segment(provider)}|{keys.encode_segment(server_path)}"
        ).encode("ascii")

    # -- codec --------------------------------------------------------------- #

    def encode(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> dict:
        """Return the dict to persist for ``token`` at the given address.

        Encryption disabled -> the legacy plaintext ``model_dump()``. Enabled ->
        a versioned AES-256-GCM envelope with a fresh random nonce.
        """
        if not self.enabled:
            return token.model_dump()
        cipher = self._cipher_for_principal(auth_method, user_id)
        nonce = os.urandom(_NONCE_BYTES)
        plaintext = token.model_dump_json().encode("utf-8")
        aad = self._aad(
            _ENVELOPE_VERSION,
            _ALGORITHM,
            _KEY_ID,
            auth_method,
            user_id,
            provider,
            server_path,
        )
        try:
            ciphertext = cipher.encrypt(nonce, plaintext, associated_data=aad)
        except Exception as exc:  # pragma: no cover - defensive; never leak plaintext
            raise SecretStoreError("Egress credential encryption failed") from exc
        return {
            "_encrypted": True,
            "version": _ENVELOPE_VERSION,
            "algorithm": _ALGORITHM,
            "key_id": _KEY_ID,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

    def decode(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        document: dict,
    ) -> StoredToken:
        """Return the :class:`StoredToken` for a persisted ``document``.

        Recognizes both the legacy plaintext representation and the versioned
        encrypted envelope. Fails closed (raises :class:`SecretStoreError`,
        never leaking plaintext/keys) on an envelope seen while encryption is
        disabled, an unsupported envelope, a bad tag (tamper / wrong user /
        wrong server), or a wrong key.
        """
        if not _is_encrypted_envelope(document):
            # Legacy plaintext entry (or a fresh write when encryption is off).
            if self._require_encrypted:
                # Terminal strict state: never accept plaintext, so a downgraded
                # or injected plaintext entry fails closed instead of vending.
                raise SecretStoreError(
                    "Egress credential is plaintext but "
                    "EGRESS_CREDENTIAL_ENCRYPTION_REQUIRE_ENCRYPTED is set"
                )
            return StoredToken(**document)

        if not self.enabled:
            raise SecretStoreError(
                "Egress credential is encrypted but EGRESS_CREDENTIAL_ENCRYPTION_KEY is not set"
            )
        if (
            document.get("version") != _ENVELOPE_VERSION
            or document.get("algorithm") != _ALGORITHM
            or document.get("key_id") != _KEY_ID
        ):
            raise SecretStoreError("Unsupported egress credential envelope version/algorithm/key")
        try:
            nonce = base64.b64decode(document["nonce"])
            ciphertext = base64.b64decode(document["ciphertext"])
        except Exception as exc:
            raise SecretStoreError("Malformed egress credential envelope") from exc

        cipher = self._cipher_for_principal(auth_method, user_id)
        aad = self._aad(
            document["version"],
            document["algorithm"],
            document["key_id"],
            auth_method,
            user_id,
            provider,
            server_path,
        )
        try:
            plaintext = cipher.decrypt(nonce, ciphertext, associated_data=aad)
        except Exception as exc:
            # Bad tag: tamper, cross-user/cross-server substitution, or wrong
            # key. Never say which, never fall back to plaintext.
            raise SecretStoreError("Egress credential failed authentication") from exc
        try:
            return StoredToken(**json.loads(plaintext))
        except Exception as exc:
            raise SecretStoreError("Egress credential payload malformed") from exc

    def needs_migration(self, document: dict) -> bool:
        """True if ``document`` is legacy plaintext and should be re-persisted encrypted.

        Used for read-repair: when encryption is enabled and a read surfaces a
        legacy plaintext entry, the store re-encodes and re-persists it.
        """
        return self.enabled and not _is_encrypted_envelope(document)


def build_credential_codec(
    root_key: str | None, require_encrypted: bool = False
) -> CredentialCodec:
    """Build a :class:`CredentialCodec` from the configured root key string.

    An empty/None key yields a disabled (passthrough) codec. Config already
    length- and placeholder-validates a non-empty key, so we only strip and
    UTF-8-encode it here as HKDF input keying material. ``require_encrypted``
    is the terminal strict switch (rejects legacy plaintext on read); it is
    inert unless a key is configured.
    """
    if not root_key or not root_key.strip():
        return CredentialCodec(root_key=None, require_encrypted=require_encrypted)
    return CredentialCodec(
        root_key=root_key.strip().encode("utf-8"), require_encrypted=require_encrypted
    )
