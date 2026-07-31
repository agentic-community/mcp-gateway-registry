"""Internal signing key manager for asymmetric JWT signing (ES256).

Manages the ES256 keypair used by auth-server to sign internal JWTs
(user-vended tokens + service-to-service tokens). Publishes the public
key(s) as a JWKS document at /.well-known/internal-jwks.json.

Supports live key rotation without restart:
- Periodically checks key file mtime (configurable interval, default 60s)
- When a new key is detected, adds it to the JWKS alongside the old key
- Signing always uses the NEWEST key; verification accepts ANY key in JWKS
- Old keys are retained in JWKS for the token max-TTL window (default 24h)
  then automatically removed

Key source priority:
1. File-mounted PEM private key (INTERNAL_SIGNING_KEY_PATH) — production
2. Auto-generated ephemeral key (INTERNAL_SIGNING_KEY_GENERATE=true) — dev
3. None — asymmetric signing disabled, falls back to HS256 (legacy)

Multi-replica:
- All replicas mount the same key file → same key → same JWKS
- On rotation, all replicas detect the file change independently
- The kid overlap window ensures tokens signed by any replica verify everywhere

Security:
- Private key is ONLY held by auth-server (never exported, never in env vars)
- Public key / JWKS is non-sensitive (served without authentication)
- kid is rotation-numbered (es256-1, es256-2, ...) for overlap during rotation
- Key generation uses OS CSPRNG via the cryptography library
"""

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

logger = logging.getLogger(__name__)

# How often to check if the key file changed (seconds)
_KEY_RELOAD_INTERVAL_SECONDS: int = int(os.environ.get("INTERNAL_SIGNING_KEY_RELOAD_SECONDS", "60"))

# How long to keep old keys in JWKS after rotation (seconds)
# Must be >= longest token TTL (24h for user tokens)
_KEY_RETENTION_SECONDS: int = int(
    os.environ.get("INTERNAL_SIGNING_KEY_RETENTION_SECONDS", str(25 * 3600))  # 25h
)


def _base64url_encode(data: bytes) -> str:
    """Base64url encode without padding (per RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _compute_jwk(public_key: ec.EllipticCurvePublicKey, kid: str) -> dict[str, Any]:
    """Derive the JWK representation of an EC P-256 public key."""
    numbers = public_key.public_numbers()
    x_bytes = numbers.x.to_bytes(32, byteorder="big")
    y_bytes = numbers.y.to_bytes(32, byteorder="big")
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _base64url_encode(x_bytes),
        "y": _base64url_encode(y_bytes),
        "kid": kid,
        "use": "sig",
        "alg": "ES256",
    }


class _KeyEntry:
    """A single signing key with metadata."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        kid: str,
        loaded_at: float,
    ):
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.kid = kid
        self.loaded_at = loaded_at
        self.jwk = _compute_jwk(self.public_key, kid)


class InternalSigningKeyManager:
    """Manages the ES256 signing keypair(s) for internal JWTs.

    Supports live rotation: periodically checks the key file for changes
    and adds new keys to the JWKS. Old keys are retained during the overlap
    window so tokens signed with the previous key remain verifiable.
    """

    def __init__(self) -> None:
        self._keys: list[_KeyEntry] = []
        self._lock = threading.Lock()
        self._key_path: str = ""
        self._last_mtime: float = 0.0
        self._last_check_time: float = 0.0

        key_path = os.environ.get("INTERNAL_SIGNING_KEY_PATH", "")
        auto_generate = os.environ.get("INTERNAL_SIGNING_KEY_GENERATE", "false").lower() == "true"

        if key_path:
            self._key_path = key_path
            self._load_key_from_file(key_path)
        elif auto_generate:
            self._generate_ephemeral_key()
        else:
            logger.info(
                "Internal asymmetric signing not configured "
                "(set INTERNAL_SIGNING_KEY_PATH or INTERNAL_SIGNING_KEY_GENERATE=true). "
                "Falling back to HS256 (legacy)."
            )

    @property
    def is_available(self) -> bool:
        """Whether asymmetric signing is configured and ready."""
        return len(self._keys) > 0

    @property
    def kid(self) -> str:
        """Current (newest) key ID for signing."""
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[-1].kid

    def get_public_jwks(self) -> dict[str, Any]:
        """Return all active public keys as a JWKS document.

        Includes the current signing key AND any old keys still within the
        retention window. Safe to serve without authentication.
        """
        self._maybe_reload()
        with self._lock:
            self._expire_old_keys()
            return {"keys": [entry.jwk for entry in self._keys]}

    def get_signing_key(self) -> ec.EllipticCurvePrivateKey | None:
        """Return the CURRENT (newest) private key for signing.

        Only auth-server should call this.
        """
        self._maybe_reload()
        with self._lock:
            if not self._keys:
                return None
            return self._keys[-1].private_key

    def get_signing_kid(self) -> str:
        """Return the kid of the current signing key."""
        self._maybe_reload()
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[-1].kid

    def get_verification_keys(self) -> dict[str, ec.EllipticCurvePublicKey]:
        """Return all active public keys indexed by kid (for local verification)."""
        self._maybe_reload()
        with self._lock:
            self._expire_old_keys()
            return {entry.kid: entry.public_key for entry in self._keys}

    def _maybe_reload(self) -> None:
        """Check if the key file changed and reload if so.

        Called on every sign/verify operation, but only actually checks the
        filesystem at the configured interval (default 60s).
        """
        if not self._key_path:
            return

        now = time.time()
        if now - self._last_check_time < _KEY_RELOAD_INTERVAL_SECONDS:
            return
        self._last_check_time = now

        try:
            current_mtime = Path(self._key_path).stat().st_mtime
            if current_mtime != self._last_mtime:
                logger.info(
                    "Key file %s changed (mtime %f → %f), reloading",
                    self._key_path,
                    self._last_mtime,
                    current_mtime,
                )
                self._load_key_from_file(self._key_path)
        except FileNotFoundError:
            pass  # File removed — keep serving existing keys
        except Exception as e:
            logger.warning("Failed to check key file mtime: %s", type(e).__name__)

    def _expire_old_keys(self) -> None:
        """Remove keys that have exceeded the retention window.

        Must be called with self._lock held.
        """
        if len(self._keys) <= 1:
            return  # Never remove the only key
        now = time.time()
        self._keys = [
            entry
            for entry in self._keys
            if entry == self._keys[-1]  # Always keep the newest
            or (now - entry.loaded_at) < _KEY_RETENTION_SECONDS
        ]

    def _load_key_from_file(self, path: str) -> None:
        """Load an ES256 private key from a PEM file.

        If this is a NEW key (different from the current signing key), it's
        added alongside existing keys (for overlap). If loading fails, the
        existing keys remain active (fail closed to current state).
        """
        try:
            with open(path, "rb") as f:
                key_data = f.read()

            private_key = serialization.load_pem_private_key(key_data, password=None)

            if not isinstance(private_key, ec.EllipticCurvePrivateKey):
                raise ValueError("Key is not an EC private key")
            if not isinstance(private_key.curve, ec.SECP256R1):
                raise ValueError(f"Key is not P-256 (got {private_key.curve.name})")

            # Determine kid — check INTERNAL_SIGNING_KEY_ID env, or auto-increment
            kid = os.environ.get("INTERNAL_SIGNING_KEY_ID", "")
            if not kid:
                with self._lock:
                    kid = f"es256-{len(self._keys) + 1}"

            # Check if this is actually a new key (compare public key bytes)
            new_pub_bytes = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with self._lock:
                for existing in self._keys:
                    existing_pub_bytes = existing.public_key.public_bytes(
                        serialization.Encoding.PEM,
                        serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                    if new_pub_bytes == existing_pub_bytes:
                        # Same key — just update mtime tracking
                        self._last_mtime = Path(path).stat().st_mtime
                        return

                # New key — add it
                entry = _KeyEntry(private_key=private_key, kid=kid, loaded_at=time.time())
                self._keys.append(entry)

            self._last_mtime = Path(path).stat().st_mtime
            logger.info("Loaded ES256 signing key from %s (kid=%s)", path, kid)

        except FileNotFoundError:
            logger.error(
                "INTERNAL_SIGNING_KEY_PATH=%s not found. "
                "Asymmetric signing disabled (falling back to HS256).",
                path,
            )
        except Exception as e:
            logger.error(
                "Failed to load signing key from %s: %s. Keeping existing keys active.",
                path,
                type(e).__name__,
            )
            # Update mtime on failure to prevent hot-loop retries.
            # The next reload check will re-attempt only if the file changes again.
            try:
                self._last_mtime = Path(path).stat().st_mtime
            except Exception:
                pass

    def _generate_ephemeral_key(self) -> None:
        """Generate an ephemeral ES256 keypair for development/testing.

        WARNING: This key is lost on restart. Single-instance only.
        """
        private_key = ec.generate_private_key(ec.SECP256R1())
        kid = os.environ.get("INTERNAL_SIGNING_KEY_ID", "es256-ephemeral")
        entry = _KeyEntry(private_key=private_key, kid=kid, loaded_at=time.time())
        with self._lock:
            self._keys.append(entry)
        logger.warning(
            "Generated EPHEMERAL ES256 signing key (kid=%s). "
            "This key is lost on restart — mount a persistent key for production.",
            kid,
        )


# Module-level singleton (initialized once at process startup)
_key_manager: InternalSigningKeyManager | None = None


_key_manager_lock = threading.Lock()


def get_internal_signing_key_manager() -> InternalSigningKeyManager:
    """Return the singleton key manager (initializes on first call, thread-safe)."""
    global _key_manager
    if _key_manager is None:
        with _key_manager_lock:
            if _key_manager is None:
                _key_manager = InternalSigningKeyManager()
    return _key_manager
