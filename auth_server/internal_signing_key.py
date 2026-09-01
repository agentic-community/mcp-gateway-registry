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
- The kid is the RFC 7638 JWK thumbprint of the public key, so every replica
  derives the identical kid for the same physical key with no shared counter

Security:
- Private key is ONLY held by auth-server (never exported, never in env vars)
- Public key / JWKS is non-sensitive (served without authentication)
- kid is the RFC 7638 thumbprint (stable per key, never reused across keys),
  so rotation and multi-replica overlap never collide two keys onto one kid
- Key generation uses OS CSPRNG via the cryptography library
"""

import base64
import hashlib
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

# How long to keep old keys in JWKS after rotation (seconds).
#
# Must be >= the longest self-signed token TTL, else a key can be expired out
# of the JWKS while tokens it signed are still valid → those tokens 401. The
# longest token TTL is the operator-configured MCP_TOKEN_MAX_TTL_HOURS (default
# 24h, hard-capped at 168h by the registry settings clamp). We default retention
# to that ceiling + 1h of overlap so raising the token cap automatically keeps
# retention ahead of it. An explicit INTERNAL_SIGNING_KEY_RETENTION_SECONDS
# override still wins.
_MCP_TOKEN_ABSOLUTE_MAX_TTL_HOURS: int = 168  # mirrors registry.core.config


def _default_retention_seconds() -> int:
    try:
        max_ttl_hours = int(os.environ.get("MCP_TOKEN_MAX_TTL_HOURS", "24"))
    except ValueError:
        max_ttl_hours = 24
    max_ttl_hours = max(1, min(max_ttl_hours, _MCP_TOKEN_ABSOLUTE_MAX_TTL_HOURS))
    return (max_ttl_hours + 1) * 3600  # +1h overlap for clock skew / rotation


_KEY_RETENTION_SECONDS: int = int(
    os.environ.get("INTERNAL_SIGNING_KEY_RETENTION_SECONDS", str(_default_retention_seconds()))
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


def _jwk_thumbprint(public_key: ec.EllipticCurvePublicKey) -> str:
    """Compute the RFC 7638 JWK thumbprint of an EC P-256 public key.

    The thumbprint is the base64url-encoded SHA-256 of the JWK's required
    members serialized as compact JSON with keys in lexicographic order
    (crv, kty, x, y). It is deterministic per key and identical across
    processes/replicas, so it is a safe, collision-free kid that is never
    reused for a different key.
    """
    numbers = public_key.public_numbers()
    x = _base64url_encode(numbers.x.to_bytes(32, byteorder="big"))
    y = _base64url_encode(numbers.y.to_bytes(32, byteorder="big"))
    canonical = f'{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}'
    digest = hashlib.sha256(canonical.encode("ascii")).digest()
    return f"es256-{_base64url_encode(digest)}"


class _KeyEntry:
    """A single signing key with metadata."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        loaded_at: float,
    ):
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.kid = _jwk_thumbprint(self.public_key)
        self.loaded_at = loaded_at
        self.jwk = _compute_jwk(self.public_key, self.kid)


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

        # The signing key path. Defaults to the standard mount point used by
        # Kubernetes (Secret volume mount) and docker-compose (bind mount).
        # Operators can override with INTERNAL_SIGNING_KEY_PATH if needed.
        _DEFAULT_KEY_PATH = "/etc/mcp-gateway/signing-key/key.pem"
        key_path = os.environ.get("INTERNAL_SIGNING_KEY_PATH", "")
        auto_generate = os.environ.get("INTERNAL_SIGNING_KEY_GENERATE", "false").lower() == "true"

        # If no explicit path, check the default mount point (key may be
        # mounted via k8s Secret without any env var configuration).
        if not key_path and os.path.isfile(_DEFAULT_KEY_PATH):
            key_path = _DEFAULT_KEY_PATH

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

    def get_signing_material(
        self,
    ) -> tuple[ec.EllipticCurvePrivateKey, str] | None:
        """Return (current private key, its kid) atomically for signing.

        Avoids the TOCTOU of calling get_signing_key() then get_signing_kid()
        separately: a rotation between the two would pair a key with the wrong
        kid. Only auth-server should call this. Returns None if no key.
        """
        self._maybe_reload()
        with self._lock:
            if not self._keys:
                return None
            entry = self._keys[-1]
            return entry.private_key, entry.kid

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
        newest = self._keys[-1]
        self._keys = [
            entry
            for entry in self._keys
            if entry is newest  # Always keep the newest
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

            # kid is derived from the key's RFC 7638 thumbprint in _KeyEntry.

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

                # New key — add it (kid = RFC 7638 thumbprint, computed by _KeyEntry)
                entry = _KeyEntry(private_key=private_key, loaded_at=time.time())
                self._keys.append(entry)

            self._last_mtime = Path(path).stat().st_mtime
            logger.info("Loaded ES256 signing key from %s (kid=%s)", path, entry.kid)

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
            except Exception:  # nosec B110 - best-effort mtime update; safe to ignore
                pass

    def _generate_ephemeral_key(self) -> None:
        """Generate an ephemeral ES256 keypair for development/testing.

        WARNING: This key is lost on restart. Single-instance only.
        """
        private_key = ec.generate_private_key(ec.SECP256R1())
        entry = _KeyEntry(private_key=private_key, loaded_at=time.time())
        with self._lock:
            self._keys.append(entry)
        logger.warning(
            "Generated EPHEMERAL ES256 signing key (kid=%s). "
            "This key is lost on restart — mount a persistent key for production.",
            entry.kid,
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
