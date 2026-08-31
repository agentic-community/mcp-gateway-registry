"""Registry-side fetcher for the auth-server internal JWKS (ES256 public keys).

Auth-server signs internal hop tokens (``mcp-registry-ui`` / ``mcp-proxy``
audiences) with its ES256 private key when ``INTERNAL_SIGNING_KEY_PATH`` is
configured, stamping a ``kid`` header. Only auth-server holds the private key;
every other service verifies with the public half published at
``/.well-known/internal-jwks.json``.

The registry verifies those tokens in ``registry/auth/proxied_token.py``. This
module gives that verifier the public keys: it fetches the JWKS over the
in-cluster network, caches it with a TTL, and serves the last known-good keys
if a later fetch fails (bounded staleness). It mirrors the fetch/cache/
last-known-good shape of ``registry/services/federation_jwks_verifier.py`` but:

- targets the INTERNAL endpoint (plain HTTP over the cluster network, not the
  external Web-PKI federation JWKS), and
- is SYNCHRONOUS, because the internal-token verifiers are sync functions
  called on the request path.

When auth-server signs with HS256 (no private key configured), tokens carry no
``kid`` and this module is never consulted — the HS256/``SECRET_KEY`` legacy
path in ``proxied_token.py`` handles them.
"""

import logging
import threading
import time
from typing import Any

import httpx
from jwt import PyJWK

from ..core.config import settings

logger = logging.getLogger(__name__)

# Reject cache older than this even when refresh keeps failing, so a key
# rotation eventually takes effect and a stale key can't be trusted forever.
_MAX_CACHE_STALENESS_SECONDS: int = 86400  # 24h

# How long a kid confirmed-absent by a forced refresh is remembered as unknown,
# so a stream of tokens carrying random/forged kid headers cannot amplify into
# repeated forced JWKS fetches (each a 5s network call). Short enough that a
# genuinely just-rotated kid is still picked up by the next normal TTL refresh.
_UNKNOWN_KID_NEGATIVE_TTL_SECONDS: int = 30

# Hard bound on the negative-cache size so an attacker streaming unique kids
# cannot grow it without limit.
_UNKNOWN_KID_CACHE_MAX: int = 4096


class _InternalJwksCache:
    """Thread-safe TTL cache of the auth-server internal JWKS, indexed by kid."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys_by_kid: dict[str, Any] = {}
        self._fetched_at: float = 0.0
        self._have_keys: bool = False
        # kid -> time it was confirmed absent by a forced refresh (negative cache).
        self._unknown_kids: dict[str, float] = {}

    def _recently_unknown(self, kid: str, now: float) -> bool:
        """Whether ``kid`` was confirmed absent within the negative-cache TTL.

        Must be called with self._lock held.
        """
        ts = self._unknown_kids.get(kid)
        return ts is not None and (now - ts) < _UNKNOWN_KID_NEGATIVE_TTL_SECONDS

    def _mark_unknown(self, kid: str, now: float) -> None:
        """Record ``kid`` as confirmed-absent, pruning/bounding the cache.

        Must be called with self._lock held.
        """
        if len(self._unknown_kids) >= _UNKNOWN_KID_CACHE_MAX:
            cutoff = now - _UNKNOWN_KID_NEGATIVE_TTL_SECONDS
            self._unknown_kids = {k: t for k, t in self._unknown_kids.items() if t >= cutoff}
            if len(self._unknown_kids) >= _UNKNOWN_KID_CACHE_MAX:
                self._unknown_kids.clear()
        self._unknown_kids[kid] = now

    def _fetch(self) -> dict[str, Any]:
        """Fetch and parse the JWKS into a {kid: key_object} map. Raises on failure."""
        url = settings.internal_jwks_url
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            jwks = resp.json()

        keys_by_kid: dict[str, Any] = {}
        for key_data in jwks.get("keys", []):
            kid = key_data.get("kid")
            if not kid:
                continue
            try:
                keys_by_kid[kid] = PyJWK(key_data).key
            except Exception as exc:  # malformed single key — skip, keep the rest
                logger.warning(
                    "Skipping malformed internal JWK (kid=%s): %s", kid, type(exc).__name__
                )
        return keys_by_kid

    def get_key(self, kid: str) -> Any | None:
        """Return the public key for ``kid``, or None if unknown.

        Serves cached keys within the TTL. On a cache miss / expiry, refreshes
        once; if the target kid is still unknown after a fresh fetch, forces a
        second refresh (handles a just-rotated key) — unless that kid was
        recently confirmed absent (negative cache), which fails fast without a
        fetch to blunt random/forged-kid amplification. On fetch failure, falls
        back to the last known-good keys up to the max staleness bound.
        """
        now = time.time()
        ttl = settings.internal_jwks_cache_ttl_seconds

        with self._lock:
            fresh_enough = self._have_keys and (now - self._fetched_at) < ttl
            if fresh_enough and kid in self._keys_by_kid:
                return self._keys_by_kid[kid]
            # A kid confirmed absent very recently: fail fast, skip the fetch
            # storm. A genuinely rotated kid is still caught once the normal TTL
            # refresh below runs after the negative-cache entry expires.
            if fresh_enough and self._recently_unknown(kid, now):
                return None

        # Need a network fetch (expired, first use, or unknown kid).
        refreshed = self._refresh(now)
        if refreshed is not None and kid in refreshed:
            with self._lock:
                self._unknown_kids.pop(kid, None)
            return refreshed[kid]

        # Unknown kid after a successful refresh could mean a just-published
        # rotation the previous fetch missed — force one more refresh, unless we
        # already confirmed this kid absent within the negative-cache window.
        if refreshed is not None and kid not in refreshed:
            with self._lock:
                if self._recently_unknown(kid, now):
                    return None
            forced = self._refresh(time.time(), force=True)
            if forced is not None:
                key = forced.get(kid)
                with self._lock:
                    if key is None:
                        self._mark_unknown(kid, time.time())
                    else:
                        self._unknown_kids.pop(kid, None)
                return key

        # Fetch failed — fall back to last known-good within staleness bound.
        with self._lock:
            if self._have_keys and (now - self._fetched_at) < _MAX_CACHE_STALENESS_SECONDS:
                logger.warning(
                    "Internal JWKS fetch failed; using cached keys (age=%ds)",
                    int(now - self._fetched_at),
                )
                return self._keys_by_kid.get(kid)
        return None

    def _refresh(self, now: float, force: bool = False) -> dict[str, Any] | None:
        """Fetch and store keys. Returns the new map, or None on failure.

        A second refresh within the same request (force=True) is allowed to
        bypass the just-updated timestamp so a rotation is picked up promptly.
        """
        with self._lock:
            # Another thread may have refreshed while we waited for the lock.
            if (
                not force
                and self._have_keys
                and (now - self._fetched_at) < settings.internal_jwks_cache_ttl_seconds
            ):
                return dict(self._keys_by_kid)
        try:
            keys_by_kid = self._fetch()
        except Exception as exc:
            logger.warning("Internal JWKS fetch error: %s", type(exc).__name__)
            return None

        if not keys_by_kid:
            # Empty JWKS (e.g. rotation slip) — do NOT clobber good cache.
            logger.warning("Internal JWKS returned no usable keys; keeping existing cache")
            with self._lock:
                return dict(self._keys_by_kid) if self._have_keys else {}

        with self._lock:
            self._keys_by_kid = keys_by_kid
            self._fetched_at = time.time()
            self._have_keys = True
            return dict(self._keys_by_kid)


_cache = _InternalJwksCache()


def get_internal_verification_key(kid: str) -> Any | None:
    """Return the ES256 public key for ``kid`` from the auth-server internal JWKS.

    Returns None when the kid is unknown or the JWKS cannot be obtained. Callers
    treat None as a verification failure (fail closed).
    """
    return _cache.get_key(kid)
