"""Shared, hardened JWKS cache for the IdP auth providers.

Every provider (Keycloak, Entra, Cognito, Okta, Auth0, PingFederate) resolves
token signing keys through the same primitive so the *whole* fleet gets the
same robustness guarantees:

- **TTL cache** keyed by ``jwks_url`` (operator-configurable, default 300s).
- **Coalesced fetch**: concurrent callers that miss the cache do at most one
  upstream request; the rest wait on the per-URL lock and reuse the result.
  The providers call this from synchronous ``requests``-based code, so the
  lock is a ``threading`` lock (uvicorn runs sync handlers in a threadpool).
- **Unknown-``kid`` refetch, coalesced + negative-cached**: an unrecognised
  ``kid`` triggers **at most one** refetch per unknown ``kid`` per short
  window, shared across concurrent callers, after which the ``kid`` is
  negative-cached briefly. This is what stops an attacker spraying random
  ``kid`` values from amplifying into a JWKS-endpoint DoS.
- **Bounded stale fallback**: when a fetch fails we serve the last-good cache
  within a bounded staleness window (previously only Okta/PingFederate did
  this; Keycloak/Entra/Cognito now inherit it), and raise only when no cache
  exists at all.

RS256 pinning and every signature / issuer / audience check stay in the
provider ``jwt.decode`` calls — this module only owns cache robustness.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Operator-configurable defaults. Read from the environment at call time (not
# import time) so tests and live re-config both see current values. The
# providers already read their own knobs straight from os.environ, so this
# stays consistent with the existing pattern rather than threading them
# through registry.core.config.
DEFAULT_TTL_SECONDS = 300
DEFAULT_STALE_TTL_SECONDS = 86400  # 24h bounded staleness fallback
DEFAULT_NEGATIVE_TTL_SECONDS = 30  # how long an unknown kid stays negative-cached
_HTTP_TIMEOUT_SECONDS = 10
_MAX_FETCH_ATTEMPTS = 2
_RETRY_SLEEP_SECONDS = 1
DEFAULT_UNKNOWN_REFETCH_COOLDOWN_SECONDS = 5  # min gap between unknown-kid refetches
MAX_UNKNOWN_KIDS = 1024  # cap on the negative-cache size per jwks_url


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%r; using default %d", name, raw, default)
        return default
    return value if value > 0 else default


def _cache_ttl() -> int:
    return _env_int("MCP_JWKS_CACHE_TTL", DEFAULT_TTL_SECONDS)


def _stale_ttl() -> int:
    return _env_int("MCP_JWKS_STALE_TTL", DEFAULT_STALE_TTL_SECONDS)


def _negative_ttl() -> int:
    return _env_int("MCP_JWKS_NEGATIVE_CACHE_TTL", DEFAULT_NEGATIVE_TTL_SECONDS)


def _unknown_refetch_cooldown() -> int:
    return _env_int("MCP_JWKS_UNKNOWN_REFETCH_COOLDOWN", DEFAULT_UNKNOWN_REFETCH_COOLDOWN_SECONDS)


def _safe_kid(kid: str | None) -> str:
    """Render a ``kid`` for logs without the whole value."""
    if not kid:
        return "<none>"
    if len(kid) <= 12:
        return kid
    return f"{kid[:8]}\u2026({len(kid)} chars)"


class _Entry:
    """Per-``jwks_url`` cache state."""

    __slots__ = ("lock", "jwks", "fetched_at", "unknown_kids", "last_unknown_refetch")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jwks: dict[str, Any] | None = None
        self.fetched_at: float = 0.0
        # kid -> negative-cache expiry timestamp
        self.unknown_kids: dict[str, float] = {}
        # Rate-limits unknown-kid refetches (see get_signing_key).
        self.last_unknown_refetch: float = 0.0


class JwksCache:
    """Reusable JWKS cache, keyed internally by ``jwks_url``.

    A provider composes exactly one of these; because providers are process
    singletons, all concurrent ``/validate`` calls for a provider share the
    same cache and lock (so coalescing and the negative cache actually bite).
    Keying by URL also supports providers whose JWKS URL is discovery-derived
    and lets a single instance serve more than one issuer if needed.
    """

    def __init__(self, provider_name: str = "") -> None:
        self._provider_name = provider_name or "idp"
        self._registry_lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}

    # -- internals ---------------------------------------------------------

    def _entry(self, jwks_url: str) -> _Entry:
        with self._registry_lock:
            entry = self._entries.get(jwks_url)
            if entry is None:
                entry = _Entry()
                self._entries[jwks_url] = entry
            return entry

    def _fetch(self, jwks_url: str) -> dict[str, Any]:
        """Fetch fresh JWKS with a bounded retry. Raises on total failure.

        Do not introduce ``verify=False``; a TLS failure is a CA-bundle
        problem to fix at deployment, not to paper over here.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_FETCH_ATTEMPTS):
            try:
                logger.debug("Fetching JWKS for %s (attempt %d)", self._provider_name, attempt + 1)
                response = requests.get(jwks_url, timeout=_HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 - surface any fetch failure to caller
                last_error = exc
                logger.warning(
                    "JWKS fetch attempt %d for %s failed: %s",
                    attempt + 1,
                    self._provider_name,
                    exc,
                )
                if attempt < _MAX_FETCH_ATTEMPTS - 1:
                    time.sleep(_RETRY_SLEEP_SECONDS)
        if last_error is not None:
            raise last_error
        raise RuntimeError("JWKS fetch failed without a captured error")

    @staticmethod
    def _find_key(jwks: dict[str, Any] | None, kid: str) -> dict[str, Any] | None:
        if not jwks:
            return None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None

    @staticmethod
    def _to_signing_key(jwk: dict[str, Any]) -> Any:
        from jwt import PyJWK

        return PyJWK(jwk).key

    # -- public API --------------------------------------------------------

    def get_jwks(self, jwks_url: str) -> dict[str, Any]:
        """Return the JWKS for ``jwks_url``, fresh or cached.

        Coalesces concurrent misses and, on fetch failure, serves the
        last-good cache within the bounded staleness window. Raises
        ``ValueError`` only when the fetch fails and no usable cache exists.
        """
        entry = self._entry(jwks_url)

        now = time.time()
        if entry.jwks is not None and (now - entry.fetched_at) < _cache_ttl():
            return entry.jwks

        with entry.lock:
            # Double-check: another thread may have refreshed while we waited.
            now = time.time()
            if entry.jwks is not None and (now - entry.fetched_at) < _cache_ttl():
                return entry.jwks

            try:
                fresh = self._fetch(jwks_url)
            except Exception as exc:  # noqa: BLE001
                if entry.jwks is not None and (now - entry.fetched_at) < _stale_ttl():
                    logger.warning(
                        "JWKS fetch failed for %s; serving stale cache (age %.0fs): %s",
                        self._provider_name,
                        now - entry.fetched_at,
                        exc,
                    )
                    return entry.jwks
                logger.error(
                    "Failed to retrieve JWKS for %s (no usable cache): %s",
                    self._provider_name,
                    exc,
                )
                raise ValueError(f"Cannot retrieve JWKS: {exc}") from exc

            entry.jwks = fresh
            entry.fetched_at = now
            # Fresh key material invalidates any negative-cached kids.
            entry.unknown_kids.clear()
            return fresh

    @staticmethod
    def _negative_cache_kid(entry: _Entry, kid: str, now: float) -> None:
        """Negative-cache an unknown ``kid``, bounding the map so a spray of
        distinct kids (especially during a JWKS outage) cannot grow it without
        limit. Drops expired entries first, then evicts the soonest-expiring."""
        entry.unknown_kids[kid] = now + _negative_ttl()
        if len(entry.unknown_kids) <= MAX_UNKNOWN_KIDS:
            return
        entry.unknown_kids = {k: v for k, v in entry.unknown_kids.items() if v > now}
        overflow = len(entry.unknown_kids) - MAX_UNKNOWN_KIDS
        if overflow > 0:
            for k in sorted(entry.unknown_kids, key=entry.unknown_kids.get)[:overflow]:
                entry.unknown_kids.pop(k, None)

    def get_signing_key(self, jwks_url: str, kid: str) -> Any:
        """Resolve the signing key for ``kid``, hardening the unknown-``kid`` path.

        On the healthy/known-``kid`` path this is just "cached JWKS + lookup".
        On an unknown ``kid`` it refetches AT MOST ONCE per unknown ``kid`` per
        window (coalesced across concurrent callers) and negative-caches the
        ``kid`` so repeated random-``kid`` tokens cannot amplify into upstream
        traffic. Raises ``ValueError`` if the key cannot be resolved.
        """
        if not kid:
            raise ValueError("Token missing 'kid' in header")

        jwks = self.get_jwks(jwks_url)
        key = self._find_key(jwks, kid)
        if key is not None:
            return self._to_signing_key(key)

        entry = self._entry(jwks_url)

        # Fast path: recently negative-cached — refuse without touching upstream.
        now = time.time()
        expiry = entry.unknown_kids.get(kid)
        if expiry is not None and now < expiry:
            logger.debug(
                "Unknown kid %s for %s is negative-cached; not refetching",
                _safe_kid(kid),
                self._provider_name,
            )
            raise ValueError(f"No matching key found for kid: {kid}")

        with entry.lock:
            now = time.time()
            # Another thread may have refetched fresh keys covering this kid.
            key = self._find_key(entry.jwks, kid)
            if key is not None:
                entry.unknown_kids.pop(kid, None)
                return self._to_signing_key(key)

            # Another thread may have just negative-cached it (coalescing).
            expiry = entry.unknown_kids.get(kid)
            if expiry is not None and now < expiry:
                raise ValueError(f"No matching key found for kid: {kid}")

            # Rate-limit unknown-kid refetches: at most one per cooldown window
            # per jwks_url, regardless of how many DISTINCT unknown kids arrive,
            # so a varied-kid spray cannot force a JWKS fetch per request or pin
            # this lock. A genuinely rotated-in kid is still picked up -- it lands
            # in entry.jwks on the next allowed refetch (or the TTL refresh in
            # get_jwks) and is then found by the fast lookup ABOVE, before this
            # negative cache is ever consulted.
            fresh = entry.jwks
            if (now - entry.last_unknown_refetch) >= _unknown_refetch_cooldown():
                entry.last_unknown_refetch = now
                logger.info(
                    "Unknown kid %s for %s; refetching JWKS (cooldown-gated)",
                    _safe_kid(kid),
                    self._provider_name,
                )
                try:
                    fresh = self._fetch(jwks_url)
                    entry.jwks = fresh
                    entry.fetched_at = now
                    # Do NOT clear unknown_kids here: rotated-in kids are resolved
                    # by the fast lookup above, and clearing would let alternating
                    # unknown kids defeat the negative cache (one fetch per kid).
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Unknown-kid refetch failed for %s: %s", self._provider_name, exc
                    )
                    fresh = entry.jwks

            key = self._find_key(fresh, kid)
            if key is not None:
                entry.unknown_kids.pop(kid, None)
                return self._to_signing_key(key)

            # Still unknown: negative-cache the kid (bounded map).
            self._negative_cache_kid(entry, kid, now)
            raise ValueError(f"No matching key found for kid: {kid}")
