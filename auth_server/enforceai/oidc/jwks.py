from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx

from ..config import (
    OIDCIssuerConfig,
)
from ..errors import (
    DependencyUnavailableError,
)

FetchJWKSCallable = Callable[[str], Awaitable[dict[str, Any]]]
NowCallable = Callable[[], float]


def _normalize_jwks_payload(
    payload: Any,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("JWKS response must be a JSON object")

    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise ValueError("JWKS response must contain a 'keys' list")

    return payload


async def _fetch_jwks_httpx(
    jwks_uri: str,
) -> dict[str, Any]:
    timeout = httpx.Timeout(
        connect=2.0,
        read=3.0,
        write=3.0,
        pool=3.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/json"},
    ) as client:
        response = await client.get(jwks_uri)
        response.raise_for_status()
        return _normalize_jwks_payload(response.json())


@dataclass(frozen=True)
class _JWKSCacheEntry:
    jwks: dict[str, Any]
    fetched_at: float


class JWKSCache:
    """In-memory per-issuer JWKS cache with TTL and refresh.

    This cache performs network fetches only via an injected fetcher callable
    (default: httpx). Tests must inject a fake fetcher; unit tests must not
    require network access.
    """

    def __init__(
        self,
        *,
        fetcher: Optional[FetchJWKSCallable] = None,
        now: Optional[NowCallable] = None,
    ) -> None:
        self._fetcher = fetcher or _fetch_jwks_httpx
        self._now = now or time.monotonic
        self._cache: dict[str, _JWKSCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(
        self,
        issuer: str,
    ) -> asyncio.Lock:
        lock = self._locks.get(issuer)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[issuer] = lock
        return lock

    def _is_fresh(
        self,
        *,
        entry: _JWKSCacheEntry,
        ttl_seconds: int,
    ) -> bool:
        age = self._now() - entry.fetched_at
        return age >= 0 and age < ttl_seconds

    async def _refresh_jwks_locked(
        self,
        *,
        issuer: str,
        issuer_config: OIDCIssuerConfig,
    ) -> dict[str, Any]:
        try:
            jwks = await self._fetcher(issuer_config.jwks_uri)
            jwks = _normalize_jwks_payload(jwks)
        except Exception as exc:  # noqa: BLE001 - always map to dependency failure
            raise DependencyUnavailableError(
                f"Failed to fetch JWKS for issuer={issuer}",
                public_message="OIDC key set unavailable",
            ) from exc

        self._cache[issuer] = _JWKSCacheEntry(
            jwks=jwks,
            fetched_at=self._now(),
        )
        return jwks

    async def refresh_jwks(
        self,
        *,
        issuer: str,
        issuer_config: OIDCIssuerConfig,
    ) -> dict[str, Any]:
        """Force refresh JWKS for issuer and return the new payload."""

        lock = self._get_lock(issuer)
        async with lock:
            return await self._refresh_jwks_locked(
                issuer=issuer,
                issuer_config=issuer_config,
            )

    async def get_jwks(
        self,
        *,
        issuer: str,
        issuer_config: OIDCIssuerConfig,
    ) -> dict[str, Any]:
        """Return cached JWKS for issuer; refresh when expired.

        Failure behavior:
        - If no cached entry exists and fetch fails, raise `DependencyUnavailableError` (503).
        - If a cached entry exists but is expired and refresh fails, raise `DependencyUnavailableError` (fail closed).
        """

        existing = self._cache.get(issuer)
        if existing is not None and self._is_fresh(
            entry=existing,
            ttl_seconds=issuer_config.jwks_cache_ttl_seconds,
        ):
            return existing.jwks

        lock = self._get_lock(issuer)
        async with lock:
            existing_after_lock = self._cache.get(issuer)
            if existing_after_lock is not None and self._is_fresh(
                entry=existing_after_lock,
                ttl_seconds=issuer_config.jwks_cache_ttl_seconds,
            ):
                return existing_after_lock.jwks

            return await self._refresh_jwks_locked(
                issuer=issuer,
                issuer_config=issuer_config,
            )
