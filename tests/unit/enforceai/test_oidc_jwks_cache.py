"""
Unit tests for Stage 3.3 JWKS fetch + in-memory cache behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from auth_server.enforceai.config import OIDCIssuerConfig
from auth_server.enforceai.errors import DependencyUnavailableError
from auth_server.enforceai.oidc.jwks import JWKSCache


def _issuer_config(
    *,
    jwks_uri: str = "https://issuer.example/jwks.json",
    audiences: list[str] | None = None,
    ttl_seconds: int = 60,
) -> OIDCIssuerConfig:
    return OIDCIssuerConfig.model_validate(
        {
            "jwks_uri": jwks_uri,
            "audiences": audiences or ["mcp-registry"],
            "jwks_cache_ttl_seconds": ttl_seconds,
        }
    )


@pytest.mark.unit
class TestJWKSCache:
    async def test_first_request_fetches_and_caches(self):
        calls: list[str] = []

        async def fetcher(uri: str) -> dict[str, Any]:
            calls.append(uri)
            return {"keys": [{"kty": "RSA", "kid": "kid-1"}]}

        now_value = 100.0

        def now() -> float:
            return now_value

        cache = JWKSCache(
            fetcher=fetcher,
            now=now,
        )
        config = _issuer_config(ttl_seconds=60)

        jwks = await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )

        assert jwks["keys"][0]["kid"] == "kid-1"
        assert calls == [config.jwks_uri]

    async def test_repeated_requests_within_ttl_do_not_refetch(self):
        calls: list[str] = []

        async def fetcher(uri: str) -> dict[str, Any]:
            calls.append(uri)
            return {"keys": [{"kty": "RSA", "kid": "kid-1"}]}

        now_value = 100.0

        def now() -> float:
            return now_value

        cache = JWKSCache(
            fetcher=fetcher,
            now=now,
        )
        config = _issuer_config(ttl_seconds=60)

        await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )
        now_value = 120.0
        await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )

        assert calls == [config.jwks_uri]

    async def test_ttl_expiry_triggers_refresh(self):
        call_count = 0

        async def fetcher(uri: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"keys": [{"kty": "RSA", "kid": f"kid-{call_count}"}]}

        now_value = 100.0

        def now() -> float:
            return now_value

        cache = JWKSCache(
            fetcher=fetcher,
            now=now,
        )
        config = _issuer_config(ttl_seconds=10)

        first = await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )
        now_value = 111.0
        second = await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )

        assert first["keys"][0]["kid"] == "kid-1"
        assert second["keys"][0]["kid"] == "kid-2"
        assert call_count == 2

    async def test_fetch_failure_without_cache_maps_to_dependency_failure(self):
        async def fetcher(uri: str) -> dict[str, Any]:
            raise RuntimeError("network down")

        cache = JWKSCache(fetcher=fetcher)
        config = _issuer_config()

        with pytest.raises(DependencyUnavailableError, match="Failed to fetch JWKS"):
            await cache.get_jwks(
                issuer="https://issuer.example",
                issuer_config=config,
            )

    async def test_invalid_jwks_payload_is_rejected(self):
        async def fetcher(uri: str) -> dict[str, Any]:
            return {"not_keys": []}

        cache = JWKSCache(fetcher=fetcher)
        config = _issuer_config()

        with pytest.raises(DependencyUnavailableError, match="Failed to fetch JWKS"):
            await cache.get_jwks(
                issuer="https://issuer.example",
                issuer_config=config,
            )

    async def test_refresh_is_forced_even_with_fresh_cache(self):
        call_count = 0

        async def fetcher(uri: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"keys": [{"kty": "RSA", "kid": f"kid-{call_count}"}]}

        now_value = 100.0

        def now() -> float:
            return now_value

        cache = JWKSCache(
            fetcher=fetcher,
            now=now,
        )
        config = _issuer_config(ttl_seconds=9999)

        await cache.get_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )
        refreshed = await cache.refresh_jwks(
            issuer="https://issuer.example",
            issuer_config=config,
        )

        assert call_count == 2
        assert refreshed["keys"][0]["kid"] == "kid-2"
