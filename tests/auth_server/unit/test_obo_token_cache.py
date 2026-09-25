"""Unit tests for the optional OBO exchanged-token cache.

Covers the LLD's key proofs: default-off delegates without touching the store,
keying on the immutable ``egress_user`` under a reserved namespace, full-digest
multi-IdP scope normalization, ``expires_in`` gating + TTL clamp + read-skew,
fail-closed on every store/lease error path, lease TTL decoupled from skew, and
-- the round-1 regression guard -- a concurrent cold-miss herd collapses to
exactly one exchange.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from auth_server import obo_token_cache as otc
from auth_server.egress_obo import _TOKEN_EXCHANGE_TIMEOUT_SECONDS, OboExchangeError
from auth_server.obo_token_cache import OboTokenCache, _obo_identity
from registry.egress_auth.schemas import StoredToken
from registry.egress_auth.service import is_per_user_auth_method
from registry.secrets.interfaces import SecretStoreError


class _EntraProvider:
    client_id = "gw-client"


class _FakeStore:
    def __init__(self):
        self.data: dict = {}
        self.get_calls = 0
        self.put_calls = 0
        self.delete_calls = 0
        self.get_error: Exception | None = None
        self.put_error: Exception | None = None
        self.get_delay = 0.0

    async def get_token(self, auth_method, user_id, provider, server_path):
        self.get_calls += 1
        if self.get_delay:
            await asyncio.sleep(self.get_delay)
        if self.get_error:
            raise self.get_error
        return self.data.get((auth_method, user_id, provider, server_path))

    async def put_token(self, auth_method, user_id, provider, server_path, token):
        self.put_calls += 1
        if self.put_error:
            raise self.put_error
        self.data[(auth_method, user_id, provider, server_path)] = token

    async def delete_token(self, auth_method, user_id, provider, server_path):
        self.delete_calls += 1
        self.data.pop((auth_method, user_id, provider, server_path), None)


class _FakeLease:
    """Holder-fenced single-flight lease; ``acquire`` is atomic (no await point
    between the held-check and the add), so exactly one coroutine wins per key."""

    def __init__(self):
        self._held: set = set()
        self.acquire_error: Exception | None = None

    async def acquire(self, key, holder, ttl_seconds):
        if self.acquire_error:
            raise self.acquire_error
        if key in self._held:
            return False
        self._held.add(key)
        return True

    async def release(self, key, holder):
        self._held.discard(key)


class _TtlLease:
    """Simulates a CRASHED lease-holder: the lease is 'held' by a dead holder
    (``acquire`` -> False) for the first ``held_polls`` attempts (its TTL window),
    then reclaimable by exactly ONE caller, and held by that new owner thereafter.
    Exercises promotion-not-stampede without waiting a real TTL."""

    def __init__(self, held_polls: int):
        self._held_polls = held_polls
        self._attempts = 0
        self._held_keys: set = set()  # keyed on the lease key (like the real lease)
        self._reclaimed = False  # the dead lease is reclaimable exactly ONCE

    async def acquire(self, key, holder, ttl_seconds):
        if key in self._held_keys:
            return False  # currently held (by the reclaiming waiter)
        if self._reclaimed:
            # Already reclaimed once and released after storing -> the token is
            # now cached, so waiters must re-read a HIT, never re-exchange.
            return False
        self._attempts += 1
        if self._attempts <= self._held_polls:
            return False  # dead holder's lease has not yet expired
        self._held_keys.add(key)  # TTL elapsed -> reclaimed by exactly this caller
        self._reclaimed = True
        return True

    async def release(self, key, holder):
        self._held_keys.discard(key)


def _patch_exchange(monkeypatch, *, token="tok", expires_in=3600, raises=None, delay=0.0):
    calls = {"n": 0}

    async def _exchange(idp_provider, subject_token, target_audience, scopes=None):
        calls["n"] += 1
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return token, expires_in

    monkeypatch.setattr(otc, "obo_exchange", _exchange)
    return calls


def _cache(store, lease, *, enabled=True, max_ttl_s=300, skew_s=30, **kw):
    return OboTokenCache(store, lease, enabled=enabled, max_ttl_s=max_ttl_s, skew_s=skew_s, **kw)


def _call(**overrides):
    base = {
        "idp_provider": _EntraProvider(),
        "subject_token": "ingress-jwt",
        "egress_user": "user-sub-1",
        "target_audience": "api://srv",
        "scopes": ["s1"],
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestOboTokenCache:
    @pytest.mark.asyncio
    async def test_disabled_delegates_no_store(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        tok = await _cache(store, lease, enabled=False).get_or_exchange(**_call())
        assert tok == "tok"
        assert calls["n"] == 1
        assert store.get_calls == 0 and store.put_calls == 0

    @pytest.mark.asyncio
    async def test_empty_egress_user_bypasses_cache(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        await _cache(store, lease).get_or_exchange(**_call(egress_user=""))
        assert calls["n"] == 1
        assert store.get_calls == 0 and store.put_calls == 0

    @pytest.mark.asyncio
    async def test_hit_reuses_no_second_exchange(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        cache = _cache(store, lease)
        a = await cache.get_or_exchange(**_call())
        b = await cache.get_or_exchange(**_call())
        assert a == b == "tok"
        assert calls["n"] == 1  # second call was a cache hit
        assert store.put_calls == 1

    @pytest.mark.asyncio
    async def test_keys_on_egress_user(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        cache = _cache(store, lease)
        await cache.get_or_exchange(**_call())
        # A different egress_user (same audience/scopes) must NOT hit the entry.
        await cache.get_or_exchange(**_call(egress_user="user-sub-2"))
        assert calls["n"] == 2
        assert all(k[0] == otc._OBO_NS_AUTH_METHOD for k in store.data)
        assert {k[1] for k in store.data} == {"user-sub-1", "user-sub-2"}

    @pytest.mark.asyncio
    async def test_namespace_reserved_and_invisible(self, monkeypatch):
        _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        await _cache(store, lease).get_or_exchange(**_call())
        assert is_per_user_auth_method(otc._OBO_NS_AUTH_METHOD) is False
        assert all(k[0] == "obo-cache" for k in store.data)

    def test_scope_digest_normalization_multi_idp(self):
        a = _obo_identity("u", "entra", "api://srv", ["a", "b"])
        b = _obo_identity("u", "entra", "api://srv", ["b", "a", "b"])
        assert a == b  # order + duplicates normalized to the same identity
        # idp is carried in the lease key (multi-IdP: entra vs keycloak differ).
        kc = _obo_identity("u", "keycloak", "api://srv", ["a", "b"])
        assert kc.key != a.key
        assert "entra" in a.key and "keycloak" in kc.key
        # full 256-bit hex digest (64 hex chars) in the server_path segment
        assert len(a.server_path.split("|")[-1]) == 64

    @pytest.mark.asyncio
    async def test_unknown_expires_in_not_cached(self, monkeypatch):
        calls = _patch_exchange(monkeypatch, expires_in=None)
        store, lease = _FakeStore(), _FakeLease()
        cache = _cache(store, lease)
        await cache.get_or_exchange(**_call())
        await cache.get_or_exchange(**_call())
        assert calls["n"] == 2  # opaque lifetime -> never cached -> exchange each time
        assert store.put_calls == 0

    @pytest.mark.asyncio
    async def test_born_expired_not_cached(self, monkeypatch):
        # expires_in <= skew -> the entry would be unreadable; skip caching.
        calls = _patch_exchange(monkeypatch, expires_in=20)
        store, lease = _FakeStore(), _FakeLease()
        await _cache(store, lease, skew_s=30).get_or_exchange(**_call())
        assert calls["n"] == 1
        assert store.put_calls == 0

    @pytest.mark.asyncio
    async def test_ttl_clamped_to_max_plus_skew(self, monkeypatch):
        _patch_exchange(monkeypatch, expires_in=100000)
        store, lease = _FakeStore(), _FakeLease()
        await _cache(store, lease, max_ttl_s=300, skew_s=30).get_or_exchange(**_call())
        (stored,) = list(store.data.values())
        secs = otc._seconds_until(stored.expires_at)
        assert 320 <= secs <= 331  # capped at max_ttl + skew (330), not raw 100000

    @pytest.mark.asyncio
    async def test_near_expiry_is_miss_no_delete(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        ident = _obo_identity("user-sub-1", "entra", "api://srv", ["s1"])
        near = (datetime.now(UTC) + timedelta(seconds=10)).isoformat()
        store.data[(otc._OBO_NS_AUTH_METHOD, "user-sub-1", "entra", ident.server_path)] = (
            StoredToken(access_token="stale", expires_at=near)
        )
        await _cache(store, lease, skew_s=30).get_or_exchange(**_call())
        assert calls["n"] == 1  # near-expiry -> miss -> fresh exchange
        assert store.delete_calls == 0  # no delete-on-read write churn

    @pytest.mark.asyncio
    async def test_read_error_is_miss(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        store.get_error = SecretStoreError("store down")
        tok = await _cache(store, lease).get_or_exchange(**_call())
        assert tok == "tok" and calls["n"] == 1  # error == miss, exchange proceeds

    @pytest.mark.asyncio
    async def test_get_timeout_is_miss(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        store.get_delay = 0.3
        tok = await _cache(store, lease, get_timeout_s=0.05).get_or_exchange(**_call())
        assert tok == "tok" and calls["n"] == 1  # slow store -> timeout -> miss

    @pytest.mark.asyncio
    async def test_write_error_still_returns_token(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        store.put_error = SecretStoreError("write conflict")
        tok = await _cache(store, lease).get_or_exchange(**_call())
        assert tok == "tok" and calls["n"] == 1  # write failed but the token is returned

    @pytest.mark.asyncio
    async def test_lease_backend_error_degrades(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        store, lease = _FakeStore(), _FakeLease()
        lease.acquire_error = RuntimeError("mongo down")
        tok = await _cache(store, lease).get_or_exchange(**_call())
        # Exchange-without-single-flight: a lease-backend error never blocks the call.
        assert tok == "tok" and calls["n"] == 1

    @pytest.mark.asyncio
    async def test_exchange_error_propagates(self, monkeypatch):
        _patch_exchange(monkeypatch, raises=OboExchangeError("idp down"))
        store, lease = _FakeStore(), _FakeLease()
        with pytest.raises(OboExchangeError):
            await _cache(store, lease).get_or_exchange(**_call())

    def test_lease_ttl_covers_full_hold_and_beats_wait(self):
        # Lease TTL covers get + exchange + put (+margin), independent of skew, and
        # the waiter always outlasts it (promotion-not-stampede invariant).
        c0 = _cache(_FakeStore(), _FakeLease(), skew_s=0)
        assert c0._lease_ttl_s == int(2 * c0._get_timeout_s + _TOKEN_EXCHANGE_TIMEOUT_SECONDS) + 5
        assert c0._lease_ttl_s > int(_TOKEN_EXCHANGE_TIMEOUT_SECONDS)  # decoupled from skew
        assert c0._wait_total_s > c0._lease_ttl_s

    @pytest.mark.asyncio
    async def test_crashed_holder_promotes_one_waiter(self, monkeypatch):
        # A crashed holder never stores; its lease frees after TTL. Exactly ONE
        # waiter is then promoted and exchanges -- the rest re-read the hit, not
        # stampede. calls == 1 across the herd.
        calls = _patch_exchange(monkeypatch, delay=0.02)
        # wait_total_s kept above the default lease_ttl (19) to mirror the production
        # invariant; a real stampede regression fails fast (calls>1 on the first pass),
        # not by waiting this out.
        cache = _cache(_FakeStore(), _TtlLease(held_polls=5), wait_total_s=25)
        results = await asyncio.gather(*(cache.get_or_exchange(**_call()) for _ in range(15)))
        assert results == ["tok"] * 15
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_repeat_calls_one_exchange(self, monkeypatch):
        calls = _patch_exchange(monkeypatch)
        cache = _cache(_FakeStore(), _FakeLease())
        for _ in range(10):
            assert await cache.get_or_exchange(**_call()) == "tok"
        assert calls["n"] == 1  # 1 exchange, 9 hits

    @pytest.mark.asyncio
    async def test_single_flight_cold_miss_herd(self, monkeypatch):
        # THE round-1 regression guard: N concurrent cold-miss callers -> exactly
        # one exchange; the other N-1 wait-and-re-read the stored token.
        calls = _patch_exchange(monkeypatch, delay=0.05)
        cache = _cache(_FakeStore(), _FakeLease())
        results = await asyncio.gather(*(cache.get_or_exchange(**_call()) for _ in range(20)))
        assert results == ["tok"] * 20
        assert calls["n"] == 1

    def test_wait_total_exceeds_lease_ttl(self):
        # a waiter must outlast a dead lease-holder, else a slow IdP
        # or a crashed holder makes every waiter stampede the IdP.
        c = _cache(_FakeStore(), _FakeLease())
        assert c._wait_total_s > c._lease_ttl_s

    @pytest.mark.asyncio
    async def test_namespaced_store_refuses_foreign_auth_method(self):
        # Capability-scoped handle: the clamp refuses any auth_method
        # other than obo-cache and never touches the inner store for a foreign one.
        inner = _FakeStore()
        clamped = otc._NamespacedSecretStore(inner, otc._OBO_NS_AUTH_METHOD)
        await clamped.put_token(
            otc._OBO_NS_AUTH_METHOD, "u", "entra", "sp", StoredToken(access_token="t")
        )
        assert inner.put_calls == 1
        with pytest.raises(SecretStoreError):
            await clamped.get_token("oauth2", "u", "entra", "sp")
        with pytest.raises(SecretStoreError):
            await clamped.put_token(
                "self_signed", "u", "entra", "sp", StoredToken(access_token="t")
            )
        assert inner.get_calls == 0  # foreign reads never reach the inner store
