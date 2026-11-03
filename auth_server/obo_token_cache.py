"""Optional OBO exchanged-token cache.

Opt-in (``EGRESS_OBO_CACHE_ENABLED``, default off) reuse of exchanged OBO tokens
so repeated MCP calls by the same principal to the same audience reuse a
still-valid token instead of re-exchanging against the IdP on every request.

- Storage: the existing per-user ``SecretStore`` (egress purpose) under a
  RESERVED ``auth_method="obo-cache"`` namespace. ``is_per_user_auth_method``
  returns False for it, so the connections API, vend and PAT gates never see
  it, and it lives in a distinct principal prefix that cannot collide with real
  PAT/3LO entries.
- Key: the immutable ``egress_user`` (OIDC ``sub``) + IdP kind + target audience
  + a full-SHA-256 scope digest. A token minted for one
  ``(egress_user, idp, audience, scope-set)`` is never served for another. An
  empty ``egress_user`` bypasses the cache entirely (fail closed; no mutable
  ``preferred_username`` fallback).
- Single-flight: a cross-replica holder-fenced lease collapses a cold-miss herd
  to ONE exchange; non-acquirers wait and re-read. The lease is an optimization,
  not a gate: a lease-backend error degrades to exchange-without-single-flight
  and never blocks a request the stateless path would serve.
- Latency budget: every store and lease call is time-bounded, and the waiter
  deadline is sized so the slowest path through ``get_or_exchange`` stays under
  the nginx headroom above the upstream MCP timeout
  (``MCP_PROXY_NGINX_READ_TIMEOUT_BUFFER_SECONDS``).
- Fail closed: read error, near-expiry or unknown lifetime is a miss (never a
  stale or wrong-principal token); write error still returns the freshly
  exchanged token; exchange error propagates.

Multi-IdP: the ``idp`` dimension (entra | keycloak) is carried in BOTH the lease
key and the store address, so Entra and Keycloak OBO never share a cache entry.
Today only providers with a working ``obo_exchange`` populate it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

from pymongo.errors import PyMongoError

try:
    from egress_obo import (
        _TOKEN_EXCHANGE_TIMEOUT_SECONDS,
        OboExchangeError,
        _idp_kind,
        obo_exchange,
    )
    from observability import meters
except ImportError:
    from auth_server.egress_obo import (
        _TOKEN_EXCHANGE_TIMEOUT_SECONDS,
        OboExchangeError,
        _idp_kind,
        obo_exchange,
    )
    from auth_server.observability import meters
from registry.egress_auth.schemas import StoredToken
from registry.secrets.interfaces import SecretStoreBase, SecretStoreError
from registry.secrets.keys import EGRESS_PURPOSE

if TYPE_CHECKING:
    from registry.egress_auth.service import LeaseManager

logger = logging.getLogger(__name__)

# Reserved auth_method: ``is_per_user_auth_method`` is False for it, so the
# connections API / vend / PAT gates skip it, and it occupies a distinct
# per-principal storage prefix that cannot collide with real PAT/3LO entries.
_OBO_NS_AUTH_METHOD: str = "obo-cache"

# Cached entries live in the egress address space. The egress purpose keeps the
# vault prefix unchanged, which is what the scoped IAM resource and OpenBao
# policy path (``<prefix>/b2JvLWNhY2hl/*``) are written against.
_OBO_CACHE_PURPOSE: str = EGRESS_PURPOSE

_POLL_INTERVAL_SECONDS: float = 0.075
_MAX_POLL_INTERVAL_SECONDS: float = 1.0

# Bounds on each store / lease round trip. A timeout is a miss (read) or a skipped
# write, never a failed request.
_STORE_READ_TIMEOUT_SECONDS: float = 1.0
_STORE_WRITE_TIMEOUT_SECONDS: float = 2.0
_LEASE_OP_TIMEOUT_SECONDS: float = 1.0

# Extra lease lifetime beyond the holder's bounded hold (read + exchange + write).
_LEASE_MARGIN_SECONDS: int = 2

# How long a non-holder waits for the holder to publish before running its own
# exchange. Sized so the slowest path (initial read + wait + one more read and
# acquire + a full holder hold) fits the 30s nginx headroom above the upstream
# MCP timeout; see test_worst_case_latency_fits_nginx_headroom. A holder that
# crashes mid-hold keeps its lease for the lease TTL, which outlasts this wait,
# so waiters then exchange on their own: the same IdP load as the cache being off.
_WAIT_TOTAL_SECONDS: float = 11.0


class OboToken(NamedTuple):
    """An OBO access token and whether it came from the cache."""

    access_token: str
    from_cache: bool


@dataclass(frozen=True)
class _Ident:
    key: str  # lease key (single-flight)
    server_path: str  # store address segment


@dataclass(frozen=True)
class _ExchangeRequest:
    """Everything one ``get_or_exchange`` call needs, bundled once."""

    idp_provider: object
    subject_token: str
    target_audience: str
    scopes: list[str]
    idp: str
    egress_user: str
    ident: _Ident


def _obo_identity(
    egress_user: str,
    idp: str,
    target_audience: str,
    scopes: list[str],
) -> _Ident:
    """Derive the lease key and store ``server_path`` from ONE identity tuple.

    Both encode the same ``(idp, egress_user, audience, scope-digest)``, so a
    waiter polls exactly the ``server_path`` the acquirer populates (single-flight
    is never silently bypassed). The scope digest is the FULL SHA-256.
    """
    norm = sorted({s for s in (scopes or []) if s})
    digest = hashlib.sha256("\x1f".join(norm).encode()).hexdigest()
    server_path = f"{target_audience}|{digest}"
    key = f"obo|{idp}|{egress_user}|{server_path}"
    return _Ident(key=key, server_path=server_path)


def _seconds_until(
    expires_at: str | None,
) -> float:
    """Seconds until an ISO8601 ``expires_at``; 0.0 when absent/unparseable."""
    if not expires_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(expires_at)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - datetime.now(UTC)).total_seconds()


class _NamespacedSecretStore:
    """A SecretStore facade clamped to a single reserved ``auth_method``.

    Defense in depth for the OBO cache: ``get_or_exchange`` only ever passes
    ``obo-cache``, but this refuses any other ``auth_method`` in process, so a bug
    or a future caller cannot read or overwrite a user's PAT/3LO secret even if
    the backend IAM/OpenBao policy were mis-scoped. It complements the scoped
    backend policy and does not replace it.
    """

    def __init__(
        self,
        inner: SecretStoreBase,
        allowed_auth_method: str,
    ) -> None:
        self._inner = inner
        self._allowed = allowed_auth_method

    def _guard(
        self,
        auth_method: str,
    ) -> None:
        if auth_method != self._allowed:
            # A clamp violation is a programming error (the cache only ever passes
            # the reserved namespace), not a store blip, so log it distinctly. The
            # callers (_safe_get/_safe_put) still fail closed on the raised error.
            logger.error(
                "obo_cache: refused foreign auth_method=%r (namespaced to %r)",
                auth_method,
                self._allowed,
            )
            raise SecretStoreError(
                f"OBO cache store is namespaced to {self._allowed!r}; "
                f"refusing auth_method={auth_method!r}"
            )

    async def get_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        self._guard(auth_method)
        return await self._inner.get_token(
            auth_method, user_id, provider, server_path, purpose=_OBO_CACHE_PURPOSE
        )

    async def put_token(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        self._guard(auth_method)
        await self._inner.put_token(
            auth_method, user_id, provider, server_path, token, purpose=_OBO_CACHE_PURPOSE
        )


class OboTokenCache:
    """Opt-in cache wrapping ``obo_exchange`` (see module docstring).

    Constructed only when ``EGRESS_OBO_CACHE_ENABLED`` is true; with the flag off
    the auth-server holds no instance and the OBO hop exchanges statelessly.
    """

    def __init__(
        self,
        store: SecretStoreBase,
        lease_manager: LeaseManager,
        *,
        max_ttl_s: int,
        skew_s: int,
        get_timeout_s: float = _STORE_READ_TIMEOUT_SECONDS,
        put_timeout_s: float = _STORE_WRITE_TIMEOUT_SECONDS,
        lease_timeout_s: float = _LEASE_OP_TIMEOUT_SECONDS,
        wait_total_s: float = _WAIT_TOTAL_SECONDS,
    ) -> None:
        # Capability-scoped handle: clamp the store to the reserved obo-cache
        # namespace so a bug or a future caller can never reach a user's PAT/3LO
        # secrets in process, independent of the backend IAM/OpenBao policy.
        self._store = _NamespacedSecretStore(store, _OBO_NS_AUTH_METHOD)
        self._lease = lease_manager
        self._max_ttl_s = max_ttl_s
        self._skew_s = skew_s
        self._get_timeout_s = get_timeout_s
        self._put_timeout_s = put_timeout_s
        self._lease_timeout_s = lease_timeout_s
        self._wait_total_s = wait_total_s
        # The lease must outlive the holder's whole bounded hold (re-read +
        # exchange, wall-clock capped in _exchange, + write), or a waiter could
        # acquire mid-hold and run a duplicate exchange. Independent of skew.
        self._lease_ttl_s = (
            int(self._get_timeout_s + _TOKEN_EXCHANGE_TIMEOUT_SECONDS + self._put_timeout_s)
            + _LEASE_MARGIN_SECONDS
        )
        # Globally unique across replica processes: the holder-fence anchor.
        self._holder = uuid.uuid4().hex

    async def get_or_exchange(
        self,
        *,
        idp_provider: object,
        subject_token: str,
        egress_user: str,
        target_audience: str,
        scopes: list[str] | None,
    ) -> OboToken:
        """Return a cached OBO token for this identity, or exchange and cache one.

        Args:
            idp_provider: The gateway's IdP provider object (passed to ``obo_exchange``).
            subject_token: The verified ingress JWT to exchange.
            egress_user: The immutable per-user vault id; empty bypasses the cache.
            target_audience: The registry-derived audience of the MCP server.
            scopes: Requested scopes for the exchanged token.

        Returns:
            The access token and whether it was served from the cache.

        Raises:
            OboExchangeError: A live exchange failed or timed out.
        """
        scope_list = list(scopes or [])
        if not egress_user:
            # No stable principal: never cache (fail closed).
            token, _ = await obo_exchange(idp_provider, subject_token, target_audience, scope_list)
            return OboToken(token, from_cache=False)

        idp = _idp_kind(idp_provider)
        req = _ExchangeRequest(
            idp_provider=idp_provider,
            subject_token=subject_token,
            target_audience=target_audience,
            scopes=scope_list,
            idp=idp,
            egress_user=egress_user,
            ident=_obo_identity(egress_user, idp, target_audience, scope_list),
        )
        hit = await self._cached(req)
        if hit is not None:
            return hit
        meters.obo_cache_miss.add(1, {"idp": idp})
        return await self._wait_or_lead(req)

    async def _wait_or_lead(
        self,
        req: _ExchangeRequest,
    ) -> OboToken:
        """Single-flight loop: lead the exchange if the lease is free, else wait."""
        deadline = time.monotonic() + self._wait_total_s
        poll = _POLL_INTERVAL_SECONDS
        while True:
            # Re-read FIRST each pass: a peer may have populated the cache, in
            # which case we issue NO lease write (avoids herd contention on the
            # single lease doc).
            hit = await self._cached(req)
            if hit is not None:
                return hit
            acquired = await self._try_acquire(req.ident.key, deadline)
            if acquired is None:
                # Lease backend unavailable: exchange without single-flight and
                # without storing (no fence against a concurrent writer).
                return await self._exchange_and_store(req, store=False)
            if acquired:
                return await self._lead(req)
            # A peer holds the lease and is exchanging. A fast peer failure
            # releases the holder-fenced lease, so a later pass promotes us.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Peer slow or gone: run our own bounded exchange (no store, to
                # avoid racing the holder's write).
                return await self._exchange_and_store(req, store=False)
            # Exponential backoff so a herd waiting on a slow holder does not
            # hammer the SecretStore; never sleep past the deadline.
            await asyncio.sleep(min(poll, remaining))
            poll = min(poll * 2, _MAX_POLL_INTERVAL_SECONDS)

    async def _lead(
        self,
        req: _ExchangeRequest,
    ) -> OboToken:
        """Holder path: re-read under the lease, else exchange and publish."""
        try:
            hit = await self._cached(req)
            if hit is not None:
                return hit
            return await self._exchange_and_store(req, store=True)
        finally:
            await self._release(req.ident.key)

    async def _cached(
        self,
        req: _ExchangeRequest,
    ) -> OboToken | None:
        tok = await self._safe_get(req.egress_user, req.idp, req.ident.server_path)
        if tok is None:
            return None
        meters.obo_cache_hit.add(1, {"idp": req.idp})
        return OboToken(tok.access_token, from_cache=True)

    async def _try_acquire(
        self,
        key: str,
        deadline: float,
    ) -> bool | None:
        """Acquire the single-flight lease; None when the lease backend failed.

        The Mongo lease answers at once, but the in-process fallback blocks until
        the holder releases. So the call may take until the waiter's deadline (at
        least ``lease_timeout_s``), and running out of time means "not acquired",
        not a backend failure.
        """
        timeout = max(deadline - time.monotonic(), self._lease_timeout_s)
        try:
            return await asyncio.wait_for(
                self._lease.acquire(key, self._holder, self._lease_ttl_s),
                timeout=timeout,
            )
        except TimeoutError:
            logger.debug("obo_cache: lease not acquired before the wait deadline")
            return False
        except PyMongoError:
            meters.obo_cache_store_error.add(1, {"op": "lease", "kind": "backend"})
            logger.warning("obo_cache: lease acquire failed; exchanging without single-flight")
            return None
        except Exception:
            meters.obo_cache_store_error.add(1, {"op": "lease", "kind": "unexpected"})
            logger.exception("obo_cache: unexpected lease acquire error")
            return None

    async def _release(
        self,
        key: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._lease.release(key, self._holder),
                timeout=self._lease_timeout_s,
            )
        except (PyMongoError, TimeoutError):
            # The holder-fenced lease expires on its own; never mask the token.
            logger.warning("obo_cache: lease release failed; lease will expire")
        except Exception:
            logger.exception("obo_cache: unexpected lease release error")

    async def _exchange_and_store(
        self,
        req: _ExchangeRequest,
        *,
        store: bool,
    ) -> OboToken:
        token, expires_in = await self._exchange(req)
        if not store:
            return OboToken(token, from_cache=False)
        # Cache ONLY with a known, positive lifetime beyond the read skew; else the
        # entry would be born unreadable (fail closed on opaque/short tokens).
        if expires_in is not None and expires_in > self._skew_s:
            await self._safe_put(req, token, expires_in)
        else:
            # Exchanged but the IdP returned no usable lifetime, so it cannot be
            # cached. Distinct from a normal miss so a permanently uncacheable IdP
            # is observable (the hit ratio would otherwise sit silently at 0).
            meters.obo_cache_unstorable.add(1, {"idp": req.idp})
        return OboToken(token, from_cache=False)

    async def _exchange(
        self,
        req: _ExchangeRequest,
    ) -> tuple[str, int | None]:
        try:
            # Outer wall-clock bound: obo_exchange only sets httpx PER-PHASE timeouts
            # (connect/read/write), so without this the holder could outlive its
            # lease TTL and let a waiter fire a duplicate exchange.
            token, expires_in = await asyncio.wait_for(
                obo_exchange(req.idp_provider, req.subject_token, req.target_audience, req.scopes),
                timeout=_TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            meters.obo_exchange_failure.add(1, {"idp": req.idp})
            raise OboExchangeError("OBO token exchange timed out") from exc
        except OboExchangeError:
            meters.obo_exchange_failure.add(1, {"idp": req.idp})
            raise
        meters.obo_exchange_performed.add(1, {"idp": req.idp})
        return token, expires_in

    async def _safe_get(
        self,
        egress_user: str,
        idp: str,
        server_path: str,
    ) -> StoredToken | None:
        try:
            tok = await asyncio.wait_for(
                self._store.get_token(_OBO_NS_AUTH_METHOD, egress_user, idp, server_path),
                timeout=self._get_timeout_s,
            )
        except (SecretStoreError, TimeoutError):
            # Fail closed: a store error or slow read is a miss, never a stale serve.
            meters.obo_cache_store_error.add(1, {"op": "get", "kind": "backend"})
            logger.debug("obo_cache: get_token failed", exc_info=True)
            return None
        except Exception:
            # Not a store blip: a programming or contract error. Still a miss so the
            # OBO hop keeps working, but loud enough to be noticed.
            meters.obo_cache_store_error.add(1, {"op": "get", "kind": "unexpected"})
            logger.exception("obo_cache: unexpected get_token error")
            return None
        # Skew applied ONCE, here (read side): within skew of expiry is a miss.
        if not tok or _seconds_until(tok.expires_at) <= self._skew_s:
            return None
        return tok

    async def _safe_put(
        self,
        req: _ExchangeRequest,
        token: str,
        expires_in: int,
    ) -> None:
        now = datetime.now(UTC)
        # Store the RAW usable window capped by MAX_TTL(+skew); the read applies
        # skew, so skew is subtracted exactly once (no born-unreadable entries).
        reuse_cap = min(int(expires_in), self._max_ttl_s + self._skew_s)
        stored = StoredToken(
            access_token=token,
            expires_at=(now + timedelta(seconds=reuse_cap)).isoformat(),
            scopes=sorted({s for s in req.scopes if s}),
            client_id=getattr(req.idp_provider, "client_id", None) or None,
            created_at=now.isoformat(),
            last_refreshed_at=now.isoformat(),
        )
        try:
            await asyncio.wait_for(
                self._store.put_token(
                    _OBO_NS_AUTH_METHOD, req.egress_user, req.idp, req.ident.server_path, stored
                ),
                timeout=self._put_timeout_s,
            )
        except (SecretStoreError, TimeoutError):
            # Write failure degrades to no-cache; the fresh token is still returned.
            meters.obo_cache_store_error.add(1, {"op": "put", "kind": "backend"})
            logger.debug("obo_cache: put_token failed", exc_info=True)
        except Exception:
            meters.obo_cache_store_error.add(1, {"op": "put", "kind": "unexpected"})
            logger.exception("obo_cache: unexpected put_token error")
