"""Optional OBO exchanged-token cache.

Opt-in (``EGRESS_OBO_CACHE_ENABLED``, default off) reuse of exchanged OBO tokens
so repeated MCP calls by the same principal to the same audience reuse a
still-valid token instead of re-exchanging against the IdP on every request.

- Storage: the existing per-user ``SecretStore`` under a RESERVED
  ``auth_method="obo-cache"`` namespace -- ``is_per_user_auth_method`` returns
  False for it, so it is structurally invisible to the connections API / vend /
  PAT gates, and it lives in a distinct principal prefix that cannot collide
  with real PAT/3LO entries.
- Key: the immutable ``egress_user`` (OIDC ``sub``) + IdP kind + target audience
  + a full-SHA-256 scope digest. A token minted for one
  ``(egress_user, idp, audience, scope-set)`` can never be served for another.
  An empty ``egress_user`` bypasses the cache entirely (fail-closed; no mutable
  ``preferred_username`` fallback).
- Single-flight: a cross-replica holder-fenced lease collapses a cold-miss herd
  to ONE exchange; non-acquirers wait and re-read. The lease is an optimization,
  not a gate -- a lease-backend error degrades to exchange-without-single-flight
  and never blocks a request the stateless path would serve.
- Fail-closed: read error / near-expiry / unknown lifetime -> miss (never serve
  a stale or wrong-principal token); write error -> still return the freshly
  exchanged token; exchange error -> propagate.

Multi-IdP: the ``idp`` dimension (entra | keycloak) is carried in BOTH the lease
key and the store address, so Entra and Keycloak OBO never share a
cache entry. Today only providers with a working ``obo_exchange`` populate it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
from registry.secrets.interfaces import SecretStoreError

logger = logging.getLogger(__name__)

# Reserved auth_method: ``is_per_user_auth_method`` is False for it, so the
# connections API / vend / PAT gates skip it, and it occupies a distinct
# per-principal storage prefix that cannot collide with real PAT/3LO entries.
_OBO_NS_AUTH_METHOD = "obo-cache"

_POLL_INTERVAL_SECONDS = 0.075
_MAX_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class _Ident:
    key: str  # lease key (single-flight)
    server_path: str  # store address segment


def _obo_identity(egress_user: str, idp: str, target_audience: str, scopes: list[str]) -> _Ident:
    """Derive the lease key and store ``server_path`` from ONE identity tuple.

    Both encode the same ``(idp, egress_user, audience, scope-digest)``, so a
    waiter polls exactly the ``server_path`` the acquirer populates (single-flight
    is never silently bypassed). The scope digest is the FULL SHA-256 (hygiene).
    """
    norm = sorted({s for s in (scopes or []) if s})
    digest = hashlib.sha256("\x1f".join(norm).encode()).hexdigest()
    server_path = f"{target_audience}|{digest}"
    key = f"obo|{idp}|{egress_user}|{server_path}"
    return _Ident(key=key, server_path=server_path)


def _seconds_until(expires_at: str | None) -> float:
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

    Defense-in-depth for the OBO cache: ``get_or_exchange`` only
    ever passes ``obo-cache``, but this refuses any other ``auth_method`` in
    process, so a bug or a future caller cannot read or overwrite a user's
    PAT/3LO secret even if the backend IAM/OpenBao policy were mis-scoped. It
    complements -- does not replace -- the scoped backend policy.
    """

    def __init__(self, inner, allowed_auth_method: str) -> None:
        self._inner = inner
        self._allowed = allowed_auth_method

    def _guard(self, auth_method: str) -> None:
        if auth_method != self._allowed:
            # A clamp violation is a programming error (the cache only ever passes
            # the reserved namespace), not a store blip -- log it distinctly. The
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

    async def get_token(self, auth_method, user_id, provider, server_path):
        self._guard(auth_method)
        return await self._inner.get_token(auth_method, user_id, provider, server_path)

    async def put_token(self, auth_method, user_id, provider, server_path, token):
        self._guard(auth_method)
        return await self._inner.put_token(auth_method, user_id, provider, server_path, token)


class OboTokenCache:
    """Opt-in cache wrapping ``obo_exchange`` (see module docstring)."""

    def __init__(
        self,
        store,
        lease_manager,
        *,
        enabled: bool,
        max_ttl_s: int,
        skew_s: int,
        get_timeout_s: float = 2.0,
        wait_total_s: float | None = None,
    ) -> None:
        # Capability-scoped handle: clamp the store to the reserved obo-cache namespace
        # so a bug or a future caller can never reach a user's PAT/3LO secrets in-process,
        # independent of the backend IAM/OpenBao policy.
        self._store = (
            _NamespacedSecretStore(store, _OBO_NS_AUTH_METHOD) if store is not None else None
        )
        self._lease = lease_manager
        self._enabled = enabled
        self._max_ttl_s = max_ttl_s
        self._skew_s = skew_s
        self._get_timeout_s = get_timeout_s
        # Lease TTL must cover the acquirer's WHOLE hold: the pre-exchange re-read
        # (get_timeout) + the bounded exchange (exchange_timeout, wall-clock capped
        # in _exchange) + the put (get_timeout), plus margin. DECOUPLED from skew
        # (skew=0 stays safe).
        self._lease_ttl_s = int(2 * self._get_timeout_s + _TOKEN_EXCHANGE_TIMEOUT_SECONDS) + 5
        # A waiter MUST outlast a dead lease-holder (lease_ttl) plus one exchange of
        # headroom -- else a slow IdP or a crashed holder makes every waiter give up
        # and stampede the IdP, defeating single-flight exactly when it is needed.
        # Invariant: get/put_timeout < exchange_timeout < lease_ttl < wait_total_s.
        self._wait_total_s = wait_total_s or (
            self._lease_ttl_s + int(_TOKEN_EXCHANGE_TIMEOUT_SECONDS)
        )
        # Globally unique across replica processes -- the holder-fence anchor.
        self._holder = uuid.uuid4().hex

    async def get_or_exchange(
        self,
        *,
        idp_provider,
        subject_token: str,
        egress_user: str,
        target_audience: str,
        scopes: list[str] | None,
    ) -> str:
        scopes = list(scopes or [])
        if not self._enabled or not egress_user:
            # Disabled or no stable principal -> never cache (fail-closed).
            token, _ = await obo_exchange(idp_provider, subject_token, target_audience, scopes)
            return token

        idp = _idp_kind(idp_provider)
        ident = _obo_identity(egress_user, idp, target_audience, scopes)

        tok = await self._safe_get(egress_user, idp, ident.server_path)
        if tok:
            meters.obo_cache_hit.add(1, {"idp": idp})
            return tok.access_token
        meters.obo_cache_miss.add(1, {"idp": idp})

        deadline = time.monotonic() + self._wait_total_s
        poll = _POLL_INTERVAL_SECONDS
        while True:
            # Re-read FIRST each pass: a peer may have populated the cache, in
            # which case we issue NO lease write (avoids herd contention on the
            # single lease doc).
            tok = await self._safe_get(egress_user, idp, ident.server_path)
            if tok:
                meters.obo_cache_hit.add(1, {"idp": idp})
                return tok.access_token
            try:
                acquired = await self._lease.acquire(ident.key, self._holder, self._lease_ttl_s)
            except Exception:  # noqa: BLE001 - lease backend down -> fail-open on the
                # optimization, NEVER block the token the stateless path would serve.
                return await self._exchange_and_store(
                    idp_provider,
                    subject_token,
                    target_audience,
                    scopes,
                    idp,
                    egress_user,
                    ident.server_path,
                    store=False,
                )
            if acquired:
                try:
                    tok = await self._safe_get(egress_user, idp, ident.server_path)
                    if tok:
                        meters.obo_cache_hit.add(1, {"idp": idp})
                        return tok.access_token
                    return await self._exchange_and_store(
                        idp_provider,
                        subject_token,
                        target_audience,
                        scopes,
                        idp,
                        egress_user,
                        ident.server_path,
                        store=True,
                    )
                finally:
                    try:
                        await self._lease.release(ident.key, self._holder)
                    except Exception:  # noqa: BLE001 - never mask the token result
                        logger.debug("obo_cache: lease release failed", exc_info=True)
            # A peer holds the lease and is exchanging. Wait, then loop (re-read
            # first). A fast peer failure releases the holder-fenced lease, so the
            # next pass's acquire promotes this waiter.
            if time.monotonic() >= deadline:
                # Peer slow/failed -> own bounded exchange (no store, to avoid a race).
                return await self._exchange_and_store(
                    idp_provider,
                    subject_token,
                    target_audience,
                    scopes,
                    idp,
                    egress_user,
                    ident.server_path,
                    store=False,
                )
            # Exponential backoff so a herd waiting on a slow holder does not hammer
            # the SecretStore (re-read every pass) at a fixed 75ms cadence.
            await asyncio.sleep(poll)
            poll = min(poll * 2, _MAX_POLL_INTERVAL_SECONDS)

    async def _exchange_and_store(
        self,
        idp_provider,
        subject_token,
        target_audience,
        scopes,
        idp,
        egress_user,
        server_path,
        *,
        store,
    ) -> str:
        token, expires_in = await self._exchange(
            idp_provider, subject_token, target_audience, scopes, idp
        )
        # Cache ONLY with a known, positive lifetime beyond the read-skew; else the
        # entry would be born unreadable (fail-closed on opaque/short tokens).
        if store:
            if expires_in is not None and expires_in > self._skew_s:
                await self._safe_put(
                    egress_user,
                    idp,
                    server_path,
                    token,
                    scopes,
                    expires_in,
                    getattr(idp_provider, "client_id", None) or None,
                )
            else:
                # Exchanged but the IdP returned no usable lifetime -> cannot cache.
                # Distinct from a normal miss so a permanently-uncacheable IdP is
                # observable (hit ratio would otherwise sit silently at 0).
                meters.obo_cache_unstorable.add(1, {"idp": idp})
        return token

    async def _exchange(self, idp_provider, subject_token, target_audience, scopes, idp):
        try:
            # Outer wall-clock bound: obo_exchange only sets httpx PER-PHASE timeouts
            # (connect/read/write), so without this the acquirer could hold the lease
            # past its TTL and let a waiter fire a duplicate exchange.
            token, expires_in = await asyncio.wait_for(
                obo_exchange(idp_provider, subject_token, target_audience, scopes),
                timeout=_TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            meters.obo_exchange_failure.add(1, {"idp": idp})
            raise OboExchangeError("OBO token exchange timed out") from exc
        except OboExchangeError:
            meters.obo_exchange_failure.add(1, {"idp": idp})
            raise
        meters.obo_exchange_performed.add(1, {"idp": idp})
        return token, expires_in

    async def _safe_get(self, egress_user, idp, server_path):
        try:
            tok = await asyncio.wait_for(
                self._store.get_token(_OBO_NS_AUTH_METHOD, egress_user, idp, server_path),
                timeout=self._get_timeout_s,
            )
        except (SecretStoreError, TimeoutError):
            meters.obo_cache_store_error.add(1, {"op": "get"})
            return None  # fail-closed: a store error/blip is a miss, never a stale serve
        except Exception:  # noqa: BLE001 - any store fault is a miss, never fatal
            meters.obo_cache_store_error.add(1, {"op": "get"})
            logger.debug("obo_cache: get_token failed", exc_info=True)
            return None
        # Skew applied ONCE, here (read side): within skew of expiry -> miss.
        if not tok or _seconds_until(tok.expires_at) <= self._skew_s:
            return None
        return tok

    async def _safe_put(self, egress_user, idp, server_path, token, scopes, expires_in, client_id):
        now = datetime.now(UTC)
        # Store the RAW usable window capped by MAX_TTL(+skew); the read applies
        # skew, so skew is subtracted exactly once (no born-unreadable entries).
        reuse_cap = min(int(expires_in), self._max_ttl_s + self._skew_s)
        stored = StoredToken(
            access_token=token,
            expires_at=(now + timedelta(seconds=reuse_cap)).isoformat(),
            scopes=sorted({s for s in (scopes or []) if s}),
            client_id=client_id,
            created_at=now.isoformat(),
            last_refreshed_at=now.isoformat(),
        )
        try:
            # Timeout-bounded like _safe_get so the lease is held for at most
            # exchange + ~2s put, comfortably under lease_ttl.
            await asyncio.wait_for(
                self._store.put_token(_OBO_NS_AUTH_METHOD, egress_user, idp, server_path, stored),
                timeout=self._get_timeout_s,
            )
        except Exception:  # noqa: BLE001 - write failure degrades to no-cache; the
            # freshly exchanged token is still returned to the caller.
            meters.obo_cache_store_error.add(1, {"op": "put"})
            logger.debug("obo_cache: put_token failed", exc_info=True)
