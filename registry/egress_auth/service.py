"""EgressAuthService -- orchestration for the per-user egress credential vault.

Ties together the provider table, the OAuth engine, the AEAD state codec, and
the SecretStore. The SecretStore is the only persistence dependency for token
material; a small ``ReplayGuard`` protocol covers single-use ``state`` nonces.

Handles here: consent-URL build, callback handling (verify/decrypt state +
account-swap guard + single-use nonce + code exchange + store), lazy-on-vend
refresh with an in-process single-flight lock, list, disconnect.

Not handled here: the cross-replica Mongo lease lock, the egress injection in
``mcp_proxy``, and the HTTP routers, which are layered on separately. The
single-flight here is in-process only; the cross-replica double-check/lease
wraps it.
"""

import asyncio
import logging
import secrets
from datetime import UTC, datetime
from typing import Protocol

from registry.egress_auth import oauth_engine
from registry.egress_auth.providers import resolve_provider
from registry.egress_auth.schemas import (
    EgressConnection,
    OAuthState,
    StoredToken,
)
from registry.egress_auth.state_codec import InvalidState, decode_state, encode_state
from registry.secrets.interfaces import SecretStoreBase

logger = logging.getLogger(__name__)

# auth_method values that represent a real per-user identity and may own a vault
# bucket (canonical values; cookie users are "oauth2", not
# "session_cookie"). The denylist below is authoritative; this set is the
# positive cross-check for the drift test.
PER_USER_AUTH_METHODS: frozenset[str] = frozenset(
    {
        "oauth2",
        "self_signed",
        "jwt",
        "boto3",
        "cognito",
        "keycloak",
        "okta",
        "auth0",
        "entra",
        "pingfederate",
    }
)
# Non-per-user callers: their sub is a constant or operator-chosen and must
# never address a per-user vault bucket. Fail-closed for unknown/empty too.
NON_PER_USER_AUTH_METHODS: frozenset[str] = frozenset({"federation-static", "network-trusted"})


class EgressAuthError(Exception):
    """Base error for egress-auth orchestration failures."""


class ConsentRequired(EgressAuthError):
    """No usable token; the caller must complete consent at ``authorize_url``."""

    def __init__(self, authorize_url: str | None = None) -> None:
        super().__init__("egress consent required")
        self.authorize_url = authorize_url


class ReplayGuard(Protocol):
    """Single-use guard for OAuth ``state`` nonces (a Mongo-TTL-backed implementation backs this).

    ``check_and_consume`` returns True if the nonce was unused (and records it),
    False if it has already been consumed (replay).
    """

    async def check_and_consume(self, nonce: str, ttl_seconds: int) -> bool: ...


class _InMemoryReplayGuard:
    """Default single-process replay guard (used in single-process runs / tests).

    A Mongo-TTL-backed implementation is swapped in for cross-replica safety.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = asyncio.Lock()

    async def check_and_consume(self, nonce: str, ttl_seconds: int) -> bool:
        async with self._lock:
            if nonce in self._seen:
                return False
            self._seen.add(nonce)
            return True


class LeaseManager(Protocol):
    """Cross-replica single-flight lease for refresh (a Mongo-backed implementation backs this).

    ``acquire`` returns True if this caller now holds the lease for ``key``;
    ``release`` drops it iff still held. The post-acquire double-check in the
    service is the correctness anchor -- the lease only prevents refresh storms.
    """

    async def acquire(self, key: str, holder: str, ttl_seconds: int) -> bool: ...
    async def release(self, key: str, holder: str) -> None: ...


class _InProcessLeaseManager:
    """Default single-process lease (used in single-process runs / tests). Per-key asyncio.Lock."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()
        self._held: set[str] = set()

    async def acquire(self, key: str, holder: str, ttl_seconds: int) -> bool:
        async with self._guard:
            lock = self._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        self._held.add(key)
        return True

    async def release(self, key: str, holder: str) -> None:
        lock = self._locks.get(key)
        if lock and lock.locked():
            self._held.discard(key)
            lock.release()


# The single per-user egress bucket. Browser consent keys the vault on this
# (the cookie session's auth_method is enforced == "oauth2"), so vend-read must
# resolve here too for one human identity. Mirror of auth_server's
# _EGRESS_CANONICAL_PER_USER_METHOD / _PER_USER_IDP_METHODS.
_EGRESS_CANONICAL_PER_USER_METHOD: str = "oauth2"
_PER_USER_IDP_METHODS: frozenset[str] = frozenset(
    {
        "oauth2",
        "session_cookie",
        "self_signed",
        "jwt",
        "boto3",
        "keycloak",
        "entra",
        "cognito",
        "okta",
        "auth0",
        "pingfederate",
    }
)


def canonical_auth_method(validation_result: dict) -> str:
    """The ONE canonical egress principal method.

    Different token FORMATS / IdP provider names denote the same kind of per-user
    principal and must all canonicalize to a single vault bucket (``oauth2``) so
    consent-write and vend-read agree. In particular a Dynamic-Client-Registration
    bearer issued directly by the IdP reports ``method == "keycloak"`` (etc.),
    which must fold into the same bucket the cookie-consent path wrote
    (``oauth2``) -- otherwise the DCR vend misses and the user loops on consent.
    NON-per-user methods (``federation-static``/``network-trusted``) and unknown
    methods pass through raw so ``is_per_user_auth_method`` still rejects them.
    Mirrors ``auth_server.server._canonical_auth_method``.
    """
    method = validation_result.get("method") or ""
    if method in _PER_USER_IDP_METHODS:
        return _EGRESS_CANONICAL_PER_USER_METHOD
    return method


def is_per_user_auth_method(auth_method: str) -> bool:
    """Denylist-first (fail-closed): only real per-user methods may vend."""
    if not auth_method or auth_method in NON_PER_USER_AUTH_METHODS:
        return False
    return auth_method in PER_USER_AUTH_METHODS


class EgressAuthService:
    """Orchestrates consent, vend, refresh, and disconnect over a SecretStore."""

    def __init__(
        self,
        secret_store: SecretStoreBase,
        callback_base_url: str,
        refresh_skew_seconds: int = 300,
        state_ttl_seconds: int = 600,
        replay_guard: ReplayGuard | None = None,
        lease_manager: LeaseManager | None = None,
        lease_ttl_seconds: int = 30,
    ) -> None:
        self._store = secret_store
        self._callback_url = callback_base_url.rstrip("/") + "/oauth2/egress/callback"
        self._skew = refresh_skew_seconds
        self._state_ttl = state_ttl_seconds
        self._replay = replay_guard or _InMemoryReplayGuard()
        self._lease = lease_manager or _InProcessLeaseManager()
        self._lease_ttl = lease_ttl_seconds
        # Stable per-process holder id for lease ownership/release.
        self._holder = f"egress-{id(self)}"

    # -- helpers -------------------------------------------------------------- #

    @staticmethod
    def _client_secret(egress_oauth: dict) -> str:
        from registry.utils.credential_encryption import decrypt_credential

        enc = egress_oauth.get("client_secret_encrypted")
        if not enc:
            raise EgressAuthError("egress_oauth.client_secret_encrypted missing")
        secret = decrypt_credential(enc)
        if secret is None:
            raise EgressAuthError("could not decrypt egress client_secret (SECRET_KEY changed?)")
        return secret

    def _is_near_expiry(self, token: StoredToken) -> bool:
        if not token.expires_at:
            return False  # no expiry info -> treat as long-lived; refresh on 401 elsewhere
        try:
            exp = datetime.fromisoformat(token.expires_at)
        except ValueError:
            return True
        remaining = (exp - datetime.now(UTC)).total_seconds()
        return remaining <= self._skew

    @staticmethod
    def _is_expired(expires_at_iso: str) -> bool:
        """True if the ISO8601 timestamp is in the past (fail-closed on malformed).

        Used by the ``pat`` sink check: an unparsable or past ``expires_at`` is
        treated as expired so a stale/garbled credential is never vended.
        """
        try:
            exp = datetime.fromisoformat(expires_at_iso)
        except ValueError:
            return True
        return (exp - datetime.now(UTC)).total_seconds() <= 0

    # -- consent -------------------------------------------------------------- #

    def build_consent_url(
        self,
        auth_method: str,
        user_id: str,
        client_id_audit: str,
        session_id: str,
        server_path: str,
        egress_oauth: dict,
    ) -> str:
        """Build the provider authorize URL with an AEAD-encrypted, single-use state."""
        cfg = resolve_provider(egress_oauth)
        verifier = oauth_engine.generate_pkce_verifier() if cfg.use_pkce else None
        challenge = oauth_engine.pkce_challenge_s256(verifier) if verifier else None
        state = OAuthState(
            user_id=user_id,
            auth_method=auth_method,
            client_id=client_id_audit,
            provider=egress_oauth["provider"],
            server_path=server_path,
            session_id=session_id,
            pkce_verifier=verifier,
            nonce=secrets.token_urlsafe(16),
            issued_at=datetime.now(UTC).isoformat(),
        )
        return oauth_engine.build_authorize_url(
            cfg=cfg,
            client_id=egress_oauth["client_id"],
            redirect_uri=self._callback_url,
            scopes=list(egress_oauth.get("scopes") or []),
            state=encode_state(state),
            pkce_challenge=challenge,
        )

    # -- callback ------------------------------------------------------------- #

    async def handle_callback(
        self,
        code: str,
        state_blob: str,
        egress_oauth: dict,
        current_user_id: str | None = None,
        current_auth_method: str | None = None,
    ) -> EgressConnection:
        """Verify state, exchange the code, and store the token.

        Security checks before any token write:
        - state decrypts/authenticates (AEAD) -- else InvalidState.
        - state is within TTL.
        - state nonce is single-use (replay guard).
        - account-swap guard: the live principal (if provided) matches the
          principal bound in the signed state -- bind to (user_id, auth_method),
          NOT session_id (a legit "same user, new session" must still work).
        """
        try:
            state = decode_state(state_blob)
        except InvalidState as exc:
            raise EgressAuthError(f"invalid state: {exc}") from exc

        # TTL
        try:
            issued = datetime.fromisoformat(state.issued_at)
        except ValueError as exc:
            raise EgressAuthError("state issued_at malformed") from exc
        if (datetime.now(UTC) - issued).total_seconds() > self._state_ttl:
            raise EgressAuthError("state expired")

        # Single-use
        if not await self._replay.check_and_consume(state.nonce, self._state_ttl):
            raise EgressAuthError("state already used (replay)")

        # Account-swap guard (bind to canonical principal, not session_id)
        if current_user_id is not None and current_user_id != state.user_id:
            raise EgressAuthError("state user mismatch")
        if current_auth_method is not None and current_auth_method != state.auth_method:
            raise EgressAuthError("state auth_method mismatch")

        cfg = resolve_provider(egress_oauth)
        token = await oauth_engine.exchange_code(
            cfg=cfg,
            client_id=egress_oauth["client_id"],
            client_secret=self._client_secret(egress_oauth),
            code=code,
            redirect_uri=self._callback_url,
            pkce_verifier=state.pkce_verifier,
        )
        await self._store.put_token(
            state.auth_method, state.user_id, state.provider, state.server_path, token
        )
        return EgressConnection(
            provider=state.provider,
            server_path=state.server_path,
            scopes=token.scopes,
            expires_at=token.expires_at,
            status=token.status,
            last_refreshed_at=token.last_refreshed_at,
        )

    # -- vend ----------------------------------------------------------------- #

    async def get_valid_token(
        self,
        auth_method: str,
        user_id: str,
        server_path: str,
        egress_oauth: dict,
    ) -> str | None:
        """Vend a valid access token, refreshing if near expiry. None on miss.

        Primary refresh mechanism (lazy-on-vend). Returns None when there is no
        connection, the connection is dead (refresh_failed), the caller is not a
        per-user principal, or the stored client_id no longer matches (rotated
        provider app -> force re-consent).
        """
        if not is_per_user_auth_method(auth_method):
            return None

        provider = egress_oauth["provider"]
        token = await self._store.get_token(auth_method, user_id, provider, server_path)
        if token is None or token.status == "refresh_failed":
            return None

        # client-id binding: rotated provider app -> re-consent.
        if token.client_id and token.client_id != egress_oauth.get("client_id"):
            logger.info(
                "egress vend: stored client_id != configured client_id for %s/%s; forcing re-consent",
                provider,
                server_path,
            )
            return None

        if self._is_near_expiry(token):
            token = await self._refresh_single_flight(
                auth_method, user_id, server_path, egress_oauth, token
            )
        return token.access_token if token else None

    async def _refresh_single_flight(
        self,
        auth_method: str,
        user_id: str,
        server_path: str,
        egress_oauth: dict,
        token: StoredToken,
    ) -> StoredToken | None:
        """Single-flight refresh: cross-replica lease + post-acquire double-check.

        The post-acquire re-read is the CORRECTNESS anchor (a second waiter that
        finds a fresh token after acquiring does nothing); the lease only prevents
        refresh storms / rotating-refresh churn across replicas. The lease key is
        the canonical vault tuple so it matches the vault namespacing exactly.
        """
        provider = egress_oauth["provider"]
        key = f"{auth_method}|{user_id}|{provider}|{server_path}"

        acquired = await self._lease.acquire(key, self._holder, self._lease_ttl)
        if not acquired:
            # Could not take the lease (another replica is refreshing). Re-read
            # once -- if it refreshed, use that; else fall back to the stale token
            # rather than racing a concurrent refresh against a rotating provider.
            current = await self._store.get_token(auth_method, user_id, provider, server_path)
            return current if current and current.status != "refresh_failed" else None

        try:
            current = await self._store.get_token(auth_method, user_id, provider, server_path)
            if current and not self._is_near_expiry(current):
                return current  # another waiter already refreshed
            if current is None or not current.refresh_token:
                return None
            cfg = resolve_provider(egress_oauth)
            try:
                new = await oauth_engine.refresh_token(
                    cfg=cfg,
                    client_id=egress_oauth["client_id"],
                    client_secret=self._client_secret(egress_oauth),
                    refresh_token_value=current.refresh_token,
                )
            except oauth_engine.DeadRefreshTokenError:
                # Dead refresh token: mark the entry failed so the next vend
                # returns None -> consent URL, rather than retrying forever.
                failed = current.model_copy(update={"status": "refresh_failed"})
                await self._store.put_token(auth_method, user_id, provider, server_path, failed)
                logger.warning(
                    "egress refresh failed (dead refresh token) for %s/%s; marked refresh_failed",
                    provider,
                    server_path,
                )
                return None
            await self._store.put_token(auth_method, user_id, provider, server_path, new)
            return new
        finally:
            await self._lease.release(key, self._holder)

    # -- list / disconnect ---------------------------------------------------- #

    async def list_connections(
        self,
        auth_method: str,
        user_id: str,
    ) -> list[EgressConnection]:
        """List the principal's connections (tokens stripped)."""
        if not is_per_user_auth_method(auth_method):
            return []
        out: list[EgressConnection] = []
        for provider, server_path, token in await self._store.list_for_user(auth_method, user_id):
            out.append(
                EgressConnection(
                    provider=provider,
                    server_path=server_path,
                    scopes=token.scopes,
                    expires_at=token.expires_at,
                    status=token.status,
                    last_refreshed_at=token.last_refreshed_at,
                )
            )
        return out

    async def disconnect(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        """Delete the vault entry (idempotent). Provider-side revoke is a future follow-on."""
        await self._store.delete_token(auth_method, user_id, provider, server_path)

    # -- pat (static per-user PAT / API-key) ---------------------------------- #

    async def get_pat(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> str | None:
        """Vend a stored per-user PAT, enforcing the bounded lifetime. None on miss.

        Deliberately does NOT touch the OAuth refresh/lease machinery: a PAT is
        static (no refresh token) and stores ``client_id=None`` (which the OAuth
        vend path would reject). A PAT with no ``expires_at`` or one in the past is
        treated as a MISS so a stale credential can never linger.

        Args:
            auth_method: The caller's canonical per-user auth method.
            user_id: The vault principal (verified egress user).
            provider: The provider namespace key for this server.
            server_path: The slash-prefixed server path.

        Returns:
            The stored PAT string, or None on a miss (never submitted, expired,
            or a non-per-user caller).
        """
        if not is_per_user_auth_method(auth_method):
            return None
        token = await self._store.get_token(auth_method, user_id, provider, server_path)
        if token is None:
            return None
        # Enforce the bounded lifetime at the sink: an expired PAT is a MISS.
        if not token.expires_at or self._is_expired(token.expires_at):
            return None
        return token.access_token

    async def set_pat(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
        token: StoredToken,
    ) -> None:
        """Store a per-user PAT (thin wrapper over the SecretStore)."""
        await self._store.put_token(auth_method, user_id, provider, server_path, token)

    async def get_pat_status(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> StoredToken | None:
        """Read the stored PAT entry for status (presence + expiry only).

        The caller MUST NOT return ``access_token`` from this to any API
        consumer; the PAT is write-only. Returns None on a miss.
        """
        if not is_per_user_auth_method(auth_method):
            return None
        return await self._store.get_token(auth_method, user_id, provider, server_path)

    async def delete_pat(
        self,
        auth_method: str,
        user_id: str,
        provider: str,
        server_path: str,
    ) -> None:
        """Delete the caller's stored PAT (idempotent)."""
        await self._store.delete_token(auth_method, user_id, provider, server_path)
