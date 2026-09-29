"""Backend OAuth (client_credentials) token acquisition + in-process cache.

Lets the REGISTRY authenticate to an OAuth-backed MCP server as an OAuth2
client (RFC 6749 §4.4 ``client_credentials`` grant) when it performs health
checks and tool discovery -- the machine-to-machine analogue of the static
``bearer`` / ``api_key`` backend-auth schemes. This is the registry's OWN
credential to reach the upstream; it is unrelated to the per-user egress vault
(``registry/egress_auth/``), which brokers END-USER tokens through the gateway.

Config lives on the server record under ``auth_scheme == "oauth"`` +
``backend_oauth`` (token_url, client_id, encrypted client_secret, scopes,
token_auth_style, resource). The client_secret is Fernet-encrypted with the
same key as every other backend credential.

The header builders in ``registry/core/mcp_client.py`` and
``registry/health/service.py`` are synchronous, but acquiring a token is an
async network call. So the async discovery/health entrypoints call
:func:`with_bearer` to resolve+cache the token and stash it on a shallow copy of
``server_info`` under :data:`RESOLVED_BEARER_KEY`; the sync builders read that
key for the ``oauth`` scheme. Resolution is cached per server path with a
single-flight lock so concurrent health checks don't stampede the token
endpoint, and is invalidated when the config fingerprint changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from registry.core.config import settings
from registry.egress_auth import oauth_engine
from registry.egress_auth.schemas import OAuthProviderConfig, TokenEndpointAuthStyle
from registry.egress_auth.upstream_binding import base_url
from registry.secrets import keys
from registry.utils.credential_encryption import decrypt_credential

logger = logging.getLogger(__name__)

# Key under which the resolved bearer token is stashed on a server_info copy for
# the synchronous header builders to consume.
RESOLVED_BEARER_KEY = "_backend_oauth_bearer"

# TTL applied when the token endpoint returns no expiry hint (no ``expires_in``
# and no JWT ``exp``). Short, so an opaque token is re-acquired each cycle rather
# than cached indefinitely.
_DEFAULT_TTL_SECONDS = 60

# Absolute floor on the freshness margin. The cap at half-lifetime keeps short-lived
# tokens cacheable, but without a floor the margin trends to zero and a token can
# expire between the freshness check and the upstream reading the header.
_MIN_FRESHNESS_MARGIN_SECONDS = 5

# Default Entra login base URL for the gateway's own token endpoint. Sovereign
# clouds override via ``settings.entra_login_base_url`` (mirrors the auth-server).
_DEFAULT_ENTRA_LOGIN_BASE_URL = "https://login.microsoftonline.com"

# Rate limit for the "designated but not connected" warning, per server path. The
# health loop revisits every ~38s and the state persists until a human acts, so an
# unbounded warning is pure noise after the first line.
_UNCONNECTED_WARN_INTERVAL_SECONDS = 900
_unconnected_warned_at: dict[str, float] = {}


@dataclass
class _CacheEntry:
    access_token: str
    # Epoch seconds; None means "no expiry known" -> use the default short TTL.
    expires_at_epoch: float | None
    fingerprint: str
    # Wall-clock epoch when acquired (used only for the no-expiry default TTL).
    acquired_epoch: float


_cache: dict[str, _CacheEntry] = {}
_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


def _server_path(server_info: dict) -> str | None:
    return server_info.get("service_path") or server_info.get("path")


def _warn_discovery_unconnected(server_path: str | None) -> None:
    """Warn that a designated discovery identity has no vaulted token, at most
    once per server per :data:`_UNCONNECTED_WARN_INTERVAL_SECONDS`.

    WARNING rather than INFO because nothing here self-heals. The designation is
    durable and the token is gone, so every health cycle takes this branch until
    a human reconnects. It is a degraded state needing action, not routine
    information, and at INFO it sits below the level anyone watches.

    Rate-limited because the health loop runs every ~38s: one incident produced
    472 identical lines in five hours, which trains readers to ignore the channel
    rather than telling them anything the first line did not. The first
    occurrence still logs immediately, so the cause is present from the start.
    """
    now = time.monotonic()
    last = _unconnected_warned_at.get(server_path or "")
    if last is not None and (now - last) < _UNCONNECTED_WARN_INTERVAL_SECONDS:
        return
    _unconnected_warned_at[server_path or ""] = now
    logger.warning(
        "oauth discovery: no valid vaulted token for path=%s. A discovery identity "
        "is designated but not connected (never consented, revoked, or the vault "
        "lost it -- OpenBao in dev mode discards tokens on restart). Headless "
        "discovery, health checks and security scans for this server run WITHOUT a "
        "credential until someone reconnects it.",
        server_path,
    )


def _fingerprint(bo: dict, client_secret_encrypted: str | None, upstream: str) -> str:
    """Stable hash of the config so an edited config invalidates the cache.

    Includes the encrypted secret (ciphertext) so a rotated secret forces a
    re-acquire without ever hashing plaintext.

    Includes the registered UPSTREAM so the cached token is destination-bound, the way
    the per-user vault binds its entries. Without it, repointing ``proxy_pass_url``
    leaves the fingerprint unchanged, so the next health cycle ships the token minted
    for the old host to the new one -- no re-mint, no signal. The SSRF guard only
    excludes private and metadata addresses, so any public host passes it.
    """
    material = "|".join(
        [
            str(bo.get("token_url") or ""),
            str(bo.get("client_id") or ""),
            str(bo.get("token_auth_style") or "post_body"),
            str(bo.get("scope_separator") or " "),
            str(bo.get("resource") or ""),
            ",".join(bo.get("scopes") or []),
            client_secret_encrypted or "",
            upstream,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _to_epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def _is_fresh(entry: _CacheEntry) -> bool:
    now = datetime.now(UTC).timestamp()
    if entry.expires_at_epoch is not None:
        # egress_token_refresh_skew_seconds (default 300) is tuned for long-lived
        # DELEGATED user tokens. Applied verbatim to a machine token it can exceed the
        # whole lifetime: any token issued with expires_in <= skew would never be
        # considered fresh, so it would be stored, returned once, and re-minted on the
        # next call -- turning this cache into a no-op and hammering the IdP once per
        # server per health cycle. Cap the margin at half the token's own lifetime so
        # a short-lived token still gets cached for a useful fraction of it -- but
        # FLOOR it, because capping alone lets the margin collapse toward zero: a token
        # with expires_in=2 would read as fresh with 1s left and could expire in flight
        # between this check and the upstream receiving it.
        lifetime = max(0.0, entry.expires_at_epoch - entry.acquired_epoch)
        skew = min(max(0, settings.egress_token_refresh_skew_seconds), lifetime / 2)
        skew = max(skew, _MIN_FRESHNESS_MARGIN_SECONDS)
        return entry.expires_at_epoch - now > skew
    # No known expiry: honor the short default TTL.
    return now - entry.acquired_epoch < _DEFAULT_TTL_SECONDS


def _build_cfg(bo: dict) -> OAuthProviderConfig:
    token_url = bo.get("token_url") or ""
    style_raw = bo.get("token_auth_style") or TokenEndpointAuthStyle.POST_BODY.value
    try:
        style = TokenEndpointAuthStyle(style_raw)
    except ValueError:
        style = TokenEndpointAuthStyle.POST_BODY
    return OAuthProviderConfig(
        name="backend-client-credentials",
        display_name="Backend OAuth (client credentials)",
        # authorize_url is unused for client_credentials but the model requires a
        # value; mirror token_url so the config is self-consistent.
        authorize_url=token_url,
        token_url=token_url,
        scope_separator=bo.get("scope_separator") or " ",
        token_endpoint_auth_style=style,
        use_pkce=False,
        resource=bo.get("resource") or None,
        is_builtin=False,
    )


async def _lock_for(key: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        return lock


async def resolve_bearer(server_info: dict) -> str | None:
    """Return a valid backend OAuth access token for ``server_info``, or None.

    Returns None (rather than raising) on any misconfiguration or token-endpoint
    failure so the caller simply omits the Authorization header -- the health
    check then fails/records unhealthy, which is the correct signal. Cached per
    server path with a single-flight lock and config-fingerprint invalidation.

    ``token_url`` MUST be HTTPS. The request carries the operator's ``client_secret``, so
    it runs under ``CREDENTIALED_OAUTH_PROFILE``, whose ``require_https`` cannot be
    relaxed -- ``EGRESS_OAUTH_TRUSTED_IDP_HOSTS`` admits a private *host* but never a
    plaintext scheme. A plain-HTTP IdP is therefore un-onboardable through this tier,
    including this project's own bundled Keycloak. ``PUT /oauth-config`` validates the
    URL eagerly so that surfaces as a 400 at config time rather than a silent
    health-cycle failure -- except for a host that only *resolves* private, since that
    validation runs with ``resolve=False`` to stay DNS-independent.
    """
    if server_info.get("auth_scheme") != "oauth":
        return None
    bo = server_info.get("backend_oauth") or {}
    token_url = bo.get("token_url")
    client_id = bo.get("client_id")
    if not token_url or not client_id:
        logger.warning(
            "backend oauth misconfigured (missing token_url/client_id) path=%s",
            _server_path(server_info),
        )
        return None

    secret_encrypted = bo.get("client_secret_encrypted")
    fingerprint = _fingerprint(
        bo, secret_encrypted, base_url(server_info.get("proxy_pass_url") or "")
    )
    key = _server_path(server_info) or fingerprint

    # Fast path: fresh cache hit for the current config.
    entry = _cache.get(key)
    if entry and entry.fingerprint == fingerprint and _is_fresh(entry):
        return entry.access_token

    lock = await _lock_for(key)
    async with lock:
        # Double-check under the lock: another waiter may have just acquired.
        entry = _cache.get(key)
        if entry and entry.fingerprint == fingerprint and _is_fresh(entry):
            return entry.access_token

        # decrypt_credential returns None rather than raising on failure (e.g. after a
        # SECRET_KEY rotation), so a bare `if secret_encrypted else None` would leave
        # client_secret None and fall through to POSTing just the client_id -- silently
        # downgrading a CONFIDENTIAL client to a public-client request against the
        # operator's real token endpoint, and masking the actual cause. Fail closed.
        client_secret = None
        if secret_encrypted:
            client_secret = decrypt_credential(secret_encrypted)
            if not client_secret:
                logger.error(
                    "backend oauth: stored client_secret for path=%s could not be "
                    "decrypted (SECRET_KEY rotated?); refusing to request a token as a "
                    "public client. Re-save the OAuth config to re-encrypt it.",
                    _server_path(server_info),
                )
                return None
        cfg = _build_cfg(bo)
        scopes = bo.get("scopes") or []
        try:
            token = await oauth_engine.client_credentials_token(
                cfg, client_id, client_secret, scopes
            )
        except oauth_engine.OAuthEngineError as exc:
            logger.warning(
                "backend oauth token acquisition failed path=%s error=%s",
                _server_path(server_info),
                exc,
            )
            return None

        _cache[key] = _CacheEntry(
            access_token=token.access_token,
            expires_at_epoch=_to_epoch(token.expires_at),
            fingerprint=fingerprint,
            acquired_epoch=datetime.now(UTC).timestamp(),
        )
        logger.info(
            "backend oauth token acquired path=%s expires_at=%s",
            _server_path(server_info),
            token.expires_at,
        )
        return token.access_token


async def resolve_discovery_bearer(server_info: dict) -> str | None:
    """Borrow the designated per-user identity's vaulted token for headless
    discovery against an OAuth 2.1 (per-user) server. None when not configured,
    not connected, or the vaulted token is dead/near-expiry-unrefreshable.

    Uses the server's OWN backend-auth discovery config (``oauth_discovery.oauth``)
    rather than ``egress_oauth`` -- the two are separate configurations. It does,
    however, DEPEND ON THE EGRESS FEATURE: the borrow reads the per-user OAuth vault
    via ``get_valid_token``, and the consent that fills that vault is served by the
    egress OAuth facade. So ``EGRESS_AUTH_ENABLED`` is a hard requirement, not an
    independent concern -- without it there is no vault to borrow from and no front
    door to populate it. Enforcing that here keeps the resolver honest with the
    router, which only mounts ``/oauth2/egress/connect`` under the same flag.

    Precedence: like :func:`resolve_obo_discovery_bearer`, this only supplies a
    credential when the operator configured no explicit static one. A stored
    ``bearer``/``api_key`` scan token is the operator's deliberate choice of
    discovery credential and MUST win -- without this guard the borrowed token
    is stashed under :data:`RESOLVED_BEARER_KEY`, and the header builders
    early-return on that key, silently dropping the static credential.
    ``auth_scheme == 'oauth'`` is likewise excluded: that is tier 1's own
    configuration, and :func:`with_bearer` must not substitute a human identity
    when a machine grant was asked for.
    """
    disc = server_info.get("oauth_discovery") or {}
    if not disc.get("enabled"):
        return None
    # Gate AFTER the enabled check, deliberately. Checking the flag first cannot tell
    # "no discovery configured" from "configured, designated, consented, then the flag
    # was turned off" -- and the latter is the case an operator needs told, because the
    # UI hides the config while the record still exists, so servers silently flip
    # unhealthy with nothing naming the cause.
    if not settings.egress_auth_enabled:
        logger.warning(
            "oauth discovery configured for path=%s but EGRESS_AUTH_ENABLED is false; "
            "the borrow reads the per-user vault, so discovery cannot run and this "
            "server will record unhealthy. Enable the egress feature, or remove the "
            "discovery identity (DELETE /servers/%s/oauth-discovery).",
            _server_path(server_info),
            (_server_path(server_info) or "").lstrip("/"),
        )
        return None
    if (server_info.get("auth_scheme") or "none") != "none":
        return None
    oauth_cfg = disc.get("oauth")
    auth_method = disc.get("auth_method")
    user_id = disc.get("user_id")
    if not oauth_cfg or not auth_method or not user_id:
        logger.warning(
            "oauth discovery misconfigured (missing oauth config/principal) path=%s",
            _server_path(server_info),
        )
        return None
    server_path = _server_path(server_info)
    try:
        from registry.egress_auth.factory import get_egress_auth_service

        # The vaulted token is destination-bound (issue: newer egress upstream
        # binding). Discovery hits this server's own endpoint, so bind the borrow
        # to the server's registered upstream base. A mismatch simply yields None
        # (discovery degrades), never a cross-upstream credential leak.
        token = await get_egress_auth_service().get_valid_token(
            auth_method=auth_method,
            user_id=user_id,
            server_path=server_path,
            egress_oauth=oauth_cfg,
            requested_upstream=base_url(server_info.get("proxy_pass_url") or ""),
            # Only an entry vaulted BY a discovery consent. An egress entry at the same
            # address belongs to the user's own runtime use, and borrowing it would put
            # the registry's headless calls on a credential the user consented for
            # themselves -- the separation this whole path exists to provide.
            purpose=keys.DISCOVERY_PURPOSE,
        )
    except Exception as exc:  # egress/vault/refresh failure -> degrade, don't crash discovery
        logger.warning(
            "oauth discovery token borrow failed path=%s error=%s",
            server_path,
            type(exc).__name__,
        )
        return None
    if not token:
        _warn_discovery_unconnected(server_path)
    return token


def _gateway_idp_client() -> tuple[str, str, str] | None:
    """The gateway's OWN IdP client as ``(client_id, client_secret, token_url)``
    for a machine (client_credentials) grant, or None when the configured provider
    is unsupported or its client is not configured.

    This is the SAME app registration the gateway uses for ingress; the OBO
    discovery path borrows it to mint an app-only token audienced to an internal
    server -- no per-server secret, no user, no vault.
    """
    provider = (settings.auth_provider or "").lower()
    if provider == "entra":
        client_id = settings.entra_client_id or ""
        client_secret = settings.entra_client_secret or ""
        tenant = settings.entra_tenant_id or ""
        if not (client_id and client_secret and tenant):
            return None
        # ``settings.entra_login_base_url`` carries the same default, so the
        # fallback below only covers an override explicitly blanked by an
        # operator (``ENTRA_LOGIN_BASE_URL=`` in a .env file).
        login_base = (settings.entra_login_base_url or _DEFAULT_ENTRA_LOGIN_BASE_URL).rstrip("/")
        return client_id, client_secret, f"{login_base}/{tenant}/oauth2/v2.0/token"
    if provider == "keycloak":
        # NOT SUPPORTED, and deliberately fails closed rather than nearly working.
        #
        # Two things are missing, and only one of them is visible. The token request
        # carries the gateway's client_secret, so it goes through
        # CREDENTIALED_OAUTH_PROFILE, which requires HTTPS and refuses private hosts;
        # KEYCLOAK_URL is an in-cluster plain-HTTP URL on every shipped deployment
        # (``http://keycloak:8080`` on compose, the headless Service on Helm), so the
        # guard rejects it. Relaxing that guard is not the fix -- it exists to stop us
        # posting a client secret in cleartext.
        #
        # The second is the dangerous one. Keycloak binds the audience through a
        # server-side audience mapper rather than a request scope, so
        # :func:`_obo_discovery_scopes` sends NO scope. Neither the charts nor the realm
        # bootstrap create that mapper. An operator who fixes only the HTTPS problem
        # therefore gets past this function and mints a token audienced to whatever
        # Keycloak defaults to -- not ``target_audience`` -- which is then sent to a
        # third-party MCP server as ``Authorization: Bearer``. Nothing downstream
        # re-checks the audience, so the failure is silent and the token is real.
        #
        # Returning None unconditionally is the only honest state until the mapper is
        # created and the scope/audience contract is testable. The previous message told
        # the operator to set an HTTPS URL "to use obo_exchange discovery", which led
        # straight into that path.
        logger.warning(
            "obo discovery unavailable: AUTH_PROVIDER=keycloak is not supported for "
            "backend discovery. Keycloak binds the token audience with a server-side "
            "audience mapper that this deployment does not create, so the gateway cannot "
            "prove a token is audienced to the target. Use AUTH_PROVIDER=entra for "
            "obo_exchange discovery."
        )
        return None
    return None


def _obo_discovery_scopes(provider: str, target: str) -> list[str]:
    """Scopes for the OBO discovery client_credentials request.

    Entra: ``<target>/.default`` requests the application permissions (app roles)
    the gateway app has been granted on the target resource -- the only scope form
    Entra accepts for an app-only (client_credentials) token.

    Only Entra reaches this. An empty list would mean "mint a token with no audience
    constraint and send it upstream anyway", which is why :func:`_gateway_idp_client`
    refuses every other provider rather than letting one fall through to here.
    """
    if provider == "entra":
        return [f"{target}/.default"]
    return []


def _obo_fingerprint(
    client_id: str, token_url: str, target: str, scopes: list[str], upstream: str
) -> str:
    """Stable hash so an edited target/scope or a re-pointed gateway client forces
    a re-acquire. The gateway client_secret is env-based (not per-server) and short
    TTLs re-mint after a rotation, so it is intentionally excluded (never hash a
    plaintext secret).

    ``upstream`` makes the cached token destination-bound. This matters MORE here than
    for tier 1: the token is the gateway's own app-only credential for
    ``target_audience``, which the server owner never held. Without it, repointing
    ``proxy_pass_url`` would ship that credential to a newly registered host on the
    next health cycle with no re-mint and no signal.
    """
    material = "|".join([client_id, token_url, target, ",".join(scopes), upstream])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def resolve_obo_discovery_bearer(server_info: dict) -> str | None:
    """Mint a machine (client_credentials) token audienced to an ``obo_exchange``
    server's ``target_audience`` for headless health/tool-discovery, or None.

    This is the machine-identity analogue of the per-user OBO exchange: OBO
    re-audiences a USER's ingress JWT at request time and cannot run headless (no
    subject token), so discovery authenticates as the gateway app ITSELF. The only
    credential is the gateway's existing IdP client secret -- no per-user token, no
    vault, no interactive connect. The Entra app must be granted an application
    permission (app role) on the target server's app, admin-consented, so
    ``<target>/.default`` yields a token (see docs/obo-token-exchange.md).

    Precedence: this is the FALLBACK backend discovery credential for an obo
    server. It fires only when the operator configured no explicit backend
    ``auth_scheme`` -- a ``bearer``/``api_key`` scan token or ``oauth``
    (client_credentials) always wins and is never shadowed by this derived
    token. A designated OAuth 2.1 discovery identity is recorded in the separate
    ``oauth_discovery`` block and leaves ``auth_scheme`` at ``none``, so it is not
    excluded by that guard; it wins earlier, at tier 2 of :func:`with_bearer`, via
    :func:`resolve_discovery_bearer`.

    Returns None (not raises) on any misconfiguration/failure so the caller omits
    the header -- discovery then records the server unhealthy, the correct signal.
    Cached per server path with a single-flight lock and fingerprint invalidation,
    exactly like :func:`resolve_bearer`.

    Requires ``EGRESS_AUTH_ENABLED``, but for a NARROWER reason than
    :func:`resolve_discovery_bearer`. Tier 2 needs the flag because it reads the
    per-user vault; this tier touches no vault -- it uses only the gateway's own IdP
    client. What it does depend on is egress CONFIG: ``egress_auth_mode`` and
    ``egress_oauth.target_audience``, neither of which can be set while the feature is
    off. So an ``obo_exchange`` server seen with the flag disabled is stored config
    surviving a flag flip -- a misconfiguration -- and we fail closed on
    misconfiguration and log it.

    Deliberately NOT justified as "a server whose runtime path is off should not report
    healthy". That claim proves too much: tier 1 happily authenticates servers no user
    can reach for a dozen other reasons (no scope grant, no group membership), and the
    registry does not otherwise suppress health checks because traffic would be
    unauthorized.
    """
    if server_info.get("egress_auth_mode") != "obo_exchange":
        return None
    if not settings.egress_auth_enabled:
        logger.warning(
            "obo discovery: path=%s has egress_auth_mode='obo_exchange' but "
            "EGRESS_AUTH_ENABLED is false -- stored egress config surviving a flag "
            "flip. Failing closed; this server will record unhealthy.",
            _server_path(server_info),
        )
        return None
    # Fallback only: an explicit backend discovery credential (any auth_scheme
    # other than none) is the operator's chosen credential and MUST win. Without
    # this, a static bearer/api_key scan token would be silently shadowed because
    # the resolved-bearer key takes precedence in the sync header builders.
    if (server_info.get("auth_scheme") or "none") != "none":
        return None
    eo = server_info.get("egress_oauth") or {}
    target = (eo.get("target_audience") or "").strip()
    if not target:
        logger.warning(
            "obo discovery misconfigured (missing target_audience) path=%s",
            _server_path(server_info),
        )
        return None

    # Defense-in-depth: re-run the SAME target-audience control the registration
    # path enforces, so discovery can never mint a broad or reflected token even
    # for a stored record that predates the check.
    #
    # This calls _validate_obo_egress_config, NOT just _is_disallowed_obo_audience.
    # The registration path runs five clauses and the first-party floor is only
    # one; _is_gateway_own_audience matters MORE here than it does for the runtime
    # grant. A same-app OBO exchange is rejected by Entra at runtime, so
    # jwt-bearer has an IdP-side backstop -- but `api://<our-own-client-id>/.default`
    # is a perfectly valid app-only request that Entra will happily issue. That
    # token is then sent upstream as `Authorization: Bearer` to a third-party MCP
    # server, in the exact form this gateway's own ingress accepts. Reflecting our
    # own audience to an upstream is the confused-deputy case the control exists
    # to stop, and client_credentials is the one grant with no backstop for it.
    #
    # scopes=None: tier 3 builds its own `<target>/.default` and never reads the
    # stored scopes, so the scope-binding clause has nothing to check.
    from registry.core.schemas import _validate_obo_egress_config

    try:
        _validate_obo_egress_config(target, None)
    except ValueError as exc:
        logger.warning(
            "obo discovery target audience refused path=%s reason=%s",
            _server_path(server_info),
            exc,
        )
        return None

    client = _gateway_idp_client()
    if client is None:
        logger.warning(
            "obo discovery unavailable: no gateway IdP client usable for backend "
            "discovery with provider=%s (only 'entra' is supported) path=%s",
            settings.auth_provider,
            _server_path(server_info),
        )
        return None
    client_id, client_secret, token_url = client
    provider = (settings.auth_provider or "").lower()
    scopes = _obo_discovery_scopes(provider, target)

    fingerprint = _obo_fingerprint(
        client_id, token_url, target, scopes, base_url(server_info.get("proxy_pass_url") or "")
    )
    key = _server_path(server_info) or fingerprint

    entry = _cache.get(key)
    if entry and entry.fingerprint == fingerprint and _is_fresh(entry):
        return entry.access_token

    lock = await _lock_for(key)
    async with lock:
        entry = _cache.get(key)
        if entry and entry.fingerprint == fingerprint and _is_fresh(entry):
            return entry.access_token

        cfg = OAuthProviderConfig(
            name="obo-discovery-client-credentials",
            display_name="OBO discovery (client credentials)",
            # authorize_url is unused for client_credentials; mirror token_url.
            authorize_url=token_url,
            token_url=token_url,
            scope_separator=" ",
            token_endpoint_auth_style=TokenEndpointAuthStyle.POST_BODY,
            use_pkce=False,
            is_builtin=False,
        )
        try:
            token = await oauth_engine.client_credentials_token(
                cfg, client_id, client_secret, scopes
            )
        except oauth_engine.OAuthEngineError as exc:
            logger.warning(
                "obo discovery token acquisition failed path=%s error=%s",
                _server_path(server_info),
                exc,
            )
            return None

        _cache[key] = _CacheEntry(
            access_token=token.access_token,
            expires_at_epoch=_to_epoch(token.expires_at),
            fingerprint=fingerprint,
            acquired_epoch=datetime.now(UTC).timestamp(),
        )
        logger.info(
            "obo discovery token acquired path=%s audience=%s expires_at=%s",
            _server_path(server_info),
            target,
            token.expires_at,
        )
        return token.access_token


async def with_bearer(server_info: dict) -> dict:
    """Return ``server_info`` unchanged, or a shallow copy carrying a resolved
    OAuth bearer token under :data:`RESOLVED_BEARER_KEY`.

    Resolves at most one credential, from mutually exclusive tiers:

    1. ``auth_scheme == 'oauth'`` -> client_credentials (:func:`resolve_bearer`).
       TERMINAL: the operator configured a machine grant, so a failure here is
       "no credential", never "try a different principal". Falling through would
       substitute a borrowed HUMAN token on a transient token-endpoint error --
       a silent privilege-class change that is harder to notice than an outright
       failure.
    2. ``oauth_discovery.enabled`` with no explicit ``auth_scheme`` -> a borrowed
       per-user discovery identity (:func:`resolve_discovery_bearer`).
    3. an ``obo_exchange`` server with no explicit ``auth_scheme`` -> a gateway
       machine token (:func:`resolve_obo_discovery_bearer`).

    Tiers 2 and 3 both bow out unless ``auth_scheme`` is ``none``, so a static
    ``bearer``/``api_key`` scan token is never shadowed. No-op when no tier
    applies, so callers invoke it unconditionally before building headers.
    """
    if not server_info:
        return server_info
    token: str | None = None
    if server_info.get("auth_scheme") == "oauth":
        # Terminal tier -- see (1) above. Do not fall through on None.
        token = await resolve_bearer(server_info)
    else:
        token = await resolve_discovery_bearer(server_info)
        if token is None:
            token = await resolve_obo_discovery_bearer(server_info)
    if not token:
        return server_info
    return {**server_info, RESOLVED_BEARER_KEY: token}


def invalidate(server_path: str) -> None:
    """Drop any cached token AND its single-flight lock for a server path.

    Call after a config change or when a server is deleted. Dropping the lock too
    matters for deletion: ``_locks`` is keyed the same way as ``_cache`` and nothing
    else prunes it, so a registry with server churn would accumulate one ``asyncio``
    Lock per path ever seen.

    Racing an in-flight acquire is benign but NOT a no-op, so do not read more into
    this than it does. A concurrent acquirer already holds its own reference to the
    popped lock, so nothing breaks -- but the next caller creates a FRESH lock, so
    single-flight is lost for that window, and the in-flight task will write its
    pre-invalidation token into ``_cache`` after the invalidation. The fingerprint
    check on the next read discards it, so the cost is one extra mint, not a stale
    credential.
    """
    _cache.pop(server_path, None)
    _locks.pop(server_path, None)
