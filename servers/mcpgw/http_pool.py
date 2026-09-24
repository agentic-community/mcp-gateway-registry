"""Process-lifetime, connection-pooled HTTP client for mcpgw egress.

Every mcpgw tool used to build and tear down a fresh ``httpx.AsyncClient`` per
call, so each registry API request and each Keycloak M2M token fetch paid a full
TCP (+TLS) handshake with no keep-alive reuse. ``search_registry`` is on the
discovery hot path and is called once per agent turn, so the handshake churn is
paid per turn under concurrency.

This module owns ONE pooled client shared by every egress hop mcpgw makes ITSELF:

  * the 7 registry API calls (``/api/servers``, ``/api/agents``, ``/api/skills``,
    skill content, ``/api/search/semantic`` x2, ``/api/servers/health``), and
  * the Keycloak M2M ``client_credentials`` token POST.

Partially covered: when ``OIDC_ENABLED=true``, fastmcp's ``OAuthProxy`` also reaches
``KEYCLOAK_INTERNAL_URL`` for ``/token`` (login, client refresh, proactive refresh)
and ``/revoke``. Those clients are built inline inside fastmcp 3.4.7 with no
injection point, so those four call sites are neither pooled nor guarded here nor
counted by the reset metric. They fire per login/refresh, never per tool call, and
their URLs are operator env config. The ``/certs`` JWKS fetch IS covered --
``JWTVerifier`` takes a public ``http_client=``, wired in ``server.py``.

It mirrors ``registry/utils/url_guard.py`` rather than importing it:
``docker/Dockerfile.mcp-server`` is generic over ``servers/*`` and copies only
``${SERVER_DIR}/`` into the image, and ``url_guard`` pulls in ``registry.exceptions``
plus (lazily) ``registry.core.config``. Sharing the code would therefore mean
vendoring a registry subtree and pydantic-settings into every MCP server image, so
the duplication here is a deliberate trade against restructuring that image -- not
an impossibility. The four ``EGRESS_HTTP_POOL_*`` env vars are parsed here with
``os.getenv``, using the same defaults and bounds as ``registry/core/config.py`` so
one operator value configures all three processes identically.

Security posture
----------------
Every request goes through ``_GuardedAsyncTransport``, which validates and PINS
the destination per request, before pool checkout. mcpgw's two destinations are
operator env config (``REGISTRY_BASE_URL``, ``KEYCLOAK_INTERNAL_URL``), never
request- or registrant-derived, so there is no *policy* decision to make -- but
configuration trust says nothing about what those hostnames RESOLVE to. Both
requests carry a privileged credential (the registry API token or the M2M access
token) and both return the response body to the MCP caller, so a DNS answer that
points ``registry`` or ``keycloak`` at a link-local/metadata address would be a
credential-exfil primitive. The guard therefore:

  * restricts the request to the exact ``(host, port)`` pairs this process is
    configured for (``_allowed_destinations``), so a future tool cannot quietly
    borrow this client -- and its credentials -- for a third destination;
  * hard-denies cloud metadata / workload-identity endpoints (EC2 IMDS, ECS task
    creds, EKS Pod Identity, Alibaba), link-local, unspecified, multicast, and
    reserved addresses -- none of it relaxable;
  * unwraps IPv6 forms that embed an IPv4 address (IPv4-mapped, NAT64, 6to4,
    Teredo) and classifies the embedded IPv4, so ``::ffff:169.254.169.254`` cannot
    smuggle IMDS past the category checks;
  * rejects URL userinfo (credentials belong in a per-request header);
  * ALLOWS private-unicast, CGNAT, and loopback -- unlike the registry's default
    profile, matching its ``EGRESS_UPSTREAM_PROFILE`` instead -- because mcpgw
    legitimately talks to in-cluster service names (``registry``, ``keycloak``) and
    to ``localhost`` in dev/stdio mode.

Pinning is what keeps pooling rebind-safe: the connect host is rewritten to the
validated IP (``Host`` header and TLS SNI keep the original hostname), so the
httpcore pool is keyed by the pinned IP. A hostname that later resolves elsewhere
opens a new connection instead of reusing the old one.

What the guard here deliberately does NOT carry over from the registry: operator
allowlists/profiles, the structural ``validate_url`` checks it does not need, and
``coerce_ip_literal``'s obfuscated IPv4-literal spellings (decimal/octal/hex).
Those exist for attacker-SUPPLIED URLs. An obfuscated literal is still handled
here, just by a different mechanism -- it is not parseable as an IP, so it falls
through to the resolver branch, where getaddrinfo canonicalizes it and every
answer is classified (see the comment in ``handle_async_request``).

The denial table (``_CREDENTIAL_ENDPOINT_IPS``, the embedded-IPv4 prefixes, and
``_ip_denial_reason``) is hand-duplicated from ``url_guard``;
``tests/unit/servers/mcpgw/test_http_pool.py`` asserts equivalence with
``url_guard._ip_denial_reason(ip, allow_private=True)`` so drift on either side
fails a test rather than silently narrowing the guard.

Other posture, unchanged from the per-call clients this replaces:

  * No shared default headers -- every credential rides a PER-REQUEST header, so
    a pooled connection carries no caller identity between requests.
  * Cookie persistence disabled (``_NoStoreCookieJar``): a ``Set-Cookie`` is never
    stored, so it can never be replayed onto a later OR concurrent request sharing
    the client. Credentials never ride cookies here; this is defense-in-depth.
  * Timeouts are passed PER REQUEST at the call site (30s registry, 15s token);
    the client default is only a fallback.
"""

from __future__ import annotations

import asyncio
import http.cookiejar
import ipaddress
import logging
import os
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# Fallback client timeout. Every call site passes its own per-request timeout; this
# only bounds a future caller that forgets to.
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# Bounds mirror registry/core/config.py's Settings fields so an operator value that
# the registry accepts behaves identically here (and a typo cannot produce an
# unbounded pool or a negative expiry).
_MAX_CONNECTIONS_DEFAULT = 100
_MAX_KEEPALIVE_DEFAULT = 20
_KEEPALIVE_EXPIRY_DEFAULT = 30.0
_CONNECT_RETRIES_DEFAULT = 1

# Bound DNS work independently of the HTTP connect/read timeout so a slow or
# adversarial resolver cannot stall the event loop.
_DNS_RESOLUTION_TIMEOUT_SECONDS: float = 5.0

# Defaults for the two operator-configured egress destinations. Defined HERE and
# imported by ``server.py`` so the URL the tools build and the destination the guard
# admits can never disagree: if they did, an unset ``KEYCLOAK_INTERNAL_URL`` would make
# the guard reject the exact token/JWKS endpoint the server is calling.
DEFAULT_REGISTRY_BASE_URL = "http://localhost"
DEFAULT_KEYCLOAK_INTERNAL_URL = "http://keycloak:8080"

# Cloud metadata and workload-identity credential endpoints. NEVER reachable, and
# deliberately checked as an explicit set BEFORE any category logic: several of
# these are themselves link-local or private, so allowing private-unicast (which
# mcpgw must, to reach in-cluster service names) would otherwise reopen them.
# Kept in sync with registry/utils/url_guard.py::_CREDENTIAL_ENDPOINT_IPS.
_CREDENTIAL_ENDPOINT_IPS: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # EC2 IMDS (IPv4)
        "fd00:ec2::254",  # EC2 IMDS (IPv6)
        "169.254.170.2",  # ECS task credentials
        "169.254.170.23",  # EKS Pod Identity (IPv4)
        "fd00:ec2::23",  # EKS Pod Identity (IPv6)
        "100.100.100.200",  # Alibaba Cloud ECS metadata
    }
)

# IPv6 transports that embed a reachable IPv4 address. Python classifies the
# wrapper as neither link-local nor private, so the embedded IPv4 must be
# extracted and category-checked instead of the wrapper.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052
_6TO4_PREFIX = ipaddress.ip_network("2002::/16")  # RFC 3056: 2002:V4:V4::/48
_TEREDO_PREFIX = ipaddress.ip_network("2001::/32")  # RFC 4380: client v4, XOR'd

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class EgressTargetError(ValueError):
    """A request was aimed at a destination mcpgw must never connect to.

    Subclasses ``ValueError`` so the tool call sites' existing ``except ValueError``
    branch reports it and fails closed, exactly like an invalid tool argument.
    """


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    """Return a bounded int from the environment, falling back on any bad value."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning("%s=%s is outside [%s, %s]; using %s", name, value, low, high, default)
        return default
    return value


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    """Return a bounded float from the environment, falling back on any bad value."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning("%s=%s is outside [%s, %s]; using %s", name, value, low, high, default)
        return default
    return value


def _pool_limits() -> httpx.Limits:
    """Build the pool limits from EGRESS_HTTP_POOL_*, clamping keepalive to max."""
    max_connections = _env_int(
        "EGRESS_HTTP_POOL_MAX_CONNECTIONS", _MAX_CONNECTIONS_DEFAULT, low=1, high=10000
    )
    max_keepalive = _env_int(
        "EGRESS_HTTP_POOL_MAX_KEEPALIVE", _MAX_KEEPALIVE_DEFAULT, low=0, high=10000
    )
    # max_keepalive must not exceed max_connections (same clamp as the registry's
    # Settings validator) -- httpx would otherwise keep more idle connections than
    # the pool ceiling allows.
    max_keepalive = min(max_keepalive, max_connections)
    keepalive_expiry = _env_float(
        "EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS",
        _KEEPALIVE_EXPIRY_DEFAULT,
        low=0.0,
        high=600.0,
    )
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
        keepalive_expiry=keepalive_expiry,
    )


def _connect_retries() -> int:
    """Return the transport-level connect-establishment retry count."""
    return _env_int("EGRESS_HTTP_POOL_CONNECT_RETRIES", _CONNECT_RETRIES_DEFAULT, low=0, high=5)


def _unwrap_embedded_ipv4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return the embedded IPv4 for mapped/NAT64/6to4/Teredo forms, else ``ip``.

    ``is_link_local`` / ``is_private`` are False for these IPv6 wrappers, so the
    embedded IPv4 must be category-checked instead of the wrapper.
    """
    if not isinstance(ip, ipaddress.IPv6Address):
        return ip

    # A scope id (``%eth0``) is a routing hint, not part of address identity, but
    # ``str(ip)`` keeps it -- which would make the exact credential-endpoint
    # comparison miss ``fd00:ec2::254%eth0``. Rebuilding from the integer strips it.
    if ip.scope_id is not None:
        ip = ipaddress.IPv6Address(int(ip))

    if ip.ipv4_mapped is not None:  # ::ffff:0:0/96
        return ip.ipv4_mapped
    if ip in _NAT64_PREFIX:  # embedded v4 in the low 32 bits
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in _6TO4_PREFIX:  # embedded v4 in bits [16, 48)
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    if ip in _TEREDO_PREFIX:  # client v4 in the low 32 bits, XOR'd
        return ipaddress.IPv4Address((int(ip) & 0xFFFFFFFF) ^ 0xFFFFFFFF)
    return ip


def _ip_denial_reason(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    """Return why ``ip`` must not be an mcpgw egress target, or None if allowed.

    Private-unicast, CGNAT, and loopback are allowed (in-cluster service names and
    local dev). Everything the registry hard-denies stays denied.
    """
    ip = _unwrap_embedded_ipv4(ip)

    if str(ip) in _CREDENTIAL_ENDPOINT_IPS:
        return "cloud/workload credential endpoint"
    if ip.is_link_local:
        return "link-local"
    if ip.is_unspecified:
        return "unspecified address"
    if ip.is_multicast:
        return "multicast"
    # IPv6 loopback is also classified as reserved, so allow loopback first.
    if ip.is_loopback:
        return None
    if ip.is_reserved:
        return "reserved"
    return None


def _validated_ip(hostname: str, raw: str) -> str:
    """Return ``raw`` as a connectable IP string, or raise if it is denied."""
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError as exc:  # a resolver answer is always a literal
        raise EgressTargetError(f"{hostname}: unparseable address {raw!r}") from exc
    reason = _ip_denial_reason(ip)
    if reason is not None:
        raise EgressTargetError(f"{hostname} resolves to a blocked address ({reason}): {raw}")
    return str(ip)


async def _resolve_and_validate(hostname: str, port: int) -> list[str]:
    """Resolve ``hostname`` under a deadline and return every validated address.

    EVERY answer is validated, not just the one used: a resolver that returns a mix
    of a legitimate address and a metadata address must fail closed rather than
    depend on answer ordering. The full list is returned (de-duplicated, resolver
    order preserved) so the transport keeps httpx's multi-address connect fallback
    -- pinning to only the first answer would break a dual-stack host whose AAAA is
    unreachable (e.g. ``localhost`` -> ``::1`` with an IPv4-only listener, the mcpgw
    default ``REGISTRY_BASE_URL``).
    """
    loop = asyncio.get_running_loop()
    try:
        addr_info = await asyncio.wait_for(
            loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM),
            timeout=_DNS_RESOLUTION_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise EgressTargetError(f"{hostname}: DNS resolution timed out") from exc
    except OSError as exc:
        raise EgressTargetError(f"{hostname}: DNS resolution failed ({exc})") from exc

    ips: list[str] = []
    for info in addr_info:
        ip = _validated_ip(hostname, info[4][0])
        if ip not in ips:
            ips.append(ip)
    if not ips:
        raise EgressTargetError(f"{hostname}: DNS returned no addresses")
    return ips


def _destination(raw_url: str) -> tuple[str, int] | None:
    """Return the ``(host, effective_port)`` of a configured base URL, or None."""
    parsed = urlsplit((raw_url or "").strip())
    if not parsed.hostname or parsed.scheme not in _ALLOWED_SCHEMES:
        return None
    return (parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80))


def _allowed_destinations() -> frozenset[tuple[str, int]]:
    """Return the exact ``(host, port)`` pairs this process is configured to call.

    mcpgw has exactly two egress destinations, both operator env config: the registry
    API and the Keycloak token/JWKS endpoints. Pinning the transport to that set turns
    "any in-cluster address" into "the destinations this process was deployed to talk
    to", and makes a future tool that quietly adds a third destination fail closed
    instead of inheriting this client's credentials.

    Read from the environment rather than passed in from ``server.py`` so the transport
    owns its own policy and cannot be constructed without one. The defaults MUST match
    ``server.py``'s, which is why both modules read them from the constants above: a
    mismatch would silently reject the very destination the tools are calling. Keycloak
    is included even when M2M and OIDC are both off -- a spurious extra entry is
    harmless, a missing one is an outage.

    An empty set disables the check (the IP classification still applies). That branch
    is a safety valve, not a supported mode: it is unreachable through normal config,
    because both values fall back to parseable defaults and an unparseable
    ``REGISTRY_BASE_URL`` makes httpx reject the tools' request URLs anyway.
    """
    destinations = {
        dest
        for dest in (
            _destination(os.getenv("REGISTRY_BASE_URL", DEFAULT_REGISTRY_BASE_URL)),
            _destination(os.getenv("KEYCLOAK_INTERNAL_URL", DEFAULT_KEYCLOAK_INTERNAL_URL)),
        )
        if dest is not None
    }
    return frozenset(destinations)


class _GuardedAsyncTransport(httpx.AsyncHTTPTransport):
    """Transport that validates and pins every request before pool checkout.

    Ported from ``registry/utils/url_guard.py::GuardedAsyncTransport`` (which the
    mcpgw image cannot import). Running per request -- not per client -- is what
    makes a long-lived pooled client safe: validation cannot be skipped by a warm
    connection, and because the connect host becomes the pinned IP, the pool is
    keyed by that IP rather than by the hostname.
    """

    def __init__(
        self,
        *,
        allowed_destinations: frozenset[tuple[str, int]] = frozenset(),
        **kwargs: Any,
    ) -> None:
        self._allowed_destinations = allowed_destinations
        # Last address that connected, per (hostname, port). Tried first on the next
        # request so a warm keep-alive to it is reused; see handle_async_request.
        # Bounded by the destination allowlist (two entries in practice).
        self._last_good_ip: dict[tuple[str, int], str] = {}
        super().__init__(**kwargs)

    @staticmethod
    def _pin(
        request: httpx.Request,
        url: httpx.URL,
        hostname: str,
        ip: str,
        connect_timeout: float | None,
    ) -> httpx.Request:
        """Rewrite only the connect host, retaining Host and TLS SNI identity."""
        request.url = url.copy_with(host=ip)
        request.headers["Host"] = hostname if url.port is None else f"{hostname}:{url.port}"
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = hostname
        timeout = dict(extensions.get("timeout") or {})
        if connect_timeout is not None:
            timeout["connect"] = connect_timeout
            extensions["timeout"] = timeout
        request.extensions = extensions
        return request

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.scheme not in _ALLOWED_SCHEMES:
            raise EgressTargetError(f"unsupported scheme {url.scheme!r}")
        hostname = url.host
        if not hostname:
            raise EgressTargetError(f"{url} has no hostname")
        if url.userinfo:
            # Credentials belong in a per-request header, never in the URL, and
            # ``copy_with(host=ip)`` would carry the userinfo onto the pinned request.
            raise EgressTargetError(f"{hostname}: URL userinfo is not allowed")

        port = url.port or (443 if url.scheme == "https" else 80)
        if (
            self._allowed_destinations
            and (hostname.lower(), port) not in self._allowed_destinations
        ):
            raise EgressTargetError(
                f"{hostname}:{port} is not a configured mcpgw destination "
                "(REGISTRY_BASE_URL / KEYCLOAK_INTERNAL_URL)"
            )

        # A host that is not parseable as an IP literal -- including an obfuscated
        # spelling such as 0xA9FEA9FE or 0251.0376.0251.0376 -- deliberately falls
        # through to the resolver branch below, where getaddrinfo canonicalizes it
        # (glibc parses those forms via inet_aton) and every answer is classified.
        # Do NOT "optimize" an unparseable host into an early reject: that is what
        # makes dropping the registry's ``coerce_ip_literal`` safe here.
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None

        if literal is not None:
            # Already an address: classify it, and leave the URL alone (there is
            # nothing to pin -- httpcore will connect to exactly this address).
            reason = _ip_denial_reason(literal)
            if reason is not None:
                raise EgressTargetError(f"blocked address ({reason}): {hostname}")
            return await super().handle_async_request(request)

        validated_ips = await _resolve_and_validate(hostname, port)
        # Try the address that last connected first, then the rest in resolver order.
        # Because the pool is keyed by the PINNED IP, resolver order alone would make
        # every request open a fresh connect to an unreachable first answer (e.g.
        # `localhost` -> ::1 with an IPv4-only listener) before it could reach the warm
        # keep-alive on the second, and the failed connection attempts evict the idle
        # ones -- pooling would be off entirely. The preferred address is only used
        # while it is still among THIS request's validated answers, so every request
        # is still classified and pinned from a fresh resolution (rebind-safe).
        key = (hostname.lower(), port)
        preferred = self._last_good_ip.get(key)
        if preferred in validated_ips and validated_ips[0] != preferred:
            validated_ips.remove(preferred)
            validated_ips.insert(0, preferred)

        # Unguarded httpx hands the HOSTNAME to the connect layer, which walks the
        # addresses itself inside ONE connect budget; pinning replaces that, so the
        # budget is split across the attempts here. Without the split, an N-address
        # host would multiply the caller's timeout by N (times the transport's connect
        # retries) -- a 30s call against a dead dual-stack upstream would hang for
        # ~120s. A connect-phase failure happens before any request body is written,
        # so moving to the next address is a clean re-send (and mcpgw never streams a
        # request body).
        connect_budget = request.extensions.get("timeout", {}).get("connect")
        per_attempt = connect_budget / len(validated_ips) if connect_budget else None
        last_error: Exception | None = None
        for ip in validated_ips:
            try:
                response = await super().handle_async_request(
                    self._pin(request, url, hostname, ip, per_attempt)
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Keep the evidence for the next dual-stack/ECS debugging session:
                # only the LAST error survives to the caller.
                logger.debug("connect to %s (%s) failed: %s", hostname, ip, exc)
                last_error = exc
                continue
            self._last_good_ip[key] = ip
            return response
        if last_error is not None:
            raise last_error
        raise EgressTargetError(f"{hostname}: no validated address to connect to")


class _NoStoreCookieJar(http.cookiejar.CookieJar):
    """A cookie jar that silently drops every cookie.

    Makes cookie handling stateless on a shared client: a ``Set-Cookie`` is never
    stored, so it can never be replayed onto a later or concurrent request. This is
    concurrency-safe by construction (there is no shared cookie state to race),
    unlike clearing the jar after each response.
    """

    def set_cookie(self, cookie: http.cookiejar.Cookie) -> None:  # noqa: D102
        return  # never persist


_shared_client: httpx.AsyncClient | None = None


def shared_async_client() -> httpx.AsyncClient:
    """Return the process-lifetime, connection-pooled mcpgw egress client.

    Pass the timeout PER REQUEST. NEVER use the result as an ``async with`` target:
    ``AsyncClient.__aexit__`` closes the pool for every other caller. The
    ``is_closed`` guard lets the pool self-heal if it was closed (lifespan shutdown,
    a test) and then used again.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        # ``limits`` MUST go to the TRANSPORT: httpx.AsyncClient ignores its own
        # ``limits=`` whenever an explicit ``transport=`` is supplied
        # (``AsyncClient._init_transport`` returns the given transport untouched), so
        # passing it on the client would silently leave the pool on httpx's defaults
        # (100/20/5s) and make every EGRESS_HTTP_POOL_* value inert.
        client = httpx.AsyncClient(
            transport=_GuardedAsyncTransport(
                allowed_destinations=_allowed_destinations(),
                retries=_connect_retries(),
                limits=_pool_limits(),
            ),
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        client.cookies.jar = _NoStoreCookieJar()
        _shared_client = client
    return _shared_client


async def aclose_shared_client() -> None:
    """Close the pooled client (called from the FastMCP lifespan shutdown).

    Best-effort: a failure here must not mask the real shutdown path, and
    ``shared_async_client`` lazily rebuilds the pool if it is used again.
    """
    global _shared_client
    if _shared_client is not None:
        try:
            await _shared_client.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("error closing the shared mcpgw egress client", exc_info=True)
        _shared_client = None


def reset_shared_client_for_tests() -> None:
    """Drop the pooled-client reference without closing it (tests only)."""
    global _shared_client
    _shared_client = None


async def request_with_reconnect(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    on_reset: Callable[[], None] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request with a single transparent retry on a dead pooled keep-alive.

    A connection that the peer (registry, nginx, an LB, Keycloak) closed while it sat
    idle in the pool is indistinguishable from a live one until it is used: the first
    request after the idle gap raises ``RemoteProtocolError``. ``retries=`` on the
    transport only covers connection ESTABLISHMENT, not a reset on connection REUSE,
    and httpx never auto-retries a POST.

    Retrying is safe for every mcpgw hop because all of them are idempotent: the
    registry calls are reads, the two ``/api/search/semantic`` POSTs are searches, and
    the Keycloak ``client_credentials`` grant is not single-use (unlike an
    authorization_code / refresh_token grant, which is why the registry's 3LO refresh
    is deliberately NOT retried). The dominant case -- peer sent FIN before the
    request was written -- is a clean resend; the rare residual re-sends into a
    terminal error the caller already fails closed on. The retry re-enters the guarded
    transport, so it is validated and pinned again rather than trusting the first
    resolution.
    """
    try:
        return await client.request(method, url, **kwargs)
    except httpx.RemoteProtocolError:
        if on_reset is not None:
            on_reset()
        return await client.request(method, url, **kwargs)
