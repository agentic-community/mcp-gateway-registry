"""Hardened URL validation and rebinding-safe fetch guard (SSRF protection).

This module is the single source of truth for validating user- and
registry-controlled URLs before the registry stores them or fetches them
server-side. It consolidates the strengths of the previous partial
implementations (``skill_service._is_safe_url`` and
``ard_net_guard.assert_fetchable``) into one fail-closed guard:

- Only ``http`` / ``https`` schemes are accepted.
- The host must resolve **exclusively** to public IP addresses. Private,
  loopback, link-local, reserved, multicast, and unspecified ranges are
  blocked, along with the cloud metadata endpoint (``169.254.169.254``).
- IPv4-mapped IPv6 addresses (``::ffff:10.0.0.1``) are unwrapped before the
  range check so a private target cannot be smuggled through an IPv6 literal.
- DNS-rebinding is defeated by pinning: the fetch connects only to an IP that
  was validated inside the same transport call, so there is no window between
  the check and the connect for the hostname to rebind to a private address.
  Redirects are re-validated on every hop because httpx re-invokes the pinned
  transport for each redirect.

The guard fails closed: any error, resolution failure, or ambiguity results in
rejection rather than a permissive fallback.

Two validation profiles exist because the registry has two distinct outbound
surfaces:

- **Skill fetches** (``SKILL_PROFILE``): public-only, with an operator bypass
  allowlist read from ``settings.github_extra_hosts`` so GitHub Enterprise
  Server on an internal network stays reachable. Built-in public forge domains
  are NOT auto-trusted — they get full IP validation, closing the
  "internal host masquerading as github.com" bypass.

- **Server / agent targets** (``PROXY_PROFILE``): the same public-only default,
  but operators who legitimately proxy to internal MCP servers can opt those
  targets in via ``settings.ssrf_allowed_hosts`` / ``settings.ssrf_allowed_cidrs``.
  The cloud metadata address is never allowlistable in either profile.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

import httpx

from ..core.config import settings
from ..exceptions import UrlValidationError

logger = logging.getLogger(__name__)

# Default connect/read timeout applied to guarded fetches when a caller does not
# supply its own. Keeps a hung internal target from tying up a worker.
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# The cloud metadata endpoint can never be reached, regardless of allowlists.
_CLOUD_METADATA_IPS: frozenset[str] = frozenset({"169.254.169.254", "fd00:ec2::254"})

# The gateway's own bundled registry-tools MCP server (airegistry-tools ->
# mcpgw-server), reached over private container/service DNS (Docker Compose
# service name, ECS Service Connect alias, Kubernetes service name). This is a
# first-party component of the gateway itself, not an operator-supplied target,
# so the SSRF guard trusts it by default -- an operator upgrading to a build with
# the SSRF guard should not have to hand-configure SSRF_ALLOWED_HOSTS just to
# keep the built-in registry-tools server healthy. Operator-supplied
# ssrf_allowed_hosts are UNIONED with this set, never replace it. The cloud
# metadata endpoint is still never reachable. (The demo servers -- currenttime,
# realserverfaketools -- are opt-in via enable_demo_servers and are NOT trusted
# by default; operators who enable them add them to SSRF_ALLOWED_HOSTS.)
_BUILTIN_PROXY_ALLOWED_HOSTS: frozenset[str] = frozenset({"mcpgw-server"})

# Nginx metacharacters that must never appear in a proxy_pass_url. A valid URL
# never legitimately contains these; their presence indicates an attempt to
# break out of an nginx directive/string context (config injection).
_NGINX_METACHARACTERS: frozenset[str] = frozenset(
    {
        "\r",
        "\n",
        ";",
        "{",
        "}",
        "#",
        '"',
        "'",
        "\\",
        " ",
        "\t",
        "$",
        "\x00",
    }
)


@dataclass(frozen=True)
class _Allowlist:
    """A resolved set of hosts/CIDRs that may bypass the private-IP block."""

    hosts: frozenset[str] = field(default_factory=frozenset)
    cidrs: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    def allows_host(
        self,
        hostname_lower: str,
    ) -> bool:
        """Return True if the hostname is explicitly allowlisted."""
        return hostname_lower in self.hosts

    def allows_ip(
        self,
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        """Return True if the IP falls inside an allowlisted CIDR."""
        return any(ip in net for net in self.cidrs)


def _parse_hosts(
    raw: str,
) -> frozenset[str]:
    """Parse a comma-separated host list into a normalized frozenset."""
    return frozenset(h.strip().lower() for h in (raw or "").split(",") if h.strip())


def _parse_cidrs(
    raw: str,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse a comma-separated CIDR list, skipping malformed entries."""
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            nets.append(ipaddress.ip_network(chunk, strict=False))
        except ValueError:
            logger.warning("SSRF guard: ignoring malformed CIDR in allowlist: %r", chunk)
    return tuple(nets)


@lru_cache(maxsize=1)
def _skill_allowlist() -> _Allowlist:
    """Return the skill-fetch bypass allowlist (github_extra_hosts only).

    Built-in public forge domains are intentionally absent: they get full IP
    validation. Only operator-configured GHES hosts skip the private-IP block.
    Cached because settings are immutable per-process.
    """
    return _Allowlist(hosts=_parse_hosts(settings.github_extra_hosts))


@lru_cache(maxsize=1)
def _proxy_allowlist() -> _Allowlist:
    """Return the server/agent target bypass allowlist.

    Reads ``settings.ssrf_allowed_hosts`` and ``settings.ssrf_allowed_cidrs`` so
    operators can proxy to internal MCP servers. The bundled first-party MCP
    server hostnames (_BUILTIN_PROXY_ALLOWED_HOSTS) are always unioned in so an
    upgrade to an SSRF-guarded build keeps them healthy with zero configuration.
    Cached because settings are immutable per-process.
    """
    return _Allowlist(
        hosts=_BUILTIN_PROXY_ALLOWED_HOSTS | _parse_hosts(settings.ssrf_allowed_hosts),
        cidrs=_parse_cidrs(settings.ssrf_allowed_cidrs),
    )


@dataclass(frozen=True)
class _Profile:
    """A named validation profile: which allowlist and scheme rules apply."""

    name: str
    allowlist_factory: object  # callable returning _Allowlist


SKILL_PROFILE = _Profile(name="skill", allowlist_factory=_skill_allowlist)
PROXY_PROFILE = _Profile(name="proxy", allowlist_factory=_proxy_allowlist)


def _unwrap_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Unwrap an IPv4-mapped IPv6 address to its embedded IPv4 address."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_metadata_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return True for cloud metadata endpoints (never allowlistable)."""
    return str(ip) in _CLOUD_METADATA_IPS


def _is_blocked_ip(
    ip_str: str,
    allowlist: _Allowlist,
) -> bool:
    """Return True if an IP must not be the target of a server-side fetch.

    Blocks private, loopback, link-local, reserved, multicast, and unspecified
    ranges. The cloud metadata endpoint is always blocked. An operator CIDR
    allowlist can re-permit private ranges (but never the metadata endpoint).
    Any unparseable address is treated as blocked (fail closed).

    Args:
        ip_str: IP address string to check.
        allowlist: The profile allowlist (CIDRs that re-permit private ranges).

    Returns:
        True if the IP is unsafe to connect to, False if it is acceptable.
    """
    try:
        ip = _unwrap_ip(ipaddress.ip_address(ip_str))
    except ValueError:
        return True

    # The metadata endpoint is never reachable, even via an allowlist.
    if _is_metadata_ip(ip):
        return True

    is_dangerous = (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
    if not is_dangerous:
        return False

    # Dangerous range: only acceptable if an operator CIDR allowlist re-permits.
    return not allowlist.allows_ip(ip)


def _resolve_public_ips(
    hostname: str,
    port: int,
    allowlist: _Allowlist,
) -> list[str]:
    """Resolve a hostname and require all IPs to be acceptable.

    Args:
        hostname: The host to resolve.
        port: The destination port (passed to ``getaddrinfo``).
        allowlist: The profile allowlist (CIDRs re-permitting private ranges).

    Returns:
        The list of resolved IP address strings (all validated).

    Raises:
        UrlValidationError: If resolution fails or any resolved IP is blocked.
    """
    try:
        addr_info = socket.getaddrinfo(
            hostname,
            port,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as e:
        raise UrlValidationError(hostname, f"DNS resolution failed: {e}") from e

    ips: list[str] = []
    for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str, allowlist):
            raise UrlValidationError(
                hostname,
                f"resolves to blocked/private IP {ip_str}",
            )
        ips.append(ip_str)

    if not ips:
        raise UrlValidationError(hostname, "resolved to no addresses")

    return ips


def contains_nginx_metacharacters(
    value: str,
) -> bool:
    """Return True if a string contains characters that could break nginx config.

    Used as defense-in-depth on proxy_pass_url so a crafted URL cannot terminate
    an nginx directive or string literal even before the value reaches the
    nginx-specific escaping.

    Args:
        value: The candidate string (typically a proxy_pass_url).

    Returns:
        True if any nginx metacharacter is present.
    """
    return any(ch in value for ch in _NGINX_METACHARACTERS)


def validate_url(
    url: str,
    *,
    profile: _Profile = SKILL_PROFILE,
    require_https: bool = False,
    reject_nginx_metacharacters: bool = False,
    resolve: bool = True,
) -> list[str]:
    """Validate a URL for scheme, host, and public-IP resolution (fail closed).

    This is the registration-time / pre-fetch check. It resolves DNS and
    requires every resolved IP to be acceptable for the profile, so it also
    serves as the resolution step feeding the pinned transport.

    Args:
        url: The URL to validate.
        profile: Which allowlist/scheme rules apply (SKILL_PROFILE default,
            PROXY_PROFILE for server/agent targets).
        require_https: When True, reject non-https schemes (http is denied).
        reject_nginx_metacharacters: When True, reject URLs containing nginx
            metacharacters (used for proxy_pass_url).
        resolve: When True (default), resolve DNS and require every resolved IP
            to be acceptable. When False, only the static checks run (scheme,
            metacharacters, host presence, and literal-IP private/metadata
            block) — used at registration time, where the authoritative
            rebinding-safe defense is the pinned transport at fetch time and a
            live DNS lookup would be a fragile, network-dependent TOCTOU.

    Returns:
        The list of validated IP strings the host resolves to. Empty list when
        the host is allowlisted or when ``resolve`` is False (no pinning info).

    Raises:
        UrlValidationError: On any validation failure (fails closed).
    """
    if not url or not isinstance(url, str):
        raise UrlValidationError(str(url), "URL is empty or not a string")

    if reject_nginx_metacharacters and contains_nginx_metacharacters(url):
        raise UrlValidationError(url, "contains disallowed nginx metacharacters")

    try:
        parsed = urlparse(url)
    except Exception as e:  # pragma: no cover - urlparse rarely raises
        raise UrlValidationError(url, f"could not be parsed: {e}") from e

    allowed_schemes = ("https",) if require_https else ("http", "https")
    if parsed.scheme not in allowed_schemes:
        raise UrlValidationError(url, f"scheme '{parsed.scheme}' is not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError(url, "URL has no hostname")

    hostname_lower = hostname.lower()
    allowlist: _Allowlist = profile.allowlist_factory()  # type: ignore[operator]

    # A hostname that is itself a literal IP must still pass the range check
    # (this is always enforced, even when resolve=False, because it needs no
    # network and catches the most direct SSRF payloads like the metadata IP).
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(hostname, allowlist):
            raise UrlValidationError(url, f"targets blocked/private IP {hostname}")
        return [str(literal)]

    # Host explicitly allowlisted by the operator (e.g. GHES, or an internal
    # MCP-server host): skip the IP block. We do not pin these (the transport
    # falls back to normal DNS for them).
    if allowlist.allows_host(hostname_lower):
        logger.debug("URL guard[%s]: host '%s' is allowlisted", profile.name, hostname_lower)
        return []

    if not resolve:
        # Registration-time structural validation only. The pinned transport
        # performs the authoritative resolve-and-block at fetch time.
        return []

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _resolve_public_ips(hostname, port, allowlist)


def validate_proxy_pass_url(
    url: str,
) -> None:
    """Validate a server ``proxy_pass_url`` at registration time (fail closed).

    Rejects non-http(s) schemes, nginx metacharacters, and literal
    private/metadata IP targets. This does NOT perform a live DNS lookup: the
    authoritative rebinding-safe block for hostname targets happens at fetch
    time via the pinned guarded client (health checks). Raises on failure.

    Raises:
        UrlValidationError: On any validation failure.
    """
    validate_url(
        url,
        profile=PROXY_PROFILE,
        reject_nginx_metacharacters=True,
        resolve=False,
    )


def validate_server_path(
    path: str,
) -> None:
    """Validate a server registration ``path`` for nginx-safe characters.

    The path is interpolated into nginx ``location`` directives, so it must not
    contain characters that could terminate a directive or comment out
    surrounding config (``"``, ``;``, ``{``, ``}``, ``#``, ``$``, whitespace,
    control chars, backslash). Legitimate paths only use URL path characters, so
    this rejects rather than escapes. Fails closed.

    Args:
        path: The server path (e.g. ``/github``).

    Raises:
        UrlValidationError: If the path is empty or contains disallowed
            nginx metacharacters.
    """
    if not path or not isinstance(path, str):
        raise UrlValidationError(str(path), "server path is empty or not a string")
    if contains_nginx_metacharacters(path):
        raise UrlValidationError(path, "server path contains disallowed nginx metacharacters")


def validate_agent_url(
    url: str,
) -> None:
    """Validate an agent URL at registration time (fail closed).

    Rejects non-http(s) schemes and literal private/metadata IP targets. Like
    :func:`validate_proxy_pass_url`, this does not perform a live DNS lookup;
    the pinned guarded client blocks hostname targets that resolve private at
    fetch time (agent health check / card pull). Raises on failure.

    Raises:
        UrlValidationError: On any validation failure.
    """
    validate_url(url, profile=PROXY_PROFILE, resolve=False)


class _PinnedResolverMixin:
    """Shared logic for pinning a request to a validated IP.

    Rewrites the outgoing request so httpx connects only to an IP that this
    transport just validated, preserving the original Host header and TLS SNI.
    Because the resolve+validate+connect all happen inside the transport call
    for every request (including each redirect hop), there is no rebinding
    window and no bypassable pre-check.
    """

    _guard_profile: _Profile = SKILL_PROFILE

    def _pin_request(
        self,
        request: httpx.Request,
    ) -> httpx.Request:
        """Validate the request host and rewrite it to a pinned IP.

        Raises:
            UrlValidationError: If the target host is unsafe (fails closed).
        """
        url = request.url
        scheme = url.scheme
        if scheme not in ("http", "https"):
            raise UrlValidationError(str(url), f"scheme '{scheme}' is not allowed")

        hostname = url.host
        if not hostname:
            raise UrlValidationError(str(url), "URL has no hostname")

        hostname_lower = hostname.lower()
        allowlist: _Allowlist = self._guard_profile.allowlist_factory()  # type: ignore[operator]

        # Literal-IP host: validate directly, no rewrite needed.
        try:
            ipaddress.ip_address(hostname)
            is_literal = True
        except ValueError:
            is_literal = False

        if is_literal:
            if _is_blocked_ip(hostname, allowlist):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return request

        # Allowlisted host: do not pin (resolve normally).
        if allowlist.allows_host(hostname_lower):
            return request

        port = url.port or (443 if scheme == "https" else 80)
        pinned_ips = _resolve_public_ips(hostname, port, allowlist)
        pinned_ip = pinned_ips[0]

        # Rewrite the connection target to the validated IP while preserving the
        # original hostname for the Host header and TLS SNI.
        request.url = url.copy_with(host=pinned_ip)
        request.headers["Host"] = hostname if url.port is None else f"{hostname}:{url.port}"
        request.extensions = dict(request.extensions)
        request.extensions["sni_hostname"] = hostname
        return request


class GuardedTransport(_PinnedResolverMixin, httpx.HTTPTransport):
    """Synchronous httpx transport that pins requests to validated IPs."""

    def __init__(
        self,
        *,
        guard_profile: _Profile = SKILL_PROFILE,
        **kwargs: object,
    ) -> None:
        self._guard_profile = guard_profile
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        request = self._pin_request(request)
        return super().handle_request(request)


class GuardedAsyncTransport(_PinnedResolverMixin, httpx.AsyncHTTPTransport):
    """Async httpx transport that pins requests to validated IPs."""

    def __init__(
        self,
        *,
        guard_profile: _Profile = SKILL_PROFILE,
        **kwargs: object,
    ) -> None:
        self._guard_profile = guard_profile
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        request = self._pin_request(request)
        return await super().handle_async_request(request)


def guarded_client(
    *,
    profile: _Profile = SKILL_PROFILE,
    timeout: float | httpx.Timeout | None = None,
    verify: bool = True,
    **kwargs: object,
) -> httpx.Client:
    """Return a sync httpx.Client that is SSRF/rebinding-safe.

    Every request (and redirect hop) made through this client is validated and
    pinned by :class:`GuardedTransport`. Use this in place of ``httpx.Client``
    for any fetch built from user/registry-controlled URLs.
    """
    resolved_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS
    return httpx.Client(
        transport=GuardedTransport(guard_profile=profile, verify=verify),
        timeout=resolved_timeout,
        **kwargs,  # type: ignore[arg-type]
    )


def guarded_async_client(
    *,
    profile: _Profile = SKILL_PROFILE,
    timeout: float | httpx.Timeout | None = None,
    verify: bool = True,
    **kwargs: object,
) -> httpx.AsyncClient:
    """Return an async httpx.AsyncClient that is SSRF/rebinding-safe.

    Every request (and redirect hop) made through this client is validated and
    pinned by :class:`GuardedAsyncTransport`. Use this in place of
    ``httpx.AsyncClient`` for any fetch built from user/registry-controlled
    URLs.
    """
    resolved_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS
    return httpx.AsyncClient(
        transport=GuardedAsyncTransport(guard_profile=profile, verify=verify),
        timeout=resolved_timeout,
        **kwargs,  # type: ignore[arg-type]
    )
