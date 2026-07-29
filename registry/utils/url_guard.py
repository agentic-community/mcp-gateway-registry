"""Hardened URL validation and rebinding-safe fetch guard (SSRF protection).

This module is the single source of truth for validating user- and
registry-controlled URLs before the registry stores them or fetches them
server-side. It consolidates the strengths of the previous partial
implementations (``skill_service._is_safe_url`` and
``ard_net_guard.assert_fetchable``) into one fail-closed guard:

- Only ``http`` / ``https`` schemes are accepted.
- The host must resolve **exclusively** to public IP addresses. IP classification
  is delegated to the shared ``registry.utils.ip_guard`` (one classifier
  repo-wide): private, loopback, link-local, reserved, multicast, unspecified, and
  CGNAT ranges are blocked, along with the cloud metadata endpoints
  (``169.254.169.254`` AND the IPv6 ``fd00:ec2::254``) which are NEVER reachable —
  not via the coarse bool nor an explicit CIDR allowlist.
- Obfuscated IPv4 literals (hex/octal/decimal/trailing-dot) and embedded-IPv4 IPv6
  transports (mapped ``::ffff:``, NAT64, 6to4, Teredo) are recognized and
  category-checked, so a private/metadata target cannot be smuggled through a
  non-canonical spelling.
- Explicit host/CIDR allowlists can admit private, loopback, and CGNAT targets,
  but cloud/workload credential endpoints, link-local, unspecified, reserved, and
  multicast destinations remain hard-denied.
- Hostname trust (``ssrf_allowed_hosts`` / ``github_extra_hosts``) is a
  post-resolution relaxation, not a DNS bypass. Trusted names are still resolved,
  every answer is classified, and the request is pinned to a validated answer.
  Explicit hostname trust can admit non-credential private/internal addresses,
  but EC2/ECS/EKS credential endpoints remain hard-denied before that relaxation.
- The bundled ``mcpgw-server`` name is reserved and is not in ``PROXY_PROFILE``'s
  global hostname set. A dedicated profile is selected only for the exact built-in
  MCP entity ``/airegistry-tools/`` with target ``http://mcpgw-server:8003/``.
- DNS-rebinding is defeated by pinning: the fetch connects only to an IP that
  was validated inside the same transport call, so there is no window between
  the check and the connect for the hostname to rebind to a private address.
  Redirects are re-validated on every hop because httpx re-invokes the pinned
  transport for each redirect.

The guard fails closed: any error, resolution failure, or ambiguity results in
rejection rather than a permissive fallback.

Validation profiles separate the registry's distinct outbound trust surfaces:

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

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx

from ..core.config import settings
from ..exceptions import UrlValidationError
from .ip_guard import coerce_ip_literal, ip_denial_reason

logger = logging.getLogger(__name__)

# Default connect/read timeout applied to guarded fetches when a caller does not
# supply its own. Keeps a hung internal target from tying up a worker.
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# NOTE: IP classification (cloud-metadata hard-deny incl. the v6 fd00:ec2::254,
# CGNAT, obfuscated literals, embedded-IPv4 unwrapping, the private/loopback/
# reserved/multicast categories, and the two-tier allowlist) lives in the shared
# ``registry.utils.ip_guard`` module — this file delegates to it via
# ``_is_blocked_ip`` so there is ONE classifier repo-wide.

# Server path names that collide with the cross-server wildcard sentinels. A
# registration path like ``/all`` or ``/*`` normalizes (lstrip("/")) to the
# server-scope value ``all`` / ``*``, which the scope resolver promotes to a
# full cross-server wildcard, silently granting the registrant access to every
# server. These names are therefore reserved and rejected at registration.
#
# Kept case-insensitive on match: lstrip("/") preserves case, and while the
# current resolver compare is case-sensitive, we reject the whole family so a
# future or operator-UI case-insensitive compare cannot reopen the escalation.
#
# MUST stay in sync with registry.auth.access_resolver._WILDCARD_VALUES (the
# read-side sentinel set). Defined locally rather than imported to keep this
# low-level util free of a dependency on the auth/repository layers.
_RESERVED_SERVER_PATH_NAMES: frozenset[str] = frozenset({"all", "*"})

# Bound DNS work independently of the HTTP connect/read timeout. Async guarded
# transports use the event loop resolver under this deadline, so a slow or
# adversarial resolver cannot block the event loop indefinitely.
_DNS_RESOLUTION_TIMEOUT_SECONDS: float = 5.0

# The only implicit private-host trust in the product: the bundled
# airegistry-tools MCP server. Trust is selected only when ALL three identity
# dimensions match (MCP entity type, registered path, normalized full target).
# The hostname remains reserved in every ordinary profile so a custom entity,
# skill, agent, or differently named MCP server cannot borrow this exception.
_BUILTIN_AIREGISTRY_TOOLS_ENTITY_TYPE = "mcp_server"
_BUILTIN_AIREGISTRY_TOOLS_PATH = "/airegistry-tools/"
_BUILTIN_AIREGISTRY_TOOLS_TARGET = "http://mcpgw-server:8003/"
# Exact request identities emitted by current MCP/health clients for the
# built-in base URL. Query strings are intentionally absent and therefore
# rejected; path, host, effective port, and query all participate in identity.
_BUILTIN_AIREGISTRY_TOOLS_OUTBOUND_IDENTITIES: frozenset[str] = frozenset(
    {
        _BUILTIN_AIREGISTRY_TOOLS_TARGET,
        "http://mcpgw-server:8003/mcp",
        "http://mcpgw-server:8003/mcp/",
    }
)
_RESERVED_BUILTIN_PROXY_HOSTS: frozenset[str] = frozenset({"mcpgw-server"})

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
    """A resolved set of hosts/CIDRs that relax the IP block.

    ``cidrs`` explicitly permit private, loopback, and CGNAT destinations.
    ``hosts`` is a post-resolution trust signal with the same limited
    relaxation. Hard-denied credential, link-local, unspecified, reserved, and
    multicast categories remain closed in both cases.
    """

    hosts: frozenset[str] = field(default_factory=frozenset)
    cidrs: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    def allows_host(
        self,
        hostname_lower: str,
    ) -> bool:
        """Return True if the hostname is explicitly allowlisted."""
        return hostname_lower in self.hosts


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


def normalize_url_identity(url: str) -> str:
    """Return the canonical full URL used for target identity comparisons.

    Scheme and hostname are lower-cased, IDNs are converted to ASCII, default
    ports are omitted, an empty path becomes ``/``, and the complete path and
    query are preserved. Userinfo and fragments are rejected because neither is
    a legitimate proxy target and both create ambiguous/log-sensitive identities.
    """
    if not url or not isinstance(url, str):
        raise UrlValidationError(str(url), "URL is empty or not a string")
    try:
        parsed = urlsplit(url)
        port = parsed.port  # force validation of malformed/out-of-range ports
    except (TypeError, ValueError) as exc:
        raise UrlValidationError(url, f"could not be parsed: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UrlValidationError(url, f"scheme '{parsed.scheme}' is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UrlValidationError(url, "URL userinfo is not allowed")
    if parsed.fragment:
        raise UrlValidationError(url, "URL fragments are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError(url, "URL has no hostname")

    try:
        normalized_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise UrlValidationError(url, f"hostname is invalid: {exc}") from exc
    host_for_netloc = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    default_port = 443 if scheme == "https" else 80
    netloc = host_for_netloc if port in (None, default_port) else f"{host_for_netloc}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def sanitized_url_for_log(url: str) -> str:
    """Return a URL safe for logs with userinfo, query, and fragment removed."""
    try:
        normalized = normalize_url_identity(url)
        parsed = urlsplit(normalized)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except Exception:
        return "[invalid URL]"


def _normalized_registered_path(path: str | None) -> str:
    """Normalize a registry path for exact built-in identity matching."""
    clean = (path or "").strip("/")
    return f"/{clean}/" if clean else "/"


def is_builtin_airegistry_tools_target(
    entity_type: str,
    entity_path: str | None,
    target_url: str | None,
) -> bool:
    """Return whether an entity is the exact bundled airegistry-tools target."""
    if entity_type != _BUILTIN_AIREGISTRY_TOOLS_ENTITY_TYPE:
        return False
    if _normalized_registered_path(entity_path) != _BUILTIN_AIREGISTRY_TOOLS_PATH:
        return False
    try:
        return normalize_url_identity(target_url) == _BUILTIN_AIREGISTRY_TOOLS_TARGET
    except UrlValidationError:
        return False


@lru_cache(maxsize=1)
def _skill_allowlist() -> _Allowlist:
    """Return the skill-fetch bypass allowlist (github_extra_hosts only).

    Built-in public forge domains are intentionally absent: they get full IP
    validation. Only operator-configured GHES hosts may relax private-IP checks,
    and those hosts are still resolved, classified, and pinned.
    Cached because settings are immutable per-process.
    """
    return _Allowlist(hosts=_parse_hosts(settings.github_extra_hosts))


@lru_cache(maxsize=1)
def _proxy_allowlist() -> _Allowlist:
    """Return the ordinary server/agent/generic target allowlist.

    The bundled ``mcpgw-server`` hostname is intentionally absent. It is reserved
    and only admitted through ``BUILTIN_AIREGISTRY_TOOLS_PROFILE`` after exact
    entity/path/full-target matching.
    """
    return _Allowlist(
        hosts=_parse_hosts(settings.ssrf_allowed_hosts),
        cidrs=_parse_cidrs(settings.ssrf_allowed_cidrs),
    )


@lru_cache(maxsize=1)
def _builtin_airegistry_tools_allowlist() -> _Allowlist:
    """Return proxy policy plus the one exact built-in private hostname."""
    ordinary = _proxy_allowlist()
    return _Allowlist(
        hosts=ordinary.hosts | _RESERVED_BUILTIN_PROXY_HOSTS,
        cidrs=ordinary.cidrs,
    )


def _federation_allowlist() -> _Allowlist:
    """Return the peer-federation allowlist: deliberately empty (no bypass).

    Peer federation attaches a bearer credential to server-side requests and
    connects to a registrant-supplied endpoint, so it must never inherit any
    private-IP bypass. An empty allowlist means every private/loopback/
    link-local/reserved/metadata address is blocked outright — an operator
    ``github_extra_hosts``/``ssrf_allowed_hosts`` entry cannot re-permit a
    private target on the federation path. This must match the empty allowlist
    the write-time endpoint guard uses so write-time and fetch-time validation
    share one trust boundary.
    """
    return _Allowlist()


@dataclass(frozen=True)
class _Profile:
    """A named validation profile and optional exact outbound identities."""

    name: str
    allowlist_factory: object  # callable returning _Allowlist
    allowed_url_identities: frozenset[str] | None = None


SKILL_PROFILE = _Profile(name="skill", allowlist_factory=_skill_allowlist)
PROXY_PROFILE = _Profile(name="proxy", allowlist_factory=_proxy_allowlist)
BUILTIN_AIREGISTRY_TOOLS_PROFILE = _Profile(
    name="builtin-airegistry-tools",
    allowlist_factory=_builtin_airegistry_tools_allowlist,
    allowed_url_identities=_BUILTIN_AIREGISTRY_TOOLS_OUTBOUND_IDENTITIES,
)
FEDERATION_PROFILE = _Profile(name="federation", allowlist_factory=_federation_allowlist)


def proxy_profile_for_entity_target(
    entity_type: str,
    entity_path: str | None,
    registered_target_url: str | None,
    outbound_url: str | None = None,
) -> _Profile:
    """Select trust using the registered identity and exact outbound identity.

    A record whose registered target is the built-in base may use only the
    small, explicit set of URLs emitted by the MCP and health clients. A
    different override URL fails closed instead of silently falling back to the
    ordinary profile, even when that different URL is otherwise public.
    """
    if not is_builtin_airegistry_tools_target(entity_type, entity_path, registered_target_url):
        return PROXY_PROFILE

    candidate = outbound_url or registered_target_url
    normalized = normalize_url_identity(candidate)
    if normalized not in _BUILTIN_AIREGISTRY_TOOLS_OUTBOUND_IDENTITIES:
        raise UrlValidationError(
            candidate,
            "outbound URL does not match the exact built-in airegistry-tools identity",
        )
    return BUILTIN_AIREGISTRY_TOOLS_PROFILE


def _is_blocked_ip(
    ip_str: str,
    allowlist: _Allowlist,
    *,
    trusted_hostname: bool = False,
) -> bool:
    """Return True if an IP must not be the target of a server-side fetch.

    ``trusted_hostname`` represents an explicit operator hostname allowlist or
    the exact built-in profile. It preserves the old ability to reach internal
    addresses, but only *after* DNS resolution and canonical classification. The
    resolved address is added as an exact CIDR, so ``ip_denial_reason`` still
    executes its cloud/workload credential hard-deny before the relaxation.
    """
    ip = coerce_ip_literal(ip_str)
    if ip is None:
        return True
    allowed_cidrs = allowlist.cidrs
    if trusted_hostname:
        allowed_cidrs = (*allowed_cidrs, ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}"))
    return (
        ip_denial_reason(
            ip,
            allow_private=False,
            allowed_cidrs=allowed_cidrs,
        )
        is not None
    )


def _validate_resolved_ips(
    hostname: str,
    addr_info: list[tuple],
    allowlist: _Allowlist,
) -> list[str]:
    """Validate every resolver answer and return a de-duplicated IP list."""
    trusted_hostname = allowlist.allows_host(hostname.lower())
    ips: list[str] = []
    for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
        ip_str = str(sockaddr[0])
        if _is_blocked_ip(ip_str, allowlist, trusted_hostname=trusted_hostname):
            raise UrlValidationError(hostname, f"resolves to blocked/private IP {ip_str}")
        if ip_str not in ips:
            ips.append(ip_str)
    if not ips:
        raise UrlValidationError(hostname, "resolved to no addresses")
    return ips


def _resolve_public_ips(
    hostname: str,
    port: int,
    allowlist: _Allowlist,
) -> list[str]:
    """Synchronously resolve, classify, and return every destination IP."""
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlValidationError(hostname, f"DNS resolution failed: {exc}") from exc
    return _validate_resolved_ips(hostname, addr_info, allowlist)


async def _resolve_public_ips_async(
    hostname: str,
    port: int,
    allowlist: _Allowlist,
) -> list[str]:
    """Resolve without blocking the event loop, under a fixed DNS deadline."""
    loop = asyncio.get_running_loop()
    try:
        addr_info = await asyncio.wait_for(
            loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP),
            timeout=_DNS_RESOLUTION_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise UrlValidationError(hostname, "DNS resolution timed out") from exc
    except socket.gaierror as exc:
        raise UrlValidationError(hostname, f"DNS resolution failed: {exc}") from exc
    return _validate_resolved_ips(hostname, addr_info, allowlist)


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
        The list of validated IP strings the host resolves to. Empty only when
        ``resolve`` is False (no pinning information).

    Raises:
        UrlValidationError: On any validation failure (fails closed).
    """
    if not url or not isinstance(url, str):
        raise UrlValidationError(str(url), "URL is empty or not a string")

    if reject_nginx_metacharacters and contains_nginx_metacharacters(url):
        raise UrlValidationError(url, "contains disallowed nginx metacharacters")

    normalized = normalize_url_identity(url)
    if (
        profile.allowed_url_identities is not None
        and normalized not in profile.allowed_url_identities
    ):
        raise UrlValidationError(url, "URL does not match the selected exact outbound identity")
    parsed = urlparse(normalized)

    allowed_schemes = ("https",) if require_https else ("http", "https")
    if parsed.scheme not in allowed_schemes:
        raise UrlValidationError(url, f"scheme '{parsed.scheme}' is not allowed")

    hostname = parsed.hostname
    if not hostname:  # normalize_url_identity already enforces this; defensive
        raise UrlValidationError(url, "URL has no hostname")

    hostname_lower = hostname.lower()
    allowlist: _Allowlist = profile.allowlist_factory()  # type: ignore[operator]

    # A hostname that is itself a literal IP must still pass the range check
    # (this is always enforced, even when resolve=False, because it needs no
    # network and catches the most direct SSRF payloads like the metadata IP).
    # Use coerce_ip_literal (not ipaddress.ip_address) so obfuscated spellings
    # (hex/octal/decimal/embedded-v4) are recognized as IPs and category-checked,
    # not mistaken for opaque hostnames.
    literal = coerce_ip_literal(hostname)
    if literal is not None:
        if _is_blocked_ip(hostname, allowlist):
            raise UrlValidationError(url, f"targets blocked/private IP {hostname}")
        return [str(literal)]

    # ``mcpgw-server`` is a first-party internal identity, not a globally trusted
    # destination. Ordinary entities are rejected even before DNS; only the
    # exact airegistry-tools entity/path/full-target selector receives the
    # dedicated profile that can admit it.
    if (
        hostname_lower in _RESERVED_BUILTIN_PROXY_HOSTS
        and profile is not BUILTIN_AIREGISTRY_TOOLS_PROFILE
    ):
        raise UrlValidationError(
            url, "hostname is reserved for the built-in airegistry-tools server"
        )

    if not resolve:
        # Registration-time structural validation only. The DNS-aware service
        # check and pinned transport perform authoritative resolution.
        return []

    # Explicitly trusted hostnames are still resolved and classified. Their
    # resolved non-credential addresses are relaxed in _validate_resolved_ips,
    # then returned for connection pinning; cloud/workload credential endpoints
    # remain hard-denied before that relaxation.
    if allowlist.allows_host(hostname_lower):
        logger.debug("URL guard[%s]: resolving trusted host '%s'", profile.name, hostname_lower)

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _resolve_public_ips(hostname, port, allowlist)


def validate_proxy_pass_url(
    url: str,
    *,
    server_path: str | None = None,
) -> None:
    """Validate a server ``proxy_pass_url`` at registration time (fail closed).

    Rejects non-http(s) schemes, nginx metacharacters, and literal
    private/metadata IP targets. This does NOT perform a live DNS lookup: the
    authoritative rebinding-safe block for hostname targets happens at fetch
    time via the pinned guarded client (health checks). Raises on failure.

    Raises:
        UrlValidationError: On any validation failure.
    """
    profile = proxy_profile_for_entity_target("mcp_server", server_path, url)
    validate_url(
        url,
        profile=profile,
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

    The path is also turned into a scope ``server`` value (via ``lstrip("/")``),
    so a path that normalizes to a cross-server wildcard sentinel (``all`` /
    ``*``) is rejected: such a value would silently grant access to every server
    in the registry (see :data:`_RESERVED_SERVER_PATH_NAMES`).

    Args:
        path: The server path (e.g. ``/github``).

    Raises:
        UrlValidationError: If the path is empty, contains disallowed nginx
            metacharacters, or normalizes to a reserved cross-server wildcard
            name.
    """
    if not path or not isinstance(path, str):
        raise UrlValidationError(str(path), "server path is empty or not a string")
    if contains_nginx_metacharacters(path):
        raise UrlValidationError(path, "server path contains disallowed nginx metacharacters")

    # Normalize the same way the scope layer does (add_server_scope does
    # server_path.lstrip("/")), then also drop trailing slashes so "/all/" and
    # "//all//" collapse to the same reserved name. Compare case-insensitively.
    # A path that normalizes to empty (e.g. "/") is left to existing handling:
    # an empty server name is falsy and grants no access in the resolver, so it
    # is not part of this escalation and must stay registerable.
    normalized = path.strip("/")
    if normalized.lower() in _RESERVED_SERVER_PATH_NAMES:
        raise UrlValidationError(
            path,
            f"server path '{normalized}' is reserved (collides with the cross-server wildcard)",
        )


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
    """Shared validate-and-pin logic for sync and async guarded transports."""

    _guard_profile: _Profile = SKILL_PROFILE

    def _request_target(
        self,
        request: httpx.Request,
    ) -> tuple[httpx.URL, str, str, int, _Allowlist]:
        """Perform structural checks and return normalized pinning inputs."""
        url = request.url
        # Reuse canonical validation for scheme/host/userinfo/fragment/reserved
        # built-in checks, but deliberately leave DNS to the transport-specific
        # resolver below.
        validate_url(str(url), profile=self._guard_profile, resolve=False)
        scheme = url.scheme
        hostname = url.host
        if not hostname:  # validate_url already enforces this; defensive
            raise UrlValidationError(str(url), "URL has no hostname")
        port = url.port or (443 if scheme == "https" else 80)
        allowlist: _Allowlist = self._guard_profile.allowlist_factory()  # type: ignore[operator]
        return url, scheme, hostname, port, allowlist

    @staticmethod
    def _rewrite_to_pinned_ip(
        request: httpx.Request,
        url: httpx.URL,
        hostname: str,
        pinned_ip: str,
    ) -> httpx.Request:
        """Rewrite only the connect host, retaining Host and TLS SNI identity."""
        request.url = url.copy_with(host=pinned_ip)
        request.headers["Host"] = hostname if url.port is None else f"{hostname}:{url.port}"
        request.extensions = dict(request.extensions)
        request.extensions["sni_hostname"] = hostname
        return request

    def _pin_request(
        self,
        request: httpx.Request,
    ) -> httpx.Request:
        """Synchronously validate, resolve, and pin a request."""
        url, _scheme, hostname, port, allowlist = self._request_target(request)
        if coerce_ip_literal(hostname) is not None:
            if _is_blocked_ip(hostname, allowlist):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return request
        pinned_ip = _resolve_public_ips(hostname, port, allowlist)[0]
        return self._rewrite_to_pinned_ip(request, url, hostname, pinned_ip)

    async def _pin_request_async(
        self,
        request: httpx.Request,
    ) -> httpx.Request:
        """Asynchronously validate, resolve under deadline, and pin a request."""
        url, _scheme, hostname, port, allowlist = self._request_target(request)
        if coerce_ip_literal(hostname) is not None:
            if _is_blocked_ip(hostname, allowlist):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return request
        pinned_ip = (await _resolve_public_ips_async(hostname, port, allowlist))[0]
        return self._rewrite_to_pinned_ip(request, url, hostname, pinned_ip)


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
        request = await self._pin_request_async(request)
        return await super().handle_async_request(request)


def guarded_client(
    *,
    profile: _Profile = SKILL_PROFILE,
    timeout: float | httpx.Timeout | None = None,
    verify: bool | str = True,
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
    verify: bool | str = True,
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
