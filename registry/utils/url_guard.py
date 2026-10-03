"""Hardened URL validation and rebinding-safe fetch guard (SSRF protection).

This module is the single source of truth for validating user- and
registry-controlled URLs before the registry stores them or fetches them
server-side. It consolidates the strengths of the previous partial
implementations (``skill_service._is_safe_url`` and
``ard_net_guard.assert_fetchable``) into one fail-closed guard:

- Only ``http`` / ``https`` schemes are accepted.
- The host must resolve **exclusively** to public IP addresses. IP classification
  lives in this module (the ``coerce_ip_literal`` / ``_ip_denial_reason`` pair,
  the one classifier repo-wide): private, loopback, link-local, reserved,
  multicast, unspecified, and
  CGNAT ranges are blocked, along with the cloud metadata endpoints
  (including AWS endpoints, IPv6 ``fd00:ec2::*``, and Alibaba
  ``100.100.100.200``) which are NEVER reachable —
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
  Cloud/workload credential endpoints are never allowlistable in any profile.

- **Credential-bearing OAuth token endpoints**
  (``CREDENTIALED_OAUTH_PROFILE``): HTTPS-only, and public-only unless the
  operator names a trusted IdP host in ``settings.egress_oauth_trusted_idp_hosts``
  (hosts only — no CIDRs, no wildcards — and empty by default). Token POSTs carry
  client secrets, refresh tokens, or user assertions, so this profile must never
  inherit the proxy profile's internal-target bypass: it reads only its own
  setting, and an ``ssrf_allowed_hosts``/``github_extra_hosts`` entry cannot
  re-permit a token endpoint.

Request lifecycle (validate -> resolve -> pin -> re-validate per redirect)::

    registration time                 fetch time (per request AND per redirect)
    -----------------                 -----------------------------------------
    validate_url(resolve=False)       guarded_client / guarded_async_client
      | scheme / userinfo / host         |
      | nginx metacharacters             v
      | literal-IP category check     GuardedTransport.handle_request
      | (metadata/private denied)        |
      v                                  v
    store target                      _pin_request  (once per hop)
                                         | validate_url(resolve=False)  <- static re-check
                                         | coerce_ip_literal(host)?
                                         |    yes -> _is_blocked_ip -> deny metadata/private
                                         |    no  -> _resolve_public_ips (DNS now)
                                         v         every A/AAAA classified, else deny
                                      rewrite connect host -> pinned public IP
                                         | keep Host header + TLS SNI = original name
                                         v
                                      super().handle_request  (only public IP is dialed)
                                         |
                                         v
                                      3xx? httpx re-issues Location through this
                                           same transport -> _pin_request runs again,
                                           so a redirect to 169.254.169.254 is denied
                                           at the SECOND hop before any connect.

    Fail closed everywhere: any resolution failure, un-encodable host, identity
    mismatch, or ambiguity raises UrlValidationError instead of falling through.

Forward-proxy egress (``EGRESS_FORWARD_PROXY_ENABLED``, OFF by default)
----------------------------------------------------------------------

httpx wires environment-proxy support only into its own default transport
(``allow_env_proxies = trust_env and transport is None``), so a custom transport
such as :class:`GuardedAsyncTransport` never sees ``HTTP_PROXY`` / ``HTTPS_PROXY``.
Where the registry has no direct internet egress, every guarded fetch therefore
times out. Setting ``proxy=`` on the enclosing client is NOT the fix: httpx would
mount a plain transport under ``all://`` that shadows the guarded one, silently
removing every check in this module. The routing decision lives inside the
transport instead, so the proxied path cannot skip the guard.

When the flag is on, a target is proxied only if ALL of these hold; otherwise it
takes the direct, pinned path above, byte for byte:

1. a proxy is configured for the request's scheme, and
2. ``NO_PROXY`` does not name the host, and
3. every address the host resolves to is on the public internet.

The relaxation this buys is narrow and is the one thing to know: **for a proxied
target, IP-rebind pinning is dropped.** It cannot be kept. A ``CONNECT`` tunnel
builds its TLS handshake with ``server_hostname`` taken from the request URL
(``httpcore/_async/http_proxy.py``) and never reads the ``sni_hostname``
extension that pinning sets, so a pinned request through a tunnel would verify
the certificate against an IP and fail. The compensating controls:

- Resolution and classification still run, in full, BEFORE any ``CONNECT``: the
  cloud/workload-credential, metadata, link-local, unspecified, reserved and
  multicast hard-denies, and the profile's allowlist. One blocked answer denies
  the request.
- Routing is derived from that classification rather than from an operator list,
  so every internal target keeps its pin automatically, whether or not anyone
  remembered to list it in ``NO_PROXY``. A mixed public/internal answer set
  counts as internal and keeps the pin.
- Per-redirect re-validation is unchanged: httpx re-invokes the transport per
  hop, so the routing decision is remade per hop.
- Credential-bearing profiles accept only ``https`` through a proxy, so a secret
  rides inside the tunnel and the proxy sees only ``host:port``.

What remains is that the proxy re-resolves the name when it dials, so a name
that passes classification here and then rebinds is not caught by this module.
Such a dial is made from the PROXY's network position, not the registry's, so
the operator guidance is for the proxy itself to deny ``CONNECT`` to RFC-1918,
loopback and link-local destinations. See ``docs/forward-proxy-egress.md``.
"""

from __future__ import annotations

import asyncio
import http.cookiejar
import ipaddress
import logging
import os
import socket
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlsplit, urlunsplit

import certifi
import httpx

from ..common.log_redaction import redact_url
from ..exceptions import UrlValidationError

if TYPE_CHECKING:
    from ..core.config import Settings

logger = logging.getLogger(__name__)


# Registry settings are bound lazily on first use rather than imported at module
# load: importing ``registry.core.config`` eagerly builds the full ``Settings()``
# (SECRET_KEY, DocumentDB, etc.), which a standalone tool that only needs the
# SSRF/URL guard (e.g. the AgentCore sync sidecar in ``cli/agentcore``) must not
# require. This stays ``None`` until an allowlist factory actually needs operator
# config, so ``import registry.utils.url_guard`` — and ``validate_url`` /
# ``guarded_*`` with the default (empty) allowlist, e.g. FEDERATION_PROFILE —
# work with zero registry server configuration. Tests patch this attribute
# directly (``patch("registry.utils.url_guard.settings", ...)``).
settings: Settings | None = None


def _get_settings() -> Settings:
    """Return the registry settings object, binding it lazily on first use.

    Honors a caller/test-supplied module-level ``settings`` (patched in tests);
    otherwise imports ``registry.core.config.settings`` on first access so the
    heavy config is never built merely by importing this module.
    """
    if settings is not None:
        return settings
    from ..core.config import settings as _resolved

    return _resolved


# Default connect/read timeout applied to guarded fetches when a caller does not
# supply its own. Keeps a hung internal target from tying up a worker.
_DEFAULT_TIMEOUT_SECONDS: float = 15.0

# IP-category SSRF classification. This is the ONE place in the project that
# knows about the bypass tricks — obfuscated IPv4 literal encodings
# (decimal/octal/hex/short-form), IPv6 transports that embed an IPv4 address
# (IPv4-mapped ::ffff:0:0/96, NAT64 64:ff9b::/96, 6to4 2002::/16, Teredo
# 2001::/32), and the always-deny categories (link-local/metadata, unspecified,
# reserved, multicast). Every outbound-guard call site funnels through
# ``_is_blocked_ip`` (literal targets) or ``_resolve_public_ips`` (hostnames),
# both of which delegate to ``coerce_ip_literal`` + ``_ip_denial_reason`` below.
#
# Scope caveat: default-mode completeness (``allow_private=False``) relies on
# CPython's ``ipaddress`` category flags, whose treatment of special-purpose
# ranges can evolve between Python releases. The repo pins Python 3.14 and the
# tests pin the security-relevant ranges so a downgrade or semantic change fails
# loudly.

# IPv6 transports that embed an IPv4 address. Each carries a reachable IPv4 in
# its bits, but Python classifies the wrapper as neither link-local nor private
# (except where it happens to fall in ::/8), so without explicit unwrapping a URL
# like http://[64:ff9b::a9fe:a9fe]/ or http://[2002:a9fe:a9fe::]/ would reach the
# IPv4 metadata endpoint. We extract and category-check the embedded v4.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052
_6TO4_PREFIX = ipaddress.ip_network("2002::/16")  # RFC 3056: 2002:V4:V4::/48
_TEREDO_PREFIX = ipaddress.ip_network("2001::/32")  # RFC 4380: client v4 in low 32 bits, XOR'd

# CGNAT shared address space (RFC 6598). CPython's is_private does NOT flag this
# range, but it is internally routable (carrier NAT / on-cluster overlays), so we
# treat it like private-unicast: denied by default, relaxable with allow_private.
_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")

# Cloud metadata and workload-identity credential endpoints. NEVER reachable:
# not relaxable by ``allow_private`` NOR by an explicit CIDR/hostname allowlist.
# Checked as an explicit set BEFORE any category logic because several addresses
# are private/link-local and would otherwise be reopened by a relaxation:
# - EC2 IMDS: 169.254.169.254 / fd00:ec2::254
# - ECS task credentials: 169.254.170.2
# - EKS Pod Identity: 169.254.170.23 / fd00:ec2::23
# - Alibaba Cloud ECS metadata: 100.100.100.200
_CREDENTIAL_ENDPOINT_IPS: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
        "169.254.170.2",
        "169.254.170.23",
        "fd00:ec2::23",
        "100.100.100.200",
    }
)


def _parse_inet_aton_part(part: str) -> int | None:
    """Parse one IPv4 part with inet_aton radix rules, or None if not numeric.

    inet_aton reads ``0x``-prefixed parts as hex, other leading-``0`` parts as
    OCTAL (so ``0251`` == 169, not 251), and the rest as decimal. Python's
    ``int(x, 0)`` rejects bare leading-zero octal (``0251``), so we handle the
    radices explicitly to match glibc/nginx exactly.
    """
    if part == "":
        return None
    low = part.lower()
    try:
        if low.startswith("0x"):
            return int(part, 16)
        if part.startswith("0") and part != "0":
            return int(part, 8)
        return int(part, 10)
    except ValueError:
        return None


def coerce_ip_literal(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a host as an IP, including non-canonical IPv4 spellings.

    ``ipaddress.ip_address`` only accepts canonical dotted-quad / hex-IPv6, so an
    attacker could smuggle the metadata IP past a naive check as a decimal
    (``2852039166``), octal (``0251.0376.0251.0376``), hex (``0xA9FEA9FE``), or
    trailing-dot (``169.254.169.254.``) literal — all of which ``inet_aton`` (and
    therefore glibc's resolver and nginx) interpret as a real IPv4 address. We
    match inet_aton for all realistic literal spellings; Python's ``int()`` is a
    lenient superset on exotic inputs (underscores, ``0o`` prefixes), but only in
    the safe direction — those parse to a number and get category-checked (more
    denial), never fewer.

    Args:
        host: The host portion of a URL (brackets already stripped).

    Returns:
        The parsed address (IPv4 or IPv6), or None if ``host`` is a genuine
        hostname that must be resolved downstream before it can be checked.
    """
    # Canonical parse first (covers all IPv6 and canonical dotted-quad IPv4).
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass

    # inet_aton-style IPv4: 1-4 parts, tolerating a single trailing dot.
    candidate = host[:-1] if host.endswith(".") else host
    parts = candidate.split(".")
    if not (1 <= len(parts) <= 4):
        return None
    parsed = [_parse_inet_aton_part(p) for p in parts]
    if any(v is None or v < 0 for v in parsed):
        return None
    values = [v for v in parsed if v is not None]  # None/negative ruled out above

    # inet_aton packing: the last part absorbs all remaining low-order bytes;
    # earlier parts are one byte each.
    packed = 0
    for i, v in enumerate(values):
        if i == len(values) - 1:
            max_val = 1 << (8 * (4 - i))
            if v >= max_val:
                return None
            packed |= v
        elif v > 0xFF:
            return None
        else:
            packed |= v << (8 * (3 - i))
    return ipaddress.IPv4Address(packed)


def _unwrap_embedded_ipv4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return the embedded IPv4 for mapped/NAT64/compat forms, else ``ip``.

    is_link_local/is_private return False for these IPv6 wrappers, so the
    embedded IPv4 must be category-checked instead of the wrapper.
    """
    if not isinstance(ip, ipaddress.IPv6Address):
        return ip

    # Scope identifiers (for example ``%eth0`` or the URL-encoded ``%25eth0``)
    # are routing hints, not part of the address identity.  ``str(ip)`` retains
    # them, which would make an exact credential-endpoint comparison miss
    # ``fd00:ec2::254%eth0`` and let a private/CIDR relaxation reopen IMDS.
    # Reconstructing from the integer strips the scope before every category,
    # embedded-IPv4, and hard-denial check below.
    if ip.scope_id is not None:
        ip = ipaddress.IPv6Address(int(ip))

    if ip.ipv4_mapped is not None:  # ::ffff:0:0/96
        return ip.ipv4_mapped
    if ip in _NAT64_PREFIX:  # 64:ff9b::/96 — embedded v4 in the low 32 bits
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in _6TO4_PREFIX:  # 2002:V4:V4::/48 — embedded v4 in bits [16, 48)
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    if ip in _TEREDO_PREFIX:  # 2001:0::/32 — client v4 in the low 32 bits, XOR'd
        return ipaddress.IPv4Address((int(ip) & 0xFFFFFFFF) ^ 0xFFFFFFFF)
    return ip


def _ip_denial_reason(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    allow_private: bool,
    allowed_cidrs: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (),
) -> str | None:
    """Return a denial reason if ``ip`` is an unsafe egress target.

    Embedded IPv4 forms are unwrapped first. Cloud/workload credential
    endpoints, link-local, unspecified, multicast, and reserved destinations
    are hard-denied before any relaxation. ``allow_private`` and explicit CIDRs
    may relax only loopback, private-unicast, and CGNAT destinations.
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

    explicitly_allowed = any(ip in net for net in allowed_cidrs)

    # IPv6 loopback is also classified as reserved, so handle it first.
    if ip.is_loopback:
        return None if allow_private or explicitly_allowed else "loopback"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private or (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NET):
        return None if allow_private or explicitly_allowed else "private"
    return None


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

# Server path names that shadow a built-in monitoring / health route. The server
# management routes embed the registration path in the request URL via
# {service_path:path}/{path:path} (e.g. POST /api/toggle/<path>,
# POST /api/servers/<path>/rescan). A server registered under "health" therefore
# produces management-plane request URLs like /api/toggle/health, which the audit
# middleware would misclassify as a health check and drop from the audit trail.
# Reserving these names blocks the shadow at the SOURCE (registration), so no such
# collision can exist -- defense in depth alongside the exact-match fix in the
# audit middleware (_is_health_check_path). Unlike _RESERVED_SERVER_PATH_NAMES,
# these are not wildcard sentinels and are kept in a separate set (they must NOT
# be added to access_resolver._WILDCARD_VALUES). Compared case-insensitively.
_RESERVED_MONITORING_PATH_NAMES: frozenset[str] = frozenset(
    {
        "health",
        "healthcheck",
        "metrics",
        "well-known",
        ".well-known",
    }
)

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
    allow_private: bool = False

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


# ---------------------------------------------------------------------------
# Forward-proxy egress configuration.
# ---------------------------------------------------------------------------
# Read from os.environ directly, NOT through _get_settings(): cli/agentcore
# imports this module with zero registry configuration (see the `settings = None`
# comment above), so the guard must never force Settings to be built. Matching
# Settings fields exist for documentation, validation, and the System Config
# page; tests/unit/utils/test_url_guard_forward_proxy.py asserts the two readers
# agree, because a config page showing `true` while the guard behaves as `false`
# is worse than the feature being off.
_FORWARD_PROXY_ENABLED_ENV: str = "EGRESS_FORWARD_PROXY_ENABLED"
_FORWARD_PROXY_CA_BUNDLE_ENV: str = "EGRESS_FORWARD_PROXY_CA_BUNDLE"

# Spellings accepted as true. Must stay a superset-compatible subset of what
# Pydantic's bool coercion accepts for the matching Settings field, which is why
# the single-letter and whitespace forms are here.
_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "t", "yes", "y", "on"})


@dataclass(frozen=True)
class _ForwardProxyConfig:
    """Resolved forward-proxy configuration, parsed once from the environment."""

    enabled: bool = False
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RoutedRequest:
    """A validated request plus the proxy it must be sent through.

    ``proxy_url is None`` means the direct, IP-pinned path.
    """

    request: httpx.Request
    proxy_url: str | None


def _env_first(
    *names: str,
) -> str | None:
    """Return the first non-empty environment value among ``names``, or None.

    Callers pass the upper-case spelling first, so upper case wins over lower
    case. That matches curl and httpx. Deliberate: do not "fix" it to prefer the
    lower-case form, which some other tooling does for ``no_proxy``.
    """
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_no_proxy(
    raw: str,
) -> tuple[str, ...]:
    """Parse NO_PROXY into normalized match patterns.

    A leading dot is stripped so ``.example.com`` and ``example.com`` behave
    identically, matching curl. A CIDR-shaped entry is kept (it can still match
    an exact IP literal) but warned about, because neither curl nor httpx expands
    a CIDR, so ``10.0.0.0/8`` silently fails to match ``10.1.2.3`` and sends
    in-cluster traffic to the corporate proxy.
    """
    patterns: list[str] = []
    for entry in (raw or "").split(","):
        cleaned = entry.strip().lower().lstrip(".")
        if not cleaned:
            continue
        if "/" in cleaned:
            logger.warning(
                "SSRF guard: NO_PROXY entry %r looks like a CIDR, which is not "
                "expanded (curl and httpx do not either). Use an exact host or a "
                "domain suffix, or those addresses will be sent to the proxy.",
                cleaned,
            )
        patterns.append(cleaned)
    return tuple(patterns)


@lru_cache(maxsize=1)
def _forward_proxy_config() -> _ForwardProxyConfig:
    """Parse forward-proxy configuration once per process.

    Cached like the allowlist factories, because the environment is immutable per
    process. Tests call ``_forward_proxy_config.cache_clear()`` after patching
    ``os.environ``.

    A flag that is on with no proxy configured logs one WARNING and returns a
    DISABLED config, so the deployment behaves exactly as if the flag were off.
    That is deliberately not an error: an operator may enable the flag in a shared
    values file while only some environments set a proxy.
    """
    raw_enabled = (os.environ.get(_FORWARD_PROXY_ENABLED_ENV) or "").strip().lower()
    if raw_enabled not in _TRUE_VALUES:
        return _ForwardProxyConfig()

    http_proxy = _env_first("HTTP_PROXY", "http_proxy")
    https_proxy = _env_first("HTTPS_PROXY", "https_proxy")

    if http_proxy is None and https_proxy is None:
        logger.warning(
            "SSRF guard: %s is true but neither HTTP_PROXY nor HTTPS_PROXY is set; "
            "egress stays direct and pinned",
            _FORWARD_PROXY_ENABLED_ENV,
        )
        return _ForwardProxyConfig()

    no_proxy = _parse_no_proxy(_env_first("NO_PROXY", "no_proxy") or "")
    logger.info(
        "SSRF guard: forward proxy enabled (http=%s https=%s no_proxy_entries=%d). "
        "IP-rebind pinning is relaxed for targets that resolve exclusively public.",
        redact_url(http_proxy) if http_proxy else "unset",
        redact_url(https_proxy) if https_proxy else "unset",
        len(no_proxy),
    )
    return _ForwardProxyConfig(
        enabled=True,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        no_proxy=no_proxy,
    )


def _validate_proxy_url(
    proxy_url: str,
) -> None:
    """Raise unless a configured proxy URL is a usable http(s) endpoint.

    Fails closed on a malformed proxy rather than falling back to a direct dial,
    which would hide the misconfiguration behind the same ConnectTimeout the
    operator is trying to fix. The proxy URL may carry basic-auth userinfo, so it
    is redacted in the error message as well as in logs.
    """
    try:
        parsed = urlsplit(proxy_url)
        parsed.port  # noqa: B018 - forces validation of a malformed/out-of-range port
    except (TypeError, ValueError) as exc:
        raise UrlValidationError(
            redact_url(proxy_url), f"proxy URL could not be parsed: {exc}"
        ) from exc
    if parsed.scheme.lower() not in ("http", "https"):
        raise UrlValidationError(
            redact_url(proxy_url),
            f"proxy URL scheme '{parsed.scheme}' is not supported (use http or https)",
        )
    if not parsed.hostname:
        raise UrlValidationError(redact_url(proxy_url), "proxy URL has no hostname")


@lru_cache(maxsize=1)
def _forward_proxy_ssl_context() -> ssl.SSLContext | bool:
    """Return the TLS context for proxied egress, or True for the default store.

    Cached with ``maxsize=1`` for two reasons: the file is read once per process,
    and ``shared_guarded_async_client`` keys its pool on ``(profile.name, verify)``
    where an ``ssl.SSLContext`` hashes by identity, so a fresh context per call
    would mint a new pool key every time and multiply the connection pool.

    The bundle is loaded ON TOP of the default roots. This is why the setting is
    not ``SSL_CERT_FILE``: ``ssl.create_default_context(cafile=...)`` REPLACES the
    certifi store rather than adding to it, so an operator who points
    ``SSL_CERT_FILE`` at the single CA their proxy team handed them loses TLS to
    every direct-path target and gets a second, unrelated outage.

    "Default roots" means certifi plus any OS-provided store, in that order,
    because certifi is what httpx itself would have used
    (``httpx._config.create_ssl_context`` builds from ``certifi.where()``) and
    httpx does NOT read the operating system trust store. Seeding from certifi
    explicitly keeps the proxied path's baseline trust identical to the direct
    path's instead of depending on whether the base image ships
    ``ca-certificates``.

    Fails closed: a missing, unreadable, or malformed bundle raises rather than
    falling back, because a silent fallback surfaces as a certificate error on
    every proxied request, which reads like a bug in this feature.
    """
    path = (os.environ.get(_FORWARD_PROXY_CA_BUNDLE_ENV) or "").strip()
    if not path:
        return True  # httpx builds its own default context; unchanged behavior

    if not os.path.isfile(path):
        raise UrlValidationError(
            path,
            f"{_FORWARD_PROXY_CA_BUNDLE_ENV} points to '{path}', which is not a file",
        )
    context = ssl.create_default_context(cafile=certifi.where())
    # Additive: picks up an OS-provided store as well, where one exists.
    context.load_default_certs()
    try:
        context.load_verify_locations(cafile=path)
    except (OSError, ssl.SSLError) as exc:
        raise UrlValidationError(
            path, f"failed to load {_FORWARD_PROXY_CA_BUNDLE_ENV} '{path}': {exc}"
        ) from exc
    logger.info("SSRF guard: loaded forward-proxy CA bundle from %s", path)
    return context


def _no_proxy_matches(
    patterns: tuple[str, ...],
    hostname: str,
    port: int,
) -> bool:
    """Return True if the target is excluded from proxying by NO_PROXY.

    Supports the conventional forms: ``*`` (everything direct), an exact host, a
    ``host:port`` pair, and a domain suffix (``example.com``, which matches
    ``a.example.com``; a leading dot was already stripped by ``_parse_no_proxy``).
    CIDR entries are NOT expanded, matching curl and httpx; ``_parse_no_proxy``
    warns about one.

    Module-level rather than a method because both the guarded transports and the
    plain proxy-routed transport need it, and one matcher is the point.
    """
    if not patterns:
        return False
    host = hostname.lower()
    host_port = f"{host}:{port}"
    for pattern in patterns:
        if pattern == "*":
            return True
        if pattern in (host, host_port):
            return True
        if host.endswith(f".{pattern}"):
            return True
    return False


async def _resolves_exclusively_public_async(
    hostname: str,
    port: int,
) -> bool | None:
    """Return True if every address a host resolves to is on the public internet.

    This is a ROUTING question, not an access-control one, and it never raises.
    Used by the plain (un-guarded) transport, whose whole purpose is reaching
    operator-configured targets the SSRF guard would reject, such as an in-cluster
    ``http://keycloak:8080`` at a private address.

    Returns:
        True when every resolved address is public, so the target may be proxied.
        False when any address is internal, so the target must stay direct.
        None when resolution failed, in which case the caller dials direct and
        lets httpx surface its own error rather than inventing a new failure mode.
    """
    literal = coerce_ip_literal(hostname)
    if literal is not None:
        return not _is_blocked_ip(hostname, _Allowlist(), allow_private=False)
    try:
        loop = asyncio.get_running_loop()
        addr_info = await asyncio.wait_for(
            loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP),
            timeout=_DNS_RESOLUTION_TIMEOUT_SECONDS,
        )
    except (TimeoutError, socket.gaierror, OSError):
        logger.debug(
            "plain egress: could not resolve %s for a proxy routing decision; dialing direct",
            hostname,
        )
        return None
    addresses = {str(info[4][0]) for info in addr_info}
    if not addresses:
        return None
    return all(not _is_blocked_ip(ip, _Allowlist(), allow_private=False) for ip in addresses)


def _build_proxy(
    proxy_url: str,
) -> httpx.Proxy:
    """Build the ``httpx.Proxy`` a delegate transport connects through.

    A ``Proxy`` object rather than the raw URL string, for two reasons. httpx
    hands ``proxy.ssl_context`` to httpcore as ``proxy_ssl_context``, and a
    ``Proxy`` built from a string has ``ssl_context=None``, so an ``https://``
    proxy signed by an internal CA would be verified against httpcore's default
    context and fail. ``httpx.Proxy`` also extracts userinfo from the URL into
    ``raw_auth`` itself, which is what keeps proxy basic auth working.
    """
    context = _forward_proxy_ssl_context()
    return httpx.Proxy(
        url=proxy_url,
        ssl_context=context if isinstance(context, ssl.SSLContext) else None,
    )


def _delegate_kwargs_with_ca_bundle(
    base: dict[str, object],
) -> dict[str, object]:
    """Return delegate transport kwargs, applying the CA bundle to ``verify``.

    The bundle has to reach the UPSTREAM leg (inside the tunnel) as well as the
    leg to the proxy, and that leg is governed by ``verify``. Only the default
    trust store is replaced: a caller that passed ``verify=False`` or its own
    bundle path made an explicit choice that is left alone.
    """
    context = _forward_proxy_ssl_context()
    kwargs = dict(base)
    if isinstance(context, ssl.SSLContext) and kwargs.get("verify", True) is True:
        kwargs["verify"] = context
    return kwargs


# Cloud credential endpoints that MUST be excluded from the forward proxy.
#
# These are link-local, so a proxy dialing one reaches ITS OWN host's metadata
# service, not the caller's. Setting HTTP_PROXY re-points every proxy-aware SDK
# in the process, and botocore is one: its credential providers will send these
# requests to the proxy. The endpoints speak plain HTTP, so there is no CONNECT
# tunnel and the proxy sees the full response body, which is the credential
# document (AccessKeyId, SecretAccessKey, Token) in cleartext.
#
# AWS_EC2_METADATA_DISABLED=true is NOT a sufficient substitute. It is read only
# by botocore's IMDSFetcher (botocore/utils.py), so it covers 169.254.169.254
# alone. ContainerProvider (botocore/credentials.py) is gated purely on
# AWS_CONTAINER_CREDENTIALS_RELATIVE_URI / _FULL_URI being present, with no such
# check, so the ECS and EKS task-credential endpoints stay exposed. NO_PROXY is
# the only control that covers all three.
#
# The IPv6 (fd00:ec2::*) and Alibaba (100.100.100.200) endpoints in
# _CREDENTIAL_ENDPOINT_IPS are deliberately not required here: no SDK dials them
# through HTTP_PROXY in practice, and NO_PROXY matching of a bracketed IPv6
# literal is inconsistent across clients. This module's own transports hard-deny
# every one of them regardless, so the requirement below exists purely to protect
# the OTHER libraries sharing the process.
_PROXY_REQUIRED_NO_PROXY_IPS: tuple[str, ...] = (
    "169.254.169.254",  # EC2 IMDS
    "169.254.170.2",  # ECS task credentials
    "169.254.170.23",  # EKS Pod Identity
)


def _assert_credential_endpoints_bypass_proxy(
    config: _ForwardProxyConfig,
) -> None:
    """Refuse to start when a credential endpoint could be sent to the proxy.

    Fails closed rather than warning. The blocked combination is never a correct
    configuration: no deployment legitimately wants its IAM credential requests
    routed through a forward proxy. A warning would also be read too late, since
    the leak happens on the first boto3 call, which can be long after startup.

    This can only fire for a deployment that explicitly enabled
    EGRESS_FORWARD_PROXY_ENABLED, so it cannot affect an existing one, and the
    remedy is a single NO_PROXY entry.
    """
    if not config.enabled:
        return
    missing = [
        ip for ip in _PROXY_REQUIRED_NO_PROXY_IPS if not _no_proxy_matches(config.no_proxy, ip, 80)
    ]
    if not missing:
        return
    raise UrlValidationError(
        ",".join(missing),
        f"{_FORWARD_PROXY_ENABLED_ENV} is true but NO_PROXY does not exclude the "
        f"cloud credential endpoint(s) {', '.join(missing)}. Setting HTTP_PROXY "
        "re-points every proxy-aware SDK in this process, so the AWS SDK would "
        "send IAM credential requests to the forward proxy. Those endpoints speak "
        "plain HTTP, so the proxy would see the returned AccessKeyId, "
        "SecretAccessKey and Token in cleartext, and because the addresses are "
        "link-local the proxy would be answering from its own host rather than "
        "this one. Refusing to start. Fix: add "
        f"'{','.join(_PROXY_REQUIRED_NO_PROXY_IPS)}' to NO_PROXY on this service "
        "(AWS_EC2_METADATA_DISABLED=true is NOT sufficient: botocore applies it "
        "only to 169.254.169.254, not to the container-credential endpoints). "
        "See docs/forward-proxy-egress.md.",
    )


def validate_forward_proxy_config() -> None:
    """Resolve and validate forward-proxy configuration at process startup.

    Call this from each app's lifespan so a bad CA-bundle path fails the pod
    rather than surfacing on the first proxied request, hours later and far from
    the cause. The CA bundle is validated even when the feature flag is off, so a
    typo is caught before the flag is flipped.

    Raises:
        UrlValidationError: If the CA bundle path is missing, unreadable, or not
            valid PEM, or if the proxy is enabled without NO_PROXY excluding the
            cloud credential endpoints.
    """
    config = _forward_proxy_config()
    _assert_credential_endpoints_bypass_proxy(config)
    _forward_proxy_ssl_context()


@lru_cache(maxsize=1)
def _egress_route_counter() -> object | None:
    """Return the egress-route counter, or None when metrics are unavailable.

    Imported lazily and best-effort: this module must stay importable from the
    auth-server process and from cli/agentcore, neither of which owns the
    registry meter. A missing meter makes the counter a no-op rather than an
    error, because a metric must never be the reason egress fails.
    """
    try:
        from ..observability.meters import egress_forward_proxy_requests_total
    except Exception:  # noqa: BLE001 - metrics are strictly optional here
        logger.debug("SSRF guard: egress route counter unavailable", exc_info=True)
        return None
    return egress_forward_proxy_requests_total


def _annotate_current_span(
    profile_name: str,
    route: str,
) -> None:
    """Annotate the enclosing httpx client span with the routing decision.

    No new span is created: opentelemetry-instrumentation-httpx already produces
    one per request, and annotating it is what lets an operator see in a trace
    whether a slow request went through the proxy. Imported lazily and
    best-effort, for the same reason as the counter.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return
        span.set_attribute("egress.via_forward_proxy", route == "proxied")
        span.set_attribute("egress.guard_profile", profile_name)
    except Exception:  # noqa: BLE001 - tracing must never break egress
        logger.debug("SSRF guard: failed to annotate the egress span", exc_info=True)


def _record_egress_route(
    profile_name: str,
    route: str,
    outcome: str,
) -> None:
    """Record one egress routing decision as a metric and a span attribute.

    Never raises: observability must never be the reason an egress request fails.
    """
    _annotate_current_span(profile_name, route)
    counter = _egress_route_counter()
    if counter is None:
        return
    try:
        counter.add(1, {"profile": profile_name, "route": route, "outcome": outcome})  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a metric must never break egress
        logger.debug("SSRF guard: failed to record egress route metric", exc_info=True)


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
    return _Allowlist(hosts=_parse_hosts(_get_settings().github_extra_hosts))


@lru_cache(maxsize=1)
def _proxy_allowlist() -> _Allowlist:
    """Return the ordinary server/agent/generic target allowlist.

    The bundled ``mcpgw-server`` hostname is intentionally absent. It is reserved
    and only admitted through ``BUILTIN_AIREGISTRY_TOOLS_PROFILE`` after exact
    entity/path/full-target matching.
    """
    _settings = _get_settings()
    return _Allowlist(
        hosts=_parse_hosts(_settings.ssrf_allowed_hosts),
        cidrs=_parse_cidrs(_settings.ssrf_allowed_cidrs),
        allow_private=bool(getattr(_settings, "gateway_proxy_allow_private_targets", False)),
    )


@lru_cache(maxsize=1)
def _builtin_airegistry_tools_allowlist() -> _Allowlist:
    """Return proxy policy plus the one exact built-in private hostname."""
    ordinary = _proxy_allowlist()
    return _Allowlist(
        hosts=ordinary.hosts | _RESERVED_BUILTIN_PROXY_HOSTS,
        cidrs=ordinary.cidrs,
        allow_private=ordinary.allow_private,
    )


@lru_cache(maxsize=1)
def _credentialed_oauth_allowlist() -> _Allowlist:
    """Return the OAuth token-endpoint allowlist: trusted IdP hosts only.

    Credential-bearing token POSTs may include a client secret, refresh token,
    or user assertion. They must not inherit any operator proxy/skill bypass, so
    this reads its own dedicated setting and deliberately does NOT consult
    ``ssrf_allowed_hosts``/``ssrf_allowed_cidrs``/``github_extra_hosts``: an
    entry there still cannot re-permit a token endpoint.

    ``egress_oauth_trusted_idp_hosts`` exists because a self-hosted IdP
    (Keycloak, Entra behind Private Link, ...) legitimately resolves to a
    private address, while the gateway is already required to trust that same
    IdP for its own authentication via ``KEYCLOAK_URL``. Unlike a federation
    peer or a registrant-supplied proxy target, the IdP is operator-configured
    infrastructure. It is hosts-only (no CIDRs, no wildcards) so each IdP must
    be named exactly — deliberately narrower than the proxy profile — and
    defaults to empty, so an existing deployment is unchanged.

    This is a post-resolution relaxation, not a DNS bypass: the profile still
    enforces HTTPS at both structural validation and transport, answers are
    still resolved, classified and pinned, and cloud/workload credential,
    metadata, link-local, reserved and multicast addresses stay hard-denied.
    Cached because settings are immutable per-process.
    """
    return _Allowlist(hosts=_parse_hosts(_get_settings().egress_oauth_trusted_idp_hosts))


def _egress_upstream_allowlist() -> _Allowlist:
    """Empty allowlist for the egress-injected proxy hop.

    Deliberately touches no settings so it is importable from the auth-server
    process (which does not build the registry Settings). Combined with
    ``allow_private=True`` on EGRESS_UPSTREAM_PROFILE, the hop pins to the
    resolved IP and permits private / hostname-private MCP upstreams, while the
    cloud/workload credential + metadata endpoints stay hard-denied.
    """
    return _Allowlist()


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
    require_https: bool = False
    allow_private: bool = False
    # Profiles whose requests carry a secret (client secret, refresh token,
    # bearer, injected egress credential). On the forward-proxied path these are
    # HTTPS-only, so the secret rides inside a CONNECT tunnel and the proxy sees
    # only host:port. An http target would instead be sent via httpcore's
    # absolute-URI forward path, handing the full URL and the Authorization
    # header to the proxy in cleartext. A field rather than a check against
    # profile names, so a future profile has to make an explicit choice.
    credential_bearing: bool = False


SKILL_PROFILE = _Profile(name="skill", allowlist_factory=_skill_allowlist)
PROXY_PROFILE = _Profile(name="proxy", allowlist_factory=_proxy_allowlist)
BUILTIN_AIREGISTRY_TOOLS_PROFILE = _Profile(
    name="builtin-airegistry-tools",
    allowlist_factory=_builtin_airegistry_tools_allowlist,
    allowed_url_identities=_BUILTIN_AIREGISTRY_TOOLS_OUTBOUND_IDENTITIES,
)
FEDERATION_PROFILE = _Profile(
    name="federation",
    allowlist_factory=_federation_allowlist,
    credential_bearing=True,  # attaches a peer bearer token
)
CREDENTIALED_OAUTH_PROFILE = _Profile(
    name="credentialed-oauth",
    allowlist_factory=_credentialed_oauth_allowlist,
    require_https=True,
    credential_bearing=True,  # belt and braces: require_https already denies http
)
EGRESS_UPSTREAM_PROFILE = _Profile(
    name="egress-upstream",
    allowlist_factory=_egress_upstream_allowlist,
    allow_private=True,
    credential_bearing=True,  # carries the injected egress credential
)


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
    allow_private: bool = False,
) -> bool:
    """Return True if an IP must not be the target of a server-side fetch.

    ``trusted_hostname`` represents an explicit operator hostname allowlist or
    the exact built-in profile. It preserves the old ability to reach internal
    addresses, but only *after* DNS resolution and canonical classification. The
    resolved address is added as an exact CIDR, so ``_ip_denial_reason`` still
    executes its cloud/workload credential hard-deny before the relaxation.
    """
    ip = coerce_ip_literal(ip_str)
    if ip is None:
        return True
    allowed_cidrs = allowlist.cidrs
    if trusted_hostname:
        allowed_cidrs = (*allowed_cidrs, ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}"))
    return (
        _ip_denial_reason(
            ip,
            # allow_private is honored from EITHER source: the threaded flag
            # (profile.allow_private, e.g. EGRESS_UPSTREAM_PROFILE=True) or the
            # allowlist (allowlist.allow_private, set from
            # gateway_proxy_allow_private_targets). Metadata/credential/link-local
            # /reserved/multicast stay hard-denied in _ip_denial_reason before
            # any relaxation, so this only ever relaxes loopback/private/CGNAT.
            allow_private=allow_private or allowlist.allow_private,
            allowed_cidrs=allowed_cidrs,
        )
        is not None
    )


def _validate_resolved_ips(
    hostname: str,
    addr_info: list[tuple],
    allowlist: _Allowlist,
    *,
    allow_private: bool = False,
) -> list[str]:
    """Validate every resolver answer and return a de-duplicated IP list."""
    trusted_hostname = allowlist.allows_host(hostname.lower())
    ips: list[str] = []
    for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
        ip_str = str(sockaddr[0])
        if _is_blocked_ip(
            ip_str, allowlist, trusted_hostname=trusted_hostname, allow_private=allow_private
        ):
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
    *,
    allow_private: bool = False,
) -> list[str]:
    """Synchronously resolve, classify, and return every destination IP."""
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlValidationError(hostname, f"DNS resolution failed: {exc}") from exc
    return _validate_resolved_ips(hostname, addr_info, allowlist, allow_private=allow_private)


async def _resolve_public_ips_async(
    hostname: str,
    port: int,
    allowlist: _Allowlist,
    *,
    allow_private: bool = False,
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
    return _validate_resolved_ips(hostname, addr_info, allowlist, allow_private=allow_private)


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

    allowed_schemes = ("https",) if require_https or profile.require_https else ("http", "https")
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
        if _is_blocked_ip(hostname, allowlist, allow_private=profile.allow_private):
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
        logger.debug(f"URL guard[{profile.name}]: resolving trusted host '{hostname_lower}'")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _resolve_public_ips(hostname, port, allowlist, allow_private=profile.allow_private)


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

    A path that normalizes to empty (e.g. ``/``, ``//``) is also rejected: the
    nginx location for a server is rendered with a trailing slash (issue #1501),
    so a slashes-only path becomes ``location /`` -- a root prefix match that
    subjects every URL on the gateway to the ``auth_request /validate`` subrequest
    and shadows/duplicates the static catch-all. No real server registers at the
    root, so this is a footgun with no legitimate use; fail closed.

    A path that normalizes to a built-in monitoring-route name (``health`` and
    friends, see :data:`_RESERVED_MONITORING_PATH_NAMES`) is also rejected: the
    management routes embed the path in the request URL, so such a name would let
    a mutating admin action masquerade as a health check and be dropped from the
    audit trail.

    Args:
        path: The server path (e.g. ``/github``).

    Raises:
        UrlValidationError: If the path is empty, contains disallowed nginx
            metacharacters, normalizes to an empty (slashes-only) path, or
            normalizes to a reserved cross-server wildcard or monitoring-route
            name.
    """
    if not path or not isinstance(path, str):
        raise UrlValidationError(str(path), "server path is empty or not a string")
    if contains_nginx_metacharacters(path):
        raise UrlValidationError(path, "server path contains disallowed nginx metacharacters")

    # Normalize the same way the scope layer does (add_server_scope does
    # server_path.lstrip("/")), then also drop trailing slashes so "/all/" and
    # "//all//" collapse to the same reserved name. Compare case-insensitively.
    normalized = path.strip("/")

    # A slashes-only path normalizes to empty. It renders as `location /` after
    # the trailing-slash normalization (issue #1501), hijacking the whole gateway
    # into /validate. Reject it: fail closed.
    if not normalized:
        raise UrlValidationError(
            path,
            "server path must have a non-empty segment (a root/slashes-only path "
            "would render as a gateway-wide 'location /' block)",
        )

    if normalized.lower() in _RESERVED_SERVER_PATH_NAMES:
        raise UrlValidationError(
            path,
            f"server path '{normalized}' is reserved (collides with the cross-server wildcard)",
        )

    # Reject names that shadow a built-in monitoring/health route. Such a name
    # would let a management-plane mutation (e.g. /api/toggle/health) masquerade
    # as a health check and be dropped from the audit trail. Fail closed at the
    # source so the collision cannot be created in the first place.
    if normalized.lower() in _RESERVED_MONITORING_PATH_NAMES:
        raise UrlValidationError(
            path,
            f"server path '{normalized}' is reserved (shadows a built-in monitoring route)",
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
            if _is_blocked_ip(hostname, allowlist, allow_private=self._guard_profile.allow_private):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return request
        pinned_ip = _resolve_public_ips(
            hostname, port, allowlist, allow_private=self._guard_profile.allow_private
        )[0]
        return self._rewrite_to_pinned_ip(request, url, hostname, pinned_ip)

    async def _pin_request_async(
        self,
        request: httpx.Request,
    ) -> httpx.Request:
        """Asynchronously validate, resolve under deadline, and pin a request."""
        url, _scheme, hostname, port, allowlist = self._request_target(request)
        if coerce_ip_literal(hostname) is not None:
            if _is_blocked_ip(hostname, allowlist, allow_private=self._guard_profile.allow_private):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return request
        pinned_ip = (
            await _resolve_public_ips_async(
                hostname, port, allowlist, allow_private=self._guard_profile.allow_private
            )
        )[0]
        return self._rewrite_to_pinned_ip(request, url, hostname, pinned_ip)

    # -- Forward-proxy routing (see the module docstring) --------------------

    def _forward_proxy_candidate(
        self,
        scheme: str,
        hostname: str,
        port: int,
    ) -> str | None:
        """Return the proxy this target COULD use, or None for certainly-direct.

        A candidate, not a decision: the final call needs the resolved addresses
        (see ``_route_request_async``), because a target that resolves to an
        internal address goes direct even when a proxy is configured. This covers
        only the reasons a target can never be proxied, none of which needs DNS:
        the feature is off, ``NO_PROXY`` names the host, or no proxy is configured
        for this scheme.
        """
        config = _forward_proxy_config()
        if not config.enabled:
            return None
        if _no_proxy_matches(config.no_proxy, hostname, port):
            return None
        # No warning when the other scheme's proxy is set: a deployment with only
        # HTTPS_PROXY is a normal configuration, and an http:// target there is
        # correctly dialed directly.
        proxy_url = config.https_proxy if scheme == "https" else config.http_proxy
        if proxy_url is None:
            return None
        _validate_proxy_url(proxy_url)
        return proxy_url

    def _assert_proxy_allowed_for_profile(
        self,
        url: httpx.URL,
        scheme: str,
    ) -> None:
        """Reject an http target through a proxy for credential-bearing profiles.

        An https target is tunneled with CONNECT, so the proxy sees only
        ``host:port`` and the secret stays inside TLS. An http target is sent via
        the absolute-URI forward path, which hands the full URL and the
        Authorization header to the proxy in cleartext. Fail closed.
        """
        if not self._guard_profile.credential_bearing or scheme == "https":
            return
        logger.warning(
            "SSRF guard[%s]: refusing http target %s through the forward proxy "
            "(the credential would reach the proxy in cleartext)",
            self._guard_profile.name,
            redact_url(str(url)),
        )
        raise UrlValidationError(
            str(url),
            "credential-bearing profile refuses an http target through a forward "
            "proxy (the credential would reach the proxy in cleartext). Use https, "
            "or add this host to NO_PROXY so it is dialed directly and pinned",
        )

    @staticmethod
    def _is_internal_address(
        ip_str: str,
    ) -> bool:
        """Return True if an address is internal, ignoring every relaxation.

        Delegates to the canonical classifier with an EMPTY allowlist and
        ``allow_private=False``, so this introduces no second opinion about what
        counts as internal. An address the classifier rejects with no relaxations
        applied is internal by definition; one it accepts is on the public
        internet.

        This is the ROUTING signal, not an access decision. Access was already
        decided by ``_validated_addresses*``, which applied the profile's
        allowlist and raised on anything genuinely forbidden.
        """
        return _is_blocked_ip(ip_str, _Allowlist(), allow_private=False)

    def _validated_addresses(
        self,
        url: httpx.URL,
        hostname: str,
        port: int,
        allowlist: _Allowlist,
    ) -> list[str]:
        """Resolve and classify ONCE, returning every permitted address.

        Resolving a second time to make the routing decision would both waste a
        lookup and open a window for the two answers to disagree, which is the
        exact TOCTOU gap the pin exists to close. One lookup, reused by both
        branches.
        """
        if coerce_ip_literal(hostname) is not None:
            if _is_blocked_ip(hostname, allowlist, allow_private=self._guard_profile.allow_private):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return [hostname]
        # Raises if ANY answer is blocked under the profile's allowlist.
        return _resolve_public_ips(
            hostname, port, allowlist, allow_private=self._guard_profile.allow_private
        )

    async def _validated_addresses_async(
        self,
        url: httpx.URL,
        hostname: str,
        port: int,
        allowlist: _Allowlist,
    ) -> list[str]:
        """Async twin of :meth:`_validated_addresses`."""
        if coerce_ip_literal(hostname) is not None:
            if _is_blocked_ip(hostname, allowlist, allow_private=self._guard_profile.allow_private):
                raise UrlValidationError(str(url), f"targets blocked/private IP {hostname}")
            return [hostname]
        return await _resolve_public_ips_async(
            hostname, port, allowlist, allow_private=self._guard_profile.allow_private
        )

    def _route_on_addresses(
        self,
        request: httpx.Request,
        url: httpx.URL,
        scheme: str,
        hostname: str,
        addresses: list[str],
        candidate: str,
    ) -> _RoutedRequest:
        """Choose proxied or direct given the already-resolved addresses.

        Shared by the sync and async routers so the two cannot drift. A proxy
        carries a target only when EVERY resolved address is on the public
        internet, so rebinding protection is surrendered for public destinations
        alone.
        """
        if any(self._is_internal_address(ip) for ip in addresses):
            # Internal target, permitted by an operator allowlist. Keep the pin. A
            # mixed answer set counts as internal: deliberately conservative,
            # because the safe direction preserves a control.
            logger.debug(
                "SSRF guard[%s]: %s resolves internal, staying direct and pinned",
                self._guard_profile.name,
                hostname,
            )
            _record_egress_route(self._guard_profile.name, "direct", "ok")
            if coerce_ip_literal(hostname) is not None:
                return _RoutedRequest(request, None)  # already an address, nothing to pin
            return _RoutedRequest(
                self._rewrite_to_pinned_ip(request, url, hostname, addresses[0]), None
            )

        # Public destination. The resolved addresses are logged at DEBUG and then
        # deliberately discarded: they are the only forensic handle on the
        # rebinding gap, since the proxy resolves the name again when it dials.
        self._assert_proxy_allowed_for_profile(url, scheme)
        logger.debug(
            "SSRF guard[%s]: %s resolves public %s, routing via forward proxy",
            self._guard_profile.name,
            hostname,
            addresses,
        )
        _record_egress_route(self._guard_profile.name, "proxied", "ok")
        # Handed over UNMODIFIED so the tunnel can name the host and verify its
        # certificate against that name.
        return _RoutedRequest(request, candidate)

    def _excluded_from_proxy(
        self,
        hostname: str,
    ) -> None:
        """Record a target the proxy is configured for but will not carry.

        Only reached when the feature is ON, so the flag-off path stays free of
        any observability cost. Recording this is what lets an operator see a
        proxy-only deployment still sending traffic direct, which usually means
        NO_PROXY is too broad or no proxy is set for that scheme.
        """
        logger.debug(
            "SSRF guard[%s]: %s excluded from the forward proxy (NO_PROXY or no "
            "proxy for this scheme), staying direct and pinned",
            self._guard_profile.name,
            hostname,
        )
        _record_egress_route(self._guard_profile.name, "direct", "ok")

    def _route_request(
        self,
        request: httpx.Request,
    ) -> _RoutedRequest:
        """Validate, resolve, then route on where the target actually lives."""
        url, scheme, hostname, port, allowlist = self._request_target(request)
        # NOTE: _request_target runs twice on every direct branch below, once
        # here and once inside _pin_request. It is pure CPU validation with no
        # DNS, and keeping _pin_request self-contained is what preserves the
        # existing transport tests that call it directly. Do not "optimize" away.
        if not _forward_proxy_config().enabled:
            # Feature off: the existing path, byte for byte, and no metric.
            return _RoutedRequest(self._pin_request(request), None)

        candidate = self._forward_proxy_candidate(scheme, hostname, port)
        if candidate is None:
            self._excluded_from_proxy(hostname)
            return _RoutedRequest(self._pin_request(request), None)

        addresses = self._validated_addresses(url, hostname, port, allowlist)
        return self._route_on_addresses(request, url, scheme, hostname, addresses, candidate)

    async def _route_request_async(
        self,
        request: httpx.Request,
    ) -> _RoutedRequest:
        """Async twin of :meth:`_route_request`."""
        url, scheme, hostname, port, allowlist = self._request_target(request)
        # See the note in _route_request about the deliberate double
        # _request_target call on the direct branches.
        if not _forward_proxy_config().enabled:
            return _RoutedRequest(await self._pin_request_async(request), None)

        candidate = self._forward_proxy_candidate(scheme, hostname, port)
        if candidate is None:
            self._excluded_from_proxy(hostname)
            return _RoutedRequest(await self._pin_request_async(request), None)

        addresses = await self._validated_addresses_async(url, hostname, port, allowlist)
        return self._route_on_addresses(request, url, scheme, hostname, addresses, candidate)


class GuardedTransport(_PinnedResolverMixin, httpx.HTTPTransport):
    """Synchronous httpx transport that pins requests to validated IPs.

    When a forward proxy applies (see the module docstring), the validated request
    is handed to a delegate transport built with the same TLS and retry settings
    plus ``proxy=``, because httpx builds proxy support into the transport and a
    custom transport otherwise never sees ``HTTP_PROXY``. The guard runs in THIS
    class either way, so the proxied path cannot skip validation.
    """

    def __init__(
        self,
        *,
        guard_profile: _Profile = SKILL_PROFILE,
        **kwargs: object,
    ) -> None:
        self._guard_profile = guard_profile
        # Captured so a proxy delegate inherits verify / retries / limits exactly,
        # rather than silently getting different TLS settings from the direct path.
        self._delegate_kwargs = dict(kwargs)
        self._proxy_delegates: dict[str, httpx.HTTPTransport] = {}
        # The sync transport can be shared across threads, so delegate creation is
        # double-checked under a lock (the async twin needs no lock: its dict miss
        # has no await between the read and the write).
        self._delegate_lock = threading.Lock()
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def _proxy_delegate(
        self,
        proxy_url: str,
    ) -> httpx.HTTPTransport:
        """Return (creating once) the delegate transport for this proxy."""
        delegate = self._proxy_delegates.get(proxy_url)
        if delegate is not None:
            return delegate
        with self._delegate_lock:
            delegate = self._proxy_delegates.get(proxy_url)
            if delegate is None:
                delegate = httpx.HTTPTransport(
                    proxy=_build_proxy(proxy_url),
                    **_delegate_kwargs_with_ca_bundle(self._delegate_kwargs),  # type: ignore[arg-type]
                )
                self._proxy_delegates[proxy_url] = delegate
        return delegate

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        routed = self._route_request(request)
        if routed.proxy_url is None:
            return super().handle_request(routed.request)
        return self._proxy_delegate(routed.proxy_url).handle_request(routed.request)

    def close(self) -> None:
        for delegate in list(self._proxy_delegates.values()):
            try:
                delegate.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("error closing a forward-proxy delegate", exc_info=True)
        self._proxy_delegates.clear()
        super().close()


class _AsyncProxyDelegateMixin:
    """Owns the per-proxy delegate transports for an async transport.

    Shared by the guarded transport and the plain proxy-routed transport so the
    delegate construction and teardown exist once. Copy-pasting them would be how
    the two drift apart on TLS settings.
    """

    _delegate_kwargs: dict[str, object]
    _proxy_delegates: dict[str, httpx.AsyncHTTPTransport]

    def _init_proxy_delegates(
        self,
        kwargs: dict[str, object],
    ) -> None:
        """Capture the transport kwargs a delegate must inherit."""
        self._delegate_kwargs = dict(kwargs)
        self._proxy_delegates = {}

    def _proxy_delegate(
        self,
        proxy_url: str,
    ) -> httpx.AsyncHTTPTransport:
        """Return (creating once) the delegate transport for this proxy.

        A plain synchronous dict miss with no ``await`` between the read and the
        write, so two coroutines on one event loop cannot interleave and create
        two delegates.
        """
        delegate = self._proxy_delegates.get(proxy_url)
        if delegate is None:
            delegate = httpx.AsyncHTTPTransport(
                proxy=_build_proxy(proxy_url),
                **_delegate_kwargs_with_ca_bundle(self._delegate_kwargs),  # type: ignore[arg-type]
            )
            self._proxy_delegates[proxy_url] = delegate
        return delegate

    async def _aclose_proxy_delegates(self) -> None:
        """Close every delegate, best effort."""
        for delegate in list(self._proxy_delegates.values()):
            try:
                await delegate.aclose()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("error closing a forward-proxy delegate", exc_info=True)
        self._proxy_delegates.clear()


class GuardedAsyncTransport(
    _AsyncProxyDelegateMixin, _PinnedResolverMixin, httpx.AsyncHTTPTransport
):
    """Async httpx transport that pins requests to validated IPs.

    See :class:`GuardedTransport` for how the forward-proxy delegate works.
    Setting ``proxy=`` on the enclosing ``httpx.AsyncClient`` instead would mount
    a plain transport under ``all://`` that shadows this one and removes every
    SSRF check while leaving the factory named ``guarded_async_client``.
    """

    def __init__(
        self,
        *,
        guard_profile: _Profile = SKILL_PROFILE,
        **kwargs: object,
    ) -> None:
        self._guard_profile = guard_profile
        self._init_proxy_delegates(kwargs)
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        routed = await self._route_request_async(request)
        if routed.proxy_url is None:
            return await super().handle_async_request(routed.request)
        delegate = self._proxy_delegate(routed.proxy_url)
        return await delegate.handle_async_request(routed.request)

    async def aclose(self) -> None:
        await self._aclose_proxy_delegates()
        await super().aclose()


class PlainProxyRoutedAsyncTransport(_AsyncProxyDelegateMixin, httpx.AsyncHTTPTransport):
    """Un-guarded transport that routes PUBLIC targets through the forward proxy.

    Backs :func:`shared_plain_async_client` (issue #1834). That client serves
    operator-configured, already-trusted targets: the login IdP token/userinfo
    endpoints and the in-cluster registry vends. Two of those are public when the
    IdP is hosted (Entra, Cognito, Auth0, Okta) and two are in-cluster by
    construction, so the routing has to tell them apart.

    Deliberately NOT a :class:`GuardedAsyncTransport` subclass, and deliberately
    NOT pinning or denying. Its whole purpose is reaching targets the SSRF guard
    rejects, such as an in-cluster ``http://keycloak:8080`` at a private address;
    inheriting the guard would deny exactly the requests this client exists for.
    Classification is used for the ROUTING decision alone, and
    ``_resolves_exclusively_public_async`` never raises.

    With the flag off this is byte-for-byte the previous behavior: no resolution,
    no delegate, a direct dial.

    Why this is needed at all: pooling the login callback behind an explicit
    transport (commit 05bbd1a4) silently removed httpx's environment-proxy
    support, because httpx applies its env-proxy mounts only when no custom
    transport is supplied. Before that the callback used a bare
    ``httpx.AsyncClient()``, which honored ``HTTPS_PROXY`` through the default
    ``trust_env=True``. In a proxy-only network that regression is a complete
    sign-in outage.
    """

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        self._init_proxy_delegates(kwargs)
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def _proxy_url_for(
        self,
        request: httpx.Request,
    ) -> str | None:
        """Return the proxy to send this request through, or None for direct."""
        config = _forward_proxy_config()
        if not config.enabled:
            return None

        url = request.url
        scheme = url.scheme
        hostname = url.host
        if not hostname:
            return None
        port = url.port or (443 if scheme == "https" else 80)

        if _no_proxy_matches(config.no_proxy, hostname, port):
            return None
        proxy_url = config.https_proxy if scheme == "https" else config.http_proxy
        if proxy_url is None:
            return None

        public = await _resolves_exclusively_public_async(hostname, port)
        if public is not True:
            # Internal, mixed, or unresolvable: stay direct. An in-cluster IdP and
            # the registry vends land here, which is why they never depend on
            # NO_PROXY being correct.
            return None

        # A public target on this client carries an operator client_secret (the
        # IdP token POST), so refuse cleartext: httpcore's plain forward path
        # sends the absolute URI and merges headers, handing the secret to the
        # proxy. An in-cluster http IdP never reaches here, it resolved private.
        if scheme != "https":
            raise UrlValidationError(
                str(url),
                "refusing an http target through a forward proxy on the plain "
                "egress client (an IdP token request carries a client secret, "
                "which the proxy would see in cleartext). Use https, or add this "
                "host to NO_PROXY so it is dialed directly",
            )

        _validate_proxy_url(proxy_url)
        logger.debug("plain egress: routing %s via the forward proxy", hostname)
        _record_egress_route("plain", "proxied", "ok")
        return proxy_url

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        proxy_url = await self._proxy_url_for(request)
        if proxy_url is None:
            return await super().handle_async_request(request)
        return await self._proxy_delegate(proxy_url).handle_async_request(request)

    async def aclose(self) -> None:
        await self._aclose_proxy_delegates()
        await super().aclose()


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


# ---------------------------------------------------------------------------
# Pooled, process-lifetime egress clients.
# ---------------------------------------------------------------------------
# The per-call ``guarded_async_client`` / ``httpx.AsyncClient(...)`` pattern opens
# a fresh TCP+TLS connection for every egress request (no keep-alive reuse). The
# accessors below return process-lifetime, connection-pooled clients safe to share
# across requests and users:
#   * SSRF safety is unchanged -- GuardedAsyncTransport validates + pins EVERY
#     request before pool checkout, and the pool is keyed by the pinned IP, so a
#     rebound hostname re-resolves to a new key and never reuses a stale connection.
#     NOTE: because the pool key is the pinned IP (not the hostname), two different
#     hostnames that both validate to the SAME public IP:port can coalesce onto one
#     TLS connection whose cert was verified for whichever host opened it. This is
#     safe -- each request is independently pinned and its Host header is correct,
#     and reaching host C over host B's connection requires C to already resolve to
#     that IP -- but it is a behavioral change vs the old per-call clients. Be precise
#     about what is lost: cert scope is checked only for the hostname that OPENS the
#     connection, since a reusing request performs no handshake. A request to C can
#     therefore succeed over B's connection where C's own cert would have failed. Do
#     NOT enable http2 on these clients (it would coalesce far more aggressively).
#   * No shared default identity state -- no default auth headers (callers pass the
#     credential per request). Cookie persistence is disabled via a no-store cookie
#     jar (see _disable_cookie_persistence / _NoStoreCookieJar): a Set-Cookie is never
#     stored, so it can never be replayed onto a later OR concurrent request sharing
#     the client. Credentials never ride cookies on these paths, so this is
#     defense-in-depth.
# Timeouts are passed PER REQUEST at the call site; the client default is only a
# fallback. Owned by each app's FastAPI lifespan (``aclose_shared_clients`` on
# shutdown). ``verify`` is part of the key so a ``verify=False`` client can never be
# reused where verification is expected.
_shared_guarded_clients: dict[tuple[str, bool | str], httpx.AsyncClient] = {}
_shared_plain_client: httpx.AsyncClient | None = None


def _pool_limits() -> httpx.Limits:
    s = _get_settings()
    return httpx.Limits(
        max_connections=s.egress_http_pool_max_connections,
        max_keepalive_connections=s.egress_http_pool_max_keepalive,
        keepalive_expiry=s.egress_http_pool_keepalive_expiry_seconds,
    )


class _NoStoreCookieJar(http.cookiejar.CookieJar):
    """A cookie jar that silently drops every cookie.

    Installing this on a pooled, shared client makes cookie handling stateless: a
    ``Set-Cookie`` is never stored, so it can never be replayed onto a later or
    concurrent request sharing the client. This is concurrency-safe by construction
    (there is no shared cookie state to race), unlike clearing the jar after each
    response. Credentials never ride cookies on the egress paths; this is
    defense-in-depth against a future endpoint that sets one.
    """

    def set_cookie(self, cookie: http.cookiejar.Cookie) -> None:  # noqa: D102
        return  # never persist


def _disable_cookie_persistence(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Swap in a no-store cookie jar so the shared client never persists cookies."""
    client.cookies.jar = _NoStoreCookieJar()
    return client


def shared_guarded_async_client(
    *,
    profile: _Profile = SKILL_PROFILE,
    verify: bool | str = True,
) -> httpx.AsyncClient:
    """Return a process-lifetime, connection-pooled, SSRF-guarded client, one per
    ``(profile, verify)``. Pass the timeout PER REQUEST. Do NOT pass per-client
    kwargs (``follow_redirects``/``base_url``/``headers``): a caller needing those
    keeps its own instance client (see the federation clients)."""
    key = (profile.name, verify)
    client = _shared_guarded_clients.get(key)
    if client is None or client.is_closed:
        client = _disable_cookie_persistence(
            httpx.AsyncClient(
                transport=GuardedAsyncTransport(
                    guard_profile=profile,
                    verify=verify,
                    retries=_get_settings().egress_http_pool_connect_retries,
                ),
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                limits=_pool_limits(),
            )
        )
        _shared_guarded_clients[key] = client
    return client


def shared_plain_async_client() -> httpx.AsyncClient:
    """Return a process-lifetime, connection-pooled PLAIN (un-SSRF-guarded) client
    for operator-configured, already-trusted targets ONLY: the in-cluster registry
    egress-token vend and the login IdP token/userinfo endpoints (from
    ``oauth2_providers.yml``, keyed by ``KEYCLOAK_URL`` etc.). Those are static
    operator config -- they may be in-cluster and HTTP, so they cannot use the
    HTTPS-only credentialed-OAuth guard -- and they are never request- or
    stored-URL-derived. NEVER use this for a request/stored-URL-derived target (a
    registrant proxy_pass_url, a federation peer): use ``shared_guarded_async_client``.

    Forward-proxy aware since issue #1834. A HOSTED IdP (Entra, Cognito, Auth0,
    Okta) is a public target, and in a proxy-only network a direct dial to it
    times out, which is a complete sign-in outage. The transport therefore routes
    a target that resolves exclusively to public addresses through the configured
    proxy, and keeps everything else direct. No pinning and no SSRF denial are
    added: an in-cluster ``http://keycloak:8080`` resolves private, so it stays
    direct and needs no ``NO_PROXY`` entry, and nor do the registry vends."""
    global _shared_plain_client
    if _shared_plain_client is None or _shared_plain_client.is_closed:
        _shared_plain_client = _disable_cookie_persistence(
            httpx.AsyncClient(
                transport=PlainProxyRoutedAsyncTransport(
                    retries=_get_settings().egress_http_pool_connect_retries,
                ),
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                limits=_pool_limits(),
            )
        )
    return _shared_plain_client


async def aclose_shared_clients() -> None:
    """Close all pooled egress clients (call from each app's lifespan shutdown).
    Accessors lazily rebuild the pool if used again afterwards."""
    global _shared_plain_client
    for client in list(_shared_guarded_clients.values()):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("error closing a shared guarded client", exc_info=True)
    _shared_guarded_clients.clear()
    if _shared_plain_client is not None:
        try:
            await _shared_plain_client.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("error closing the shared plain client", exc_info=True)
        _shared_plain_client = None


def reset_shared_clients_for_tests() -> None:
    """Drop pooled-client references without closing them (tests only)."""
    global _shared_plain_client
    _shared_guarded_clients.clear()
    _shared_plain_client = None


async def post_with_reconnect(
    client: httpx.AsyncClient,
    url: str,
    *,
    on_reset: Callable[[], None] | None = None,
    **kwargs: object,
) -> httpx.Response:
    """POST with a single transparent retry when a POOLED keep-alive connection was
    closed server/LB-side while idle. httpx does not auto-retry a non-idempotent
    POST on a half-open connection, so the first POST after an idle gap can raise
    ``RemoteProtocolError``. The dominant case (peer sent FIN before the request was
    written) is a clean re-POST; the rare residual (peer closed after processing)
    re-POSTs into a terminal error that the caller fails closed on (never a silent
    corruption). ``retries=`` on the transport covers connect failures; this covers
    a reset on connection REUSE before the response. Streaming POSTs are NOT wrapped
    (they are context managers); those rely on ``keepalive_expiry`` alone."""
    try:
        return await client.post(url, **kwargs)  # type: ignore[arg-type]
    except httpx.RemoteProtocolError:
        if on_reset is not None:
            on_reset()
        return await client.post(url, **kwargs)  # type: ignore[arg-type]
