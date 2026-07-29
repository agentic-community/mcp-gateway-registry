"""Canonical IP-category SSRF checks shared across egress guards.

This module centralizes the "is this host a denied network target?" logic so the
project has ONE place that knows about the bypass tricks — obfuscated IPv4
literal encodings (decimal/octal/hex/short-form), IPv6 transports that embed an
IPv4 address (IPv4-mapped ::ffff:0:0/96, NAT64 64:ff9b::/96, 6to4 2002::/16,
Teredo 2001::/32), and the always-deny categories (link-local/metadata,
unspecified, reserved, multicast). Prior to this module the logic was hand-rolled
in several call sites and had already drifted (each copy knew a different subset
of the tricks), which is exactly how an SSRF gap slips in.

Scope caveat: default-mode completeness (``allow_private=False``) relies on
CPython's ``ipaddress`` category flags, whose treatment of special-purpose
ranges can evolve between Python releases. The repo pins Python 3.14 and the
tests pin the security-relevant ranges so a downgrade or semantic change fails
loudly.

Two entry points:
- ``coerce_ip_literal(host)``: parse a host string to an IP address, matching
  ``inet_aton``/glibc/nginx semantics (so non-canonical spellings of the metadata
  IP are recognized, not treated as opaque hostnames). Returns None for genuine
  hostnames (which must be resolved before they can be category-checked).
- ``ip_denial_reason(ip, allow_private, allowed_cidrs=())``: classify an
  already-parsed IP in a two-tier model — (1) cloud/workload credential
  endpoints are hard-denied and never overridable; (2) an explicit
  ``allowed_cidrs`` entry re-permits loopback/private/CGNAT only; hard-denied
  link-local, unspecified, reserved, and multicast categories remain closed;
  (3) the coarse ``allow_private`` bool relaxes only loopback, private, and
  CGNAT destinations. Embedded IPv4 forms (mapped/NAT64/6to4/Teredo) are
  unwrapped first.

Callers that accept hostnames should resolve first (``socket.getaddrinfo``) and
run every resolved IP through ``ip_denial_reason``. Callers that only guard
literal targets statically use ``coerce_ip_literal`` + ``ip_denial_reason``.
"""

import ipaddress

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


def ip_denial_reason(
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
