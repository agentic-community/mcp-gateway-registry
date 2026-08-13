"""Unit tests for the url_guard IP-category SSRF classifier (registry.utils.url_guard).

These lock in the two properties the whole gateway-proxy egress control rests on:
1. ``coerce_ip_literal`` recognizes every spelling of an IPv4 literal that
   inet_aton/glibc/nginx would (decimal/octal/hex/short-form/trailing-dot), so an
   obfuscated metadata IP is category-checked, not treated as an opaque hostname;
   and it falls back to "hostname" (None) for overflow / genuine names.
2. ``_ip_denial_reason`` denies the metadata endpoint via every wrapper (mapped,
   NAT64) and keeps link-local/unspecified/reserved/multicast denied regardless
   of the allow-private flag, which relaxes ONLY loopback/private-unicast.
"""

import ipaddress

import pytest

from registry.utils.url_guard import _ip_denial_reason, coerce_ip_literal

_METADATA = ipaddress.ip_address("169.254.169.254")
_LOOPBACK = ipaddress.ip_address("127.0.0.1")


@pytest.mark.unit
class TestCoerceIpLiteralMetadataSpellings:
    """Every inet_aton spelling of the metadata IP resolves to the same address."""

    @pytest.mark.parametrize(
        "host",
        [
            "169.254.169.254",  # canonical
            "169.254.169.254.",  # trailing dot
            "2852039166",  # 1-part decimal
            "0xA9FEA9FE",  # 1-part hex
            "0251.0376.0251.0376",  # 4-part dotted octal
            "0xA9.0xFE.0xA9.0xFE",  # 4-part dotted hex
        ],
    )
    def test_metadata_spellings(self, host):
        assert coerce_ip_literal(host) == _METADATA


@pytest.mark.unit
class TestCoerceIpLiteralShortForms:
    """inet_aton short forms: the last part absorbs the remaining low bytes."""

    def test_two_part_short_form(self):
        # 127.1 -> 127.0.0.1 (1 absorbs the low 24 bits)
        assert coerce_ip_literal("127.1") == _LOOPBACK

    def test_three_part_short_form(self):
        # 127.0.1 -> 127.0.0.1 (last part absorbs low 16 bits)
        assert coerce_ip_literal("127.0.1") == _LOOPBACK

    def test_one_part_loopback(self):
        assert coerce_ip_literal("2130706433") == _LOOPBACK

    def test_zero_is_unspecified(self):
        assert coerce_ip_literal("0") == ipaddress.ip_address("0.0.0.0")

    def test_single_int_loopback_denied_end_to_end(self):
        # coercion + classification together: the decimal loopback form is denied.
        ip = coerce_ip_literal("2130706433")
        assert ip is not None
        assert _ip_denial_reason(ip, allow_private=False) is not None


@pytest.mark.unit
class TestCoerceIpLiteralHostnameFallback:
    """Overflow and genuine names return None (treated as hostnames)."""

    @pytest.mark.parametrize(
        "host",
        [
            "256.1.1.1",  # part > 255
            "4294967296",  # > 2^32
            "1.2.3.4.5",  # too many parts
            "08.0.0.1",  # invalid octal digit
            "-1.2.3.4",  # negative part
            "api.github.com",  # real hostname
            "example.com",
            "",  # empty
            "not-an-ip",
        ],
    )
    def test_returns_none(self, host):
        assert coerce_ip_literal(host) is None


@pytest.mark.unit
class TestCgnatAssumption:
    """Document WHY _CGNAT_NET exists: CPython does not flag CGNAT as private."""

    def test_cgnat_not_is_private(self):
        # If a future CPython flags 100.64.0.0/10 as is_private this assertion
        # fails loudly — at which point the explicit _CGNAT_NET range is redundant
        # but harmless. The point is to notice the semantics change.
        assert ipaddress.ip_address("100.64.0.1").is_private is False


@pytest.mark.unit
class TestCoerceIpLiteralPublicPreserved:
    """A legitimate non-canonical PUBLIC IP still parses (no over-deny)."""

    def test_public_decimal(self):
        # 93.184.216.34 (example.com) as decimal
        packed = (93 << 24) | (184 << 16) | (216 << 8) | 34
        assert coerce_ip_literal(str(packed)) == ipaddress.ip_address("93.184.216.34")


@pytest.mark.unit
class TestIpDenialReasonAlwaysDenied:
    """Denied regardless of allow_private, including via IPv6 wrappers."""

    @pytest.mark.parametrize("allow_private", [False, True])
    @pytest.mark.parametrize(
        "ip",
        [
            "169.254.169.254",  # EC2 IMDS
            "169.254.170.2",  # ECS task credentials
            "169.254.170.23",  # EKS Pod Identity (IPv4)
            "100.100.100.200",  # Alibaba Cloud ECS metadata
            "fd00:ec2::23",  # EKS Pod Identity (IPv6)
            "fd00:ec2::23%eth0",  # scoped EKS Pod Identity (scope is not identity)
            "fd00:ec2::254",  # EC2 IMDS (IPv6)
            "fd00:ec2::254%eth0",  # scoped EC2 IMDS (must not bypass exact match)
            "fe80::1",  # IPv6 link-local
            "::ffff:169.254.169.254",  # IPv4-mapped IMDS
            "::ffff:169.254.170.2",  # IPv4-mapped ECS credentials
            "::ffff:100.100.100.200",  # IPv4-mapped Alibaba metadata
            "64:ff9b::6464:64c8",  # NAT64-embedded Alibaba metadata
            "2002:6464:64c8::",  # 6to4-embedded Alibaba metadata
            "2001::9b9b:9b37",  # Teredo-embedded Alibaba metadata
            "64:ff9b::a9fe:aa17",  # NAT64-embedded EKS Pod Identity IPv4
            "64:ff9b::a9fe:a9fe",  # NAT64-embedded metadata (RFC 6052)
            "2002:a9fe:a9fe::",  # 6to4-embedded metadata (RFC 3056)
            "2001::5601:5601",  # Teredo-embedded metadata (client v4 XOR'd, RFC 4380)
            "::169.254.169.254",  # IPv4-compatible (::/8 -> reserved safety net)
            "64:ff9b:1::a9fe:a9fe",  # RFC 8215 local-use NAT64 (not unwrapped; ::/8 reserved)
            "0.0.0.0",  # unspecified
            "::",  # IPv6 unspecified
            "240.0.0.1",  # reserved
            "255.255.255.255",  # limited broadcast (reserved)
            "224.0.0.1",  # multicast
        ],
    )
    def test_always_denied(self, allow_private, ip):
        assert _ip_denial_reason(ipaddress.ip_address(ip), allow_private=allow_private) is not None


@pytest.mark.unit
class TestIpDenialReasonPrivateGate:
    """Loopback/private/CGNAT/ULA relaxed ONLY when allow_private is set."""

    _RELAXABLE = [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # private A
        "192.168.1.1",  # private C
        "::1",  # IPv6 loopback (also is_reserved — must still relax)
        "100.64.0.1",  # CGNAT (RFC 6598) — is_private=False, denied by explicit range
        "fd00::1",  # IPv6 ULA
    ]

    @pytest.mark.parametrize("ip", _RELAXABLE)
    def test_denied_by_default(self, ip):
        assert _ip_denial_reason(ipaddress.ip_address(ip), allow_private=False) is not None

    @pytest.mark.parametrize("ip", _RELAXABLE)
    def test_allowed_when_flag_set(self, ip):
        assert _ip_denial_reason(ipaddress.ip_address(ip), allow_private=True) is None

    def test_reserved_stays_denied_even_with_flag(self):
        # blast-radius check: allow_private must NOT open reserved/multicast
        assert _ip_denial_reason(ipaddress.ip_address("240.0.0.1"), allow_private=True) is not None
        assert _ip_denial_reason(ipaddress.ip_address("224.0.0.1"), allow_private=True) is not None


@pytest.mark.unit
class TestIpDenialReasonWrappedRelaxation:
    """allow_private relaxes an embedded-IPv4 wrapper by its unwrapped category."""

    def test_mapped_private_relaxable(self):
        ip = ipaddress.ip_address("::ffff:10.0.0.1")
        assert _ip_denial_reason(ip, allow_private=False) is not None
        assert _ip_denial_reason(ip, allow_private=True) is None

    def test_nat64_private_relaxable(self):
        # 64:ff9b::0a00:0001 embeds 10.0.0.1
        ip = ipaddress.ip_address("64:ff9b::0a00:0001")
        assert _ip_denial_reason(ip, allow_private=False) is not None
        assert _ip_denial_reason(ip, allow_private=True) is None

    def test_6to4_private_relaxable(self):
        # 2002:0a00:0001:: embeds 10.0.0.1
        ip = ipaddress.ip_address("2002:0a00:0001::")
        assert _ip_denial_reason(ip, allow_private=False) is not None
        assert _ip_denial_reason(ip, allow_private=True) is None

    def test_mapped_public_allowed_both(self):
        ip = ipaddress.ip_address("::ffff:8.8.8.8")
        assert _ip_denial_reason(ip, allow_private=False) is None
        assert _ip_denial_reason(ip, allow_private=True) is None


@pytest.mark.unit
class TestIpDenialReasonPublicAllowed:
    """Public unicast is allowed in both flag states."""

    @pytest.mark.parametrize("allow_private", [False, True])
    @pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
    def test_public_allowed(self, allow_private, ip):
        assert _ip_denial_reason(ipaddress.ip_address(ip), allow_private=allow_private) is None


@pytest.mark.unit
class TestCredentialEndpointsCannotBeRelaxed:
    @pytest.mark.parametrize(
        "ip,cidr",
        [
            ("169.254.169.254", "169.254.0.0/16"),
            ("169.254.170.2", "169.254.0.0/16"),
            ("169.254.170.23", "169.254.0.0/16"),
            ("100.100.100.200", "100.64.0.0/10"),
            ("fd00:ec2::23", "fd00:ec2::/64"),
            ("fd00:ec2::23%eth0", "fd00:ec2::/64"),
            ("fd00:ec2::254", "fd00:ec2::/64"),
            ("fd00:ec2::254%eth0", "fd00:ec2::/64"),
        ],
    )
    def test_allow_private_and_cidr_do_not_override(self, ip, cidr):
        assert (
            _ip_denial_reason(
                ipaddress.ip_address(ip),
                allow_private=True,
                allowed_cidrs=(ipaddress.ip_network(cidr),),
            )
            == "cloud/workload credential endpoint"
        )


@pytest.mark.unit
class TestHardDeniedCategoriesCannotBeRelaxed:
    @pytest.mark.parametrize(
        "ip,cidr",
        [
            ("169.254.10.10", "169.254.0.0/16"),
            ("fe80::1", "fe80::/10"),
            ("0.0.0.0", "0.0.0.0/0"),
            ("::", "::/0"),
        ],
    )
    def test_allow_private_and_cidr_do_not_override(self, ip, cidr):
        assert (
            _ip_denial_reason(
                ipaddress.ip_address(ip),
                allow_private=True,
                allowed_cidrs=(ipaddress.ip_network(cidr),),
            )
            is not None
        )


@pytest.mark.unit
class TestAlibabaMetadataPrecision:
    """Only Alibaba's documented metadata IP is hard-denied, not a broad range."""

    @pytest.mark.parametrize("ip", ["100.100.100.199", "100.100.100.201", "100.63.255.255"])
    def test_adjacent_addresses_are_not_hard_denied_as_metadata(self, ip):
        reason = _ip_denial_reason(
            ipaddress.ip_address(ip),
            allow_private=True,
            allowed_cidrs=(ipaddress.ip_network("100.64.0.0/10"),),
        )
        assert reason is None
