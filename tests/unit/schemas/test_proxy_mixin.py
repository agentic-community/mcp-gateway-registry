"""Unit tests for ProxyableMixin, resolve_proxy_target, and the SSRF egress guard.

_assert_egress_allowed is a best-effort STATIC egress check that delegates to the
canonical registry.utils.url_guard (PROXY_PROFILE, resolve=False), whose inline
IP classifier is exhaustively tested in tests/unit/utils/test_url_guard.py. It is deliberately
weaker than the fetch-time pinned transport (it does not resolve DNS); the
authoritative rebind defense is that transport + the network egress policy. These
tests confirm the guard is wired into the model edge correctly and rejects
literal-IP bypasses; they do not claim it is the sole SSRF control. They lock in
the vectors so a future edit cannot silently weaken it:

- always-denied regardless of the allow-private flag: link-local (incl. the cloud
  metadata IP 169.254.169.254), IPv6 link-local, and the unspecified address
  (0.0.0.0 / ::) which Linux routes to localhost on connect();
- IPv4-mapped IPv6 (::ffff:169.254.169.254) normalized BEFORE the category check
  so the metadata IP cannot be smuggled past the guard;
- loopback / private / reserved / multicast denied unless allow-private is set;
- hostnames pass the static guard (DNS is resolved downstream; the network-layer
  egress policy is the real rebind defense) — documented best-effort behavior;
- non-http(s) schemes rejected.
"""

import pytest

from registry.schemas.proxy_mixin import (
    EgressPolicyError,
    ProxyableMixin,
    _assert_egress_allowed,
    resolve_proxy_target,
)


def _set_allow_private(monkeypatch, value: bool) -> None:
    """Point settings.gateway_proxy_allow_private_targets at ``value``.

    _assert_egress_allowed now delegates to url_guard, whose PROXY_PROFILE
    allowlist is @lru_cached (settings are immutable per production process). The
    cache must be cleared for a test that flips the flag, or the delegated guard
    reads a stale allowlist.
    """
    from registry.core import config
    from registry.utils import url_guard

    monkeypatch.setattr(config.settings, "gateway_proxy_allow_private_targets", value, raising=True)
    url_guard._proxy_allowlist.cache_clear()


@pytest.mark.unit
class TestEgressGuardAlwaysDenied:
    """Vectors denied regardless of the allow-private flag."""

    @pytest.mark.parametrize("allow_private", [False, True])
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # AWS/GCP/Azure metadata
            "https://169.254.169.254/",
            "http://[fe80::1]/",  # IPv6 link-local
            "http://[::ffff:169.254.169.254]/",  # IPv4-mapped metadata IP
            "http://0.0.0.0/",  # unspecified -> localhost on connect
            "http://[::]/",  # IPv6 unspecified
        ],
    )
    def test_denied_even_when_allow_private(self, monkeypatch, allow_private, url):
        _set_allow_private(monkeypatch, allow_private)
        with pytest.raises(EgressPolicyError):
            _assert_egress_allowed(url)

    @pytest.mark.parametrize("allow_private", [False, True])
    @pytest.mark.parametrize(
        "url",
        [
            "http://2852039166/",  # decimal for 169.254.169.254 (metadata)
            "http://0xA9FEA9FE/",  # hex for 169.254.169.254
            "http://0251.0376.0251.0376/",  # dotted-octal for 169.254.169.254
            "http://169.254.169.254./",  # trailing-dot canonical
            "http://2130706433/",  # decimal for 127.0.0.1 (loopback)
            "http://0x7f000001/",  # hex for 127.0.0.1
        ],
    )
    def test_alternate_ip_encodings_denied(self, monkeypatch, allow_private, url):
        """Non-canonical IPv4 spellings of metadata/loopback must not slip through.

        inet_aton (and therefore glibc's resolver and nginx) interpret decimal,
        hex, octal, and trailing-dot forms as real IPv4 addresses. A guard that
        only parses canonical dotted-quad would let http://2852039166/ reach the
        metadata endpoint. loopback forms here are denied because allow_private
        never relaxes link-local/metadata, and the loopback ones fall under the
        private gate — but the metadata forms must be denied in BOTH flag states.
        """
        _set_allow_private(monkeypatch, allow_private)
        # metadata forms: always denied; loopback forms: denied unless allow_private.
        is_loopback_form = url in ("http://2130706433/", "http://0x7f000001/")
        if is_loopback_form and allow_private:
            _assert_egress_allowed(url)  # loopback permitted when opted in
        else:
            with pytest.raises(EgressPolicyError):
                _assert_egress_allowed(url)


@pytest.mark.unit
class TestEgressGuardPrivateGatedByFlag:
    """Loopback/private/reserved/multicast: denied by default, allowed when opted in."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/",  # loopback
            "http://10.0.0.5/",  # private class A
            "http://192.168.1.10/",  # private class C
            "http://172.16.0.1/",  # private class B
            "http://[::1]/",  # IPv6 loopback
            "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
            "http://240.0.0.1/",  # reserved
            "http://224.0.0.1/",  # multicast
        ],
    )
    def test_denied_by_default(self, monkeypatch, url):
        _set_allow_private(monkeypatch, False)
        with pytest.raises(EgressPolicyError):
            _assert_egress_allowed(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/",
            "http://10.0.0.5/",
            "http://192.168.1.10/",
            "http://[::1]/",
        ],
    )
    def test_allowed_when_flag_set(self, monkeypatch, url):
        _set_allow_private(monkeypatch, True)
        # Must not raise.
        _assert_egress_allowed(url)


@pytest.mark.unit
class TestEgressGuardAllowed:
    """Public IPs and hostnames pass the static guard."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://93.184.216.34/",  # public IPv4 (example.com)
            "https://api.github.com/repos",  # hostname -> resolved downstream
            "https://dashboard.internal.example.com/app",
        ],
    )
    def test_allowed(self, monkeypatch, url):
        _set_allow_private(monkeypatch, False)
        _assert_egress_allowed(url)


@pytest.mark.unit
class TestEgressGuardScheme:
    """Only http(s) is accepted."""

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/",
            "file:///etc/passwd",
            "gopher://example.com/",
            "ws://example.com/",
            "not-a-url",
        ],
    )
    def test_non_http_scheme_rejected(self, monkeypatch, url):
        _set_allow_private(monkeypatch, False)
        with pytest.raises(ValueError):
            _assert_egress_allowed(url)


@pytest.mark.unit
class TestProxyableMixinValidator:
    """The field_validator surfaces the egress guard at the model edge."""

    def test_none_target_ok(self, monkeypatch):
        _set_allow_private(monkeypatch, False)
        m = ProxyableMixin(is_proxied=False, proxy_target_url=None)
        assert m.proxy_target_url is None

    def test_public_target_ok(self, monkeypatch):
        _set_allow_private(monkeypatch, False)
        m = ProxyableMixin(is_proxied=True, proxy_target_url="https://api.github.com/")
        assert m.is_proxied is True

    def test_mixin_itself_is_read_safe(self, monkeypatch):
        """The mixin does NOT raise on a denied target at construction.

        Storage models inherit the mixin and are reconstructed from the DB on
        every read; raising here would vanish a bypass-written record on load.
        The egress raise lives on the request/patch models (API edge) via
        egress_guard_validator, and the authoritative net is the persist/render
        guard. See the ProxyableMixin note and egress_guard_validator.
        """
        _set_allow_private(monkeypatch, False)
        # Must NOT raise — read-safe.
        m = ProxyableMixin(is_proxied=True, proxy_target_url="http://169.254.169.254/")
        assert m.proxy_target_url == "http://169.254.169.254/"

    def test_egress_guard_validator_raises_at_edge(self, monkeypatch):
        """The reusable edge validator (attached to request models) DOES raise."""
        from registry.schemas.proxy_mixin import EgressPolicyError, egress_guard_validator

        _set_allow_private(monkeypatch, False)
        with pytest.raises(EgressPolicyError):
            egress_guard_validator("http://169.254.169.254/")
        assert egress_guard_validator(None) is None
        assert egress_guard_validator("https://ok.example/") == "https://ok.example/"

    def test_defaults_are_safe(self):
        m = ProxyableMixin()
        assert m.is_proxied is False
        assert m.proxy_target_url is None
        assert m.proxy_resolved_ips == []
        assert m.proxy_target_host is None
        assert m.proxy_disabled_reason is None


@pytest.mark.unit
class TestResolveProxyTarget:
    """Per-entity-type effective-target resolution."""

    def test_not_proxied_returns_none(self):
        assert resolve_proxy_target("skill", {"is_proxied": False}) is None

    def test_explicit_target_wins(self):
        doc = {"is_proxied": True, "proxy_target_url": "https://x.example.com/"}
        assert resolve_proxy_target("skill", doc) == "https://x.example.com/"

    def test_mcp_server_falls_back_to_proxy_pass_url(self):
        doc = {"is_proxied": True, "proxy_pass_url": "https://backend:9000/"}
        assert resolve_proxy_target("mcp_server", doc) == "https://backend:9000/"

    def test_mcp_local_deployment_never_proxied(self):
        doc = {"is_proxied": True, "deployment": "local", "proxy_pass_url": "http://x/"}
        assert resolve_proxy_target("mcp_server", doc) is None

    def test_agent_falls_back_to_url(self):
        doc = {"is_proxied": True, "url": "https://agent.example.com/"}
        assert resolve_proxy_target("a2a_agent", doc) == "https://agent.example.com/"

    def test_skill_requires_explicit_target(self):
        # No proxy_target_url and no native URL field -> not proxyable.
        assert resolve_proxy_target("skill", {"is_proxied": True}) is None

    def test_disabled_reason_treated_as_not_proxied(self):
        doc = {
            "is_proxied": True,
            "proxy_target_url": "https://x.example.com/",
            "proxy_disabled_reason": "target now resolves to a denied IP",
        }
        assert resolve_proxy_target("skill", doc) is None


@pytest.mark.unit
class TestEgressGuardErrorTaxonomy:
    """Every rejection is an EgressPolicyError (a ValueError subclass), so the
    Pydantic edge surfaces it as a clean 4xx.

    NOTE: after converging onto url_guard (one code path for register + fetch),
    scheme errors and network denials both raise EgressPolicyError rather than the
    former split (plain ValueError for scheme). Nothing catches EgressPolicyError
    *specifically* — every consumer uses ``except ValueError`` — so collapsing the
    taxonomy has no functional effect; it just removes a distinction with no
    consumer. Both remain ValueError subclasses.
    """

    def test_network_denial_is_egress_policy_error(self, monkeypatch):
        _set_allow_private(monkeypatch, False)
        with pytest.raises(EgressPolicyError):
            _assert_egress_allowed("http://169.254.169.254/")

    def test_bad_scheme_is_value_error(self, monkeypatch):
        # A non-http(s) scheme is rejected as a ValueError (EgressPolicyError), so
        # the Pydantic field validator surfaces it as a 4xx rather than a 500.
        _set_allow_private(monkeypatch, False)
        with pytest.raises(ValueError):
            _assert_egress_allowed("file:///etc/passwd")

    def test_userinfo_host_still_checked(self, monkeypatch):
        """A userinfo@host URL must be judged on the real host, not the userinfo."""
        _set_allow_private(monkeypatch, False)
        with pytest.raises(EgressPolicyError):
            _assert_egress_allowed("http://user:pass@169.254.169.254/")

    def test_empty_host_rejected(self, monkeypatch):
        """An http URL with no host is now rejected (url_guard requires a host) —
        stricter than the former passthrough, and safer."""
        _set_allow_private(monkeypatch, False)
        with pytest.raises(ValueError):
            _assert_egress_allowed("http:///path")


@pytest.mark.unit
class TestResolveProxyTargetHostnamePassthrough:
    """resolve_proxy_target does NOT re-run the egress guard — documents the
    contract that hostname re-validation is the refresh/network layer's job."""

    def test_hostname_target_returned_verbatim(self):
        doc = {"is_proxied": True, "proxy_target_url": "https://api.github.com/"}
        assert resolve_proxy_target("skill", doc) == "https://api.github.com/"


@pytest.mark.unit
class TestClientMaxBodySizeValidator:
    """The nginx size token is validated at config load (config-injection guard)."""

    @pytest.mark.parametrize("good", ["1m", "512k", "2G", "10485760", "1M", "1024K"])
    def test_valid_tokens_accepted(self, monkeypatch, good):
        from registry.core.config import Settings

        s = Settings(gateway_generic_client_max_body_size=good)
        assert s.gateway_generic_client_max_body_size == good

    @pytest.mark.parametrize("bad", ["1mb", "abc", "1 m", "-1m", "1m; rm -rf", "", "1.5m"])
    def test_invalid_tokens_rejected(self, monkeypatch, bad):
        from pydantic import ValidationError

        from registry.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(gateway_generic_client_max_body_size=bad)
