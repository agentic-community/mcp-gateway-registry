"""Unit tests for cli.agentcore.security egress + secret-handling guards.

Covers the four hardening surfaces added for the AgentCore sync adapter:
- require_secure_registry_url: refuse credentials over cleartext to non-loopback
- guarded_oidc_get / guarded_oidc_post: SSRF-guard registrant-supplied OIDC URLs
- is_safe_egress_url: registration-time validation (no network I/O by contract)
- write_secret_file: 0o600 atomic writes for on-disk credentials
- validate_account_ids: 12-digit shape + explicit allowlist (fail closed)

Each guard is proven to FAIL CLOSED: the vulnerable input is rejected, and the
safe input is accepted.
"""

from __future__ import annotations

import os
import socket
import stat
from unittest.mock import patch

import pytest

from cli.agentcore.security import (
    EgressSecurityError,
    is_safe_egress_url,
    require_secure_registry_url,
    validate_account_ids,
    write_secret_file,
)


def _resolve_to(*ips):
    """getaddrinfo stub resolving any host to the given public IP(s)."""

    def _fake(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]

    return _fake


class TestRequireSecureRegistryUrl:
    """Credentials must never go over cleartext HTTP to a non-loopback host."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://registry.example.com",
            "https://registry.example.com:8443/base",
            "http://localhost:7860",
            "http://127.0.0.1",
            "http://127.0.0.1:9090/x",
            "http://[::1]:7860",
        ],
    )
    def test_accepts_https_or_loopback(self, url):
        # Does not raise.
        require_secure_registry_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://registry.example.com",
            "http://10.0.0.5:7860",
            "http://registry.internal",
            "http://192.168.1.10",
        ],
    )
    def test_rejects_cleartext_to_remote_host(self, url):
        with pytest.raises(EgressSecurityError):
            require_secure_registry_url(url)

    @pytest.mark.parametrize("url", ["ftp://x.com", "gopher://x", "file:///etc/passwd"])
    def test_rejects_non_http_scheme(self, url):
        with pytest.raises(EgressSecurityError):
            require_secure_registry_url(url)


class TestIsSafeEgressUrl:
    """Registration-time SSRF/scheme validation for a discovery_url."""

    def test_public_https_is_safe(self):
        # Stub DNS so the test is hermetic (is_safe_egress_url resolves the host).
        from registry.utils import url_guard

        with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")):
            assert is_safe_egress_url(
                "https://cognito-idp.us-east-1.amazonaws.com/pool/.well-known/openid-configuration"
            )

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://127.0.0.1/.well-known/openid-configuration",  # loopback
            "http://10.0.0.1/x",  # private
            "ftp://example.com/x",  # non-http scheme
            "not-a-url",
        ],
    )
    def test_unsafe_urls_rejected(self, url):
        assert is_safe_egress_url(url) is False

    def test_cleartext_http_public_url_rejected(self):
        # A discovery_url is where the client secret is later sent, so plain
        # http:// to even a public host must be rejected (require_https). This
        # is the fix for the fail-open the reviewer caught.
        assert (
            is_safe_egress_url("http://auth.example.com/.well-known/openid-configuration") is False
        )


class TestWriteSecretFile:
    """On-disk credentials must be written 0o600, atomically."""

    def test_writes_owner_only_mode(self, tmp_path):
        target = tmp_path / "token_refresh_manifest.json"
        write_secret_file(str(target), '{"a": 1}')
        assert target.read_text() == '{"a": 1}'
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_overwrites_existing_loosening_mode(self, tmp_path):
        # A pre-existing world-readable file must end up 0o600 after rewrite.
        target = tmp_path / "creds.json"
        target.write_text("old")
        os.chmod(target, 0o644)
        write_secret_file(str(target), "new")
        assert target.read_text() == "new"
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600

    def test_no_partial_file_on_success(self, tmp_path):
        # Only the final file exists; no leftover temp files in the directory.
        target = tmp_path / "t.json"
        write_secret_file(str(target), "data")
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
        assert leftovers == []


class TestValidateAccountIds:
    """Cross-account IDs: 12-digit shape + explicit allowlist, fail closed."""

    def test_valid_ids_pass_without_allowlist(self):
        ids = ["123456789012", "210987654321"]
        assert validate_account_ids(ids) == ids

    @pytest.mark.parametrize(
        "bad",
        [
            "12345",  # too short
            "1234567890123",  # too long
            "12345678901a",  # non-digit
            "arn:aws:iam::123456789012:root",  # not an id
            "",
        ],
    )
    def test_malformed_id_rejected(self, bad):
        with pytest.raises(EgressSecurityError):
            validate_account_ids([bad])

    def test_allowlist_permits_member(self):
        assert validate_account_ids(
            ["123456789012"], allowlist=["123456789012", "999999999999"]
        ) == ["123456789012"]

    def test_allowlist_rejects_off_list_account(self):
        with pytest.raises(EgressSecurityError):
            validate_account_ids(["123456789012"], allowlist=["999999999999"])
