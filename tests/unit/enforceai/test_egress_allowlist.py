"""
Unit tests for EnforceAI egress allowlist matcher (Phase 2).
"""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from auth_server.enforceai.egress.allowlist import (
    check_proxy_pass_url,
    normalize_allowlist_entry_value,
)
from auth_server.enforceai.models.egress_allowlist import (
    EgressAllowlistEntryRecord,
)


@pytest.mark.unit
class TestEgressAllowlistMatcher:
    def test_hostname_match(
        self,
    ) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entries = [
            EgressAllowlistEntryRecord(
                entry_id=1,
                kind="hostname",
                value="example.com",
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        ]
        decision = check_proxy_pass_url(
            proxy_pass_url="https://example.com/mcp",
            entries=entries,
            now=now,
        )
        assert decision.allowed is True
        assert decision.matched_entry is not None
        assert decision.matched_entry.entry_id == 1

    def test_domain_suffix_match(
        self,
    ) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entries = [
            EgressAllowlistEntryRecord(
                entry_id=1,
                kind="domain-suffix",
                value="example.com",
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        ]
        assert (
            check_proxy_pass_url(
                proxy_pass_url="https://api.example.com/",
                entries=entries,
                now=now,
            ).allowed
            is True
        )
        assert (
            check_proxy_pass_url(
                proxy_pass_url="https://example.com/",
                entries=entries,
                now=now,
            ).allowed
            is True
        )
        assert (
            check_proxy_pass_url(
                proxy_pass_url="https://evil.com/",
                entries=entries,
                now=now,
            ).allowed
            is False
        )

    def test_ip_cidr_match(
        self,
    ) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entries = [
            EgressAllowlistEntryRecord(
                entry_id=1,
                kind="ip-cidr",
                value="127.0.0.0/8",
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        ]
        decision = check_proxy_pass_url(
            proxy_pass_url="http://127.0.0.1:8080/sse",
            entries=entries,
            now=now,
        )
        assert decision.allowed is True

    def test_ttl_expires_entry(
        self,
    ) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entries = [
            EgressAllowlistEntryRecord(
                entry_id=1,
                kind="hostname",
                value="example.com",
                expires_at=now - timedelta(seconds=1),
                created_at=now,
                updated_at=now,
            )
        ]
        decision = check_proxy_pass_url(
            proxy_pass_url="https://example.com/",
            entries=entries,
            now=now,
        )
        assert decision.allowed is False

    def test_rejects_userinfo_and_non_http_scheme(
        self,
    ) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entries: list[EgressAllowlistEntryRecord] = []

        assert (
            check_proxy_pass_url(
                proxy_pass_url="file:///etc/passwd",
                entries=entries,
                now=now,
            ).allowed
            is False
        )
        assert (
            check_proxy_pass_url(
                proxy_pass_url="http://user:pass@example.com/",
                entries=entries,
                now=now,
            ).allowed
            is False
        )


@pytest.mark.unit
class TestAllowlistEntryValueNormalization:
    def test_normalizes_domain_suffix(
        self,
    ) -> None:
        assert (
            normalize_allowlist_entry_value(kind="domain-suffix", value=" .Example.COM. ")
            == "example.com"
        )

    def test_rejects_domain_suffix_with_wildcard(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="must not include"):
            normalize_allowlist_entry_value(kind="domain-suffix", value="*.example.com")

    def test_normalizes_ip_cidr(
        self,
    ) -> None:
        assert (
            normalize_allowlist_entry_value(kind="ip-cidr", value="127.0.0.1")
            == "127.0.0.1/32"
        )

