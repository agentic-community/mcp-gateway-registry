"""Tests for the lightweight security-scan summary helpers.

These back the N+1 fix: list endpoints attach a per-item scan summary built from
a single bulk repository read instead of one fetch per card.
"""

from unittest.mock import AsyncMock

import pytest

from registry.services.scan_summary import (
    build_scan_summary,
    build_scan_summary_map,
)


class TestBuildScanSummary:
    """Tests for build_scan_summary (single document -> icon summary)."""

    def test_extracts_severity_counts_and_failure_flag(self):
        scan = {
            "server_path": "/cloudflare-docs",
            "scan_failed": False,
            "critical_issues": 1,
            "high_severity": 2,
            "medium_severity": 3,
            "low_severity": 4,
            "raw_output": {"big": "blob"},  # heavy field must be dropped
        }

        summary = build_scan_summary(scan)

        assert summary == {
            "scan_failed": False,
            "critical_issues": 1,
            "high_severity": 2,
            "medium_severity": 3,
            "low_severity": 4,
        }
        assert "raw_output" not in summary

    def test_missing_fields_default_to_safe_zeros(self):
        summary = build_scan_summary({"server_path": "/x"})

        assert summary == {
            "scan_failed": False,
            "critical_issues": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
        }


class TestBuildScanSummaryMap:
    """Tests for build_scan_summary_map (list -> path-keyed summary map)."""

    def test_keys_by_server_path(self):
        scans = [
            {"server_path": "/a", "critical_issues": 1},
            {"server_path": "/b", "high_severity": 2},
        ]

        result = build_scan_summary_map(scans)

        assert set(result) == {"/a", "/b"}
        assert result["/a"]["critical_issues"] == 1
        assert result["/b"]["high_severity"] == 2

    def test_keys_by_agent_path(self):
        result = build_scan_summary_map([{"agent_path": "/code-reviewer"}])
        assert "/code-reviewer" in result

    def test_keys_by_skill_path(self):
        result = build_scan_summary_map([{"skill_path": "/skills/pdf"}])
        assert "/skills/pdf" in result

    def test_skips_documents_without_a_path_key(self):
        result = build_scan_summary_map([{"critical_issues": 9}])
        assert result == {}

    def test_empty_input_returns_empty_map(self):
        assert build_scan_summary_map([]) == {}

    def test_keeps_latest_scan_per_path_regardless_of_input_order(self):
        # The DocumentDB repos store one doc per scan run (insert, not upsert),
        # so list_all() returns history with a path appearing multiple times.
        # The map must reflect the newest scan_timestamp, not input/cursor order.
        scans = [
            {"server_path": "/a", "scan_timestamp": "2026-01-01T00:00:00Z", "critical_issues": 5},
            {"server_path": "/a", "scan_timestamp": "2026-06-01T00:00:00Z", "critical_issues": 0},
            {"server_path": "/a", "scan_timestamp": "2026-03-01T00:00:00Z", "critical_issues": 2},
        ]

        # Newest-first (how list_all sorts it)...
        newest_first = build_scan_summary_map(
            sorted(scans, key=lambda s: s["scan_timestamp"], reverse=True)
        )
        # ...and oldest-first, to prove the result is order-independent.
        oldest_first = build_scan_summary_map(sorted(scans, key=lambda s: s["scan_timestamp"]))

        assert newest_first["/a"]["critical_issues"] == 0
        assert oldest_first["/a"]["critical_issues"] == 0

    def test_normalizes_trailing_slash_to_match_list_item_paths(self):
        # get_latest matches "/foo" and "/foo/"; the summary map must collapse
        # them so a scan stored as "/foo/" still keys onto a list item at "/foo".
        result = build_scan_summary_map([{"server_path": "/foo/", "high_severity": 3}])
        assert "/foo" in result
        assert result["/foo"]["high_severity"] == 3

    def test_document_missing_timestamp_does_not_overwrite_a_timestamped_one(self):
        scans = [
            {"server_path": "/a", "scan_timestamp": "2026-06-01T00:00:00Z", "critical_issues": 7},
            {"server_path": "/a", "critical_issues": 0},  # no timestamp
        ]
        result = build_scan_summary_map(scans)
        assert result["/a"]["critical_issues"] == 7


class TestServiceGetScanSummaries:
    """The scanner services build the map from the repo's list_latest()."""

    @pytest.mark.asyncio
    async def test_server_scanner_reads_list_latest_not_list_all(self):
        from registry.services.security_scanner import SecurityScannerService

        service = SecurityScannerService()
        service._scan_repo = AsyncMock()
        service._scan_repo.list_latest.return_value = [
            {"server_path": "/a", "critical_issues": 1},
        ]

        summaries = await service.get_scan_summaries()

        # Must use the collapsed-at-the-data-layer read, never the full history.
        service._scan_repo.list_latest.assert_awaited_once()
        service._scan_repo.list_all.assert_not_called()
        assert summaries["/a"]["critical_issues"] == 1

    @pytest.mark.asyncio
    async def test_server_scanner_swallows_repo_errors(self):
        from registry.services.security_scanner import SecurityScannerService

        service = SecurityScannerService()
        service._scan_repo = AsyncMock()
        service._scan_repo.list_latest.side_effect = RuntimeError("db down")

        # A bulk-load failure must degrade to "no summaries", never break the list.
        assert await service.get_scan_summaries() == {}

    @pytest.mark.asyncio
    async def test_agent_scanner_reads_list_latest(self):
        from registry.services.agent_scanner import AgentScannerService

        service = AgentScannerService()
        service._scan_repo = AsyncMock()
        service._scan_repo.list_latest.return_value = [
            {"agent_path": "/code-reviewer", "high_severity": 2},
        ]

        summaries = await service.get_scan_summaries()

        service._scan_repo.list_latest.assert_awaited_once()
        assert summaries["/code-reviewer"]["high_severity"] == 2

    @pytest.mark.asyncio
    async def test_skill_scanner_reads_list_latest(self):
        from registry.services.skill_scanner import SkillScannerService

        service = SkillScannerService()
        service._scan_repo = AsyncMock()
        service._scan_repo.list_latest.return_value = [
            {"skill_path": "/skills/pdf", "low_severity": 4},
        ]

        summaries = await service.get_scan_summaries()

        service._scan_repo.list_latest.assert_awaited_once()
        assert summaries["/skills/pdf"]["low_severity"] == 4
