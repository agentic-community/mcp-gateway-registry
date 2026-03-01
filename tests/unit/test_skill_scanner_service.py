"""
Tests for SkillScannerService: property tests and unit tests.

# Feature: skill-scanner-integration
# Property 2: Scanner output parsing preserves findings
# Property 3: Safety determination invariant

**Validates: Requirements 3.2, 3.3, 3.5, 3.6, 3.7, 8.1, 8.2, 8.3, 9.1, 9.2**
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from registry.services.skill_scanner import SkillScannerService


VALID_SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
VALID_ANALYZERS = ["static", "behavioral", "llm", "meta", "virustotal", "ai-defense"]

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scanner_json_output(findings: list) -> str:
    """Build a valid JSON string mimicking skill-scanner CLI output."""
    return json.dumps({"findings": findings})


def _make_finding(severity: str, analyzer: str) -> dict:
    """Create a minimal finding dict."""
    return {
        "severity": severity,
        "analyzer": analyzer,
        "threat_names": [f"test-{severity.lower()}"],
        "threat_summary": f"Test {severity} finding",
        "is_safe": severity not in ("CRITICAL", "HIGH"),
    }


def _create_service() -> SkillScannerService:
    """Create a SkillScannerService with a mocked repository."""
    with patch(
        "registry.services.skill_scanner.get_skill_security_scan_repository"
    ) as mock_factory:
        mock_repo = MagicMock()
        mock_repo.create = MagicMock(return_value=True)
        mock_repo.get_latest = MagicMock(return_value=None)
        mock_factory.return_value = mock_repo
        service = SkillScannerService()
        # Force the lazy property to use our mock
        service._scan_repo = mock_repo
    return service


# ---------------------------------------------------------------------------
# Property 2: Scanner output parsing preserves findings
# ---------------------------------------------------------------------------


def _finding_strategy():
    """Strategy for generating a single finding dict."""
    return st.fixed_dictionaries(
        {
            "severity": st.sampled_from(VALID_SEVERITIES),
            "analyzer": st.sampled_from(VALID_ANALYZERS),
            "threat_names": st.lists(st.text(min_size=1, max_size=20), max_size=3),
            "threat_summary": st.text(max_size=100),
            "is_safe": st.booleans(),
        }
    )


def _ansi_prefix_strategy():
    """Strategy for optional ANSI escape code prefix."""
    return st.one_of(
        st.just(""),
        st.just("\x1b[31m"),
        st.just("\x1b[0m"),
        st.just("\x1b[1;32m"),
    )


class TestParseOutputPreservesFindings:
    """Property 2: Scanner output parsing preserves findings."""

    @given(
        findings=st.lists(_finding_strategy(), min_size=0, max_size=10),
        ansi_prefix=_ansi_prefix_strategy(),
    )
    @settings(max_examples=100)
    def test_all_findings_preserved_by_analyzer(self, findings, ansi_prefix):
        """Parsing output preserves all findings under their correct analyzer keys."""
        service = _create_service()
        raw_json = _build_scanner_json_output(findings)
        stdout = ansi_prefix + raw_json

        parsed = service._parse_scanner_output(stdout)

        # Count total findings across all analyzers in parsed output
        total_parsed = 0
        for analyzer_data in parsed["analysis_results"].values():
            total_parsed += len(analyzer_data.get("findings", []))

        assert total_parsed == len(findings)

        # Verify each finding is under the correct analyzer key
        for finding in findings:
            analyzer = finding["analyzer"]
            analyzer_findings = parsed["analysis_results"].get(analyzer, {}).get("findings", [])
            assert finding in analyzer_findings


# ---------------------------------------------------------------------------
# Property 3: Safety determination invariant
# ---------------------------------------------------------------------------


def _severity_list_strategy():
    """Strategy for generating a list of severity strings."""
    return st.lists(st.sampled_from(VALID_SEVERITIES), min_size=0, max_size=20)


class TestSafetyDeterminationInvariant:
    """Property 3: Safety determination invariant."""

    @given(severities=_severity_list_strategy())
    @settings(max_examples=100)
    def test_safety_matches_severity_counts(self, severities):
        """is_safe is True iff critical==0 and high==0; severity sum equals total findings."""
        service = _create_service()

        # Build raw_output with findings
        findings = [_make_finding(sev, "static") for sev in severities]
        raw_output = {"analysis_results": {"static": {"findings": findings}}}

        is_safe, critical, high, medium, low = service._analyze_scan_results(raw_output)

        expected_critical = severities.count("CRITICAL")
        expected_high = severities.count("HIGH")
        expected_medium = severities.count("MEDIUM")
        expected_low = severities.count("LOW")

        assert critical == expected_critical
        assert high == expected_high
        assert medium == expected_medium
        assert low == expected_low
        assert is_safe == (expected_critical == 0 and expected_high == 0)
        assert critical + high + medium + low == len(severities)


# ---------------------------------------------------------------------------
# Unit tests for skill scanner service (Task 5.4)
# ---------------------------------------------------------------------------


class TestSkillScannerServiceUnit:
    """Unit tests for SkillScannerService edge cases and error handling."""

    def test_parse_safe_fixture(self):
        """Parsing safe fixture produces no findings and is_safe=True."""
        service = _create_service()
        with open(FIXTURES_DIR / "skill_scan_safe_output.json") as f:
            raw_json = f.read()

        parsed = service._parse_scanner_output(raw_json)
        is_safe, critical, high, medium, low = service._analyze_scan_results(parsed)

        assert is_safe is True
        assert critical == 0
        assert high == 0
        assert medium == 0
        assert low == 0

    def test_parse_unsafe_fixture(self):
        """Parsing unsafe fixture produces correct severity counts and is_safe=False."""
        service = _create_service()
        with open(FIXTURES_DIR / "skill_scan_unsafe_output.json") as f:
            raw_json = f.read()

        parsed = service._parse_scanner_output(raw_json)
        is_safe, critical, high, medium, low = service._analyze_scan_results(parsed)

        assert is_safe is False
        assert critical == 1
        assert high == 1

    def test_parse_medium_fixture(self):
        """Parsing medium fixture produces correct counts and is_safe=True."""
        service = _create_service()
        with open(FIXTURES_DIR / "skill_scan_medium_output.json") as f:
            raw_json = f.read()

        parsed = service._parse_scanner_output(raw_json)
        is_safe, critical, high, medium, low = service._analyze_scan_results(parsed)

        assert is_safe is True
        assert critical == 0
        assert high == 0
        assert medium == 1
        assert low == 1

    def test_run_skill_scanner_timeout(self):
        """CLI timeout raises RuntimeError with timeout message."""
        service = _create_service()

        with patch("registry.services.skill_scanner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="skill-scanner", timeout=5)

            with pytest.raises(RuntimeError, match="timed out"):
                service._run_skill_scanner(
                    skill_path="/test",
                    skill_content_path="/tmp/test",
                    analyzers="static",
                    timeout=5,
                )

    def test_run_skill_scanner_nonzero_exit(self):
        """CLI non-zero exit raises RuntimeError with stderr."""
        service = _create_service()

        with patch("registry.services.skill_scanner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="skill-scanner", stderr="scanner error"
            )

            with pytest.raises(RuntimeError, match="Skill scanner failed"):
                service._run_skill_scanner(
                    skill_path="/test",
                    skill_content_path="/tmp/test",
                    analyzers="static",
                    timeout=120,
                )

    def test_run_skill_scanner_no_target_raises(self):
        """Missing both content_path and md_url raises ValueError."""
        service = _create_service()

        with pytest.raises(ValueError, match="Either skill_content_path or skill_md_url"):
            service._run_skill_scanner(
                skill_path="/test",
                analyzers="static",
                timeout=120,
            )

    def test_analyze_empty_results(self):
        """Empty analysis_results returns safe with zero counts."""
        service = _create_service()
        is_safe, critical, high, medium, low = service._analyze_scan_results(
            {"analysis_results": {}}
        )

        assert is_safe is True
        assert critical == 0
        assert high == 0
        assert medium == 0
        assert low == 0

    def test_parse_strips_ansi_codes(self):
        """ANSI escape codes are stripped before JSON parsing."""
        service = _create_service()
        ansi_json = '\x1b[31m{"findings": []}\x1b[0m'

        parsed = service._parse_scanner_output(ansi_json)
        assert parsed["scan_results"] == {"findings": []}
        assert parsed["analysis_results"] == {}
