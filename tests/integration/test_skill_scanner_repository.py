"""
Property-based test for skill security scan repository create-then-retrieve round-trip.

# Feature: skill-scanner-integration, Property 5: Repository create-then-retrieve round-trip

**Validates: Requirements 6.2**
"""

import asyncio
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from registry.repositories.file.skill_security_scan_repository import (
    FileSkillSecurityScanRepository,
)

VALID_ANALYZERS = ["static", "behavioral", "llm", "meta", "virustotal", "ai-defense"]


def _scan_result_dict_strategy():
    """Strategy for generating valid scan result dicts with realistic fields."""
    return st.fixed_dictionaries(
        {
            "skill_path": st.from_regex(r"/[a-z][a-z0-9\-]{0,30}", fullmatch=True),
            "scan_timestamp": st.from_regex(
                r"2026-0[1-9]-[012][0-9]T[01][0-9]:[0-5][0-9]:[0-5][0-9]Z",
                fullmatch=True,
            ),
            "is_safe": st.booleans(),
            "critical_issues": st.integers(min_value=0, max_value=50),
            "high_severity": st.integers(min_value=0, max_value=50),
            "medium_severity": st.integers(min_value=0, max_value=50),
            "low_severity": st.integers(min_value=0, max_value=50),
            "analyzers_used": st.lists(
                st.sampled_from(VALID_ANALYZERS),
                min_size=1,
                max_size=6,
                unique=True,
            ),
            "raw_output": st.just({}),
            "scan_failed": st.booleans(),
            "error_message": st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        }
    )


class TestRepositoryCreateRetrieveRoundTrip:
    """Property 5: Repository create-then-retrieve round-trip."""

    @given(scan_result=_scan_result_dict_strategy())
    @settings(max_examples=50)
    def test_create_then_retrieve_preserves_fields(self, scan_result):
        """Persisting a scan result via create() and retrieving via get_latest() preserves all fields."""

        async def _run():
            with tempfile.TemporaryDirectory() as tmp_dir:
                repo = FileSkillSecurityScanRepository()
                repo._scans_dir = Path(tmp_dir) / "skill_security_scans"
                repo._scans = {}

                created = await repo.create(scan_result)
                assert created is True

                retrieved = await repo.get_latest(scan_result["skill_path"])
                assert retrieved is not None

                for key, value in scan_result.items():
                    assert key in retrieved, f"Missing key: {key}"
                    assert retrieved[key] == value, (
                        f"Mismatch for key '{key}': expected {value!r}, got {retrieved[key]!r}"
                    )

        asyncio.get_event_loop().run_until_complete(_run())
