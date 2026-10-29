"""Exit-code handling of the A2A security scanner.

The a2a-scanner CLI prints its full JSON result to stdout and then exits 1 when
the result holds a HIGH finding. An uncaught error in the CLI also exits 1, but
prints a traceback and no JSON. The registry must treat the first as a completed
scan with findings and the second as a scanner failure; treating both as a
failure dropped the real finding and left the agent enabled when
SECURITY_BLOCK_ON_SCAN_FAILURE was false.
"""

import json
import subprocess  # nosec B404 - only CalledProcessError is used, to mimic subprocess.run
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from registry.schemas.agent_security import AgentSecurityScanResult
from registry.services.agent_scanner import AgentScannerService

AGENT_PATH: str = "/weather-agent"
ANALYZERS: str = "yara,spec"
SCANNER_TIMEOUT_SECONDS: int = 30
SCANNER_FAILED_MESSAGE: str = "Agent security scanner failed"
TRACEBACK_STDERR: str = (
    "Traceback (most recent call last):\n"
    '  File "a2ascanner/cli.py", line 1, in scan_card\n'
    "RuntimeError: analyzer crashed\n"
)

# Marks "argument not given" for _scanner_stdout, where None means "omit the key".
_UNSET: Any = object()

AGENT_CARD: dict = {
    "name": "Weather Agent",
    "description": "Answers questions about the weather forecast for a city.",
    "url": "https://agent.example.com/a2a",
    "path": AGENT_PATH,
}


def _finding(
    severity: str,
    threat_name: str = "AGENT CARD SPOOFING",
    analyzer: str = "yara",
) -> dict:
    """Build one scanner finding as the a2a-scanner CLI prints it."""
    return {
        "severity": severity,
        "threat_name": threat_name,
        "analyzer": analyzer,
        "summary": "Card claims an identity it does not control.",
    }


def _completed_process(
    returncode: int,
    stdout: str = "",
    stderr: str = "",
) -> MagicMock:
    """Build the object subprocess.run returns for the mocked scanner."""
    completed = MagicMock()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = stderr
    return completed


def _scanner_stdout(
    findings: list[dict],
    high_severity_count: Any = _UNSET,
) -> str:
    """Build the JSON document the scanner prints for the given findings.

    high_severity_count defaults to the number of printed findings whose severity
    is HIGH, which is what the scanner reports when taxonomy and in-memory
    severities agree. Pass a value to model a divergence, or None to leave the key
    out the way older scanner output does.
    """
    if high_severity_count is _UNSET:
        high_severity_count = sum(1 for finding in findings if finding["severity"] == "HIGH")
    result: dict = {
        "target_name": AGENT_CARD["name"],
        "target_type": "agent_card",
        "status": "completed",
        "analyzers": ["yara"],
        "findings": findings,
        "total_findings": len(findings),
    }
    if high_severity_count is not None:
        result["high_severity_count"] = high_severity_count
    return json.dumps(result, indent=2)


def _fake_subprocess_run(
    completed: MagicMock,
):
    """Build a stand-in for subprocess.run that honors check=True like the real one."""

    def _run(cmd, *args, **kwargs):
        if kwargs.get("check") and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                cmd,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed

    return _run


def _make_service() -> AgentScannerService:
    """Build an AgentScannerService whose scan repository is a mock."""
    with patch("registry.services.agent_scanner.get_security_scan_repository"):
        service = AgentScannerService()
    service._scan_repo = MagicMock(create=AsyncMock())
    return service


def _run_scanner(
    completed: MagicMock,
) -> dict:
    """Run _run_a2a_scanner against a mocked scanner process."""
    service = _make_service()
    with patch(
        "registry.services.agent_scanner.subprocess.run",
        side_effect=_fake_subprocess_run(completed),
    ):
        return service._run_a2a_scanner(
            agent_card=AGENT_CARD,
            agent_path=AGENT_PATH,
            analyzers=ANALYZERS,
            timeout=SCANNER_TIMEOUT_SECONDS,
        )


async def _scan_agent(
    completed: MagicMock,
) -> tuple[AgentSecurityScanResult, AgentScannerService]:
    """Run scan_agent against a mocked scanner process; return the result and service."""
    service = _make_service()
    with patch(
        "registry.services.agent_scanner.subprocess.run",
        side_effect=_fake_subprocess_run(completed),
    ):
        result = await service.scan_agent(
            agent_card=AGENT_CARD,
            agent_path=AGENT_PATH,
            analyzers=ANALYZERS,
            timeout=SCANNER_TIMEOUT_SECONDS,
        )
    return result, service


@pytest.mark.unit
class TestScannerExitOneWithHighSeverityCount:
    """Exit 1 with a positive high_severity_count is a completed scan.

    The scanner exits 1 when its in-memory HIGH count is above zero, and prints
    that count as high_severity_count. The severity printed on each finding comes
    from its threat taxonomy, so it can differ from the in-memory one.
    """

    def test_high_finding_is_returned_in_raw_output(self):
        high = _finding("HIGH")
        completed = _completed_process(1, stdout=_scanner_stdout([high]))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [high]}}
        assert raw_output["scan_results"]["findings"] == [high]

    def test_finding_printed_as_critical_is_accepted_when_scanner_counted_high(self):
        critical = _finding("CRITICAL", threat_name="CODE EXECUTION")
        completed = _completed_process(1, stdout=_scanner_stdout([critical], high_severity_count=1))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [critical]}}

    def test_finding_printed_as_medium_is_accepted_when_scanner_counted_high(self):
        medium = _finding("MEDIUM", threat_name="CONTEXT POISONING", analyzer="llm")
        completed = _completed_process(1, stdout=_scanner_stdout([medium], high_severity_count=1))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"llm": {"findings": [medium]}}
        assert raw_output["scan_results"]["high_severity_count"] == 1

    def test_ansi_colored_stdout_is_still_parsed(self):
        high = _finding("HIGH")
        stdout = "\x1b[31mscanning card\x1b[0m\n" + _scanner_stdout([high])
        completed = _completed_process(1, stdout=stdout)

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"]["yara"]["findings"] == [high]

    @pytest.mark.asyncio
    async def test_scan_agent_records_unsafe_scan_not_a_failed_scan(self):
        completed = _completed_process(1, stdout=_scanner_stdout([_finding("HIGH")]))

        result, service = await _scan_agent(completed)

        assert result.is_safe is False
        assert result.scan_failed is False
        assert result.high_severity == 1
        service._scan_repo.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scan_agent_counts_a_high_the_scanner_printed_as_medium(self):
        medium = _finding("MEDIUM", threat_name="CONTEXT POISONING", analyzer="llm")
        completed = _completed_process(1, stdout=_scanner_stdout([medium], high_severity_count=1))

        result, _ = await _scan_agent(completed)

        assert result.is_safe is False
        assert result.scan_failed is False
        assert result.high_severity >= 1


@pytest.mark.unit
class TestScannerExitOneWithoutHighSeverityCount:
    """Without a usable high_severity_count, the printed severities decide."""

    def test_high_finding_is_accepted_when_the_count_key_is_absent(self):
        high = _finding("HIGH")
        completed = _completed_process(1, stdout=_scanner_stdout([high], high_severity_count=None))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [high]}}

    def test_critical_finding_is_accepted_when_the_count_key_is_absent(self):
        critical = _finding("CRITICAL", threat_name="CODE EXECUTION")
        completed = _completed_process(
            1, stdout=_scanner_stdout([critical], high_severity_count=None)
        )

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [critical]}}

    def test_severity_match_is_case_insensitive_in_the_fallback(self):
        completed = _completed_process(
            1, stdout=_scanner_stdout([_finding("High")], high_severity_count=None)
        )

        raw_output = _run_scanner(completed)

        assert len(raw_output["analysis_results"]["yara"]["findings"]) == 1

    def test_only_lower_severities_without_the_count_key_raises(self):
        completed = _completed_process(
            1,
            stdout=_scanner_stdout([_finding("MEDIUM")], high_severity_count=None),
            stderr="boom",
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_bool_count_is_not_a_count_and_falls_back_to_printed_severities(self):
        high = _finding("HIGH")
        completed = _completed_process(1, stdout=_scanner_stdout([high], high_severity_count=True))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [high]}}


@pytest.mark.unit
class TestScannerExitOneIsAFailure:
    """Exit 1 is a scanner failure unless stdout shows the scanner found a HIGH issue."""

    def test_traceback_without_json_raises(self):
        completed = _completed_process(1, stdout="", stderr=TRACEBACK_STDERR)

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE) as excinfo:
            _run_scanner(completed)

        assert "analyzer crashed" in str(excinfo.value)

    def test_zero_high_severity_count_with_lower_severities_raises(self):
        findings = [_finding("MEDIUM"), _finding("LOW")]
        completed = _completed_process(
            1, stdout=_scanner_stdout(findings, high_severity_count=0), stderr="boom"
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_zero_high_severity_count_wins_over_a_printed_high(self):
        completed = _completed_process(
            1,
            stdout=_scanner_stdout([_finding("HIGH")], high_severity_count=0),
            stderr="boom",
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_bool_count_with_only_lower_severities_raises(self):
        completed = _completed_process(
            1,
            stdout=_scanner_stdout([_finding("MEDIUM")], high_severity_count=True),
            stderr="boom",
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_json_without_findings_raises(self):
        completed = _completed_process(
            1, stdout=_scanner_stdout([], high_severity_count=0), stderr="boom"
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_json_that_is_not_an_object_raises(self):
        completed = _completed_process(1, stdout='[{"severity": "HIGH"}]', stderr="boom")

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    @pytest.mark.asyncio
    async def test_scan_agent_records_failed_scan_for_traceback(self):
        completed = _completed_process(1, stdout="", stderr=TRACEBACK_STDERR)

        result, _ = await _scan_agent(completed)

        assert result.scan_failed is True
        assert result.is_safe is False
        assert result.high_severity == 0


@pytest.mark.unit
class TestScannerOtherExitCodes:
    """Only exit 0 and exit 1 with a HIGH finding are accepted."""

    @pytest.mark.parametrize("returncode", [2, 127, -9])
    def test_other_nonzero_exit_raises_even_with_high_json(self, returncode):
        completed = _completed_process(
            returncode, stdout=_scanner_stdout([_finding("HIGH")]), stderr="usage error"
        )

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE) as excinfo:
            _run_scanner(completed)

        assert "usage error" in str(excinfo.value)

    def test_exit_zero_still_parses_and_returns_findings(self):
        medium = _finding("MEDIUM", analyzer="spec")
        completed = _completed_process(0, stdout=_scanner_stdout([medium]))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"spec": {"findings": [medium]}}
        assert raw_output["scan_results"]["findings"] == [medium]

    def test_exit_zero_with_no_json_still_raises_value_error(self):
        completed = _completed_process(0, stdout="scan complete, nothing to report")

        with pytest.raises(ValueError, match="No JSON found"):
            _run_scanner(completed)


@pytest.mark.unit
class TestExitZeroVerdictIsUnchanged:
    """A scan that exited 0 has high_severity_count 0, so its verdict is as before."""

    @pytest.mark.asyncio
    async def test_medium_only_scan_with_zero_count_stays_safe(self):
        completed = _completed_process(
            0, stdout=_scanner_stdout([_finding("MEDIUM")], high_severity_count=0)
        )

        result, _ = await _scan_agent(completed)

        assert result.is_safe is True
        assert result.scan_failed is False
        assert result.high_severity == 0
        assert result.medium_severity == 1

    @pytest.mark.asyncio
    async def test_printed_high_with_zero_count_is_still_counted_once(self):
        completed = _completed_process(
            0, stdout=_scanner_stdout([_finding("HIGH")], high_severity_count=0)
        )

        result, _ = await _scan_agent(completed)

        assert result.is_safe is False
        assert result.high_severity == 1

    @pytest.mark.asyncio
    async def test_scan_without_the_count_key_is_counted_from_findings_only(self):
        completed = _completed_process(
            0, stdout=_scanner_stdout([_finding("LOW")], high_severity_count=None)
        )

        result, _ = await _scan_agent(completed)

        assert result.is_safe is True
        assert result.low_severity == 1
        assert result.high_severity == 0


@pytest.mark.unit
class TestExitOneWithMalformedResults:
    """Exit 1 results whose shape the registry cannot trust."""

    def test_positive_count_with_no_findings_is_accepted(self):
        completed = _completed_process(1, stdout=_scanner_stdout([], high_severity_count=1))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {}
        assert raw_output["scan_results"]["high_severity_count"] == 1

    @pytest.mark.asyncio
    async def test_positive_count_with_no_findings_is_still_unsafe(self):
        completed = _completed_process(1, stdout=_scanner_stdout([], high_severity_count=1))

        result, _ = await _scan_agent(completed)

        assert result.is_safe is False
        assert result.scan_failed is False
        assert result.high_severity == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("findings", [None, "none", 3])
    async def test_positive_count_with_non_list_findings_is_recorded_as_a_failed_scan(
        self, findings
    ):
        stdout = json.dumps({"high_severity_count": 1, "findings": findings})
        completed = _completed_process(1, stdout=stdout)

        result, _ = await _scan_agent(completed)

        assert result.scan_failed is True
        assert result.is_safe is False

    @pytest.mark.parametrize("findings", [None, "none", {"severity": "HIGH"}])
    def test_non_list_findings_without_a_count_raise(self, findings):
        stdout = json.dumps({"target_name": "Weather Agent", "findings": findings})
        completed = _completed_process(1, stdout=stdout, stderr="boom")

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_non_integer_count_falls_back_to_printed_severities(self):
        stdout = json.dumps({"high_severity_count": "1", "findings": [_finding("MEDIUM")]})
        completed = _completed_process(1, stdout=stdout, stderr="boom")

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)

    def test_non_dict_findings_entries_are_ignored_by_the_fallback(self):
        stdout = json.dumps({"findings": ["HIGH", None]})
        completed = _completed_process(1, stdout=stdout, stderr="boom")

        with pytest.raises(RuntimeError, match=SCANNER_FAILED_MESSAGE):
            _run_scanner(completed)


@pytest.mark.unit
class TestScannerOutputParsing:
    """How the scanner's stdout becomes raw_output."""

    def test_findings_from_one_analyzer_are_grouped_together(self):
        first = _finding("MEDIUM", threat_name="FIRST")
        second = _finding("LOW", threat_name="SECOND")
        completed = _completed_process(0, stdout=_scanner_stdout([first, second]))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [first, second]}}

    def test_findings_from_different_analyzers_are_kept_apart(self):
        yara = _finding("MEDIUM", analyzer="yara")
        spec = _finding("LOW", analyzer="spec")
        completed = _completed_process(0, stdout=_scanner_stdout([yara, spec]))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {
            "yara": {"findings": [yara]},
            "spec": {"findings": [spec]},
        }

    def test_finding_without_an_analyzer_is_filed_under_unknown(self):
        finding = {"severity": "LOW", "threat_name": "NO ANALYZER"}
        stdout = json.dumps({"findings": [finding]})
        completed = _completed_process(0, stdout=stdout)

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"unknown": {"findings": [finding]}}

    def test_json_array_output_is_returned_without_grouping(self):
        completed = _completed_process(0, stdout='[{"severity": "HIGH"}]')

        raw_output = _run_scanner(completed)

        assert raw_output == {
            "analysis_results": {},
            "scan_results": [{"severity": "HIGH"}],
        }

    def test_json_array_after_log_lines_is_found(self):
        completed = _completed_process(0, stdout='scanning card\n[{"severity": "LOW"}]')

        raw_output = _run_scanner(completed)

        assert raw_output["scan_results"] == [{"severity": "LOW"}]

    def test_json_object_after_log_lines_is_found(self):
        finding = _finding("LOW")
        completed = _completed_process(0, stdout="scanning card\n" + _scanner_stdout([finding]))

        raw_output = _run_scanner(completed)

        assert raw_output["analysis_results"] == {"yara": {"findings": [finding]}}

    def test_log_lines_around_unparseable_json_raise_value_error(self):
        completed = _completed_process(0, stdout="scanning card\n{not json")

        with pytest.raises(ValueError):
            _run_scanner(completed)

    @pytest.mark.asyncio
    async def test_array_output_from_a_clean_exit_is_recorded_as_safe_with_no_findings(self):
        completed = _completed_process(0, stdout="[]")

        result, _ = await _scan_agent(completed)

        assert result.is_safe is True
        assert result.scan_failed is False
        assert result.high_severity == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
