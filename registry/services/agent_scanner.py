"""
Agent Scanner Service

This service provides security scanning functionality for A2A agents during registration.
It wraps the CLI A2A scanner and makes it available to API endpoints with proper
configuration and error handling.
"""

import asyncio
import json
import logging
import os
import re
import subprocess  # nosec B404
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.config import settings
from ..repositories.factory import get_security_scan_repository
from ..schemas.agent_security import AgentSecurityScanConfig, AgentSecurityScanResult
from .scanner_llm_analyzer import (
    ensure_llm_analyzer_available,
)

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "agent_security_scans"

# a2a-scanner (pinned commit 8f6e27f) prints its full JSON result to stdout and
# then exits 1 when its in-memory result holds a HIGH-severity finding. That count
# is printed as "high_severity_count". The severities printed per finding come
# from the scanner's threat taxonomy and can differ from the in-memory one (a HIGH
# finding may print as MEDIUM or CRITICAL). An uncaught scanner error also exits 1,
# but with a traceback and no JSON, so the exit code alone cannot tell the two apart.
A2A_SCANNER_FINDINGS_EXIT_CODE: int = 1
A2A_SCANNER_BLOCKING_SEVERITIES: frozenset[str] = frozenset({"high", "critical"})


def _drop_none_values(
    value: Any,
) -> Any:
    """Return a copy of a JSON-like value with every None removed, at any depth.

    The registry builds the scan card with AgentCard.model_dump(), which writes
    every unset optional field as null ("signature": null, "signatures": null,
    and "examples": null on each skill). The agent never sent those. The A2A
    scanner's YARA rule UnauthorizedNetworkAccess_UnsignedMessages flags
    '"signature": null' as a missing signature (HIGH), and the spec analyzer
    flags a null skill 'examples' as invalid, so every unsigned agent was marked
    unsafe and, by default, disabled (issue #1921). In the A2A card format a
    null field means the same as an absent one, so dropping it loses nothing;
    real values such as "verify_signature": false or an empty-string signature
    are kept and still scanned.
    """
    if isinstance(value, dict):
        return {key: _drop_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value if item is not None]
    return value


def _parse_a2a_scanner_stdout(
    stdout: str,
) -> Any:
    """Parse the JSON the a2a-scanner CLI prints to stdout.

    Strips ANSI color codes, then tries a direct json.loads and falls back to the
    first line that starts a JSON object or array. Raises ValueError when no JSON
    can be found or parsed.
    """
    stdout = stdout.strip()

    # Remove ANSI color codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    stdout = ansi_escape.sub("", stdout)

    # Try to parse as JSON directly
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # If direct parse fails, try to find JSON in output
        json_start = -1
        for i in range(len(stdout) - 1):
            if stdout[i] == "{" and (i == 0 or stdout[i - 1] in "\n\r"):
                json_start = i
                break

        if json_start == -1:
            # Try array format
            for i in range(len(stdout) - 1):
                if stdout[i] == "[" and (i == 0 or stdout[i - 1] in "\n\r"):
                    json_start = i
                    break

        if json_start == -1:
            raise ValueError("No JSON found in A2A scanner output")

        json_str = stdout[json_start:]
        return json.loads(json_str)


def _build_a2a_raw_output(
    scan_results: Any,
) -> dict:
    """Wrap parsed a2a-scanner results in the raw_output shape the registry stores."""
    raw_output: dict = {
        "analysis_results": {},
        "scan_results": scan_results,
    }

    # Extract findings and organize by analyzer
    if isinstance(scan_results, dict):
        findings = scan_results.get("findings", [])
        # Findings is always a list from a2a-scanner
        for finding in findings:
            analyzer_name = finding.get("analyzer", "unknown")
            if analyzer_name not in raw_output["analysis_results"]:
                raw_output["analysis_results"][analyzer_name] = {"findings": []}
            raw_output["analysis_results"][analyzer_name]["findings"].append(finding)

    return raw_output


def _scanner_high_severity_count(
    scan_results: Any,
) -> int | None:
    """Return the scanner's own "high_severity_count", or None when unusable.

    The count is the predicate the scanner uses for its exit-1 decision. A missing
    key, a non-integer value, and a bool (an int subclass in Python) are all unusable.
    """
    if not isinstance(scan_results, dict):
        return None
    count = scan_results.get("high_severity_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return None
    return count


def _has_blocking_findings(
    scan_results: Any,
) -> bool:
    """Return True when parsed scanner results show the scanner found a HIGH issue.

    Uses the scanner's own "high_severity_count" when it is usable. Falls back to
    looking for a HIGH or CRITICAL severity in the printed findings only when the
    count is absent or unusable (older scanner output).
    """
    if not isinstance(scan_results, dict):
        return False
    high_count = _scanner_high_severity_count(scan_results)
    if high_count is not None:
        return high_count > 0
    findings = scan_results.get("findings")
    if not isinstance(findings, list):
        return False
    return any(
        isinstance(finding, dict)
        and str(finding.get("severity", "")).lower() in A2A_SCANNER_BLOCKING_SEVERITIES
        for finding in findings
    )


class AgentScannerService:
    """Service for scanning A2A agents for security vulnerabilities."""

    def __init__(self):
        """Initialize the agent scanner service."""
        self._ensure_output_directory()
        self._scan_repo = get_security_scan_repository()

    def _ensure_output_directory(self) -> Path:
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    def get_scan_config(self) -> AgentSecurityScanConfig:
        """Get agent security scan configuration from settings."""
        return AgentSecurityScanConfig(
            enabled=settings.agent_security_scan_enabled,
            scan_on_registration=settings.agent_security_scan_on_registration,
            block_unsafe_agents=settings.agent_security_block_unsafe_agents,
            analyzers=settings.agent_security_analyzers,
            scan_timeout_seconds=settings.agent_security_scan_timeout,
            llm_api_key=settings.a2a_scanner_llm_api_key or os.getenv("A2A_SCANNER_LLM_API_KEY"),
            add_security_pending_tag=settings.agent_security_add_pending_tag,
            block_on_scan_failure=settings.security_block_on_scan_failure,
        )

    async def scan_agent(
        self,
        agent_card: dict,
        agent_path: str,
        analyzers: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> AgentSecurityScanResult:
        """
        Scan an A2A agent for security vulnerabilities.

        Args:
            agent_card: Agent card dictionary to scan
            agent_path: Path identifier for the agent (e.g., /code-reviewer)
            analyzers: Comma-separated list of analyzers to use (overrides config)
            api_key: Azure OpenAI API key for LLM-based analysis (overrides config)
            timeout: Scan timeout in seconds (overrides config)

        Returns:
            AgentSecurityScanResult containing scan results

        Raises:
            Exception: If scan completely fails
        """
        config = self.get_scan_config()

        # Use config values if not provided
        if analyzers is None:
            analyzers = config.analyzers
        if api_key is None:
            api_key = config.llm_api_key
        if timeout is None:
            timeout = config.scan_timeout_seconds

        logger.info(f"Starting agent security scan for {agent_path} with analyzers: {analyzers}")

        try:
            # Fails the scan before the scanner starts if 'llm' is requested but
            # LiteLLM is absent; the A2A scanner would otherwise skip it silently.
            ensure_llm_analyzer_available(analyzers, "AGENT_SECURITY_ANALYZERS")
            # Run the scan in a thread pool to avoid blocking
            raw_output = await asyncio.to_thread(
                self._run_a2a_scanner,
                agent_card=agent_card,
                agent_path=agent_path,
                analyzers=analyzers,
                api_key=api_key,
                timeout=timeout,
            )

            # Analyze results
            is_safe, critical, high, medium, low = self._analyze_scan_results(raw_output)

            # Record the URL that was actually scanned: the real backend
            # (proxy_pass_url) in reverse-proxy mode, else the advertised url.
            agent_url = (
                agent_card.get("proxy_pass_url")
                or agent_card.get("proxyPassUrl")
                or agent_card.get("url")
            )

            # Create result object
            result = AgentSecurityScanResult(
                agent_path=agent_path,
                agent_url=str(agent_url) if agent_url else None,
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=is_safe,
                critical_issues=critical,
                high_severity=high,
                medium_severity=medium,
                low_severity=low,
                analyzers_used=analyzers.split(","),
                raw_output=raw_output,
                output_file="",  # Repository handles storage
                scan_failed=False,
            )

            # Save scan result via repository
            await self._scan_repo.create(result.model_dump())

            logger.info(
                f"Agent security scan completed for {agent_path}. "
                f"Safe: {is_safe}, Critical: {critical}, High: {high}, Medium: {medium}, Low: {low}"
            )

            return result

        except Exception as e:
            logger.error(f"Agent security scan failed for {agent_path}: {e}")

            # Create error output
            raw_output = {
                "error": str(e),
                "analysis_results": {},
                "scan_failed": True,
            }

            # Return error result
            result = AgentSecurityScanResult(
                agent_path=agent_path,
                agent_url=(
                    agent_card.get("proxy_pass_url")
                    or agent_card.get("proxyPassUrl")
                    or agent_card.get("url")
                ),
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=False,  # Treat scanner failures as unsafe
                critical_issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                analyzers_used=analyzers.split(",") if analyzers else [],
                raw_output=raw_output,
                output_file="",  # Repository handles storage
                scan_failed=True,
                error_message=str(e),
            )

            # Save error result via repository
            await self._scan_repo.create(result.model_dump())

            return result

    def _run_a2a_scanner(
        self,
        agent_card: dict,
        agent_path: str,
        analyzers: str,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> dict:
        """
        Run a2a-scanner command and return raw output.

        This is a synchronous method that runs in a thread pool.
        """
        logger.info(f"Running A2A security scan on: {agent_path}")
        logger.info(f"Using analyzers: {analyzers}")

        # In A2A reverse-proxy mode the card's advertised url is the gateway
        # address, so scan the real backend instead: point the scanned card's url
        # at proxy_pass_url when present. Fall back to url for agents registered
        # before the flag was on (proxy_pass_url absent) -- backwards compatible.
        # Copy the card so we never mutate the caller's dict, without the null
        # fields model_dump() adds (see _drop_none_values, issue #1921).
        scan_card = _drop_none_values(agent_card)
        backend_url = scan_card.get("proxy_pass_url") or scan_card.get("proxyPassUrl")
        if backend_url:
            scan_card["url"] = backend_url

        # Create temporary file for agent card
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(scan_card, tmp_file, indent=2, default=str)
            tmp_file_path = tmp_file.name

        try:
            # Build command
            cmd = [
                "a2a-scanner",
                "scan-card",
                tmp_file_path,
                "--analyzers",
                analyzers,
                "--format",
                "json",
            ]

            # Set environment variable for API key if provided
            env = os.environ.copy()
            if api_key:
                env["AZURE_OPENAI_API_KEY"] = api_key

            # Run scanner with timeout. check=False because the scanner exits 1 after
            # printing a complete result that holds a HIGH finding (see below).
            try:
                result = subprocess.run(  # nosec B603 - args are hardcoded flags and validated config values
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    timeout=timeout,
                )

                # Log raw output for debugging
                logger.debug(f"Raw A2A scanner stdout:\n{result.stdout[:500]}")

                if result.returncode == A2A_SCANNER_FINDINGS_EXIT_CODE:
                    # Exit 1 is how the scanner reports a completed scan with a HIGH
                    # finding (its high_severity_count). Accept it only when stdout
                    # carries that result; an uncaught scanner error also exits 1
                    # but prints no JSON.
                    try:
                        scan_results = _parse_a2a_scanner_stdout(result.stdout)
                    except ValueError:
                        scan_results = None
                    if _has_blocking_findings(scan_results):
                        logger.warning(
                            f"A2A scanner exited {result.returncode} after reporting "
                            f"high-severity findings for {agent_path}; "
                            "treating it as a completed scan"
                        )
                        return _build_a2a_raw_output(scan_results)
                if result.returncode != 0:
                    logger.error(f"A2A scanner command failed with exit code {result.returncode}")
                    logger.error(f"stderr: {result.stderr}")
                    raise RuntimeError(f"Agent security scanner failed: {result.stderr}")

                # Parse JSON output - scanner outputs JSON
                scan_results = _parse_a2a_scanner_stdout(result.stdout)

                # Wrap in expected format with analysis_results
                raw_output = _build_a2a_raw_output(scan_results)

                logger.debug(
                    f"A2A scanner output:\n{json.dumps(raw_output, indent=2, default=str)}"
                )
                return raw_output

            except subprocess.TimeoutExpired as e:
                logger.error(f"A2A scanner command timed out after {timeout} seconds")
                raise RuntimeError(f"Agent security scan timed out after {timeout} seconds") from e

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary agent card file: {e}")

    def _analyze_scan_results(self, raw_output: dict) -> tuple[bool, int, int, int, int]:
        """
        Analyze scan results and extract severity counts.

        Returns:
            Tuple of (is_safe, critical_count, high_count, medium_count, low_count)
        """
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        # Navigate the raw output structure to find findings
        analysis_results = raw_output.get("analysis_results", {})

        for _analyzer_name, analyzer_data in analysis_results.items():
            if isinstance(analyzer_data, dict):
                findings = analyzer_data.get("findings", [])
                for finding in findings:
                    severity = finding.get("severity", "").lower()
                    if severity == "critical":
                        critical_count += 1
                    elif severity == "high":
                        high_count += 1
                    elif severity == "medium":
                        medium_count += 1
                    elif severity == "low":
                        low_count += 1

        # The scanner decides its exit code from its in-memory HIGH count, while the
        # severities printed per finding come from its threat taxonomy and can read
        # MEDIUM for a finding the scanner counted as HIGH. Never report fewer HIGH
        # findings than the scanner counted. It is 0 for a scan that exited 0.
        scanner_high_count = _scanner_high_severity_count(raw_output.get("scan_results"))
        if scanner_high_count is not None:
            high_count = max(high_count, scanner_high_count)

        # Determine if safe: no critical or high severity issues
        is_safe = critical_count == 0 and high_count == 0

        logger.info("Agent security analysis results:")
        logger.info(f"  Critical Issues: {critical_count}")
        logger.info(f"  High Severity: {high_count}")
        logger.info(f"  Medium Severity: {medium_count}")
        logger.info(f"  Low Severity: {low_count}")
        logger.info(f"  Overall Assessment: {'SAFE' if is_safe else 'UNSAFE'}")

        return is_safe, critical_count, high_count, medium_count, low_count

    async def get_scan_summaries(self) -> dict[str, dict]:
        """
        Bulk-load lightweight scan summaries for all agents.

        Returns a path -> summary map (scan_failed + per-severity counts) so the
        agents list endpoint can attach icon data inline instead of forcing the
        frontend to fetch each agent's scan individually (N+1).

        Returns:
            Mapping of agent path to its lightweight scan summary. Empty on error.
        """
        from .scan_summary import build_scan_summary_map

        try:
            # list_latest() collapses scan history to one (newest) doc per path
            # at the data layer, so we don't transfer the full history. The map
            # builder still dedups defensively in case a backend returns dupes.
            scans = await self._scan_repo.list_latest()
            return build_scan_summary_map(scans)
        except Exception:
            logger.exception("Failed to bulk-load agent security scan summaries")
            return {}

    async def get_scan_result(self, agent_path: str) -> dict | None:
        """
        Get the latest scan result for an agent.

        Args:
            agent_path: Agent path (e.g., /code-reviewer)

        Returns:
            Dictionary containing scan results, or None if no scan found
        """
        try:
            # Get latest scan from repository
            scan_result = await self._scan_repo.get_latest(agent_path)

            if scan_result:
                logger.info(f"Loaded agent scan results for {agent_path} from repository")
                # Convert to dict if needed
                if hasattr(scan_result, "model_dump"):
                    return scan_result.model_dump()
                return scan_result

            logger.warning(f"No scan results found for agent: {agent_path}")
            return None

        except Exception as e:
            logger.exception(f"Unexpected error loading agent scan results for {agent_path}")
            return None


# Global singleton instance
agent_scanner_service = AgentScannerService()
