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

from ..core.config import settings
from ..repositories.factory import get_security_scan_repository
from ..schemas.agent_security import AgentSecurityScanConfig, AgentSecurityScanResult

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "agent_security_scans"


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

            # Get agent URL if available
            agent_url = agent_card.get("url")

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
                agent_url=agent_card.get("url"),
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

        # Create temporary file for agent card
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(agent_card, tmp_file, indent=2, default=str)
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

            # Run scanner with timeout
            try:
                result = subprocess.run(  # nosec B603 - args are hardcoded flags and validated config values
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    timeout=timeout,
                )

                # Log raw output for debugging
                logger.debug(f"Raw A2A scanner stdout:\n{result.stdout[:500]}")

                # Parse JSON output - scanner outputs JSON
                stdout = result.stdout.strip()

                # Remove ANSI color codes
                ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
                stdout = ansi_escape.sub("", stdout)

                # Try to parse as JSON directly
                try:
                    scan_results = json.loads(stdout)
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
                    scan_results = json.loads(json_str)

                # Wrap in expected format with analysis_results
                raw_output = {
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

                logger.debug(
                    f"A2A scanner output:\n{json.dumps(raw_output, indent=2, default=str)}"
                )
                return raw_output

            except subprocess.TimeoutExpired as e:
                logger.error(f"A2A scanner command timed out after {timeout} seconds")
                raise RuntimeError(f"Agent security scan timed out after {timeout} seconds") from e
            except subprocess.CalledProcessError as e:
                logger.error(f"A2A scanner command failed with exit code {e.returncode}")
                logger.error(f"stderr: {e.stderr}")
                raise RuntimeError(f"Agent security scanner failed: {e.stderr}") from e

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

        # Determine if safe: no critical or high severity issues
        is_safe = critical_count == 0 and high_count == 0

        logger.info("Agent security analysis results:")
        logger.info(f"  Critical Issues: {critical_count}")
        logger.info(f"  High Severity: {high_count}")
        logger.info(f"  Medium Severity: {medium_count}")
        logger.info(f"  Low Severity: {low_count}")
        logger.info(f"  Overall Assessment: {'SAFE' if is_safe else 'UNSAFE'}")

        return is_safe, critical_count, high_count, medium_count, low_count

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
