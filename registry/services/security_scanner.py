"""
Security Scanner Service

This service provides security scanning functionality for MCP servers during registration.
It wraps the CLI security scanner and makes it available to API endpoints with proper
configuration and error handling.
"""

import asyncio
import json
import logging
import os
import re
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.endpoint_utils import get_endpoint_url
from ..schemas.security import SecurityScanResult, SecurityScanConfig
from ..repositories.factory import get_security_scan_repository


logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "security_scans"


def _extract_bearer_token_from_headers(headers: str) -> Optional[str]:
    """
    Extract bearer token from headers JSON string.

    Args:
        headers: JSON string containing headers

    Returns:
        Bearer token if found, None otherwise

    Raises:
        ValueError: If headers JSON is invalid
    """
    logger.info("Adding custom headers for scanning")
    try:
        headers_dict = json.loads(headers)
        # Check for X-Authorization header with Bearer token
        auth_header = headers_dict.get("X-Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header.replace("Bearer ", "")
            logger.info("Using bearer token authentication")
            return bearer_token
        else:
            logger.warning("Headers provided but no Bearer token found in X-Authorization header")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse headers JSON: {e}")
        raise ValueError(f"Invalid headers JSON: {headers}") from e


def _parse_scanner_json_output(stdout: str) -> list:
    """
    Parse JSON output from scanner stdout.

    Args:
        stdout: Raw stdout from scanner command

    Returns:
        Parsed JSON array of tool results

    Raises:
        ValueError: If no valid JSON array found in output
        json.JSONDecodeError: If JSON parsing fails
    """
    # Remove ANSI color codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean_stdout = ansi_escape.sub("", stdout)

    # Find the start of JSON array
    json_start = -1

    # Try to find JSON array start
    for i in range(len(clean_stdout) - 1):
        if clean_stdout[i] == "[" and (i == 0 or clean_stdout[i - 1] in "\n\r"):
            json_start = i
            break

    # Fallback: find any '[' followed by whitespace and '{'
    if json_start == -1:
        pattern = r"\[\s*\{"
        match = re.search(pattern, clean_stdout)
        if match:
            json_start = match.start()

    if json_start == -1:
        raise ValueError("No JSON array found in scanner output")

    # Extract and parse JSON
    json_str = clean_stdout[json_start:]
    tool_results = json.loads(json_str)
    return tool_results


def _organize_findings_by_analyzer(tool_results: list) -> dict:
    """
    Organize findings from tool results by analyzer.

    Args:
        tool_results: List of tool results from scanner

    Returns:
        Dictionary organized by analyzer name with findings
    """
    organized_results = {}

    for tool_result in tool_results:
        findings_dict = tool_result.get("findings", {})
        for analyzer_name, analyzer_findings in findings_dict.items():
            if analyzer_name not in organized_results:
                organized_results[analyzer_name] = {"findings": []}

            # Convert analyzer findings to expected format
            if isinstance(analyzer_findings, dict):
                finding = {
                    "tool_name": tool_result.get("tool_name"),
                    "severity": analyzer_findings.get("severity", "unknown"),
                    "threat_names": analyzer_findings.get("threat_names", []),
                    "threat_summary": analyzer_findings.get("threat_summary", ""),
                    "is_safe": tool_result.get("is_safe", True),
                }
                organized_results[analyzer_name]["findings"].append(finding)

    return organized_results


class SecurityScannerService:
    """Service for scanning MCP servers for security vulnerabilities."""

    def __init__(self) -> None:
        """Initialize the security scanner service."""
        self._ensure_output_directory()
        self._scan_repo = get_security_scan_repository()

    def _ensure_output_directory(self) -> Path:
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    def get_scan_config(self) -> SecurityScanConfig:
        """Get security scan configuration from settings."""
        return SecurityScanConfig(
            enabled=settings.security_scan_enabled,
            scan_on_registration=settings.security_scan_on_registration,
            block_unsafe_servers=settings.security_block_unsafe_servers,
            analyzers=settings.security_analyzers,
            scan_timeout_seconds=settings.security_scan_timeout,
            llm_api_key=settings.mcp_scanner_llm_api_key or os.getenv("MCP_SCANNER_LLM_API_KEY"),
            add_security_pending_tag=settings.security_add_pending_tag,
        )

    async def scan_server(
        self,
        server_url: str,
        server_path: Optional[str] = None,
        analyzers: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        timeout: Optional[int] = None,
        mcp_endpoint: Optional[str] = None,
    ) -> SecurityScanResult:
        """
        Scan an MCP server for security vulnerabilities.

        Args:
            server_url: URL of the MCP server to scan (proxy_pass_url)
            server_path: Optional path identifier for the server
            analyzers: Comma-separated list of analyzers to use (overrides config)
            api_key: OpenAI API key for LLM-based analysis (overrides config)
            headers: JSON string of headers to include in requests
            timeout: Scan timeout in seconds (overrides config)
            mcp_endpoint: Optional explicit MCP endpoint URL. If set, used directly
                instead of appending /mcp to server_url.

        Returns:
            SecurityScanResult containing scan results

        Raises:
            subprocess.TimeoutExpired: If scan times out
            subprocess.CalledProcessError: If scanner command fails
            ValueError: If invalid input provided
            RuntimeError: If scan fails for other reasons
        """
        config = self.get_scan_config()

        # Use config values if not provided
        if analyzers is None:
            analyzers = config.analyzers
        if api_key is None:
            api_key = config.llm_api_key
        if timeout is None:
            timeout = config.scan_timeout_seconds

        # Resolve endpoint URL using centralized utility
        # Priority: explicit mcp_endpoint > URL detection > append /mcp
        server_url = get_endpoint_url(
            proxy_pass_url=server_url,
            transport_type="streamable-http",
            mcp_endpoint=mcp_endpoint,
        )

        logger.info(f"Starting security scan for {server_url} with analyzers: {analyzers}")

        try:
            # Run the scan in a thread pool to avoid blocking
            raw_output = await asyncio.to_thread(
                self._run_mcp_scanner,
                server_url=server_url,
                analyzers=analyzers,
                api_key=api_key,
                headers=headers,
                timeout=timeout,
            )

            # Analyze results
            is_safe, critical, high, medium, low = self._analyze_scan_results(raw_output)

            # Create result object
            result = SecurityScanResult(
                server_url=server_url,
                server_path=server_path
                or server_url,  # Use server_path if provided, fallback to URL
                scan_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
                f"Security scan completed for {server_url}. "
                f"Safe: {is_safe}, Critical: {critical}, High: {high}, Medium: {medium}, Low: {low}"
            )

            return result

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.error(f"Security scan failed for {server_url}: {e}")

            # Create error output
            raw_output = {
                "error": str(e),
                "analysis_results": {},
                "tool_results": [],
                "scan_failed": True,
            }

            # Return error result
            result = SecurityScanResult(
                server_url=server_url,
                scan_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            await self._scan_repo.create(result)

            return result
        except Exception as e:
            logger.exception(f"Unexpected error during security scan for {server_url}")

            # Create error output
            raw_output = {
                "error": str(e),
                "analysis_results": {},
                "tool_results": [],
                "scan_failed": True,
            }

            # Return error result
            result = SecurityScanResult(
                server_url=server_url,
                scan_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            await self._scan_repo.create(result)

            return result

    def _run_mcp_scanner(
        self,
        server_url: str,
        analyzers: str,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Run mcp-scanner command and return raw output.

        This is a synchronous method that runs in a thread pool.

        Args:
            server_url: URL of the MCP server to scan
            analyzers: Comma-separated list of analyzers to use
            api_key: OpenAI API key for LLM-based analysis
            headers: JSON string of headers to include in requests
            timeout: Scan timeout in seconds

        Returns:
            Dictionary containing analysis results and tool results

        Raises:
            subprocess.TimeoutExpired: If scan times out
            subprocess.CalledProcessError: If scanner command fails
            ValueError: If headers are invalid or output cannot be parsed
            RuntimeError: If scan fails for other reasons
        """
        logger.info(f"Running security scan on: {server_url}")
        logger.info(f"Using analyzers: {analyzers}")

        # Build command
        cmd = [
            "mcp-scanner",
            "--analyzers",
            analyzers,
            "--raw",  # Use raw format instead of summary
            "remote",  # Subcommand to scan remote MCP server
            "--server-url",
            server_url,
        ]

        # Add headers if provided - parse JSON and extract bearer token
        if headers:
            bearer_token = _extract_bearer_token_from_headers(headers)
            if bearer_token:
                cmd.extend(["--bearer-token", bearer_token])

        # Set environment variable for API key if provided
        env = os.environ.copy()
        if api_key:
            env["MCP_SCANNER_LLM_API_KEY"] = api_key

        # Run scanner with timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=timeout,
            )

            # Log raw output for debugging
            logger.debug(f"Raw scanner stdout:\n{result.stdout[:500]}")

            # Parse JSON output - scanner outputs JSON array after log messages
            stdout = result.stdout.strip()
            tool_results = _parse_scanner_json_output(stdout)

            # Wrap in expected format with analysis_results
            raw_output = {"analysis_results": {}, "tool_results": tool_results}

            # Extract findings from tool results and organize by analyzer
            raw_output["analysis_results"] = _organize_findings_by_analyzer(tool_results)

            logger.debug(f"Scanner output:\n{json.dumps(raw_output, indent=2, default=str)}")
            return raw_output

        except subprocess.TimeoutExpired as e:
            logger.error(f"Scanner command timed out after {timeout} seconds")
            raise RuntimeError(f"Security scan timed out after {timeout} seconds") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"Scanner command failed with exit code {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Security scanner failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scanner output as JSON: {e}")
            logger.error(f"Raw stdout: {result.stdout[:1000]}")
            raise RuntimeError("Failed to parse security scanner output") from e

    def _analyze_scan_results(self, raw_output: dict) -> tuple[bool, int, int, int, int]:
        """
        Analyze scan results and extract severity counts.

        Args:
            raw_output: Dictionary containing scanner results

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

        logger.info("Security analysis results:")
        logger.info(f"  Critical Issues: {critical_count}")
        logger.info(f"  High Severity: {high_count}")
        logger.info(f"  Medium Severity: {medium_count}")
        logger.info(f"  Low Severity: {low_count}")
        logger.info(f"  Overall Assessment: {'SAFE' if is_safe else 'UNSAFE'}")

        return is_safe, critical_count, high_count, medium_count, low_count

    async def get_scan_result(self, server_path: str) -> Optional[dict]:
        """
        Get the latest scan result for a server.

        Args:
            server_path: Server path (e.g., /cloudflare-docs)

        Returns:
            Dictionary containing scan results, or None if no scan found
        """
        try:
            # Get latest scan from repository
            scan_result = await self._scan_repo.get_latest(server_path)

            if scan_result:
                logger.info(f"Loaded security scan results for {server_path} from repository")
                # Convert to dict if needed
                if hasattr(scan_result, "model_dump"):
                    return scan_result.model_dump()
                return scan_result

            logger.warning(f"No security scan results found for server {server_path}")
            return None

        except Exception as e:
            logger.exception(f"Unexpected error loading security scan results for {server_path}")
            return None


# Global singleton instance
security_scanner_service = SecurityScannerService()
