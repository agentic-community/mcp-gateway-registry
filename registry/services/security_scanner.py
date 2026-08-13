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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common.log_redaction import redact_url
from ..core.config import settings
from ..core.endpoint_utils import get_endpoint_url
from ..repositories.factory import get_security_scan_repository
from ..schemas.security import SecurityScanConfig, SecurityScanResult

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "security_scans"


def _extract_bearer_token_from_headers(headers: str) -> str | None:
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
    except json.JSONDecodeError as exc:
        # Never echo the raw headers string or parser detail: either may expose
        # credential-bearing context through a higher-level error log/response.
        logger.error("Failed to parse scanner headers JSON")
        raise ValueError("Invalid headers JSON") from exc


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

    # Extract and parse JSON. mcp-scanner may emit extra output after the JSON
    # array (a trailing summary / log line), so parse only the first JSON value
    # and ignore trailing data. Plain json.loads() raises "Extra data" here.
    json_str = clean_stdout[json_start:]
    tool_results, _ = json.JSONDecoder().raw_decode(json_str)
    return tool_results


def _organize_findings_by_analyzer(tool_results: list) -> dict:
    """
    Organize findings from tool results by analyzer.

    Args:
        tool_results: List of tool results from scanner

    Returns:
        Dictionary organized by analyzer name with findings
    """
    organized_results: dict[str, Any] = {}

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
        server_path: str | None = None,
        analyzers: str | None = None,
        api_key: str | None = None,
        headers: str | None = None,
        timeout: int | None = None,
        mcp_endpoint: str | None = None,
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

        # Fail closed on SSRF before spawning the scanner subprocess: the
        # scanner performs its own outbound requests to server_url, so a
        # proxy_pass_url that resolves to a private/metadata target must be
        # rejected here even though registration already validated it (defence
        # in depth against stored data predating validation, and a live
        # resolve catches a host that has since been pointed at an internal
        # address). Raises UrlValidationError -> surfaced by the caller.
        from ..exceptions import UrlValidationError
        from ..utils.url_guard import proxy_profile_for_entity_target, validate_url

        # Select the SSRF profile by the server's registered identity. The bundled
        # airegistry-tools server targets the internal mcpgw-server host, which the
        # ordinary PROXY_PROFILE deliberately does not allowlist; it is admitted only
        # through its exact-identity built-in profile. Any other server gets
        # PROXY_PROFILE unchanged. entity_type is always "mcp_server" here -- this
        # service scans MCP servers. Both the raw proxy_pass_url and the resolved
        # endpoint URL below are validated against this same profile.
        scan_profile = proxy_profile_for_entity_target(
            entity_type="mcp_server",
            entity_path=server_path,
            registered_target_url=server_url,
        )
        validate_url(server_url, profile=scan_profile)

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

        safe_server_url = redact_url(server_url)
        logger.info(
            f"Starting security scan endpoint={safe_server_url} analyzers={analyzers}",
        )

        try:
            # Re-validate the FINAL resolved URL before spawning the scanner.
            # get_endpoint_url() may return an operator-supplied mcp_endpoint
            # verbatim, which is a different host/URL than the proxy_pass_url
            # validated above. The scanner runs in an external subprocess that
            # the pinned guarded client cannot protect, so this is the only
            # place we can block an mcp_endpoint that points at a
            # private/metadata/loopback address (or that has since been rebound
            # there). resolve=True forces a live DNS lookup so a rebind is
            # caught at fetch time. Fail closed: a validation failure is treated
            # like any other scan failure below (recorded as unsafe, subprocess
            # never spawned). Same profile as the pre-resolution check so the
            # built-in server's exact-identity endpoint (/mcp) is admitted.
            validate_url(server_url, profile=scan_profile)

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
                f"Security scan completed endpoint={safe_server_url} safe={is_safe} critical={critical:d} high={high:d} medium={medium:d} low={low:d}",
            )

            return result

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
            RuntimeError,
            UrlValidationError,
        ) as exc:
            failure = f"security scan failed ({type(exc).__name__})"
            logger.error(
                f"Security scan failed endpoint={safe_server_url} type={type(exc).__name__}",
            )

            raw_output = {
                "error": failure,
                "analysis_results": {},
                "tool_results": [],
                "scan_failed": True,
            }

            result = SecurityScanResult(
                server_url=server_url,
                server_path=server_path or server_url,
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=False,
                critical_issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                analyzers_used=analyzers.split(",") if analyzers else [],
                raw_output=raw_output,
                output_file="",
                scan_failed=True,
                error_message=failure,
            )

            await self._scan_repo.create(result.model_dump())
            return result
        except Exception as exc:
            failure = f"security scan failed ({type(exc).__name__})"
            logger.error(
                f"Unexpected security scan failure endpoint={safe_server_url} type={type(exc).__name__}",
            )
            raw_output = {
                "error": failure,
                "analysis_results": {},
                "tool_results": [],
                "scan_failed": True,
            }
            result = SecurityScanResult(
                server_url=server_url,
                server_path=server_path or server_url,
                scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                is_safe=False,
                critical_issues=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                analyzers_used=analyzers.split(",") if analyzers else [],
                raw_output=raw_output,
                output_file="",
                scan_failed=True,
                error_message=failure,
            )
            await self._scan_repo.create(result.model_dump())
            return result

    def _run_mcp_scanner(
        self,
        server_url: str,
        analyzers: str,
        api_key: str | None = None,
        headers: str | None = None,
        timeout: int | None = None,
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
        logger.info(f"Running security scan endpoint={redact_url(server_url)}")
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
            result = subprocess.run(  # nosec B603 - args are hardcoded flags passed to mcp-scanner tool
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=timeout,
            )

            # Parse scanner output without logging it: stdout can contain
            # credential-bearing response data from the scanned endpoint.
            stdout = result.stdout.strip()
            tool_results = _parse_scanner_json_output(stdout)

            raw_output = {"analysis_results": {}, "tool_results": tool_results}
            raw_output["analysis_results"] = _organize_findings_by_analyzer(tool_results)
            logger.info(f"Security scanner output parsed findings={len(tool_results):d}")
            return raw_output

        except subprocess.TimeoutExpired as exc:
            logger.error(f"Scanner command timed out timeout_seconds={timeout:d}")
            raise RuntimeError("Security scan timed out") from exc
        except subprocess.CalledProcessError as exc:
            logger.error(f"Scanner command failed exit_code={exc.returncode:d}")
            raise RuntimeError("Security scanner command failed") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"Failed to parse security scanner output type={type(exc).__name__}")
            raise RuntimeError("Failed to parse security scanner output") from exc

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

    async def get_scan_summaries(self) -> dict[str, dict]:
        """
        Bulk-load lightweight scan summaries for all servers.

        Returns a path -> summary map (scan_failed + per-severity counts) so the
        servers list endpoint can attach icon data inline instead of forcing the
        frontend to fetch each server's scan individually (N+1).

        Returns:
            Mapping of server path to its lightweight scan summary. Empty on error.
        """
        from .scan_summary import build_scan_summary_map

        try:
            # list_latest() collapses scan history to one (newest) doc per path
            # at the data layer, so we don't transfer the full history. The map
            # builder still dedups defensively in case a backend returns dupes.
            scans = await self._scan_repo.list_latest()
            return build_scan_summary_map(scans)
        except Exception as exc:
            logger.error(
                f"Failed to bulk-load server security scan summaries type={type(exc).__name__}",
            )
            return {}

    async def get_scan_result(self, server_path: str) -> dict | None:
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
                logger.info("Loaded security scan results from repository")
                if hasattr(scan_result, "model_dump"):
                    return scan_result.model_dump()
                return scan_result

            logger.warning("No security scan results found for server")
            return None

        except Exception as exc:
            logger.error(
                f"Unexpected security scan result lookup failure type={type(exc).__name__}",
            )
            return None


# Global singleton instance
security_scanner_service = SecurityScannerService()
