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
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

from ..common.log_redaction import redact_url
from ..core.config import settings
from ..core.endpoint_utils import get_endpoint_url
from ..repositories.factory import get_security_scan_repository
from ..schemas.security import SecurityScanConfig, SecurityScanResult
from .scanner_llm_analyzer import (
    MCP_SCANNER_LITELLM_STANDIN,
    LLMAnalyzerUnavailableError,
    ensure_llm_analyzer_available,
)

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "security_scans"

# Name of the environment variable carrying the scan credential to the child. Private to
# this module and the shim below; deliberately NOT one of mcp-scanner's own variables.
_SCANNER_BEARER_ENV = "MCP_GATEWAY_SCAN_BEARER"  # nosec B105 - env var name, not a secret

# Python source run inside the scanner subprocess, after the litellm stand-in. It fixes
# when mcp-scanner falls back from MCP 2026-07-28 to the initialize handshake.
#
# mcp-scanner 4.8.7 sends server/discover first. When that fails it falls back to
# initialize only for one exact message, "Bad Request: Missing session ID". A server on
# the mcp 1.x Python SDK (FastMCP 2/3, the gateway's own mcpgw) checks the
# MCP-Protocol-Version header first and answers -32600 "Bad Request: Unsupported
# protocol version: 2026-07-28. Supported versions: ...", so the scan failed and,
# with SECURITY_BLOCK_ON_SCAN_FAILURE on, the registry disabled the server.
#
# The 2026-07-28 spec's rule is: only the errors that 2026-07-28 itself defines
# (-32020 HeaderMismatch, -32021 MissingRequiredClientCapability, -32022
# UnsupportedProtocolVersion) identify a modern server. Any other error means a
# handshake-era server, so fall back to initialize. The patch keeps every fallback
# mcp-scanner already makes and adds that one. If the module cannot be imported or the
# method is ever renamed, the patch does nothing rather than break the scan.
# Spec: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#backward-compatibility
# Remove once cisco-ai-mcp-scanner falls back on any non-2026 discover error:
# https://github.com/cisco-ai-defense/mcp-scanner/issues/302
MCP_SCANNER_DISCOVER_FALLBACK_PATCH: str = (
    "try:\n"
    "    import mcpscanner.core.scanner as _mcps\n"
    "except ImportError:\n"
    "    _mcps=None\n"
    "_mcps_original=getattr(getattr(_mcps,'Scanner',None),'_should_fallback_to_initialize',None)\n"
    "if _mcps_original is not None:\n"
    "    def _mcps_should_fallback(error,*,legacy_stdio=False,legacy_http=False):\n"
    "        if _mcps_original(error,legacy_stdio=legacy_stdio,legacy_http=legacy_http):\n"
    "            return True\n"
    "        code=getattr(error,'code',None)\n"
    "        if code is None:\n"
    "            code=getattr(getattr(error,'error',None),'code',None)\n"
    "        return code not in (-32020,-32021,-32022)\n"
    "    _mcps.Scanner._should_fallback_to_initialize=staticmethod(_mcps_should_fallback)\n"
)

# Launched as `python -c <shim>`, so the scan credential never appears in any argv. The
# shim pops it from the environment and re-attaches mcp-scanner's own flag inside the
# child, then hands off to the same entry point the console script uses. Popping it also
# keeps it out of the environment the scanner might pass to anything it spawns.
# MCP_SCANNER_LITELLM_STANDIN runs first: mcp-scanner imports litellm at module
# load even for a YARA-only scan, and the registry no longer installs it (#1826).
# MCP_SCANNER_DISCOVER_FALLBACK_PATCH runs second, before the CLI starts a scan.
_SCANNER_SHIM = (
    MCP_SCANNER_LITELLM_STANDIN
    + MCP_SCANNER_DISCOVER_FALLBACK_PATCH
    + (
        "import os,sys;"
        "from mcpscanner.cli import cli_entry_point;"
        f"_t=os.environ.pop({_SCANNER_BEARER_ENV!r},None);"
        "sys.argv=['mcp-scanner']+sys.argv[1:]+(['--bearer-token',_t] if _t else []);"
        "sys.exit(cli_entry_point())"
    )
)


def _scan_failure_message(
    exc: BaseException,
    headers: str | None,
    credential_expected: bool | None = None,
    appended_mcp_path: bool = False,
) -> str:
    """Build the failure string the UI's scan modal shows.

    This is the one place an operator reads about a failed scan, so it has to carry
    something they can act on. "security scan failed (RuntimeError)" names a Python
    exception and nothing else, which is what sent one investigation through DNS,
    TLS and firewall rules before the real cause turned up in a log line.

    When the scan carried no credential, say so. Against a server that requires
    authentication that IS the whole explanation, and it is the common cause here: a
    discovery identity designated but never connected, or one whose vaulted token is
    gone (OpenBao in dev mode discards tokens on restart).

    ``headers`` is the resolved auth header the caller built, so empty means the scan
    went out unauthenticated. Shared by both failure paths -- the specific exception
    list and the catch-all -- because a message written into only one of them is a
    message half the failures will not carry.
    """
    # The 'llm' analyzer error already says what to change; a credential hint
    # would send the operator the wrong way.
    if isinstance(exc, LLMAnalyzerUnavailableError):
        return str(exc)
    message = f"security scan failed ({type(exc).__name__})"
    # Issue #2032: the hint used to appear on every unauthenticated failure, even
    # for a server configured with no credential at all, where it points the wrong
    # way. None means the caller did not say, which keeps the old behavior.
    if not headers and credential_expected is not False:
        message += (
            " - the scan ran without a credential. A server that requires "
            "authentication will refuse it. Check this server's scan/discovery "
            "credential, then rescan."
        )
    # The scanner reports only "connection cancelled", never the HTTP status, so
    # name the registry's own guess instead. Nothing from the URL goes into this
    # message: it is shown in the UI and a URL can carry secrets.
    if appended_mcp_path:
        message += (
            " - the registry added /mcp to the server URL to reach its MCP endpoint. "
            "If the endpoint is at a different path, set mcp_endpoint, then rescan."
        )
    return message


def scan_credential_expected(
    server_info: dict | None,
) -> bool:
    """Return True when the server is configured to scan with a credential.

    Mirrors the resolver behind the scan's credential (backend_oauth.with_bearer
    and the static header fallback): a backend auth scheme, a designated OAuth
    discovery identity, or an OBO exchange.

    Args:
        server_info: The server record.

    Returns:
        True when one of those is configured.
    """
    info = server_info or {}
    if (info.get("auth_scheme") or "none") != "none":
        return True
    if (info.get("oauth_discovery") or {}).get("enabled"):
        return True
    return info.get("egress_auth_mode") == "obo_exchange"


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
            allow_unsafe_servers=settings.security_allow_unsafe_servers,
            analyzers=settings.security_analyzers,
            scan_timeout_seconds=settings.security_scan_timeout,
            llm_api_key=settings.mcp_scanner_llm_api_key or os.getenv("MCP_SCANNER_LLM_API_KEY"),
            add_security_pending_tag=settings.security_add_pending_tag,
            block_on_scan_failure=settings.security_block_on_scan_failure,
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
        credential_expected: bool | None = None,
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
            credential_expected: Whether the server is configured with a scan
                credential (see scan_credential_expected). Decides whether a failed
                unauthenticated scan blames the credential. None keeps the hint.

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
        registered_url = server_url
        server_url = get_endpoint_url(
            proxy_pass_url=server_url,
            transport_type="streamable-http",
            mcp_endpoint=mcp_endpoint,
        )
        # True when the registry guessed the endpoint by adding /mcp, which is
        # what a scan of the wrong URL usually comes down to.
        appended_mcp_path = not mcp_endpoint and server_url.rstrip("/") != registered_url.rstrip(
            "/"
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
            ensure_llm_analyzer_available(analyzers, "SECURITY_ANALYZERS")

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
            failure = _scan_failure_message(
                exc,
                headers,
                credential_expected=credential_expected,
                appended_mcp_path=appended_mcp_path,
            )
            logger.error(
                f"Security scan failed endpoint={safe_server_url} type={type(exc).__name__} "
                f"authenticated={bool(headers)}",
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
            failure = _scan_failure_message(
                exc,
                headers,
                credential_expected=credential_expected,
                appended_mcp_path=appended_mcp_path,
            )
            logger.error(
                f"Unexpected security scan failure endpoint={safe_server_url} "
                f"type={type(exc).__name__} authenticated={bool(headers)}",
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

        # Build command. The scanner is launched through a tiny in-process shim rather
        # than its console script, so the credential can be handed over in the child's
        # ENVIRONMENT instead of its argv.
        #
        # `mcp-scanner remote` only accepts a credential as `--bearer-token` / `--header`,
        # both of which land in the child's argv and are therefore world-readable via
        # `ps` and /proc/<pid>/cmdline for the life of the scan. That was tolerable when
        # the only thing passed was an operator's static scan token; it is not once the
        # resolver can hand over a *delegated human* OAuth token (a borrowed discovery
        # identity) or the gateway's own app-only token. The shim re-attaches the flag
        # inside the child, where the value is no longer externally visible.
        #
        # The environment is already this subprocess's channel for secrets
        # (MCP_SCANNER_LLM_API_KEY, below), so this follows the existing convention
        # rather than inventing one. /proc/<pid>/environ is restricted to the same
        # uid/root, unlike cmdline, and `ps` never renders it.
        cmd = [
            sys.executable,
            "-c",
            _SCANNER_SHIM,
            "--analyzers",
            analyzers,
            "--raw",  # Use raw format instead of summary
            "remote",  # Subcommand to scan remote MCP server
            "--server-url",
            server_url,
        ]

        env = os.environ.copy()
        # mcp-scanner 4.8.7 runs on MCP SDK 2.x, whose httpx2 client reads only the
        # operating system trust store, and the registry image ships none. Give
        # it certifi, the trust 4.8.6 had through httpx. An operator's own
        # SSL_CERT_FILE is left alone.
        env.setdefault("SSL_CERT_FILE", certifi.where())
        # Parse the resolved auth headers and hand the credential over out-of-band.
        if headers:
            bearer_token = _extract_bearer_token_from_headers(headers)
            if bearer_token:
                env[_SCANNER_BEARER_ENV] = bearer_token
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
