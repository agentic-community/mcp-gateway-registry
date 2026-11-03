#!/usr/bin/env python3
"""
MCP Security Scanner CLI Tool

Scans MCP servers for security vulnerabilities using cisco-ai-mcp-scanner.
With --auto-disable it disables a server that fails its scan through the registry API.
"""

import argparse
import json
import logging
import os
import re
import subprocess  # nosec B404
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Constants
DEFAULT_ANALYZERS = "yara"
LLM_API_KEY_ENV = "MCP_SCANNER_LLM_API_KEY"
# Bearer token for authenticating to the scanned MCP server. Preferred over any
# command-line delivery because process arguments are world-readable via `ps` /
# /proc/<pid>/cmdline while the scan runs; the environment of another user's
# process is not.
BEARER_TOKEN_ENV = "MCP_SCAN_BEARER_TOKEN"  # nosec B105 - env var name, not a secret value
# Use absolute path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "security_scans"
# Registry used by --auto-disable to disable a server that fails its scan.
REGISTRY_URL_ENV = "REGISTRY_URL"
DEFAULT_REGISTRY_URL = "http://localhost"
DEFAULT_REGISTRY_TOKEN_FILE = ".token"  # nosec B105 - file path, not a secret value

sys.path.insert(0, str(PROJECT_ROOT / "api"))
from registry_client import RegistryClient  # noqa: E402


class SecurityScanResult(BaseModel):
    """Security scan result model."""

    server_url: str = Field(..., description="URL of the scanned MCP server")
    scan_timestamp: str = Field(..., description="ISO timestamp of the scan")
    is_safe: bool = Field(..., description="Overall safety assessment")
    critical_issues: int = Field(default=0, description="Count of critical severity issues")
    high_severity: int = Field(default=0, description="Count of high severity issues")
    medium_severity: int = Field(default=0, description="Count of medium severity issues")
    low_severity: int = Field(default=0, description="Count of low severity issues")
    raw_output: dict = Field(..., description="Full scanner output")
    output_file: str = Field(..., description="Path to detailed JSON output file")


def _get_llm_api_key(cli_value: str | None = None) -> str:
    """Retrieve LLM API key from CLI argument or environment variable.

    Args:
        cli_value: API key provided via command line

    Returns:
        LLM API key for security scanning

    Raises:
        ValueError: If API key is not found
    """
    if cli_value:
        return cli_value

    env_value = os.getenv(LLM_API_KEY_ENV)
    if env_value:
        return env_value

    raise ValueError(f"LLM API key must be provided via --api-key or {LLM_API_KEY_ENV} env var")


def _get_bearer_token(headers: str | None = None) -> str | None:
    """Resolve the MCP server bearer token, preferring the environment.

    The token is read from ``MCP_SCAN_BEARER_TOKEN`` first so callers can hand it
    off without ever placing it on this process's command line (argv is readable
    by any local user via ``ps`` / ``/proc/<pid>/cmdline``). A ``--headers`` JSON
    string carrying ``X-Authorization: Bearer <token>`` is still honored for
    backward compatibility, but it is a discouraged path for the same reason.

    Args:
        headers: Optional JSON string of request headers (legacy delivery).

    Returns:
        The bearer token, or None if neither source supplies one.

    Raises:
        ValueError: If ``headers`` is provided but is not valid JSON.
    """
    env_token = os.getenv(BEARER_TOKEN_ENV)
    if env_token:
        return env_token

    if not headers:
        return None

    try:
        headers_dict = json.loads(headers)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse headers JSON: {e}")
        raise ValueError("Invalid headers JSON") from e

    auth_header = headers_dict.get("X-Authorization", "")
    if auth_header.startswith("Bearer "):
        logger.warning(
            "Bearer token supplied via --headers; prefer the %s environment "
            "variable so the token is not exposed on the command line.",
            BEARER_TOKEN_ENV,
        )
        return auth_header[len("Bearer ") :]

    logger.warning("Headers provided but no Bearer token found in X-Authorization header")
    return None


def _ensure_output_directory() -> Path:
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _run_mcp_scanner(
    server_url: str,
    analyzers: str = DEFAULT_ANALYZERS,
    api_key: str | None = None,
    bearer_token: str | None = None,
) -> dict:
    """Run mcp-scanner command and return raw output.

    The LLM API key is handed to the external scanner via its environment so it
    never lands on the command line. The bearer token is passed to the external
    ``mcp-scanner`` on argv because that tool only accepts it as an argument and
    exposes no environment-variable path; this is a documented residual that is
    scoped to the short-lived child process (see module docstring / SECURITY
    guidelines). We keep the token off *this* process's argv by resolving it from
    the environment upstream.

    Args:
        server_url: URL of the MCP server to scan
        analyzers: Comma-separated list of analyzers to use
        api_key: OpenAI API key for LLM-based analysis
        bearer_token: Bearer token for authenticating to the scanned server

    Returns:
        Dictionary containing raw scanner output

    Raises:
        subprocess.CalledProcessError: If scanner command fails
    """
    logger.info(f"Running security scan on: {server_url}")
    logger.info(f"Using analyzers: {analyzers}")

    # Build command - global options before subcommand, subcommand options after
    cmd = [
        "mcp-scanner",
        "--analyzers",
        analyzers,
        "--raw",  # Use raw format instead of summary
        "remote",  # Subcommand to scan remote MCP server
        "--server-url",
        server_url,
    ]

    # The external mcp-scanner accepts the bearer token only as a command-line
    # argument (no env-var path exists in its `remote` subcommand). This is the
    # single unavoidable argv exposure and is limited to the scanner child's
    # lifetime; every hop we control keeps the token in the environment instead.
    if bearer_token:
        cmd.extend(["--bearer-token", bearer_token])
        logger.info("Using bearer token authentication")

    # Set environment variable for API key if provided
    env = os.environ.copy()
    if api_key:
        env[LLM_API_KEY_ENV] = api_key

    # Run scanner
    try:
        result = subprocess.run(  # nosec B603 - mcp-scanner tool with validated args
            cmd, capture_output=True, text=True, check=True, env=env
        )

        # Log raw output for debugging
        logger.debug(f"Raw scanner stdout:\n{result.stdout[:500]}")

        # Parse JSON output - scanner outputs JSON array after log messages
        stdout = result.stdout.strip()

        # Remove ANSI color codes that can interfere with JSON parsing
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        stdout = ansi_escape.sub("", stdout)

        # Find the start of JSON array - look for '[\n  {' pattern (array with objects)
        # This is more robust than just finding first '[' or '{'
        json_start = -1

        # Try to find JSON array start
        for i in range(len(stdout) - 1):
            if stdout[i] == "[" and (i == 0 or stdout[i - 1] in "\n\r"):
                # Found '[' at start of line, likely start of JSON
                json_start = i
                break

        # Fallback: find any '[' followed by whitespace and '{'
        if json_start == -1:
            pattern = r"\[\s*\{"
            match = re.search(pattern, stdout)
            if match:
                json_start = match.start()

        if json_start == -1:
            raise ValueError("No JSON array found in scanner output")

        # Extract and parse JSON
        json_str = stdout[json_start:]
        tool_results = json.loads(json_str)

        # Wrap in expected format with analysis_results
        # Convert array of tool results to the expected structure
        raw_output = {"analysis_results": {}, "tool_results": tool_results}

        # Extract findings from tool results and organize by analyzer
        for tool_result in tool_results:
            findings_dict = tool_result.get("findings", {})
            for analyzer_name, analyzer_findings in findings_dict.items():
                if analyzer_name not in raw_output["analysis_results"]:
                    raw_output["analysis_results"][analyzer_name] = {"findings": []}

                # Convert analyzer findings to expected format
                if isinstance(analyzer_findings, dict):
                    finding = {
                        "tool_name": tool_result.get("tool_name"),
                        "severity": analyzer_findings.get("severity", "unknown"),
                        "threat_names": analyzer_findings.get("threat_names", []),
                        "threat_summary": analyzer_findings.get("threat_summary", ""),
                        "is_safe": tool_result.get("is_safe", True),
                    }
                    raw_output["analysis_results"][analyzer_name]["findings"].append(finding)

        logger.debug(f"Scanner output:\n{json.dumps(raw_output, indent=2, default=str)}")
        return raw_output

    except subprocess.CalledProcessError as e:
        logger.error(f"Scanner command failed with exit code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse scanner output as JSON: {e}")
        logger.error(f"Raw stdout: {result.stdout[:1000]}")
        raise


def _analyze_scan_results(raw_output: dict) -> tuple[bool, int, int, int, int]:
    """Analyze scan results and extract severity counts.

    Args:
        raw_output: Raw scanner output dictionary

    Returns:
        Tuple of (is_safe, critical_count, high_count, medium_count, low_count)
    """
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    # Navigate the raw output structure to find findings
    # Structure: raw_output -> analysis_results -> [analyzer_name] -> findings
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


def _save_scan_output(server_url: str, raw_output: dict) -> str:
    """Save detailed scan output to JSON file.

    Saves in two locations:
    1. security_scans/YYYY-MM-DD/scan_<server>_<timestamp>.json (archived)
    2. security_scans/scan_<server>_latest.json (always current)

    Args:
        server_url: URL of the scanned server
        raw_output: Raw scanner output

    Returns:
        Path to saved output file (latest version)
    """
    output_dir = _ensure_output_directory()

    # Generate safe filename from server URL
    safe_url = server_url.replace("https://", "").replace("http://", "").replace("/", "_")

    # Create date-based subdirectory for archival
    timestamp = datetime.now(UTC)
    date_folder = timestamp.strftime("%Y-%m-%d")
    archive_dir = output_dir / date_folder
    archive_dir.mkdir(exist_ok=True)

    # Save timestamped version in date folder (archived)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    archived_filename = f"scan_{safe_url}_{timestamp_str}.json"
    archived_file = archive_dir / archived_filename

    with open(archived_file, "w") as f:
        json.dump(raw_output, f, indent=2, default=str)

    logger.info(f"Archived scan output saved to: {archived_file}")

    # Save latest version in root security_scans folder (always current)
    # Extract server name from URL for cleaner filename
    # e.g., http://localhost/realserverfaketools/mcp -> realserverfaketools_mcp.json
    server_name = safe_url.replace("localhost_", "")
    latest_filename = f"{server_name}.json"
    latest_file = output_dir / latest_filename

    with open(latest_file, "w") as f:
        json.dump(raw_output, f, indent=2, default=str)

    logger.info(f"Latest scan output saved to: {latest_file}")

    return str(latest_file)


def _load_registry_token(
    token_file: str,
) -> str:
    """Read the registry access token from a token file.

    Accepts both layouts the repo's token files use: a top-level
    ``access_token`` and the nested ``tokens.access_token``.

    Args:
        token_file: Path to the JSON token file

    Returns:
        The access token

    Raises:
        ValueError: If the file holds no access token
    """
    with open(token_file) as f:
        data = json.load(f)
    token = data.get("access_token") or (data.get("tokens") or {}).get("access_token")
    if not token:
        raise ValueError(f"No access_token found in {token_file}")
    return token


def _disable_unsafe_server(
    server_path: str,
    registry_url: str,
    token_file: str,
) -> bool:
    """Disable a server that failed its security scan, via the registry API.

    Reads the current state first: the toggle endpoint flips state, so calling
    it on an already-disabled server would enable it.

    Args:
        server_path: Path of the server to disable (e.g., /mcpgw)
        registry_url: Base URL of the registry
        token_file: Path to a token file for a registry admin

    Returns:
        True if the server is disabled afterwards, False otherwise
    """
    logger.info(f"Disabling unsafe server: {server_path}")

    try:
        client = RegistryClient(registry_url=registry_url, token=_load_registry_token(token_file))
        if not client.get_server(server_path).is_enabled:
            logger.info(f"Server {server_path} is already disabled")
            return True
        result = client.toggle_service(server_path)
        if result.is_enabled:
            logger.error(f"Registry reports {server_path} still enabled after toggle")
            return False
        logger.info(f"Server {server_path} disabled successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to disable server {server_path}: {e}")
        return False


def _extract_server_path_from_url(server_url: str) -> str | None:
    """Extract server path from URL.

    Args:
        server_url: Full server URL (e.g., http://localhost/mcpgw/mcp)

    Returns:
        Server path (e.g., /mcpgw) or None if cannot be extracted
    """
    try:
        # Parse URL to extract path component
        # Expected format: http://localhost/server-path/mcp
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        path_parts = [p for p in parsed.path.split("/") if p and p != "mcp"]

        if path_parts:
            server_path = f"/{path_parts[0]}"
            logger.debug(f"Extracted server path '{server_path}' from URL '{server_url}'")
            return server_path
        else:
            logger.warning(f"Could not extract server path from URL: {server_url}")
            return None

    except Exception as e:
        logger.error(f"Error parsing server URL {server_url}: {e}")
        return None


def scan_server(
    server_url: str,
    analyzers: str = DEFAULT_ANALYZERS,
    api_key: str | None = None,
    output_json: bool = False,
    auto_disable: bool = False,
    bearer_token: str | None = None,
    registry_url: str = DEFAULT_REGISTRY_URL,
    registry_token_file: str = DEFAULT_REGISTRY_TOKEN_FILE,
) -> SecurityScanResult:
    """Scan an MCP server for security vulnerabilities.

    Args:
        server_url: URL of the MCP server to scan
        analyzers: Comma-separated list of analyzers to use
        api_key: OpenAI API key for LLM-based analysis
        output_json: If True, output raw mcp-scanner JSON directly
        auto_disable: If True, automatically disable servers that fail security scan
        bearer_token: Bearer token for authenticating to the scanned server
        registry_url: Registry base URL used by auto_disable
        registry_token_file: Registry admin token file used by auto_disable

    Returns:
        SecurityScanResult containing scan results
    """
    # Run scanner
    try:
        raw_output = _run_mcp_scanner(server_url, analyzers, api_key, bearer_token)
    except subprocess.CalledProcessError as e:
        # Scanner failed - create error output and save it
        logger.error(f"Scanner failed with exit code {e.returncode}")
        raw_output = {
            "error": str(e),
            "stderr": e.stderr if hasattr(e, "stderr") else "",
            "analysis_results": {},
            "tool_results": [],
            "scan_failed": True,
        }
        # Save the error output
        output_file = _save_scan_output(server_url, raw_output)

        # Create error result
        result = SecurityScanResult(
            server_url=server_url,
            scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            is_safe=False,  # Treat scanner failures as unsafe
            critical_issues=0,
            high_severity=0,
            medium_severity=0,
            low_severity=0,
            raw_output=raw_output,
            output_file=output_file,
        )

        # Output result
        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            print("\n" + "=" * 60)
            print("SECURITY SCAN FAILED")
            print("=" * 60)
            print(f"Server URL: {result.server_url}")
            print(f"Scan Time: {result.scan_timestamp}")
            print("\nError: Scanner failed to complete scan")
            print(f"Details: {e}")
            print("\nMarking server as UNSAFE due to scanner failure")
            print(f"\nDetailed output saved to: {result.output_file}")
            print("=" * 60 + "\n")

        return result

    # Analyze results
    is_safe, critical, high, medium, low = _analyze_scan_results(raw_output)

    # Save detailed output
    output_file = _save_scan_output(server_url, raw_output)

    # Auto-disable server if unsafe
    disabled = False
    if auto_disable and not is_safe:
        logger.warning("Server marked as UNSAFE - attempting to disable")
        server_path = _extract_server_path_from_url(server_url)
        if server_path:
            disabled = _disable_unsafe_server(server_path, registry_url, registry_token_file)
            if disabled:
                logger.info(f"✓ Server {server_path} has been disabled for security reasons")
            else:
                logger.error(f"✗ Failed to disable server {server_path}")
        else:
            logger.error("✗ Could not extract server path from URL - manual intervention required")

    # Create result object
    result = SecurityScanResult(
        server_url=server_url,
        scan_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        is_safe=is_safe,
        critical_issues=critical,
        high_severity=high,
        medium_severity=medium,
        low_severity=low,
        raw_output=raw_output,
        output_file=output_file,
    )

    # Output result
    if output_json:
        # Output raw mcp-scanner format directly (same as --raw)
        print(json.dumps(raw_output, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("SECURITY SCAN SUMMARY")
        print("=" * 60)
        print(f"Server URL: {result.server_url}")
        print(f"Scan Time: {result.scan_timestamp}")
        print("\nEXECUTIVE SUMMARY OF ISSUES:")
        print(f"  Critical Issues: {result.critical_issues}")
        print(f"  High Severity: {result.high_severity}")
        print(f"  Medium Severity: {result.medium_severity}")
        print(f"  Low Severity: {result.low_severity}")
        print(f"\nOverall Assessment: {'SAFE ✓' if result.is_safe else 'UNSAFE ✗'}")

        # Show auto-disable status if applicable
        if auto_disable and not result.is_safe:
            server_path = _extract_server_path_from_url(server_url)
            if disabled:
                print(
                    f"\n⚠️  ACTION TAKEN: Server {server_path} has been DISABLED due to security issues"
                )
            else:
                print("\n⚠️  WARNING: Could not auto-disable server - manual intervention required")

        print(f"\nDetailed output saved to: {result.output_file}")
        print("=" * 60 + "\n")

    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Scan MCP servers for security vulnerabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic scan with YARA analyzer (default)
    uv run cli/mcp_security_scanner.py --server-url https://mcp.deepwki.com/mcp

    # Scan with both YARA and LLM analyzers
    export MCP_SCANNER_LLM_API_KEY=sk-...
    uv run cli/mcp_security_scanner.py --server-url https://example.com/mcp --analyzers yara,llm

    # Scan with LLM only, passing API key directly
    uv run cli/mcp_security_scanner.py --server-url https://example.com/mcp --analyzers llm --api-key sk-...

    # Authenticate to the target server (token via environment, not argv)
    export MCP_SCAN_BEARER_TOKEN=eyJhbGci...
    uv run cli/mcp_security_scanner.py --server-url https://example.com/mcp

    # Output as JSON
    uv run cli/mcp_security_scanner.py --server-url https://example.com/mcp --json
""",
    )

    parser.add_argument("--server-url", required=True, help="URL of the MCP server to scan")

    parser.add_argument(
        "--analyzers",
        default=DEFAULT_ANALYZERS,
        help=f"Comma-separated list of analyzers to use (default: {DEFAULT_ANALYZERS})",
    )

    parser.add_argument(
        "--api-key",
        help=f"LLM API key for security scanning (can also use {LLM_API_KEY_ENV} env var)",
    )

    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--auto-disable",
        action="store_true",
        help=(
            "Disable a server that fails its scan (is_safe: false) through the registry "
            "API. Needs a registry admin token (see --registry-url and --registry-token-file)"
        ),
    )

    parser.add_argument(
        "--registry-url",
        default=os.environ.get(REGISTRY_URL_ENV, DEFAULT_REGISTRY_URL),
        help=f"Registry base URL for --auto-disable (default: ${REGISTRY_URL_ENV} or "
        f"{DEFAULT_REGISTRY_URL})",
    )

    parser.add_argument(
        "--registry-token-file",
        default=DEFAULT_REGISTRY_TOKEN_FILE,
        help=f"Registry admin token file for --auto-disable (default: {DEFAULT_REGISTRY_TOKEN_FILE})",
    )

    parser.add_argument(
        "--headers",
        help=(
            "DEPRECATED and insecure: JSON string of headers carrying an "
            f"X-Authorization Bearer token. Prefer the {BEARER_TOKEN_ENV} "
            "environment variable so the token is never placed on the command line."
        ),
    )

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Get API key if needed for LLM analyzer
        api_key = None
        if "llm" in args.analyzers.lower():
            api_key = _get_llm_api_key(args.api_key)

        # Resolve the bearer token, preferring the environment over --headers so
        # it does not appear on this process's argv.
        bearer_token = _get_bearer_token(args.headers)

        # Run scan
        result = scan_server(
            server_url=args.server_url,
            analyzers=args.analyzers,
            api_key=api_key,
            output_json=args.json,
            auto_disable=args.auto_disable,
            bearer_token=bearer_token,
            registry_url=args.registry_url,
            registry_token_file=args.registry_token_file,
        )

        # Exit with non-zero code if unsafe
        sys.exit(0 if result.is_safe else 1)

    except Exception as e:
        logger.exception(f"Security scan failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
