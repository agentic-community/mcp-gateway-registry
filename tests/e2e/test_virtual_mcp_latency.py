"""
MCP Virtual Server Latency Benchmarks.

Measures and compares latency between the virtual MCP server (routed through
nginx/Lua) and direct backend MCP servers. Reports min, max, mean, median,
p95, and p99 for each method, plus routing overhead.

Usage:
    python3 tests/e2e/test_virtual_mcp_latency.py
"""

import json
import logging
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIRTUAL_SERVER_URL = "http://localhost/virtual/e2e-multi-backend/mcp"
DIRECT_CURRENTTIME_URL = "http://localhost:8000/mcp"
DIRECT_FAKETOOLS_URL = "http://localhost:8002/mcp"

WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 20

REQUEST_TIMEOUT_SECONDS = 30


def _refresh_token() -> str:
    """Refresh the admin-bot M2M token and return the access token."""
    script_path = os.path.join(PROJECT_ROOT, "scripts", "refresh_m2m_token.sh")
    token_path = os.path.join(PROJECT_ROOT, ".oauth-tokens", "admin-bot-token.json")

    logger.info("Refreshing admin-bot token...")
    result = subprocess.run(
        ["bash", script_path, "admin-bot"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        logger.error("Token refresh failed: %s", result.stderr)
        raise RuntimeError(f"Token refresh failed: {result.stderr}")

    with open(token_path, "r") as f:
        token_data = json.load(f)

    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("No access_token found in token file")

    logger.info("Token refreshed successfully")
    return access_token


def _parse_sse_response(
    raw_body: str,
) -> Optional[Dict[str, Any]]:
    """Parse an SSE or plain JSON response body.

    SSE responses have lines like:
        event: message
        data: {"jsonrpc":"2.0", ...}

    Plain JSON responses are just the JSON object directly.

    Returns the parsed JSON-RPC result dict, or None on failure.
    """
    raw_body = raw_body.strip()

    # Try plain JSON first
    if raw_body.startswith("{"):
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            pass

    # Parse SSE: find last data: line (some servers send multiple events)
    last_data_line = None
    for line in raw_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            last_data_line = stripped[len("data:") :].strip()

    if last_data_line:
        try:
            return json.loads(last_data_line)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE data line: %s", last_data_line)

    logger.warning("Could not parse response body:\n%s", raw_body[:500])
    return None


def _send_mcp_request(
    url: str,
    payload: Dict[str, Any],
    token: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Send an MCP JSON-RPC request and return (parsed_response, session_id).

    Args:
        url: The MCP endpoint URL.
        payload: JSON-RPC request body.
        token: Optional Bearer token for authorization.
        session_id: Optional MCP session ID header.

    Returns:
        Tuple of (parsed JSON-RPC response dict, session ID from response).
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_session = resp.headers.get("Mcp-Session-Id", session_id)
            parsed = _parse_sse_response(resp_body)
            return parsed, resp_session
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        logger.error("HTTP %d from %s: %s", e.code, url, error_body[:300])
        return None, session_id
    except urllib.error.URLError as e:
        logger.error("URL error for %s: %s", url, e.reason)
        return None, session_id
    except Exception as e:
        logger.error("Request to %s failed: %s", url, e)
        return None, session_id


def _initialize_session(
    url: str,
    token: Optional[str] = None,
) -> Optional[str]:
    """Send an MCP initialize request and return the session ID."""
    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "latency-benchmark",
                "version": "1.0.0",
            },
        },
    }

    resp, session_id = _send_mcp_request(url, payload, token=token)
    if resp is None:
        logger.error("Failed to initialize session at %s", url)
        return None

    if "error" in resp:
        logger.error("Initialize error at %s: %s", url, resp["error"])
        return None

    logger.info("Session initialized at %s, session_id=%s", url, session_id)

    # Send initialized notification
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    _send_mcp_request(url, notification, token=token, session_id=session_id)

    return session_id


def _timed_request(
    url: str,
    payload: Dict[str, Any],
    token: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Send an MCP request and return (elapsed_ms, parsed_response)."""
    start = time.perf_counter()
    resp, _ = _send_mcp_request(url, payload, token=token, session_id=session_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, resp


def _run_benchmark(
    label: str,
    url: str,
    payload: Dict[str, Any],
    token: Optional[str] = None,
    session_id: Optional[str] = None,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = MEASURED_ITERATIONS,
) -> Optional[Dict[str, float]]:
    """Run a benchmark: warmup + measured iterations.

    Returns dict with min, max, mean, median, p95, p99 in ms,
    or None if all iterations failed.
    """
    logger.info("Benchmarking [%s] ...", label)

    # Warmup
    for i in range(warmup):
        elapsed, resp = _timed_request(url, payload, token=token, session_id=session_id)
        if resp is None:
            logger.warning("  warmup %d/%d: FAILED", i + 1, warmup)
        else:
            logger.debug("  warmup %d/%d: %.1f ms", i + 1, warmup, elapsed)

    # Measured iterations
    latencies: List[float] = []
    failures = 0
    for i in range(iterations):
        elapsed, resp = _timed_request(url, payload, token=token, session_id=session_id)
        if resp is None or "error" in (resp or {}):
            failures += 1
            logger.warning(
                "  iteration %d/%d: FAILED (resp=%s)",
                i + 1,
                iterations,
                resp,
            )
        else:
            latencies.append(elapsed)
            logger.debug("  iteration %d/%d: %.1f ms", i + 1, iterations, elapsed)

    if not latencies:
        logger.error("  All %d iterations failed for [%s]", iterations, label)
        return None

    if failures > 0:
        logger.warning("  %d/%d iterations failed for [%s]", failures, iterations, label)

    latencies.sort()
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p99_idx = max(0, int(len(latencies) * 0.99) - 1)

    result = {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": latencies[p95_idx],
        "p99": latencies[p99_idx],
        "count": len(latencies),
        "failures": failures,
    }

    logger.info(
        "  [%s] mean=%.1f ms, median=%.1f ms, p95=%.1f ms (%d ok, %d fail)",
        label,
        result["mean"],
        result["median"],
        result["p95"],
        len(latencies),
        failures,
    )
    return result


def _compute_overhead(
    virtual_stats: Dict[str, float],
    direct_stats: Dict[str, float],
) -> Dict[str, float]:
    """Compute overhead = virtual - direct for each stat."""
    return {
        key: virtual_stats[key] - direct_stats[key]
        for key in ("min", "max", "mean", "median", "p95", "p99")
    }


def _print_table(
    rows: List[Tuple[str, Optional[Dict[str, float]]]],
) -> None:
    """Print a formatted results table."""
    header = (
        f"{'Method':<34} | {'Min(ms)':>8} | {'Max(ms)':>8} | "
        f"{'Mean(ms)':>9} | {'Median(ms)':>11} | {'P95(ms)':>8} | {'P99(ms)':>8}"
    )
    separator = (
        f"{'-' * 34}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 11}-+-{'-' * 8}-+-{'-' * 8}"
    )

    print()
    print("=" * len(header))
    print("MCP LATENCY BENCHMARK RESULTS")
    print(f"  Warmup: {WARMUP_ITERATIONS}, Measured: {MEASURED_ITERATIONS} iterations")
    print("=" * len(header))
    print()
    print(header)
    print(separator)

    for label, stats in rows:
        if stats is None:
            print(f"{label:<34} | {'FAILED':>8} | {'':>8} | {'':>9} | {'':>11} | {'':>8} | {'':>8}")
        else:
            print(
                f"{label:<34} | {stats['min']:>8.1f} | {stats['max']:>8.1f} | "
                f"{stats['mean']:>9.1f} | {stats['median']:>11.1f} | "
                f"{stats['p95']:>8.1f} | {stats['p99']:>8.1f}"
            )

    print(separator)
    print()


def _build_method_configs() -> List[Dict[str, Any]]:
    """Build the list of method benchmark configurations.

    Returns a list of dicts, each with:
        name: display name for the method
        virtual_payload: JSON-RPC payload for virtual server
        direct_url: URL of the direct backend (or None to skip)
        direct_payload: JSON-RPC payload for direct backend (or None)
    """
    return [
        {
            "name": "ping",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            },
            "direct_url": DIRECT_CURRENTTIME_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            },
        },
        {
            "name": "tools/list",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
            "direct_url": DIRECT_CURRENTTIME_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
        },
        {
            "name": "tools/call get_time",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_time",
                    "arguments": {"timezone": "UTC"},
                },
            },
            "direct_url": DIRECT_CURRENTTIME_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "current_time_by_timezone",
                    "arguments": {"timezone": "UTC"},
                },
            },
        },
        {
            "name": "tools/call quantum_flux",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "quantum_flux_analyzer",
                    "arguments": {"energy_level": 5},
                },
            },
            "direct_url": DIRECT_FAKETOOLS_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "quantum_flux_analyzer",
                    "arguments": {"energy_level": 5},
                },
            },
        },
        {
            "name": "resources/list",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/list",
            },
            "direct_url": DIRECT_CURRENTTIME_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/list",
            },
        },
        {
            "name": "prompts/list",
            "virtual_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "prompts/list",
            },
            "direct_url": DIRECT_CURRENTTIME_URL,
            "direct_payload": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "prompts/list",
            },
        },
    ]


def main() -> None:
    """Run the MCP latency benchmarks."""
    start_time = time.time()

    # Refresh token
    token = _refresh_token()

    # Initialize sessions
    logger.info("Initializing virtual server session...")
    virtual_session = _initialize_session(VIRTUAL_SERVER_URL, token=token)
    if not virtual_session:
        logger.error("Failed to initialize virtual server session. Aborting.")
        sys.exit(1)

    logger.info("Initializing direct currenttime session...")
    direct_ct_session = _initialize_session(DIRECT_CURRENTTIME_URL)
    if not direct_ct_session:
        logger.error("Failed to initialize direct currenttime session. Aborting.")
        sys.exit(1)

    logger.info("Initializing direct realserverfaketools session...")
    direct_ft_session = _initialize_session(DIRECT_FAKETOOLS_URL)
    if not direct_ft_session:
        logger.error("Failed to initialize direct faketools session. Aborting.")
        sys.exit(1)

    # Map direct URLs to their sessions
    direct_sessions = {
        DIRECT_CURRENTTIME_URL: direct_ct_session,
        DIRECT_FAKETOOLS_URL: direct_ft_session,
    }

    method_configs = _build_method_configs()
    table_rows: List[Tuple[str, Optional[Dict[str, float]]]] = []

    for config in method_configs:
        method_name = config["name"]

        # Benchmark virtual server
        virtual_stats = _run_benchmark(
            label=f"virtual {method_name}",
            url=VIRTUAL_SERVER_URL,
            payload=config["virtual_payload"],
            token=token,
            session_id=virtual_session,
        )
        table_rows.append((f"[virtual] {method_name}", virtual_stats))

        # Benchmark direct backend
        direct_url = config.get("direct_url")
        direct_payload = config.get("direct_payload")
        direct_stats = None

        if direct_url and direct_payload:
            direct_session = direct_sessions.get(direct_url)
            direct_stats = _run_benchmark(
                label=f"direct  {method_name}",
                url=direct_url,
                payload=direct_payload,
                session_id=direct_session,
            )
            table_rows.append((f"[direct]  {method_name}", direct_stats))

            # Compute overhead
            if virtual_stats and direct_stats:
                overhead = _compute_overhead(virtual_stats, direct_stats)
                table_rows.append((f"  overhead", overhead))
            else:
                table_rows.append(("  overhead", None))
        else:
            table_rows.append((f"[direct]  {method_name}", None))
            table_rows.append(("  overhead", None))

        # Blank separator row placeholder - we add a visual break in printing
        table_rows.append(("", None))

    # Filter out blank separator entries for the table but print blank lines
    _print_results_table(table_rows)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        logger.info("Benchmark completed in %d minutes and %.1f seconds", minutes, seconds)
    else:
        logger.info("Benchmark completed in %.1f seconds", seconds)


def _print_results_table(
    rows: List[Tuple[str, Optional[Dict[str, float]]]],
) -> None:
    """Print formatted results table with visual grouping."""
    header = (
        f"{'Method':<34} | {'Min(ms)':>8} | {'Max(ms)':>8} | "
        f"{'Mean(ms)':>9} | {'Median(ms)':>11} | {'P95(ms)':>8} | {'P99(ms)':>8}"
    )
    separator = (
        f"{'-' * 34}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 11}-+-{'-' * 8}-+-{'-' * 8}"
    )

    print()
    print("=" * len(header))
    print("MCP LATENCY BENCHMARK RESULTS")
    print(f"  Warmup: {WARMUP_ITERATIONS}, Measured: {MEASURED_ITERATIONS} iterations")
    print("=" * len(header))
    print()
    print(header)
    print(separator)

    for label, stats in rows:
        if label == "":
            # Visual separator between method groups
            print(separator)
            continue

        if stats is None:
            print(f"{label:<34} | {'FAILED':>8} | {'':>8} | {'':>9} | {'':>11} | {'':>8} | {'':>8}")
        else:
            print(
                f"{label:<34} | {stats['min']:>8.1f} | {stats['max']:>8.1f} | "
                f"{stats['mean']:>9.1f} | {stats['median']:>11.1f} | "
                f"{stats['p95']:>8.1f} | {stats['p99']:>8.1f}"
            )

    print()


if __name__ == "__main__":
    main()
