"""
MCP Virtual Server stress tests.

Runs concurrent workloads against the virtual MCP server endpoint
to validate behavior under load. Measures throughput, latency
percentiles, and error rates across multiple scenarios.

Usage:
    python3 tests/e2e/test_virtual_mcp_stress.py
"""

import json
import logging
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/home/ubuntu/mcp-gateway-registry-MAIN"
TOKEN_SCRIPT = f"{PROJECT_ROOT}/scripts/refresh_m2m_token.sh"
TOKEN_FILE = f"{PROJECT_ROOT}/.oauth-tokens/admin-bot-token.json"
MCP_ENDPOINT = "http://localhost/virtual/e2e-multi-backend/mcp"
CLIENT_NAME = "admin-bot"

NUM_THREADS = 20
REQUESTS_PER_THREAD = 50
SESSION_STORM_THREADS = 10
SESSION_STORM_CALLS = 10

ERROR_RATE_THRESHOLD = 10.0

_request_id_counter = 0
_request_id_lock = threading.Lock()


def _next_request_id() -> int:
    """Return a globally unique, thread-safe request ID."""
    global _request_id_counter
    with _request_id_lock:
        _request_id_counter += 1
        return _request_id_counter


def _refresh_token() -> str:
    """Refresh the OAuth token and return the access_token string."""
    subprocess.run(
        ["bash", TOKEN_SCRIPT, CLIENT_NAME],
        check=True,
        capture_output=True,
    )
    with open(TOKEN_FILE, "r") as f:
        data = json.load(f)
    token = data["access_token"]
    logger.info("Token refreshed successfully (length=%d)", len(token))
    return token


def _parse_sse_response(body: str) -> Optional[Dict[str, Any]]:
    """Parse an SSE response body, extracting the JSON from data: lines."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            payload = stripped[len("data:") :].strip()
            if payload:
                return json.loads(payload)
    return None


def _parse_response(body: str) -> Optional[Dict[str, Any]]:
    """Parse either plain JSON or SSE response body."""
    body = body.strip()
    if not body:
        return None
    # Try plain JSON first
    if body.startswith("{"):
        return json.loads(body)
    # Try SSE
    return _parse_sse_response(body)


def _send_request(
    payload: Dict[str, Any],
    token: str,
    session_id: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    """Send an MCP JSON-RPC request and return (parsed_body, response_headers)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        MCP_ENDPOINT,
        data=data,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    req.add_header("Authorization", f"Bearer {token}")
    if session_id:
        req.add_header("Mcp-Session-Id", session_id)

    resp = urllib.request.urlopen(req, timeout=timeout)
    headers = {k.lower(): v for k, v in resp.getheaders()}
    body = resp.read().decode("utf-8")
    parsed = _parse_response(body)
    return parsed, headers


def _initialize_session(token: str) -> str:
    """Perform an MCP initialize handshake and return the session ID."""
    init_payload = {
        "jsonrpc": "2.0",
        "id": _next_request_id(),
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "stress-test", "version": "1.0.0"},
        },
    }
    _, headers = _send_request(init_payload, token)
    session_id = headers.get("mcp-session-id", "")
    if not session_id:
        raise RuntimeError("No Mcp-Session-Id header in initialize response")
    logger.info("Session initialized: %s", session_id)

    # Send initialized notification (no id field)
    notif_payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    _send_request(notif_payload, token, session_id=session_id)
    return session_id


def _build_payload(method: str) -> Dict[str, Any]:
    """Build a JSON-RPC payload for the given method shorthand."""
    rid = _next_request_id()
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "method": "ping"}
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "method": "tools/list"}
    elif method == "tools/call_get_time":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {"name": "get_time", "arguments": {"timezone": "UTC"}},
        }
    elif method == "tools/call_quantum":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {"name": "quantum_flux_analyzer", "arguments": {"energy_level": 5}},
        }
    elif method == "resources/list":
        return {"jsonrpc": "2.0", "id": rid, "method": "resources/list"}
    elif method == "prompts/list":
        return {"jsonrpc": "2.0", "id": rid, "method": "prompts/list"}
    else:
        raise ValueError(f"Unknown method: {method}")


class _StressResult:
    """Thread-safe accumulator for stress test results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.successes: int = 0
        self.failures: int = 0
        self.latencies: List[float] = []
        self.errors: List[str] = []

    def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self.successes += 1
            self.latencies.append(latency_ms)

    def record_failure(self, error: str) -> None:
        with self._lock:
            self.failures += 1
            self.errors.append(error)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.failures / self.total) * 100.0


def _print_scenario_report(name: str, result: _StressResult, duration: float) -> None:
    """Print a formatted report for a single scenario."""
    throughput = result.total / duration if duration > 0 else 0.0

    p50 = p95 = p99 = max_lat = 0.0
    if result.latencies:
        sorted_lat = sorted(result.latencies)
        p50 = _percentile(sorted_lat, 50)
        p95 = _percentile(sorted_lat, 95)
        p99 = _percentile(sorted_lat, 99)
        max_lat = sorted_lat[-1]

    print(f"\n=== Scenario: {name} ===")
    print(f"Total requests: {result.total}")
    print(f"Successful:     {result.successes}")
    print(f"Failed:         {result.failures}")
    print(f"Error rate:     {result.error_rate:.1f}%")
    print(f"Duration:       {duration:.1f}s")
    print(f"Throughput:     {throughput:.1f} req/s")
    print(f"Latency (ms):   p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}  max={max_lat:.1f}")

    if result.errors:
        unique_errors = {}
        for e in result.errors:
            short = e[:120]
            unique_errors[short] = unique_errors.get(short, 0) + 1
        print(f"Error summary ({len(result.errors)} total):")
        for err, count in sorted(unique_errors.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{count}x] {err}")


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Compute a percentile from sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def _worker_repeated_requests(
    method: str,
    token: str,
    session_id: str,
    count: int,
    result: _StressResult,
) -> None:
    """Worker function: send `count` requests of a given method."""
    for _ in range(count):
        payload = _build_payload(method)
        start = time.perf_counter()
        try:
            _send_request(payload, token, session_id=session_id)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_success(elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_failure(str(exc))


def _worker_mixed_requests(
    token: str,
    session_id: str,
    count: int,
    result: _StressResult,
) -> None:
    """Worker function: send `count` requests with random method mix."""
    methods = [
        "ping",
        "tools/list",
        "tools/call_get_time",
        "tools/call_quantum",
        "resources/list",
        "prompts/list",
    ]
    for _ in range(count):
        method = random.choice(methods)
        payload = _build_payload(method)
        start = time.perf_counter()
        try:
            _send_request(payload, token, session_id=session_id)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_success(elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_failure(str(exc))


def _worker_session_storm(
    token: str,
    calls_per_session: int,
    result: _StressResult,
) -> None:
    """Worker function: create a new session, then make tool calls on it."""
    try:
        sid = _initialize_session(token)
    except Exception as exc:
        # Count the initialize failure plus all planned calls as failures
        for _ in range(calls_per_session + 1):
            result.record_failure(f"session init: {exc}")
        return

    for _ in range(calls_per_session):
        payload = _build_payload("tools/call_get_time")
        start = time.perf_counter()
        try:
            _send_request(payload, token, session_id=sid)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_success(elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.record_failure(str(exc))


def _run_scenario_concurrent(
    name: str,
    method: str,
    token: str,
    session_id: str,
    num_threads: int = NUM_THREADS,
    requests_per_thread: int = REQUESTS_PER_THREAD,
) -> _StressResult:
    """Run a scenario where all threads call the same method."""
    result = _StressResult()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                _worker_repeated_requests,
                method,
                token,
                session_id,
                requests_per_thread,
                result,
            )
            for _ in range(num_threads)
        ]
        for f in as_completed(futures):
            f.result()  # propagate exceptions from workers

    duration = time.perf_counter() - start
    _print_scenario_report(name, result, duration)
    return result


def _run_scenario_mixed(
    name: str,
    token: str,
    session_id: str,
    num_threads: int = NUM_THREADS,
    requests_per_thread: int = REQUESTS_PER_THREAD,
) -> _StressResult:
    """Run a mixed-workload scenario."""
    result = _StressResult()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                _worker_mixed_requests,
                token,
                session_id,
                requests_per_thread,
                result,
            )
            for _ in range(num_threads)
        ]
        for f in as_completed(futures):
            f.result()

    duration = time.perf_counter() - start
    _print_scenario_report(name, result, duration)
    return result


def _run_scenario_session_storm(
    name: str,
    token: str,
    num_threads: int = SESSION_STORM_THREADS,
    calls_per_session: int = SESSION_STORM_CALLS,
) -> _StressResult:
    """Run the session-storm scenario: each thread creates its own session."""
    result = _StressResult()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(
                _worker_session_storm,
                token,
                calls_per_session,
                result,
            )
            for _ in range(num_threads)
        ]
        for f in as_completed(futures):
            f.result()

    duration = time.perf_counter() - start
    _print_scenario_report(name, result, duration)
    return result


def main() -> int:
    """Run all stress test scenarios and return 0 if all pass, 1 otherwise."""
    print("=" * 60)
    print("MCP Virtual Server Stress Tests")
    print("=" * 60)

    # Refresh token
    logger.info("Refreshing OAuth token...")
    token = _refresh_token()

    # Initialize a shared session for scenarios 1-3
    logger.info("Initializing shared MCP session...")
    session_id = _initialize_session(token)

    results: List[Tuple[str, _StressResult]] = []

    # Scenario 1: Concurrent tools/list
    logger.info("Starting scenario: Concurrent tools/list")
    r1 = _run_scenario_concurrent(
        "Concurrent tools/list",
        "tools/list",
        token,
        session_id,
    )
    results.append(("Concurrent tools/list", r1))

    # Refresh token between scenarios to avoid expiry
    token = _refresh_token()

    # Scenario 2: Concurrent tools/call (get_time)
    logger.info("Starting scenario: Concurrent tools/call (get_time)")
    r2 = _run_scenario_concurrent(
        "Concurrent tools/call (get_time)",
        "tools/call_get_time",
        token,
        session_id,
    )
    results.append(("Concurrent tools/call (get_time)", r2))

    # Refresh token
    token = _refresh_token()

    # Scenario 3: Mixed workload
    logger.info("Starting scenario: Mixed workload")
    r3 = _run_scenario_mixed(
        "Mixed workload",
        token,
        session_id,
    )
    results.append(("Mixed workload", r3))

    # Refresh token
    token = _refresh_token()

    # Scenario 4: Session storm
    logger.info("Starting scenario: Session storm")
    r4 = _run_scenario_session_storm(
        "Session storm",
        token,
    )
    results.append(("Session storm", r4))

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "PASS" if result.error_rate < ERROR_RATE_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {name}: error_rate={result.error_rate:.1f}%")

    if all_passed:
        print("\nAll scenarios PASSED (error rate < 10%)")
        return 0
    else:
        print("\nSome scenarios FAILED (error rate >= 10%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
