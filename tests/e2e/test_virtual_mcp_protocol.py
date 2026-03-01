#!/usr/bin/env python3
"""
E2E tests for the Virtual MCP Server protocol.

Tests the full MCP JSON-RPC protocol through the virtual server endpoint,
verifying initialize, ping, tools/list, tools/call, resources, prompts,
and error handling behaviors.

Usage:
    python3 tests/e2e/test_virtual_mcp_protocol.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIRTUAL_SERVER_ENDPOINT = "http://localhost/virtual/e2e-multi-backend/mcp"
TOKEN_REFRESH_SCRIPT = str(PROJECT_ROOT / "scripts" / "refresh_m2m_token.sh")
TOKEN_FILE = str(PROJECT_ROOT / ".oauth-tokens" / "admin-bot-token.json")
CLIENT_NAME = "admin-bot"

EXPECTED_TOOLS = [
    "get_time",
    "quantum_flux_analyzer",
    "synth_patterns",
    "synthetic_data_generator",
]


def _refresh_token() -> str:
    """Refresh the OAuth token and return the access token string.

    Returns:
        The access token string.

    Raises:
        RuntimeError: If the token refresh fails.
    """
    result = subprocess.run(
        ["bash", TOKEN_REFRESH_SCRIPT, CLIENT_NAME],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Token refresh failed (exit {result.returncode}):\n{result.stderr}")

    with open(TOKEN_FILE, "r") as f:
        token_data = json.load(f)

    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("No access_token in token file after refresh")

    return access_token


def _build_headers(
    token: str,
    session_id: Optional[str] = None,
) -> Dict[str, str]:
    """Build HTTP headers for MCP requests."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {token}",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def _parse_response(
    raw: str,
    content_type: str,
) -> Dict[str, Any]:
    """Parse a response that may be JSON or SSE format.

    Returns:
        Parsed JSON dict from the response body.
    """
    if "text/event-stream" in content_type:
        for line in raw.strip().split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise ValueError("No valid JSON data line found in SSE response")
    return json.loads(raw)


def _send_request(
    payload: Dict[str, Any],
    token: str,
    session_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Send a JSON-RPC request to the virtual MCP endpoint.

    Returns:
        Tuple of (parsed_response_body, response_headers_dict).
    """
    headers = _build_headers(token, session_id)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        VIRTUAL_SERVER_ENDPOINT,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            content_type = resp.headers.get("content-type", "")
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            resp_headers["_status"] = str(resp.status)
            parsed = _parse_response(raw, content_type)
            return parsed, resp_headers
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        content_type = e.headers.get("content-type", "") if e.headers else ""
        resp_headers = {"_status": str(e.code)}
        if e.headers:
            resp_headers.update({k.lower(): v for k, v in e.headers.items()})
        try:
            parsed = _parse_response(error_body, content_type)
        except (json.JSONDecodeError, ValueError):
            parsed = {"raw_error": error_body, "http_code": e.code}
        return parsed, resp_headers


def _send_raw_http(
    method: str,
    token: str,
    session_id: Optional[str] = None,
    body: Optional[bytes] = None,
) -> Tuple[int, str, Dict[str, str]]:
    """Send a raw HTTP request (GET/DELETE/POST) and return status, body, headers.

    Returns:
        Tuple of (http_status_code, response_body, response_headers_dict).
    """
    headers = _build_headers(token, session_id)
    if method == "GET":
        headers["Accept"] = "text/event-stream"

    req = urllib.request.Request(
        VIRTUAL_SERVER_ENDPOINT,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, raw, resp_headers
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        resp_headers = {}
        if e.headers:
            resp_headers = {k.lower(): v for k, v in e.headers.items()}
        return e.code, error_body, resp_headers


class VirtualMCPProtocolTests:
    """E2E test suite for the Virtual MCP Server protocol."""

    def __init__(self) -> None:
        self._token: str = ""
        self._session_id: Optional[str] = None
        self._request_id: int = 0
        self._results: List[Tuple[str, bool, str]] = []

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _record(
        self,
        name: str,
        passed: bool,
        detail: str = "",
    ) -> None:
        self._results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)

    def setup(self) -> None:
        """Refresh the token before running tests."""
        print("Refreshing OAuth token...")
        self._token = _refresh_token()
        print("Token obtained successfully.\n")

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def test_01_initialize(self) -> None:
        """Verify initialize returns capabilities and session ID."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test-client", "version": "1.0.0"},
            },
        }
        body, headers = _send_request(payload, self._token)

        try:
            result = body.get("result", {})
            assert "protocolVersion" in result, "Missing protocolVersion"
            assert result["protocolVersion"] == "2025-11-25", (
                f"Expected negotiated version '2025-11-25', got '{result['protocolVersion']}'"
            )
            caps = result.get("capabilities", {})
            assert "tools" in caps, "Missing tools capability"
            assert "resources" in caps, "Missing resources capability"
            assert "prompts" in caps, "Missing prompts capability"
            assert "serverInfo" in result, "Missing serverInfo"

            session_id = headers.get("mcp-session-id", "")
            assert session_id.startswith("vs-"), (
                f"Session ID should start with 'vs-', got: {session_id}"
            )
            self._session_id = session_id

            self._record("initialize", True)
        except AssertionError as e:
            self._record("initialize", False, str(e))

    def test_01a_initialize_version_negotiation(self) -> None:
        """Verify server echoes back supported protocol version."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test-client", "version": "1.0.0"},
            },
        }
        body, _ = _send_request(payload, self._token)

        try:
            result = body.get("result", {})
            assert result.get("protocolVersion") == "2024-11-05", (
                f"Expected '2024-11-05' echoed back, got '{result.get('protocolVersion')}'"
            )
            self._record("initialize version negotiation (old)", True)
        except AssertionError as e:
            self._record("initialize version negotiation (old)", False, str(e))

    def test_01b_initialize_version_unsupported(self) -> None:
        """Verify server returns its latest version for unsupported client version."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "9999-01-01",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test-client", "version": "1.0.0"},
            },
        }
        body, _ = _send_request(payload, self._token)

        try:
            result = body.get("result", {})
            version = result.get("protocolVersion", "")
            assert version == "2025-11-25", (
                f"Expected server's latest version '2025-11-25', got '{version}'"
            )
            self._record("initialize version negotiation (unsupported)", True)
        except AssertionError as e:
            self._record("initialize version negotiation (unsupported)", False, str(e))

    def test_01c_notifications_initialized_returns_202(self) -> None:
        """Verify notifications/initialized returns HTTP 202 with no body."""
        payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        data = json.dumps(payload).encode("utf-8")
        status, body, _ = _send_raw_http("POST", self._token, self._session_id, data)

        try:
            assert status == 202, f"Expected HTTP 202 Accepted, got {status}"
            assert body.strip() == "", f"Expected empty body for 202, got: '{body[:100]}'"
            self._record("notifications/initialized -> 202", True)
        except AssertionError as e:
            self._record("notifications/initialized -> 202", False, str(e))

    def test_01d_get_returns_405(self) -> None:
        """Verify HTTP GET returns 405 Method Not Allowed (no SSE support)."""
        status, _, _ = _send_raw_http("GET", self._token, self._session_id)

        try:
            assert status == 405, f"Expected HTTP 405 for GET, got {status}"
            self._record("GET -> 405 Method Not Allowed", True)
        except AssertionError as e:
            self._record("GET -> 405 Method Not Allowed", False, str(e))

    def test_01e_delete_returns_405(self) -> None:
        """Verify HTTP DELETE returns 405 Method Not Allowed."""
        status, _, _ = _send_raw_http("DELETE", self._token, self._session_id)

        try:
            assert status == 405, f"Expected HTTP 405 for DELETE, got {status}"
            self._record("DELETE -> 405 Method Not Allowed", True)
        except AssertionError as e:
            self._record("DELETE -> 405 Method Not Allowed", False, str(e))

    def test_02_ping(self) -> None:
        """Verify ping returns empty result."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "ping",
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            assert "result" in body, f"No result key in response: {body}"
            assert body["result"] == {}, f"Expected empty result, got: {body['result']}"
            self._record("ping", True)
        except AssertionError as e:
            self._record("ping", False, str(e))

    def test_03_tools_list(self) -> None:
        """Verify tools/list returns exactly 4 expected tools."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            tools = result.get("tools", [])
            tool_names = sorted([t["name"] for t in tools])
            assert tool_names == sorted(EXPECTED_TOOLS), (
                f"Expected tools {sorted(EXPECTED_TOOLS)}, got {tool_names}"
            )

            for tool in tools:
                assert "inputSchema" in tool, f"Tool '{tool['name']}' missing inputSchema key"

            self._record("tools/list", True, f"{len(tools)} tools found")
        except (AssertionError, KeyError) as e:
            self._record("tools/list", False, str(e))

    def test_04_call_get_time(self) -> None:
        """Call get_time with timezone=UTC and verify response."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "get_time",
                "arguments": {"timezone": "UTC"},
            },
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            content = result.get("content", [])
            assert len(content) > 0, "Expected non-empty content array"
            assert content[0].get("type") == "text", (
                f"Expected type 'text', got '{content[0].get('type')}'"
            )
            assert content[0].get("text"), "Expected non-empty text"
            self._record("tools/call get_time", True)
        except (AssertionError, KeyError, IndexError) as e:
            self._record("tools/call get_time", False, str(e))

    def test_05_call_quantum_flux_analyzer(self) -> None:
        """Call quantum_flux_analyzer with energy_level=7."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "quantum_flux_analyzer",
                "arguments": {"energy_level": 7},
            },
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            content = result.get("content", [])
            assert len(content) > 0, "Expected non-empty content array"
            assert content[0].get("type") == "text", (
                f"Expected type 'text', got '{content[0].get('type')}'"
            )
            assert content[0].get("text"), "Expected non-empty text"
            self._record("tools/call quantum_flux_analyzer", True)
        except (AssertionError, KeyError, IndexError) as e:
            self._record("tools/call quantum_flux_analyzer", False, str(e))

    def test_06_call_synth_patterns(self) -> None:
        """Call synth_patterns with input_patterns=["a","b"]."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "synth_patterns",
                "arguments": {"input_patterns": ["a", "b"]},
            },
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            content = result.get("content", [])
            assert len(content) > 0, "Expected non-empty content array"
            assert content[0].get("type") == "text", (
                f"Expected type 'text', got '{content[0].get('type')}'"
            )
            assert content[0].get("text"), "Expected non-empty text"
            self._record("tools/call synth_patterns", True)
        except (AssertionError, KeyError, IndexError) as e:
            self._record("tools/call synth_patterns", False, str(e))

    def test_07_call_synthetic_data_generator(self) -> None:
        """Call synthetic_data_generator with schema and record_count."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "synthetic_data_generator",
                "arguments": {
                    "schema": {"name": "string"},
                    "record_count": 2,
                },
            },
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            content = result.get("content", [])
            assert len(content) > 0, "Expected non-empty content array"
            assert content[0].get("type") == "text", (
                f"Expected type 'text', got '{content[0].get('type')}'"
            )
            assert content[0].get("text"), "Expected non-empty text"
            self._record("tools/call synthetic_data_generator", True)
        except (AssertionError, KeyError, IndexError) as e:
            self._record("tools/call synthetic_data_generator", False, str(e))

    def test_08_resources_list(self) -> None:
        """Verify resources/list returns a response with resources key."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/list",
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            assert "resources" in result, f"Expected 'resources' key in result, got: {result}"
            self._record("resources/list", True)
        except AssertionError as e:
            self._record("resources/list", False, str(e))

    def test_09_resources_read_error(self) -> None:
        """Verify resources/read for non-existent resource returns error."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/read",
            "params": {"uri": "config://app"},
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            assert "error" in body, f"Expected error response, got: {body}"
            self._record("resources/read error", True)
        except AssertionError as e:
            self._record("resources/read error", False, str(e))

    def test_10_prompts_list(self) -> None:
        """Verify prompts/list returns a response with prompts key."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/list",
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            result = body.get("result", {})
            assert "prompts" in result, f"Expected 'prompts' key in result, got: {result}"
            self._record("prompts/list", True)
        except AssertionError as e:
            self._record("prompts/list", False, str(e))

    def test_11_prompts_get_error(self) -> None:
        """Verify prompts/get for non-existent prompt returns error."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/get",
            "params": {"name": "system_prompt_for_agent"},
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            assert "error" in body, f"Expected error response, got: {body}"
            self._record("prompts/get error", True)
        except AssertionError as e:
            self._record("prompts/get error", False, str(e))

    def test_12_error_nonexistent_tool(self) -> None:
        """Verify calling a non-existent tool returns an error."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            assert "error" in body, f"Expected error for nonexistent tool, got: {body}"
            self._record("error: non-existent tool", True)
        except AssertionError as e:
            self._record("error: non-existent tool", False, str(e))

    def test_13_error_unknown_method(self) -> None:
        """Verify sending an unknown method returns an error."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "unknown/method",
        }
        body, _ = _send_request(payload, self._token, self._session_id)

        try:
            assert "error" in body, f"Expected error for unknown method, got: {body}"
            self._record("error: unknown method", True)
        except AssertionError as e:
            self._record("error: unknown method", False, str(e))

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run_all(self) -> bool:
        """Run all tests and return True if all passed."""
        print("=" * 60)
        print("Virtual MCP Protocol E2E Tests")
        print("=" * 60)
        print(f"Endpoint: {VIRTUAL_SERVER_ENDPOINT}")
        print()

        try:
            self.setup()
        except Exception as e:
            print(f"SETUP FAILED: {e}")
            return False

        test_methods = sorted(
            [m for m in dir(self) if m.startswith("test_")],
        )

        start = time.time()
        for method_name in test_methods:
            method = getattr(self, method_name)
            try:
                method()
            except Exception as e:
                test_label = method_name.replace("test_", "").lstrip("0123456789_")
                self._record(test_label, False, f"Unhandled exception: {e}")

        elapsed = time.time() - start

        # Summary
        passed = sum(1 for _, ok, _ in self._results if ok)
        failed = sum(1 for _, ok, _ in self._results if not ok)
        total = len(self._results)

        print()
        print("-" * 60)
        print(f"Results: {passed}/{total} passed, {failed} failed ({elapsed:.1f}s)")
        print("-" * 60)

        if failed > 0:
            print("\nFailed tests:")
            for name, ok, detail in self._results:
                if not ok:
                    print(f"  - {name}: {detail}")

        return failed == 0


def main() -> None:
    suite = VirtualMCPProtocolTests()
    success = suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
