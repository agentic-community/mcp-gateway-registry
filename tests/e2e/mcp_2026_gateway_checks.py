#!/usr/bin/env python3
"""Check the gateway's MCP 2026-07-28 behaviors against a LIVE gateway.

Each check sends raw HTTP requests, so it pins exactly what the gateway puts
on the wire:

    discover-direct      server/discover through a direct route lists 2026-07-28
    discover-virtual     the virtual router answers server/discover itself
    legacy-handshake     a 2025-06-18 initialize still works on a direct route
    mismatch-name        an Mcp-Name that disagrees with params.name gets 400 -32020
    mismatch-method      an Mcp-Method that disagrees with the body gets 400 -32020
    mismatch-virtual     the same check on a virtual route
    cache-scope          a per-user filtered tools/list is marked cacheScope private
    one-challenge-401    a request without a token gets exactly one WWW-Authenticate
    challenge-403        a request the scopes deny names resource_metadata and no
                         insufficient_scope (only with --restricted-token-file)

The routes need the MCP compatibility test upstreams from cli/examples/
(mcp-everything.json and virtual-server-mcp-compat.json). The script prints
one line per check and exits non-zero if any check fails.

Usage:
    # Local gateway, admin token at ./.token
    uv run python tests/e2e/mcp_2026_gateway_checks.py

    # Also check the 403 challenge with a user who cannot reach /everything/
    uv run python tests/e2e/mcp_2026_gateway_checks.py --restricted-token-file .token2
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_FILE: Path = PROJECT_ROOT / ".token"
DEFAULT_BASE_URL: str = "http://localhost"
DIRECT_ROUTE: str = "/everything/mcp"
VIRTUAL_ROUTE: str = "/virtual/compat-test/mcp"
DIRECT_TOOL: str = "echo"
VIRTUAL_TOOL: str = "pm_echo"
MODERN_PROTOCOL_VERSION: str = "2026-07-28"
LEGACY_PROTOCOL_VERSION: str = "2025-06-18"
HEADER_MISMATCH_CODE: int = -32020
REQUEST_TIMEOUT_SECONDS: float = 30.0
META: dict[str, Any] = {
    "io.modelcontextprotocol/protocolVersion": MODERN_PROTOCOL_VERSION,
    "io.modelcontextprotocol/clientInfo": {"name": "mcp-2026-gateway-checks", "version": "1.0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}


def _load_token(
    token_file: Path,
) -> str:
    """Read the access token from a token file (nested or flat JSON)."""
    data = json.loads(token_file.read_text())
    token = (data.get("tokens") or {}).get("access_token") or data.get("access_token")
    if not token:
        raise ValueError(f"No access_token found in {token_file}")
    return token


def _last_jsonrpc_message(
    text: str,
) -> dict[str, Any]:
    """Return the last JSON-RPC message in a JSON or SSE response body."""
    data_lines = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    return json.loads(data_lines[-1] if data_lines else text)


def _modern_post(
    url: str,
    token: str | None,
    method: str,
    params: dict[str, Any],
    header_method: str | None = None,
    header_name: str | None = None,
) -> requests.Response:
    """POST a 2026-07-28 request, with the routing headers set from the arguments."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MODERN_PROTOCOL_VERSION,
        "Mcp-Method": header_method or method,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if header_name:
        headers["Mcp-Name"] = header_name
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {**params, "_meta": META}}
    return requests.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)


def _check_discover(
    url: str,
    token: str,
) -> tuple[bool, str]:
    response = _modern_post(url, token, "server/discover", {})
    versions = _last_jsonrpc_message(response.text).get("result", {}).get("supportedVersions", [])
    return MODERN_PROTOCOL_VERSION in versions, f"HTTP {response.status_code}, versions={versions}"


def _check_legacy_handshake(
    url: str,
    token: str,
) -> tuple[bool, str]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp-2026-gateway-checks", "version": "1.0"},
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    response = requests.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    version = _last_jsonrpc_message(response.text).get("result", {}).get("protocolVersion")
    return response.status_code == 200 and bool(
        version
    ), f"HTTP {response.status_code}, version={version}"


def _check_mismatch(
    url: str,
    token: str,
    tool: str,
    header_method: str | None,
    header_name: str | None,
) -> tuple[bool, str]:
    params = {"name": tool, "arguments": {"message": "x"}}
    response = _modern_post(url, token, "tools/call", params, header_method, header_name)
    code = _last_jsonrpc_message(response.text).get("error", {}).get("code")
    passed = response.status_code == 400 and code == HEADER_MISMATCH_CODE
    return passed, f"HTTP {response.status_code}, code={code}"


def _check_cache_scope(
    url: str,
    token: str,
) -> tuple[bool, str]:
    response = _modern_post(url, token, "tools/list", {})
    scope = _last_jsonrpc_message(response.text).get("result", {}).get("cacheScope")
    return scope == "private", f"HTTP {response.status_code}, cacheScope={scope}"


def _check_one_challenge(
    url: str,
) -> tuple[bool, str]:
    response = requests.post(
        url,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    challenges = [v for k, v in response.raw.headers.items() if k.lower() == "www-authenticate"]
    passed = (
        response.status_code == 401
        and len(challenges) == 1
        and "resource_metadata" in challenges[0]
    )
    return passed, f"HTTP {response.status_code}, {len(challenges)} challenge(s)"


def _check_forbidden_challenge(
    url: str,
    restricted_token: str,
) -> tuple[bool, str]:
    response = _modern_post(url, restricted_token, "tools/list", {})
    challenge = response.headers.get("www-authenticate", "")
    passed = (
        response.status_code == 403
        and "resource_metadata" in challenge
        and "insufficient_scope" not in challenge
    )
    return passed, f"HTTP {response.status_code}, WWW-Authenticate={challenge[:60]!r}"


def _build_checks(
    base_url: str,
    token: str,
    restricted_token: str | None,
) -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    """Name every check with a zero-argument callable that runs it."""
    direct = base_url.rstrip("/") + DIRECT_ROUTE
    virtual = base_url.rstrip("/") + VIRTUAL_ROUTE
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("discover-direct", lambda: _check_discover(direct, token)),
        ("discover-virtual", lambda: _check_discover(virtual, token)),
        ("legacy-handshake", lambda: _check_legacy_handshake(direct, token)),
        ("mismatch-name", lambda: _check_mismatch(direct, token, DIRECT_TOOL, None, "get-sum")),
        (
            "mismatch-method",
            lambda: _check_mismatch(direct, token, DIRECT_TOOL, "tools/list", DIRECT_TOOL),
        ),
        (
            "mismatch-virtual",
            lambda: _check_mismatch(virtual, token, VIRTUAL_TOOL, None, "ping_echo"),
        ),
        ("cache-scope", lambda: _check_cache_scope(direct, token)),
        ("one-challenge-401", lambda: _check_one_challenge(direct)),
    ]
    if restricted_token:
        checks.append(
            ("challenge-403", lambda: _check_forbidden_challenge(direct, restricted_token))
        )
    return checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the gateway's MCP 2026-07-28 behaviors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base", default=DEFAULT_BASE_URL, help="Gateway base URL")
    parser.add_argument(
        "--token-file", type=Path, default=DEFAULT_TOKEN_FILE, help="Admin token file"
    )
    parser.add_argument(
        "--restricted-token-file",
        type=Path,
        default=None,
        help="Token of a user who cannot reach the direct route, for the 403 check",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    restricted = _load_token(args.restricted_token_file) if args.restricted_token_file else None
    checks = _build_checks(args.base, _load_token(args.token_file), restricted)
    failures = 0
    for name, run in checks:
        try:
            passed, detail = run()
        except (requests.RequestException, ValueError) as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        failures += 0 if passed else 1
        print(f"{'PASS' if passed else 'FAIL'}  {name:20} {detail}")
    if failures:
        logger.error(f"{failures} of {len(checks)} check(s) failed")
        return 1
    logger.info(f"All {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
