#!/usr/bin/env python3
"""Call one tool on each gateway route with old and new MCP clients.

Runs against a LIVE gateway. Each route is called with four client styles:

    raw-2025-03-26  a hand-written handshake (initialize, then tools/call), as
                    tests/scripts/call_mcp_tool.py and the CLI do
    sdk-legacy      MCP SDK 2.x, mode="legacy" (initialize with 2025-11-25)
    sdk-auto        MCP SDK 2.x, mode="auto" (server/discover, falls back to initialize)
    sdk-2026        MCP SDK 2.x pinned to 2026-07-28 (no handshake)

Every cell has an expected outcome. A pairing that cannot work by design (for
example a 2026-07-28-only client against a handshake-only server on a direct
route, which the gateway does not translate) is expected to fail. The script
prints a table and exits non-zero if any cell differs from its expectation.

The default routes need the MCP compatibility test upstreams from
cli/examples/ (mcp-everything.json, mcp-everything-legacy.json, pingmcp-modern.json,
pingmcp-dual.json, virtual-server-mcp-compat.json) plus the built-in airegistry-tools server.

Usage:
    # Local gateway, admin token at ./.token
    uv run python tests/e2e/mcp_client_matrix.py

    # Also check servers that use the caller's own egress connection
    # (GitHub OAuth and Slack must already be connected for this user)
    uv run python tests/e2e/mcp_client_matrix.py --include-egress

    # Another gateway and token file
    uv run python tests/e2e/mcp_client_matrix.py \\
        --base https://mcpgateway.example.com --token-file .oauth-tokens/ingress.json

    # Your own routes: a JSON list of
    # {"route": "/x/mcp", "tool": "echo", "arguments": {}, "expected_ok": [true, true, true, true]}
    # (expected_ok follows the client styles above; it defaults to all true)
    uv run python tests/e2e/mcp_client_matrix.py --targets-file my-routes.json
"""

import argparse
import asyncio
import json
import logging
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi
import httpx2
import requests
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

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
RAW_PROTOCOL_VERSION: str = "2025-03-26"
MODERN_PROTOCOL_VERSION: str = "2026-07-28"
REQUEST_TIMEOUT_SECONDS: float = 60.0
CLIENT_STYLES: tuple[str, ...] = ("raw-2025-03-26", "sdk-legacy", "sdk-auto", "sdk-2026")
ALL_OK: tuple[bool, ...] = (True, True, True, True)
CELL_WIDTH: int = 26
ROUTE_WIDTH: int = 56


@dataclass(frozen=True)
class Target:
    """One route and tool, with the expected outcome per client style."""

    route: str
    tool: str
    arguments: dict[str, Any]
    expected_ok: tuple[bool, ...]
    note: str = ""


DEFAULT_TARGETS: tuple[Target, ...] = (
    Target("/airegistry-tools/mcp", "healthcheck", {}, ALL_OK, "mcpgw on fastmcp 4, dual-era"),
    Target("/everything/mcp", "echo", {"message": "matrix"}, ALL_OK, "reference server, dual-era"),
    Target("/pingmcp-dual/mcp", "echo", {"message": "matrix"}, ALL_OK, "dual-era"),
    Target(
        "/pingmcp/mcp",
        "echo",
        {"message": "matrix"},
        (True, True, True, False),
        "handshake-only server; the direct route does not translate for a 2026-only client",
    ),
    Target(
        "/pingmcp-modern/mcp",
        "echo",
        {"message": "matrix"},
        (False, False, True, True),
        "2026-only server; the direct route does not translate for handshake clients",
    ),
    Target(
        "/cloudflare-docs/mcp",
        "search_cloudflare_documentation",
        {"query": "R2 pricing"},
        ALL_OK,
        "public HTTPS server",
    ),
    Target(
        "/virtual/compat-test/mcp",
        "pm_echo",
        {"message": "matrix"},
        ALL_OK,
        "virtual route over a 2026-only backend",
    ),
    Target(
        "/virtual/compat-test/mcp",
        "ping_echo",
        {"message": "matrix"},
        ALL_OK,
        "virtual route over a handshake-only backend; the router translates",
    ),
)

EGRESS_TARGETS: tuple[Target, ...] = (
    Target(
        "/com-github-github-mcp-server/mcp",
        "get_me",
        {},
        ALL_OK,
        "GitHub OAuth egress (needs this user's connection)",
    ),
    Target(
        "/slack/mcp",
        "slack_search_users",
        {"query": "a"},
        (True, True, True, False),
        "Slack OAuth egress (needs this user's connection); Slack speaks only the handshake",
    ),
)


def _load_targets(
    targets_file: Path,
) -> tuple[Target, ...]:
    """Read routes to test from a JSON list of route, tool, arguments, expected_ok."""
    entries = json.loads(targets_file.read_text())
    targets = []
    for entry in entries:
        expected = tuple(entry.get("expected_ok", ALL_OK))
        if len(expected) != len(CLIENT_STYLES):
            raise ValueError(f"expected_ok for {entry['route']} needs {len(CLIENT_STYLES)} values")
        targets.append(
            Target(
                entry["route"],
                entry["tool"],
                entry.get("arguments", {}),
                expected,
                entry.get("note", ""),
            )
        )
    return tuple(targets)


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


def _raw_handshake_call(
    url: str,
    token: str,
    target: Target,
) -> tuple[bool, str]:
    """Call the tool with a hand-written 2025-03-26 handshake."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": RAW_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp-client-matrix", "version": "1.0"},
        },
    }
    response = requests.post(url, json=initialize, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        return False, f"initialize HTTP {response.status_code}"
    session_id = response.headers.get("mcp-session-id")
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    headers["MCP-Protocol-Version"] = RAW_PROTOCOL_VERSION
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    requests.post(url, json=initialized, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": target.tool, "arguments": target.arguments},
    }
    response = requests.post(url, json=call, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        return False, f"tools/call HTTP {response.status_code}"
    message = _last_jsonrpc_message(response.text)
    if "error" in message:
        return False, f"error {message['error'].get('code')}"
    if message.get("result", {}).get("isError"):
        return False, "tool isError"
    return True, "ok"


async def _sdk_call(
    url: str,
    token: str,
    target: Target,
    mode: str,
) -> tuple[bool, str]:
    """Call the tool with the MCP SDK 2.x client in the given mode."""
    tls = ssl.create_default_context(cafile=certifi.where())
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx2.AsyncClient(
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=tls,
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport, mode=mode) as client:
            version = getattr(client, "protocol_version", "?")
            result = await client.call_tool(target.tool, target.arguments)
    if result.is_error:
        return False, f"tool isError ({version})"
    return True, f"ok ({version})"


def _root_error(
    error: BaseException,
) -> BaseException:
    """Unwrap an ExceptionGroup to its first leaf exception."""
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]
    return error


def _run_cell(
    style: str,
    url: str,
    token: str,
    target: Target,
) -> tuple[bool, str]:
    """Run one client style against one target; never raises."""
    try:
        if style == "raw-2025-03-26":
            return _raw_handshake_call(url, token, target)
        mode = {"sdk-legacy": "legacy", "sdk-auto": "auto", "sdk-2026": MODERN_PROTOCOL_VERSION}[
            style
        ]
        return asyncio.run(_sdk_call(url, token, target, mode))
    except Exception as error:  # a failed call is a result for the table
        root = _root_error(error)
        return False, f"{type(root).__name__}: {str(root)[:40]}"


def _format_cell(
    ok: bool,
    detail: str,
    expected: bool,
) -> str:
    """Render one table cell, flagging outcomes that differ from expectation."""
    if ok == expected:
        marker = "" if ok else "(expected) "
        return f"{marker}{detail}"[:CELL_WIDTH]
    return f"UNEXPECTED {detail}"[:CELL_WIDTH]


def _run_matrix(
    base_url: str,
    token: str,
    targets: tuple[Target, ...],
) -> int:
    """Print the matrix and return the number of unexpected outcomes."""
    header = f"{'route / tool':{ROUTE_WIDTH}} " + " | ".join(
        f"{s:{CELL_WIDTH}}" for s in CLIENT_STYLES
    )
    print(header)
    print("-" * len(header))
    unexpected = 0
    for target in targets:
        url = base_url.rstrip("/") + target.route
        cells = []
        for style, expected in zip(CLIENT_STYLES, target.expected_ok, strict=True):
            ok, detail = _run_cell(style, url, token, target)
            if ok != expected:
                unexpected += 1
            cells.append(_format_cell(ok, detail, expected))
        label = f"{target.route} {target.tool}"[:ROUTE_WIDTH]
        print(f"{label:{ROUTE_WIDTH}} " + " | ".join(f"{c:{CELL_WIDTH}}" for c in cells))
    return unexpected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call one tool per gateway route with old and new MCP clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base", default=DEFAULT_BASE_URL, help="Gateway base URL")
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="Token file with tokens.access_token or access_token",
    )
    parser.add_argument(
        "--include-egress",
        action="store_true",
        help="Also check GitHub and Slack, which need this user's egress connections",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="JSON list of routes to test instead of the default compatibility routes",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # The SDK and httpx log every request at INFO; keep the table readable.
        for noisy in ("httpx", "httpx2", "mcp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    base_targets = _load_targets(args.targets_file) if args.targets_file else DEFAULT_TARGETS
    targets = base_targets + (EGRESS_TARGETS if args.include_egress else ())
    logger.info(f"Gateway {args.base}, {len(targets)} routes, {len(CLIENT_STYLES)} client styles")
    start = time.time()
    unexpected = _run_matrix(args.base, _load_token(args.token_file), targets)
    elapsed = time.time() - start
    if unexpected:
        logger.error(f"{unexpected} unexpected outcome(s) in {elapsed:.1f} seconds")
        return 1
    logger.info(f"All outcomes as expected in {elapsed:.1f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
