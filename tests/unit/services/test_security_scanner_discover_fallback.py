"""Tests for MCP_SCANNER_DISCOVER_FALLBACK_PATCH in registry.services.security_scanner.

The patch runs inside the scanner subprocess, so these tests run it in a subprocess
too. That keeps the monkeypatch out of the test process and exercises the exact
source string the registry ships.
"""

import json
import subprocess
import sys

import pytest

from registry.services.scanner_llm_analyzer import MCP_SCANNER_LITELLM_STANDIN
from registry.services.security_scanner import (
    _SCANNER_SHIM,
    MCP_SCANNER_DISCOVER_FALLBACK_PATCH,
)

MCP_1X_UNSUPPORTED_VERSION: str = (
    "Bad Request: Unsupported protocol version: 2026-07-28. "
    "Supported versions: 2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25"
)
SUBPROCESS_TIMEOUT_SECONDS: int = 60

# Each case: (label, error code, error message, legacy_http, expected decision).
CASES: list[tuple[str, int, str, bool, bool]] = [
    ("mcp 1.x server rejects the 2026 header", -32600, MCP_1X_UNSUPPORTED_VERSION, True, True),
    ("missing session id (already handled)", -32600, "Bad Request: Missing session ID", True, True),
    ("legacy method not found", -32601, "Method not found", True, True),
    ("legacy generic invalid params", -32602, "Invalid params", True, True),
    ("2026 header mismatch stays modern", -32020, "Header mismatch", True, False),
]


def _decisions(
    patched: bool,
) -> dict[str, bool]:
    """Run the scanner's fallback rule in a subprocess and return each decision."""
    setup = MCP_SCANNER_LITELLM_STANDIN + (MCP_SCANNER_DISCOVER_FALLBACK_PATCH if patched else "")
    probe = (
        "import json\n"
        "from mcp.shared.exceptions import MCPError\n"
        "from mcpscanner.core.scanner import Scanner\n"
        f"cases = {CASES!r}\n"
        "out = {}\n"
        "for label, code, message, legacy_http, _ in cases:\n"
        "    error = MCPError(code=code, message=message)\n"
        "    out[label] = Scanner._should_fallback_to_initialize(error, legacy_http=legacy_http)\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-owned source
        [sys.executable, "-c", setup + probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.unit
def test_unpatched_scanner_refuses_mcp_1x_fallback():
    """Documents the upstream bug the patch exists for. If this starts failing,
    mcp-scanner fixed it and the patch can be removed:
    https://github.com/cisco-ai-defense/mcp-scanner/issues/302"""
    decisions = _decisions(patched=False)

    assert decisions["mcp 1.x server rejects the 2026 header"] is False


@pytest.mark.unit
@pytest.mark.parametrize(("label", "code", "message", "legacy_http", "expected"), CASES)
def test_patched_fallback_follows_the_2026_spec(label, code, message, legacy_http, expected):
    """Any non-2026 discover error falls back to initialize; 2026 errors do not."""
    decisions = _decisions(patched=True)

    assert decisions[label] is expected


@pytest.mark.unit
def test_shim_applies_patch_after_litellm_standin_and_before_cli():
    """The patch imports mcpscanner, so the litellm stand-in must already be in place,
    and the CLI must not start a scan before the patch is applied."""
    standin_at = _SCANNER_SHIM.index(MCP_SCANNER_LITELLM_STANDIN)
    patch_at = _SCANNER_SHIM.index(MCP_SCANNER_DISCOVER_FALLBACK_PATCH)
    cli_at = _SCANNER_SHIM.index("from mcpscanner.cli import cli_entry_point")

    assert standin_at < patch_at < cli_at


@pytest.mark.unit
def test_patch_is_a_no_op_when_the_method_is_missing():
    """A future mcp-scanner without the method must still start, unpatched."""
    probe = (
        "import types, sys\n"
        "pkg = types.ModuleType('mcpscanner'); pkg.__path__ = []\n"
        "core = types.ModuleType('mcpscanner.core'); core.__path__ = []\n"
        "fake = types.ModuleType('mcpscanner.core.scanner')\n"
        "class Scanner: pass\n"
        "fake.Scanner = Scanner\n"
        "pkg.core = core; core.scanner = fake\n"
        "sys.modules.update({'mcpscanner': pkg, 'mcpscanner.core': core, 'mcpscanner.core.scanner': fake})\n"
        + MCP_SCANNER_DISCOVER_FALLBACK_PATCH
        + "print(hasattr(Scanner, '_should_fallback_to_initialize'))\n"
    )
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-owned source
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.stdout.strip() == "False"


@pytest.mark.unit
def test_patch_is_a_no_op_when_mcpscanner_cannot_be_imported():
    """If mcpscanner.core.scanner cannot be imported, the patch must not raise."""
    probe = (
        "import types, sys\n"
        "sys.modules['mcpscanner'] = types.ModuleType('mcpscanner')\n"
        + MCP_SCANNER_DISCOVER_FALLBACK_PATCH
        + "print('started')\n"
    )
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-owned source
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.stdout.strip() == "started"
