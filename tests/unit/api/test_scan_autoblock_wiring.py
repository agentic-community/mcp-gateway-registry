"""The scan-completion path must act on SECURITY_ALLOW_UNSAFE_SERVERS.

PR #1719 landed `reconcile_security_blocks` and the `allow_unsafe_servers` flag
without connecting either to the scan path, so the whole feature was unreachable:
the flag had no reader and the reconcile function had no production caller. These
tests pin the wiring so it cannot quietly come loose again.

The case worth reading twice is `test_flag_on_but_no_tool_blamed_still_disables`.
A scan can mark a server unsafe on a server-level finding that blames no single
tool, and `reconcile_security_blocks` then returns an empty map. Leaving the
server enabled there would turn the opt-in into a silent bypass of
`block_unsafe_servers`, so the code falls back to disabling the whole server.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.api.server_routes import _perform_security_scan_on_registration
from registry.schemas.security import SecurityScanConfig, SecurityScanResult

pytestmark = [pytest.mark.unit, pytest.mark.api, pytest.mark.servers]

SERVER_PATH = "/unsafe-server"


def _scan_config(
    allow_unsafe: bool,
    block_unsafe: bool = True,
) -> SecurityScanConfig:
    """A scan config with scanning on, varying only the two flags under test."""
    return SecurityScanConfig(
        enabled=True,
        scan_on_registration=True,
        block_unsafe_servers=block_unsafe,
        allow_unsafe_servers=allow_unsafe,
        add_security_pending_tag=False,
    )


def _failed_scan(tool_name: str | None = "bad_tool") -> SecurityScanResult:
    """An unsafe scan result, optionally blaming one named tool.

    With tool_name=None the scan is unsafe but attributes nothing to a tool,
    which is the server-level finding case.
    """
    raw: dict = {"tool_results": []}
    if tool_name:
        raw["tool_results"].append(
            {
                "tool_name": tool_name,
                "findings": {
                    "yara_analyzer": {
                        "severity": "HIGH",
                        "threat_names": ["PROMPT INJECTION"],
                    }
                },
            }
        )
    return SecurityScanResult(
        server_url="https://example.com/mcp",
        server_path=SERVER_PATH,
        scan_timestamp="2026-09-21T00:00:00Z",
        is_safe=False,
        critical_issues=0,
        high_severity=1,
        raw_output=raw,
    )


async def _run(
    scan_config: SecurityScanConfig,
    scan_result: SecurityScanResult,
    reconcile_returns: dict,
) -> dict:
    """Drive the scan path once and report what it did.

    Returns a dict with `disabled` (whether the server was turned off) and
    `reconciled` (whether the per-tool reconcile ran).
    """
    scanner = AsyncMock()
    scanner.scan_server = AsyncMock(return_value=scan_result)
    scanner.get_scan_config = lambda: scan_config

    svc = AsyncMock()
    svc.reconcile_security_blocks = AsyncMock(return_value=reconcile_returns)

    with (
        patch("registry.api.server_routes.security_scanner_service", scanner),
        patch("registry.api.server_routes.server_service", svc),
        patch("registry.api.server_routes._disable_server_for_security", AsyncMock()) as disable,
        patch("registry.api.server_routes.fire_scan_complete_event"),
    ):
        await _perform_security_scan_on_registration(
            SERVER_PATH,
            "https://example.com/mcp",
            {"server_name": "Unsafe", "path": SERVER_PATH},
        )
        return {
            "disabled": disable.await_count == 1,
            "reconciled": svc.reconcile_security_blocks.await_count == 1,
        }


class TestScanAutoBlockWiring:
    """The scan path honours allow_unsafe_servers, and fails closed."""

    async def test_flag_off_disables_whole_server(self):
        """Default behaviour is unchanged: no reconcile, server goes down."""
        out = await _run(_scan_config(allow_unsafe=False), _failed_scan(), {})

        assert out["disabled"] is True
        assert out["reconciled"] is False

    async def test_flag_on_blocks_tools_and_keeps_server_up(self):
        """The opt-in path blocks the flagged tool and leaves the server enabled."""
        blocked = {"bad_tool": {"blocked": True, "source": "security_scan"}}

        out = await _run(_scan_config(allow_unsafe=True), _failed_scan(), blocked)

        assert out["reconciled"] is True
        assert out["disabled"] is False

    async def test_flag_on_but_no_tool_blamed_still_disables(self):
        """Fail closed when the reconcile blocks nothing.

        An unsafe server with no blockable tool must not stay enabled, or the
        opt-in becomes a way to bypass block_unsafe_servers entirely.
        """
        out = await _run(_scan_config(allow_unsafe=True), _failed_scan(tool_name=None), {})

        assert out["reconciled"] is True
        assert out["disabled"] is True

    async def test_block_unsafe_off_disables_nothing(self):
        """allow_unsafe_servers has no effect while block_unsafe_servers is off."""
        out = await _run(
            _scan_config(allow_unsafe=True, block_unsafe=False),
            _failed_scan(),
            {"bad_tool": {"blocked": True}},
        )

        assert out["disabled"] is False
        assert out["reconciled"] is False
