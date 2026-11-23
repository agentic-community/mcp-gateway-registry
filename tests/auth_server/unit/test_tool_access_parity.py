"""The tool access listing and the gateway give the same answer (issue #2102).

For every rule shape below, the listing's "can call" and "can open a session"
checks must agree with validate_server_tool_access, the function /validate
uses for every MCP call. If the gateway's matching changes, this test fails
until the listing changes with it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.services.tool_access_service import (
    GroupInfo,
    _call_check,
    _connect_check,
    _group_context,
    _index_rules,
    _rules_for_server,
)

SERVERS: list[str] = ["/docs/", "/virtual/eng", "/other/"]
TOOLS: list[str] = ["search", "search_docs", "install", "uninstall_all", "*"]

RULE_SHAPES: dict[str, list[dict]] = {
    "named-tool": [{"server": "docs", "methods": ["initialize"], "tools": ["search"]}],
    "tools-wildcard-star": [{"server": "docs", "methods": ["initialize"], "tools": ["*"]}],
    "tools-wildcard-all": [{"server": "docs", "methods": [], "tools": ["all"]}],
    "methods-wildcard-no-tools": [{"server": "docs", "methods": ["all"], "tools": []}],
    "tools-call-method-only": [{"server": "docs", "methods": ["tools/call"], "tools": []}],
    "server-star": [{"server": "*", "methods": ["initialize"], "tools": ["search"]}],
    "server-all-legacy": [{"server": "all", "methods": ["all"], "tools": ["all"]}],
    "leading-slash": [{"server": "/docs/", "methods": ["initialize"], "tools": ["search_docs"]}],
    "missing-tools-key": [{"server": "docs", "methods": ["initialize"]}],
    "tools-none": [{"server": "docs", "methods": ["initialize"], "tools": None}],
    "bare-string-star": [{"server": "docs", "methods": ["initialize"], "tools": "*"}],
    "bare-string-name": [{"server": "docs", "methods": ["initialize"], "tools": "search_docs"}],
    "bare-string-contains-all": [{"server": "docs", "methods": [], "tools": "uninstall_all"}],
    "bare-string-methods": [{"server": "docs", "methods": "initialize", "tools": []}],
    "initialize-in-tools": [{"server": "docs", "methods": [], "tools": ["initialize"]}],
    "virtual-path": [{"server": "virtual/eng", "methods": ["initialize"], "tools": ["search"]}],
    "two-rules": [
        {"server": "docs", "methods": ["initialize"], "tools": []},
        {"server": "docs", "methods": [], "tools": ["install"]},
    ],
    "other-server": [{"server": "other", "methods": ["all"], "tools": ["*"]}],
    "empty": [],
}


def _group(
    name: str,
) -> GroupInfo:
    """Build a GroupInfo for one rule shape, with no UI permissions."""
    rules = RULE_SHAPES[name]
    context = _group_context(name, {name: rules}, {})
    return GroupInfo(
        name=name,
        idp_groups=[],
        rules=rules,
        context=context,
        is_admin=False,
        rules_by_server=_index_rules(rules),
    )


async def _gateway_allows(
    shape: str,
    server_path: str,
    method: str,
    tool: str,
) -> bool:
    """Ask the gateway's own check, with this shape as the user's only scope."""
    from auth_server.server import validate_server_tool_access

    repo = AsyncMock()
    repo.get_server_scopes = AsyncMock(return_value=RULE_SHAPES[shape])
    with (
        patch("auth_server.server.get_scope_repository", return_value=repo),
        patch("auth_server.server._get_blocked_tools", new=AsyncMock(return_value=set())),
    ):
        return await validate_server_tool_access(server_path.strip("/"), method, tool, [shape])


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(RULE_SHAPES))
async def test_call_check_matches_gateway(
    shape: str,
) -> None:
    group = _group(shape)
    for server_path in SERVERS:
        server_rules = _rules_for_server([group], server_path)
        for tool in TOOLS:
            listing = _call_check(server_rules, tool)(group)
            gateway = await _gateway_allows(shape, server_path, "tools/call", tool)
            assert listing == gateway, (shape, server_path, tool, listing, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(RULE_SHAPES))
async def test_connect_check_matches_gateway(
    shape: str,
) -> None:
    group = _group(shape)
    for server_path in SERVERS:
        listing = _connect_check(_rules_for_server([group], server_path))(group)
        gateway = await _gateway_allows(shape, server_path, "initialize", "")
        assert listing == gateway, (shape, server_path, listing, gateway)


@pytest.mark.asyncio
async def test_bare_string_is_an_exact_name() -> None:
    """A string tools value names one tool; it is never a substring test."""
    assert await _gateway_allows("bare-string-name", "/docs/", "tools/call", "search_docs")
    assert not await _gateway_allows("bare-string-name", "/docs/", "tools/call", "search")
    assert not await _gateway_allows("bare-string-contains-all", "/docs/", "tools/call", "search")
