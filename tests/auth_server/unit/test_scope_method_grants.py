"""Tests for which MCP methods a scope grants in validate_server_tool_access.

Covers the wildcard-tools fallback (a tools list of "*" must not grant every
METHOD), the methods any access to a server allows (server/discover and client
notifications), and the URL-derived transport names a body-less request is
authorized as.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from registry.services.scope_service import STANDARD_METHODS

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
UNRESTRICTED_SEED_FILES: tuple[str, ...] = (
    "scripts/mcp-servers-unrestricted-read.json",
    "scripts/mcp-servers-unrestricted-execute.json",
)
NEW_DEFAULT_METHODS: tuple[str, ...] = ("resources/read", "prompts/list", "prompts/get")

SCOPES: dict[str, list[dict]] = {
    "wildcard-tools": [
        {"server": "demo", "methods": ["initialize", "tools/list", "tools/call"], "tools": ["*"]}
    ],
    "explicit-tools": [
        {"server": "demo", "methods": ["initialize", "tools/list"], "tools": ["search"]}
    ],
    "method-in-tools": [{"server": "demo", "methods": ["initialize"], "tools": ["resources/read"]}],
    "string-tools": [
        {"server": "demo", "methods": ["initialize"], "tools": "resources/read-and-more"}
    ],
    "no-methods": [{"server": "demo", "methods": [], "tools": []}],
    "other-server": [{"server": "elsewhere", "methods": ["*"], "tools": ["*"]}],
}


async def _check(
    method: str,
    scope: str,
    tool_name: str | None = None,
    is_http_verb: bool = False,
) -> bool:
    """Run validate_server_tool_access against the SCOPES fixture."""
    from auth_server.server import validate_server_tool_access

    repo = AsyncMock()
    repo.get_server_scopes = AsyncMock(side_effect=lambda name: SCOPES.get(name))
    with (
        patch("auth_server.server.get_scope_repository", return_value=repo),
        patch("auth_server.server._get_blocked_tools", AsyncMock(return_value=set())),
    ):
        return await validate_server_tool_access(
            "demo", method, tool_name, [scope], is_http_verb=is_http_verb
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    ["resources/read", "prompts/get", "logging/setLevel", "completion/complete", "tasks/get"],
)
async def test_wildcard_tools_do_not_grant_unlisted_methods(method):
    """tools: ["*"] grants every tool for tools/call, not every method."""
    assert await _check(method, "wildcard-tools") is False


@pytest.mark.asyncio
async def test_listed_methods_and_wildcard_tool_calls_still_work():
    assert await _check("tools/list", "wildcard-tools") is True
    assert await _check("tools/call", "wildcard-tools", tool_name="anything") is True


@pytest.mark.asyncio
async def test_nameless_tools_call_is_denied():
    """A tools/call without a tool name is malformed; nothing grants it."""
    assert await _check("tools/call", "wildcard-tools", tool_name=None) is False


@pytest.mark.asyncio
async def test_method_named_in_tools_list_keeps_working():
    """Backward compatibility: a method listed by name in tools is granted."""
    assert await _check("resources/read", "method-in-tools") is True


@pytest.mark.asyncio
async def test_string_tools_value_never_grants_by_substring():
    """A tools value stored as a string must not grant a method it merely contains."""
    assert await _check("resources/read", "string-tools") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["server/discover", "notifications/cancelled", "notifications/initialized"]
)
async def test_any_server_access_grants_discover_and_notifications(method):
    assert await _check(method, "no-methods") is True
    assert await _check(method, "explicit-tools") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["server/discover", "notifications/cancelled"])
async def test_no_server_access_still_denies_discover_and_notifications(method):
    assert await _check(method, "other-server") is False


@pytest.mark.asyncio
async def test_server_access_rule_never_applies_to_http_verbs():
    assert await _check("server/discover", "no-methods", is_http_verb=True) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["sse", "mcp", "messages"])
async def test_transport_open_keeps_its_wildcard_tools_grant(endpoint):
    """A body-less request (opening an HTTP+SSE stream) is authorized as its URL
    tail. Wildcard-tool scopes allowed it before, and still do; others do not."""
    assert await _check(endpoint, "wildcard-tools") is True
    assert await _check(endpoint, "explicit-tools") is False


def test_standard_methods_include_new_defaults():
    for method in NEW_DEFAULT_METHODS:
        assert method in STANDARD_METHODS


@pytest.mark.parametrize("seed_file", UNRESTRICTED_SEED_FILES)
def test_unrestricted_seed_scopes_grant_new_defaults(seed_file):
    """The unrestricted scopes use tools "*", so after the fallback fix they need
    the new methods listed explicitly."""
    data = json.loads((REPO_ROOT / seed_file).read_text())
    mcp_entries = [
        entry
        for entry in data.get("server_access", [])
        if isinstance(entry, dict) and "tools/list" in entry.get("methods", [])
    ]
    assert mcp_entries, f"no MCP server entry found in {seed_file}"
    for entry in mcp_entries:
        for method in NEW_DEFAULT_METHODS:
            assert method in entry["methods"], f"{seed_file} entry {entry['server']} lacks {method}"
