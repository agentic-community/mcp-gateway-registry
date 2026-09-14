"""Unit tests for the generic-hop verb->scope authorization.

These lock in the privilege-escalation guard and the HTTP-verb matching semantics
of validate_server_tool_access(..., is_http_verb=True):

- the legacy MCP "*"/"all" methods wildcard does NOT authorize an HTTP verb (a
  pre-existing methods:["*"] scope must not start granting DELETE when its entity
  is flipped is_proxied) — only explicit enumeration or the distinct "http:*"
  token grants;
- HTTP verbs match the methods list case-INSENSITIVELY (a scope authored "get"
  authorizes GET), while MCP tokens stay case-sensitive;
- an HTTP verb never falls through to the tools list (so tools:["*"] can't grant
  a verb either).
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from auth_server.server import validate_server_tool_access  # noqa: E402

pytestmark = pytest.mark.unit


def _scope_repo(server_access_by_scope: dict[str, list[dict]]):
    """Mock scope repo whose get_server_scopes returns the given rule lists."""
    repo = AsyncMock()

    async def _get(scope_name: str):
        return server_access_by_scope.get(scope_name, [])

    repo.get_server_scopes.side_effect = _get
    return repo


async def _access(server_access, method, scopes, is_http_verb, tool_name=None):
    repo = _scope_repo(server_access)
    with patch("auth_server.server.get_scope_repository", return_value=repo):
        return await validate_server_tool_access(
            "skill/skills/proxy-demo", method, tool_name, scopes, is_http_verb=is_http_verb
        )


class TestHttpVerbWildcardSplit:
    async def test_mcp_star_wildcard_does_not_grant_http_verb(self):
        """Privilege escalation on upgrade: methods:["*"] means all MCP
        methods, NOT all HTTP verbs. A DELETE must be denied."""
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["*"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is False

    async def test_mcp_all_wildcard_does_not_grant_http_verb(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["all"]}]}
        assert await _access(access, "PUT", ["s1"], is_http_verb=True) is False

    async def test_http_wildcard_token_grants_verb(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["http:*"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is True

    async def test_explicit_verb_enumeration_grants(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["GET", "HEAD"]}]}
        assert await _access(access, "GET", ["s1"], is_http_verb=True) is True

    async def test_verb_not_enumerated_denied(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["GET", "HEAD"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is False

    async def test_read_only_scope_cannot_delete(self):
        access = {"ro": [{"server": "skill/skills/proxy-demo", "methods": ["GET"]}]}
        assert await _access(access, "DELETE", ["ro"], is_http_verb=True) is False


class TestHttpVerbCaseInsensitive:
    async def test_lowercase_stored_verb_authorizes_uppercase_request(self):
        # A scope authored "get" (lower) must authorize a GET request — the read-
        # side canonicalization backstop covers pre-existing / any-write-path data.
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["get", "head"]}]}
        assert await _access(access, "GET", ["s1"], is_http_verb=True) is True

    async def test_mixed_case_stored_verb(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["Delete"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is True


class TestHttpVerbDoesNotFallThroughToTools:
    async def test_tools_wildcard_does_not_grant_verb(self):
        # An HTTP verb must match methods ONLY; tools:["*"] must not authorize it.
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": [], "tools": ["*"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is False

    async def test_verb_in_tools_list_does_not_grant(self):
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": [], "tools": ["DELETE"]}]}
        assert await _access(access, "DELETE", ["s1"], is_http_verb=True) is False


class TestMcpPathUnaffected:
    async def test_mcp_star_still_grants_mcp_method(self):
        # Regression guard: the split must NOT change MCP behavior — "*" still
        # grants tools/list for a normal (non-generic) MCP request.
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["*"]}]}
        assert await _access(access, "tools/list", ["s1"], is_http_verb=False) is True

    async def test_mcp_method_case_sensitive(self):
        # MCP tokens stay case-sensitive: "TOOLS/LIST" must NOT match "tools/list".
        access = {"s1": [{"server": "skill/skills/proxy-demo", "methods": ["tools/list"]}]}
        assert await _access(access, "TOOLS/LIST", ["s1"], is_http_verb=False) is False
