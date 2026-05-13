"""Unit tests for registry/auth/access_resolver.py.

Covers:
- resolve_scope_access wildcard, restricted, multi-scope-union, and
  fail-closed semantics.
- UserAccess dataclass shape.
- get_user_accessible_servers / get_user_accessible_tools thin wrappers.
- audit_legacy_scopes_on_startup warning emission.

Mapped to testing.md section 2.5 (user_context shape) and LLD Step 1 / 9.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from registry.auth.access_resolver import (
    UserAccess,
    audit_legacy_scopes_on_startup,
    get_user_accessible_servers,
    get_user_accessible_tools,
    resolve_scope_access,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """AsyncMock scope repository configurable per test via side_effect."""
    repo = AsyncMock()
    repo.get_server_scopes = AsyncMock(return_value=[])
    repo.list_scope_names = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def patched_factory(mock_repo):
    """Patch the factory import path used inside access_resolver."""
    with patch(
        "registry.repositories.factory.get_scope_repository",
        return_value=mock_repo,
    ):
        yield mock_repo


# =============================================================================
# resolve_scope_access
# =============================================================================


@pytest.mark.asyncio
async def test_resolve_scope_access_empty_scopes(patched_factory):
    """No scopes returns an empty UserAccess."""
    # Arrange: default patched_factory returns [] from get_server_scopes

    # Act
    access = await resolve_scope_access([])

    # Assert
    assert isinstance(access, UserAccess)
    assert access.servers == []
    assert access.tools == {}


@pytest.mark.asyncio
async def test_resolve_scope_access_admin_star_server_and_tools(patched_factory):
    """server='*' with tools=['all'] produces the cross-server wildcard sentinel."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "*", "methods": ["all"], "tools": ["all"]}
    ]

    # Act
    access = await resolve_scope_access(["tla-admin"])

    # Assert
    assert access.servers == ["*"]
    assert access.tools == {"*": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_admin_all_string_server(patched_factory):
    """server='all' behaves identically to server='*'."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "all", "methods": ["all"], "tools": ["all"]}
    ]

    # Act
    access = await resolve_scope_access(["tla-admin-all"])

    # Assert
    assert access.servers == ["*"]
    assert access.tools == {"*": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_restricted_single_scope(patched_factory):
    """A restricted scope yields a per-server allowlist."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {
            "server": "current_time",
            "methods": ["tools/call", "tools/list"],
            "tools": ["current_time_by_timezone"],
        }
    ]

    # Act
    access = await resolve_scope_access(["tla-consumer-restricted"])

    # Assert
    assert access.servers == ["current_time"]
    assert access.tools == {"current_time": {"current_time_by_timezone"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_per_server_wildcard_star(patched_factory):
    """tools=['*'] yields the per-server wildcard sentinel."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["all"], "tools": ["*"]}
    ]

    # Act
    access = await resolve_scope_access(["tla-consumer-wildcard"])

    # Assert
    assert access.servers == ["current_time"]
    assert access.tools == {"current_time": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_per_server_wildcard_all_string(patched_factory):
    """tools=['all'] behaves identically to tools=['*']."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["all"], "tools": ["all"]}
    ]

    # Act
    access = await resolve_scope_access(["tla-consumer-all-string"])

    # Assert
    assert access.tools == {"current_time": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_union_of_two_scopes(patched_factory):
    """Two restricted scopes on the same server produce the union of tools."""

    # Arrange
    async def _get_server_scopes(scope: str):
        if scope == "scope-a":
            return [
                {
                    "server": "current_time",
                    "methods": ["tools/call"],
                    "tools": ["current_time_by_timezone"],
                }
            ]
        if scope == "scope-b":
            return [
                {
                    "server": "current_time",
                    "methods": ["tools/call"],
                    "tools": ["current_time_utc"],
                }
            ]
        return []

    patched_factory.get_server_scopes.side_effect = _get_server_scopes

    # Act
    access = await resolve_scope_access(["scope-a", "scope-b"])

    # Assert
    assert access.tools == {
        "current_time": {"current_time_by_timezone", "current_time_utc"}
    }


@pytest.mark.asyncio
async def test_resolve_scope_access_wildcard_wins_over_restricted(patched_factory):
    """Wildcard on any scope is sticky across later restricted scopes."""

    # Arrange
    async def _get_server_scopes(scope: str):
        if scope == "restricted":
            return [
                {
                    "server": "current_time",
                    "methods": ["tools/call"],
                    "tools": ["current_time_by_timezone"],
                }
            ]
        if scope == "wildcard":
            return [{"server": "current_time", "methods": ["all"], "tools": ["*"]}]
        return []

    patched_factory.get_server_scopes.side_effect = _get_server_scopes

    # Act: both orderings must collapse to {"*"}
    access_a = await resolve_scope_access(["restricted", "wildcard"])
    access_b = await resolve_scope_access(["wildcard", "restricted"])

    # Assert
    assert access_a.tools == {"current_time": {"*"}}
    assert access_b.tools == {"current_time": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_missing_tools_field_fails_closed(patched_factory):
    """A rule without a `tools` key records an empty set (fail closed)."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["tools/call"]}
    ]

    # Act
    access = await resolve_scope_access(["tla-consumer-legacy"])

    # Assert
    assert access.servers == ["current_time"]
    assert access.tools == {"current_time": set()}


@pytest.mark.asyncio
async def test_resolve_scope_access_empty_tools_list_fails_closed(patched_factory):
    """tools=[] yields an empty set, not wildcard."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["tools/list"], "tools": []}
    ]

    # Act
    access = await resolve_scope_access(["tla-consumer-empty"])

    # Assert
    assert access.servers == ["current_time"]
    assert access.tools == {"current_time": set()}


@pytest.mark.asyncio
async def test_resolve_scope_access_non_list_non_string_tools_ignored(patched_factory):
    """A `tools` value that is neither a string nor a list (e.g. dict) is
    treated as no allowlist; the server is still added with an empty set
    so subsequent rules can still merge into it.
    """
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["tools/call"], "tools": {"oops": True}}
    ]

    # Act
    access = await resolve_scope_access(["bad-scope"])

    # Assert
    assert access.servers == ["current_time"]
    assert access.tools == {"current_time": set()}


@pytest.mark.asyncio
async def test_resolve_scope_access_scope_repo_error_returns_empty(patched_factory):
    """get_server_scopes raising leaves the user with no access for that scope."""
    # Arrange
    patched_factory.get_server_scopes.side_effect = RuntimeError("db down")

    # Act
    access = await resolve_scope_access(["any-scope"])

    # Assert
    assert access.servers == []
    assert access.tools == {}


# =============================================================================
# Thin wrappers
# =============================================================================


@pytest.mark.asyncio
async def test_get_user_accessible_servers_thin_wrapper(patched_factory):
    """get_user_accessible_servers returns the servers list from the resolver."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["all"], "tools": ["*"]}
    ]

    # Act
    servers = await get_user_accessible_servers(["tla-consumer-wildcard"])

    # Assert
    assert servers == ["current_time"]


@pytest.mark.asyncio
async def test_get_user_accessible_tools_thin_wrapper(patched_factory):
    """get_user_accessible_tools returns the tools map from the resolver."""
    # Arrange
    patched_factory.get_server_scopes.return_value = [
        {
            "server": "current_time",
            "methods": ["tools/call"],
            "tools": ["current_time_by_timezone"],
        }
    ]

    # Act
    tools = await get_user_accessible_tools(["tla-consumer-restricted"])

    # Assert
    assert tools == {"current_time": {"current_time_by_timezone"}}


# =============================================================================
# audit_legacy_scopes_on_startup
# =============================================================================


@pytest.mark.asyncio
async def test_audit_legacy_scopes_no_warnings_on_clean_deployment(
    patched_factory, caplog
):
    """A deployment with only well-formed scopes emits zero warnings."""
    # Arrange
    patched_factory.list_scope_names.return_value = ["scope-a", "scope-b"]

    async def _get(scope: str):
        if scope == "scope-a":
            return [{"server": "current_time", "methods": ["all"], "tools": ["all"]}]
        if scope == "scope-b":
            return [
                {
                    "server": "current_time",
                    "methods": ["tools/call"],
                    "tools": ["current_time_by_timezone"],
                }
            ]
        return []

    patched_factory.get_server_scopes.side_effect = _get

    # Act
    with caplog.at_level(logging.INFO, logger="registry.auth.access_resolver"):
        count = await audit_legacy_scopes_on_startup()

    # Assert
    assert count == 0
    assert any("no issues found" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_audit_legacy_scopes_warns_on_missing_tools_key(
    patched_factory, caplog
):
    """A scope row missing the `tools` key triggers a WARN line."""
    # Arrange
    patched_factory.list_scope_names.return_value = ["tla-consumer-legacy"]
    patched_factory.get_server_scopes.return_value = [
        {"server": "current_time", "methods": ["tools/list", "tools/call"]}
    ]

    # Act
    with caplog.at_level(logging.WARNING, logger="registry.auth.access_resolver"):
        count = await audit_legacy_scopes_on_startup()

    # Assert
    assert count == 1
    warn_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("legacy_scope_missing_tools" in msg for msg in warn_messages)


@pytest.mark.asyncio
async def test_audit_legacy_scopes_warns_on_empty_tools_with_call_method(
    patched_factory, caplog
):
    """tools=[] combined with a call-capable method triggers a WARN."""
    # Arrange
    patched_factory.list_scope_names.return_value = ["tla-consumer-empty"]
    patched_factory.get_server_scopes.return_value = [
        {
            "server": "current_time",
            "methods": ["tools/call"],
            "tools": [],
        }
    ]

    # Act
    with caplog.at_level(logging.WARNING, logger="registry.auth.access_resolver"):
        count = await audit_legacy_scopes_on_startup()

    # Assert
    assert count == 1
    warn_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("empty_tools_list_with_call_method" in msg for msg in warn_messages)


@pytest.mark.asyncio
async def test_audit_legacy_scopes_handles_repo_error(patched_factory, caplog):
    """list_scope_names raising is logged and the function returns 0 without raising."""
    # Arrange
    patched_factory.list_scope_names.side_effect = RuntimeError("boom")

    # Act
    with caplog.at_level(logging.ERROR, logger="registry.auth.access_resolver"):
        count = await audit_legacy_scopes_on_startup()

    # Assert
    assert count == 0
    error_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any("list_scope_names failed" in msg for msg in error_messages)


# =============================================================================
# Regression: DocumentDB rows where `tools` is a bare string instead of a list
# (Issue #1026 — UI submits "*" rather than ["*"] when the user picks "All
# tools"; auth_server's validate_server_tool_access already tolerates this
# via Python's substring `in` check, so the resolver must agree).
# =============================================================================


@pytest.mark.asyncio
async def test_resolve_scope_access_tools_wildcard_as_bare_string(monkeypatch):
    """tools: "*" (string, not list) is treated as wildcard."""
    # Arrange
    mock_repo = AsyncMock()
    mock_repo.get_server_scopes.return_value = [
        {"server": "currenttime", "methods": ["all"], "tools": "*"},
    ]
    monkeypatch.setattr(
        "registry.repositories.factory.get_scope_repository", lambda: mock_repo
    )

    # Act
    access = await resolve_scope_access(["public-mcp-users"])

    # Assert
    assert access.servers == ["currenttime"]
    assert access.tools == {"currenttime": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_tools_all_as_bare_string(monkeypatch):
    """tools: "all" (string) is treated as wildcard same as ["all"]."""
    # Arrange
    mock_repo = AsyncMock()
    mock_repo.get_server_scopes.return_value = [
        {"server": "currenttime", "methods": ["all"], "tools": "all"},
    ]
    monkeypatch.setattr(
        "registry.repositories.factory.get_scope_repository", lambda: mock_repo
    )

    # Act
    access = await resolve_scope_access(["s"])

    # Assert
    assert access.tools == {"currenttime": {"*"}}


@pytest.mark.asyncio
async def test_resolve_scope_access_tools_single_tool_as_bare_string(monkeypatch):
    """tools: "foo" (string) is treated as a single-element allowlist."""
    # Arrange
    mock_repo = AsyncMock()
    mock_repo.get_server_scopes.return_value = [
        {"server": "currenttime", "methods": ["tools/call"], "tools": "current_time_by_timezone"},
    ]
    monkeypatch.setattr(
        "registry.repositories.factory.get_scope_repository", lambda: mock_repo
    )

    # Act
    access = await resolve_scope_access(["s"])

    # Assert
    assert access.tools == {"currenttime": {"current_time_by_timezone"}}
