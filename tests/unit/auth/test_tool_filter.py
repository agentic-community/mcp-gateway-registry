"""Unit tests for registry/auth/tool_filter.py.

Covers:
- filter_tools_for_user admin / wildcard / restricted / fail-closed paths.
- tool_allowed_for_user boolean wrapper.
- _tool_identity normalization across `name` and `tool_name` shapes.
- _is_admin_or_cross_server_wildcard bypass rules.
- audit emission on prune and graceful handling of sink failure.

Mapped to testing.md sections 1.1.2, 1.1.4, 1.1.5, 1.1.18 and LLD Step 3.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from registry.auth.tool_filter import (
    _is_admin_or_cross_server_wildcard,
    _tool_identity,
    filter_tools_for_user,
    tool_allowed_for_user,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def restricted_user_context() -> dict[str, Any]:
    """User context for a restricted consumer of current_time."""
    return {
        "username": "alice",
        "is_admin": False,
        "scopes": ["tla-consumer-restricted"],
        "accessible_servers": ["current_time"],
        "accessible_tools": {"current_time": {"current_time_by_timezone"}},
    }


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Admin user context with is_admin True."""
    return {
        "username": "admin",
        "is_admin": True,
        "scopes": ["tla-admin"],
        "accessible_servers": ["*"],
        "accessible_tools": {"*": {"*"}},
    }


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Three tool dicts using the `name` key (registry tool_list shape)."""
    return [
        {"name": "current_time_by_timezone", "description": "tz"},
        {"name": "current_time_utc", "description": "utc"},
        {"name": "current_time_epoch", "description": "epoch"},
    ]


# =============================================================================
# _tool_identity
# =============================================================================


def test_tool_identity_uses_name_key():
    """`name` key is used when present."""
    assert _tool_identity({"name": "foo"}) == "foo"


def test_tool_identity_falls_back_to_tool_name():
    """`tool_name` key is used when `name` is absent."""
    assert _tool_identity({"tool_name": "bar"}) == "bar"


def test_tool_identity_prefers_name_over_tool_name():
    """When both keys are present, `name` wins."""
    assert _tool_identity({"name": "foo", "tool_name": "bar"}) == "foo"


def test_tool_identity_returns_none_for_non_dict():
    """Non-dict inputs yield None."""
    assert _tool_identity("not a dict") is None


# =============================================================================
# _is_admin_or_cross_server_wildcard
# =============================================================================


def test_bypass_admin_flag():
    """is_admin=True bypasses filtering."""
    assert _is_admin_or_cross_server_wildcard({"is_admin": True}) is True


def test_bypass_star_in_accessible_servers():
    """accessible_servers containing '*' bypasses filtering."""
    ctx = {"is_admin": False, "accessible_servers": ["*"]}
    assert _is_admin_or_cross_server_wildcard(ctx) is True


def test_bypass_all_string_in_accessible_servers():
    """accessible_servers containing 'all' bypasses filtering."""
    ctx = {"is_admin": False, "accessible_servers": ["all"]}
    assert _is_admin_or_cross_server_wildcard(ctx) is True


def test_bypass_star_key_in_accessible_tools():
    """accessible_tools['*'] == {'*'} bypasses filtering."""
    ctx = {"is_admin": False, "accessible_servers": [], "accessible_tools": {"*": {"*"}}}
    assert _is_admin_or_cross_server_wildcard(ctx) is True


def test_no_bypass_for_restricted_user():
    """A restricted user with concrete server list does not bypass."""
    ctx = {
        "is_admin": False,
        "accessible_servers": ["current_time"],
        "accessible_tools": {"current_time": {"current_time_by_timezone"}},
    }
    assert _is_admin_or_cross_server_wildcard(ctx) is False


# =============================================================================
# filter_tools_for_user - bypass paths
# =============================================================================


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_admin_bypass(mock_audit, admin_user_context, sample_tools):
    """Admin user_context returns tools unchanged without auditing."""
    # Act
    result = filter_tools_for_user("current_time", sample_tools, admin_user_context)

    # Assert
    assert result == sample_tools
    mock_audit.assert_not_called()


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_cross_server_wildcard_star(mock_audit, sample_tools):
    """accessible_servers=['*'] bypasses filtering."""
    # Arrange
    ctx = {"is_admin": False, "accessible_servers": ["*"], "accessible_tools": {}}

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == sample_tools
    mock_audit.assert_not_called()


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_cross_server_wildcard_all_string(mock_audit, sample_tools):
    """accessible_servers=['all'] bypasses filtering."""
    # Arrange
    ctx = {"is_admin": False, "accessible_servers": ["all"], "accessible_tools": {}}

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == sample_tools
    mock_audit.assert_not_called()


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_accessible_tools_star_key(mock_audit, sample_tools):
    """accessible_tools={'*': {'*'}} bypasses filtering."""
    # Arrange
    ctx = {
        "is_admin": False,
        "accessible_servers": [],
        "accessible_tools": {"*": {"*"}},
    }

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == sample_tools
    mock_audit.assert_not_called()


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_per_server_wildcard_sentinel(mock_audit, sample_tools):
    """accessible_tools[server]={'*'} returns the full list untouched."""
    # Arrange
    ctx = {
        "is_admin": False,
        "accessible_servers": ["current_time"],
        "accessible_tools": {"current_time": {"*"}},
    }

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == sample_tools
    mock_audit.assert_not_called()


# =============================================================================
# filter_tools_for_user - restricted paths
# =============================================================================


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_restricted_keeps_only_allowed_name(
    mock_audit, restricted_user_context, sample_tools
):
    """A restricted allowlist keeps only matching `name` entries."""
    # Act
    result = filter_tools_for_user("current_time", sample_tools, restricted_user_context)

    # Assert
    assert result == [{"name": "current_time_by_timezone", "description": "tz"}]


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_restricted_handles_tool_name_alias(mock_audit, restricted_user_context):
    """Semantic search shape (`tool_name`) is normalized by _tool_identity."""
    # Arrange: semantic search `matching_tools` entries use tool_name, not name.
    search_tools = [
        {"tool_name": "current_time_by_timezone", "description": "tz"},
        {"tool_name": "current_time_utc", "description": "utc"},
    ]

    # Act
    result = filter_tools_for_user("current_time", search_tools, restricted_user_context)

    # Assert
    assert result == [{"tool_name": "current_time_by_timezone", "description": "tz"}]


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_restricted_prefers_name_over_tool_name(mock_audit, restricted_user_context):
    """When both keys are present, the tool name from `name` is used."""
    # Arrange: `name` points to the allowed tool, `tool_name` to a disallowed one.
    tools = [
        {"name": "current_time_by_timezone", "tool_name": "current_time_utc"},
    ]

    # Act
    result = filter_tools_for_user("current_time", tools, restricted_user_context)

    # Assert: kept because `name` matches the allowlist.
    assert result == tools


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_missing_allowlist_returns_empty_list(mock_audit, sample_tools):
    """No entry for the requested server returns [] (fail closed)."""
    # Arrange
    ctx = {
        "username": "bob",
        "is_admin": False,
        "scopes": ["tla-consumer-other"],
        "accessible_servers": ["other_server"],
        "accessible_tools": {"other_server": {"some_tool"}},
    }

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == []


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_empty_allowlist_set_returns_empty_list(mock_audit, sample_tools):
    """An explicitly empty allowlist set returns [] (fail closed)."""
    # Arrange
    ctx = {
        "username": "empty-user",
        "is_admin": False,
        "scopes": ["tla-consumer-empty"],
        "accessible_servers": ["current_time"],
        "accessible_tools": {"current_time": set()},
    }

    # Act
    result = filter_tools_for_user("current_time", sample_tools, ctx)

    # Assert
    assert result == []


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_none_tools_input_returns_empty(mock_audit, restricted_user_context):
    """Passing tools=None yields [] without raising."""
    # Act
    result = filter_tools_for_user("current_time", None, restricted_user_context)

    # Assert
    assert result == []


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_drops_non_dict_entries(mock_audit, restricted_user_context):
    """Non-dict entries are silently dropped during filtering."""
    # Arrange
    tools = [
        {"name": "current_time_by_timezone"},
        "oops not a dict",
        None,
        42,
    ]

    # Act
    result = filter_tools_for_user("current_time", tools, restricted_user_context)

    # Assert
    assert result == [{"name": "current_time_by_timezone"}]


# =============================================================================
# Audit event emission
# =============================================================================


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_emits_audit_event_when_prunes(
    mock_audit, restricted_user_context, sample_tools
):
    """When tools are pruned, the audit emitter is invoked once with the prune details."""
    # Act
    filter_tools_for_user(
        "current_time",
        sample_tools,
        restricted_user_context,
        endpoint="tools_service",
    )

    # Assert
    assert mock_audit.call_count == 1
    kwargs = mock_audit.call_args.kwargs
    assert kwargs["endpoint"] == "tools_service"
    assert kwargs["server_name"] == "current_time"
    assert len(kwargs["kept"]) == 1
    # The pruned entries include the two disallowed tools.
    pruned_names = sorted(t.get("name") for t in kwargs["pruned"])
    assert pruned_names == ["current_time_epoch", "current_time_utc"]


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_does_not_emit_audit_event_when_admin(
    mock_audit, admin_user_context, sample_tools
):
    """Admin bypass path never triggers the audit emitter."""
    # Act
    filter_tools_for_user("current_time", sample_tools, admin_user_context)

    # Assert
    mock_audit.assert_not_called()


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_does_not_emit_audit_event_when_nothing_pruned(mock_audit):
    """When the allowlist covers every tool, no audit is emitted."""
    # Arrange
    ctx = {
        "username": "alice",
        "is_admin": False,
        "scopes": ["tla-consumer-restricted"],
        "accessible_servers": ["current_time"],
        "accessible_tools": {"current_time": {"a", "b"}},
    }
    tools = [{"name": "a"}, {"name": "b"}]

    # Act
    result = filter_tools_for_user("current_time", tools, ctx)

    # Assert
    assert result == tools
    mock_audit.assert_not_called()


def test_filter_audit_failure_does_not_break_request(
    restricted_user_context, sample_tools
):
    """A failing audit sink does not propagate out of filter_tools_for_user.

    Simulates the realistic failure path: the sink module raises while
    emitting an event. The internal try/except in `_emit_tool_filter_audit`
    must swallow that so the request response is unaffected.
    """
    # Arrange: patch the sink so `emit_audit_event` raises when invoked.
    with patch(
        "registry.audit.sink.emit_audit_event",
        side_effect=RuntimeError("audit sink down"),
    ):
        # Act: the filter call must not propagate the sink failure.
        try:
            result = filter_tools_for_user(
                "current_time",
                sample_tools,
                restricted_user_context,
                endpoint="tools_service",
            )
        except RuntimeError:
            pytest.fail("audit failure must not propagate out of filter_tools_for_user")

    # Assert
    assert result == [{"name": "current_time_by_timezone", "description": "tz"}]


# =============================================================================
# tool_allowed_for_user
# =============================================================================


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_tool_allowed_for_user_true(mock_audit, restricted_user_context):
    """Allowed tool returns True."""
    # Act
    allowed = tool_allowed_for_user(
        "current_time", "current_time_by_timezone", restricted_user_context
    )

    # Assert
    assert allowed is True


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_tool_allowed_for_user_false_restricted(mock_audit, restricted_user_context):
    """Disallowed tool returns False."""
    # Act
    allowed = tool_allowed_for_user(
        "current_time", "current_time_utc", restricted_user_context
    )

    # Assert
    assert allowed is False


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_tool_allowed_for_user_admin_bypass(mock_audit, admin_user_context):
    """Admin always sees every tool."""
    # Act
    allowed = tool_allowed_for_user(
        "current_time", "never_declared_tool", admin_user_context
    )

    # Assert
    assert allowed is True


# =============================================================================
# Display-name / technical-name / path normalization (regression, Issue #1026
# Section 1 live test)
# =============================================================================


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_matches_when_called_with_display_name_but_scope_uses_path(mock_audit):
    """
    Regression guard: scope rows are stored keyed on the technical path
    (e.g. "airegistry-tools"), but registry endpoints pass the display
    name ("AI Registry tools") when calling the filter. The lookup must
    normalize slashes and fall through to the server_path candidate so
    the allowlist still matches.
    """
    # Arrange
    ctx = {
        "username": "bob",
        "is_admin": False,
        "scopes": ["tla-consumer-restricted"],
        "accessible_servers": ["airegistry-tools"],
        "accessible_tools": {"airegistry-tools": {"intelligent_tool_finder"}},
    }
    tools = [
        {"name": "intelligent_tool_finder"},
        {"name": "list_services"},
    ]

    # Act - caller passes the display name plus technical server_path
    result = filter_tools_for_user(
        "AI Registry tools",
        tools,
        ctx,
        endpoint="servers",
        server_path="/airegistry-tools/",
    )

    # Assert
    assert [t["name"] for t in result] == ["intelligent_tool_finder"]


@patch("registry.auth.tool_filter._emit_tool_filter_audit")
def test_filter_matches_when_scope_key_has_slashes_but_caller_passes_bare_name(mock_audit):
    """Slash-stripped normalization works the other way round too."""
    # Arrange: scope stored with path form
    ctx = {
        "username": "carol",
        "is_admin": False,
        "accessible_servers": ["/airegistry-tools/"],
        "accessible_tools": {"/airegistry-tools/": {"intelligent_tool_finder"}},
    }
    tools = [{"name": "intelligent_tool_finder"}, {"name": "list_services"}]

    # Act - caller has no path, just the technical name
    result = filter_tools_for_user(
        "airegistry-tools",
        tools,
        ctx,
        endpoint="servers",
    )

    # Assert
    assert [t["name"] for t in result] == ["intelligent_tool_finder"]
