"""Unit tests for auth_server.group_filter.

Covers the login-time IdP group filter:
- Design B (ALLOWED_IDP_GROUPS allowlist) takes precedence when set.
- Design C (auto-derive from scope mappings) is the default.
- Fail-open only when Design C cannot run (repo error / no mappings seeded).
- A successful-but-empty Design C result stores empty (genuinely no-access).
- Authorization is unchanged (scope-equivalence) when dropped groups are
  unmapped.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from auth_server import group_filter


def _make_scope_repo(mapped: set[str]) -> AsyncMock:
    """Build a scope-repo mock whose get_all_mapped_group_names returns mapped."""
    repo = AsyncMock()
    repo.get_all_mapped_group_names.return_value = mapped
    return repo


class TestFilterSessionGroups:
    """Tests for filter_session_groups."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """An empty group list short-circuits to empty."""
        result = await group_filter.filter_session_groups([], username_hash="h")
        assert result == []

    @pytest.mark.asyncio
    async def test_scope_derived_keeps_only_mapped(self):
        """Design C keeps only groups present in the scope-mapped set."""
        repo = _make_scope_repo({"registry-admins", "registry-readonly"})
        groups = ["registry-admins", "noise-1", "registry-readonly"] + [f"g{i}" for i in range(500)]
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(groups, username_hash="h")
        assert set(result) == {"registry-admins", "registry-readonly"}

    @pytest.mark.asyncio
    async def test_allowlist_mode_overrides_scope_derived(self):
        """Design B (allowlist) takes precedence; scope repo is not consulted."""
        repo = _make_scope_repo({"should-not-be-used"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["keep-a", "keep-b"]):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ) as mock_get_repo:
                result = await group_filter.filter_session_groups(
                    ["keep-a", "drop-x", "keep-b"],
                    username_hash="h",
                )
        assert set(result) == {"keep-a", "keep-b"}
        mock_get_repo.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_mappings_fail_open(self):
        """No scope mappings seeded -> fail open (store full list)."""
        repo = _make_scope_repo(set())
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(
                    ["a", "b", "c"],
                    username_hash="h",
                )
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_repo_error_fail_open(self):
        """Repository error -> fail open (login must not break)."""
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                side_effect=RuntimeError("db down"),
            ):
                result = await group_filter.filter_session_groups(
                    ["a", "b"],
                    username_hash="h",
                )
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_identifier_mismatch_stores_empty(self):
        """Mappings keyed by names but token emits GUIDs -> empty intersection.

        A successful Design C run with no overlap stores empty (no-access); it
        does NOT chain to the allowlist.
        """
        repo = _make_scope_repo({"registry-admins"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch(
                "registry.repositories.factory.get_scope_repository",
                return_value=repo,
            ):
                result = await group_filter.filter_session_groups(
                    ["7f3a-guid", "9b2c-guid"],
                    username_hash="h",
                )
        assert result == []


class TestGroupNameNormalization:
    """Group names are compared in canonical form on both sides (issue #1689).

    Keycloak's Group Membership mapper emits full paths (``/mcp-admins``) when
    *Full group path* is on. Before normalisation such a claim matched nothing
    and was silently dropped from the session.
    """

    @pytest.mark.asyncio
    async def test_full_path_claim_matches_bare_mapping(self):
        """``/mcp-admins`` in the claim must match ``mcp-admins`` in the mapping."""
        repo = _make_scope_repo({"mcp-admins", "mcp-techs"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                result = await group_filter.filter_session_groups(
                    ["/mcp-admins", "/unrelated"],
                    username_hash="h",
                )
        assert result == ["mcp-admins"]

    @pytest.mark.asyncio
    async def test_bare_claim_matches_full_path_mapping(self):
        """The reverse: a mapping stored with a slash still matches a bare claim."""
        repo = _make_scope_repo({"/mcp-admins"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                result = await group_filter.filter_session_groups(
                    ["mcp-admins"],
                    username_hash="h",
                )
        assert result == ["mcp-admins"]

    @pytest.mark.asyncio
    async def test_returned_names_are_canonical(self):
        """The session stores the canonical form, not the raw claim."""
        repo = _make_scope_repo({"mcp-admins"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                result = await group_filter.filter_session_groups(
                    [" /mcp-admins/ "],
                    username_hash="h",
                )
        assert result == ["mcp-admins"]

    @pytest.mark.asyncio
    async def test_allowlist_matches_full_path_claim(self):
        """Design B entries cannot contain slashes, so the claim is canonicalised."""
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["keep-a"]):
            result = await group_filter.filter_session_groups(
                ["/keep-a", "/drop-x"],
                username_hash="h",
            )
        assert result == ["keep-a"]

    @pytest.mark.asyncio
    async def test_duplicate_forms_collapse(self):
        """``/team`` and ``team`` in the same claim are one group."""
        repo = _make_scope_repo({"team"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                result = await group_filter.filter_session_groups(
                    ["/team", "team"],
                    username_hash="h",
                )
        assert result == ["team"]


class TestDroppedGroupsAreLogged:
    """A dropped group must be named, not just counted (issue #1689).

    ``5 -> 1`` cannot tell an operator which group went missing, and every
    later log line faithfully reports the reduced set. This is the one place
    the information still exists.
    """

    @pytest.mark.asyncio
    async def test_scope_derived_drop_names_the_groups(self, caplog):
        repo = _make_scope_repo({"mcp-admins"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                with caplog.at_level(logging.WARNING, logger="auth_server.group_filter"):
                    await group_filter.filter_session_groups(
                        ["mcp-admins", "/finance-team", "hr-team"],
                        username_hash="h",
                    )
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("dropped 2 group(s)" in m for m in warnings), warnings
        assert any("finance-team" in m and "hr-team" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_allowlist_drop_names_the_groups(self, caplog):
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", ["keep-a"]):
            with caplog.at_level(logging.WARNING, logger="auth_server.group_filter"):
                await group_filter.filter_session_groups(
                    ["keep-a", "drop-x"],
                    username_hash="h",
                )
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("drop-x" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_nothing_dropped_logs_no_warning(self, caplog):
        """A clean pass must not cry wolf."""
        repo = _make_scope_repo({"mcp-admins", "mcp-techs"})
        with patch.object(group_filter, "ALLOWED_IDP_GROUPS", []):
            with patch("registry.repositories.factory.get_scope_repository", return_value=repo):
                with caplog.at_level(logging.INFO, logger="auth_server.group_filter"):
                    await group_filter.filter_session_groups(
                        ["mcp-admins", "mcp-techs"],
                        username_hash="h",
                    )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
        infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("2 -> 2" in m for m in infos), infos
