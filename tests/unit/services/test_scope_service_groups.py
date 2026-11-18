"""Tests for adding a server to groups and removing it again (scope_service).

The server list matches a group's ``list_service`` entries against the server's
path name, so these tests pin that add writes the path name and that remove
cleans up both the path name and the display name older code wrote.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.services import scope_service

SERVER_PATH = "/atlassian-rovo/"
DISPLAY_NAME = "Atlassian Rovo MCP"


def _server_info() -> dict:
    """A stored server with a display name that differs from its path."""
    return {
        "path": SERVER_PATH,
        "server_name": DISPLAY_NAME,
        "tool_list": [{"name": "getJiraIssue"}, {"name": "searchJiraIssuesUsingJql"}],
    }


@pytest.fixture
def scope_repo() -> AsyncMock:
    """A scope repository where every group exists and every write succeeds."""
    repo = AsyncMock()
    repo.group_exists.return_value = True
    return repo


@pytest.fixture
def patched(scope_repo: AsyncMock):
    """Patch the repository, the server lookup and the auth-server reload."""
    with (
        patch.object(scope_service, "get_scope_repository", return_value=scope_repo),
        patch.object(
            scope_service.server_service,
            "get_server_info",
            new=AsyncMock(return_value=_server_info()),
        ) as get_info,
        patch.object(scope_service, "_trigger_auth_server_reload", new=AsyncMock()),
    ):
        yield get_info


@pytest.mark.unit
class TestAddServerToGroups:
    """add_server_to_groups writes a scope rule and a list_service entry per group."""

    @pytest.mark.asyncio
    async def test_list_service_entry_uses_the_path_name(self, scope_repo, patched):
        """The UI entry is the path name, which the server list matches on."""
        ok = await scope_service.add_server_to_groups(SERVER_PATH, ["hr-team"])

        assert ok is True
        scope_repo.add_server_to_ui_scopes.assert_awaited_once_with(
            group_name="hr-team",
            server_name="atlassian-rovo",
        )

    @pytest.mark.asyncio
    async def test_scope_rule_carries_the_stored_tools(self, scope_repo, patched):
        """The server_access rule lists the tools from the stored tool list."""
        await scope_service.add_server_to_groups(SERVER_PATH, ["hr-team"])

        kwargs = scope_repo.add_server_scope.await_args.kwargs
        assert kwargs["server_path"] == SERVER_PATH
        assert kwargs["tools"] == ["getJiraIssue", "searchJiraIssuesUsingJql"]

    @pytest.mark.asyncio
    async def test_missing_group_is_skipped(self, scope_repo, patched):
        """A group that does not exist gets no rule and no UI entry."""
        scope_repo.group_exists.return_value = False

        await scope_service.add_server_to_groups(SERVER_PATH, ["no-such-group"])

        scope_repo.add_server_scope.assert_not_awaited()
        scope_repo.add_server_to_ui_scopes.assert_not_awaited()


@pytest.mark.unit
class TestRemoveServerFromGroups:
    """remove_server_from_groups removes the rule and every UI entry form."""

    @pytest.mark.asyncio
    async def test_removes_the_path_name_and_the_old_display_name(self, scope_repo, patched):
        """Entries written before the fix used the display name, so both go."""
        ok = await scope_service.remove_server_from_groups(SERVER_PATH, ["hr-team"])

        assert ok is True
        removed = [
            call.kwargs["server_name"]
            for call in scope_repo.remove_server_from_ui_scopes.await_args_list
        ]
        assert removed == ["atlassian-rovo", DISPLAY_NAME]

    @pytest.mark.asyncio
    async def test_unknown_server_removes_only_the_path_name(self, scope_repo, patched):
        """With no stored server there is no display name to clean up."""
        patched.return_value = None

        await scope_service.remove_server_from_groups(SERVER_PATH, ["hr-team"])

        scope_repo.remove_server_from_ui_scopes.assert_awaited_once_with(
            group_name="hr-team",
            server_name="atlassian-rovo",
        )

    @pytest.mark.asyncio
    async def test_display_name_equal_to_the_path_name_is_removed_once(self, scope_repo, patched):
        """A server whose display name is its path name gets one removal, not two."""
        patched.return_value = {**_server_info(), "server_name": "atlassian-rovo"}

        await scope_service.remove_server_from_groups(SERVER_PATH, ["hr-team"])

        assert scope_repo.remove_server_from_ui_scopes.await_count == 1
