"""Unit tests for the reserved-wildcard sink guard in scope_repository.

Defense-in-depth: even if a caller bypasses the registration-time
``validate_server_path`` guard, ``add_server_scope`` must refuse to write a
scope row whose server name collides with the cross-server wildcard sentinel
(``all`` / ``*``). Such a row would grant access to every server in the
registry. The guard fails closed by returning False and logging, matching the
method's existing error convention (it must NOT crash and must NOT write).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb.scope_repository import DocumentDBScopeRepository


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    collection.update_one = AsyncMock()
    collection.replace_one = AsyncMock()
    return collection


@pytest.fixture
def repo(mock_collection):
    r = DocumentDBScopeRepository.__new__(DocumentDBScopeRepository)
    r._collection = mock_collection
    r._collection_name = "mcp_scopes_test"
    r._scopes_cache = {}
    return r


class TestAddServerScopeReservedWildcard:
    """add_server_scope must fail closed on reserved wildcard names."""

    @pytest.mark.parametrize(
        "server_path",
        ["/all", "all", "/ALL", "/All", "all/", "//all//", "/*", "*"],
    )
    @pytest.mark.asyncio
    async def test_refuses_reserved_wildcard_and_does_not_write(
        self, repo, mock_collection, server_path
    ):
        # Arrange
        # (repo/collection from fixtures)

        # Act
        result = await repo.add_server_scope(
            server_path=server_path,
            scope_name="mcp-servers-unrestricted/read",
            methods=["tools/list"],
            tools=["*"],
        )

        # Assert
        assert result is False
        mock_collection.update_one.assert_not_called()
        assert repo._scopes_cache == {}

    @pytest.mark.parametrize("server_path", ["/", "//", "///", ""])
    @pytest.mark.asyncio
    async def test_refuses_empty_server_name_and_does_not_write(
        self, repo, mock_collection, server_path
    ):
        """Issue #1501: an empty/slashes-only server name renders as a
        gateway-wide `location /` block, so the registration guard rejects it and
        this mirror must too. Fail closed: no write."""
        result = await repo.add_server_scope(
            server_path=server_path,
            scope_name="mcp-servers-unrestricted/read",
            methods=["tools/list"],
            tools=["*"],
        )

        assert result is False
        mock_collection.update_one.assert_not_called()
        assert repo._scopes_cache == {}

    @pytest.mark.asyncio
    async def test_allows_adjacent_name_and_writes(self, repo, mock_collection):
        # Arrange
        mock_collection.update_one.return_value = MagicMock(matched_count=1)

        # Act
        result = await repo.add_server_scope(
            server_path="/all-tools",
            scope_name="mcp-servers-unrestricted/read",
            methods=["tools/list"],
            tools=["*"],
        )

        # Assert
        assert result is True
        mock_collection.update_one.assert_awaited_once()
        assert repo._scopes_cache["mcp-servers-unrestricted/read"][0]["server"] == "all-tools"


class TestImportGroupReservedWildcard:
    """import_group writes server_access directly (bypassing add_server_scope),
    so it needs its own reserved-wildcard sink guard."""

    @pytest.mark.parametrize(
        "server_rule",
        [
            {"server": "all", "methods": ["tools/list"], "tools": ["*"]},
            {"server": "*", "methods": ["tools/list"], "tools": ["*"]},
            {"server": "/all", "methods": ["tools/list"], "tools": ["*"]},
            {"server": "ALL", "methods": ["tools/list"], "tools": ["*"]},
        ],
    )
    @pytest.mark.asyncio
    async def test_grouped_shape_reserved_server_refused(self, repo, mock_collection, server_rule):
        # Arrange: grouped access_rules shape
        server_access = [{"scope_name": "myscope", "access_rules": [server_rule]}]

        # Act
        result = await repo.import_group(
            group_name="myscope",
            server_access=server_access,
        )

        # Assert: refused, no write
        assert result is False
        mock_collection.replace_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_direct_rule_shape_reserved_server_refused(self, repo, mock_collection):
        # Arrange: direct server-rule shape (no access_rules wrapper)
        server_access = [{"server": "all", "methods": ["tools/list"], "tools": ["*"]}]

        # Act
        result = await repo.import_group(group_name="myscope", server_access=server_access)

        # Assert
        assert result is False
        mock_collection.replace_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_reserved_server_refused_even_with_allow_privileged(self, repo, mock_collection):
        # A reserved wildcard server name is never legitimate, so even an
        # admin (allow_privileged=True) cannot import it.
        server_access = [{"server": "all", "methods": ["tools/list"], "tools": ["*"]}]

        result = await repo.import_group(
            group_name="myscope",
            server_access=server_access,
            allow_privileged=True,
        )

        assert result is False
        mock_collection.replace_one.assert_not_called()

    @pytest.mark.parametrize(
        "server_rule",
        [
            {"server": "", "methods": ["tools/list"], "tools": ["*"]},
            {"server": "/", "methods": ["tools/list"], "tools": ["*"]},
            {"server": "//", "methods": ["tools/list"], "tools": ["*"]},
            # A present-but-degenerate server value coerces to empty and must be
            # refused (the resolver would grant nothing, but the guard mirrors
            # validate_server_path which rejects the value outright). str(x or "")
            # normalizes None / non-string falsy values to "".
            {"server": None, "methods": ["tools/list"], "tools": ["*"]},
            {"server": 0, "methods": ["tools/list"], "tools": ["*"]},
            {"server": [], "methods": ["tools/list"], "tools": ["*"]},
        ],
    )
    @pytest.mark.asyncio
    async def test_empty_server_rule_refused(self, repo, mock_collection, server_rule):
        """Issue #1501: an empty/slashes-only (or falsy, coercing-to-empty)
        server rule renders as a gateway-wide `location /` block, so import must
        refuse it too."""
        server_access = [{"scope_name": "myscope", "access_rules": [server_rule]}]

        result = await repo.import_group(group_name="myscope", server_access=server_access)

        assert result is False
        mock_collection.replace_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_benign_server_access_imports(self, repo, mock_collection):
        # Arrange: legitimate, non-wildcard server rule imports normally
        mock_collection.replace_one.return_value = MagicMock(upserted_id="myscope")
        server_access = [
            {
                "scope_name": "myscope",
                "access_rules": [
                    {"server": "all-tools", "methods": ["tools/list"], "tools": ["*"]}
                ],
            }
        ]

        # Act
        result = await repo.import_group(group_name="myscope", server_access=server_access)

        # Assert
        assert result is True
        mock_collection.replace_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_only_rule_imports(self, repo, mock_collection):
        """An agent rule has no `server` key and must not be caught by the
        server-rule sink guard (it is inspected by validate_a2a_agent_access,
        not this scan)."""
        mock_collection.replace_one.return_value = MagicMock(upserted_id="myscope")
        server_access = [
            {
                "scope_name": "myscope",
                "access_rules": [{"agent": "flight-booking-agent", "actions": ["message/send"]}],
            }
        ]

        result = await repo.import_group(group_name="myscope", server_access=server_access)

        assert result is True
        mock_collection.replace_one.assert_awaited_once()
