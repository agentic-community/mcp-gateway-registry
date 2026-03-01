"""
Unit tests for Audit Filter Options and Statistics endpoints.

Tests the GET /audit/filter-options and GET /audit/statistics
endpoints, plus the repository distinct() and aggregate() methods.

Validates: Issue #572
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.repositories.audit_repository import DocumentDBAuditRepository


# =============================================================================
# Repository: distinct() method
# =============================================================================


class TestDistinct:
    """Tests for DocumentDBAuditRepository.distinct() method."""

    async def test_returns_sorted_distinct_values(self):
        """distinct() returns a sorted list of distinct string values."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(
            return_value=["charlie", "alice", "bob"]
        )

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.distinct("identity.username")

            assert result == ["alice", "bob", "charlie"]
            mock_collection.distinct.assert_called_once_with(
                "identity.username", {}
            )

    async def test_filters_out_none_and_empty(self):
        """distinct() filters out None and empty string values."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(
            return_value=["admin", None, "", "user1"]
        )

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.distinct("identity.username")

            assert result == ["admin", "user1"]

    async def test_passes_query_filter(self):
        """distinct() passes the query filter to MongoDB."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(return_value=["admin"])
        query = {"log_type": "registry_api_access"}

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.distinct("identity.username", query)

            mock_collection.distinct.assert_called_once_with(
                "identity.username", query
            )
            assert result == ["admin"]

    async def test_returns_empty_on_error(self):
        """distinct() returns empty list on error."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(side_effect=Exception("DB error"))

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.distinct("identity.username")

            assert result == []


# =============================================================================
# Repository: aggregate() method
# =============================================================================


class TestAggregate:
    """Tests for DocumentDBAuditRepository.aggregate() method."""

    async def test_returns_aggregation_results(self):
        """aggregate() returns list of aggregation result docs."""
        mock_collection = MagicMock()
        test_results = [
            {"_id": "admin", "count": 100},
            {"_id": "user1", "count": 50},
        ]

        async def async_iter():
            for doc in test_results:
                yield doc

        mock_collection.aggregate = MagicMock(return_value=async_iter())

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            pipeline = [
                {"$match": {"log_type": "registry_api_access"}},
                {"$group": {"_id": "$identity.username", "count": {"$sum": 1}}},
            ]
            result = await repo.aggregate(pipeline)

            assert len(result) == 2
            assert result[0]["_id"] == "admin"
            assert result[0]["count"] == 100

    async def test_returns_empty_list_on_no_results(self):
        """aggregate() returns empty list when no results."""
        mock_collection = MagicMock()

        async def async_iter():
            return
            yield

        mock_collection.aggregate = MagicMock(return_value=async_iter())

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.aggregate([{"$match": {}}])

            assert result == []

    async def test_returns_empty_on_error(self):
        """aggregate() returns empty list on error."""
        mock_collection = MagicMock()
        mock_collection.aggregate = MagicMock(side_effect=Exception("DB error"))

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            result = await repo.aggregate([{"$match": {}}])

            assert result == []


# =============================================================================
# API Endpoint: GET /audit/filter-options
# =============================================================================


class TestFilterOptionsEndpoint:
    """Tests for GET /api/audit/filter-options endpoint."""

    async def test_returns_usernames_for_registry_stream(self):
        """Returns usernames for registry_api stream."""
        mock_repo = MagicMock()
        mock_repo.distinct = AsyncMock(
            side_effect=lambda field, query: ["admin", "user1"]
        )

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            from registry.audit.routes import get_filter_options

            result = await get_filter_options(
                user_context={"is_admin": True, "username": "admin"},
                stream="registry_api",
            )

            assert result.usernames == ["admin", "user1"]
            assert result.server_names == []

    async def test_returns_usernames_and_servers_for_mcp_stream(self):
        """Returns both usernames and server names for mcp_access stream."""
        mock_repo = MagicMock()

        async def mock_distinct(field, query):
            if field == "identity.username":
                return ["admin", "user1"]
            elif field == "mcp_server.name":
                return ["fininfo-server", "currenttime-server"]
            return []

        mock_repo.distinct = AsyncMock(side_effect=mock_distinct)

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            from registry.audit.routes import get_filter_options

            result = await get_filter_options(
                user_context={"is_admin": True, "username": "admin"},
                stream="mcp_access",
            )

            assert result.usernames == ["admin", "user1"]
            assert result.server_names == ["fininfo-server", "currenttime-server"]


# =============================================================================
# API Endpoint: GET /audit/statistics
# =============================================================================


class TestStatisticsEndpoint:
    """Tests for GET /api/audit/statistics endpoint."""

    async def test_returns_statistics_for_registry_stream(self):
        """Returns aggregated statistics for registry_api stream."""
        mock_repo = MagicMock()
        mock_repo.count = AsyncMock(return_value=500)

        # Top users
        top_users = [
            {"_id": "admin", "count": 300},
            {"_id": "user1", "count": 200},
        ]
        # Top operations
        top_ops = [
            {"_id": "list", "count": 250},
            {"_id": "read", "count": 150},
        ]
        # Timeline
        timeline = [
            {"_id": "2026-02-27", "count": 200},
            {"_id": "2026-02-28", "count": 300},
        ]
        # Status distribution
        status_dist = [
            {"_id": "2xx", "count": 450},
            {"_id": "4xx", "count": 40},
            {"_id": "5xx", "count": 10},
        ]
        # Per-user activity breakdown
        user_activity = [
            {
                "_id": "admin",
                "total": 300,
                "operations": [
                    {"name": "list", "count": 200},
                    {"name": "read", "count": 100},
                ],
            },
            {
                "_id": "user1",
                "total": 200,
                "operations": [{"name": "read", "count": 200}],
            },
        ]

        # aggregate() is called 5 times for registry_api (no server aggregation)
        mock_repo.aggregate = AsyncMock(
            side_effect=[top_users, top_ops, timeline, status_dist, user_activity]
        )

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            from registry.audit.routes import get_statistics

            result = await get_statistics(
                user_context={"is_admin": True, "username": "admin"},
                stream="registry_api",
                days=7,
                username=None,
            )

            assert result.total_events == 500
            assert len(result.top_users) == 2
            assert result.top_users[0].name == "admin"
            assert result.top_users[0].count == 300
            assert len(result.top_operations) == 2
            assert len(result.activity_timeline) == 2
            assert result.status_distribution.status_2xx == 450
            assert result.status_distribution.status_4xx == 40
            assert result.status_distribution.status_5xx == 10
            assert result.top_servers == []
            assert len(result.user_activity) == 2
            assert result.user_activity[0].username == "admin"
            assert result.user_activity[0].total == 300
            assert len(result.user_activity[0].operations) == 2

    async def test_returns_statistics_for_mcp_stream(self):
        """Returns aggregated statistics for mcp_access stream including servers."""
        mock_repo = MagicMock()
        mock_repo.count = AsyncMock(return_value=200)

        top_users = [{"_id": "admin", "count": 200}]
        top_ops = [{"_id": "tools/call", "count": 100}]
        timeline = [{"_id": "2026-02-28", "count": 200}]
        status_dist = [
            {"_id": "success", "count": 180},
            {"_id": "error", "count": 20},
        ]
        # Per-user activity breakdown
        user_activity = [
            {
                "_id": "admin",
                "total": 200,
                "operations": [{"name": "tools/call", "count": 100}],
            },
        ]
        top_servers = [
            {"_id": "fininfo-server", "count": 89},
            {"_id": "currenttime-server", "count": 67},
        ]

        # aggregate() is called 6 times for mcp_access (includes user_activity + server aggregation)
        mock_repo.aggregate = AsyncMock(
            side_effect=[top_users, top_ops, timeline, status_dist, user_activity, top_servers]
        )

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            from registry.audit.routes import get_statistics

            result = await get_statistics(
                user_context={"is_admin": True, "username": "admin"},
                stream="mcp_access",
                days=7,
                username=None,
            )

            assert result.total_events == 200
            assert len(result.top_servers) == 2
            assert result.top_servers[0].name == "fininfo-server"
            # MCP success -> status_2xx
            assert result.status_distribution.status_2xx == 180
            # MCP error -> status_5xx
            assert result.status_distribution.status_5xx == 20
            assert len(result.user_activity) == 1
            assert result.user_activity[0].username == "admin"

    async def test_handles_empty_results(self):
        """Returns zero counts when no events exist."""
        mock_repo = MagicMock()
        mock_repo.count = AsyncMock(return_value=0)
        mock_repo.aggregate = AsyncMock(return_value=[])

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            from registry.audit.routes import get_statistics

            result = await get_statistics(
                user_context={"is_admin": True, "username": "admin"},
                stream="registry_api",
                days=7,
                username=None,
            )

            assert result.total_events == 0
            assert result.top_users == []
            assert result.top_operations == []
            assert result.activity_timeline == []
            assert result.status_distribution.status_2xx == 0
            assert result.status_distribution.status_4xx == 0
            assert result.status_distribution.status_5xx == 0
            assert result.user_activity == []
