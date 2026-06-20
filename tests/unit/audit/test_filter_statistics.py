"""
Unit tests for Audit Filter Options and Statistics endpoints.

Tests the GET /audit/filter-options and GET /audit/statistics
endpoints, plus the repository distinct() and aggregate() methods.

Validates: Issue #572
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.core.config import settings
from registry.repositories.audit_repository import DocumentDBAuditRepository

# =============================================================================
# Repository: distinct() method
# =============================================================================


class TestDistinct:
    """Tests for DocumentDBAuditRepository.distinct() method."""

    async def test_returns_sorted_distinct_values(self):
        """distinct() returns a sorted list of distinct string values."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(return_value=["charlie", "alice", "bob"])

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
            mock_collection.distinct.assert_called_once_with("identity.username", {})

    async def test_filters_out_none_and_empty(self):
        """distinct() filters out None and empty string values."""
        mock_collection = AsyncMock()
        mock_collection.distinct = AsyncMock(return_value=["admin", None, "", "user1"])

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

            mock_collection.distinct.assert_called_once_with("identity.username", query)
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
        mock_repo.distinct = AsyncMock(side_effect=lambda field, query: ["admin", "user1"])

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

        # Prior-window timeline
        timeline_prior = [
            {"_id": "2026-02-20", "count": 100},
            {"_id": "2026-02-21", "count": 120},
        ]

        # aggregate() is called 6 times for registry_api (no server aggregation)
        mock_repo.aggregate = AsyncMock(
            side_effect=[top_users, top_ops, timeline, status_dist, user_activity, timeline_prior]
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
            assert len(result.activity_timeline_prior) == 2
            assert result.activity_timeline_prior[0].period == "2026-02-20"
            assert result.activity_timeline_prior[0].count == 100

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
        timeline_prior = [{"_id": "2026-02-21", "count": 150}]
        top_servers = [
            {"_id": "fininfo-server", "count": 89},
            {"_id": "currenttime-server", "count": 67},
        ]

        # aggregate() is called 7 times for mcp_access
        # (user_activity + prior timeline + server aggregation)
        mock_repo.aggregate = AsyncMock(
            side_effect=[
                top_users,
                top_ops,
                timeline,
                status_dist,
                user_activity,
                timeline_prior,
                top_servers,
            ]
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


# =============================================================================
# API Endpoint: GET /audit/executive-summary
# =============================================================================


def _is_agent_query(query: dict) -> bool:
    """Return True when the query filters for agent (bearer-token) traffic."""
    return query.get("identity.credential_type") == "bearer_token"


def _is_human_query(query: dict) -> bool:
    """Return True when the query filters for human (session-cookie) traffic."""
    return query.get("identity.credential_type") == "session_cookie"


class TestExecutiveSummaryEndpoint:
    """Tests for GET /api/audit/executive-summary endpoint."""

    @pytest.fixture(autouse=True)
    def _clear_summary_cache(self):
        """Clear the module-level summary cache so tests do not see stale data."""
        import registry.audit.routes as routes

        routes._exec_summary_cache.clear()
        yield
        routes._exec_summary_cache.clear()

    async def test_happy_path_computes_metrics(self):
        """Computes governance, active users, traffic split, and momentum."""
        mock_repo = MagicMock()

        async def mock_distinct(field, query):
            # MCP-only metrics
            if field == "mcp_server.name":
                return ["fininfo-server", "currenttime-server", ""]
            if field == "mcp_request.tool_name":
                return ["get_quote", "get_time", None]
            # Username distinct counts: agents have only one identity
            if _is_agent_query(query):
                return ["agent-bot"]
            return ["alice", "bob", "anonymous", ""]

        async def mock_count(query):
            # Human events vs agent events for traffic split
            if _is_human_query(query):
                return 60
            if _is_agent_query(query):
                return 40
            # Non-anonymous event counts (current vs prior window)
            ts = query.get("timestamp", {})
            gte = ts.get("$gte")
            # Prior window has the older start; current has the newer start
            return 100 if gte is not None else 0

        mock_repo.distinct = AsyncMock(side_effect=mock_distinct)
        mock_repo.count = AsyncMock(side_effect=mock_count)

        from registry.audit.routes import RegisteredAssets

        with (
            patch(
                "registry.audit.routes.get_audit_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.audit.routes._get_registered_assets",
                new=AsyncMock(return_value=RegisteredAssets()),
            ),
        ):
            from registry.audit.routes import get_executive_summary

            result = await get_executive_summary(
                user_context={"is_admin": True, "username": "admin"},
                days=7,
            )

            # Governance: falsy values dropped
            assert result.governance.mcp_servers_governed == 2
            assert result.governance.tools_under_policy == 2
            # identities_active drops anonymous/empty -> alice, bob
            assert result.governance.identities_active == 2

            # Active users: same non-agent distinct -> 2 each
            assert result.active_users.dau == 2
            assert result.active_users.wau == 2
            assert result.active_users.mau == 2

            # Active agents: agent-filtered distinct returns one identity
            assert result.active_agents.daa == 1
            assert result.active_agents.waa == 1
            assert result.active_agents.maa == 1
            assert result.active_agents.waa_available is True
            assert result.active_agents.maa_available is False

            # Registered assets come from the patched helper (zeros here)
            assert result.registered_assets.servers == 0
            assert result.registered_assets.tools == 0

            # Traffic split: 60 human, 40 agent -> 60.0 / 40.0
            assert result.traffic_split.human_events == 60
            assert result.traffic_split.agent_events == 40
            assert result.traffic_split.human_pct == 60.0
            assert result.traffic_split.agent_pct == 40.0

            assert result.window_days == 7
            # Default retention (7d) covers WAU but not the 30d MAU window
            assert result.retention_days == 7
            assert result.active_users.wau_available is True
            assert result.active_users.mau_available is False

    async def test_no_prior_data_sets_wow_none(self):
        """events_wow_pct is None and has_prior_data False when prior window is empty."""
        mock_repo = MagicMock()

        async def mock_distinct(field, query):
            if field in ("mcp_server.name", "mcp_request.tool_name"):
                return []
            return ["alice"]

        # Distinguish current vs prior non-anonymous event counts by window start.
        # The prior window starts further in the past (days*2) than the current
        # window (days). We return 0 for the older start to simulate no prior data.
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        cur_start_threshold = now - timedelta(days=10)

        async def mock_count(query):
            if _is_agent_query(query):
                return 5
            gte = query.get("timestamp", {}).get("$gte")
            if gte is not None and gte < cur_start_threshold:
                return 0  # prior window
            return 50  # current window

        mock_repo.distinct = AsyncMock(side_effect=mock_distinct)
        mock_repo.count = AsyncMock(side_effect=mock_count)

        from registry.audit.routes import RegisteredAssets

        with (
            patch(
                "registry.audit.routes.get_audit_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.audit.routes._get_registered_assets",
                new=AsyncMock(return_value=RegisteredAssets()),
            ),
        ):
            from registry.audit.routes import get_executive_summary

            result = await get_executive_summary(
                user_context={"is_admin": True, "username": "admin"},
                days=7,
            )

            assert result.momentum.events_prior == 0
            assert result.momentum.events_wow_pct is None
            assert result.momentum.has_prior_data is False
            assert result.momentum.events_current == 50

    async def test_wow_pct_rounding(self):
        """events_wow_pct rounds to one decimal when prior data exists."""
        mock_repo = MagicMock()

        async def mock_distinct(field, query):
            if field in ("mcp_server.name", "mcp_request.tool_name"):
                return []
            return ["alice"]

        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        cur_start_threshold = now - timedelta(days=10)

        async def mock_count(query):
            if _is_agent_query(query):
                return 1
            gte = query.get("timestamp", {}).get("$gte")
            if gte is not None and gte < cur_start_threshold:
                return 30  # prior window
            return 40  # current window -> (40-30)/30*100 = 33.333...

        mock_repo.distinct = AsyncMock(side_effect=mock_distinct)
        mock_repo.count = AsyncMock(side_effect=mock_count)

        from registry.audit.routes import RegisteredAssets

        # Retention must cover the prior window (days*2 = 14) for WoW to be honest.
        with (
            patch(
                "registry.audit.routes.get_audit_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.audit.routes._get_registered_assets",
                new=AsyncMock(return_value=RegisteredAssets()),
            ),
            patch.object(settings, "audit_log_mongodb_ttl_days", 30),
        ):
            from registry.audit.routes import get_executive_summary

            result = await get_executive_summary(
                user_context={"is_admin": True, "username": "admin"},
                days=7,
            )

            assert result.momentum.events_prior == 30
            assert result.momentum.events_current == 40
            assert result.momentum.events_wow_pct == 33.3
            assert result.momentum.has_prior_data is True
            # Retention now covers the 30d window too
            assert result.active_users.mau_available is True

    async def test_short_retention_hides_wow(self):
        """has_prior_data is False when retention does not cover the prior window."""
        mock_repo = MagicMock()

        async def mock_distinct(field, query):
            if field in ("mcp_server.name", "mcp_request.tool_name"):
                return []
            return ["alice"]

        # Prior window reports events, but retention is too short to trust them.
        async def mock_count(query):
            if _is_agent_query(query):
                return 1
            return 30

        mock_repo.distinct = AsyncMock(side_effect=mock_distinct)
        mock_repo.count = AsyncMock(side_effect=mock_count)

        from registry.audit.routes import RegisteredAssets

        # days=7 needs 14d retention for the prior window; 7d is not enough.
        with (
            patch(
                "registry.audit.routes.get_audit_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.audit.routes._get_registered_assets",
                new=AsyncMock(return_value=RegisteredAssets()),
            ),
            patch.object(settings, "audit_log_mongodb_ttl_days", 7),
        ):
            from registry.audit.routes import get_executive_summary

            result = await get_executive_summary(
                user_context={"is_admin": True, "username": "admin"},
                days=7,
            )

            # Prior count is non-zero, but retention gating still hides WoW.
            assert result.momentum.events_prior == 30
            assert result.momentum.has_prior_data is False
            assert result.momentum.events_wow_pct is None
            assert result.active_users.mau_available is False

    async def test_registered_assets_counts(self):
        """_get_registered_assets gathers counts from each asset repository."""
        server_repo = MagicMock()
        server_repo.count = AsyncMock(return_value=5)
        server_repo.count_tools = AsyncMock(return_value=13)
        agent_repo = MagicMock()
        agent_repo.count = AsyncMock(return_value=4)
        skill_repo = MagicMock()
        skill_repo.count = AsyncMock(return_value=7)
        custom_entity_repo = MagicMock()
        custom_entity_repo.count_all = AsyncMock(return_value=2)

        with (
            patch(
                "registry.repositories.factory.get_server_repository",
                return_value=server_repo,
            ),
            patch(
                "registry.repositories.factory.get_agent_repository",
                return_value=agent_repo,
            ),
            patch(
                "registry.repositories.factory.get_skill_repository",
                return_value=skill_repo,
            ),
            patch(
                "registry.repositories.factory.get_custom_entity_repository",
                return_value=custom_entity_repo,
            ),
        ):
            from registry.audit.routes import _get_registered_assets

            assets = await _get_registered_assets()

            assert assets.servers == 5
            assert assets.tools == 13
            assert assets.agents == 4
            assert assets.skills == 7
            assert assets.custom_entities == 2

    async def test_summary_is_cached_within_ttl(self):
        """A second call within the TTL returns the cache without recomputing."""
        from registry.audit.routes import (
            ExecutiveSummaryResponse,
            RegisteredAssets,
            get_executive_summary,
        )

        compute = AsyncMock(
            return_value=ExecutiveSummaryResponse(
                window_days=7,
                registered_assets=RegisteredAssets(),
            )
        )

        with patch(
            "registry.audit.routes._compute_executive_summary",
            new=compute,
        ):
            user_context = {"is_admin": True, "username": "admin"}
            await get_executive_summary(user_context=user_context, days=7)
            await get_executive_summary(user_context=user_context, days=7)

        # Two requests, but the expensive computation ran only once
        assert compute.await_count == 1
