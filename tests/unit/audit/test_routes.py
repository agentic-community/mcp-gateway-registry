"""
Unit tests for Audit API routes.

Validates: Requirements 7.1, 7.2, 7.5, 7.6
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from hypothesis import given, settings, strategies as st

from registry.audit.routes import _build_query, _generate_csv, _generate_jsonl, require_admin


# =============================================================================
# Property 11: Admin-Only Audit API Access
# =============================================================================


class TestAdminOnlyAccess:
    """Property 11: Admin-only audit API access."""

    @given(
        st.fixed_dictionaries({
            "username": st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            "is_admin": st.just(False),
        })
    )
    @settings(max_examples=50)
    def test_rejects_non_admin_users(self, user_context: dict):
        """require_admin raises 403 for any non-admin user."""
        with pytest.raises(HTTPException) as exc_info:
            require_admin(user_context)
        assert exc_info.value.status_code == 403

    def test_allows_admin_users(self):
        """require_admin allows admin users."""
        user_context = {"username": "admin", "is_admin": True}
        result = require_admin(user_context)
        assert result["is_admin"] is True


# =============================================================================
# Query Building
# =============================================================================


class TestBuildQuery:
    """Tests for _build_query function."""

    def test_stream_only(self):
        """Build query with only stream parameter."""
        query = _build_query(stream="registry_api", from_time=None, to_time=None,
                            username=None, operation=None, resource_type=None,
                            resource_id=None, status_min=None, status_max=None, auth_decision=None)
        assert query == {"log_type": "registry_api_access"}

    def test_with_filters(self):
        """Build query with multiple filters."""
        from_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        query = _build_query(stream="registry_api", from_time=from_time, to_time=None,
                            username="admin", operation="create", resource_type="server",
                            resource_id=None, status_min=400, status_max=499, auth_decision=None)

        # Username uses case-insensitive regex for partial matching
        assert query["identity.username"]["$regex"] == "admin"
        assert query["identity.username"]["$options"] == "i"
        assert query["action.operation"] == "create"
        assert query["response.status_code"]["$gte"] == 400


# =============================================================================
# Export Format Generation
# =============================================================================


class TestExportFormats:
    """Tests for export format generation."""

    def test_generate_jsonl(self):
        """Generate JSONL from events."""
        events = [{"request_id": "req-1"}, {"request_id": "req-2"}]
        result = list(_generate_jsonl(events))
        assert len(result) == 2
        assert all(line.endswith("\n") for line in result)

    def test_generate_csv(self):
        """Generate CSV from events."""
        events = [{
            "timestamp": datetime(2025, 1, 15, tzinfo=timezone.utc),
            "request_id": "req-1",
            "identity": {"username": "admin"},
            "request": {"method": "GET", "path": "/api/test"},
            "response": {"status_code": 200, "duration_ms": 50.0},
            "action": {"operation": "read", "resource_type": "server"},
        }]
        result = list(_generate_csv(events))
        csv_content = result[0]
        assert "timestamp" in csv_content
        assert "req-1" in csv_content


# =============================================================================
# API Endpoints
# =============================================================================


class TestAuditEventsEndpoint:
    """Tests for GET /api/audit/events endpoint."""

    async def test_returns_paginated_results(self):
        """GET /events returns paginated audit events."""
        mock_repo = MagicMock()
        mock_repo.find = AsyncMock(return_value=[{"request_id": "req-1"}])
        mock_repo.count = AsyncMock(return_value=1)

        with patch("registry.audit.routes.get_audit_repository", return_value=mock_repo):
            from registry.audit.routes import get_audit_events

            result = await get_audit_events(
                user_context={"is_admin": True},
                stream="registry_api",
                from_time=None,
                to_time=None,
                username=None,
                operation=None,
                resource_type=None,
                resource_id=None,
                status_min=None,
                status_max=None,
                auth_decision=None,
                limit=50,
                offset=0,
                sort_order=-1,
            )

            assert result.total == 1
            assert len(result.events) == 1


class TestAuditEventDetailEndpoint:
    """Tests for GET /api/audit/events/{request_id} endpoint."""

    async def test_returns_404_when_not_found(self):
        """GET /events/{request_id} returns 404 when event not found."""
        mock_repo = MagicMock()
        mock_repo.find = AsyncMock(return_value=[])

        with patch("registry.audit.routes.get_audit_repository", return_value=mock_repo):
            from registry.audit.routes import get_audit_event

            with pytest.raises(HTTPException) as exc_info:
                await get_audit_event(
                    request_id="nonexistent",
                    user_context={"is_admin": True},
                    log_type=None,
                )

            assert exc_info.value.status_code == 404
