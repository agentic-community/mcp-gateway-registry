"""
Unit tests for audit events composite key (request_id, log_type).

Validates that both MCPServerAccessRecord and RegistryApiAccessRecord
can coexist for the same request_id, and that the detail endpoint
returns multiple events.

Related: GitHub issue #527
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pymongo.errors import DuplicateKeyError

from registry.audit.models import (
    Identity,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPServerAccessRecord,
    RegistryApiAccessRecord,
    Request,
    Response,
)
from registry.audit.routes import get_audit_event
from registry.repositories.audit_repository import DocumentDBAuditRepository


def _make_registry_record(
    request_id: str = "req-123",
) -> RegistryApiAccessRecord:
    """Create a test RegistryApiAccessRecord."""
    return RegistryApiAccessRecord(
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        identity=Identity(
            username="testuser",
            auth_method="oauth2",
            credential_type="bearer_token",
        ),
        request=Request(
            method="POST",
            path="/cloudflare-docs/mcp",
            client_ip="127.0.0.1",
        ),
        response=Response(
            status_code=200,
            duration_ms=150.0,
        ),
    )


def _make_mcp_record(
    request_id: str = "req-123",
) -> MCPServerAccessRecord:
    """Create a test MCPServerAccessRecord."""
    return MCPServerAccessRecord(
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        identity=Identity(
            username="testuser",
            auth_method="oauth2",
            credential_type="bearer_token",
        ),
        mcp_server=MCPServer(
            name="cloudflare-docs",
            path="/cloudflare-docs",
            proxy_target="http://localhost:8001",
        ),
        mcp_request=MCPRequest(
            method="tools/call",
            tool_name="search_cloudflare_documentation",
        ),
        mcp_response=MCPResponse(
            status="success",
            duration_ms=120.0,
        ),
    )


class TestCompositeKeyInsert:
    """Tests for composite unique key (request_id, log_type) insert behavior."""

    async def test_both_record_types_insert_with_same_request_id(self):
        """Both MCPServerAccessRecord and RegistryApiAccessRecord can be inserted
        with the same request_id (different log_type values)."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="new_id")

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            mcp_record = _make_mcp_record(request_id="req-123")
            result1 = await repo.insert(mcp_record)
            assert result1 is True

            registry_record = _make_registry_record(request_id="req-123")
            result2 = await repo.insert(registry_record)
            assert result2 is True

            assert mock_collection.insert_one.call_count == 2

    async def test_true_duplicate_returns_true(self):
        """DuplicateKeyError is caught for true duplicates
        (same request_id AND same log_type)."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.side_effect = DuplicateKeyError(
            "duplicate key error"
        )

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            result = await repo.insert(_make_registry_record())
            assert result is True

    async def test_record_log_type_defaults_are_distinct(self):
        """Verify the two record types have distinct log_type defaults."""
        mcp_record = _make_mcp_record()
        registry_record = _make_registry_record()

        assert mcp_record.log_type == "mcp_server_access"
        assert registry_record.log_type == "registry_api_access"
        assert mcp_record.log_type != registry_record.log_type


class TestDetailEndpointMultipleEvents:
    """Tests for GET /events/{request_id} with composite key."""

    async def test_returns_multiple_events(self):
        """Endpoint returns all events for a given request_id."""
        mock_repo = AsyncMock()
        mock_repo.find.return_value = [
            {
                "request_id": "req-123",
                "log_type": "mcp_server_access",
            },
            {
                "request_id": "req-123",
                "log_type": "registry_api_access",
            },
        ]

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            response = await get_audit_event(
                request_id="req-123",
                user_context={"username": "admin"},
            )

            assert response["request_id"] == "req-123"
            assert len(response["events"]) == 2

    async def test_filters_by_log_type(self):
        """Endpoint filters events by log_type query parameter."""
        mock_repo = AsyncMock()
        mock_repo.find.return_value = [
            {
                "request_id": "req-123",
                "log_type": "registry_api_access",
            },
        ]

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            response = await get_audit_event(
                request_id="req-123",
                user_context={"username": "admin"},
                log_type="registry_api_access",
            )

            assert len(response["events"]) == 1
            mock_repo.find.assert_called_once_with(
                {
                    "request_id": "req-123",
                    "log_type": "registry_api_access",
                },
                limit=10,
            )

    async def test_returns_404_when_not_found(self):
        """Endpoint returns 404 for unknown request_id."""
        mock_repo = AsyncMock()
        mock_repo.find.return_value = []

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_audit_event(
                    request_id="nonexistent",
                    user_context={"username": "admin"},
                )

            assert exc_info.value.status_code == 404

    async def test_without_log_type_queries_all(self):
        """Endpoint queries without log_type filter when not provided."""
        mock_repo = AsyncMock()
        mock_repo.find.return_value = [
            {"request_id": "req-123", "log_type": "mcp_server_access"},
        ]

        with patch(
            "registry.audit.routes.get_audit_repository",
            return_value=mock_repo,
        ):
            await get_audit_event(
                request_id="req-123",
                user_context={"username": "admin"},
                log_type=None,
            )

            mock_repo.find.assert_called_once_with(
                {"request_id": "req-123"},
                limit=10,
            )
