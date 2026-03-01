"""
Unit tests for Audit Repository.

Validates: Requirements 6.1, 6.2
"""

from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.audit.models import Identity, RegistryApiAccessRecord, Request, Response
from registry.repositories.audit_repository import DocumentDBAuditRepository


def make_test_record(request_id: str = "test-123") -> RegistryApiAccessRecord:
    """Create a test audit record."""
    return RegistryApiAccessRecord(
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        identity=Identity(
            username="testuser", auth_method="oauth2", credential_type="bearer_token"
        ),
        request=Request(method="GET", path="/api/test", client_ip="127.0.0.1"),
        response=Response(status_code=200, duration_ms=50.5),
    )


class TestFind:
    """Tests for find() method."""

    async def test_returns_list_of_events(self):
        """find() returns a list of audit events."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        test_docs = [{"request_id": "req-1"}, {"request_id": "req-2"}]

        async def async_iter():
            for doc in test_docs:
                yield doc

        mock_cursor.__aiter__ = lambda self: async_iter()
        mock_collection.find = MagicMock(return_value=mock_cursor)

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            results = await repo.find({})

            assert len(results) == 2

    async def test_applies_pagination(self):
        """find() applies limit and offset."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        async def async_iter():
            return
            yield

        mock_cursor.__aiter__ = lambda self: async_iter()
        mock_collection.find = MagicMock(return_value=mock_cursor)

        with patch.object(
            DocumentDBAuditRepository,
            "_get_collection",
            new_callable=AsyncMock,
            return_value=mock_collection,
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection
            await repo.find({}, limit=25, offset=50)

            mock_cursor.skip.assert_called_once_with(50)
            mock_cursor.limit.assert_called_once_with(25)


class TestInsert:
    """Tests for insert() method."""

    async def test_writes_record(self):
        """insert() writes the audit record to MongoDB."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="new_id")

        with patch.object(
            DocumentDBAuditRepository, "_get_collection", return_value=mock_collection
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            result = await repo.insert(make_test_record())

            assert result is True
            mock_collection.insert_one.assert_called_once()

    async def test_returns_false_on_error(self):
        """insert() returns False when an error occurs."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.side_effect = Exception("Database error")

        with patch.object(
            DocumentDBAuditRepository, "_get_collection", return_value=mock_collection
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            result = await repo.insert(make_test_record())

            assert result is False

    async def test_returns_true_on_duplicate_key(self):
        """insert() returns True when a duplicate audit event already exists."""
        mock_collection = AsyncMock()
        mock_collection.insert_one.side_effect = DuplicateKeyError("duplicate key error")

        with patch.object(
            DocumentDBAuditRepository, "_get_collection", return_value=mock_collection
        ):
            repo = DocumentDBAuditRepository()
            repo._collection = mock_collection

            result = await repo.insert(make_test_record())

            assert result is True
