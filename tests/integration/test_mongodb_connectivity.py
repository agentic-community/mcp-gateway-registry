"""
Simple MongoDB connectivity tests.

These tests verify basic MongoDB connectivity and CRUD operations
without complex fixture dependencies.
"""

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from registry.core.config import Settings


@pytest.mark.integration
@pytest.mark.asyncio
class TestMongoDBConnectivity:
    """Test basic MongoDB connectivity."""

    @pytest.mark.skip(reason="Requires MongoDB running - not available in CI environment")
    async def test_mongodb_connection(self):
        """Test that we can connect to MongoDB."""
        # Arrange - Use localhost with directConnection for single server
        client = AsyncIOMotorClient(
            "mongodb://localhost:27017",
            directConnection=True,  # Bypass replica set discovery
            serverSelectionTimeoutMS=5000,
        )

        # Act & Assert - connection happens on first operation
        try:
            # Ping the server
            await client.admin.command("ping")
            assert True, "Successfully connected to MongoDB"
        finally:
            client.close()

    @pytest.mark.skip(reason="Requires MongoDB running - not available in CI environment")
    async def test_mongodb_create_and_read_document(self):
        """Test basic CRUD: create and read a document."""
        # Arrange - Use localhost with directConnection
        client = AsyncIOMotorClient("mongodb://localhost:27017", directConnection=True)
        db = client["test_mcp_registry"]
        collection = db["test_connectivity"]

        try:
            # Act - Insert a test document
            test_doc = {
                "test_id": "connectivity_test_1",
                "message": "Hello MongoDB",
                "status": "testing",
            }
            result = await collection.insert_one(test_doc)

            # Assert - Document was inserted
            assert result.inserted_id is not None

            # Act - Read the document back
            found_doc = await collection.find_one({"test_id": "connectivity_test_1"})

            # Assert - Document matches what we inserted
            assert found_doc is not None
            assert found_doc["message"] == "Hello MongoDB"
            assert found_doc["status"] == "testing"

        finally:
            # Cleanup
            await collection.delete_many({"test_id": "connectivity_test_1"})
            client.close()

    @pytest.mark.skip(reason="Requires MongoDB running - not available in CI environment")
    async def test_mongodb_update_and_delete_document(self):
        """Test basic CRUD: update and delete a document."""
        # Arrange - Use localhost with directConnection
        client = AsyncIOMotorClient("mongodb://localhost:27017", directConnection=True)
        db = client["test_mcp_registry"]
        collection = db["test_connectivity"]

        try:
            # Act - Insert a test document
            test_doc = {"test_id": "connectivity_test_2", "value": 100, "status": "initial"}
            await collection.insert_one(test_doc)

            # Act - Update the document
            await collection.update_one(
                {"test_id": "connectivity_test_2"}, {"$set": {"value": 200, "status": "updated"}}
            )

            # Assert - Document was updated
            updated_doc = await collection.find_one({"test_id": "connectivity_test_2"})
            assert updated_doc["value"] == 200
            assert updated_doc["status"] == "updated"

            # Act - Delete the document
            delete_result = await collection.delete_one({"test_id": "connectivity_test_2"})

            # Assert - Document was deleted
            assert delete_result.deleted_count == 1

            # Verify document is gone
            deleted_doc = await collection.find_one({"test_id": "connectivity_test_2"})
            assert deleted_doc is None

        finally:
            # Cleanup (just in case)
            await collection.delete_many({"test_id": "connectivity_test_2"})
            client.close()
