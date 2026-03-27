"""Unit tests for RegistryCard repository."""

from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from registry.repositories.documentdb.registry_card_repository import (
    DocumentDBRegistryCardRepository,
)
from registry.schemas.registry_card import (
    RegistryAuthConfig,
    RegistryCapabilities,
    RegistryCard,
    RegistryContact,
)


@pytest.fixture
def mock_collection():
    """Fixture for mock MongoDB collection."""
    collection = AsyncMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.replace_one = AsyncMock()
    return collection


@pytest.fixture
def sample_registry_card():
    """Fixture for sample RegistryCard."""
    return RegistryCard(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Test Registry",
        description="A test registry",
        federation_endpoint="https://registry.example.com/api/v1/federation",
        contact=RegistryContact(email="admin@example.com"),
        metadata={"region": "us-east-1"},
    )


@pytest.mark.unit
class TestDocumentDBRegistryCardRepository:
    """Tests for DocumentDB RegistryCard repository."""

    @pytest.mark.asyncio
    async def test_get_when_no_card_exists(self, mock_collection):
        """Test get() returns None when no card exists."""
        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.get()

        assert result is None
        mock_collection.find_one.assert_called_once_with({"_id": "default"})

    @pytest.mark.asyncio
    async def test_get_when_card_exists(self, mock_collection, sample_registry_card):
        """Test get() returns RegistryCard when it exists."""
        stored_doc = sample_registry_card.model_dump(mode="json")
        stored_doc["_id"] = "default"
        stored_doc["created_at"] = "2024-01-01T00:00:00Z"
        stored_doc["updated_at"] = "2024-01-01T00:00:00Z"
        mock_collection.find_one.return_value = stored_doc

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.get()

        assert result is not None
        assert isinstance(result, RegistryCard)
        assert str(result.id) == str(sample_registry_card.id)
        assert result.name == sample_registry_card.name
        assert result.description == sample_registry_card.description
        mock_collection.find_one.assert_called_once_with({"_id": "default"})

    @pytest.mark.asyncio
    async def test_save_creates_new_card(self, mock_collection, sample_registry_card):
        """Test save() creates a new card when none exists."""
        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.save(sample_registry_card)

        assert result == sample_registry_card
        mock_collection.replace_one.assert_called_once()

        # Verify the document structure
        call_args = mock_collection.replace_one.call_args
        filter_dict = call_args[0][0]
        document = call_args[0][1]
        options = call_args[1]

        assert filter_dict == {"_id": "default"}
        assert document["_id"] == "default"
        assert document["id"] == str(sample_registry_card.id)
        assert document["name"] == sample_registry_card.name
        assert "created_at" in document
        assert "updated_at" in document
        assert options["upsert"] is True

    @pytest.mark.asyncio
    async def test_save_updates_existing_card(self, mock_collection, sample_registry_card):
        """Test save() updates an existing card."""
        existing_doc = sample_registry_card.model_dump(mode="json")
        existing_doc["_id"] = "default"
        existing_doc["created_at"] = "2024-01-01T00:00:00Z"
        existing_doc["updated_at"] = "2024-01-01T00:00:00Z"
        mock_collection.find_one.return_value = existing_doc

        # Create updated card
        updated_card = RegistryCard(
            id=sample_registry_card.id,
            name="Updated Name",
            description="Updated description",
            federation_endpoint=sample_registry_card.federation_endpoint,
        )

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.save(updated_card)

        assert result == updated_card
        mock_collection.replace_one.assert_called_once()

        # Verify the document preserves created_at but updates updated_at
        call_args = mock_collection.replace_one.call_args
        document = call_args[0][1]

        assert document["created_at"] == "2024-01-01T00:00:00Z"
        assert document["updated_at"] != "2024-01-01T00:00:00Z"
        assert document["name"] == "Updated Name"
        assert document["description"] == "Updated description"

    @pytest.mark.asyncio
    async def test_save_preserves_all_fields(self, mock_collection):
        """Test save() preserves all RegistryCard fields."""
        contact = RegistryContact(
            email="admin@full.example.com", url="https://full.example.com/contact"
        )
        card = RegistryCard(
            schema_version="1.1.0",
            id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            name="Full Test Registry",
            description="Complete test",
            federation_api_version="2.0.0",
            federation_endpoint="https://full.example.com/api/v1/federation",
            contact=contact,
            capabilities=RegistryCapabilities(servers=False, agents=True),
            authentication=RegistryAuthConfig(
                schemes=["bearer"], oauth2_issuer="https://auth.test.com"
            ),
            visibility_policy="authenticated",
            metadata={"tier": "production", "region": "us-west-2"},
        )

        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        await repo.save(card)

        call_args = mock_collection.replace_one.call_args
        document = call_args[0][1]

        assert document["schema_version"] == "1.1.0"
        assert document["id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        assert document["name"] == "Full Test Registry"
        assert document["description"] == "Complete test"
        assert document["federation_api_version"] == "2.0.0"
        assert document["contact"]["email"] == "admin@full.example.com"
        assert document["contact"]["url"] == "https://full.example.com/contact"
        assert document["capabilities"]["servers"] is False
        assert document["capabilities"]["agents"] is True
        assert document["authentication"]["schemes"] == ["bearer"]
        assert document["visibility_policy"] == "authenticated"
        assert document["metadata"]["tier"] == "production"

    @pytest.mark.asyncio
    async def test_fixed_id_always_default(self, mock_collection, sample_registry_card):
        """Test that repository always uses fixed _id: 'default'."""
        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        await repo.save(sample_registry_card)

        # Check find_one was called with default ID
        mock_collection.find_one.assert_called_with({"_id": "default"})

        # Check replace_one was called with default ID
        call_args = mock_collection.replace_one.call_args
        filter_dict = call_args[0][0]
        document = call_args[0][1]

        assert filter_dict == {"_id": "default"}
        assert document["_id"] == "default"

    @pytest.mark.asyncio
    async def test_upsert_option_enabled(self, mock_collection, sample_registry_card):
        """Test that upsert option is enabled for replace_one."""
        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        await repo.save(sample_registry_card)

        call_args = mock_collection.replace_one.call_args
        options = call_args[1]

        assert options["upsert"] is True

    @pytest.mark.asyncio
    async def test_get_handles_missing_optional_fields(self, mock_collection):
        """Test get() handles documents with missing optional fields gracefully."""
        # Minimal document with only required fields
        minimal_doc = {
            "_id": "default",
            "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "name": "Minimal Registry",
            "federation_endpoint": "https://minimal.example.com/api/v1/federation",
            "federation_api_version": "1.0",
            "schema_version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        mock_collection.find_one.return_value = minimal_doc

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.get()

        assert result is not None
        assert str(result.id) == "cccccccc-cccc-cccc-cccc-cccccccccccc"
        assert result.name == "Minimal Registry"
        assert result.description is None
        assert result.contact is None

    @pytest.mark.asyncio
    async def test_collection_name_is_correct(self):
        """Test that repository uses correct collection name."""
        repo = DocumentDBRegistryCardRepository()
        # The collection name should follow the pattern from get_collection_name
        assert "registry_cards" in repo._collection_name

    @pytest.mark.asyncio
    async def test_lazy_initialization_of_collection(self):
        """Test that collection is lazily initialized."""
        repo = DocumentDBRegistryCardRepository()

        # Initially, collection should be None
        assert repo._collection is None

        # After first operation, collection should be initialized
        with patch.object(repo, "_get_collection", new_callable=AsyncMock) as mock_get:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=None)
            mock_get.return_value = mock_collection

            await repo.get()

            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_handles_none_optional_fields(self, mock_collection):
        """Test save() correctly handles None values in optional fields."""
        card = RegistryCard(
            id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
            name="Test",
            federation_endpoint="https://example.com/api/v1/federation",
            description=None,
            contact=None,
        )

        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        await repo.save(card)

        call_args = mock_collection.replace_one.call_args
        document = call_args[0][1]

        assert document["description"] is None
        assert document["contact"] is None

    @pytest.mark.asyncio
    async def test_get_returns_card_with_default_capabilities(self, mock_collection):
        """Test get() returns card with default capabilities if not specified."""
        doc = {
            "_id": "default",
            "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "name": "Test",
            "federation_endpoint": "https://example.com/api/v1/federation",
            "federation_api_version": "1.0",
            "schema_version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            # capabilities and authentication not explicitly set
        }
        mock_collection.find_one.return_value = doc

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        result = await repo.get()

        assert result is not None
        # Pydantic should apply defaults
        assert result.capabilities.servers is True
        assert result.authentication.schemes == ["oauth2", "bearer"]

    @pytest.mark.asyncio
    async def test_save_timestamps_are_iso_format(self, mock_collection, sample_registry_card):
        """Test that save() creates ISO format timestamps."""
        mock_collection.find_one.return_value = None

        repo = DocumentDBRegistryCardRepository()
        repo._collection = mock_collection

        await repo.save(sample_registry_card)

        call_args = mock_collection.replace_one.call_args
        document = call_args[0][1]

        created_at = document["created_at"]
        updated_at = document["updated_at"]

        # Verify ISO format by parsing
        assert isinstance(created_at, str)
        assert isinstance(updated_at, str)
        # Should be valid ISO timestamps
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
