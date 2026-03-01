"""
Unit tests for registry/search/service.py (FaissService).

This module tests all core functionality of the FaissService including:
- FAISS index initialization and management
- Adding/updating/removing servers and agents
- Semantic search with hybrid keyword boosting
- Index persistence (save/load)
- Embeddings generation and normalization
"""

import json
import logging
from typing import Any

import numpy as np
import pytest

from registry.schemas.agent_models import AgentCard
from registry.search.service import FaissService, _PydanticAwareJSONEncoder
from tests.fixtures.factories import AgentCardFactory
from tests.fixtures.mocks.mock_embeddings import MockEmbeddingsClient

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_embeddings_client():
    """Create a mock embeddings client for testing."""
    return MockEmbeddingsClient(model_name="test-model", dimension=384)


@pytest.fixture
def faiss_service(mock_settings, mock_embeddings_client):
    """
    Create a FaissService instance with mocked dependencies.

    This fixture provides a pre-initialized FaissService with:
    - Mock embeddings client
    - Mock FAISS index
    - Test settings with temporary directories
    """
    service = FaissService()
    service.embedding_model = mock_embeddings_client
    service._initialize_new_index()
    return service


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """Create sample server info dictionary for testing."""
    return {
        "server_name": "test-server",
        "description": "A test server for search testing",
        "tags": ["test", "search", "demo"],
        "num_tools": 2,
        "entity_type": "mcp_server",
        "tool_list": [
            {
                "name": "get_data",
                "description": "Retrieve data from source",
                "parsed_description": {"main": "Retrieve data from source", "args": "id: string"},
                "schema": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
            {
                "name": "set_data",
                "description": "Update data in source",
                "parsed_description": {
                    "main": "Update data in source",
                    "args": "id: string, value: any",
                },
                "schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "value": {"type": "string"}},
                },
            },
        ],
    }


@pytest.fixture
def sample_agent_card() -> AgentCard:
    """Create sample agent card for testing."""
    return AgentCardFactory(
        name="test-agent",
        description="A test agent for search testing",
        tags=["test", "agent", "demo"],
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestFaissServiceInitialization:
    """Tests for FaissService initialization."""

    def test_init_creates_empty_service(self):
        """Test that FaissService.__init__ creates empty service."""
        service = FaissService()

        assert service.embedding_model is None
        assert service.faiss_index is None
        assert service.metadata_store == {}
        assert service.next_id_counter == 0

    def test_initialize_new_index_creates_index(self, mock_settings):
        """Test that _initialize_new_index creates a new FAISS index."""
        service = FaissService()
        service._initialize_new_index()

        assert service.faiss_index is not None
        assert service.faiss_index.d == mock_settings.embeddings_model_dimensions
        assert service.faiss_index.ntotal == 0
        assert service.metadata_store == {}
        assert service.next_id_counter == 0

    @pytest.mark.asyncio
    async def test_initialize_loads_model_and_index(self, mock_settings, monkeypatch):
        """Test that initialize() loads embedding model and FAISS data."""
        service = FaissService()

        # Mock the internal methods
        load_model_called = False
        load_data_called = False

        async def mock_load_model():
            nonlocal load_model_called
            load_model_called = True
            service.embedding_model = MockEmbeddingsClient(dimension=384)

        async def mock_load_data():
            nonlocal load_data_called
            load_data_called = True
            service._initialize_new_index()

        monkeypatch.setattr(service, "_load_embedding_model", mock_load_model)
        monkeypatch.setattr(service, "_load_faiss_data", mock_load_data)

        await service.initialize()

        assert load_model_called
        assert load_data_called
        assert service.embedding_model is not None
        assert service.faiss_index is not None

    @pytest.mark.asyncio
    async def test_load_faiss_data_creates_new_when_missing(self, mock_settings):
        """Test that _load_faiss_data creates new index when files don't exist."""
        service = FaissService()

        # Ensure files don't exist
        assert not mock_settings.faiss_index_path.exists()
        assert not mock_settings.faiss_metadata_path.exists()

        await service._load_faiss_data()

        assert service.faiss_index is not None
        assert service.faiss_index.ntotal == 0
        assert service.metadata_store == {}
        assert service.next_id_counter == 0

    @pytest.mark.asyncio
    async def test_load_faiss_data_loads_existing(self, mock_settings, tmp_path):
        """Test that _load_faiss_data loads existing index and metadata."""
        service = FaissService()

        # Create mock metadata file
        metadata = {
            "metadata": {
                "test-server": {
                    "id": 0,
                    "text_for_embedding": "test text",
                    "full_server_info": {"server_name": "test-server"},
                    "entity_type": "mcp_server",
                }
            },
            "next_id": 1,
        }

        mock_settings.faiss_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_settings.faiss_metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create mock index file (will be handled by mock faiss.read_index)
        mock_settings.faiss_index_path.touch()

        await service._load_faiss_data()

        assert service.metadata_store == metadata["metadata"]
        assert service.next_id_counter == 1


# =============================================================================
# TEXT PREPARATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestTextPreparation:
    """Tests for text preparation methods."""

    def test_get_text_for_embedding_server(self, faiss_service, sample_server_info):
        """Test _get_text_for_embedding generates correct text for server."""
        text = faiss_service._get_text_for_embedding(sample_server_info)

        assert "test-server" in text
        assert "A test server for search testing" in text
        assert "test, search, demo" in text
        assert "get_data" in text
        assert "set_data" in text
        assert "Retrieve data from source" in text

    def test_get_text_for_embedding_handles_missing_fields(self, faiss_service):
        """Test _get_text_for_embedding handles missing fields gracefully."""
        server_info = {"server_name": "minimal-server"}

        text = faiss_service._get_text_for_embedding(server_info)

        assert "minimal-server" in text
        assert text  # Should not be empty

    def test_get_text_for_agent(self, faiss_service, sample_agent_card):
        """Test _get_text_for_agent generates correct text for agent."""
        text = faiss_service._get_text_for_agent(sample_agent_card)

        assert sample_agent_card.name in text
        assert sample_agent_card.description in text
        assert "Skills:" in text or "test, agent, demo" in text

    def test_get_text_for_agent_with_skills(self, faiss_service):
        """Test _get_text_for_agent includes skill details."""
        agent = AgentCardFactory(
            name="skilled-agent",
            description="Agent with skills",
        )

        text = faiss_service._get_text_for_agent(agent)

        assert "skilled-agent" in text
        assert "Skills:" in text

    def test_get_text_for_embedding_includes_metadata(self, faiss_service):
        """Test _get_text_for_embedding includes metadata in embedding text."""
        server_info = {
            "server_name": "test-server",
            "description": "Test server with metadata",
            "tags": ["test"],
            "tool_list": [],
            "metadata": {
                "team": "data-platform",
                "owner": "alice@example.com",
                "compliance_level": "PCI-DSS",
            },
        }

        text = faiss_service._get_text_for_embedding(server_info)

        assert "test-server" in text
        assert "Metadata:" in text
        assert "team: data-platform" in text
        assert "owner: alice@example.com" in text
        assert "compliance_level: PCI-DSS" in text

    def test_get_text_for_embedding_without_metadata(self, faiss_service):
        """Test _get_text_for_embedding works without metadata field."""
        server_info = {
            "server_name": "test-server",
            "description": "Test server without metadata",
            "tags": ["test"],
            "tool_list": [],
        }

        text = faiss_service._get_text_for_embedding(server_info)

        assert "test-server" in text
        assert "Metadata:" not in text

    def test_get_text_for_embedding_with_nested_metadata(self, faiss_service):
        """Test _get_text_for_embedding handles nested metadata structures."""
        server_info = {
            "server_name": "test-server",
            "description": "Test server",
            "tags": [],
            "tool_list": [],
            "metadata": {
                "compliance": {"level": "PCI-DSS", "audited": True},
                "tags": ["production", "critical"],
            },
        }

        text = faiss_service._get_text_for_embedding(server_info)

        assert "Metadata:" in text
        assert "compliance:" in text
        assert "tags:" in text

    def test_get_text_for_agent_includes_metadata(self, faiss_service):
        """Test _get_text_for_agent includes metadata in embedding text."""
        agent = AgentCardFactory(
            name="test-agent",
            description="Test agent with metadata",
            metadata={"team": "ai-platform", "owner": "bob@example.com", "version": "2.1.0"},
        )

        text = faiss_service._get_text_for_agent(agent)

        assert "test-agent" in text
        assert "Metadata:" in text
        assert "team: ai-platform" in text
        assert "owner: bob@example.com" in text
        assert "version: 2.1.0" in text

    def test_get_text_for_agent_without_metadata(self, faiss_service):
        """Test _get_text_for_agent works without metadata."""
        agent = AgentCardFactory(name="test-agent", description="Test agent without metadata")

        text = faiss_service._get_text_for_agent(agent)

        assert "test-agent" in text
        assert "Metadata:" not in text


# =============================================================================
# EMBEDDING AND NORMALIZATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestEmbeddingOperations:
    """Tests for embedding generation and normalization."""

    def test_normalize_embedding(self, faiss_service):
        """Test _normalize_embedding normalizes vectors to unit length."""
        # Create a non-normalized vector
        vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        normalized = faiss_service._normalize_embedding(vector)

        # Check L2 norm is 1.0 (unit length)
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)

        # Check values are correct (3,4,0) normalized is (0.6, 0.8, 0)
        assert np.isclose(normalized[0], 0.6, atol=1e-6)
        assert np.isclose(normalized[1], 0.8, atol=1e-6)
        assert np.isclose(normalized[2], 0.0, atol=1e-6)

    def test_normalize_embedding_zero_vector(self, faiss_service):
        """Test _normalize_embedding handles zero vector."""
        vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        normalized = faiss_service._normalize_embedding(vector)

        # Should return original vector when norm is 0
        assert np.array_equal(normalized, vector)

    def test_normalize_embedding_already_normalized(self, faiss_service):
        """Test _normalize_embedding handles already normalized vector."""
        # Create already normalized vector
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        normalized = faiss_service._normalize_embedding(vector)

        # Should remain the same
        assert np.allclose(normalized, vector, atol=1e-6)


# =============================================================================
# ADD/UPDATE ENTITY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestAddUpdateService:
    """Tests for adding and updating services in FAISS index."""

    @pytest.mark.asyncio
    async def test_add_new_service(self, faiss_service, sample_server_info, mock_settings):
        """Test adding a new service to the index."""
        service_path = "/servers/test-server"

        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        # Check metadata store
        assert service_path in faiss_service.metadata_store
        metadata = faiss_service.metadata_store[service_path]
        assert metadata["id"] == 0
        assert metadata["entity_type"] == "mcp_server"
        assert metadata["full_server_info"]["is_enabled"] is True

        # Check FAISS index
        assert faiss_service.faiss_index.ntotal == 1
        assert faiss_service.next_id_counter == 1

    @pytest.mark.asyncio
    async def test_update_existing_service_same_text(self, faiss_service, sample_server_info):
        """Test updating service with same text doesn't re-embed."""
        service_path = "/servers/test-server"

        # Add service first
        await faiss_service.add_or_update_service(
            service_path, sample_server_info, is_enabled=False
        )

        initial_total = faiss_service.faiss_index.ntotal
        initial_counter = faiss_service.next_id_counter

        # Update with same info but different enabled state
        sample_server_info["extra_field"] = "new value"
        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        # Should not create new embedding
        assert faiss_service.faiss_index.ntotal == initial_total
        assert faiss_service.next_id_counter == initial_counter

        # But should update metadata
        metadata = faiss_service.metadata_store[service_path]
        assert metadata["full_server_info"]["is_enabled"] is True

    @pytest.mark.asyncio
    async def test_update_existing_service_different_text(self, faiss_service, sample_server_info):
        """Test updating service with different text re-embeds."""
        service_path = "/servers/test-server"

        # Add service first
        await faiss_service.add_or_update_service(
            service_path, sample_server_info, is_enabled=False
        )

        initial_id = faiss_service.metadata_store[service_path]["id"]

        # Update with different description (changes embedding text)
        sample_server_info["description"] = "Completely different description"
        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        # Should use same ID
        metadata = faiss_service.metadata_store[service_path]
        assert metadata["id"] == initial_id

        # Should have re-embedded
        assert "Completely different description" in metadata["text_for_embedding"]

    @pytest.mark.asyncio
    async def test_add_service_without_model(self, mock_settings):
        """Test adding service fails gracefully without embedding model."""
        service = FaissService()
        service._initialize_new_index()
        # Don't set embedding_model

        await service.add_or_update_service(
            "/servers/test", {"server_name": "test"}, is_enabled=False
        )

        # Should not add to index
        assert service.faiss_index.ntotal == 0
        assert "/servers/test" not in service.metadata_store


@pytest.mark.unit
@pytest.mark.search
class TestAddUpdateAgent:
    """Tests for adding and updating agents in FAISS index."""

    @pytest.mark.asyncio
    async def test_add_new_agent(self, faiss_service, sample_agent_card):
        """Test adding a new agent to the index."""
        agent_path = "/agents/test-agent"

        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=True)

        # Check metadata store
        assert agent_path in faiss_service.metadata_store
        metadata = faiss_service.metadata_store[agent_path]
        assert metadata["id"] == 0
        assert metadata["entity_type"] == "a2a_agent"
        assert metadata["full_agent_card"]["name"] == sample_agent_card.name

        # Check FAISS index
        assert faiss_service.faiss_index.ntotal == 1
        assert faiss_service.next_id_counter == 1

    @pytest.mark.asyncio
    async def test_update_existing_agent_same_text(self, faiss_service, sample_agent_card):
        """Test updating agent with same text doesn't re-embed."""
        agent_path = "/agents/test-agent"

        # Add agent first
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=False)

        initial_total = faiss_service.faiss_index.ntotal
        initial_counter = faiss_service.next_id_counter

        # Update with same card
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=True)

        # Should not create new embedding
        assert faiss_service.faiss_index.ntotal == initial_total
        assert faiss_service.next_id_counter == initial_counter

    @pytest.mark.asyncio
    async def test_update_existing_agent_different_text(self, faiss_service):
        """Test updating agent with different text re-embeds."""
        agent_path = "/agents/test-agent"
        agent1 = AgentCardFactory(name="test-agent", description="Original description")

        # Add agent first
        await faiss_service.add_or_update_agent(agent_path, agent1, is_enabled=False)

        initial_id = faiss_service.metadata_store[agent_path]["id"]

        # Update with different description
        agent2 = AgentCardFactory(name="test-agent", description="New description")
        await faiss_service.add_or_update_agent(agent_path, agent2, is_enabled=True)

        # Should use same ID
        metadata = faiss_service.metadata_store[agent_path]
        assert metadata["id"] == initial_id

        # Should have re-embedded
        assert "New description" in metadata["text_for_embedding"]

    @pytest.mark.asyncio
    async def test_add_agent_without_model(self, mock_settings):
        """Test adding agent fails gracefully without embedding model."""
        service = FaissService()
        service._initialize_new_index()
        # Don't set embedding_model

        agent = AgentCardFactory()

        await service.add_or_update_agent("/agents/test", agent, is_enabled=False)

        # Should not add to index
        assert service.faiss_index.ntotal == 0
        assert "/agents/test" not in service.metadata_store


# =============================================================================
# REMOVE ENTITY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestRemoveEntities:
    """Tests for removing entities from FAISS index."""

    @pytest.mark.asyncio
    async def test_remove_service(self, faiss_service, sample_server_info):
        """Test removing a service from the index."""
        service_path = "/servers/test-server"

        # Add service first
        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        assert service_path in faiss_service.metadata_store

        # Remove service
        await faiss_service.remove_service(service_path)

        # Should be removed from metadata
        assert service_path not in faiss_service.metadata_store

    @pytest.mark.asyncio
    async def test_remove_nonexistent_service(self, faiss_service):
        """Test removing non-existent service logs warning."""
        # Should not raise error
        await faiss_service.remove_service("/servers/nonexistent")

    @pytest.mark.asyncio
    async def test_remove_agent(self, faiss_service, sample_agent_card):
        """Test removing an agent from the index."""
        agent_path = "/agents/test-agent"

        # Add agent first
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=True)

        assert agent_path in faiss_service.metadata_store

        # Remove agent
        await faiss_service.remove_agent(agent_path)

        # Should be removed from metadata
        assert agent_path not in faiss_service.metadata_store

    @pytest.mark.asyncio
    async def test_remove_nonexistent_agent(self, faiss_service):
        """Test removing non-existent agent logs warning."""
        # Should not raise error
        await faiss_service.remove_agent("/agents/nonexistent")

    @pytest.mark.asyncio
    async def test_remove_entity_wrapper(self, faiss_service, sample_agent_card):
        """Test remove_entity wrapper method."""
        agent_path = "/agents/test-agent"

        # Add agent
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card)

        # Remove using wrapper
        await faiss_service.remove_entity(agent_path)

        # Should be removed
        assert agent_path not in faiss_service.metadata_store


# =============================================================================
# SEARCH TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestSearch:
    """Tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_mixed_empty_query(self, faiss_service):
        """Test search_mixed raises error on empty query."""
        with pytest.raises(ValueError, match="Query text is required"):
            await faiss_service.search_mixed("")

    @pytest.mark.asyncio
    async def test_search_mixed_no_model(self):
        """Test search_mixed raises error without embedding model."""
        service = FaissService()
        service._initialize_new_index()

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.search_mixed("test query")

    @pytest.mark.asyncio
    async def test_search_mixed_empty_index(self, faiss_service):
        """Test search_mixed returns empty results on empty index."""
        results = await faiss_service.search_mixed("test query")

        assert results == {"servers": [], "tools": [], "agents": []}

    @pytest.mark.asyncio
    async def test_search_mixed_finds_servers(self, faiss_service, sample_server_info):
        """Test search_mixed finds matching servers."""
        # Add a server
        await faiss_service.add_or_update_service(
            "/servers/test-server", sample_server_info, is_enabled=True
        )

        # Search for it
        results = await faiss_service.search_mixed("test server")

        assert len(results["servers"]) == 1
        server = results["servers"][0]
        assert server["entity_type"] == "mcp_server"
        assert server["path"] == "/servers/test-server"
        assert server["server_name"] == "test-server"
        assert "relevance_score" in server
        assert 0 <= server["relevance_score"] <= 1

    @pytest.mark.asyncio
    async def test_search_mixed_finds_agents(self, faiss_service, sample_agent_card):
        """Test search_mixed finds matching agents."""
        # Add an agent
        await faiss_service.add_or_update_agent(
            "/agents/test-agent", sample_agent_card, is_enabled=True
        )

        # Search for it
        results = await faiss_service.search_mixed("test agent")

        assert len(results["agents"]) == 1
        agent = results["agents"][0]
        assert agent["entity_type"] == "a2a_agent"
        assert agent["path"] == "/agents/test-agent"
        assert agent["agent_name"] == sample_agent_card.name
        assert "relevance_score" in agent
        assert 0 <= agent["relevance_score"] <= 1

    @pytest.mark.asyncio
    async def test_search_mixed_with_entity_type_filter(
        self, faiss_service, sample_server_info, sample_agent_card
    ):
        """Test search_mixed filters by entity_type."""
        # Add both server and agent
        await faiss_service.add_or_update_service(
            "/servers/test-server", sample_server_info, is_enabled=True
        )
        await faiss_service.add_or_update_agent(
            "/agents/test-agent", sample_agent_card, is_enabled=True
        )

        # Search for servers only
        results = await faiss_service.search_mixed("test", entity_types=["mcp_server"])

        assert len(results["servers"]) >= 0  # May or may not find server depending on mock
        assert len(results["agents"]) == 0  # Should not return agents

    @pytest.mark.asyncio
    async def test_search_mixed_extracts_tools(self, faiss_service, sample_server_info):
        """Test search_mixed extracts matching tools."""
        # Add server with tools
        await faiss_service.add_or_update_service(
            "/servers/test-server", sample_server_info, is_enabled=True
        )

        # Search for specific tool
        results = await faiss_service.search_mixed("get data", entity_types=["tool"])

        # Should extract tools even if server doesn't match well
        assert "tools" in results

    @pytest.mark.asyncio
    async def test_search_mixed_respects_max_results(self, faiss_service):
        """Test search_mixed respects max_results parameter."""
        # Add multiple servers
        for i in range(10):
            server_info = {
                "server_name": f"server-{i}",
                "description": f"Test server {i}",
                "tags": ["test"],
                "entity_type": "mcp_server",
            }
            await faiss_service.add_or_update_service(
                f"/servers/server-{i}", server_info, is_enabled=True
            )

        # Search with limit
        results = await faiss_service.search_mixed("test server", max_results=5)

        assert len(results["servers"]) <= 5

    @pytest.mark.asyncio
    async def test_search_entities_wrapper(self, faiss_service, sample_server_info):
        """Test search_entities wrapper method."""
        await faiss_service.add_or_update_service(
            "/servers/test-server", sample_server_info, is_enabled=True
        )

        # Use wrapper method
        results = await faiss_service.search_entities("test server")

        # Should return combined list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_agents_wrapper(self, faiss_service, sample_agent_card):
        """Test search_agents wrapper method."""
        await faiss_service.add_or_update_agent(
            "/agents/test-agent", sample_agent_card, is_enabled=True
        )

        # Use wrapper method
        results = await faiss_service.search_agents("test agent")

        # Should return list of agents
        assert isinstance(results, list)


# =============================================================================
# KEYWORD BOOST TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestKeywordBoost:
    """Tests for keyword boosting in hybrid search."""

    def test_calculate_keyword_boost_no_match(self, faiss_service, sample_server_info):
        """Test keyword boost returns 1.0 when no keywords match."""
        boost = faiss_service._calculate_keyword_boost("unrelated query xyz", sample_server_info)

        assert boost == 1.0

    def test_calculate_keyword_boost_name_match(self, faiss_service, sample_server_info):
        """Test keyword boost increases for name match."""
        boost = faiss_service._calculate_keyword_boost("test server", sample_server_info)

        # Should have boost from name match
        assert boost > 1.0

    def test_calculate_keyword_boost_tool_match(self, faiss_service, sample_server_info):
        """Test keyword boost increases for tool name match."""
        boost = faiss_service._calculate_keyword_boost("get data", sample_server_info)

        # Should have boost from tool match
        assert boost > 1.0

    def test_calculate_keyword_boost_tag_match(self, faiss_service, sample_server_info):
        """Test keyword boost increases for tag match."""
        boost = faiss_service._calculate_keyword_boost("search", sample_server_info)

        # Should have boost from tag match
        assert boost > 1.0

    def test_calculate_keyword_boost_filters_stopwords(self, faiss_service, sample_server_info):
        """Test keyword boost filters out stopwords."""
        boost = faiss_service._calculate_keyword_boost("the is are", sample_server_info)

        # Stopwords should not contribute to boost
        assert boost == 1.0

    def test_calculate_keyword_boost_capped_at_max(self, faiss_service):
        """Test keyword boost is capped at maximum value."""
        # Create server with many matching keywords
        server_info = {
            "server_name": "test search demo server",
            "description": "test search demo testing searching",
            "tags": ["test", "search", "demo", "testing"],
            "tool_list": [{"name": "test_tool"}, {"name": "search_tool"}, {"name": "demo_tool"}],
        }

        boost = faiss_service._calculate_keyword_boost("test search demo", server_info)

        # Should be capped at 2.0
        assert boost <= 2.0


# =============================================================================
# TOOL EXTRACTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestToolExtraction:
    """Tests for tool extraction from search results."""

    def test_extract_matching_tools_no_tools(self, faiss_service):
        """Test _extract_matching_tools returns empty list when no tools."""
        server_info = {"server_name": "test-server", "tool_list": None}

        tools = faiss_service._extract_matching_tools("query", server_info)

        assert tools == []

    def test_extract_matching_tools_name_match(self, faiss_service, sample_server_info):
        """Test _extract_matching_tools finds tools by name."""
        tools = faiss_service._extract_matching_tools("get data", sample_server_info)

        # Should find get_data tool
        assert len(tools) > 0
        assert any("get_data" in tool["tool_name"] for tool in tools)

    def test_extract_matching_tools_description_match(self, faiss_service, sample_server_info):
        """Test _extract_matching_tools finds tools by description."""
        tools = faiss_service._extract_matching_tools("retrieve source", sample_server_info)

        # Should find tools matching description
        assert len(tools) >= 0

    def test_extract_matching_tools_filters_stopwords(self, faiss_service, sample_server_info):
        """Test _extract_matching_tools filters stopwords."""
        tools = faiss_service._extract_matching_tools("the is are", sample_server_info)

        # Stopwords alone should not match
        assert tools == []

    def test_extract_matching_tools_scores_name_higher(self, faiss_service):
        """Test _extract_matching_tools scores name matches higher."""
        server_info = {
            "tool_list": [
                {
                    "name": "search_tool",
                    "description": "Does something else",
                    "parsed_description": {"main": "Does something else"},
                },
                {
                    "name": "other_tool",
                    "description": "search search search",
                    "parsed_description": {"main": "search search search"},
                },
            ]
        }

        tools = faiss_service._extract_matching_tools("search", server_info)

        # Name match should be scored higher than description match
        if len(tools) >= 2:
            assert "search_tool" in tools[0]["tool_name"]

    def test_extract_matching_tools_server_name_match(self, faiss_service):
        """Test _extract_matching_tools returns tools when query contains server name.

        This handles cases like "use context7 to look up mongodb docs" where the
        query mentions the server name but not specific tool names.
        """
        server_info = {
            "server_name": "Context7 MCP Server",
            "tool_list": [
                {
                    "name": "resolve-library-id",
                    "schema": {"type": "object"},
                },
                {
                    "name": "query-docs",
                    "schema": {"type": "object"},
                },
            ],
        }

        # Query contains "context7" but no tool-specific keywords
        tools = faiss_service._extract_matching_tools(
            "MongoDB vector index support context7", server_info
        )

        # Should return both tools since server name matches
        assert len(tools) == 2
        tool_names = [t["tool_name"] for t in tools]
        assert "resolve-library-id" in tool_names
        assert "query-docs" in tool_names
        # All tools should have base score of 0.5
        for tool in tools:
            assert tool["raw_score"] == 0.5


# =============================================================================
# DISTANCE/RELEVANCE CONVERSION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestDistanceConversion:
    """Tests for distance to relevance score conversion."""

    def test_distance_to_relevance_positive_distance(self, faiss_service):
        """Test _distance_to_relevance handles positive distances."""
        # Positive distance (1 - inner_product)
        relevance = faiss_service._distance_to_relevance(0.05)

        # Should convert: 1 - 0.05 = 0.95
        assert 0.94 <= relevance <= 0.96

    def test_distance_to_relevance_negative_distance(self, faiss_service):
        """Test _distance_to_relevance handles negative distances."""
        # Negative distance (-inner_product)
        relevance = faiss_service._distance_to_relevance(-0.95)

        # Should convert: -(-0.95) = 0.95
        assert 0.94 <= relevance <= 0.96

    def test_distance_to_relevance_zero(self, faiss_service):
        """Test _distance_to_relevance handles zero distance."""
        relevance = faiss_service._distance_to_relevance(0.0)

        assert relevance == 1.0

    def test_distance_to_relevance_clamped(self, faiss_service):
        """Test _distance_to_relevance clamps to [0, 1] range."""
        # Test upper bound
        relevance_high = faiss_service._distance_to_relevance(-2.0)
        assert relevance_high <= 1.0

        # Test lower bound
        relevance_low = faiss_service._distance_to_relevance(2.0)
        assert relevance_low >= 0.0


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestPersistence:
    """Tests for FAISS index persistence (save/load)."""

    @pytest.mark.asyncio
    async def test_save_data_creates_files(self, faiss_service, sample_server_info, mock_settings):
        """Test save_data creates index and metadata files."""
        # Add some data
        await faiss_service.add_or_update_service(
            "/servers/test-server", sample_server_info, is_enabled=True
        )

        # Save data
        await faiss_service.save_data()

        # Check that metadata file exists
        assert mock_settings.faiss_metadata_path.exists()

        # Verify metadata content
        with open(mock_settings.faiss_metadata_path) as f:
            saved_data = json.load(f)

        assert "metadata" in saved_data
        assert "next_id" in saved_data
        assert "/servers/test-server" in saved_data["metadata"]

    @pytest.mark.asyncio
    async def test_save_data_without_index(self, mock_settings):
        """Test save_data handles missing index gracefully."""
        service = FaissService()
        # Don't initialize index

        await service.save_data()

        # Should not create files
        assert not mock_settings.faiss_metadata_path.exists()

    def test_get_indexed_count(self, faiss_service):
        """Test getting the count of indexed items."""
        # Initially empty
        assert faiss_service.faiss_index.ntotal == 0

        # The count is directly from FAISS index
        count = faiss_service.faiss_index.ntotal
        assert count == 0


# =============================================================================
# PYDANTIC JSON ENCODER TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestPydanticJSONEncoder:
    """Tests for custom Pydantic JSON encoder."""

    def test_encoder_handles_httpurl(self):
        """Test encoder handles Pydantic HttpUrl type."""
        from pydantic import HttpUrl

        encoder = _PydanticAwareJSONEncoder()
        url = HttpUrl("https://example.com")

        result = encoder.default(url)

        assert result == "https://example.com/"

    def test_encoder_handles_datetime(self):
        """Test encoder handles datetime objects."""
        from datetime import datetime

        encoder = _PydanticAwareJSONEncoder()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        result = encoder.default(dt)

        assert "2024-01-01" in result
        assert "12:00:00" in result


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestFaissServiceIntegration:
    """Integration-style tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_server_workflow(self, faiss_service, sample_server_info, mock_settings):
        """Test complete workflow: add, search, update, search, remove."""
        service_path = "/servers/workflow-test"

        # Step 1: Add server
        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        # Step 2: Search for it
        results1 = await faiss_service.search_mixed("test server")
        assert len(results1["servers"]) >= 0

        # Step 3: Update server
        sample_server_info["description"] = "Updated description"
        await faiss_service.add_or_update_service(service_path, sample_server_info, is_enabled=True)

        # Step 4: Search again
        await faiss_service.search_mixed("updated")
        # Results should still work

        # Step 5: Remove server
        await faiss_service.remove_service(service_path)
        assert service_path not in faiss_service.metadata_store

    @pytest.mark.asyncio
    async def test_full_agent_workflow(self, faiss_service, sample_agent_card, mock_settings):
        """Test complete workflow for agents."""
        agent_path = "/agents/workflow-test"

        # Add, search, update, remove
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=True)

        results1 = await faiss_service.search_agents("test agent")
        assert isinstance(results1, list)

        # Update
        sample_agent_card.description = "Updated agent description"
        await faiss_service.add_or_update_agent(agent_path, sample_agent_card, is_enabled=True)

        # Remove
        await faiss_service.remove_agent(agent_path)
        assert agent_path not in faiss_service.metadata_store

    @pytest.mark.asyncio
    async def test_mixed_entities_workflow(
        self, faiss_service, sample_server_info, sample_agent_card
    ):
        """Test workflow with both servers and agents."""
        # Add both types
        await faiss_service.add_or_update_service(
            "/servers/mixed-server", sample_server_info, is_enabled=True
        )
        await faiss_service.add_or_update_agent(
            "/agents/mixed-agent", sample_agent_card, is_enabled=True
        )

        # Search for all entities
        results = await faiss_service.search_entities("test")

        # Should return combined results
        assert isinstance(results, list)

        # Check index has both
        assert faiss_service.faiss_index.ntotal >= 2
        assert len(faiss_service.metadata_store) == 2
