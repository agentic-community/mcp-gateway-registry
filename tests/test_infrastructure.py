"""
Test to verify test infrastructure is working correctly.

This test file validates that all mocking, fixtures, and test utilities
are functioning as expected.
"""

import numpy as np

from tests.fixtures.constants import TEST_AGENT_NAME_1, TEST_SERVER_NAME_1
from tests.fixtures.factories import AgentCardFactory, ServerDetailFactory
from tests.fixtures.helpers import create_minimal_agent_dict, create_minimal_server_dict
from tests.fixtures.mocks.mock_auth import MockJWTValidator
from tests.fixtures.mocks.mock_embeddings import MockEmbeddingsClient
from tests.fixtures.mocks.mock_faiss import MockFaissIndex
from tests.fixtures.mocks.mock_http import MockResponse


class TestInfrastructure:
    """Test the test infrastructure components."""

    def test_constants_imported(self):
        """Test that constants can be imported and accessed."""
        assert TEST_SERVER_NAME_1 == "com.example.test-server-1"
        assert TEST_AGENT_NAME_1 == "test-agent-1"

    def test_mock_faiss_index(self):
        """Test MockFaissIndex basic functionality."""
        index = MockFaissIndex(dimension=384)

        assert index.d == 384
        assert index.ntotal == 0

        # Add some vectors
        vectors = np.random.randn(5, 384).astype(np.float32)
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        index.add_with_ids(vectors, ids)

        assert index.ntotal == 5

        # Search
        query = np.random.randn(1, 384).astype(np.float32)
        distances, indices = index.search(query, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_mock_embeddings_client(self):
        """Test MockEmbeddingsClient."""
        client = MockEmbeddingsClient(dimension=384)

        texts = ["test sentence 1", "test sentence 2"]
        embeddings = client.encode(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

    def test_mock_jwt_validator(self):
        """Test MockJWTValidator."""
        validator = MockJWTValidator()

        token = validator.create_token(
            username="testuser", groups=["users"], scopes=["read:servers"]
        )

        assert isinstance(token, str)
        assert len(token) > 0

        # Validate the token
        payload = validator.validate_token(token)
        assert payload["username"] == "testuser"
        assert "users" in payload["groups"]

    def test_mock_http_response(self):
        """Test MockResponse."""
        response = MockResponse(status_code=200, json_data={"message": "success"})

        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_server_factory(self):
        """Test ServerDetailFactory."""
        server = ServerDetailFactory()

        assert server.name is not None
        assert server.version is not None
        assert server.description is not None

    def test_agent_factory(self):
        """Test AgentCardFactory."""
        agent = AgentCardFactory()

        assert agent.name is not None
        assert agent.url is not None
        assert agent.protocol_version == "1.0"

    def test_helpers_minimal_server(self):
        """Test helper function for creating minimal server."""
        server_dict = create_minimal_server_dict("test.server")

        assert server_dict["name"] == "test.server"
        assert server_dict["description"] == "Test server"
        assert server_dict["version"] == "1.0.0"

    def test_helpers_minimal_agent(self):
        """Test helper function for creating minimal agent."""
        agent_dict = create_minimal_agent_dict(name="test-agent", url="http://localhost:9000")

        assert agent_dict["name"] == "test-agent"
        assert agent_dict["url"] == "http://localhost:9000"
        assert agent_dict["protocolVersion"] == "1.0"

    def test_settings_fixture(self, test_settings):
        """Test that test_settings fixture works."""
        assert test_settings.secret_key == "test-secret-key-for-testing-only"
        assert test_settings.admin_user == "testadmin"

    def test_sample_fixtures(self, sample_server_info, sample_agent_card):
        """Test sample data fixtures."""
        assert sample_server_info["name"] == "com.example.test-server"
        assert sample_agent_card["name"] == "test-agent"
