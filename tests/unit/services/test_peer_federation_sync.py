"""
Unit tests for Peer Federation Service Sync Methods.

Tests for sync_peer, sync_all_peers, and storage methods
(_store_synced_servers and _store_synced_agents).

Updated for async/repository pattern.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.schemas.agent_models import AgentCard
from registry.schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
    SyncResult,
)
from registry.services.peer_federation_service import (
    PeerFederationService,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    PeerFederationService._instance = None
    yield
    PeerFederationService._instance = None


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    mock_repo = AsyncMock()
    mock_repo.get_peer = AsyncMock(return_value=None)
    mock_repo.list_peers = AsyncMock(return_value=[])
    mock_repo.create_peer = AsyncMock()
    mock_repo.update_peer = AsyncMock()
    mock_repo.delete_peer = AsyncMock(return_value=True)
    mock_repo.get_sync_status = AsyncMock(return_value=None)
    mock_repo.update_sync_status = AsyncMock()
    mock_repo.list_sync_statuses = AsyncMock(return_value=[])
    mock_repo.load_all = AsyncMock()
    return mock_repo


@pytest.fixture
def mock_server_service():
    """Mock server_service for storage tests."""
    mock = AsyncMock()
    mock.get_server_info = AsyncMock(return_value=None)
    mock.get_all_servers = AsyncMock(return_value={})
    mock.register_server = AsyncMock(return_value={"success": True})
    mock.update_server = AsyncMock(return_value=True)
    mock.remove_server = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_agent_service():
    """Mock agent_service for storage tests."""
    mock = AsyncMock()
    mock.get_agent_info = AsyncMock(return_value=None)
    mock.get_all_agents = AsyncMock(return_value=[])
    mock.register_agent = AsyncMock(return_value=MagicMock(spec=AgentCard))
    mock.update_agent = AsyncMock(return_value=MagicMock(spec=AgentCard))
    mock.remove_agent = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def sample_peer_config():
    """Sample peer config for testing."""
    return PeerRegistryConfig(
        peer_id="test-peer",
        name="Test Peer Registry",
        endpoint="https://peer.example.com",
        enabled=True,
        sync_mode="all",
    )


@pytest.fixture
def sample_peer_config_disabled():
    """Sample disabled peer config for testing."""
    return PeerRegistryConfig(
        peer_id="disabled-peer",
        name="Disabled Peer Registry",
        endpoint="https://disabled.example.com",
        enabled=False,
    )


@pytest.fixture
def sample_peer_config_whitelist():
    """Sample peer config with whitelist mode."""
    return PeerRegistryConfig(
        peer_id="whitelist-peer",
        name="Whitelist Peer Registry",
        endpoint="https://whitelist.example.com",
        enabled=True,
        sync_mode="whitelist",
        whitelist_servers=["/server1", "/server2"],
        whitelist_agents=["/agent1"],
    )


@pytest.fixture
def sample_peer_config_tag_filter():
    """Sample peer config with tag_filter mode."""
    return PeerRegistryConfig(
        peer_id="tag-filter-peer",
        name="Tag Filter Peer Registry",
        endpoint="https://tag-filter.example.com",
        enabled=True,
        sync_mode="tag_filter",
        tag_filters=["production", "public"],
    )


@pytest.fixture
def sample_server_data():
    """Sample server data returned from peer."""
    return {
        "path": "/test-server",
        "name": "Test Server",
        "description": "A test server",
        "url": "http://test.example.com:8000",
    }


@pytest.fixture
def sample_agent_data():
    """Sample agent data returned from peer."""
    return {
        "path": "/test-agent",
        "name": "Test Agent",
        "version": "1.0.0",
        "description": "A test agent",
        "url": "https://test.example.com/agent",
    }


def create_service_with_mocks(mock_repository, mock_server_service, mock_agent_service):
    """Create a PeerFederationService with mocked dependencies."""
    with patch(
        "registry.services.peer_federation_service.get_peer_federation_repository",
        return_value=mock_repository,
    ):
        with patch(
            "registry.services.peer_federation_service.server_service",
            mock_server_service,
        ):
            with patch(
                "registry.services.peer_federation_service.agent_service",
                mock_agent_service,
            ):
                service = PeerFederationService()
                return service


@pytest.mark.unit
class TestSyncPeer:
    """Tests for sync_peer method."""

    @pytest.mark.asyncio
    async def test_sync_peer_successful_with_servers_and_agents(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
    ):
        """Test successful sync with servers and agents."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peer in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )

                    # Mock PeerRegistryClient
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.return_value = [
                            {"path": "/server1", "name": "Server 1"},
                            {"path": "/server2", "name": "Server 2"},
                        ]
                        mock_client.fetch_agents.return_value = [
                            {
                                "path": "/agent1",
                                "name": "Agent 1",
                                "version": "1.0.0",
                                "description": "Agent 1 description",
                                "url": "https://example.com/agent1",
                            },
                        ]
                        mock_client_class.return_value = mock_client

                        result = await service.sync_peer(sample_peer_config.peer_id)

                        # Verify result
                        assert result.success is True
                        assert result.peer_id == sample_peer_config.peer_id
                        assert result.servers_synced == 2
                        assert result.agents_synced == 1
                        assert result.error_message is None
                        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_sync_peer_disabled_peer_raises_error(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config_disabled,
    ):
        """Test sync disabled peer raises ValueError."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up disabled peer in cache
                    service.registered_peers[sample_peer_config_disabled.peer_id] = (
                        sample_peer_config_disabled
                    )
                    service.peer_sync_status[sample_peer_config_disabled.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config_disabled.peer_id
                    )

                    with pytest.raises(ValueError, match="is disabled"):
                        await service.sync_peer(sample_peer_config_disabled.peer_id)

    @pytest.mark.asyncio
    async def test_sync_peer_nonexistent_peer_raises_error(
        self, mock_repository, mock_server_service, mock_agent_service
    ):
        """Test sync non-existent peer raises ValueError."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    with pytest.raises(ValueError, match="Peer not found"):
                        await service.sync_peer("nonexistent-peer")

    @pytest.mark.asyncio
    async def test_sync_peer_network_error_handling(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
    ):
        """Test network error handling during sync."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peer in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )

                    # Mock PeerRegistryClient to raise exception
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.side_effect = Exception("Network error")
                        mock_client_class.return_value = mock_client

                        result = await service.sync_peer(sample_peer_config.peer_id)

                        # Verify result
                        assert result.success is False
                        assert result.peer_id == sample_peer_config.peer_id
                        assert result.servers_synced == 0
                        assert result.agents_synced == 0
                        assert "Network error" in result.error_message

    @pytest.mark.asyncio
    async def test_sync_peer_handles_none_responses_from_client(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
    ):
        """
        Test sync fails when client returns None (indicates fetch error).

        Updated for issue #561 fix: None indicates an error (auth failure,
        network error, etc.), not an empty result. The sync should fail
        with a clear error message.
        """
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peer in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )

                    # Mock PeerRegistryClient - return None to simulate fetch failure
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.return_value = None
                        mock_client.fetch_agents.return_value = None
                        mock_client.fetch_security_scans.return_value = None
                        mock_client_class.return_value = mock_client

                        result = await service.sync_peer(sample_peer_config.peer_id)

                        # Should fail with error message
                        assert result.success is False
                        assert result.servers_synced == 0
                        assert result.agents_synced == 0
                        assert result.error_message is not None
                        assert "Failed to fetch" in result.error_message
                        assert (
                            "authentication" in result.error_message.lower()
                            or "network" in result.error_message.lower()
                        )

    @pytest.mark.asyncio
    async def test_sync_peer_succeeds_with_empty_list_responses(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
    ):
        """
        Test sync succeeds when client returns empty lists (legitimate empty result).

        Updated for issue #561 fix: Empty list [] indicates a legitimate
        empty result (peer has no servers/agents), not an error. This is
        different from None which indicates a fetch failure.
        """
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peer in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )

                    # Mock PeerRegistryClient - return empty lists (legitimate empty result)
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.return_value = []
                        mock_client.fetch_agents.return_value = []
                        mock_client.fetch_security_scans.return_value = []
                        mock_client_class.return_value = mock_client

                        result = await service.sync_peer(sample_peer_config.peer_id)

                        # Should succeed with 0 items
                        assert result.success is True
                        assert result.servers_synced == 0
                        assert result.agents_synced == 0
                        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_sync_peer_fails_with_partial_none_responses(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
    ):
        """
        Test sync fails when any fetch returns None (partial failure).

        If servers fetch succeeds but agents fetch fails (None), the entire
        sync should be marked as failed with a clear error message indicating
        which fetch(es) failed.
        """
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peer in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )

                    # Mock PeerRegistryClient - servers succeed, agents fail
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.return_value = [
                            {"path": "/server1", "name": "Server 1"}
                        ]
                        mock_client.fetch_agents.return_value = None  # Failure
                        mock_client.fetch_security_scans.return_value = []
                        mock_client_class.return_value = mock_client

                        result = await service.sync_peer(sample_peer_config.peer_id)

                        # Should fail even though servers fetch succeeded
                        assert result.success is False
                        assert result.error_message is not None
                        assert "agents" in result.error_message


@pytest.mark.unit
class TestSyncAllPeers:
    """Tests for sync_all_peers method."""

    @pytest.mark.asyncio
    async def test_sync_all_enabled_peers(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
        sample_peer_config,
        sample_peer_config_disabled,
    ):
        """Test sync_all syncs only enabled peers by default."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peers in cache
                    service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
                    service.registered_peers[sample_peer_config_disabled.peer_id] = (
                        sample_peer_config_disabled
                    )
                    service.peer_sync_status[sample_peer_config.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config.peer_id
                    )
                    service.peer_sync_status[sample_peer_config_disabled.peer_id] = PeerSyncStatus(
                        peer_id=sample_peer_config_disabled.peer_id
                    )

                    # Mock PeerRegistryClient
                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient"
                    ) as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.fetch_servers.return_value = []
                        mock_client.fetch_agents.return_value = []
                        mock_client_class.return_value = mock_client

                        results = await service.sync_all_peers(enabled_only=True)

                        # Only enabled peer should be synced
                        assert sample_peer_config.peer_id in results
                        assert sample_peer_config_disabled.peer_id not in results
                        assert results[sample_peer_config.peer_id].success is True

    @pytest.mark.asyncio
    async def test_sync_all_peers_continue_on_individual_failure(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test sync_all continues when individual peer fails."""
        peer1 = PeerRegistryConfig(
            peer_id="peer1",
            name="Peer 1",
            endpoint="https://peer1.example.com",
            enabled=True,
        )
        peer2 = PeerRegistryConfig(
            peer_id="peer2",
            name="Peer 2",
            endpoint="https://peer2.example.com",
            enabled=True,
        )

        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    # Set up peers in cache
                    service.registered_peers["peer1"] = peer1
                    service.registered_peers["peer2"] = peer2
                    service.peer_sync_status["peer1"] = PeerSyncStatus(peer_id="peer1")
                    service.peer_sync_status["peer2"] = PeerSyncStatus(peer_id="peer2")

                    # Mock PeerRegistryClient - first fails, second succeeds
                    call_count = [0]

                    def mock_client_factory(*args, **kwargs):
                        mock_client = MagicMock()
                        call_count[0] += 1
                        if call_count[0] == 1:
                            mock_client.fetch_servers.side_effect = Exception("Peer 1 error")
                        else:
                            mock_client.fetch_servers.return_value = [
                                {"path": "/server1", "name": "Server 1"}
                            ]
                            mock_client.fetch_agents.return_value = []
                        return mock_client

                    with patch(
                        "registry.services.peer_federation_service.PeerRegistryClient",
                        side_effect=mock_client_factory,
                    ):
                        results = await service.sync_all_peers()

                        # Both peers should have results
                        assert len(results) == 2
                        # One failed, one succeeded
                        successes = sum(1 for r in results.values() if r.success)
                        failures = sum(1 for r in results.values() if not r.success)
                        assert successes == 1
                        assert failures == 1


@pytest.mark.unit
class TestFilterServersByConfig:
    """Tests for _filter_servers_by_config method."""

    def test_sync_mode_all_returns_all_servers(self, mock_repository, sample_peer_config):
        """Test sync_mode=all returns all servers."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            servers = [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
                {"path": "/server3", "name": "Server 3"},
            ]

            result = service._filter_servers_by_config(servers, sample_peer_config)

            assert len(result) == 3
            assert result == servers

    def test_sync_mode_whitelist_filters_by_whitelist_servers(
        self, mock_repository, sample_peer_config_whitelist
    ):
        """Test sync_mode=whitelist filters servers."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            servers = [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
                {"path": "/server3", "name": "Server 3"},
            ]

            result = service._filter_servers_by_config(servers, sample_peer_config_whitelist)

            assert len(result) == 2
            paths = [s["path"] for s in result]
            assert "/server1" in paths
            assert "/server2" in paths
            assert "/server3" not in paths

    def test_sync_mode_whitelist_with_empty_whitelist_returns_empty(self, mock_repository):
        """Test sync_mode=whitelist with empty whitelist returns empty."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            peer_config = PeerRegistryConfig(
                peer_id="test-peer",
                name="Test Peer",
                endpoint="https://test.example.com",
                sync_mode="whitelist",
                whitelist_servers=[],
            )

            servers = [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
            ]

            result = service._filter_servers_by_config(servers, peer_config)

            assert len(result) == 0

    def test_sync_mode_tag_filter_filters_by_tags(
        self, mock_repository, sample_peer_config_tag_filter
    ):
        """Test sync_mode=tag_filter filters by tags."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            servers = [
                {"path": "/server1", "name": "Server 1", "tags": ["production"]},
                {"path": "/server2", "name": "Server 2", "tags": ["staging"]},
                {"path": "/server3", "name": "Server 3", "tags": ["production", "api"]},
            ]

            result = service._filter_servers_by_config(servers, sample_peer_config_tag_filter)

            # Should only include servers with "production" or "public" tags
            assert len(result) == 2
            paths = [s["path"] for s in result]
            assert "/server1" in paths
            assert "/server3" in paths
            assert "/server2" not in paths

    def test_sync_mode_tag_filter_matches_categories(self, mock_repository):
        """Test tag filter also checks categories field."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            peer_config = PeerRegistryConfig(
                peer_id="test-peer",
                name="Test Peer",
                endpoint="https://test.example.com",
                sync_mode="tag_filter",
                tag_filters=["production"],
            )

            servers = [
                {"path": "/server1", "name": "Server 1", "categories": ["production"]},
                {"path": "/server2", "name": "Server 2", "tags": ["staging"]},
            ]

            result = service._filter_servers_by_config(servers, peer_config)

            assert len(result) == 1
            assert result[0]["path"] == "/server1"


@pytest.mark.unit
class TestFilterAgentsByConfig:
    """Tests for _filter_agents_by_config method."""

    def test_sync_mode_all_returns_all_agents(self, mock_repository, sample_peer_config):
        """Test sync_mode=all returns all agents."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            agents = [
                {"path": "/agent1", "name": "Agent 1"},
                {"path": "/agent2", "name": "Agent 2"},
            ]

            result = service._filter_agents_by_config(agents, sample_peer_config)

            assert len(result) == 2
            assert result == agents

    def test_sync_mode_whitelist_filters_by_whitelist_agents(
        self, mock_repository, sample_peer_config_whitelist
    ):
        """Test sync_mode=whitelist filters agents."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            agents = [
                {"path": "/agent1", "name": "Agent 1"},
                {"path": "/agent2", "name": "Agent 2"},
            ]

            result = service._filter_agents_by_config(agents, sample_peer_config_whitelist)

            assert len(result) == 1
            assert result[0]["path"] == "/agent1"


@pytest.mark.unit
class TestMatchesTagFilter:
    """Tests for _matches_tag_filter method."""

    def test_matches_when_tag_in_tags_field(self, mock_repository):
        """Test tag filter matches tags field."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"tags": ["production", "api"]}

            result = service._matches_tag_filter(item, ["production"])

            assert result is True

    def test_matches_when_tag_in_categories_field(self, mock_repository):
        """Test tag filter matches categories field."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"categories": ["production"]}

            result = service._matches_tag_filter(item, ["production"])

            assert result is True

    def test_matches_with_multiple_filters(self, mock_repository):
        """Test tag filter with multiple filter tags."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"tags": ["staging"]}

            result = service._matches_tag_filter(item, ["production", "staging"])

            assert result is True

    def test_returns_false_when_no_match(self, mock_repository):
        """Test returns False when no match."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"tags": ["staging"]}

            result = service._matches_tag_filter(item, ["production"])

            assert result is False

    def test_returns_false_for_empty_tag_filters(self, mock_repository):
        """Test returns False for empty filter list."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"tags": ["production"]}

            result = service._matches_tag_filter(item, [])

            assert result is False

    def test_handles_missing_tags_field(self, mock_repository):
        """Test handles missing tags field gracefully."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {}

            result = service._matches_tag_filter(item, ["production"])

            assert result is False


@pytest.mark.unit
class TestStoreSyncedServers:
    """Tests for _store_synced_servers method."""

    @pytest.mark.asyncio
    async def test_store_new_server_with_sync_metadata(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test storing new server adds sync metadata."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    servers = [{"path": "/server1", "name": "Server 1"}]

                    stored_count = await service._store_synced_servers("test-peer", servers)

                    assert stored_count == 1
                    # Verify register_server was called
                    mock_server_service.register_server.assert_called_once()

                    # Check the server data has sync_metadata
                    call_args = mock_server_service.register_server.call_args
                    server_data = call_args[0][0]
                    assert "sync_metadata" in server_data
                    assert server_data["sync_metadata"]["is_federated"] is True
                    assert server_data["sync_metadata"]["source_peer_id"] == "test-peer"

    @pytest.mark.asyncio
    async def test_store_update_existing_server(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test updating existing server."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # Server already exists
                    mock_server_service.get_server_info.return_value = {
                        "path": "/peer-test-peer/server1",
                        "name": "Old Server 1",
                        "sync_metadata": {},
                    }

                    service = PeerFederationService()

                    servers = [{"path": "/server1", "name": "Server 1 Updated"}]

                    stored_count = await service._store_synced_servers("test-peer", servers)

                    assert stored_count == 1
                    # Verify update_server was called (not register_server)
                    mock_server_service.update_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_path_prefixing_with_peer_id(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test server path is prefixed with peer ID."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    servers = [{"path": "/server1", "name": "Server 1"}]

                    await service._store_synced_servers("my-peer", servers)

                    # Verify the path is prefixed with peer_id
                    # Implementation uses /{peer_id}{path}, e.g., /my-peer/server1
                    call_args = mock_server_service.register_server.call_args
                    server_data = call_args[0][0]
                    assert server_data["path"] == "/my-peer/server1"

    @pytest.mark.asyncio
    async def test_store_skip_servers_missing_path_field(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test servers without path field are skipped."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    servers = [
                        {"name": "Server without path"},
                        {"path": "/server1", "name": "Server 1"},
                    ]

                    stored_count = await service._store_synced_servers("test-peer", servers)

                    # Only one server should be stored
                    assert stored_count == 1


@pytest.mark.unit
class TestStoreSyncedAgents:
    """Tests for _store_synced_agents method."""

    @pytest.mark.asyncio
    async def test_store_new_agent_with_sync_metadata(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test storing new agent adds sync metadata."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    agents = [
                        {
                            "path": "/agent1",
                            "name": "Agent 1",
                            "version": "1.0.0",
                            "description": "Test agent",
                            "url": "https://example.com/agent",
                        }
                    ]

                    stored_count = await service._store_synced_agents("test-peer", agents)

                    assert stored_count == 1
                    # Verify register_agent was called
                    mock_agent_service.register_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_skip_agents_missing_path_field(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test agents without path field are skipped."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    service = PeerFederationService()

                    agents = [
                        {"name": "Agent without path"},
                        {
                            "path": "/agent1",
                            "name": "Agent 1",
                            "version": "1.0.0",
                            "description": "Test",
                            "url": "https://example.com",
                        },
                    ]

                    stored_count = await service._store_synced_agents("test-peer", agents)

                    # Only one agent should be stored
                    assert stored_count == 1


@pytest.mark.unit
class TestDetectOrphanedItems:
    """Tests for detect_orphaned_items method."""

    @pytest.mark.asyncio
    async def test_detects_servers_missing_from_peer(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test detects orphaned servers."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # Mock get_all_servers to return existing synced servers
                    # Path format: /{peer_id}{original_path}
                    mock_server_service.get_all_servers.return_value = {
                        "/test-peer/server1": {
                            "path": "/test-peer/server1",
                            "sync_metadata": {
                                "source_peer_id": "test-peer",
                                "is_federated": True,
                                "original_path": "/server1",
                            },
                        },
                        "/test-peer/server2": {
                            "path": "/test-peer/server2",
                            "sync_metadata": {
                                "source_peer_id": "test-peer",
                                "is_federated": True,
                                "original_path": "/server2",
                            },
                        },
                    }

                    service = PeerFederationService()

                    # Only server1 is currently in peer
                    current_server_paths = ["/server1"]
                    current_agent_paths = []

                    orphaned_servers, orphaned_agents = await service.detect_orphaned_items(
                        "test-peer", current_server_paths, current_agent_paths
                    )

                    # server2 should be detected as orphaned
                    assert len(orphaned_servers) == 1
                    assert "/test-peer/server2" in orphaned_servers

    @pytest.mark.asyncio
    async def test_returns_empty_lists_when_no_orphans(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test returns empty lists when no orphans."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # No servers exist locally
                    mock_server_service.get_all_servers.return_value = {}
                    mock_agent_service.get_all_agents.return_value = []

                    service = PeerFederationService()

                    orphaned_servers, orphaned_agents = await service.detect_orphaned_items(
                        "test-peer", ["/server1"], ["/agent1"]
                    )

                    assert len(orphaned_servers) == 0
                    assert len(orphaned_agents) == 0


@pytest.mark.unit
class TestSetLocalOverride:
    """Tests for set_local_override method."""

    @pytest.mark.asyncio
    async def test_sets_override_to_true_for_server(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test setting local override to True."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # Server exists
                    mock_server_service.get_server_info.return_value = {
                        "path": "/peer-test/server1",
                        "sync_metadata": {},
                    }

                    service = PeerFederationService()

                    result = await service.set_local_override("/peer-test/server1", "server", True)

                    assert result is True
                    mock_server_service.update_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_non_existent_server(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test handling non-existent server."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # Server doesn't exist
                    mock_server_service.get_server_info.return_value = None

                    service = PeerFederationService()

                    result = await service.set_local_override("/nonexistent", "server", True)

                    assert result is False


@pytest.mark.unit
class TestIsLocallyOverridden:
    """Tests for is_locally_overridden method."""

    def test_returns_true_when_override_is_set(self, mock_repository):
        """Test returns True when local_overrides is True."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"sync_metadata": {"local_overrides": True}}

            result = service.is_locally_overridden(item)

            assert result is True

    def test_returns_false_when_override_not_set(self, mock_repository):
        """Test returns False when local_overrides is False."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {"sync_metadata": {"local_overrides": False}}

            result = service.is_locally_overridden(item)

            assert result is False

    def test_handles_missing_sync_metadata(self, mock_repository):
        """Test handles missing sync_metadata."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()

            item = {}

            result = service.is_locally_overridden(item)

            assert result is False


@pytest.mark.unit
class TestLocalOverrideIntegration:
    """Integration tests for local override behavior during sync."""

    @pytest.mark.asyncio
    async def test_local_override_prevents_server_sync_update(
        self,
        mock_repository,
        mock_server_service,
        mock_agent_service,
    ):
        """Test locally overridden servers are not updated during sync."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service",
                mock_server_service,
            ):
                with patch(
                    "registry.services.peer_federation_service.agent_service",
                    mock_agent_service,
                ):
                    # Existing server with local override
                    mock_server_service.get_server_info.return_value = {
                        "path": "/peer-test-peer/server1",
                        "name": "Local Modified Server",
                        "sync_metadata": {
                            "source_peer_id": "test-peer",
                            "is_federated": True,
                            "local_overrides": True,
                        },
                    }

                    service = PeerFederationService()

                    servers = [{"path": "/server1", "name": "Remote Server Name"}]

                    stored_count = await service._store_synced_servers("test-peer", servers)

                    # Server should be skipped (not updated)
                    assert stored_count == 0
                    mock_server_service.update_server.assert_not_called()
                    mock_server_service.register_server.assert_not_called()
