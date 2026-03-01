"""
End-to-end integration tests for peer federation.

Tests full federation flow including:
- Peer CRUD operations via repository
- Sync operations with mock peer registry
- Orphan detection and handling
- Local override preservation

Requires MongoDB running on localhost:27017.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
)


def _mongodb_available() -> bool:
    """Check if MongoDB is available for testing."""
    try:
        import pymongo

        client = pymongo.MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=1000,
            directConnection=True,
        )
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


def _documentdb_available() -> bool:
    """Check if DocumentDB (with TLS cert) is available."""
    # Check if the TLS certificate exists
    return os.path.exists("global-bundle.pem")


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def peer_config():
    """Create a sample peer configuration for testing."""
    return PeerRegistryConfig(
        peer_id="test-peer-001",
        name="Test Peer Registry",
        endpoint="http://localhost:9999",
        enabled=True,
        sync_mode="all",
        sync_interval_minutes=30,
    )


@pytest.fixture
def peer_config_whitelist():
    """Create a peer configuration with whitelist mode."""
    return PeerRegistryConfig(
        peer_id="test-peer-whitelist",
        name="Whitelist Peer",
        endpoint="http://localhost:9998",
        enabled=True,
        sync_mode="whitelist",
        whitelist_servers=["/allowed-server-1", "/allowed-server-2"],
        whitelist_agents=["/allowed-agent-1"],
    )


@pytest.fixture
def peer_config_tag_filter():
    """Create a peer configuration with tag filter mode."""
    return PeerRegistryConfig(
        peer_id="test-peer-tags",
        name="Tag Filter Peer",
        endpoint="http://localhost:9997",
        enabled=True,
        sync_mode="tag_filter",
        tag_filters=["production", "verified"],
    )


@pytest.fixture
def mock_servers():
    """Sample server data from a peer registry."""
    return [
        {
            "path": "/server-1",
            "server_name": "Test Server 1",
            "tags": ["production"],
            "endpoint": "http://server1.example.com",
        },
        {
            "path": "/server-2",
            "server_name": "Test Server 2",
            "tags": ["development"],
            "endpoint": "http://server2.example.com",
        },
        {
            "path": "/allowed-server-1",
            "server_name": "Allowed Server 1",
            "tags": ["verified"],
            "endpoint": "http://allowed1.example.com",
        },
    ]


@pytest.fixture
def mock_agents():
    """Sample agent data from a peer registry."""
    return [
        {
            "path": "/agent-1",
            "name": "Test Agent 1",
            "description": "A test agent for production use",
            "url": "https://agent1.example.com",
            "version": "1.0.0",
            "tags": ["production", "verified"],
            "skills": [
                {
                    "id": "skill-1",
                    "name": "Skill 1",
                    "description": "A production skill",
                    "tags": ["production"],
                }
            ],
        },
        {
            "path": "/agent-2",
            "name": "Test Agent 2",
            "description": "An experimental test agent",
            "url": "https://agent2.example.com",
            "version": "0.1.0",
            "tags": ["experimental"],
            "skills": [
                {
                    "id": "skill-2",
                    "name": "Skill 2",
                    "description": "An experimental skill",
                    "tags": ["experimental"],
                }
            ],
        },
        {
            "path": "/allowed-agent-1",
            "name": "Allowed Agent 1",
            "description": "An allowed agent for whitelist testing",
            "url": "https://allowed1.example.com",
            "version": "1.0.0",
            "tags": [],
            "skills": [],
        },
    ]


# =============================================================================
# REPOSITORY INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not _documentdb_available(),
    reason="DocumentDB/TLS certificate not available",
)
class TestPeerFederationRepositoryIntegration:
    """Integration tests for peer federation repository with MongoDB."""

    async def test_documentdb_repository_crud(self, peer_config):
        """Test full CRUD cycle with DocumentDB repository."""
        # Skip if MongoDB not available
        if os.environ.get("STORAGE_BACKEND", "mongodb-ce") == "file":
            pytest.skip("Requires MongoDB storage backend")

        from registry.repositories.documentdb.peer_federation_repository import (
            DocumentDBPeerFederationRepository,
        )

        repo = DocumentDBPeerFederationRepository()

        try:
            # Create
            created = await repo.create_peer(peer_config)
            assert created.peer_id == peer_config.peer_id
            assert created.name == peer_config.name
            assert created.created_at is not None

            # Read
            retrieved = await repo.get_peer(peer_config.peer_id)
            assert retrieved is not None
            assert retrieved.peer_id == peer_config.peer_id

            # List
            peers = await repo.list_peers()
            assert any(p.peer_id == peer_config.peer_id for p in peers)

            # List enabled only
            enabled_peers = await repo.list_peers(enabled=True)
            assert any(p.peer_id == peer_config.peer_id for p in enabled_peers)

            # Update
            updated = await repo.update_peer(peer_config.peer_id, {"name": "Updated Name"})
            assert updated.name == "Updated Name"

            # Sync status
            sync_status = PeerSyncStatus(
                peer_id=peer_config.peer_id,
                is_healthy=True,
                current_generation=5,
            )
            await repo.update_sync_status(peer_config.peer_id, sync_status)

            retrieved_status = await repo.get_sync_status(peer_config.peer_id)
            assert retrieved_status is not None
            assert retrieved_status.current_generation == 5

        finally:
            # Cleanup
            try:
                await repo.delete_peer(peer_config.peer_id)
            except Exception:
                pass

    async def test_documentdb_repository_duplicate_peer_id_rejected(self, peer_config):
        """Test that duplicate peer IDs are rejected."""
        if os.environ.get("STORAGE_BACKEND", "mongodb-ce") == "file":
            pytest.skip("Requires MongoDB storage backend")

        from registry.repositories.documentdb.peer_federation_repository import (
            DocumentDBPeerFederationRepository,
        )

        repo = DocumentDBPeerFederationRepository()

        try:
            # Create first peer
            await repo.create_peer(peer_config)

            # Try to create duplicate
            duplicate = PeerRegistryConfig(
                peer_id=peer_config.peer_id,  # Same ID
                name="Duplicate Peer",
                endpoint="http://duplicate.example.com",
            )

            with pytest.raises(ValueError, match="already exists"):
                await repo.create_peer(duplicate)

        finally:
            try:
                await repo.delete_peer(peer_config.peer_id)
            except Exception:
                pass

    async def test_documentdb_repository_delete_cascade(self, peer_config):
        """Test that deleting a peer also deletes its sync status."""
        if os.environ.get("STORAGE_BACKEND", "mongodb-ce") == "file":
            pytest.skip("Requires MongoDB storage backend")

        from registry.repositories.documentdb.peer_federation_repository import (
            DocumentDBPeerFederationRepository,
        )

        repo = DocumentDBPeerFederationRepository()

        try:
            # Create peer
            await repo.create_peer(peer_config)

            # Update sync status
            sync_status = PeerSyncStatus(peer_id=peer_config.peer_id)
            await repo.update_sync_status(peer_config.peer_id, sync_status)

            # Verify sync status exists
            status = await repo.get_sync_status(peer_config.peer_id)
            assert status is not None

            # Delete peer
            await repo.delete_peer(peer_config.peer_id)

            # Verify sync status also deleted
            status_after = await repo.get_sync_status(peer_config.peer_id)
            assert status_after is None

        except Exception:
            # Cleanup if test fails
            try:
                await repo.delete_peer(peer_config.peer_id)
            except Exception:
                pass
            raise


# =============================================================================
# SERVICE INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestPeerFederationServiceIntegration:
    """Integration tests for peer federation service."""

    async def test_service_sync_with_mock_peer(
        self,
        peer_config,
        mock_servers,
        mock_agents,
    ):
        """Test sync operation with mocked peer registry client."""
        from registry.services.peer_federation_service import (
            PeerFederationService,
            get_peer_federation_service,
        )

        # Reset singleton for clean test
        PeerFederationService._instance = None

        service = get_peer_federation_service()

        # Mock the repository using AsyncMock
        mock_repo = MagicMock()
        mock_repo.create_peer = AsyncMock(return_value=peer_config)
        mock_repo.get_peer = AsyncMock(
            side_effect=lambda peer_id: peer_config if peer_id == peer_config.peer_id else None
        )
        mock_repo.update_sync_status = AsyncMock(side_effect=lambda *args: args[1])
        mock_repo.get_sync_status = AsyncMock(
            side_effect=lambda peer_id: PeerSyncStatus(peer_id=peer_id)
        )
        mock_repo.list_peers = AsyncMock(return_value=[peer_config])
        mock_repo.list_sync_statuses = AsyncMock(
            return_value=[PeerSyncStatus(peer_id=peer_config.peer_id)]
        )
        mock_repo.load_all = AsyncMock(return_value=None)

        service._repo = mock_repo

        # Add peer to cache manually (since we're mocking)
        service.registered_peers[peer_config.peer_id] = peer_config
        service.peer_sync_status[peer_config.peer_id] = PeerSyncStatus(peer_id=peer_config.peer_id)

        # Mock the peer registry client
        mock_client = MagicMock()
        mock_client.fetch_servers = MagicMock(return_value=mock_servers)
        mock_client.fetch_agents = MagicMock(return_value=mock_agents)

        # Mock server and agent services
        with patch(
            "registry.services.peer_federation_service.PeerRegistryClient",
            return_value=mock_client,
        ):
            with patch(
                "registry.services.peer_federation_service.server_service"
            ) as mock_server_svc:
                with patch(
                    "registry.services.peer_federation_service.agent_service"
                ) as mock_agent_svc:
                    mock_server_svc.registered_servers = {}
                    mock_server_svc.register_server = AsyncMock(return_value={"success": True})
                    mock_server_svc.update_server = AsyncMock(return_value=True)
                    mock_server_svc.get_server_info = AsyncMock(return_value=None)
                    mock_server_svc.get_all_servers = AsyncMock(return_value={})

                    mock_agent_svc.registered_agents = {}
                    mock_agent_svc.register_agent = AsyncMock(side_effect=lambda agent: agent)
                    mock_agent_svc.update_agent = AsyncMock(return_value=MagicMock())
                    mock_agent_svc.get_agent_info = AsyncMock(return_value=None)
                    mock_agent_svc.get_all_agents = AsyncMock(return_value=[])

                    # Execute sync
                    result = await service.sync_peer(peer_config.peer_id)

                    # Verify result
                    assert result.success is True
                    assert result.peer_id == peer_config.peer_id
                    assert result.servers_synced == len(mock_servers)
                    assert result.agents_synced == len(mock_agents)

    async def test_service_filter_by_whitelist(
        self,
        peer_config_whitelist,
        mock_servers,
        mock_agents,
    ):
        """Test that whitelist filtering works correctly."""
        from registry.services.peer_federation_service import PeerFederationService

        # Create fresh service instance
        PeerFederationService._instance = None
        service = PeerFederationService.__new__(PeerFederationService)
        service._initialized = False
        service.__init__()

        # Test server filtering
        filtered_servers = service._filter_servers_by_config(mock_servers, peer_config_whitelist)

        # Should only include whitelisted servers
        assert len(filtered_servers) == 1
        assert filtered_servers[0]["path"] == "/allowed-server-1"

        # Test agent filtering
        filtered_agents = service._filter_agents_by_config(mock_agents, peer_config_whitelist)

        # Should only include whitelisted agents
        assert len(filtered_agents) == 1
        assert filtered_agents[0]["path"] == "/allowed-agent-1"

    async def test_service_filter_by_tags(
        self,
        peer_config_tag_filter,
        mock_servers,
        mock_agents,
    ):
        """Test that tag filtering works correctly."""
        from registry.services.peer_federation_service import PeerFederationService

        # Create fresh service instance
        PeerFederationService._instance = None
        service = PeerFederationService.__new__(PeerFederationService)
        service._initialized = False
        service.__init__()

        # Test server filtering (should match "production" or "verified")
        filtered_servers = service._filter_servers_by_config(mock_servers, peer_config_tag_filter)

        # Should include server-1 (production) and allowed-server-1 (verified)
        assert len(filtered_servers) == 2
        paths = [s["path"] for s in filtered_servers]
        assert "/server-1" in paths
        assert "/allowed-server-1" in paths

        # Test agent filtering (should match "production" or "verified")
        filtered_agents = service._filter_agents_by_config(mock_agents, peer_config_tag_filter)

        # Should only include agent-1 (has both production and verified)
        assert len(filtered_agents) == 1
        assert filtered_agents[0]["path"] == "/agent-1"


# =============================================================================
# ORPHAN DETECTION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestOrphanDetection:
    """Tests for orphan detection functionality."""

    async def test_detect_orphaned_servers(self, peer_config):
        """Test detection of orphaned servers after sync."""
        from registry.services.peer_federation_service import PeerFederationService

        # Create fresh service instance
        PeerFederationService._instance = None
        service = PeerFederationService.__new__(PeerFederationService)
        service._initialized = False
        service.__init__()

        # Simulate existing synced servers (some still exist, some orphaned)
        existing_servers = {
            f"/{peer_config.peer_id}/server-1": {
                "path": f"/{peer_config.peer_id}/server-1",
                "sync_metadata": {
                    "source_peer_id": peer_config.peer_id,
                    "original_path": "/server-1",
                },
            },
            f"/{peer_config.peer_id}/server-orphan": {
                "path": f"/{peer_config.peer_id}/server-orphan",
                "sync_metadata": {
                    "source_peer_id": peer_config.peer_id,
                    "original_path": "/server-orphan",
                },
            },
        }

        # Mock server service with existing synced servers using AsyncMock
        with patch("registry.services.peer_federation_service.server_service") as mock_server_svc:
            with patch("registry.services.peer_federation_service.agent_service") as mock_agent_svc:
                # Use AsyncMock for async methods
                mock_server_svc.get_all_servers = AsyncMock(return_value=existing_servers)
                mock_agent_svc.get_all_agents = AsyncMock(return_value=[])

                # Current servers in peer (server-1 exists, server-orphan doesn't)
                current_server_paths = ["/server-1"]
                current_agent_paths = []

                orphaned_servers, orphaned_agents = await service.detect_orphaned_items(
                    peer_config.peer_id, current_server_paths, current_agent_paths
                )

                # server-orphan should be detected as orphaned
                assert len(orphaned_servers) == 1
                assert f"/{peer_config.peer_id}/server-orphan" in orphaned_servers
                assert len(orphaned_agents) == 0

    async def test_local_override_preserved(self, peer_config):
        """Test that locally overridden items are not updated during sync."""
        from registry.services.peer_federation_service import PeerFederationService

        # Create fresh service instance
        PeerFederationService._instance = None
        service = PeerFederationService.__new__(PeerFederationService)
        service._initialized = False
        service.__init__()

        # Item with local override
        overridden_item = {
            "path": "/server-overridden",
            "sync_metadata": {
                "source_peer_id": peer_config.peer_id,
                "local_overrides": True,
            },
        }

        # Test is_locally_overridden
        assert service.is_locally_overridden(overridden_item) is True

        # Item without local override
        normal_item = {
            "path": "/server-normal",
            "sync_metadata": {
                "source_peer_id": peer_config.peer_id,
                "local_overrides": False,
            },
        }

        assert service.is_locally_overridden(normal_item) is False

        # Item without sync_metadata
        new_item = {"path": "/server-new"}
        assert service.is_locally_overridden(new_item) is False


# =============================================================================
# FILE REPOSITORY INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestFilePeerFederationRepository:
    """Integration tests for file-based peer federation repository."""

    async def test_file_repository_crud(self, peer_config, tmp_path):
        """Test CRUD operations with file repository."""
        import warnings

        from registry.repositories.file.peer_federation_repository import (
            FilePeerFederationRepository,
        )

        # Suppress deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            peers_dir = tmp_path / "peers"
            sync_state_file = tmp_path / "sync_state.json"

            repo = FilePeerFederationRepository(
                peers_dir=peers_dir,
                sync_state_file=sync_state_file,
            )

            # Load (should be empty)
            await repo.load_all()
            assert len(await repo.list_peers()) == 0

            # Create
            created = await repo.create_peer(peer_config)
            assert created.peer_id == peer_config.peer_id

            # Verify file was created
            peer_file = peers_dir / f"{peer_config.peer_id}.json"
            assert peer_file.exists()

            # Read
            retrieved = await repo.get_peer(peer_config.peer_id)
            assert retrieved is not None
            assert retrieved.name == peer_config.name

            # Update
            updated = await repo.update_peer(peer_config.peer_id, {"name": "Updated Name"})
            assert updated.name == "Updated Name"

            # Sync status
            sync_status = PeerSyncStatus(peer_id=peer_config.peer_id)
            await repo.update_sync_status(peer_config.peer_id, sync_status)

            # Verify sync state file was created
            assert sync_state_file.exists()

            # Delete
            await repo.delete_peer(peer_config.peer_id)

            # Verify file was removed
            assert not peer_file.exists()

            # Verify peer is gone
            retrieved_after = await repo.get_peer(peer_config.peer_id)
            assert retrieved_after is None
