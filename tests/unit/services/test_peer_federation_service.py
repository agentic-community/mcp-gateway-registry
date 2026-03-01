"""
Unit tests for Peer Federation Service.

Tests for peer registry federation configuration management,
including CRUD operations, security, and state management.

Note: Helper functions (_validate_peer_id, _get_safe_file_path, etc.) have been
moved to registry.repositories.file.peer_federation_repository and are tested
in tests/unit/repositories/test_file_peer_federation_repository.py.
"""

from threading import Thread
from unittest.mock import AsyncMock, patch

import pytest

from registry.schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
)
from registry.services.peer_federation_service import (
    PeerFederationService,
    get_peer_federation_service,
)
from registry.repositories.file.peer_federation_repository import (
    _get_safe_file_path,
    _validate_peer_id,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    PeerFederationService._instance = None
    yield
    PeerFederationService._instance = None


@pytest.fixture
def temp_peers_dir(tmp_path):
    """Create temp directory for peer configs."""
    peers_dir = tmp_path / "peers"
    peers_dir.mkdir()
    return peers_dir


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
def sample_peer_config():
    """Sample peer config for testing."""
    return PeerRegistryConfig(
        peer_id="central-registry",
        name="Central Registry",
        endpoint="https://central.example.com",
        enabled=True,
        sync_mode="all",
        sync_interval_minutes=60,
    )


@pytest.fixture
def sample_peer_config_2():
    """Second sample peer config for testing."""
    return PeerRegistryConfig(
        peer_id="backup-registry",
        name="Backup Registry",
        endpoint="https://backup.example.com",
        enabled=False,
        sync_mode="whitelist",
        whitelist_servers=["/server1", "/server2"],
        sync_interval_minutes=120,
    )


@pytest.mark.unit
class TestValidatePeerId:
    """Tests for _validate_peer_id helper function from file repository."""

    def test_valid_peer_id(self):
        """Test that valid peer IDs pass validation."""
        # Should not raise
        _validate_peer_id("valid-peer-123")
        _validate_peer_id("peer_with_underscore")
        _validate_peer_id("alphanumeric123")

    def test_empty_peer_id_rejected(self):
        """Test that empty peer ID is rejected."""
        with pytest.raises(ValueError, match="peer_id cannot be empty"):
            _validate_peer_id("")

    def test_path_traversal_dotdot_rejected(self):
        """Test that .. path traversal is rejected."""
        with pytest.raises(ValueError, match="path traversal detected"):
            _validate_peer_id("../etc/passwd")

    def test_path_traversal_forward_slash_rejected(self):
        """Test that forward slash is rejected."""
        with pytest.raises(ValueError, match="path traversal detected"):
            _validate_peer_id("path/to/file")

    def test_path_traversal_backslash_rejected(self):
        """Test that backslash is rejected."""
        with pytest.raises(ValueError, match="path traversal detected"):
            _validate_peer_id("path\\to\\file")

    def test_invalid_character_less_than_rejected(self):
        """Test that < character is rejected."""
        with pytest.raises(ValueError, match="invalid character"):
            _validate_peer_id("peer<name")

    def test_reserved_name_con_rejected(self):
        """Test that reserved name CON is rejected."""
        with pytest.raises(ValueError, match="reserved name"):
            _validate_peer_id("con")
        with pytest.raises(ValueError, match="reserved name"):
            _validate_peer_id("CON")


@pytest.mark.unit
class TestGetSafeFilePath:
    """Tests for _get_safe_file_path helper function from file repository."""

    def test_normal_path_returns_valid(self, temp_peers_dir):
        """Test that normal peer ID returns valid path."""
        result = _get_safe_file_path("valid-peer", temp_peers_dir)
        assert result == temp_peers_dir / "valid-peer.json"

    def test_path_traversal_rejected(self, temp_peers_dir):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="path traversal detected"):
            _get_safe_file_path("../etc/passwd", temp_peers_dir)

    def test_invalid_chars_rejected(self, temp_peers_dir):
        """Test that invalid characters are rejected."""
        with pytest.raises(ValueError, match="invalid character"):
            _get_safe_file_path("peer|name", temp_peers_dir)

    def test_resolved_path_within_peers_dir(self, temp_peers_dir):
        """Test that resolved path is within peers directory."""
        result = _get_safe_file_path("normal-peer", temp_peers_dir)
        resolved = result.resolve()
        assert resolved.is_relative_to(temp_peers_dir.resolve())


@pytest.mark.unit
class TestPeerFederationServiceSingleton:
    """Tests for singleton pattern implementation."""

    def test_singleton_returns_same_instance(self, mock_repository):
        """Test that singleton returns same instance."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service1 = PeerFederationService()
            service2 = PeerFederationService()
            assert service1 is service2

    def test_get_peer_federation_service_returns_singleton(self, mock_repository):
        """Test that helper function returns singleton."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service1 = get_peer_federation_service()
            service2 = get_peer_federation_service()
            assert service1 is service2

    def test_singleton_thread_safe(self, mock_repository):
        """Test that singleton is thread-safe."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            instances = []

            def create_instance():
                instances.append(PeerFederationService())

            threads = [Thread(target=create_instance) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All instances should be the same
            first_instance = instances[0]
            assert all(inst is first_instance for inst in instances)


@pytest.mark.unit
class TestPeerFederationServiceCRUD:
    """Tests for CRUD operations on PeerFederationService."""

    @pytest.mark.asyncio
    async def test_add_peer_success(self, mock_repository, sample_peer_config):
        """Test successfully adding a peer."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            # Configure mock to return the config when create is called
            mock_repository.create_peer.return_value = sample_peer_config
            mock_repository.get_peer.return_value = None  # Peer doesn't exist yet

            service = PeerFederationService()
            result = await service.add_peer(sample_peer_config)

            assert result.peer_id == sample_peer_config.peer_id
            assert result.name == sample_peer_config.name
            mock_repository.create_peer.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_peer_duplicate_peer_id_fails(self, mock_repository, sample_peer_config):
        """Test that adding duplicate peer_id raises error."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            # Repository raises error for duplicate
            mock_repository.create_peer.side_effect = ValueError(
                f"Peer ID '{sample_peer_config.peer_id}' already exists"
            )

            service = PeerFederationService()

            with pytest.raises(ValueError, match="already exists"):
                await service.add_peer(sample_peer_config)

    @pytest.mark.asyncio
    async def test_get_peer_existing(self, mock_repository, sample_peer_config):
        """Test getting an existing peer."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = sample_peer_config

            service = PeerFederationService()
            result = await service.get_peer(sample_peer_config.peer_id)

            assert result.peer_id == sample_peer_config.peer_id
            assert result.name == sample_peer_config.name

    @pytest.mark.asyncio
    async def test_get_peer_nonexistent_raises_error(self, mock_repository):
        """Test that getting non-existent peer raises error."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = None

            service = PeerFederationService()

            with pytest.raises(ValueError, match="Peer not found"):
                await service.get_peer("nonexistent-peer")

    @pytest.mark.asyncio
    async def test_update_peer_success(self, mock_repository, sample_peer_config):
        """Test successfully updating a peer."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = sample_peer_config
            updated_config = sample_peer_config.model_copy()
            updated_config.name = "Updated Name"
            updated_config.enabled = False
            mock_repository.update_peer.return_value = updated_config

            service = PeerFederationService()

            updates = {
                "name": "Updated Name",
                "enabled": False,
            }

            result = await service.update_peer(sample_peer_config.peer_id, updates)

            assert result.name == "Updated Name"
            assert result.enabled is False

    @pytest.mark.asyncio
    async def test_update_peer_nonexistent_raises_error(self, mock_repository):
        """Test that updating non-existent peer raises error."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = None
            mock_repository.update_peer.side_effect = ValueError("Peer not found")

            service = PeerFederationService()

            with pytest.raises(ValueError, match="Peer not found"):
                await service.update_peer("nonexistent-peer", {"name": "New Name"})

    @pytest.mark.asyncio
    async def test_update_peer_preserves_federation_token(self, mock_repository):
        """
        Test that updating a peer preserves the federation_token.

        This test validates the fix for issue #561 where update_peer()
        was silently dropping encrypted federation tokens during updates.
        """
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            # Create peer with federation token
            peer_config = PeerRegistryConfig(
                peer_id="test-peer",
                name="Test Peer",
                endpoint="https://test.example.com",
                enabled=True,
                sync_mode="all",
                sync_interval_minutes=60,
                federation_token="secret-token-abc123",
            )

            # Mock repository to return peer with token before update
            mock_repository.get_peer.return_value = peer_config

            # Mock update to return updated peer with token preserved
            updated_config = peer_config.model_copy()
            updated_config.name = "Updated Name"
            updated_config.sync_interval_minutes = 120
            # Token should still be present after update
            updated_config.federation_token = "secret-token-abc123"
            mock_repository.update_peer.return_value = updated_config

            service = PeerFederationService()

            # Update non-token fields
            updates = {
                "name": "Updated Name",
                "sync_interval_minutes": 120,
            }

            result = await service.update_peer("test-peer", updates)

            # Verify token is preserved
            assert result.federation_token == "secret-token-abc123"
            assert result.name == "Updated Name"
            assert result.sync_interval_minutes == 120

    @pytest.mark.asyncio
    async def test_update_peer_token_itself(self, mock_repository):
        """Test that the federation token can be updated directly."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            # Create peer with old token
            peer_config = PeerRegistryConfig(
                peer_id="test-peer",
                name="Test Peer",
                endpoint="https://test.example.com",
                enabled=True,
                sync_mode="all",
                sync_interval_minutes=60,
                federation_token="old-token",
            )

            mock_repository.get_peer.return_value = peer_config

            # Mock update to return peer with new token
            updated_config = peer_config.model_copy()
            updated_config.federation_token = "new-token-xyz"
            mock_repository.update_peer.return_value = updated_config

            service = PeerFederationService()

            # Update just the token
            updates = {"federation_token": "new-token-xyz"}

            result = await service.update_peer("test-peer", updates)

            # Verify token was updated
            assert result.federation_token == "new-token-xyz"
            # Verify other fields unchanged
            assert result.name == "Test Peer"

    @pytest.mark.asyncio
    async def test_remove_peer_success(self, mock_repository, sample_peer_config):
        """Test successfully removing a peer."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = sample_peer_config
            mock_repository.delete_peer.return_value = True

            service = PeerFederationService()
            result = await service.remove_peer(sample_peer_config.peer_id)

            assert result is True
            mock_repository.delete_peer.assert_called_once_with(sample_peer_config.peer_id)

    @pytest.mark.asyncio
    async def test_remove_peer_nonexistent_raises_error(self, mock_repository):
        """Test that removing non-existent peer raises error."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            mock_repository.get_peer.return_value = None
            mock_repository.delete_peer.side_effect = ValueError("Peer not found")

            service = PeerFederationService()

            with pytest.raises(ValueError, match="Peer not found"):
                await service.remove_peer("nonexistent-peer")

    @pytest.mark.asyncio
    async def test_list_peers_from_cache(
        self, mock_repository, sample_peer_config, sample_peer_config_2
    ):
        """Test listing peers uses in-memory cache."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            # Manually populate cache (normally done via load_peers_and_state)
            service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
            service.registered_peers[sample_peer_config_2.peer_id] = sample_peer_config_2

            result = await service.list_peers()

            assert len(result) == 2
            peer_ids = [p.peer_id for p in result]
            assert sample_peer_config.peer_id in peer_ids
            assert sample_peer_config_2.peer_id in peer_ids

    @pytest.mark.asyncio
    async def test_list_peers_enabled_only(
        self, mock_repository, sample_peer_config, sample_peer_config_2
    ):
        """Test listing only enabled peers."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            # Manually populate cache
            service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
            service.registered_peers[sample_peer_config_2.peer_id] = sample_peer_config_2

            result = await service.list_peers(enabled=True)

            assert len(result) == 1
            assert result[0].peer_id == sample_peer_config.peer_id

    @pytest.mark.asyncio
    async def test_list_peers_disabled_only(
        self, mock_repository, sample_peer_config, sample_peer_config_2
    ):
        """Test listing only disabled peers."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            # Manually populate cache
            service.registered_peers[sample_peer_config.peer_id] = sample_peer_config
            service.registered_peers[sample_peer_config_2.peer_id] = sample_peer_config_2

            result = await service.list_peers(enabled=False)

            assert len(result) == 1
            assert result[0].peer_id == sample_peer_config_2.peer_id


@pytest.mark.unit
class TestPeerFederationServiceSyncStatus:
    """Tests for sync status operations."""

    @pytest.mark.asyncio
    async def test_get_sync_status_from_cache(self, mock_repository, sample_peer_config):
        """Test getting sync status from in-memory cache."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            # Manually populate cache
            sync_status = PeerSyncStatus(
                peer_id=sample_peer_config.peer_id,
                is_healthy=True,
                current_generation=5,
            )
            service.peer_sync_status[sample_peer_config.peer_id] = sync_status

            result = await service.get_sync_status(sample_peer_config.peer_id)

            assert result is not None
            assert result.peer_id == sample_peer_config.peer_id
            assert result.is_healthy is True

    @pytest.mark.asyncio
    async def test_get_sync_status_nonexistent(self, mock_repository):
        """Test getting sync status for non-existent peer."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            result = await service.get_sync_status("nonexistent-peer")

            assert result is None

    @pytest.mark.asyncio
    async def test_all_sync_statuses_in_cache(self, mock_repository, sample_peer_config):
        """Test all sync statuses are stored in cache."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            # Manually populate cache
            sync_statuses = [
                PeerSyncStatus(peer_id="peer1", is_healthy=True),
                PeerSyncStatus(peer_id="peer2", is_healthy=False),
            ]
            for status in sync_statuses:
                service.peer_sync_status[status.peer_id] = status

            # Verify cache contains both statuses
            assert len(service.peer_sync_status) == 2
            assert "peer1" in service.peer_sync_status
            assert "peer2" in service.peer_sync_status


@pytest.mark.unit
class TestPeerFederationServiceHelpers:
    """Tests for helper methods on the service."""

    def test_is_locally_overridden_true(self, mock_repository):
        """Test is_locally_overridden returns True when override set."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {
                "sync_metadata": {
                    "local_overrides": True,
                }
            }

            result = service.is_locally_overridden(item)
            assert result is True

    def test_is_locally_overridden_false(self, mock_repository):
        """Test is_locally_overridden returns False when no override."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {
                "sync_metadata": {
                    "local_overrides": False,
                }
            }

            result = service.is_locally_overridden(item)
            assert result is False

    def test_is_locally_overridden_missing_metadata(self, mock_repository):
        """Test is_locally_overridden returns False when no sync_metadata."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {}

            result = service.is_locally_overridden(item)
            assert result is False

    def test_matches_tag_filter_with_match(self, mock_repository):
        """Test _matches_tag_filter returns True when tags match."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {"tags": ["production", "api"]}

            result = service._matches_tag_filter(item, ["production"])
            assert result is True

    def test_matches_tag_filter_no_match(self, mock_repository):
        """Test _matches_tag_filter returns False when no tags match."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {"tags": ["staging", "api"]}

            result = service._matches_tag_filter(item, ["production"])
            assert result is False

    def test_matches_tag_filter_checks_categories(self, mock_repository):
        """Test _matches_tag_filter also checks categories field."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {"tags": [], "categories": ["production"]}

            result = service._matches_tag_filter(item, ["production"])
            assert result is True

    def test_matches_tag_filter_empty_tags(self, mock_repository):
        """Test _matches_tag_filter returns False when item has no tags."""
        with patch(
            "registry.services.peer_federation_service.get_peer_federation_repository",
            return_value=mock_repository,
        ):
            service = PeerFederationService()
            item = {}

            result = service._matches_tag_filter(item, ["production"])
            assert result is False
