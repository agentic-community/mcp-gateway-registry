"""
Unit tests for FileServerRepository.

Tests the file-based repository implementation for MCP server storage.
This includes file I/O operations, state management, and path conversions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from registry.repositories.file.server_repository import FileServerRepository

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_settings():
    """Mock settings with test directories."""
    with patch("registry.repositories.file.server_repository.settings") as mock_settings:
        # Create mock Path objects
        mock_servers_dir = MagicMock(spec=Path)
        mock_servers_dir.__truediv__ = lambda self, other: MagicMock(spec=Path)
        mock_servers_dir.mkdir = MagicMock()

        mock_state_path = MagicMock(spec=Path)
        mock_state_path.exists = MagicMock(return_value=False)

        mock_settings.servers_dir = mock_servers_dir
        mock_settings.state_file_path = mock_state_path
        yield mock_settings


@pytest.fixture
def server_repository(mock_settings):
    """Create a FileServerRepository instance for testing."""
    return FileServerRepository()


@pytest.fixture
def sample_server_dict() -> Dict[str, Any]:
    """Sample server data for testing."""
    return {
        "path": "/test-server",
        "server_name": "Test Server",
        "description": "A test server",
        "tags": ["test"],
        "num_tools": 5,
    }


# =============================================================================
# TEST: _path_to_filename Method
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestPathToFilename:
    """Tests for _path_to_filename helper method."""

    def test_path_to_filename_simple(self, server_repository):
        """Test conversion of simple path to filename."""
        # Act
        result = server_repository._path_to_filename("/test-server")

        # Assert
        assert result == "test-server.json"

    def test_path_to_filename_nested(self, server_repository):
        """Test conversion of nested path to filename."""
        # Act
        result = server_repository._path_to_filename("/api/v1/test-server")

        # Assert
        assert result == "api_v1_test-server.json"

    def test_path_to_filename_with_trailing_slash(self, server_repository):
        """Test path with trailing slash."""
        # Act
        result = server_repository._path_to_filename("/test-server/")

        # Assert
        assert result == "test-server_.json"

    def test_path_to_filename_already_has_json(self, server_repository):
        """Test path that already has .json extension."""
        # Act
        result = server_repository._path_to_filename("/test-server.json")

        # Assert
        assert result == "test-server.json"

    def test_path_to_filename_multiple_slashes(self, server_repository):
        """Test path with multiple directory levels."""
        # Act
        result = server_repository._path_to_filename("/api/v1/servers/test")

        # Assert
        assert result == "api_v1_servers_test.json"


# =============================================================================
# TEST: _save_to_file Method
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestSaveToFile:
    """Tests for _save_to_file method."""

    @pytest.mark.asyncio
    async def test_save_to_file_success(self, server_repository, sample_server_dict, mock_settings):
        """Test successful file save."""
        # Arrange
        m = mock_open()

        with patch("builtins.open", m):
            # Act
            result = await server_repository._save_to_file(sample_server_dict)

            # Assert
            assert result is True
            mock_settings.servers_dir.mkdir.assert_called_with(parents=True, exist_ok=True)
            m.assert_called_once()
            # Verify JSON was written
            handle = m()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert "Test Server" in written_data

    @pytest.mark.asyncio
    async def test_save_to_file_creates_directory(
        self, server_repository, sample_server_dict, mock_settings
    ):
        """Test that save creates directory if missing."""
        # Arrange
        m = mock_open()

        with patch("builtins.open", m):
            # Act
            await server_repository._save_to_file(sample_server_dict)

            # Assert
            mock_settings.servers_dir.mkdir.assert_called_with(parents=True, exist_ok=True)

    @pytest.mark.asyncio
    async def test_save_to_file_handles_errors(
        self, server_repository, sample_server_dict, mock_settings
    ):
        """Test error handling when save fails."""
        # Arrange
        with patch("builtins.open", side_effect=IOError("Disk full")):
            # Act
            result = await server_repository._save_to_file(sample_server_dict)

            # Assert
            assert result is False


# =============================================================================
# TEST: _save_state Method
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestSaveState:
    """Tests for _save_state method."""

    @pytest.mark.asyncio
    async def test_save_state_success(self, server_repository, mock_settings):
        """Test successful state persistence."""
        # Arrange
        server_repository._state = {"/test1": True, "/test2": False}
        m = mock_open()

        with patch("builtins.open", m):
            # Act
            await server_repository._save_state()

            # Assert
            m.assert_called_once_with(mock_settings.state_file_path, "w")
            handle = m()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            parsed_data = json.loads(written_data)
            assert parsed_data == {"/test1": True, "/test2": False}

    @pytest.mark.asyncio
    async def test_save_state_handles_errors(self, server_repository, mock_settings):
        """Test error handling when state save fails."""
        # Arrange
        server_repository._state = {"/test": True}

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            # Act - should not raise exception
            await server_repository._save_state()

            # Assert - just verify it doesn't crash
            # Error is logged, operation continues


# =============================================================================
# TEST: _load_state Method
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestLoadState:
    """Tests for _load_state method."""

    @pytest.mark.asyncio
    async def test_load_state_with_existing_file(self, server_repository, mock_settings):
        """Test loading state from existing file."""
        # Arrange
        server_repository._servers = {"/test1": {}, "/test2": {}}
        state_data = {"/test1": True, "/test2": False}
        mock_settings.state_file_path.exists.return_value = True
        m = mock_open(read_data=json.dumps(state_data))

        with patch("builtins.open", m):
            # Act
            await server_repository._load_state()

            # Assert
            assert server_repository._state == {"/test1": True, "/test2": False}

    @pytest.mark.asyncio
    async def test_load_state_no_file(self, server_repository, mock_settings):
        """Test loading state when file doesn't exist."""
        # Arrange
        server_repository._servers = {"/test1": {}, "/test2": {}}
        mock_settings.state_file_path.exists.return_value = False

        # Act
        await server_repository._load_state()

        # Assert
        # All servers should default to False (disabled)
        assert server_repository._state == {"/test1": False, "/test2": False}

    @pytest.mark.asyncio
    async def test_load_state_handles_trailing_slash_normalization(
        self, server_repository, mock_settings
    ):
        """Test state loading normalizes trailing slashes."""
        # Arrange
        server_repository._servers = {"/test": {}}
        state_data = {"/test/": True}  # State has trailing slash
        mock_settings.state_file_path.exists.return_value = True
        m = mock_open(read_data=json.dumps(state_data))

        with patch("builtins.open", m):
            # Act
            await server_repository._load_state()

            # Assert
            assert server_repository._state["/test"] is True

    @pytest.mark.asyncio
    async def test_load_state_handles_corrupt_file(self, server_repository, mock_settings):
        """Test loading state when file is corrupted."""
        # Arrange
        server_repository._servers = {"/test": {}}
        mock_settings.state_file_path.exists.return_value = True
        m = mock_open(read_data="invalid json {{{")

        with patch("builtins.open", m):
            # Act
            await server_repository._load_state()

            # Assert
            # Should fall back to default (disabled)
            assert server_repository._state == {"/test": False}


# =============================================================================
# TEST: Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestFileServerRepositoryIntegration:
    """Integration tests for file repository operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_server(
        self, server_repository, sample_server_dict, mock_settings
    ):
        """Test creating and retrieving a server."""
        # Arrange
        m = mock_open()

        with patch("builtins.open", m):
            # Act
            create_result = await server_repository.create(sample_server_dict)
            get_result = await server_repository.get("/test-server")

            # Assert
            assert create_result is True
            assert get_result == sample_server_dict
            assert server_repository._state["/test-server"] is False  # Disabled by default

    @pytest.mark.asyncio
    async def test_update_server_saves_to_file(
        self, server_repository, sample_server_dict, mock_settings
    ):
        """Test updating server writes to file."""
        # Arrange
        server_repository._servers["/test-server"] = sample_server_dict.copy()
        updated_data = sample_server_dict.copy()
        updated_data["description"] = "Updated description"

        m = mock_open()

        with patch("builtins.open", m):
            # Act
            result = await server_repository.update("/test-server", updated_data)

            # Assert
            assert result is True
            # Verify file was written
            m.assert_called()
