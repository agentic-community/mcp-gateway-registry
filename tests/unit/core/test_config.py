"""
Unit tests for registry.core.config module.

This module tests the Settings class and its configuration management,
including default values, environment variable loading, path resolution,
and computed properties.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from registry.core.config import Settings

# =============================================================================
# TEST CLASS: Settings Instantiation and Defaults
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsInstantiation:
    """Test Settings class instantiation and default values."""

    def test_settings_default_values(self, monkeypatch, tmp_path) -> None:
        """Test Settings instantiation with default values."""
        # Arrange - Clear environment variables and disable .env file loading
        monkeypatch.delenv("AUTH_SERVER_URL", raising=False)
        monkeypatch.delenv("SECRET_KEY", raising=False)

        # Change to temp directory to prevent .env file loading
        monkeypatch.chdir(tmp_path)

        # Act
        settings = Settings()

        # Assert - Auth settings
        assert settings.session_cookie_name == "mcp_gateway_session"
        assert settings.session_max_age_seconds == 60 * 60 * 8  # 8 hours
        assert settings.session_cookie_secure is False
        assert settings.session_cookie_domain is None
        assert settings.auth_server_url == "http://localhost:8888"
        assert settings.auth_server_external_url == "http://localhost:8888"

    def test_settings_embeddings_default_values(self) -> None:
        """Test embeddings-related default values."""
        # Act
        settings = Settings()

        # Assert - Embeddings settings
        assert settings.embeddings_provider == "sentence-transformers"
        assert settings.embeddings_model_name == "all-MiniLM-L6-v2"
        assert settings.embeddings_model_dimensions == 384
        assert settings.embeddings_api_key is None
        assert settings.embeddings_secret_key is None
        assert settings.embeddings_api_base is None
        assert settings.embeddings_aws_region == "us-east-1"

    def test_settings_health_check_defaults(self) -> None:
        """Test health check default values."""
        # Act
        settings = Settings()

        # Assert
        assert settings.health_check_interval_seconds == 300  # 5 minutes
        assert settings.health_check_timeout_seconds == 2

    def test_settings_websocket_defaults(self) -> None:
        """Test WebSocket performance default values."""
        # Act
        settings = Settings()

        # Assert
        assert settings.max_websocket_connections == 100
        assert settings.websocket_send_timeout_seconds == 2.0
        assert settings.websocket_broadcast_interval_ms == 10
        assert settings.websocket_max_batch_size == 20
        assert settings.websocket_cache_ttl_seconds == 1

    def test_settings_wellknown_defaults(self) -> None:
        """Test well-known discovery default values."""
        # Act
        settings = Settings()

        # Assert
        assert settings.enable_wellknown_discovery is True
        assert settings.wellknown_cache_ttl == 300  # 5 minutes

    def test_settings_container_paths_defaults(self) -> None:
        """Test container path default values."""
        # Act
        settings = Settings()

        # Assert
        assert settings.container_app_dir == Path("/app")
        assert settings.container_registry_dir == Path("/app/registry")
        assert settings.container_log_dir == Path("/app/logs")

    def test_settings_secret_key_auto_generation(self, monkeypatch, tmp_path) -> None:
        """Test that secret_key is auto-generated when not provided."""
        # Arrange - Clear SECRET_KEY env var and disable .env file loading
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        # Act
        settings = Settings()

        # Assert
        assert settings.secret_key != ""
        assert len(settings.secret_key) == 64  # 32 bytes hex = 64 chars
        assert all(c in "0123456789abcdef" for c in settings.secret_key)

    def test_settings_secret_key_not_overridden(self) -> None:
        """Test that provided secret_key is not overridden."""
        # Arrange
        custom_key = "my-custom-secret-key-12345"

        # Act
        settings = Settings(secret_key=custom_key)

        # Assert
        assert settings.secret_key == custom_key

    def test_settings_with_custom_values(self) -> None:
        """Test Settings instantiation with custom values."""
        # Arrange
        custom_values = {
            "secret_key": "test-secret",
            "session_cookie_name": "test_cookie",
            "session_max_age_seconds": 3600,
            "embeddings_provider": "litellm",
            "embeddings_model_name": "text-embedding-3-small",
            "embeddings_model_dimensions": 1024,
            "health_check_interval_seconds": 600,
        }

        # Act
        settings = Settings(**custom_values)

        # Assert
        assert settings.secret_key == custom_values["secret_key"]
        assert settings.session_cookie_name == custom_values["session_cookie_name"]
        assert settings.session_max_age_seconds == custom_values["session_max_age_seconds"]
        assert settings.embeddings_provider == custom_values["embeddings_provider"]
        assert settings.embeddings_model_name == custom_values["embeddings_model_name"]
        assert settings.embeddings_model_dimensions == custom_values["embeddings_model_dimensions"]
        assert (
            settings.health_check_interval_seconds == custom_values["health_check_interval_seconds"]
        )


# =============================================================================
# TEST CLASS: Environment Variable Loading
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsEnvironmentVariables:
    """Test Settings loading from environment variables."""

    def test_settings_load_from_env_auth(self, monkeypatch) -> None:
        """Test loading auth settings from environment variables."""
        # Arrange
        monkeypatch.setenv("SECRET_KEY", "env-secret-key")
        monkeypatch.setenv("SESSION_COOKIE_NAME", "env_session")

        # Act
        settings = Settings()

        # Assert
        assert settings.secret_key == "env-secret-key"
        assert settings.session_cookie_name == "env_session"

    def test_settings_load_from_env_embeddings(self, monkeypatch) -> None:
        """Test loading embeddings settings from environment variables."""
        # Arrange
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", "litellm")
        monkeypatch.setenv("EMBEDDINGS_MODEL_NAME", "bedrock/amazon.titan-embed-text-v2:0")
        monkeypatch.setenv("EMBEDDINGS_MODEL_DIMENSIONS", "1024")
        monkeypatch.setenv("EMBEDDINGS_API_KEY", "test-api-key")
        monkeypatch.setenv("EMBEDDINGS_AWS_REGION", "us-west-2")

        # Act
        settings = Settings()

        # Assert
        assert settings.embeddings_provider == "litellm"
        assert settings.embeddings_model_name == "bedrock/amazon.titan-embed-text-v2:0"
        assert settings.embeddings_model_dimensions == 1024
        assert settings.embeddings_api_key == "test-api-key"
        assert settings.embeddings_aws_region == "us-west-2"

    def test_settings_load_from_env_health_check(self, monkeypatch) -> None:
        """Test loading health check settings from environment variables."""
        # Arrange
        monkeypatch.setenv("HEALTH_CHECK_INTERVAL_SECONDS", "600")
        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "5")

        # Act
        settings = Settings()

        # Assert
        assert settings.health_check_interval_seconds == 600
        assert settings.health_check_timeout_seconds == 5

    def test_settings_load_from_env_websocket(self, monkeypatch) -> None:
        """Test loading WebSocket settings from environment variables."""
        # Arrange
        monkeypatch.setenv("MAX_WEBSOCKET_CONNECTIONS", "200")
        monkeypatch.setenv("WEBSOCKET_SEND_TIMEOUT_SECONDS", "5.0")
        monkeypatch.setenv("WEBSOCKET_BROADCAST_INTERVAL_MS", "20")

        # Act
        settings = Settings()

        # Assert
        assert settings.max_websocket_connections == 200
        assert settings.websocket_send_timeout_seconds == 5.0
        assert settings.websocket_broadcast_interval_ms == 20

    def test_settings_env_case_insensitive(self, monkeypatch) -> None:
        """Test that environment variables are case-insensitive."""
        # Arrange - using lowercase env var names
        monkeypatch.setenv("session_cookie_name", "lowercase_session")
        monkeypatch.setenv("AUTH_SERVER_URL", "http://uppercase:8888")

        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_name == "lowercase_session"
        assert settings.auth_server_url == "http://uppercase:8888"

    def test_settings_extra_env_ignored(self, monkeypatch) -> None:
        """Test that extra environment variables are ignored."""
        # Arrange
        monkeypatch.setenv("UNKNOWN_VARIABLE", "some_value")
        monkeypatch.setenv("ANOTHER_UNKNOWN", "another_value")

        # Act - Should not raise an error
        settings = Settings()

        # Assert
        assert not hasattr(settings, "unknown_variable")
        assert not hasattr(settings, "another_unknown")

    def test_settings_optional_fields_none(self) -> None:
        """Test that optional fields can be None."""
        # Act
        settings = Settings()

        # Assert - Optional fields should be None by default
        assert settings.embeddings_api_key is None
        assert settings.embeddings_secret_key is None
        assert settings.embeddings_api_base is None
        assert settings.session_cookie_domain is None


# =============================================================================
# TEST CLASS: Path Properties - Local Development
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsPathsLocalDev:
    """Test path properties in local development mode."""

    @patch("registry.core.config.Path")
    def test_is_local_dev_true(self, mock_path_class) -> None:
        """Test is_local_dev property when /app does not exist."""
        # Arrange
        mock_app_path = MagicMock()
        mock_app_path.exists.return_value = False
        mock_path_class.return_value = mock_app_path

        # Act
        settings = Settings()

        # Assert
        assert settings.is_local_dev is True

    @patch("registry.core.config.Path")
    def test_is_local_dev_false(self, mock_path_class) -> None:
        """Test is_local_dev property when /app exists."""
        # Arrange
        mock_app_path = MagicMock()
        mock_app_path.exists.return_value = True
        mock_path_class.return_value = mock_app_path

        # Act
        settings = Settings()

        # Assert
        assert settings.is_local_dev is False

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_servers_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test servers_dir property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.servers_dir

        # Assert
        expected = Path.cwd() / "registry" / "servers"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_static_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test static_dir property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.static_dir

        # Assert
        expected = Path.cwd() / "registry" / "static"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_templates_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test templates_dir property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.templates_dir

        # Assert
        expected = Path.cwd() / "registry" / "templates"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_log_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test log_dir property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.log_dir

        # Assert
        expected = Path.cwd() / "logs"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_log_file_path_local_dev(self, mock_is_local_dev) -> None:
        """Test log_file_path property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.log_file_path

        # Assert
        expected = Path.cwd() / "logs" / "registry.log"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_dotenv_path_local_dev(self, mock_is_local_dev) -> None:
        """Test dotenv_path property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.dotenv_path

        # Assert
        expected = Path.cwd() / ".env"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_agents_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test agents_dir property in local development mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.agents_dir

        # Assert
        expected = Path.cwd() / "registry" / "agents"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: True))
    def test_embeddings_model_dir_local_dev(self, mock_is_local_dev) -> None:
        """Test embeddings_model_dir property in local development mode."""
        # Arrange
        settings = Settings(embeddings_model_name="test-model")

        # Act
        result = settings.embeddings_model_dir

        # Assert
        expected = Path.cwd() / "registry" / "models" / "test-model"
        assert result == expected


# =============================================================================
# TEST CLASS: Path Properties - Container Mode
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsPathsContainer:
    """Test path properties in container/production mode."""

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_servers_dir_container(self, mock_is_local_dev) -> None:
        """Test servers_dir property in container mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.servers_dir

        # Assert
        expected = Path("/app/registry") / "servers"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_static_dir_container(self, mock_is_local_dev) -> None:
        """Test static_dir property in container mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.static_dir

        # Assert
        expected = Path("/app/registry") / "static"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_templates_dir_container(self, mock_is_local_dev) -> None:
        """Test templates_dir property in container mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.templates_dir

        # Assert
        expected = Path("/app/registry") / "templates"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_log_dir_container(self, mock_is_local_dev) -> None:
        """Test log_dir property in container mode (issue #987 default)."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.log_dir

        # Assert
        expected = Path("/var/log/containers/ai-registry")
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_log_file_path_container(self, mock_is_local_dev) -> None:
        """Test log_file_path property in container mode (issue #987 default)."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.log_file_path

        # Assert
        expected = Path("/var/log/containers/ai-registry") / "registry.log"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_dotenv_path_container(self, mock_is_local_dev) -> None:
        """Test dotenv_path property in container mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.dotenv_path

        # Assert
        expected = Path("/app/registry") / ".env"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_agents_dir_container(self, mock_is_local_dev) -> None:
        """Test agents_dir property in container mode."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.agents_dir

        # Assert
        expected = Path("/app/registry") / "agents"
        assert result == expected

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_embeddings_model_dir_container(self, mock_is_local_dev) -> None:
        """Test embeddings_model_dir property in container mode."""
        # Arrange
        settings = Settings(embeddings_model_name="test-model")

        # Act
        result = settings.embeddings_model_dir

        # Assert
        expected = Path("/app/registry") / "models" / "test-model"
        assert result == expected


# =============================================================================
# TEST CLASS: Fixed Path Properties
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsFixedPaths:
    """Test path properties that don't depend on is_local_dev."""

    def test_nginx_config_path(self) -> None:
        """Test nginx_config_path property."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.nginx_config_path

        # Assert
        assert result == Path("/etc/nginx/conf.d/nginx_rev_proxy.conf")

    @patch.object(
        Settings, "servers_dir", new_callable=lambda: property(lambda self: Path("/test/servers"))
    )
    def test_state_file_path(self, mock_servers_dir) -> None:
        """Test state_file_path property."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.state_file_path

        # Assert
        expected = Path("/test/servers") / "server_state.json"
        assert result == expected

    @patch.object(
        Settings, "servers_dir", new_callable=lambda: property(lambda self: Path("/test/servers"))
    )
    def test_faiss_index_path(self, mock_servers_dir) -> None:
        """Test faiss_index_path property."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.faiss_index_path

        # Assert
        expected = Path("/test/servers") / "service_index.faiss"
        assert result == expected

    @patch.object(
        Settings, "servers_dir", new_callable=lambda: property(lambda self: Path("/test/servers"))
    )
    def test_faiss_metadata_path(self, mock_servers_dir) -> None:
        """Test faiss_metadata_path property."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.faiss_metadata_path

        # Assert
        expected = Path("/test/servers") / "service_index_metadata.json"
        assert result == expected

    @patch.object(
        Settings, "agents_dir", new_callable=lambda: property(lambda self: Path("/test/agents"))
    )
    def test_agent_state_file_path(self, mock_agents_dir) -> None:
        """Test agent_state_file_path property."""
        # Arrange
        settings = Settings()

        # Act
        result = settings.agent_state_file_path

        # Assert
        expected = Path("/test/agents") / "agent_state.json"
        assert result == expected


# =============================================================================
# TEST CLASS: Embeddings Provider Configuration
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsEmbeddingsProviders:
    """Test embeddings provider configurations."""

    def test_sentence_transformers_provider(self) -> None:
        """Test sentence-transformers provider configuration."""
        # Act
        settings = Settings(
            embeddings_provider="sentence-transformers",
            embeddings_model_name="all-MiniLM-L6-v2",
            embeddings_model_dimensions=384,
        )

        # Assert
        assert settings.embeddings_provider == "sentence-transformers"
        assert settings.embeddings_model_name == "all-MiniLM-L6-v2"
        assert settings.embeddings_model_dimensions == 384
        assert settings.embeddings_api_key is None
        assert settings.embeddings_secret_key is None
        assert settings.embeddings_api_base is None

    def test_litellm_provider_with_api_key(self) -> None:
        """Test litellm provider configuration with API key."""
        # Act
        settings = Settings(
            embeddings_provider="litellm",
            embeddings_model_name="text-embedding-3-small",
            embeddings_model_dimensions=1536,
            embeddings_api_key="test-api-key",
            embeddings_api_base="https://api.openai.com/v1",
        )

        # Assert
        assert settings.embeddings_provider == "litellm"
        assert settings.embeddings_model_name == "text-embedding-3-small"
        assert settings.embeddings_model_dimensions == 1536
        assert settings.embeddings_api_key == "test-api-key"
        assert settings.embeddings_api_base == "https://api.openai.com/v1"

    def test_litellm_provider_bedrock(self) -> None:
        """Test litellm provider configuration for Amazon Bedrock."""
        # Act
        settings = Settings(
            embeddings_provider="litellm",
            embeddings_model_name="bedrock/amazon.titan-embed-text-v2:0",
            embeddings_model_dimensions=1024,
            embeddings_aws_region="us-west-2",
        )

        # Assert
        assert settings.embeddings_provider == "litellm"
        assert settings.embeddings_model_name == "bedrock/amazon.titan-embed-text-v2:0"
        assert settings.embeddings_model_dimensions == 1024
        assert settings.embeddings_aws_region == "us-west-2"
        # API key should be None for Bedrock (uses AWS credentials)
        assert settings.embeddings_api_key is None


# =============================================================================
# TEST CLASS: Settings Model Configuration
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsModelConfig:
    """Test Pydantic model configuration."""

    def test_settings_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored per model config."""
        # Act - Should not raise an error
        settings = Settings(
            unknown_field="should_be_ignored",
            another_unknown=123,
        )

        # Assert
        assert not hasattr(settings, "unknown_field")
        assert not hasattr(settings, "another_unknown")

    def test_settings_preserves_field_names(self) -> None:
        """Test that constructor uses exact field names."""
        # Act
        settings = Settings(
            session_cookie_name="test_cookie",
            auth_server_url="http://test:8888",
        )

        # Assert
        assert settings.session_cookie_name == "test_cookie"
        assert settings.auth_server_url == "http://test:8888"


# =============================================================================
# TEST CLASS: Integration with Test Fixtures
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsWithFixtures:
    """Test Settings class with pytest fixtures."""

    def test_test_settings_fixture(self, test_settings: Settings) -> None:
        """Test that test_settings fixture provides valid Settings."""
        # Assert
        assert isinstance(test_settings, Settings)
        assert test_settings.secret_key == "test-secret-key-for-testing-only"

    def test_test_settings_paths_are_temp(self, test_settings: Settings, tmp_path: Path) -> None:
        """Test that test_settings uses temporary paths."""
        # Assert - paths should be within tmp_path or be Path objects
        assert isinstance(test_settings.servers_dir, Path)
        assert isinstance(test_settings.agents_dir, Path)
        assert isinstance(test_settings.embeddings_model_dir, Path)
        assert isinstance(test_settings.log_dir, Path)


# =============================================================================
# TEST CLASS: Secret Key Generation
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsSecretKeyGeneration:
    """Test secret key generation logic."""

    def test_secret_key_generated_when_empty_string(self) -> None:
        """Test that secret key is generated when provided as empty string."""
        # Act
        settings = Settings(secret_key="")

        # Assert
        assert settings.secret_key != ""
        assert len(settings.secret_key) == 64

    def test_secret_key_different_on_each_instantiation(self) -> None:
        """Test that generated secret keys are different for each instance."""
        # Act
        settings1 = Settings(secret_key="")
        settings2 = Settings(secret_key="")

        # Assert
        assert settings1.secret_key != settings2.secret_key

    def test_secret_key_is_hex_string(self) -> None:
        """Test that generated secret key is a valid hex string."""
        # Act
        settings = Settings(secret_key="")

        # Assert
        # Should be 64 character hex string (32 bytes)
        assert len(settings.secret_key) == 64
        try:
            bytes.fromhex(settings.secret_key)
            is_valid_hex = True
        except ValueError:
            is_valid_hex = False
        assert is_valid_hex


# =============================================================================
# TEST CLASS: Session Cookie Configuration
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsSessionCookie:
    """Test session cookie configuration."""

    def test_session_cookie_secure_false_by_default(self) -> None:
        """Test that session_cookie_secure is False by default."""
        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_secure is False

    def test_session_cookie_secure_can_be_enabled(self, monkeypatch) -> None:
        """Test that session_cookie_secure can be enabled via env var."""
        # Arrange
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")

        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_secure is True

    def test_session_cookie_domain_none_by_default(self) -> None:
        """Test that session_cookie_domain is None by default."""
        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_domain is None

    def test_session_cookie_domain_can_be_set(self, monkeypatch) -> None:
        """Test that session_cookie_domain can be set via env var."""
        # Arrange
        monkeypatch.setenv("SESSION_COOKIE_DOMAIN", ".example.com")

        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_domain == ".example.com"

    def test_session_max_age_default(self) -> None:
        """Test that session_max_age_seconds has correct default."""
        # Act
        settings = Settings()

        # Assert
        assert settings.session_max_age_seconds == 28800  # 8 hours in seconds


# =============================================================================
# TEST CLASS: Auth Server URLs
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsAuthServerUrls:
    """Test auth server URL configuration."""

    def test_auth_server_urls_default_to_localhost(self, monkeypatch, tmp_path) -> None:
        """Test that auth server URLs default to localhost."""
        # Arrange - Clear AUTH_SERVER_URL env vars and disable .env file loading
        monkeypatch.delenv("AUTH_SERVER_URL", raising=False)
        monkeypatch.delenv("AUTH_SERVER_EXTERNAL_URL", raising=False)
        monkeypatch.chdir(tmp_path)

        # Act
        settings = Settings()

        # Assert
        assert settings.auth_server_url == "http://localhost:8888"
        assert settings.auth_server_external_url == "http://localhost:8888"

    def test_auth_server_urls_can_differ(self, monkeypatch) -> None:
        """Test that internal and external auth URLs can be different."""
        # Arrange
        monkeypatch.setenv("AUTH_SERVER_URL", "http://auth-internal:8888")
        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "https://auth.example.com")

        # Act
        settings = Settings()

        # Assert
        assert settings.auth_server_url == "http://auth-internal:8888"
        assert settings.auth_server_external_url == "https://auth.example.com"


# =============================================================================
# TEST CLASS: Settings Tab Visibility Feature Flags
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsTabVisibilityFeatureFlags:
    """Test SHOW_*_TAB + REGISTRY_MODE precedence in get_config() response."""

    @pytest.mark.asyncio
    async def test_settings_tab_defaults_match_current_behavior(self):
        """All defaults (true) produce same features as REGISTRY_MODE=full."""
        from registry.api.config_routes import get_config

        result = await get_config()
        features = result["features"]

        assert features["mcp_servers"] is True
        assert features["agents"] is True
        assert features["skills"] is True
        assert features["virtual_servers"] is True

    @pytest.mark.asyncio
    async def test_settings_tab_show_false_hides_feature(self, monkeypatch, tmp_path):
        """Setting SHOW_AGENTS_TAB=false hides the tab even with REGISTRY_MODE=full."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SHOW_AGENTS_TAB", "false")

        new_settings = Settings()
        with patch("registry.api.config_routes.settings", new_settings):
            from registry.api.config_routes import get_config

            result = await get_config()
            assert result["features"]["agents"] is False
            assert result["features"]["mcp_servers"] is True

    @pytest.mark.asyncio
    async def test_settings_tab_mode_disables_feature_regardless(self, monkeypatch, tmp_path):
        """REGISTRY_MODE=mcp-servers-only hides agents even if SHOW_AGENTS_TAB=true."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("REGISTRY_MODE", "mcp-servers-only")
        monkeypatch.setenv("SHOW_AGENTS_TAB", "true")

        new_settings = Settings()
        with patch("registry.api.config_routes.settings", new_settings):
            from registry.api.config_routes import get_config

            result = await get_config()
            assert result["features"]["agents"] is False
            assert result["features"]["mcp_servers"] is True

    @pytest.mark.asyncio
    async def test_settings_tab_virtual_servers_key_present(self):
        """virtual_servers key is present in features dict."""
        from registry.api.config_routes import get_config

        result = await get_config()
        assert "virtual_servers" in result["features"]
        assert result["features"]["virtual_servers"] is True

    @pytest.mark.asyncio
    async def test_settings_tab_virtual_servers_false(self, monkeypatch, tmp_path):
        """SHOW_VIRTUAL_SERVERS_TAB=false hides virtual servers."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SHOW_VIRTUAL_SERVERS_TAB", "false")

        new_settings = Settings()
        with patch("registry.api.config_routes.settings", new_settings):
            from registry.api.config_routes import get_config

            result = await get_config()
            assert result["features"]["virtual_servers"] is False

    @pytest.mark.asyncio
    async def test_settings_tab_virtual_servers_hidden_by_mode(self, monkeypatch, tmp_path):
        """REGISTRY_MODE=agents-only hides virtual servers even if SHOW_VIRTUAL_SERVERS_TAB=true."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("REGISTRY_MODE", "agents-only")
        monkeypatch.setenv("SHOW_VIRTUAL_SERVERS_TAB", "true")

        new_settings = Settings()
        with patch("registry.api.config_routes.settings", new_settings):
            from registry.api.config_routes import get_config

            result = await get_config()
            assert result["features"]["virtual_servers"] is False
            assert result["features"]["agents"] is True


# =============================================================================
# TEST CLASS: Settings Tab Visibility Startup Warnings
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsTabVisibilityStartupWarnings:
    """Test log_tab_visibility_warnings() logs correctly."""

    def test_settings_tab_warning_for_ineffective_override(self, monkeypatch, tmp_path, caplog):
        """Warning logged when SHOW_AGENTS_TAB=true but mode disables agents."""
        monkeypatch.delenv("SHOW_AGENTS_TAB", raising=False)
        monkeypatch.setenv("REGISTRY_MODE", "mcp-servers-only")
        monkeypatch.chdir(tmp_path)

        import logging

        from registry.core.config import log_tab_visibility_warnings

        s = Settings()
        with caplog.at_level(logging.WARNING):
            log_tab_visibility_warnings(s)

        assert any(
            "SHOW_AGENTS_TAB" in msg and "mcp-servers-only" in msg for msg in caplog.messages
        )

    def test_settings_tab_no_warning_when_consistent(self, monkeypatch, tmp_path, caplog):
        """No warning when all SHOW_*_TAB are consistent with REGISTRY_MODE=full."""
        monkeypatch.setenv("REGISTRY_MODE", "full")
        monkeypatch.chdir(tmp_path)

        import logging

        from registry.core.config import log_tab_visibility_warnings

        s = Settings()
        with caplog.at_level(logging.WARNING):
            log_tab_visibility_warnings(s)

        assert not any("SHOW_" in msg for msg in caplog.messages)


# =============================================================================
# TEST CLASS: APP_LOG_DIR and APP_LOG_FILE_FORMAT (Issue #987)
# =============================================================================


class TestAppLogDirValidator:
    """Tests for the APP_LOG_DIR path validator added in issue #987."""

    def test_none_value_accepted(self) -> None:
        """APP_LOG_DIR unset should leave app_log_dir=None."""
        settings = Settings(app_log_dir=None)
        assert settings.app_log_dir is None

    def test_empty_string_treated_as_unset(self) -> None:
        """Empty string should be normalized to None (uses default)."""
        settings = Settings(app_log_dir="")
        assert settings.app_log_dir is None

    def test_absolute_path_accepted(self) -> None:
        """Absolute paths are the only supported form."""
        settings = Settings(app_log_dir="/var/log/custom")
        assert settings.app_log_dir == "/var/log/custom"

    def test_relative_path_rejected(self) -> None:
        """Relative paths should be rejected at startup."""
        with pytest.raises(ValueError, match="absolute"):
            Settings(app_log_dir="relative/path")

    def test_path_with_dotdot_rejected(self) -> None:
        """Paths containing '..' segments should be rejected (defense in depth)."""
        with pytest.raises(ValueError, match="'\\.\\.'"):
            Settings(app_log_dir="/var/log/../etc")

    @patch.object(Settings, "is_local_dev", new_callable=lambda: property(lambda self: False))
    def test_override_honored_by_log_dir_property(
        self,
        mock_is_local_dev,
    ) -> None:
        """Setting app_log_dir overrides the mode-based default."""
        settings = Settings(app_log_dir="/custom/path")
        assert settings.log_dir == Path("/custom/path")


class TestAppLogFileFormatValidator:
    """Tests for the APP_LOG_FILE_FORMAT validator added in issue #987."""

    def test_json_accepted(self) -> None:
        settings = Settings(app_log_file_format="json")
        assert settings.app_log_file_format == "json"

    def test_text_accepted(self) -> None:
        settings = Settings(app_log_file_format="text")
        assert settings.app_log_file_format == "text"

    def test_case_insensitive(self) -> None:
        settings = Settings(app_log_file_format="JSON")
        assert settings.app_log_file_format == "json"

    def test_whitespace_stripped(self) -> None:
        settings = Settings(app_log_file_format="  text  ")
        assert settings.app_log_file_format == "text"

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="json.*text"):
            Settings(app_log_file_format="yaml")

    def test_default_is_json(self) -> None:
        settings = Settings()
        assert settings.app_log_file_format == "json"
