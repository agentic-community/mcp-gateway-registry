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
        """Test Settings instantiation with default values.

        SECRET_KEY is required by Settings.__init__; provide a placeholder so we
        can assert the rest of the defaults.
        """
        monkeypatch.delenv("AUTH_SERVER_URL", raising=False)
        monkeypatch.setenv("SECRET_KEY", "test-key-for-defaults-at-least-32-bytes-long")
        monkeypatch.chdir(tmp_path)

        settings = Settings()

        assert settings.session_cookie_name == "mcp_gateway_session"
        assert settings.session_max_age_seconds == 60 * 60 * 8  # 8 hours
        # Secure-by-default: the session cookie carries the Secure flag unless a
        # deployment explicitly opts out (SESSION_COOKIE_SECURE=false).
        assert settings.session_cookie_secure is True
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

    def test_settings_fails_without_secret_key(self, monkeypatch, tmp_path) -> None:
        """Settings refuses to instantiate when SECRET_KEY is unset.

        A per-replica auto-generated key would silently break cookie signing
        across replicas, so the missing-key path is now a hard startup error.
        """
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            Settings(secret_key="")

    def test_settings_fails_without_marker_secret(self, monkeypatch, tmp_path) -> None:
        """Settings refuses to instantiate when AUTH_SERVER_NGINX_MARKER_SECRET is unset.

        An empty marker makes the auth_server mint mcp-proxy tokens
        unconditionally, so a direct :8888 /validate with a forged
        X-Resolved-Upstream could bypass nginx. The missing-marker path is a
        hard startup error, like SECRET_KEY.
        """
        monkeypatch.delenv("AUTH_SERVER_NGINX_MARKER_SECRET", raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError, match="AUTH_SERVER_NGINX_MARKER_SECRET"):
            Settings(
                secret_key="present-and-at-least-32-bytes-long-key",
                auth_server_nginx_marker_secret="",
            )

    def test_settings_secret_key_not_overridden(self) -> None:
        """Test that provided secret_key is not overridden."""
        # Arrange
        custom_key = "my-custom-secret-key-12345-at-least-32-bytes"

        # Act
        settings = Settings(secret_key=custom_key)

        # Assert
        assert settings.secret_key == custom_key

    def test_settings_with_custom_values(self) -> None:
        """Test Settings instantiation with custom values."""
        # Arrange
        custom_values = {
            "secret_key": "test-secret-value-at-least-32-bytes-long",
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
        monkeypatch.setenv("SECRET_KEY", "env-secret-key-at-least-32-bytes-long-value")
        monkeypatch.setenv("SESSION_COOKIE_NAME", "env_session")

        # Act
        settings = Settings()

        # Assert
        assert settings.secret_key == "env-secret-key-at-least-32-bytes-long-value"
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
        # _env_file=None isolates the assertion from a developer's local .env
        # (which may set these keys to empty strings); CI has no .env so this
        # also matches CI behavior exactly.
        settings = Settings(_env_file=None)

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
class TestSettingsSecretKey:
    """Test SECRET_KEY validation behavior.

    SECRET_KEY is now required at construction time so multiple replicas
    cannot diverge on cookie-signing keys (see #960). Provided keys are
    accepted as-is; missing or empty values raise immediately.
    """

    def test_secret_key_empty_string_rejected(self, monkeypatch, tmp_path) -> None:
        """Empty string secret_key is rejected (no silent auto-generation)."""
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            Settings(secret_key="")

    def test_provided_secret_key_used_verbatim(self, monkeypatch) -> None:
        """A provided non-empty secret_key is used verbatim, not regenerated."""
        custom = "my-explicit-32-byte-secret-key-aaaa"
        settings = Settings(secret_key=custom)
        assert settings.secret_key == custom

    def test_short_secret_key_rejected(self, monkeypatch, tmp_path) -> None:
        """A secret_key shorter than 32 characters is rejected at startup."""
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="at least 32"):
            Settings(secret_key="too-short-key")

    def test_weak_default_secret_key_rejected(self, monkeypatch, tmp_path) -> None:
        """The historical 'development-secret-key' literal is rejected."""
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="well-known placeholder"):
            Settings(secret_key="development-secret-key")

    def test_valid_long_secret_key_accepted(self, monkeypatch) -> None:
        """A random 32+ character key is accepted."""
        custom = "a-sufficiently-long-random-secret-key-value"
        settings = Settings(secret_key=custom)
        assert settings.secret_key == custom

    def test_short_marker_secret_rejected(self, monkeypatch, tmp_path) -> None:
        """An AUTH_SERVER_NGINX_MARKER_SECRET shorter than 32 bytes is rejected."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="AUTH_SERVER_NGINX_MARKER_SECRET must be at least"):
            Settings(
                secret_key="a-valid-secret-key-of-at-least-32-bytes",
                auth_server_nginx_marker_secret="short",
            )


# =============================================================================
# TEST CLASS: Session Cookie Configuration
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsSessionCookie:
    """Test session cookie configuration."""

    def test_session_cookie_secure_true_by_default(self, monkeypatch) -> None:
        """Test that session_cookie_secure defaults to True (secure by default).

        _env_file=None isolates this from a developer's local .env, which may
        pin SESSION_COOKIE_SECURE=false for a plain-HTTP dev stack.
        """
        # Arrange
        monkeypatch.delenv("SESSION_COOKIE_SECURE", raising=False)

        # Act
        settings = Settings(_env_file=None)

        # Assert
        assert settings.session_cookie_secure is True

    def test_session_cookie_secure_can_be_disabled(self, monkeypatch) -> None:
        """Test that session_cookie_secure can be explicitly disabled for dev."""
        # Arrange
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")

        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_secure is False

    def test_session_cookie_secure_can_be_enabled(self, monkeypatch) -> None:
        """Test that session_cookie_secure can be explicitly enabled via env var."""
        # Arrange
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")

        # Act
        settings = Settings()

        # Assert
        assert settings.session_cookie_secure is True

    def test_session_cookie_domain_none_by_default(self) -> None:
        """Test that session_cookie_domain is None by default."""
        # Act
        # _env_file=None isolates this from a developer's local .env (which may
        # set SESSION_COOKIE_DOMAIN to an empty string); CI has no .env.
        settings = Settings(_env_file=None)

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
# TEST CLASS: CORS allowlist resolution
# =============================================================================


@pytest.mark.unit
@pytest.mark.core
class TestSettingsCorsAllowedOrigins:
    """Test the fail-closed CORS origin allowlist resolution."""

    def _settings(self, monkeypatch, **env: str) -> Settings:
        """Build Settings with a fixed registry_url and no local .env leakage."""
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        monkeypatch.setenv("REGISTRY_URL", "https://registry.example.com")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return Settings(_env_file=None)

    def test_arbitrary_ec2_origin_is_rejected(self, monkeypatch) -> None:
        """A non-allowlisted EC2 public DNS origin must NOT be trusted.

        This is the core regression guard: the old broad regex matched any
        *.compute*.amazonaws.com host. With an explicit allowlist, such an
        origin only appears if the operator lists it.
        """
        # Arrange - production mode, no CORS origins configured
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(monkeypatch)

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert - the attacker EC2 origin is absent; only the registry's own
        # origin is trusted (fail closed, no wildcard fallback).
        assert "http://ec2-1-2-3-4.compute-1.amazonaws.com" not in origins
        assert "https://ec2-1-2-3-4.compute-1.amazonaws.com" not in origins
        assert origins == ["https://registry.example.com"]

    def test_only_configured_origins_are_allowed(self, monkeypatch) -> None:
        """Only explicitly configured origins (plus self) are returned."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS="https://app.example.com, https://admin.example.com",
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "https://app.example.com" in origins
        assert "https://admin.example.com" in origins
        assert "https://registry.example.com" in origins
        # Nothing else leaks in.
        assert set(origins) == {
            "https://app.example.com",
            "https://admin.example.com",
            "https://registry.example.com",
        }

    def test_no_wildcard_fallback_when_unconfigured(self, monkeypatch) -> None:
        """An empty allowlist must never resolve to '*' or a regex wildcard."""
        # Arrange - production mode, registry_url that cannot resolve to an origin
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        monkeypatch.setenv("REGISTRY_URL", "not-a-valid-url")
        settings = Settings(_env_file=None)

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert - fail closed: empty list, no wildcard.
        assert origins == []
        assert "*" not in origins

    def test_malformed_origins_are_dropped(self, monkeypatch) -> None:
        """Wildcards, paths, and bad schemes are rejected, not widened."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS=(
                "https://*.example.com,"  # wildcard
                "https://good.example.com/path,"  # path component
                "ftp://bad-scheme.example.com,"  # disallowed scheme
                "https://ok.example.com"  # valid
            ),
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert - only the valid entry (and self) survive.
        assert "https://ok.example.com" in origins
        assert not any("*" in o for o in origins)
        assert not any("/path" in o for o in origins)
        assert not any(o.startswith("ftp://") for o in origins)

    def test_userinfo_and_bad_port_origins_are_rejected(self, monkeypatch) -> None:
        """Origins carrying userinfo or an invalid port are dropped."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS=(
                "https://evil@app.example.com,"  # userinfo smuggling
                "https://app.example.com:99999,"  # out-of-range port
                "https://valid.example.com"  # valid
            ),
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "https://valid.example.com" in origins
        assert not any("evil" in o for o in origins)
        assert not any("99999" in o for o in origins)

    def test_ipv6_origin_keeps_brackets(self, monkeypatch) -> None:
        """An IPv6 literal origin is normalized with brackets intact.

        Browsers send IPv6 origins bracketed (e.g. http://[::1]:8080); the
        normalized allowlist entry must match that exact bracketed form.
        """
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS="http://[::1]:8080,https://[2001:db8::1]",
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "http://[::1]:8080" in origins
        assert "https://[2001:db8::1]" in origins

    def test_origin_normalization_strips_path_and_lowercases(self, monkeypatch) -> None:
        """Configured origins are normalized to scheme://host[:port]."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS="HTTPS://App.Example.COM:8443",
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "https://app.example.com:8443" in origins

    def test_local_dev_adds_loopback_origins(self, monkeypatch) -> None:
        """A genuine local-dev stack (loopback registry) auto-trusts loopback.

        Loopback dev origins are only injected when is_local_dev is True AND the
        registry is itself reachable at a loopback host.
        """
        # Arrange - force local dev mode with a loopback registry URL
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: True))
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        monkeypatch.setenv("REGISTRY_URL", "http://localhost:7860")
        settings = Settings(_env_file=None)

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins

    def test_loopback_origins_absent_in_production(self, monkeypatch) -> None:
        """Container/production deployments must NOT auto-trust loopback."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(monkeypatch)

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "http://localhost:3000" not in origins
        assert "http://127.0.0.1:3000" not in origins

    def test_loopback_origins_absent_on_prod_host_without_app_dir(self, monkeypatch) -> None:
        """A bare-metal prod host (no /app, real registry_url) must fail closed.

        is_local_dev is a filesystem heuristic that can be True on hosts that
        merely lack the container /app directory. Loopback dev origins must NOT
        be auto-trusted there, because the registry serves a real public origin.
        """
        # Arrange - is_local_dev True (no /app) but a genuine public registry URL
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: True))
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        monkeypatch.setenv("REGISTRY_URL", "https://registry.example.com")
        settings = Settings(_env_file=None)

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert - no loopback origins leaked in; only the real self-origin.
        assert "http://localhost:3000" not in origins
        assert "http://127.0.0.1:3000" not in origins
        assert origins == ["https://registry.example.com"]

    def test_trailing_dot_host_is_normalized(self, monkeypatch) -> None:
        """A configured trailing-dot FQDN is normalized to match real origins."""
        # Arrange
        monkeypatch.setattr(Settings, "is_local_dev", property(lambda self: False))
        settings = self._settings(
            monkeypatch,
            CORS_ALLOWED_ORIGINS="https://app.example.com.",
        )

        # Act
        origins = settings.cors_allowed_origins_list

        # Assert
        assert "https://app.example.com" in origins
        assert "https://app.example.com." not in origins


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
