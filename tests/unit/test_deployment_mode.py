"""
Unit tests for deployment mode configuration and validation.

Tests the DeploymentMode/RegistryMode enums, validation logic,
and nginx_updates_enabled property.
"""

import pytest

from registry.core.config import (
    DeploymentMode,
    RegistryMode,
    Settings,
    _validate_mode_combination,
)


# =============================================================================
# TEST CLASS: Deployment Mode Validation
# =============================================================================


@pytest.mark.unit
class TestDeploymentModeValidation:
    """Test deployment mode validation logic."""

    def test_default_mode_valid(self):
        """Default modes should be valid."""
        deployment, registry, corrected = _validate_mode_combination(
            DeploymentMode.WITH_GATEWAY, RegistryMode.FULL
        )
        assert deployment == DeploymentMode.WITH_GATEWAY
        assert registry == RegistryMode.FULL
        assert corrected is False

    def test_gateway_skills_only_invalid(self):
        """Gateway + skills-only should auto-correct to registry-only."""
        deployment, registry, corrected = _validate_mode_combination(
            DeploymentMode.WITH_GATEWAY, RegistryMode.SKILLS_ONLY
        )
        assert deployment == DeploymentMode.REGISTRY_ONLY
        assert registry == RegistryMode.SKILLS_ONLY
        assert corrected is True

    def test_registry_only_full_valid(self):
        """Registry-only + full should be valid."""
        deployment, registry, corrected = _validate_mode_combination(
            DeploymentMode.REGISTRY_ONLY, RegistryMode.FULL
        )
        assert deployment == DeploymentMode.REGISTRY_ONLY
        assert registry == RegistryMode.FULL
        assert corrected is False

    def test_registry_only_skills_valid(self):
        """Registry-only + skills-only should be valid."""
        deployment, registry, corrected = _validate_mode_combination(
            DeploymentMode.REGISTRY_ONLY, RegistryMode.SKILLS_ONLY
        )
        assert deployment == DeploymentMode.REGISTRY_ONLY
        assert registry == RegistryMode.SKILLS_ONLY
        assert corrected is False

    def test_gateway_mcp_servers_only_valid(self):
        """Gateway + mcp-servers-only should be valid."""
        deployment, registry, corrected = _validate_mode_combination(
            DeploymentMode.WITH_GATEWAY, RegistryMode.MCP_SERVERS_ONLY
        )
        assert deployment == DeploymentMode.WITH_GATEWAY
        assert registry == RegistryMode.MCP_SERVERS_ONLY
        assert corrected is False


# =============================================================================
# TEST CLASS: Nginx Updates Enabled
# =============================================================================


@pytest.mark.unit
class TestNginxUpdatesEnabled:
    """Test nginx_updates_enabled property."""

    def test_enabled_with_gateway(self):
        """Should be enabled in with-gateway mode."""
        settings = Settings(deployment_mode=DeploymentMode.WITH_GATEWAY)
        assert settings.nginx_updates_enabled is True

    def test_disabled_registry_only(self):
        """Should be disabled in registry-only mode."""
        settings = Settings(deployment_mode=DeploymentMode.REGISTRY_ONLY)
        assert settings.nginx_updates_enabled is False


from unittest.mock import MagicMock, patch


# =============================================================================
# TEST CLASS: Nginx Service Deployment Mode
# =============================================================================


@pytest.mark.unit
class TestNginxServiceDeploymentMode:
    """Test nginx service respects deployment mode."""

    @patch("registry.core.nginx_service.NGINX_UPDATES_SKIPPED")
    @patch("registry.core.nginx_service.settings")
    @patch("registry.core.nginx_service.Path")
    def test_generate_config_skipped_in_registry_only(
        self,
        mock_path_class,
        mock_settings,
        mock_counter,
    ):
        """Nginx config generation should be skipped in registry-only mode."""
        mock_settings.nginx_updates_enabled = False
        mock_settings.deployment_mode = MagicMock()
        mock_settings.deployment_mode.value = "registry-only"

        # Mock Path for constructor SSL checks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        from registry.core.nginx_service import NginxConfigService

        service = NginxConfigService()

        result = service.generate_config({})

        assert result is True
        mock_counter.labels.assert_called_with(operation="generate_config")
        mock_counter.labels().inc.assert_called_once()

    @patch("registry.core.nginx_service.NGINX_UPDATES_SKIPPED")
    @patch("registry.core.nginx_service.settings")
    @patch("registry.core.nginx_service.Path")
    def test_reload_nginx_skipped_in_registry_only(
        self,
        mock_path_class,
        mock_settings,
        mock_counter,
    ):
        """Nginx reload should be skipped in registry-only mode."""
        mock_settings.nginx_updates_enabled = False
        mock_settings.deployment_mode = MagicMock()
        mock_settings.deployment_mode.value = "registry-only"

        # Mock Path for constructor SSL checks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        from registry.core.nginx_service import NginxConfigService

        service = NginxConfigService()

        result = service.reload_nginx()

        assert result is True
        mock_counter.labels.assert_called_with(operation="reload")
        mock_counter.labels().inc.assert_called_once()
