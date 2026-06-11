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


# =============================================================================
# TEST CLASS: Effective UI Title
# =============================================================================


@pytest.mark.unit
class TestEffectiveUiTitle:
    """Test effective_ui_title property — UI_TITLE override and mode-aware default."""

    def test_unset_with_gateway_default(self):
        """Unset UI_TITLE + with-gateway -> 'AI Gateway & Registry'."""
        settings = Settings(deployment_mode=DeploymentMode.WITH_GATEWAY, ui_title=None)
        assert settings.effective_ui_title == "AI Gateway & Registry"

    def test_unset_registry_only_default(self):
        """Unset UI_TITLE + registry-only -> 'AI Registry'."""
        settings = Settings(deployment_mode=DeploymentMode.REGISTRY_ONLY, ui_title=None)
        assert settings.effective_ui_title == "AI Registry"

    def test_override_with_gateway(self):
        """Set UI_TITLE wins over with-gateway default."""
        settings = Settings(
            deployment_mode=DeploymentMode.WITH_GATEWAY, ui_title="Acme Portal"
        )
        assert settings.effective_ui_title == "Acme Portal"

    def test_override_registry_only(self):
        """Set UI_TITLE wins over registry-only default."""
        settings = Settings(
            deployment_mode=DeploymentMode.REGISTRY_ONLY, ui_title="Contoso Agent Registry"
        )
        assert settings.effective_ui_title == "Contoso Agent Registry"

    def test_empty_string_treated_as_unset(self):
        """Empty UI_TITLE falls back to deployment-mode default."""
        settings = Settings(deployment_mode=DeploymentMode.WITH_GATEWAY, ui_title="")
        assert settings.effective_ui_title == "AI Gateway & Registry"

    def test_whitespace_only_treated_as_unset(self):
        """Whitespace-only UI_TITLE falls back to deployment-mode default."""
        settings = Settings(deployment_mode=DeploymentMode.REGISTRY_ONLY, ui_title="   ")
        assert settings.effective_ui_title == "AI Registry"


# =============================================================================
# TEST CLASS: /api/version exposes ui_title
# =============================================================================


@pytest.mark.unit
class TestVersionEndpointUiTitle:
    """End-to-end check that /api/version surfaces effective_ui_title.

    /api/version is unauthenticated by design (whitelisted in nginx and at
    FastAPI), so Login/Logout can render the operator-configured title before
    the user has a session. Guards against regressions that move ui_title back
    behind auth.
    """

    def _patched_client(
        self,
        monkeypatch,
        deployment_mode: DeploymentMode,
        ui_title: str | None,
    ):
        from fastapi.testclient import TestClient

        from registry.api import system_routes
        from registry.core import config as config_module
        from registry.main import app

        settings = Settings(deployment_mode=deployment_mode, ui_title=ui_title)
        # system_routes.py imports `settings` directly at module load, so we
        # patch both the source-of-truth and the route module's binding.
        monkeypatch.setattr(config_module, "settings", settings)
        monkeypatch.setattr(system_routes, "settings", settings)
        return TestClient(app)

    def test_registry_only_default_returns_ai_registry(self, monkeypatch):
        """DEPLOYMENT_MODE=registry-only, UI_TITLE unset -> 'AI Registry'."""
        client = self._patched_client(monkeypatch, DeploymentMode.REGISTRY_ONLY, None)
        response = client.get("/api/version")
        assert response.status_code == 200
        assert response.json()["ui_title"] == "AI Registry"

    def test_with_gateway_default_returns_full_title(self, monkeypatch):
        """DEPLOYMENT_MODE=with-gateway, UI_TITLE unset -> 'AI Gateway & Registry'."""
        client = self._patched_client(monkeypatch, DeploymentMode.WITH_GATEWAY, None)
        response = client.get("/api/version")
        assert response.status_code == 200
        assert response.json()["ui_title"] == "AI Gateway & Registry"

    def test_override_wins_over_mode_default(self, monkeypatch):
        """UI_TITLE='Acme Portal' is returned regardless of deployment_mode."""
        client = self._patched_client(
            monkeypatch, DeploymentMode.REGISTRY_ONLY, "Acme Portal"
        )
        response = client.get("/api/version")
        assert response.status_code == 200
        assert response.json()["ui_title"] == "Acme Portal"

    def test_endpoint_is_unauthenticated(self):
        """/api/version must be reachable without an Authorization header.

        Critical invariant: Login is unauthenticated, so its
        source for ui_title must also be unauthenticated.
        """
        from fastapi.testclient import TestClient

        from registry.main import app

        client = TestClient(app)
        response = client.get("/api/version")  # no auth header
        assert response.status_code == 200
        assert "ui_title" in response.json()


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


# =============================================================================
# TEST CLASS: Internal Deployment Classification (issue #1216)
# =============================================================================


@pytest.mark.unit
class TestInternalDeploymentClassification:
    """Test the startup correction logic for internal/workshop deployments.

    _resolve_internal_deployment_classification mutates the module-level
    settings via object.__setattr__, mirroring the deployment-mode correction.
    """

    def _run_with(self, internal_only, initial_type):
        """Apply the correction to a patched settings and return the result type."""
        from unittest.mock import patch

        from registry.core.config import InternalDeploymentType, Settings

        settings = Settings()
        object.__setattr__(settings, "internal_only_deployment", internal_only)
        object.__setattr__(settings, "internal_deployment_type", initial_type)

        with patch("registry.main.settings", settings):
            from registry.main import _resolve_internal_deployment_classification

            _resolve_internal_deployment_classification()

        return settings.internal_deployment_type, InternalDeploymentType

    def test_not_internal_keeps_none(self):
        """internal_only=False with NONE type stays NONE (the common default case)."""
        from registry.core.config import InternalDeploymentType

        result, Enum = self._run_with(False, InternalDeploymentType.NONE)
        assert result == Enum.NONE

    def test_not_internal_with_workshop_corrected_to_none(self):
        """A non-none type with internal_only=False is corrected to NONE."""
        from registry.core.config import InternalDeploymentType

        result, Enum = self._run_with(False, InternalDeploymentType.WORKSHOP)
        assert result == Enum.NONE

    def test_internal_unset_defaults_to_dev(self):
        """internal_only=True with NONE type defaults to DEV."""
        from registry.core.config import InternalDeploymentType

        result, Enum = self._run_with(True, InternalDeploymentType.NONE)
        assert result == Enum.DEV

    def test_internal_workshop_preserved(self):
        """internal_only=True with an explicit type keeps that type."""
        from registry.core.config import InternalDeploymentType

        result, Enum = self._run_with(True, InternalDeploymentType.WORKSHOP)
        assert result == Enum.WORKSHOP
