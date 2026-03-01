"""
Unit tests for registry mode filter middleware.

Tests the endpoint filtering logic based on REGISTRY_MODE setting.
"""

import pytest
from unittest.mock import MagicMock, patch

from registry.core.config import RegistryMode
from registry.middleware.mode_filter import (
    _is_path_allowed,
    _get_path_category,
    ALWAYS_ALLOWED_PREFIXES,
    SKILLS_MODE_ALLOWED_PREFIXES,
)


# =============================================================================
# TEST CLASS: Path Allowed Logic
# =============================================================================


@pytest.mark.unit
class TestPathAllowed:
    """Test _is_path_allowed function."""

    def test_health_always_allowed(self):
        """Health endpoint should always be allowed."""
        assert _is_path_allowed("/health", RegistryMode.FULL) is True
        assert _is_path_allowed("/health", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/health", RegistryMode.MCP_SERVERS_ONLY) is True
        assert _is_path_allowed("/health", RegistryMode.AGENTS_ONLY) is True

    def test_version_always_allowed(self):
        """Version endpoint should always be allowed."""
        assert _is_path_allowed("/api/version", RegistryMode.FULL) is True
        assert _is_path_allowed("/api/version", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/version", RegistryMode.MCP_SERVERS_ONLY) is True

    def test_config_always_allowed(self):
        """Config endpoint should always be allowed."""
        assert _is_path_allowed("/api/config", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/config/mode", RegistryMode.SKILLS_ONLY) is True

    def test_docs_always_allowed(self):
        """Documentation endpoints should always be allowed."""
        assert _is_path_allowed("/docs", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/openapi.json", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/redoc", RegistryMode.SKILLS_ONLY) is True

    def test_auth_always_allowed(self):
        """Auth endpoints should always be allowed."""
        assert _is_path_allowed("/api/auth/login", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/tokens/generate", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/oauth2/callback", RegistryMode.SKILLS_ONLY) is True

    def test_audit_always_allowed(self):
        """Audit endpoints should always be allowed (administrative functionality)."""
        assert _is_path_allowed("/api/audit/logs", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/audit/export", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/audit/logs", RegistryMode.MCP_SERVERS_ONLY) is True
        assert _is_path_allowed("/api/audit/logs", RegistryMode.AGENTS_ONLY) is True

    def test_management_always_allowed(self):
        """Management endpoints should always be allowed (administrative functionality)."""
        assert _is_path_allowed("/api/management/settings", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/management/", RegistryMode.MCP_SERVERS_ONLY) is True
        assert _is_path_allowed("/api/management/", RegistryMode.AGENTS_ONLY) is True

    def test_full_mode_allows_all(self):
        """Full mode should allow all endpoints."""
        assert _is_path_allowed("/api/servers", RegistryMode.FULL) is True
        assert _is_path_allowed("/api/agents", RegistryMode.FULL) is True
        assert _is_path_allowed("/api/skills", RegistryMode.FULL) is True
        assert _is_path_allowed("/api/federation", RegistryMode.FULL) is True
        assert _is_path_allowed("/api/peers", RegistryMode.FULL) is True

    def test_skills_only_allows_skills(self):
        """Skills-only mode should allow skills endpoints."""
        assert _is_path_allowed("/api/skills", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/skills/discovery", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/api/search/semantic", RegistryMode.SKILLS_ONLY) is True

    def test_skills_only_blocks_servers(self):
        """Skills-only mode should block servers endpoints."""
        assert _is_path_allowed("/api/servers", RegistryMode.SKILLS_ONLY) is False
        assert _is_path_allowed("/api/servers/test", RegistryMode.SKILLS_ONLY) is False

    def test_skills_only_blocks_agents(self):
        """Skills-only mode should block agents endpoints."""
        assert _is_path_allowed("/api/agents", RegistryMode.SKILLS_ONLY) is False
        assert _is_path_allowed("/api/agents/discover", RegistryMode.SKILLS_ONLY) is False

    def test_skills_only_blocks_federation(self):
        """Skills-only mode should block federation endpoints."""
        assert _is_path_allowed("/api/federation", RegistryMode.SKILLS_ONLY) is False
        assert _is_path_allowed("/api/peers", RegistryMode.SKILLS_ONLY) is False

    def test_skills_only_allows_wellknown(self):
        """Skills-only mode should allow well-known endpoints (returns empty list)."""
        assert _is_path_allowed("/.well-known/mcp-servers", RegistryMode.SKILLS_ONLY) is True

    def test_mcp_servers_only_allows_servers(self):
        """MCP-servers-only mode should allow servers endpoints."""
        assert _is_path_allowed("/api/servers", RegistryMode.MCP_SERVERS_ONLY) is True
        assert _is_path_allowed("/api/servers/test", RegistryMode.MCP_SERVERS_ONLY) is True
        assert _is_path_allowed("/api/search/semantic", RegistryMode.MCP_SERVERS_ONLY) is True

    def test_mcp_servers_only_blocks_agents(self):
        """MCP-servers-only mode should block agents endpoints."""
        assert _is_path_allowed("/api/agents", RegistryMode.MCP_SERVERS_ONLY) is False

    def test_mcp_servers_only_blocks_skills(self):
        """MCP-servers-only mode should block skills endpoints."""
        assert _is_path_allowed("/api/skills", RegistryMode.MCP_SERVERS_ONLY) is False

    def test_agents_only_allows_agents(self):
        """Agents-only mode should allow agents endpoints."""
        assert _is_path_allowed("/api/agents", RegistryMode.AGENTS_ONLY) is True
        assert _is_path_allowed("/api/agents/discover", RegistryMode.AGENTS_ONLY) is True
        assert _is_path_allowed("/api/search/semantic", RegistryMode.AGENTS_ONLY) is True

    def test_agents_only_blocks_servers(self):
        """Agents-only mode should block servers endpoints."""
        assert _is_path_allowed("/api/servers", RegistryMode.AGENTS_ONLY) is False

    def test_agents_only_blocks_skills(self):
        """Agents-only mode should block skills endpoints."""
        assert _is_path_allowed("/api/skills", RegistryMode.AGENTS_ONLY) is False

    def test_frontend_paths_allowed(self):
        """Frontend static paths should be allowed in all modes."""
        assert _is_path_allowed("/static/app.js", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/assets/logo.png", RegistryMode.SKILLS_ONLY) is True
        assert _is_path_allowed("/_next/static/chunks/main.js", RegistryMode.SKILLS_ONLY) is True


# =============================================================================
# TEST CLASS: Path Category Extraction
# =============================================================================


@pytest.mark.unit
class TestPathCategory:
    """Test _get_path_category function for metrics labeling."""

    def test_servers_category(self):
        """Should extract 'servers' category."""
        assert _get_path_category("/api/servers") == "servers"
        assert _get_path_category("/api/servers/test") == "servers"

    def test_agents_category(self):
        """Should extract 'agents' category."""
        assert _get_path_category("/api/agents") == "agents"
        assert _get_path_category("/api/agents/discover") == "agents"

    def test_skills_category(self):
        """Should extract 'skills' category."""
        assert _get_path_category("/api/skills") == "skills"
        assert _get_path_category("/api/skills/discovery") == "skills"

    def test_federation_category(self):
        """Should extract 'federation' category."""
        assert _get_path_category("/api/federation") == "federation"
        assert _get_path_category("/api/federation/sync") == "federation"
        assert _get_path_category("/api/peers") == "federation"

    def test_other_category(self):
        """Should return 'other' for unrecognized paths."""
        assert _get_path_category("/api/unknown") == "unknown"
        assert _get_path_category("/something/else") == "other"


# =============================================================================
# TEST CLASS: Middleware Integration
# =============================================================================


@pytest.mark.unit
class TestMiddlewareIntegration:
    """Test middleware behavior."""

    @pytest.mark.asyncio
    @patch("registry.middleware.mode_filter.settings")
    @patch("registry.middleware.mode_filter.MODE_BLOCKED_REQUESTS")
    async def test_middleware_blocks_disabled_endpoint(
        self,
        mock_metrics,
        mock_settings,
    ):
        """Middleware should return 403 for disabled endpoints."""
        mock_settings.registry_mode = RegistryMode.SKILLS_ONLY

        from registry.middleware.mode_filter import RegistryModeMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def api_servers(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/api/servers", api_servers)])
        app.add_middleware(RegistryModeMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/servers")

        assert response.status_code == 403
        data = response.json()
        assert data["error"] == "endpoint_disabled"
        assert "skills-only" in data["detail"]

    @pytest.mark.asyncio
    @patch("registry.middleware.mode_filter.settings")
    @patch("registry.middleware.mode_filter.MODE_BLOCKED_REQUESTS")
    async def test_middleware_allows_enabled_endpoint(
        self,
        mock_metrics,
        mock_settings,
    ):
        """Middleware should allow enabled endpoints."""
        mock_settings.registry_mode = RegistryMode.SKILLS_ONLY

        from registry.middleware.mode_filter import RegistryModeMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def api_skills(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/api/skills", api_skills)])
        app.add_middleware(RegistryModeMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/skills")

        assert response.status_code == 200
        assert response.text == "ok"
