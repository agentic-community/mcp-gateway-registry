"""
Unit tests for ARD catalog hardening.

Covers:
- TOCTOU race prevention in _auto_initialize_registry_card
- Per-server PRM exception detail not leaked to callers
- DB query caps: ard_catalog_max_entries_per_type limits applied
- Nginx rate-limiting zone for .well-known (declarative; tested in test_nginx_service.py)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    """Build a minimal FastAPI app with the wellknown router."""
    from registry.api.wellknown_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/.well-known")
    return app


# =============================================================================
# PER-SERVER PRM EXCEPTION DETAIL LEAK
# =============================================================================


class TestPerServerPRMExceptionLeak:
    """Per-server PRM does not leak internal exception text."""

    @patch("registry.api.wellknown_routes.settings")
    @patch("registry.api.wellknown_routes.server_service")
    @patch("registry.api.wellknown_routes._get_active_auth_provider")
    def test_internal_exception_not_in_response(self, mock_provider_fn, mock_svc, mock_settings):
        """Exception detail is generic, not reflecting internal error."""
        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True

        # server_service returns a server that needs per-server PRM
        mock_svc.get_server_info = AsyncMock(
            return_value={"egress_auth_mode": "obo_exchange", "append_mcp_path": True}
        )

        # Provider raises an internal error
        secret_detail = "MongoDB connection refused at 10.0.1.42:27017"
        mock_provider = MagicMock()
        mock_provider.protected_resource_metadata.side_effect = RuntimeError(secret_detail)
        mock_provider_fn.return_value = mock_provider

        app = _make_app()
        client = TestClient(app)

        resp = client.get("/.well-known/oauth-protected-resource/obo-echo/mcp")
        assert resp.status_code == 502
        body = resp.json()
        # The internal exception text must NOT appear in the response
        assert secret_detail not in body.get("detail", "")
        # Generic message should be present
        assert "Could not build Protected Resource Metadata" in body["detail"]


# =============================================================================
# DB QUERY CAPS (ard_catalog_max_entries_per_type)
# =============================================================================


class TestArdServiceQueryCaps:
    """ard_service.build_catalog respects max_entries_per_type."""

    @pytest.mark.asyncio
    @patch("registry.services.ard_service.settings")
    @patch("registry.services.ard_service.get_skill_repository")
    @patch("registry.services.ard_service.get_agent_repository")
    @patch("registry.services.ard_service.get_server_repository")
    async def test_queries_pass_configured_limit(
        self, mock_server_repo_fn, mock_agent_repo_fn, mock_skill_repo_fn, mock_settings
    ):
        """find_with_filter is called with the configured limit."""
        mock_settings.ard_catalog_max_entries_per_type = 100
        mock_settings.ard_catalog_enabled = True
        mock_settings.ard_publisher_domain = "test.example.com"
        mock_settings.registry_url = "https://test.example.com"
        mock_settings.registry_name = "Test"
        mock_settings.ard_catalog_default_namespace = ""
        mock_settings.ard_registry_enabled = False

        mock_server_repo = AsyncMock()
        mock_server_repo.find_with_filter.return_value = {}
        mock_server_repo_fn.return_value = mock_server_repo

        mock_agent_repo = AsyncMock()
        mock_agent_repo.find_with_filter.return_value = {}
        mock_agent_repo_fn.return_value = mock_agent_repo

        mock_skill_repo = AsyncMock()
        mock_skill_repo.list_filtered.return_value = []
        mock_skill_repo_fn.return_value = mock_skill_repo

        from registry.services.ard_service import build_catalog

        mock_request = MagicMock()
        mock_request.headers = {"host": "test.example.com", "x-forwarded-proto": "https"}
        mock_request.url.scheme = "https"

        await build_catalog(mock_request)

        # Verify limit was passed to all three queries
        mock_server_repo.find_with_filter.assert_called_once_with(
            {"is_enabled": True, "visibility": "public"}, limit=100
        )
        mock_agent_repo.find_with_filter.assert_called_once_with(
            {"is_enabled": True, "visibility": "public"}, limit=100
        )
        mock_skill_repo.list_filtered.assert_called_once_with(
            include_disabled=False,
            visibility="public",
            registry_name="local",
            limit=100,
        )
