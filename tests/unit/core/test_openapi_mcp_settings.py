"""Tests for OpenAPI MCP settings validation and the deployment-mode gate (issue #2022)."""

import pytest

from registry.core.config import (
    DeploymentMode,
    RegistryMode,
    Settings,
    apply_openapi_mcp_mode_gate,
)

GOOD_SECRET: str = "openapi-sync-0123456789abcdefghijklmnopqrstuv"


def _settings(
    **kwargs,
) -> Settings:
    return Settings(openapi_mcp_enabled=True, openapi_mcp_sync_secret=GOOD_SECRET, **kwargs)


class TestValidation:
    def test_valid_config_strips_trailing_slash(self):
        assert (
            _settings(openapi_mcp_internal_url="http://openapi-mcp:9876/").openapi_mcp_internal_url
            == "http://openapi-mcp:9876"
        )

    def test_feature_off_needs_no_secret(self):
        assert Settings(openapi_mcp_enabled=False).openapi_mcp_sync_secret == ""

    @pytest.mark.parametrize(
        "secret",
        [
            "",
            "short",
            "change-this-immediately-use-a-strong-random-key-in-production",
            "YOUR_OPENAPI_MCP_SYNC_SECRET",
        ],
    )
    def test_missing_or_weak_secret_rejected(self, secret):
        with pytest.raises((RuntimeError, ValueError)):
            Settings(openapi_mcp_enabled=True, openapi_mcp_sync_secret=secret)

    def test_secret_equal_to_secret_key_rejected(self):
        base = Settings()

        with pytest.raises(ValueError, match="differ from SECRET_KEY"):
            Settings(openapi_mcp_enabled=True, openapi_mcp_sync_secret=base.secret_key)

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://openapi-mcp:9876",
            "http://openapi-mcp:9876/s",
            "http://openapi-mcp:9876?x=1",
            "http://user:pw@openapi-mcp:9876",
            "http://10.0.0.5:9876",
            "openapi-mcp:9876",
        ],
    )
    def test_bad_internal_url_rejected(self, url):
        with pytest.raises(ValueError):
            _settings(openapi_mcp_internal_url=url)


class TestModeGate:
    def test_full_with_gateway_stays_on(self):
        s = _settings()

        assert apply_openapi_mcp_mode_gate(s) is True

    def test_registry_only_turns_off(self):
        s = _settings(deployment_mode=DeploymentMode.REGISTRY_ONLY)

        assert apply_openapi_mcp_mode_gate(s) is False
        assert s.openapi_mcp_enabled is False

    @pytest.mark.parametrize("mode", [RegistryMode.SKILLS_ONLY, RegistryMode.AGENTS_ONLY])
    def test_mode_without_mcp_servers_turns_off(self, mode):
        s = _settings(registry_mode=mode)

        assert apply_openapi_mcp_mode_gate(s) is False

    def test_mcp_servers_only_stays_on(self):
        assert (
            apply_openapi_mcp_mode_gate(_settings(registry_mode=RegistryMode.MCP_SERVERS_ONLY))
            is True
        )
