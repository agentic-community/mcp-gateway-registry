"""Tests for the OpenAPI MCP built-in identity in the SSRF guard (issue #2022)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from registry.exceptions import UrlValidationError
from registry.utils import url_guard

CONTAINER: str = "http://openapi-mcp:9876"


def _settings(
    enabled: bool = True,
    internal_url: str = CONTAINER,
    ssrf_allowed_hosts: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        openapi_mcp_enabled=enabled,
        openapi_mcp_internal_url=internal_url,
        ssrf_allowed_hosts=ssrf_allowed_hosts,
        ssrf_allowed_cidrs="",
        gateway_proxy_allow_private_targets=False,
        github_extra_hosts="",
        egress_oauth_trusted_idp_hosts="",
    )


def _resolve_to(
    ip: str,
):
    def _stub(host, port, **kw):
        return [(2, 1, 6, "", (ip, port))]

    return _stub


@pytest.fixture(autouse=True)
def _clear_caches():
    url_guard._proxy_allowlist.cache_clear()
    yield
    url_guard._proxy_allowlist.cache_clear()


class TestIdentitySelection:
    """Which records get the built-in profile."""

    def test_exact_identity_selects_builtin_profile(self):
        with patch.object(url_guard, "settings", _settings()):
            profile = url_guard.proxy_profile_for_entity_target(
                "mcp_server", "/weather-apis/", f"{CONTAINER}/s/weather-apis/mcp"
            )

        assert profile is url_guard.BUILTIN_OPENAPI_MCP_PROFILE

    def test_trailing_slash_outbound_allowed(self):
        with patch.object(url_guard, "settings", _settings()):
            profile = url_guard.proxy_profile_for_entity_target(
                "mcp_server",
                "/weather-apis",
                f"{CONTAINER}/s/weather-apis/mcp",
                f"{CONTAINER}/s/weather-apis/mcp/",
            )

        assert profile is url_guard.BUILTIN_OPENAPI_MCP_PROFILE

    def test_identity_rejects_other_slug_target(self):
        with patch.object(url_guard, "settings", _settings()):
            profile = url_guard.proxy_profile_for_entity_target(
                "mcp_server", "/a/", f"{CONTAINER}/s/b/mcp"
            )

        assert profile is url_guard.PROXY_PROFILE

    @pytest.mark.parametrize(
        "outbound",
        [f"{CONTAINER}/internal/status", f"{CONTAINER}/health", f"{CONTAINER}/s/other/mcp"],
    )
    def test_other_outbound_url_raises(self, outbound):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError, match="OpenAPI MCP server identity"):
                url_guard.proxy_profile_for_entity_target(
                    "mcp_server", "/weather-apis/", f"{CONTAINER}/s/weather-apis/mcp", outbound
                )

    @pytest.mark.parametrize("entity_type", ["a2a_agent", "skill", "workflow"])
    def test_other_entity_types_never_match(self, entity_type):
        with patch.object(url_guard, "settings", _settings()):
            assert not url_guard.is_builtin_openapi_mcp_target(
                entity_type, "/weather-apis/", f"{CONTAINER}/s/weather-apis/mcp"
            )

    def test_feature_off_never_matches(self):
        with patch.object(url_guard, "settings", _settings(enabled=False)):
            assert not url_guard.is_builtin_openapi_mcp_target(
                "mcp_server", "/weather-apis/", f"{CONTAINER}/s/weather-apis/mcp"
            )

    def test_mocked_settings_fail_closed(self):
        from unittest.mock import MagicMock

        with patch.object(url_guard, "settings", MagicMock()):
            assert not url_guard.is_builtin_openapi_mcp_target(
                "mcp_server", "/weather-apis/", f"{CONTAINER}/s/weather-apis/mcp"
            )

    def test_invalid_slug_never_matches(self):
        with patch.object(url_guard, "settings", _settings()):
            assert not url_guard.is_builtin_openapi_mcp_target(
                "mcp_server", "/Weather_APIs/", f"{CONTAINER}/s/Weather_APIs/mcp"
            )

    def test_target_url_helper(self):
        with patch.object(url_guard, "settings", _settings()):
            assert (
                url_guard.openapi_mcp_target_url("weather-apis")
                == f"{CONTAINER}/s/weather-apis/mcp"
            )


class TestProfileReach:
    """What a request through each profile may reach."""

    def test_builtin_profile_reaches_server_path(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("172.18.0.9")):
                ips = url_guard.validate_url(
                    f"{CONTAINER}/s/weather-apis/mcp", profile=url_guard.BUILTIN_OPENAPI_MCP_PROFILE
                )

        assert ips == ["172.18.0.9"]

    @pytest.mark.parametrize(
        "path", ["/internal/status", "/internal/servers/x", "/health", "/s/x/other"]
    )
    def test_builtin_profile_never_reaches_control_api(self, path):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("172.18.0.9")):
                with pytest.raises(UrlValidationError, match="identity pattern"):
                    url_guard.validate_url(
                        f"{CONTAINER}{path}", profile=url_guard.BUILTIN_OPENAPI_MCP_PROFILE
                    )

    def test_builtin_profile_denies_everything_when_feature_off(self):
        with patch.object(url_guard, "settings", _settings(enabled=False)):
            with pytest.raises(UrlValidationError, match="identity pattern"):
                url_guard.validate_url(
                    f"{CONTAINER}/s/weather-apis/mcp", profile=url_guard.BUILTIN_OPENAPI_MCP_PROFILE
                )

    def test_reserved_host_not_admitted_by_allowlist(self):
        with patch.object(url_guard, "settings", _settings(ssrf_allowed_hosts="openapi-mcp")):
            with pytest.raises(UrlValidationError, match="reserved for OpenAPI MCP"):
                url_guard.validate_url(f"{CONTAINER}/s/x/mcp", profile=url_guard.PROXY_PROFILE)

    def test_reserved_host_stays_reserved_when_feature_off(self):
        with patch.object(
            url_guard, "settings", _settings(enabled=False, ssrf_allowed_hosts="openapi-mcp")
        ):
            with pytest.raises(UrlValidationError, match="reserved for OpenAPI MCP"):
                url_guard.validate_url(f"{CONTAINER}/s/x/mcp", profile=url_guard.PROXY_PROFILE)

    def test_airegistry_profile_cannot_reach_container(self):
        with patch.object(url_guard, "settings", _settings()):
            with pytest.raises(UrlValidationError):
                url_guard.validate_url(
                    f"{CONTAINER}/s/x/mcp", profile=url_guard.BUILTIN_AIREGISTRY_TOOLS_PROFILE
                )

    def test_builtin_profile_still_blocks_metadata(self):
        with patch.object(url_guard, "settings", _settings()):
            with patch.object(url_guard.socket, "getaddrinfo", _resolve_to("169.254.169.254")):
                with pytest.raises(UrlValidationError):
                    url_guard.validate_url(
                        f"{CONTAINER}/s/weather-apis/mcp",
                        profile=url_guard.BUILTIN_OPENAPI_MCP_PROFILE,
                    )
