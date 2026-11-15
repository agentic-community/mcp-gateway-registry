"""Ownership of server records that point at the openapi-mcp container (issue #2022).

A managed record must keep pointing at its own hub, and no other record may
point at the container. Without the first rule, anyone allowed to edit the
record could repoint it and the egress vend would send every caller's
per-user keys to the new host.
"""

from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from fastapi import HTTPException

from registry.api import server_routes
from registry.services import openapi_ownership
from registry.services.server_service import ServerService
from registry.utils import url_guard
from registry.utils.url_guard import UrlValidationError

CONTAINER: str = "http://openapi-mcp:9876"
HUB_URL: str = f"{CONTAINER}/s/weather-apis/mcp"


def _settings(
    enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        openapi_mcp_enabled=enabled,
        openapi_mcp_internal_url=CONTAINER,
        ssrf_allowed_hosts="",
        ssrf_allowed_cidrs="",
        gateway_proxy_allow_private_targets=False,
        github_extra_hosts="",
        egress_oauth_trusted_idp_hosts="",
    )


def _managed(
    url: str = HUB_URL,
    path: str = "/weather-apis",
) -> dict:
    return {
        "path": path,
        "proxy_pass_url": url,
        "metadata": {"openapi_server": {"slug": "weather-apis"}},
    }


@pytest.fixture(autouse=True)
def guard_settings():
    url_guard._proxy_allowlist.cache_clear()
    with patch.object(url_guard, "settings", _settings()):
        yield
    url_guard._proxy_allowlist.cache_clear()


class TestManagedSlug:
    @pytest.mark.parametrize(
        "record",
        [
            None,
            {},
            {"metadata": None},
            {"metadata": {"openapi_server": "x"}},
            {"metadata": {"openapi_server": {"slug": 5}}},
        ],
    )
    def test_malformed_markers_are_not_managed(self, record):
        assert openapi_ownership.managed_openapi_slug(record) is None

    def test_slug_is_read_from_the_marker(self):
        assert openapi_ownership.managed_openapi_slug(_managed()) == "weather-apis"


class TestOwnershipRule:
    def test_managed_record_on_its_hub_passes(self):
        openapi_ownership.check_openapi_ownership(_managed())

    @pytest.mark.parametrize(
        "url",
        [
            "https://attacker.example/mcp",
            f"{CONTAINER}/s/other-server/mcp",
            f"{CONTAINER}/internal/status",
        ],
    )
    def test_repointed_managed_record_is_refused(self, url):
        with pytest.raises(UrlValidationError, match="must keep its openapi-mcp target"):
            openapi_ownership.check_openapi_ownership(_managed(url=url))

    def test_managed_record_at_another_path_is_refused(self):
        with pytest.raises(UrlValidationError):
            openapi_ownership.check_openapi_ownership(_managed(path="/someone-else"))

    def test_ordinary_record_on_the_container_is_refused(self):
        with pytest.raises(UrlValidationError, match="serves only OpenAPI MCP servers"):
            openapi_ownership.check_openapi_ownership(
                {"path": "/weather-apis", "proxy_pass_url": HUB_URL}
            )

    def test_ordinary_record_elsewhere_passes(self):
        openapi_ownership.check_openapi_ownership(
            {"path": "/x", "proxy_pass_url": "https://api.example.com/mcp"}
        )

    def test_feature_off_keeps_leftover_records_but_refuses_a_repoint(self):
        with patch.object(url_guard, "settings", _settings(enabled=False)):
            openapi_ownership.check_openapi_ownership(_managed())
            with pytest.raises(UrlValidationError):
                openapi_ownership.check_openapi_ownership(
                    _managed(url="https://attacker.example/mcp")
                )


class TestServerServiceWrites:
    """The rule runs in server_service, so every route that writes a target is covered."""

    @pytest.fixture
    def service(self):
        svc = ServerService()
        svc._repo = SimpleNamespace(get=AsyncMock(return_value=_managed()))
        return svc

    async def test_update_that_repoints_a_managed_record_is_refused(self, service):
        with (
            patch("registry.services.server_service.validate_proxy_pass_url"),
            pytest.raises(UrlValidationError),
        ):
            await service.update_server(
                "/weather-apis", {"proxy_pass_url": "https://attacker.example/mcp"}
            )

    async def test_update_that_drops_the_marker_is_refused(self, service):
        with pytest.raises(UrlValidationError):
            await service.update_server("/weather-apis", {"metadata": {}})

    async def test_register_of_an_ordinary_record_on_the_container_is_refused(self, service):
        with (
            patch("registry.services.server_service.validate_proxy_pass_url"),
            pytest.raises(UrlValidationError),
        ):
            await service.register_server({"path": "/weather-apis", "proxy_pass_url": HUB_URL})


class TestGenericRoutes:
    def test_managed_record_gets_a_409_pointing_to_the_openapi_routes(self):
        with pytest.raises(HTTPException) as info:
            server_routes._refuse_openapi_managed(_managed(), "remove")

        assert info.value.status_code == 409
        assert "/api/openapi-servers/weather-apis" in info.value.detail

    @pytest.mark.parametrize(
        "record", [None, {"path": "/x", "proxy_pass_url": "https://a.example"}]
    )
    def test_ordinary_or_missing_record_passes(self, record):
        server_routes._refuse_openapi_managed(record, "remove")
