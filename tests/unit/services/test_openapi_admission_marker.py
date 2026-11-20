"""The OpenAPI MCP admission marker on registry-side calls (issue #2022, security review #2).

The openapi-mcp container refuses a hub call that does not carry the marker
for that slug. Only the server's own gateway locations and the registry's own
calls to that hub may add it; a record that reaches the container by IP or
another name must never get one.
"""

from types import SimpleNamespace
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from registry.auth.openapi_mcp_sync import (
    ADMISSION_HEADER,
    admission_marker,
)
from registry.core import nginx_service
from registry.core.mcp_client import _build_headers_for_server
from registry.health.service import HealthMonitoringService
from registry.services import (
    openapi_ownership,
    security_scanner,
)
from registry.utils import url_guard

CONTAINER: str = "http://openapi-mcp:9876"
HUB_URL: str = f"{CONTAINER}/s/weather/mcp"
SYNC_SECRET: str = "admission-test-sync-secret-0123456789abcdef"  # pragma: allowlist secret
MARKER: str = admission_marker(SYNC_SECRET, "weather")


def _guard_settings() -> SimpleNamespace:
    return SimpleNamespace(
        openapi_mcp_enabled=True,
        openapi_mcp_internal_url=CONTAINER,
        ssrf_allowed_hosts="",
        ssrf_allowed_cidrs="10.0.0.0/8",
        gateway_proxy_allow_private_targets=False,
        github_extra_hosts="",
        egress_oauth_trusted_idp_hosts="",
    )


@pytest.fixture(autouse=True)
def openapi_on():
    url_guard._proxy_allowlist.cache_clear()
    with (
        patch.object(url_guard, "settings", _guard_settings()),
        patch.object(openapi_ownership.settings, "openapi_mcp_sync_secret", SYNC_SECRET),
    ):
        yield
    url_guard._proxy_allowlist.cache_clear()


def _managed(
    url: str = HUB_URL,
    path: str = "/weather",
) -> dict:
    return {
        "path": path,
        "proxy_pass_url": url,
        "metadata": {"openapi_server": {"slug": "weather"}},
    }


class TestAdmissionHeaders:
    def test_own_hub_gets_its_slugs_marker(self):
        assert openapi_ownership.admission_headers("/weather", HUB_URL) == {
            ADMISSION_HEADER: MARKER
        }

    @pytest.mark.parametrize(
        ("path", "target", "destination"),
        [
            ("/weather", "http://10.0.0.7:9876/s/weather/mcp", None),
            ("/attacker", "http://10.0.0.7:9876/s/weather/mcp", None),
            ("/attacker", HUB_URL, None),
            ("/weather", HUB_URL, f"{CONTAINER}/s/other/mcp"),
            ("/weather", HUB_URL, "https://example.com/mcp"),
            ("/weather", None, None),
        ],
        ids=["ip-alias", "ordinary-path-ip", "wrong-path", "other-hub", "other-host", "no-target"],
    )
    def test_anything_but_the_exact_own_hub_gets_nothing(self, path, target, destination):
        assert openapi_ownership.admission_headers(path, target, destination) == {}

    def test_unset_secret_gives_no_marker(self):
        with patch.object(openapi_ownership.settings, "openapi_mcp_sync_secret", ""):
            assert openapi_ownership.admission_headers("/weather", HUB_URL) == {}


class TestNginxLocations:
    def test_owned_location_sets_the_marker(self):
        block = nginx_service.NginxConfigService()._create_location_block(
            "/weather", HUB_URL, "streamable-http", _managed()
        )

        assert f'proxy_set_header {ADMISSION_HEADER} "{MARKER}";' in block

    def test_ordinary_location_clears_a_client_copy(self):
        record = {"path": "/notes", "proxy_pass_url": "http://10.0.0.7:9876/s/weather/mcp"}

        block = nginx_service.NginxConfigService()._create_location_block(
            "/notes", record["proxy_pass_url"], "streamable-http", record
        )

        assert f'proxy_set_header {ADMISSION_HEADER} "";' in block
        assert MARKER not in block

    def test_virtual_backend_directive_follows_the_backing_server(self):
        owned = nginx_service._openapi_admission_directive("/weather", HUB_URL)
        alias = nginx_service._openapi_admission_directive(
            "/notes", "http://10.0.0.7:9876/s/weather/mcp"
        )

        assert (MARKER in owned, alias.endswith(f'{ADMISSION_HEADER} "";')) == (True, True)


class TestRegistryProbes:
    def test_tool_refresh_headers_carry_the_marker(self):
        # openapi-mcp does not resolve here; this test covers what happens after
        # the destination check passes.
        with patch.object(url_guard, "validate_url"):
            headers = _build_headers_for_server(_managed(), HUB_URL)

        assert headers[ADMISSION_HEADER] == MARKER

    def test_tool_refresh_to_an_ip_alias_carries_none(self):
        record = {"path": "/notes", "proxy_pass_url": "http://10.0.0.7:9876/s/weather/mcp"}

        headers = _build_headers_for_server(record, record["proxy_pass_url"])

        assert ADMISSION_HEADER not in headers

    def test_health_headers_carry_the_marker(self):
        with patch("registry.health.service.validate_url"):
            headers = HealthMonitoringService()._build_headers_for_server(_managed(), HUB_URL)

        assert headers[ADMISSION_HEADER] == MARKER

    def test_scanner_gets_the_marker_through_its_environment(self):
        completed = MagicMock(stdout="[]", stderr="", returncode=0)
        with patch.object(security_scanner.subprocess, "run", return_value=completed) as run:
            security_scanner.SecurityScannerService()._run_mcp_scanner(
                HUB_URL, "yara", timeout=5, admission=MARKER
            )

        cmd, env = run.call_args.args[0], run.call_args.kwargs["env"]
        assert (MARKER in " ".join(cmd), env[security_scanner._SCANNER_ADMISSION_ENV]) == (
            False,
            MARKER,
        )

    def test_scanner_shim_turns_the_variable_into_a_header_flag(self):
        assert f"'--header',{ADMISSION_HEADER + ': '!r}" in security_scanner._SCANNER_SHIM
