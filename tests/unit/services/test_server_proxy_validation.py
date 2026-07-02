"""Unit tests for server-service proxy resolve-and-validate.

Locks in the SSRF layer-2 wiring on the server paths — including update_server
(the single choke point for every server PATCH / version-swap call site), which
a prior revision missed. A hostname proxy target that resolves to a metadata IP
must be rejected on BOTH register and update, not just register.
"""

import socket
from unittest.mock import AsyncMock, patch

import pytest

from registry.schemas.proxy_mixin import EgressPolicyError
from registry.services.server_service import ServerService

pytestmark = pytest.mark.unit


def _addrinfo(*ips):
    out = []
    for ip in ips:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443)))
    return out


def _service_with_repo(repo):
    with (
        patch("registry.services.server_service.get_server_repository", return_value=repo),
        patch("registry.repositories.factory.get_search_repository", return_value=AsyncMock()),
    ):
        return ServerService()


def _settings():
    m = patch("registry.core.config.settings")
    s = m.start()
    s.gateway_proxy_allow_private_targets = False
    return m


class TestRegisterServerProxy:
    async def test_register_rejects_hostname_resolving_to_metadata(self):
        repo = AsyncMock()
        repo.get.return_value = None  # no existing server
        service = _service_with_repo(repo)
        info = {
            "path": "/s",
            "server_name": "s",
            "is_proxied": True,
            "proxy_pass_url": "https://rebind.example/",
            "deployment": "remote",
        }
        m = _settings()
        try:
            with patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
                with pytest.raises(EgressPolicyError):
                    await service.register_server(info)
        finally:
            m.stop()
        repo.create.assert_not_called()  # nothing persisted

    async def test_register_pins_public_target(self):
        repo = AsyncMock()
        repo.get.return_value = None
        repo.create.return_value = True
        service = _service_with_repo(repo)
        info = {
            "path": "/s",
            "server_name": "s",
            "is_proxied": True,
            "proxy_pass_url": "https://ok.example/",
            "deployment": "remote",
        }
        m = _settings()
        try:
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
                await service.register_server(info)
        finally:
            m.stop()
        assert info["proxy_resolved_ips"] == ["93.184.216.34"]
        assert info["proxy_target_host"] == "ok.example"


class TestUpdateServerProxy:
    async def test_update_rejects_hostname_resolving_to_metadata(self):
        # Regression guard: update_server previously skipped DNS validation.
        repo = AsyncMock()
        repo.get.return_value = {
            "path": "/s",
            "server_name": "s",
            "is_proxied": False,
            "proxy_pass_url": "https://old.example/",
        }
        service = _service_with_repo(repo)
        m = _settings()
        try:
            with patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
                with pytest.raises(EgressPolicyError):
                    await service.update_server(
                        "/s",
                        {"is_proxied": True, "proxy_target_url": "https://rebind.example/"},
                    )
        finally:
            m.stop()
        repo.update.assert_not_called()  # denied before persist

    async def test_update_without_proxy_fields_skips_dns(self):
        repo = AsyncMock()
        repo.update.return_value = True
        repo.get_state.return_value = False
        service = _service_with_repo(repo)
        with patch("socket.getaddrinfo") as gai:
            await service.update_server("/s", {"description": "new desc"})
            gai.assert_not_called()  # no proxy field touched -> no DNS, no extra get

    async def test_update_reenable_clears_disabled_reason_and_pins(self):
        repo = AsyncMock()
        repo.get.return_value = {
            "path": "/s",
            "is_proxied": True,
            "proxy_pass_url": "https://ok.example/",
            "proxy_disabled_reason": "was auto-disabled",
            "deployment": "remote",
        }
        repo.update.return_value = True
        repo.get_state.return_value = False
        service = _service_with_repo(repo)
        m = _settings()
        try:
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
                await service.update_server("/s", {"is_proxied": True})
        finally:
            m.stop()
        # The persisted update dict must clear the auto-disable and carry fresh pins.
        persisted = repo.update.call_args[0][1]
        assert persisted["proxy_disabled_reason"] is None
        assert persisted["proxy_resolved_ips"] == ["93.184.216.34"]
