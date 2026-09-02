"""Unit tests for the agent-service proxy validation hook.

_validate_and_pin_agent_proxy is the register/update choke point that:
- is a no-op when the agent is not proxied;
- rejects a GRPC preferred_transport (the gateway serves HTTP only);
- resolves + validates the effective target (proxy_target_url or the agent url)
  and pins the resolved IPs, rejecting a metadata/private target.
"""

import socket
from unittest.mock import patch

import pytest

from registry.schemas.agent_models import AgentCard
from registry.services.agent_service import _validate_and_pin_agent_proxy

pytestmark = pytest.mark.unit


def _addrinfo(*ips):
    out = []
    for ip in ips:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443)))
    return out


def _card(**kw):
    base = {"name": "a", "description": "d", "url": "https://agent.example/", "version": "1"}
    base.update(kw)
    return AgentCard(**base)


class TestAgentProxyValidation:
    async def test_no_op_when_not_proxied(self):
        card = _card(is_proxied=False)
        with patch("socket.getaddrinfo") as gai:
            await _validate_and_pin_agent_proxy(card)
            gai.assert_not_called()

    async def test_grpc_transport_rejected_when_proxied(self):
        card = _card(is_proxied=True, preferred_transport="GRPC")
        with pytest.raises(ValueError, match="GRPC"):
            await _validate_and_pin_agent_proxy(card)

    async def test_grpc_transport_allowed_when_not_proxied(self):
        # GRPC is fine as long as the agent isn't proxied through the HTTP gateway.
        card = _card(is_proxied=False, preferred_transport="GRPC")
        await _validate_and_pin_agent_proxy(card)  # no raise

    async def test_pins_resolved_ips_from_agent_url_fallback(self):
        # No explicit proxy_target_url -> falls back to the agent url.
        card = _card(is_proxied=True, preferred_transport="JSONRPC")
        with patch("registry.core.config.settings") as s:
            s.gateway_proxy_allow_private_targets = False
            with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
                await _validate_and_pin_agent_proxy(card)
        assert card.proxy_resolved_ips == ["93.184.216.34"]
        assert card.proxy_target_host == "agent.example"

    async def test_metadata_target_rejected(self):
        card = _card(is_proxied=True, proxy_target_url="https://rebind.example/")
        with patch("registry.core.config.settings") as s:
            s.gateway_proxy_allow_private_targets = False
            with patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
                with pytest.raises(ValueError):
                    await _validate_and_pin_agent_proxy(card)
