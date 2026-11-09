"""Regression tests for the A2A reverse-proxy url/proxy_pass_url split helper.

The split (advertise the gateway url, keep the registrant backend in
proxy_pass_url) must be applied on EVERY write path -- register, PUT, PATCH --
not just register, or an edit that changes ``url`` would desync the advertised
gateway url from the backend (PR #1434 finding B-2). These tests exercise the
shared helper directly so all three call sites are covered by its contract.
"""

from unittest.mock import PropertyMock, patch

import pytest

from registry.api.agent_routes import _apply_a2a_reverse_proxy_split
from registry.core.config import settings
from registry.schemas.agent_models import AgentCard, AgentInterface


def _make_card(
    url: str,
    supported_protocol: str = "a2a",
    proxy_pass_url: str | None = None,
) -> AgentCard:
    return AgentCard(
        protocol_version="1.0",
        name="Flight Booking",
        description="Books flights",
        url=url,
        version="1.0.0",
        path="/flight-booking",
        supported_protocol=supported_protocol,
        proxy_pass_url=proxy_pass_url,
    )


class TestApplyA2aReverseProxySplit:
    """Contract of _apply_a2a_reverse_proxy_split (shared by register/PUT/PATCH)."""

    def test_split_applied_when_effective_and_a2a(self):
        card = _make_card("http://flight-backend:9000")
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")

        assert card.proxy_pass_url == "http://flight-backend:9000"
        assert str(card.url) == "https://gw.example.com/agent/flight-booking/"

    def test_noop_when_not_effective(self):
        """Registry-only mode: url stays the backend, no proxy_pass_url set."""
        card = _make_card("http://flight-backend:9000")
        with patch.object(
            type(settings),
            "a2a_reverse_proxy_effective",
            new_callable=PropertyMock,
            return_value=False,
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")

        assert str(card.url) == "http://flight-backend:9000"
        assert card.proxy_pass_url is None

    def test_noop_for_non_a2a_agent(self):
        card = _make_card("http://other-backend:9000", supported_protocol="other")
        with patch.object(
            type(settings),
            "a2a_reverse_proxy_effective",
            new_callable=PropertyMock,
            return_value=True,
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")

        assert str(card.url) == "http://other-backend:9000"
        assert card.proxy_pass_url is None

    def test_idempotent_preserves_existing_backend(self):
        """Re-running when url already points at the gateway must NOT clobber the
        stored backend with the gateway url (the PATCH/PUT re-apply case)."""
        card = _make_card(
            "https://gw.example.com/agent/flight-booking/",
            proxy_pass_url="http://flight-backend:9000",
        )
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")

        assert card.proxy_pass_url == "http://flight-backend:9000"
        assert str(card.url) == "https://gw.example.com/agent/flight-booking/"

    def test_patch_of_url_resyncs_backend(self):
        """Simulate a PATCH that changed url to a new backend: the split must move
        the new backend into proxy_pass_url and re-advertise the gateway url."""
        card = _make_card(
            "http://new-backend:9001",  # client changed the backend via PATCH
            proxy_pass_url="http://old-backend:9000",
        )
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")

        assert card.proxy_pass_url == "http://new-backend:9001"
        assert str(card.url) == "https://gw.example.com/agent/flight-booking/"

    def test_rejects_only_unroutable_interfaces_before_rewriting(self):
        card = _make_card("https://backend.example.com/a2a")
        card.supported_interfaces = [
            AgentInterface(
                url="https://backend.example.com/grpc",
                protocolBinding="GRPC",
                protocolVersion="1.0",
            )
        ]
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            with pytest.raises(ValueError, match="declared JSONRPC or HTTP\\+JSON"):
                _apply_a2a_reverse_proxy_split(card, "/flight-booking")
        assert card.proxy_pass_url is None
        assert card.url == "https://backend.example.com/a2a"

    def test_rejects_different_backend_path_on_update(self):
        card = _make_card(
            "https://gw.example.com/agent/flight-booking/",
            proxy_pass_url="https://backend.example.com/a2a",
        )
        card.supported_interfaces = [
            AgentInterface(
                url="https://backend.example.com/rest",
                protocolBinding="HTTP+JSON",
                protocolVersion="1.0",
            )
        ]
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            with pytest.raises(ValueError, match="registered backend URL"):
                _apply_a2a_reverse_proxy_split(card, "/flight-booking")

    def test_preserves_multiple_bindings_at_same_backend_path(self):
        card = _make_card("https://backend.example.com/a2a")
        card.supported_interfaces = [
            AgentInterface(
                url="https://backend.example.com/a2a/",
                protocolBinding="JSONRPC",
                protocolVersion="1.0",
            ),
            AgentInterface(
                url="https://backend.example.com/a2a",
                protocolBinding="HTTP+JSON",
                protocolVersion="1.0",
            ),
            AgentInterface(
                url="grpc.example.com:443", protocolBinding="GRPC", protocolVersion="1.0"
            ),
        ]
        with (
            patch.object(
                type(settings),
                "a2a_reverse_proxy_effective",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(settings, "registry_url", "https://gw.example.com"),
        ):
            _apply_a2a_reverse_proxy_split(card, "/flight-booking")
        assert card.proxy_pass_url == "https://backend.example.com/a2a"
        assert len(card.supported_interfaces) == 3
