"""The sample agent must not let the model choose where a tool call goes.

``invoke_mcp_tool`` attaches the gateway bearer token to every request it makes, so
a destination the model can influence is a token-exfiltration path. These tests pin
both halves of that: the registry URL comes from configuration, and the per-call
server path cannot replace the configured host.

Covers issue #1862.
"""

import sys
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

agent = pytest.importorskip(
    "agent",
    reason="sample agent dependencies (langgraph, langchain) are not installed",
)


@pytest.fixture
def configured_registry():
    """Point the agent at a known registry and restore the prior value after."""
    previous = agent.agent_settings.registry_url
    agent.agent_settings.registry_url = "https://registry.example.com/mcpgw/mcp"
    yield "https://registry.example.com"
    agent.agent_settings.registry_url = previous


class TestConfiguredRegistryBaseUrl:
    """The destination comes from startup configuration, never from a tool call."""

    def test_returns_scheme_and_host_from_configuration(self, configured_registry):
        assert agent._configured_registry_base_url() == configured_registry

    def test_model_cannot_supply_the_registry_url(self):
        # The parameter is gone from the tool schema, so there is no field a
        # prompt-injected instruction can set. Guard against it coming back.
        params = agent.invoke_mcp_tool.args_schema.model_fields
        assert "mcp_registry_url" not in params

    def test_unset_registry_url_is_refused(self):
        previous = agent.agent_settings.registry_url
        agent.agent_settings.registry_url = None
        try:
            with pytest.raises(agent.AgentConfigError, match="No registry URL"):
                agent._configured_registry_base_url()
        finally:
            agent.agent_settings.registry_url = previous

    @pytest.mark.parametrize(
        "configured",
        [
            "file:///etc/passwd",
            "gopher://registry.example.com",
            "data:text/plain,hello",
        ],
    )
    def test_non_http_scheme_is_refused(self, configured):
        previous = agent.agent_settings.registry_url
        agent.agent_settings.registry_url = configured
        try:
            with pytest.raises(agent.AgentConfigError, match="scheme"):
                agent._configured_registry_base_url()
        finally:
            agent.agent_settings.registry_url = previous

    def test_url_without_a_host_is_refused(self):
        previous = agent.agent_settings.registry_url
        agent.agent_settings.registry_url = "https:///mcpgw/mcp"
        try:
            with pytest.raises(agent.AgentConfigError, match="no host"):
                agent._configured_registry_base_url()
        finally:
            agent.agent_settings.registry_url = previous


class TestSafeServerPath:
    """A model-supplied server path cannot move the request to another host."""

    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("currenttime", "currenttime"),
            ("/currenttime", "currenttime"),
            ("mcpgw/mcp", "mcpgw/mcp"),
            ("  /mcpgw/mcp  ", "mcpgw/mcp"),
            ("//currenttime", "currenttime"),
        ],
    )
    def test_accepts_a_plain_registry_path(self, supplied, expected):
        assert agent._safe_server_path(supplied) == expected

    @pytest.mark.parametrize(
        "supplied",
        [
            "http://evil.example/x",
            "https://evil.example",
            "HTTP://evil.example",
            "user@evil.example",
            "currenttime?next=http://evil.example",
            "currenttime#http://evil.example",
            "current time",
            "back\\slash",
            "percent%2Fescape",
            "..",
            "../../etc/passwd",
            "mcpgw/../../evil",
        ],
    )
    def test_refuses_anything_that_could_change_the_host(self, supplied):
        with pytest.raises(agent.AgentConfigError):
            agent._safe_server_path(supplied)

    def test_empty_path_is_refused(self):
        with pytest.raises(agent.AgentConfigError, match="empty"):
            agent._safe_server_path("///")


class TestNoTokenLeavesForAnotherHost:
    """An absolute server_name used to replace the base URL outright."""

    def test_absolute_server_name_no_longer_escapes_the_base(
        self,
        configured_registry,
    ):
        from urllib.parse import urljoin

        hostile = "http://evil.example/x"

        # What the old code did: lstrip("/") then urljoin, which urljoin honors as
        # an absolute URL and so drops the configured host.
        unguarded = urljoin(configured_registry + "/", hostile.lstrip("/"))
        assert unguarded.startswith("http://evil.example")

        # The guard refuses it instead.
        with pytest.raises(agent.AgentConfigError):
            agent._safe_server_path(hostile)

    def test_guarded_path_stays_on_the_configured_host(self, configured_registry):
        from urllib.parse import urljoin

        base = agent._configured_registry_base_url()
        server_url = urljoin(base + "/", agent._safe_server_path("/currenttime"))
        assert server_url == "https://registry.example.com/currenttime"
