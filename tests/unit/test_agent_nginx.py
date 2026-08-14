"""Unit tests for A2A agent nginx reverse-proxy config generation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from registry.core.nginx_service import NginxConfigService


def _agent(
    path="/flight-booking-agent",
    url="https://flight-booking.dev.example.com",
    name="Flight Booking Agent",
    supported_protocol="a2a",
    health_status="healthy",
    proxy_pass_url=None,
):
    """Build a lightweight agent-card stand-in for the generator.

    The generator reads ``path``, ``url``, ``name``, ``supported_protocol``,
    ``health_status`` and ``proxy_pass_url``, so a namespace avoids coupling the
    test to the full AgentCard pydantic model. Defaults describe a healthy A2A
    agent (the case that produces a proxy block). ``proxy_pass_url`` defaults to
    None (the flag-off / legacy case where ``url`` is the backend).
    """
    return SimpleNamespace(
        path=path,
        url=url,
        name=name,
        supported_protocol=supported_protocol,
        health_status=health_status,
        proxy_pass_url=proxy_pass_url,
    )


@pytest.fixture
def patched_agent_service():
    """Patch the agent_service singleton imported lazily by the generator.

    Also stub _agent_backend_resolves to True so block-generation tests do not
    depend on live DNS for their (non-resolvable) example backend hosts. Tests
    that exercise the resolve-skip behavior patch it explicitly.
    """
    with (
        patch("registry.services.agent_service.agent_service") as mock_svc,
        patch.object(
            NginxConfigService,
            "_agent_backend_resolves",
            AsyncMock(return_value=True),
        ),
    ):
        mock_svc.get_enabled_agents = AsyncMock(return_value=[])
        mock_svc.get_agent_info = AsyncMock(return_value=None)
        yield mock_svc


class TestGenerateAgentLocationBlocks:
    """Tests for _generate_agent_location_blocks."""

    @pytest.mark.asyncio
    async def test_no_enabled_agents_returns_empty(self, patched_agent_service):
        """Empty string returned when there are no enabled agents."""
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_generates_blocks_for_enabled_agent(self, patched_agent_service):
        """An enabled agent produces a location block at /agent/{path}/."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "{{ROOT_PATH}}/agent/flight-booking-agent/" in result

    @pytest.mark.asyncio
    async def test_includes_agent_card_discovery_location(self, patched_agent_service):
        """The agent-card discovery location is emitted."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "{{ROOT_PATH}}/agent/flight-booking-agent/.well-known/agent-card.json" in result

    @pytest.mark.asyncio
    async def test_block_enforces_auth_request(self, patched_agent_service):
        """Generated blocks are protected by the /validate auth subrequest."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "auth_request /validate;" in result

    @pytest.mark.asyncio
    async def test_jsonrpc_block_captures_rate_limit_headers(self, patched_agent_service):
        """The JSON-RPC block captures the throttle marker + headers (issue #295).

        nginx auth_request only forwards 401/403, so a throttle leaves /validate as
        a 403; these captures let @forbidden_error rewrite it into a real 429.
        """
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "auth_request_set $rl_throttled $upstream_http_x_ratelimit_throttled;" in result
        assert "auth_request_set $rl_limit $upstream_http_x_ratelimit_limit;" in result
        assert "auth_request_set $rl_reset $upstream_http_x_ratelimit_reset;" in result
        # Retry-After is captured from the actual Retry-After response header the
        # auth-server sends (not an X-RateLimit-* alias), so nginx can re-emit it.
        assert "auth_request_set $rl_retry $upstream_http_retry_after;" in result

    @pytest.mark.asyncio
    async def test_jsonrpc_block_disables_buffering_for_sse(self, patched_agent_service):
        """The JSON-RPC block disables proxy buffering for message/stream SSE."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "proxy_buffering off;" in result

    @pytest.mark.asyncio
    async def test_proxies_to_backend_url(self, patched_agent_service):
        """The block proxies to the agent backend url (legacy: proxy_pass_url unset)."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent())
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "proxy_pass https://flight-booking.dev.example.com/;" in result

    @pytest.mark.asyncio
    async def test_proxies_to_proxy_pass_url_when_set(self, patched_agent_service):
        """In reverse-proxy mode the advertised url is the gateway address, so the
        block must proxy to proxy_pass_url (the real backend), not url."""
        agent = _agent(
            url="https://gateway.example.com/agent/flight-booking-agent/",
            proxy_pass_url="http://flight-booking-agent:9000",
        )
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=agent)
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        # Proxies to the real backend, never to the advertised gateway url.
        assert "proxy_pass http://flight-booking-agent:9000/" in result
        assert "proxy_pass https://gateway.example.com/agent/" not in result

    @pytest.mark.asyncio
    async def test_skips_agent_without_url(self, patched_agent_service):
        """An enabled agent with no backend url is skipped."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent(url=""))
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_missing_agent_card(self, patched_agent_service):
        """An enabled path with no resolvable card is skipped."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=None)
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_agent_with_unsafe_path(self, patched_agent_service):
        """An agent whose path would inject nginx directives is skipped."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/evil"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(path="evil}\n location / { return 200; }\n #")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_agent_with_unsafe_url(self, patched_agent_service):
        """An agent whose backend url is not a safe http(s) url is skipped."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/evil"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(url="https://host/ { return 200; }")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_generates_block_for_percent_encoded_agentcore_url(self, patched_agent_service):
        """A Bedrock AgentCore runtime url is routable.

        AgentCore invocation urls embed a percent-encoded ARN (built by
        cli/agentcore/models.py::build_invocation_url), so the url-safety check
        must accept %XX octets. Rejecting them made every imported AgentCore A2A
        agent silently unroutable ("unsafe agent url" -> skipped, 0 blocks).
        """
        agentcore_url = (
            "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
            "arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A123456789012%3A"
            "runtime%2Fevcharger2-qR51Wr33ZA/invocations"
        )
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/evcharger2"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(path="/evcharger2", url=agentcore_url, name="EV Charger 2")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        # Routes are emitted with the {{ROOT_PATH}} placeholder, substituted later.
        assert "/agent/evcharger2/ {" in result
        # The encoded ARN must survive into proxy_pass verbatim; re-encoding or
        # stripping it would point the route at a non-existent runtime.
        assert f"proxy_pass {agentcore_url}/;" in result

    @pytest.mark.parametrize(
        "bad_url",
        [
            "https://host/a%",  # truncated: no hex digits
            "https://host/a%2",  # truncated: one hex digit
            "https://host/a%zz",  # not hex
            "https://host/a%2y",  # second char not hex
        ],
    )
    @pytest.mark.asyncio
    async def test_skips_agent_with_incomplete_percent_escape(self, patched_agent_service, bad_url):
        """Only a COMPLETE %XX triplet is allowed, never a bare/partial '%'.

        This is what keeps the anti-injection property while permitting
        AgentCore urls: an unescaped '%' can never reach an nginx directive.
        """
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/evil"])
        patched_agent_service.get_agent_info = AsyncMock(return_value=_agent(url=bad_url))
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_percent_encoding_does_not_admit_nginx_injection(self, patched_agent_service):
        """Allowing %XX must not let config-breaking characters through.

        The characters that would break out of a location block are not hex
        digits, so they still fail even adjacent to a valid escape.
        """
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/evil"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(url="https://host/%2F; return 200; #")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_unhealthy_agent(self, patched_agent_service):
        """An unhealthy agent is skipped so no route points at a dead backend."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(health_status="unhealthy")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_agent_with_unknown_health(self, patched_agent_service):
        """An agent whose health is not yet verified (unknown) is skipped."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(health_status="unknown")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_agent_whose_backend_host_does_not_resolve(self):
        """An agent whose backend host does not resolve is skipped (fail safe): the
        block emits a literal proxy_pass and an unresolvable host would fail the
        whole nginx reload. Uses its own patches so the resolver is NOT stubbed."""
        with (
            patch("registry.services.agent_service.agent_service") as mock_svc,
            patch.object(
                NginxConfigService,
                "_agent_backend_resolves",
                AsyncMock(return_value=False),
            ),
        ):
            mock_svc.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
            mock_svc.get_agent_info = AsyncMock(return_value=_agent())
            service = NginxConfigService()

            result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_generates_block_when_backend_host_resolves(self):
        """The block IS generated when the backend host resolves (real resolve
        check stubbed True), confirming the guard is what gates generation."""
        with (
            patch("registry.services.agent_service.agent_service") as mock_svc,
            patch.object(
                NginxConfigService,
                "_agent_backend_resolves",
                AsyncMock(return_value=True),
            ),
        ):
            mock_svc.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
            mock_svc.get_agent_info = AsyncMock(return_value=_agent())
            service = NginxConfigService()

            result = await service._generate_agent_location_blocks()

        assert "{{ROOT_PATH}}/agent/flight-booking-agent/" in result

    @pytest.mark.asyncio
    async def test_skips_non_a2a_protocol_agent(self, patched_agent_service):
        """A non-A2A agent with a URL does not get a JSON-RPC proxy block."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(supported_protocol="other")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_agent_missing_protocol(self, patched_agent_service):
        """An agent with no supported_protocol is skipped (not treated as A2A)."""
        patched_agent_service.get_enabled_agents = AsyncMock(return_value=["/flight-booking-agent"])
        patched_agent_service.get_agent_info = AsyncMock(
            return_value=_agent(supported_protocol=None)
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""

    @pytest.mark.asyncio
    async def test_generates_blocks_for_multiple_agents(self, patched_agent_service):
        """Each enabled agent gets its own route."""
        patched_agent_service.get_enabled_agents = AsyncMock(
            return_value=["/flight-booking-agent", "/travel-assistant-agent"]
        )

        async def _info(path):
            return _agent(path=path, url=f"https://flight-booking{path}.example.com")

        patched_agent_service.get_agent_info = AsyncMock(side_effect=_info)
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert "{{ROOT_PATH}}/agent/flight-booking-agent/" in result
        assert "{{ROOT_PATH}}/agent/travel-assistant-agent/" in result

    @pytest.mark.asyncio
    async def test_generation_failure_returns_empty(self, patched_agent_service):
        """A failure while resolving agents must not break config rendering.

        The generator fails closed to an empty string so nginx still reloads
        (without any agent proxy blocks) instead of raising.
        """
        patched_agent_service.get_enabled_agents = AsyncMock(
            side_effect=RuntimeError("agent store unavailable")
        )
        service = NginxConfigService()

        result = await service._generate_agent_location_blocks()

        assert result == ""


class TestReverseProxyFlagDefault:
    """The reverse-proxy mode must be opt-in."""

    def test_flag_defaults_to_disabled(self):
        """A2A_REVERSE_PROXY_ENABLED defaults to False (opt-in)."""
        from registry.core.config import Settings

        assert Settings.model_fields["a2a_reverse_proxy_enabled"].default is False


class TestAgentBackendResolves:
    """The pre-emit DNS resolution guard (fail safe before a literal proxy_pass)."""

    @pytest.mark.asyncio
    async def test_empty_host_is_false(self):
        assert await NginxConfigService._agent_backend_resolves("") is False

    @pytest.mark.asyncio
    async def test_ip_literal_resolves(self):
        assert await NginxConfigService._agent_backend_resolves("127.0.0.1") is True

    @pytest.mark.asyncio
    async def test_dead_host_is_false(self):
        """A name that cannot resolve returns False (skip, do not crash reload)."""
        assert await NginxConfigService._agent_backend_resolves("no-such-host.invalid") is False


class TestCreateAgentLocationBlock:
    """Tests for _create_agent_location_block."""

    def test_external_host_uses_upstream_hostname(self):
        """An https backend sets the Host header to the upstream hostname."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "proxy_set_header Host flight-booking.dev.example.com;" in block

    def test_internal_host_preserves_original_host(self):
        """A bare internal hostname preserves the original Host header."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "http://flight-booking-agent",
            "Flight Booking Agent",
        )

        assert "proxy_set_header Host $host;" in block

    def test_captures_body_for_metrics(self):
        """The JSON-RPC block captures the request body so metrics can record the method."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;" in block

    def test_card_location_is_exact_match(self):
        """The agent-card location uses an exact match to block suffix smuggling."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert (
            "location = {{ROOT_PATH}}/agent/flight-booking-agent/.well-known/agent-card.json"
            in block
        )

    def test_card_proxy_pass_has_no_double_slash(self):
        """The agent-card proxy_pass targets the backend without a double slash."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert (
            "proxy_pass https://flight-booking.dev.example.com/.well-known/agent-card.json;"
            in block
        )

    def test_streaming_read_timeout_present(self):
        """The JSON-RPC block sets a long read timeout for SSE streaming."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "proxy_read_timeout 86400s;" in block

    def test_dotted_internal_hostname_uses_upstream_netloc(self):
        """A dotted internal hostname (with port) sets Host to the upstream netloc."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "http://svc.namespace.svc.cluster.local:8080",
            "Flight Booking Agent",
        )

        assert "proxy_set_header Host svc.namespace.svc.cluster.local:8080;" in block

    def test_unsafe_path_raises(self):
        """An agent path with nginx metacharacters raises ValueError."""
        service = NginxConfigService()

        with pytest.raises(ValueError, match="unsafe agent path"):
            service._create_agent_location_block(
                "evil}\n location / {}",
                "https://flight-booking.dev.example.com",
                "Evil",
            )

    def test_unsafe_url_raises(self):
        """A backend url with nginx metacharacters raises ValueError."""
        service = NginxConfigService()

        with pytest.raises(ValueError, match="unsafe agent url"):
            service._create_agent_location_block(
                "flight-booking-agent",
                "https://host/ { return 200; }",
                "Evil",
            )

    def test_card_location_rewrites_card_urls(self):
        """The agent-card location runs the card-rewrite body filter."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "body_filter_by_lua_file /etc/nginx/lua/agent_card_rewrite.lua;" in block

    def test_card_location_clears_content_length(self):
        """Content-Length is cleared so the rewritten card body is not truncated."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "ngx.header.content_length = nil" in block

    def test_jsonrpc_block_forwards_scopes_to_backend(self):
        """Validated scopes are captured from /validate and forwarded to the agent."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        assert "auth_request_set $auth_scopes $upstream_http_x_scopes;" in block
        assert "proxy_set_header X-Scopes $auth_scopes;" in block

    def test_multi_segment_agent_path_in_route(self):
        """A multi-segment agent path renders verbatim in the location route."""
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "lob1/travel",
            "https://flight-booking.dev.example.com",
            "Travel",
        )

        assert "{{ROOT_PATH}}/agent/lob1/travel/" in block

    def test_strips_gateway_credential_but_forwards_target_authorization(self):
        """A2A egress trust model (Design A, spec-native passthrough).

        The gateway credential travels in X-Authorization and must be stripped on
        egress (with Cookie) so a registrant-controlled agent cannot capture and
        replay it against the registry (the B1 / #1391 class of bug). The standard
        Authorization header carries the *target agent's* credential, obtained
        out-of-band by the calling agent per the A2A spec, and is forwarded
        end-to-end untouched -- the gateway is a policy gate, not a credential
        broker. So the blocks must clear X-Authorization/Cookie and must NOT clear
        or override Authorization.
        """
        service = NginxConfigService()

        block = service._create_agent_location_block(
            "flight-booking-agent",
            "https://flight-booking.dev.example.com",
            "Flight Booking Agent",
        )

        # Gateway credential + session cookie stripped on both blocks (card + RPC).
        assert block.count('proxy_set_header X-Authorization "";') == 2
        assert block.count('proxy_set_header Cookie "";') == 2
        # The target-agent Authorization is forwarded end-to-end: the block must
        # neither clear it nor rewrite it to the gateway's own credential.
        assert 'proxy_set_header Authorization "";' not in block
        assert "proxy_set_header Authorization $http_authorization;" not in block
