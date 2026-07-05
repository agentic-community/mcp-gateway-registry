"""
Unit tests for PeerRegistryClient.

Tests peer registry federation client including server/agent fetching,
health checks, and authentication integration.
"""

from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from registry.schemas.peer_federation_schema import PeerRegistryConfig
from registry.services.federation.peer_registry_client import PeerRegistryClient


@pytest.fixture(autouse=True)
def _allow_safe_endpoints():
    """Treat endpoints as SSRF-safe unless a test overrides the guard.

    The base federation client and the health check now validate every request
    URL with the shared SSRF guard (``validate_url`` with the FEDERATION
    profile), which resolves DNS and fails closed on unresolvable hosts. Most
    tests here use example.com placeholders and are not about SSRF, so the guard
    is stubbed to accept by default. The SSRF-specific tests re-patch it to
    reject inside their own ``with`` block, which overrides this fixture.
    """
    # Both the base client and the health check import validate_url
    # function-locally from registry.utils.url_guard, so patch it at the source.
    with patch("registry.utils.url_guard.validate_url", return_value=[]):
        yield


@pytest.fixture
def peer_config():
    """Create a test peer registry configuration."""
    return PeerRegistryConfig(
        peer_id="test-peer",
        name="Test Peer Registry",
        endpoint="https://peer.example.com",
        enabled=True,
        sync_mode="all",
        sync_interval_minutes=60,
    )


@pytest.fixture
def mock_auth_manager():
    """Mock FederationAuthManager."""
    with patch("registry.services.federation.peer_registry_client.FederationAuthManager") as mock:
        instance = MagicMock()
        instance.is_configured.return_value = True
        instance.get_token.return_value = "test-jwt-token"
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_http_client():
    """Mock the SSRF-guarded httpx client used for HTTP requests.

    The base federation client builds its client via ``guarded_client`` (a
    pinned, rebinding-safe transport). Patch that factory so tests get a mock
    client without any real transport or DNS resolution.
    """
    with patch("registry.services.federation.base_client.guarded_client") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


class TestPeerRegistryClientInitialization:
    """Test client initialization and configuration."""

    def test_client_initialization(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test basic client initialization."""
        # Arrange & Act
        client = PeerRegistryClient(peer_config)

        # Assert
        assert client.peer_config == peer_config
        assert client.endpoint == "https://peer.example.com"
        assert client.timeout_seconds == 30
        assert client.retry_attempts == 3

    def test_client_initialization_with_custom_params(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test client initialization with custom timeout and retries."""
        # Arrange & Act
        client = PeerRegistryClient(
            peer_config,
            timeout_seconds=60,
            retry_attempts=5,
        )

        # Assert
        assert client.timeout_seconds == 60
        assert client.retry_attempts == 5

    def test_client_warns_when_auth_not_configured(
        self,
        peer_config,
        mock_http_client,
        caplog,
    ):
        """Test that client warns when authentication is not configured."""
        # Arrange
        with patch(
            "registry.services.federation.peer_registry_client.FederationAuthManager"
        ) as mock_auth:
            instance = MagicMock()
            instance.is_configured.return_value = False
            mock_auth.return_value = instance

            # Act
            client = PeerRegistryClient(peer_config)

            # Assert
            assert "Federation authentication not configured for peer 'test-peer'" in caplog.text


class TestPeerRegistryClientFetchServers:
    """Test fetch_servers functionality."""

    def test_fetch_servers_returns_parsed_list(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that fetch_servers returns parsed list of server dictionaries."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
            ],
            "sync_generation": 100,
            "total_count": 2,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is not None
            assert len(servers) == 2
            assert servers[0]["path"] == "/server1"
            assert servers[1]["path"] == "/server2"

    def test_fetch_servers_passes_bearer_token_in_header(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that client passes JWT in Authorization Bearer header."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {"items": [], "sync_generation": 0, "total_count": 0}

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response) as mock_request:
            # Act
            client.fetch_servers()

            # Assert
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-jwt-token"

    def test_fetch_servers_without_since_generation(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetch_servers without since_generation parameter."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {"items": [], "sync_generation": 0, "total_count": 0}

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response) as mock_request:
            # Act
            client.fetch_servers()

            # Assert
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[1].get("params", {})
            assert "since_generation" not in params

    def test_fetch_servers_with_since_generation(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that since_generation parameter is correctly passed to API."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {"items": [], "sync_generation": 50, "total_count": 0}

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response) as mock_request:
            # Act
            client.fetch_servers(since_generation=42)

            # Assert
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["since_generation"] == 42

    def test_fetch_servers_with_dict_response(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetch_servers handles dict response format."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [{"path": "/server1"}],
            "sync_generation": 100,
            "total_count": 1,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is not None
            assert len(servers) == 1
            assert servers[0]["path"] == "/server1"

    def test_fetch_servers_with_direct_list_response(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetch_servers handles direct list response format."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = [
            {"path": "/server1"},
            {"path": "/server2"},
        ]

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is not None
            assert len(servers) == 2

    def test_fetch_servers_handles_auth_failure(
        self,
        peer_config,
        mock_http_client,
        caplog,
    ):
        """Test fetch_servers handles authentication failure."""
        # Arrange
        with patch(
            "registry.services.federation.peer_registry_client.FederationAuthManager"
        ) as mock_auth:
            instance = MagicMock()
            instance.is_configured.return_value = True
            instance.get_token.return_value = None  # Auth failure
            mock_auth.return_value = instance

            client = PeerRegistryClient(peer_config)

            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is None
            assert "Failed to obtain authentication token" in caplog.text

    def test_fetch_servers_handles_auth_not_configured(
        self,
        peer_config,
        mock_http_client,
        caplog,
    ):
        """Test fetch_servers handles authentication not configured."""
        # Arrange
        with patch(
            "registry.services.federation.peer_registry_client.FederationAuthManager"
        ) as mock_auth:
            instance = MagicMock()
            instance.is_configured.return_value = True
            instance.get_token.side_effect = ValueError("Not configured")
            mock_auth.return_value = instance

            client = PeerRegistryClient(peer_config)

            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is None
            assert "Cannot fetch servers" in caplog.text

    def test_fetch_servers_handles_request_failure(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
        caplog,
    ):
        """Test fetch_servers handles HTTP request failure."""
        # Arrange
        client = PeerRegistryClient(peer_config)

        # Mock the _make_request method to return None (failure)
        with patch.object(client, "_make_request", return_value=None):
            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is None
            assert "Failed to fetch servers from peer 'test-peer'" in caplog.text

    def test_fetch_servers_handles_unexpected_response_format(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
        caplog,
    ):
        """Test fetch_servers handles unexpected response format."""
        # Arrange
        client = PeerRegistryClient(peer_config)

        # Mock the _make_request method to return unexpected format
        with patch.object(client, "_make_request", return_value="invalid"):
            # Act
            servers = client.fetch_servers()

            # Assert
            assert servers is None
            assert "Unexpected response format" in caplog.text


class TestPeerRegistryClientFetchAgents:
    """Test fetch_agents functionality."""

    def test_fetch_agents_returns_parsed_list(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that fetch_agents returns parsed list of agent dictionaries."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/agent1", "name": "Agent 1"},
                {"path": "/agent2", "name": "Agent 2"},
            ],
            "sync_generation": 100,
            "total_count": 2,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            agents = client.fetch_agents()

            # Assert
            assert agents is not None
            assert len(agents) == 2
            assert agents[0]["path"] == "/agent1"
            assert agents[1]["path"] == "/agent2"

    def test_fetch_agents_passes_bearer_token_in_header(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that client passes JWT in Authorization Bearer header."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {"items": [], "sync_generation": 0, "total_count": 0}

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response) as mock_request:
            # Act
            client.fetch_agents()

            # Assert
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-jwt-token"

    def test_fetch_agents_with_since_generation(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that since_generation parameter is correctly passed to API."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {"items": [], "sync_generation": 50, "total_count": 0}

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response) as mock_request:
            # Act
            client.fetch_agents(since_generation=42)

            # Assert
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["since_generation"] == 42

    def test_fetch_agents_handles_auth_failure(
        self,
        peer_config,
        mock_http_client,
        caplog,
    ):
        """Test fetch_agents handles authentication failure."""
        # Arrange
        with patch(
            "registry.services.federation.peer_registry_client.FederationAuthManager"
        ) as mock_auth:
            instance = MagicMock()
            instance.is_configured.return_value = True
            instance.get_token.return_value = None  # Auth failure
            mock_auth.return_value = instance

            client = PeerRegistryClient(peer_config)

            # Act
            agents = client.fetch_agents()

            # Assert
            assert agents is None
            assert "Failed to obtain authentication token" in caplog.text


class TestPeerRegistryClientCheckHealth:
    """Test check_peer_health functionality."""

    def test_check_peer_health_returns_true_for_healthy_peer(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that check_peer_health returns True for healthy peer."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http_client.get.return_value = mock_response

        # Act
        is_healthy = client.check_peer_health()

        # Assert
        assert is_healthy is True
        mock_http_client.get.assert_called_once_with("https://peer.example.com/health")

    def test_check_peer_health_returns_false_for_unhealthy_peer(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that check_peer_health returns False for unhealthy peer."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = Mock()
        mock_response.status_code = 503
        mock_http_client.get.return_value = mock_response

        # Act
        is_healthy = client.check_peer_health()

        # Assert
        assert is_healthy is False

    def test_check_peer_health_accepts_2xx_status_codes(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that check_peer_health accepts various 2xx status codes."""
        # Arrange
        client = PeerRegistryClient(peer_config)

        # Test various 2xx codes
        for status_code in [200, 201, 204, 299]:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_http_client.get.return_value = mock_response

            # Act
            is_healthy = client.check_peer_health()

            # Assert
            assert is_healthy is True

    def test_check_peer_health_handles_network_errors(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
        caplog,
    ):
        """Test that check_peer_health handles network errors gracefully."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")

        # Act
        is_healthy = client.check_peer_health()

        # Assert
        assert is_healthy is False
        assert "Health check failed for peer 'test-peer'" in caplog.text

    def test_check_peer_health_handles_timeout_errors(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that check_peer_health handles timeout errors gracefully."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_http_client.get.side_effect = httpx.TimeoutException("Request timed out")

        # Act
        is_healthy = client.check_peer_health()

        # Assert
        assert is_healthy is False


class TestPeerRegistryClientRetryLogic:
    """Test retry logic inherited from BaseFederationClient."""

    def test_client_follows_base_federation_client_retry_logic(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that client follows BaseFederationClient retry logic."""
        # Arrange
        client = PeerRegistryClient(peer_config, retry_attempts=3)

        # Mock intermittent failure then success
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [{"path": "/server1"}],
            "sync_generation": 1,
            "total_count": 1,
        }

        mock_http_client.request.side_effect = [
            httpx.RequestError("Network error"),  # First attempt fails
            httpx.RequestError("Network error"),  # Second attempt fails
            mock_response,  # Third attempt succeeds
        ]

        # Act
        servers = client.fetch_servers()

        # Assert
        assert servers is not None
        assert len(servers) == 1
        assert mock_http_client.request.call_count == 3

    def test_http_4xx_errors_not_retried(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
        caplog,
    ):
        """Test that HTTP 4xx errors are not retried."""
        # Arrange
        client = PeerRegistryClient(peer_config, retry_attempts=3)

        # Mock 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_http_client.request.side_effect = httpx.HTTPStatusError(
            "Not found",
            request=Mock(),
            response=mock_response,
        )

        # Act
        servers = client.fetch_servers()

        # Assert
        assert servers is None
        # Should only attempt once (no retries for 404)
        assert mock_http_client.request.call_count == 1

    def test_http_5xx_errors_retried(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test that HTTP 5xx errors are retried."""
        # Arrange
        client = PeerRegistryClient(peer_config, retry_attempts=3)

        # Mock 500 error on all attempts
        mock_response = Mock()
        mock_response.status_code = 500
        mock_http_client.request.side_effect = httpx.HTTPStatusError(
            "Internal server error",
            request=Mock(),
            response=mock_response,
        )

        # Act
        servers = client.fetch_servers()

        # Assert
        assert servers is None
        # Should attempt 3 times
        assert mock_http_client.request.call_count == 3


class TestPeerRegistryClientFetchSingleServer:
    """Test fetch_server functionality."""

    def test_fetch_server_by_path(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetching a single server by path."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
            ],
            "sync_generation": 1,
            "total_count": 2,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            server = client.fetch_server("/server1")

            # Assert
            assert server is not None
            assert server["path"] == "/server1"
            assert server["name"] == "Server 1"

    def test_fetch_server_not_found(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
        caplog,
    ):
        """Test fetching a server that doesn't exist."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/server1", "name": "Server 1"},
            ],
            "sync_generation": 1,
            "total_count": 1,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            server = client.fetch_server("/nonexistent")

            # Assert
            assert server is None
            assert "Server '/nonexistent' not found in peer 'test-peer'" in caplog.text

    def test_fetch_server_handles_fetch_failure(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetch_server handles failure to fetch servers."""
        # Arrange
        client = PeerRegistryClient(peer_config)

        # Mock the _make_request method to return None
        with patch.object(client, "_make_request", return_value=None):
            # Act
            server = client.fetch_server("/server1")

            # Assert
            assert server is None


class TestPeerRegistryClientFetchAllServers:
    """Test fetch_all_servers functionality."""

    def test_fetch_all_servers_with_no_filter(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetching all servers without filtering."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
            ],
            "sync_generation": 1,
            "total_count": 2,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            servers = client.fetch_all_servers([])

            # Assert
            assert servers is not None
            assert len(servers) == 2

    def test_fetch_all_servers_with_filter(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetching all servers with name filtering."""
        # Arrange
        client = PeerRegistryClient(peer_config)
        mock_response = {
            "items": [
                {"path": "/server1", "name": "Server 1"},
                {"path": "/server2", "name": "Server 2"},
                {"path": "/server3", "name": "Server 3"},
            ],
            "sync_generation": 1,
            "total_count": 3,
        }

        # Mock the _make_request method
        with patch.object(client, "_make_request", return_value=mock_response):
            # Act
            servers = client.fetch_all_servers(["/server1", "/server3"])

            # Assert
            assert servers is not None
            assert len(servers) == 2
            assert servers[0]["path"] == "/server1"
            assert servers[1]["path"] == "/server3"

    def test_fetch_all_servers_handles_fetch_failure(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        """Test fetch_all_servers handles failure to fetch servers."""
        # Arrange
        client = PeerRegistryClient(peer_config)

        # Mock the _make_request method to return None
        with patch.object(client, "_make_request", return_value=None):
            # Act
            servers = client.fetch_all_servers(["/server1"])

            # Assert
            assert servers == []


class TestPeerRegistryClientSsrfGuard:
    """The federation bearer token must never be sent to an unsafe endpoint.

    _make_request is the single egress chokepoint and attaches the token via the
    Authorization header. If the target URL resolves to a blocked address the
    request must be denied before any HTTP call, so the token cannot be exfiltrated.
    """

    def test_make_request_denies_unsafe_url_without_calling_http(
        self,
        peer_config,
        mock_auth_manager,
        mock_http_client,
    ):
        from registry.exceptions import UrlValidationError

        client = PeerRegistryClient(peer_config=peer_config)

        with patch(
            "registry.utils.url_guard.validate_url",
            side_effect=UrlValidationError("http://169.254.169.254/", "blocked"),
        ):
            result = client._make_request(
                "http://169.254.169.254/latest/meta-data/",
                headers={"Authorization": "Bearer super-secret-token"},
            )

        assert result is None
        # No HTTP request was ever issued, so the token never left the process.
        client.client.request.assert_not_called()

    def test_fetch_servers_does_not_send_token_to_unsafe_endpoint(
        self,
        mock_auth_manager,
        mock_http_client,
    ):
        from registry.exceptions import UrlValidationError

        # A peer whose endpoint resolves to a private address: fetch must abort
        # before the HTTP call rather than send the bearer token there.
        peer = PeerRegistryConfig(
            peer_id="evil-peer",
            name="Evil Peer",
            endpoint="http://10.0.0.9",
            federation_token="super-secret-token",
        )
        client = PeerRegistryClient(peer_config=peer)

        with patch(
            "registry.utils.url_guard.validate_url",
            side_effect=UrlValidationError("http://10.0.0.9", "blocked"),
        ):
            result = client.fetch_servers()

        assert result is None
        client.client.request.assert_not_called()

    def test_check_peer_health_denies_unsafe_endpoint(
        self,
        mock_auth_manager,
        mock_http_client,
    ):
        from registry.exceptions import UrlValidationError

        peer = PeerRegistryConfig(
            peer_id="evil-peer",
            name="Evil Peer",
            endpoint="http://127.0.0.1:9000",
        )
        client = PeerRegistryClient(peer_config=peer)

        with patch(
            "registry.utils.url_guard.validate_url",
            side_effect=UrlValidationError("http://127.0.0.1:9000", "blocked"),
        ):
            assert client.check_peer_health() is False

        client.client.get.assert_not_called()

    def test_client_uses_pinned_federation_guarded_transport(
        self,
        peer_config,
        mock_auth_manager,
    ):
        """The outbound client must be the rebinding-safe guarded transport.

        The pre-check alone is TOCTOU-vulnerable to DNS rebinding, so the token
        could otherwise be sent to a private/metadata address that the pre-check
        just cleared. The authoritative defense is the pinned transport, which
        re-resolves and validates inside every connect. This asserts the client
        is built from the FEDERATION profile (no allowlist bypass) rather than a
        plain httpx.Client.
        """
        from registry.utils.url_guard import (
            FEDERATION_PROFILE,
            GuardedTransport,
        )

        # Build with the real guarded_client (no mock) to inspect the transport.
        client = PeerRegistryClient(peer_config=peer_config)

        transport = client.client._transport
        assert isinstance(transport, GuardedTransport)
        assert transport._guard_profile is FEDERATION_PROFILE
        # The federation profile grants no bypass: its allowlist is empty.
        allowlist = FEDERATION_PROFILE.allowlist_factory()
        assert not allowlist.hosts
        assert not allowlist.cidrs
