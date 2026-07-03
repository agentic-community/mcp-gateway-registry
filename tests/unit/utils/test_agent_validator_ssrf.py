"""
Unit tests for SSRF protection on the agent endpoint reachability probe.

``_check_endpoint_reachability`` fetches a user-controlled agent URL's
well-known card. It must resolve+validate+pin a public IP (proxy profile) so a
private/metadata target cannot be reached, and a blocked URL must degrade to
"unreachable" (advisory) rather than raise or fall through to a raw fetch.
"""

from unittest.mock import patch

from registry.utils.agent_validator import _check_endpoint_reachability


class TestReachabilitySsrfGuard:
    """The reachability probe fails closed on private/metadata targets."""

    def test_private_ip_reported_unreachable_not_fetched(self):
        """A host resolving to a private IP is reported unreachable, not fetched."""
        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [(None, None, None, None, ("10.0.0.5", 443))]

            reachable, error = _check_endpoint_reachability("https://evil.example")

            assert reachable is False
            assert error == "Endpoint URL failed SSRF validation"

    def test_metadata_ip_literal_reported_unreachable(self):
        """The cloud metadata IP literal is blocked (never allowlistable)."""
        reachable, error = _check_endpoint_reachability("http://169.254.169.254")

        assert reachable is False
        assert error == "Endpoint URL failed SSRF validation"

    def test_public_host_uses_guarded_client(self):
        """A public host resolves and is probed through the guarded client."""
        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [(None, None, None, None, ("140.82.112.3", 443))]

            with patch("registry.utils.url_guard.guarded_client") as mock_client_factory:
                mock_client = mock_client_factory.return_value.__enter__.return_value
                mock_client.get.return_value.status_code = 200

                reachable, error = _check_endpoint_reachability("https://good.example")

                assert reachable is True
                assert error is None
                mock_client.get.assert_called_once()
