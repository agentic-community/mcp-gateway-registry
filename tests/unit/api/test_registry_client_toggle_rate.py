"""Unit tests for RegistryClient.toggle_service and rate_server payloads.

These lock in the request shapes the registry API expects, guarding against the
CLI/API contract mismatches fixed alongside issue #1316:

- POST /api/servers/toggle wants form fields ``path`` + ``new_state`` (an
  explicit boolean), not a ``service_path`` field, and returns
  ``service_path`` / ``new_enabled_state`` rather than ``path`` / ``is_enabled``.
- POST /api/servers/{path}/rate wants a JSON RatingRequest body, not form data.
"""

from unittest.mock import MagicMock, patch

import pytest

from api.registry_client import RegistryClient


@pytest.fixture
def client() -> RegistryClient:
    """A RegistryClient pointed at a dummy URL with a dummy token."""
    return RegistryClient(registry_url="http://localhost", token="dummy-token-1234567890")


class TestToggleServicePayload:
    """Tests for toggle_service request/response handling."""

    def test_toggle_reads_state_then_sends_opposite(
        self,
        client: RegistryClient,
    ) -> None:
        """toggle_service reads the current state and sends new_state as its opposite."""
        # Arrange: server is currently enabled, so new_state must be "false".
        current = MagicMock(is_enabled=True)
        toggle_response = MagicMock()
        toggle_response.json.return_value = {
            "message": "Toggle request for /currenttime/ processed.",
            "service_path": "/currenttime/",
            "new_enabled_state": False,
        }

        with (
            patch.object(client, "get_server", return_value=current) as mock_get,
            patch.object(client, "_make_request", return_value=toggle_response) as mock_req,
        ):
            # Act
            result = client.toggle_service("/currenttime/")

        # Assert: read happened, and the write used the correct form payload.
        mock_get.assert_called_once_with("/currenttime/")
        mock_req.assert_called_once_with(
            method="POST",
            endpoint="/api/servers/toggle",
            data={"path": "/currenttime/", "new_state": "false"},
        )
        # Response mapped from service_path / new_enabled_state.
        assert result.path == "/currenttime/"
        assert result.is_enabled is False
        assert result.message == "Toggle request for /currenttime/ processed."

    def test_toggle_enables_a_disabled_server(
        self,
        client: RegistryClient,
    ) -> None:
        """A disabled server is sent new_state="true"."""
        current = MagicMock(is_enabled=False)
        toggle_response = MagicMock()
        toggle_response.json.return_value = {
            "message": "Toggle request for /foo/ processed.",
            "service_path": "/foo/",
            "new_enabled_state": True,
        }

        with (
            patch.object(client, "get_server", return_value=current),
            patch.object(client, "_make_request", return_value=toggle_response) as mock_req,
        ):
            result = client.toggle_service("/foo/")

        assert mock_req.call_args.kwargs["data"]["new_state"] == "true"
        assert result.is_enabled is True


class TestRateServerPayload:
    """Tests for rate_server request encoding."""

    def test_rate_sends_json_body(
        self,
        client: RegistryClient,
    ) -> None:
        """rate_server posts to the /rate endpoint with the rating in the body."""
        rate_response = MagicMock()
        rate_response.json.return_value = {
            "message": "Rating added successfully",
            "average_rating": 4.5,
        }

        with patch.object(client, "_make_request", return_value=rate_response) as mock_req:
            client.rate_server("/currenttime/", 4)

        mock_req.assert_called_once()
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["endpoint"] == "/api/servers/currenttime//rate"
        assert kwargs["data"] == {"rating": 4}

    def test_rate_endpoint_routes_to_json_branch(
        self,
        client: RegistryClient,
    ) -> None:
        """_make_request sends a JSON (not form) body for the /rate endpoint."""
        # The endpoint expects a JSON RatingRequest; confirm requests.request is
        # called with json= (not data=) for paths ending in /rate.
        fake_response = MagicMock(status_code=200)
        fake_response.raise_for_status.return_value = None

        with patch("api.registry_client.requests.request", return_value=fake_response) as mock_req:
            client._make_request(
                method="POST",
                endpoint="/api/servers/currenttime//rate",
                data={"rating": 3},
            )

        kwargs = mock_req.call_args.kwargs
        assert kwargs["json"] == {"rating": 3}
        assert kwargs.get("data") is None
