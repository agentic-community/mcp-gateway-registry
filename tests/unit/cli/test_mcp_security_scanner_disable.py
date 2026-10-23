"""Unit tests for mcp_security_scanner's --auto-disable path.

The scanner disables a server that fails its scan through the registry API.
The toggle endpoint flips state, so the scanner must read the current state
first and never toggle a server that is already disabled. Every failure must
report False so the caller knows the server may still be serving traffic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import cli.mcp_security_scanner as scanner

REGISTRY_URL: str = "http://localhost"


@pytest.fixture
def token_file(tmp_path):
    """A nested-layout token file, like the repo's .token."""
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"tokens": {"access_token": "test-token"}}))
    return str(path)


def _client(
    enabled_before: bool,
    enabled_after: bool,
) -> MagicMock:
    """Build a RegistryClient stand-in with the given states."""
    client = MagicMock()
    client.get_server.return_value.is_enabled = enabled_before
    client.toggle_service.return_value.is_enabled = enabled_after
    return client


class TestDisableUnsafeServer:
    """Tests for _disable_unsafe_server."""

    def test_disables_an_enabled_server(self, token_file):
        """An enabled server is toggled once and reported disabled."""
        client = _client(enabled_before=True, enabled_after=False)
        with patch.object(scanner, "RegistryClient", return_value=client):
            result = scanner._disable_unsafe_server("/srv", REGISTRY_URL, token_file)

        assert result is True
        client.toggle_service.assert_called_once_with("/srv")

    def test_never_toggles_an_already_disabled_server(self, token_file):
        """Toggling a disabled server would enable it, so it is left alone."""
        client = _client(enabled_before=False, enabled_after=True)
        with patch.object(scanner, "RegistryClient", return_value=client):
            result = scanner._disable_unsafe_server("/srv", REGISTRY_URL, token_file)

        assert result is True
        client.toggle_service.assert_not_called()

    def test_reports_failure_when_server_stays_enabled(self, token_file):
        """A toggle that leaves the server enabled is a failure."""
        client = _client(enabled_before=True, enabled_after=True)
        with patch.object(scanner, "RegistryClient", return_value=client):
            result = scanner._disable_unsafe_server("/srv", REGISTRY_URL, token_file)

        assert result is False

    def test_reports_failure_on_registry_error(self, token_file):
        """An API error is reported as a failed disable."""
        with patch.object(scanner, "RegistryClient", side_effect=RuntimeError("401")):
            result = scanner._disable_unsafe_server("/srv", REGISTRY_URL, token_file)

        assert result is False

    def test_reports_failure_without_a_token(self, tmp_path):
        """A token file with no access token is a failed disable, not a crash."""
        path = tmp_path / "empty.json"
        path.write_text("{}")

        result = scanner._disable_unsafe_server("/srv", REGISTRY_URL, str(path))

        assert result is False


class TestLoadRegistryToken:
    """Tests for _load_registry_token."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"access_token": "flat-token"},
            {"tokens": {"access_token": "flat-token"}},
        ],
    )
    def test_reads_both_token_file_layouts(self, tmp_path, payload):
        """Top-level and nested access tokens are both accepted."""
        path = tmp_path / "token.json"
        path.write_text(json.dumps(payload))

        assert scanner._load_registry_token(str(path)) == "flat-token"
