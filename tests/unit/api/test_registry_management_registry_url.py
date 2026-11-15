"""Where registry_management.py sends requests: --registry-url, then REGISTRY_URL, then localhost.

--registry-url used to default to http://localhost, so the flag always had a
value and REGISTRY_URL from the environment was silently ignored: a command
meant for a remote deployment went to the local stack instead.
"""

import argparse
import sys
from pathlib import Path

import pytest

_API_DIR = Path(__file__).resolve().parents[3] / "api"
sys.path.insert(0, str(_API_DIR))

import registry_management  # noqa: E402

REMOTE: str = "https://registry.example.com"


class _RecordingClient:
    """Stands in for RegistryClient and records the URL the CLI built it with."""

    urls: list[str] = []

    def __init__(self, registry_url: str, token: str) -> None:
        _RecordingClient.urls.append(registry_url)

    def list_openapi_servers(self) -> dict:
        return {"servers": []}


def test_cli_uses_registry_url_from_the_environment_without_the_flag(monkeypatch, tmp_path):
    """End to end through main(): the real --registry-url flag must not shadow the env var."""
    token_file = tmp_path / "token.json"
    token_file.write_text('{"access_token": "test-token-value"}')
    monkeypatch.setenv("REGISTRY_URL", REMOTE)
    monkeypatch.setattr(registry_management, "RegistryClient", _RecordingClient)
    monkeypatch.setattr(
        sys,
        "argv",
        ["registry_management.py", "--token-file", str(token_file), "openapi-server-list"],
    )
    _RecordingClient.urls = []

    exit_code = registry_management.main()

    assert (exit_code, _RecordingClient.urls) == (0, [REMOTE])


def test_environment_is_used_when_the_flag_is_absent(monkeypatch):
    monkeypatch.setenv("REGISTRY_URL", REMOTE)

    assert registry_management._get_registry_url(None) == REMOTE


def test_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("REGISTRY_URL", "https://other.example.com")

    assert registry_management._get_registry_url(REMOTE) == REMOTE


def test_local_default_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("REGISTRY_URL", raising=False)

    assert registry_management._get_registry_url(None) == registry_management.DEFAULT_REGISTRY_URL


@pytest.mark.parametrize("env_value", [REMOTE, None])
def test_client_is_built_for_the_resolved_url(monkeypatch, tmp_path, env_value):
    if env_value:
        monkeypatch.setenv("REGISTRY_URL", env_value)
    else:
        monkeypatch.delenv("REGISTRY_URL", raising=False)
    token_file = tmp_path / "token.json"
    token_file.write_text('{"access_token": "test-token-value"}')
    args = argparse.Namespace(registry_url=None, token_file=str(token_file))

    client = registry_management._create_client(args)

    assert client.registry_url.rstrip("/") == (
        env_value or registry_management.DEFAULT_REGISTRY_URL
    )
