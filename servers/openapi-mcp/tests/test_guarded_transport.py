"""Tests for the guarded httpx2 transport and the guard-only settings."""

import re
from pathlib import Path

import guarded_transport
import httpx
import httpx2
import pytest
from guarded_transport import (
    BackendResponseTooLargeError,
    BackendUrlBlockedError,
    GuardedHttpx2Transport,
)
from settings import GuardSettings

URL_GUARD_SOURCE: Path = Path(__file__).resolve().parents[3] / "registry" / "utils" / "url_guard.py"
SETTINGS_READ_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?:_get_settings\(\)|\b_settings|\bs)\.([a-z][a-z0-9]*_[a-z0-9_]+)"),
    re.compile(r"getattr\(\s*(?:_get_settings\(\)|_settings|s)\s*,\s*\"([a-z0-9_]+)\""),
)


def _client(
    max_bytes: int = 1_000_000,
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=GuardedHttpx2Transport(max_response_bytes=max_bytes))


class TestRealGuard:
    """The canonical guard runs on every backend call."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8080/x",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/x",
            "http://100.100.100.200/x",
            "http://[::ffff:169.254.169.254]/x",
            "http://mcpgw-server:8003/mcp",
        ],
    )
    async def test_private_and_metadata_targets_blocked(self, url):
        async with _client() as client:
            with pytest.raises(BackendUrlBlockedError):
                await client.get(url)


class TestResponseHandling:
    """Size cap and response copying, with a mock backend in place of the guard."""

    @pytest.fixture
    def big_backend(self, monkeypatch):
        def _factory(**kwargs):
            return httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 5000))

        monkeypatch.setattr(guarded_transport.url_guard, "GuardedAsyncTransport", _factory)

    async def test_response_over_cap_is_an_error(self, big_backend):
        async with _client(max_bytes=4096) as client:
            with pytest.raises(BackendResponseTooLargeError):
                await client.get("https://api.example.com/big")

    async def test_response_under_cap_is_returned(self, big_backend):
        async with _client(max_bytes=10_000) as client:
            response = await client.get("https://api.example.com/big")

        assert (response.status_code, len(response.content)) == (200, 5000)


class TestGuardSettingsCoverage:
    """GuardSettings must define every settings field url_guard reads."""

    def test_every_guard_setting_is_defined(self):
        source = URL_GUARD_SOURCE.read_text()
        read_fields: set[str] = set()
        for pattern in SETTINGS_READ_PATTERNS:
            read_fields |= set(pattern.findall(source))

        missing = sorted(read_fields - set(GuardSettings.model_fields))

        assert read_fields, "found no settings reads; the patterns no longer match url_guard.py"
        assert missing == []
