"""Unit tests for the ARD ai-catalog crawler client (issue #1296)."""

import json
from unittest.mock import MagicMock, patch

from registry.services.federation import ai_catalog_client as c


def _manifest_payload(entries):
    return {
        "specVersion": "1.0",
        "host": {"displayName": "Acme", "trustManifest": {"identity": "https://acme.com", "identityType": "https"}},
        "entries": entries,
    }


def _server_entry(name):
    return {
        "identifier": f"urn:air:acme.com:server:{name}", "displayName": name,
        "type": "application/mcp-server-card+json", "url": f"https://acme.com/{name}",
    }


def _catalog_entry(url):
    return {
        "identifier": "urn:air:acme.com:catalog:child", "displayName": "Child",
        "type": "application/ai-catalog+json", "url": url,
    }


def _fake_stream(payload=None, *, oversize=False, content_length=None):
    """Build a mock for ``client.stream(...)`` (a context manager yielding a
    streaming response with iter_bytes()/headers/raise_for_status)."""
    body = b"x" * (c._MAX_BYTES + 1) if oversize else json.dumps(payload).encode()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": content_length} if content_length else {}
    resp.iter_bytes = MagicMock(return_value=[body])
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestFetchCatalog:
    def test_fetches_root_and_validates(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream",
                         return_value=_fake_stream(_manifest_payload([_server_entry("github")]))),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 1
        manifest, _uri = docs[0]
        assert manifest.entries[0].identifier == "urn:air:acme.com:server:github"

    def test_recurses_nested_catalog_within_depth(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=2)
        root = _manifest_payload([_catalog_entry("https://acme.com/child.json"), _server_entry("a")])
        child = _manifest_payload([_server_entry("b")])
        responses = {
            "https://acme.com/.well-known/ai-catalog.json": lambda: _fake_stream(root),
            "https://acme.com/child.json": lambda: _fake_stream(child),
        }
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", side_effect=lambda method, url, **kw: responses[url]()),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 2  # root + child

    def test_loop_guard_dedupes_visited(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=5)
        root = _manifest_payload([_catalog_entry("https://acme.com/.well-known/ai-catalog.json")])
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", side_effect=lambda method, url, **kw: _fake_stream(root)),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 1  # visited set prevents re-fetch

    def test_oversized_body_aborted(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", return_value=_fake_stream(oversize=True)),
        ):
            docs = client.fetch_catalog("https://acme.com/x.json")
        assert docs == []

    def test_oversized_content_length_rejected_early(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        cm = _fake_stream(_manifest_payload([_server_entry("a")]), content_length=str(c._MAX_BYTES + 1))
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", return_value=cm),
        ):
            docs = client.fetch_catalog("https://acme.com/x.json")
        assert docs == []

    def test_blocked_url_skipped(self):
        from registry.services.ard_search_service import ArdValidationError

        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        with patch.object(c, "assert_fetchable", side_effect=ArdValidationError("blocked")):
            docs = client.fetch_catalog("https://evil.com/x.json")
        assert docs == []
