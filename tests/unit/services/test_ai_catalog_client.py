"""Unit tests for the ARD ai-catalog crawler client (issue #1296)."""

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


def _fake_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = b"x" * 100
    resp.json = MagicMock(return_value=payload)
    return resp


class TestFetchCatalog:
    def test_fetches_root_and_validates(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "get",
                         return_value=_fake_response(_manifest_payload([_server_entry("github")]))),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 1
        manifest, uri = docs[0]
        assert manifest.entries[0].identifier == "urn:air:acme.com:server:github"

    def test_recurses_nested_catalog_within_depth(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=2)
        root = _manifest_payload([_catalog_entry("https://acme.com/child.json"), _server_entry("a")])
        child = _manifest_payload([_server_entry("b")])
        responses = {
            "https://acme.com/.well-known/ai-catalog.json": _fake_response(root),
            "https://acme.com/child.json": _fake_response(child),
        }
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "get", side_effect=lambda url, **kw: responses[url]),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 2  # root + child

    def test_loop_guard_dedupes_visited(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=5)
        # Root points to itself -> must not loop forever.
        root = _manifest_payload([_catalog_entry("https://acme.com/.well-known/ai-catalog.json")])
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "get", return_value=_fake_response(root)),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 1  # visited set prevents re-fetch

    def test_oversized_document_skipped(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        big = _fake_response(_manifest_payload([_server_entry("a")]))
        big.content = b"x" * (c._MAX_BYTES + 1)
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "get", return_value=big),
        ):
            docs = client.fetch_catalog("https://acme.com/x.json")
        assert docs == []

    def test_blocked_url_skipped(self):
        from registry.services.ard_search_service import ArdValidationError

        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        with patch.object(c, "assert_fetchable", side_effect=ArdValidationError("blocked")):
            docs = client.fetch_catalog("https://evil.com/x.json")
        assert docs == []
