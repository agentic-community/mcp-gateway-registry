"""Unit tests for the ARD ai-catalog crawler client (issue #1296)."""

import json
from unittest.mock import MagicMock, patch

import httpx

from registry.services.federation import ai_catalog_client as c
from registry.utils import url_guard


def _manifest_payload(entries):
    return {
        "specVersion": "1.0",
        "host": {
            "displayName": "Acme",
            "trustManifest": {"identity": "https://acme.com", "identityType": "https"},
        },
        "entries": entries,
    }


def _server_entry(name):
    return {
        "identifier": f"urn:air:acme.com:server:{name}",
        "displayName": name,
        "type": "application/mcp-server-card+json",
        "url": f"https://acme.com/{name}",
    }


def _catalog_entry(url):
    return {
        "identifier": "urn:air:acme.com:catalog:child",
        "displayName": "Child",
        "type": "application/ai-catalog+json",
        "url": url,
    }


def _fake_stream(payload=None, *, oversize=False, content_length=None):
    """Build a mock for ``client.stream(...)`` (a context manager yielding a
    streaming response with iter_bytes()/headers/raise_for_status)."""
    body = b"x" * (c._MAX_BYTES + 1) if oversize else json.dumps(payload).encode()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.status_code = 200
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
            patch.object(
                client.client,
                "stream",
                return_value=_fake_stream(_manifest_payload([_server_entry("github")])),
            ),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 1
        manifest, _uri = docs[0]
        assert manifest.entries[0].identifier == "urn:air:acme.com:server:github"

    def test_recurses_nested_catalog_within_depth(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=2)
        root = _manifest_payload(
            [_catalog_entry("https://acme.com/child.json"), _server_entry("a")]
        )
        child = _manifest_payload([_server_entry("b")])
        responses = {
            "https://acme.com/.well-known/ai-catalog.json": lambda: _fake_stream(root),
            "https://acme.com/child.json": lambda: _fake_stream(child),
        }
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(
                client.client, "stream", side_effect=lambda method, url, **kw: responses[url]()
            ),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")
        assert len(docs) == 2  # root + child

    def test_loop_guard_dedupes_visited(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=5)
        root = _manifest_payload([_catalog_entry("https://acme.com/.well-known/ai-catalog.json")])
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(
                client.client, "stream", side_effect=lambda method, url, **kw: _fake_stream(root)
            ),
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
        cm = _fake_stream(
            _manifest_payload([_server_entry("a")]), content_length=str(c._MAX_BYTES + 1)
        )
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


def _resolve_to(*ips: str):
    """getaddrinfo stub resolving any host to the given IP(s)."""

    def _stub(host, port, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


class TestRebindingDefeat:
    """The catalog fetch must connect to the IP the guard validated, not a
    rebound one — the fetch goes through the pinned guarded client so a
    DNS-rebind between validation and connect is blocked at connect time."""

    def test_client_uses_guarded_transport(self):
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        transport = client.client._transport
        assert isinstance(transport, url_guard.GuardedTransport)
        # No auth is ever configured on the crawler client.
        assert client.client.headers.get("authorization") is None
        # Redirects are disabled so a 3xx cannot bounce off to a new host.
        assert client.client.follow_redirects is False

    def test_rebind_between_validate_and_connect_is_defeated(self):
        """A host that passes assert_fetchable() (public) but rebinds to a
        private IP before the connect is blocked by the pinned transport, and
        the crawl skips it rather than crashing."""
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        url = "https://rebind.example/.well-known/ai-catalog.json"

        settings_stub = MagicMock()
        settings_stub.github_extra_hosts = ""
        settings_stub.ssrf_allowed_hosts = ""
        settings_stub.ssrf_allowed_cidrs = ""

        with (
            # assert_fetchable succeeds (host validated public out-of-band).
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(url_guard, "settings", settings_stub),
            # The host now rebinds to a private IP; the pinned transport
            # re-resolves inside the connect and blocks it.
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to("10.0.0.5")),
        ):
            url_guard._skill_allowlist.cache_clear()
            url_guard._proxy_allowlist.cache_clear()
            docs = client.fetch_catalog(url)

        # Skipped, not fatal: the guard blocked the rebound connect.
        assert docs == []

    def test_public_ip_still_fetches(self):
        """A host that validates public and stays public is fetched normally
        (the pinned transport rewrites to the validated IP, no block)."""
        client = c.AiCatalogFederationClient(polite_interval_ms=0)
        payload = _manifest_payload([_server_entry("github")])

        settings_stub = MagicMock()
        settings_stub.github_extra_hosts = ""
        settings_stub.ssrf_allowed_hosts = ""
        settings_stub.ssrf_allowed_cidrs = ""

        # Confirm the pinned transport accepts a public IP and rewrites to it,
        # then let the (mocked) stream return the manifest body.
        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(url_guard, "settings", settings_stub),
            patch.object(url_guard.socket, "getaddrinfo", _resolve_to("93.184.216.34")),
        ):
            url_guard._skill_allowlist.cache_clear()
            url_guard._proxy_allowlist.cache_clear()
            transport = client.client._transport
            request = httpx.Request("GET", "https://acme.com/x.json")
            pinned = transport._pin_request(request)
            assert pinned.url.host == "93.184.216.34"
            assert pinned.headers["Host"] == "acme.com"

            with patch.object(client.client, "stream", return_value=_fake_stream(payload)):
                docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")

        assert len(docs) == 1


class TestCrawlResilienceAndRedirects:
    def test_unexpected_error_on_one_url_does_not_crash_crawl(self):
        """A poisoned nested URL that raises an unexpected error (e.g. an
        httpx.InvalidURL that is not an HTTPError) must skip that node, not kill
        the whole crawl — a hostile catalog cannot DoS ingestion with one link."""
        client = c.AiCatalogFederationClient(polite_interval_ms=0, max_depth=2)
        root = _manifest_payload(
            [_catalog_entry("https://acme.com/bad.json"), _server_entry("good")]
        )

        def _stream(method, url, **kw):
            if url == "https://acme.com/.well-known/ai-catalog.json":
                return _fake_stream(root)
            # The nested child raises a non-HTTPError, non-UrlValidationError.
            raise RuntimeError("boom on nested fetch")

        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", side_effect=_stream),
        ):
            docs = client.fetch_catalog("https://acme.com/.well-known/ai-catalog.json")

        # Root still processed despite the child blowing up.
        assert len(docs) == 1
        assert docs[0][0].entries[0].identifier.endswith("child")

    def test_redirect_response_is_rejected(self):
        """With follow_redirects=False, a 3xx response must be refused rather
        than have its body parsed as a catalog (redirect-body injection)."""
        client = c.AiCatalogFederationClient(polite_interval_ms=0)

        resp = MagicMock()
        resp.raise_for_status = MagicMock()  # httpx does not raise on 3xx
        resp.status_code = 302
        resp.headers = {}
        resp.iter_bytes = MagicMock(return_value=[b'{"specVersion": "1.0"}'])
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=resp)
        cm.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(c, "assert_fetchable", side_effect=lambda u, d=None: u),
            patch.object(client.client, "stream", return_value=cm),
        ):
            docs = client.fetch_catalog("https://acme.com/redir.json")

        assert docs == []
