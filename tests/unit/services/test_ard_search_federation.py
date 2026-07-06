"""Unit tests for ARD search federation modes (issue #1296, Phase 3).

Exercises none/auto/referrals filtering and per-result source attribution over a
unified index whose hits include local and synced/ingested (origin-tagged) items.
"""

from unittest.mock import AsyncMock, patch

from registry.schemas.ard_models import ArdReferral, ArdSearchResult
from registry.services import ard_search_service as s

_SOURCE_URI = "http://reg.example.com/api/ard/search"

# origin_map: known foreign origin ids -> their origin URLs.
_ORIGIN_MAP = {
    "acme": "https://acme.com/.well-known/ai-catalog.json",  # ingested catalog
    "peerb": "https://peerb.example/api/ard",                 # peer registry
}
_REFERRALS = [
    ArdReferral(identifier="urn:air:peerb.example:registry:self",
                type="application/ai-registry+json", url="https://peerb.example/api/ard"),
]

# Three server hits: one local, one ingested (acme), one peer (peerb).
_RAW = {
    "servers": [
        {"path": "/local-srv", "server_name": "Local", "tags": [], "relevance_score": 0.9},
        {"path": "/acme/github", "server_name": "AcmeGH", "tags": [], "relevance_score": 0.8},
        {"path": "/peerb/foo", "server_name": "PeerFoo", "tags": [], "relevance_score": 0.7},
    ],
    "agents": [],
    "skills": [],
}


async def _run(federation):
    repo = AsyncMock()
    repo.search = AsyncMock(return_value=_RAW)
    with (
        patch.object(s, "get_search_repository", return_value=repo),
        patch.object(s, "user_can_access_server", AsyncMock(return_value=True)),
        patch.object(s, "_resolve_publisher_domain", return_value="reg.example.com"),
        patch.object(s, "_build_origin_map", AsyncMock(return_value=(_ORIGIN_MAP, _REFERRALS))),
    ):
        return await s.search_and_scope(
            "q", None, None, 10, {"username": "u"}, _SOURCE_URI, federation=federation
        )


class TestFederationModes:
    async def test_none_returns_local_only(self):
        results, _scoped, referrals = await _run("none")
        names = {r.display_name for r in results}
        assert names == {"Local"}
        assert all(r.source == _SOURCE_URI for r in results)
        assert referrals == []

    async def test_auto_returns_all_source_tagged(self):
        results, _scoped, referrals = await _run("auto")
        by_name = {r.display_name: r for r in results}
        assert set(by_name) == {"Local", "AcmeGH", "PeerFoo"}
        assert by_name["Local"].source == _SOURCE_URI
        assert by_name["AcmeGH"].source == _ORIGIN_MAP["acme"]
        assert by_name["PeerFoo"].source == _ORIGIN_MAP["peerb"]
        assert referrals == []

    async def test_referrals_local_plus_peer_pointers(self):
        results, _scoped, referrals = await _run("referrals")
        names = {r.display_name for r in results}
        assert names == {"Local"}
        assert len(referrals) == 1
        assert referrals[0].type == "application/ai-registry+json"
        assert referrals[0].url == "https://peerb.example/api/ard"


class TestOverrideSourceDescriptor:
    async def test_rewrites_url_and_identifier_from_source(self):
        from unittest.mock import AsyncMock

        from registry.schemas.ard_models import ArdCatalogEntry

        r = ArdSearchResult(
            **ArdCatalogEntry(
                identifier="urn:air:local:server:acme-github", display_name="GH",
                type="application/mcp-server-card+json",
                url="http://reg.example.com/api/public/servers/github/server.json",
            ).model_dump(by_alias=True, exclude_none=True),
            score=90, source="https://acme.com/.well-known/ai-catalog.json",
        )
        repo = AsyncMock()
        repo.find_with_filter = AsyncMock(return_value={
            "/acme/github": {
                "ard_source_url": "https://acme.com/api/public/servers/github/server.json",
                "ard_source_identifier": "urn:air:acme.com:server:github",
            }
        })
        await s._override_source_descriptor([(r, "/acme/github")], repo)
        assert r.url == "https://acme.com/api/public/servers/github/server.json"
        assert r.identifier == "urn:air:acme.com:server:github"

    async def test_empty_list_is_noop(self):
        from unittest.mock import AsyncMock

        repo = AsyncMock()
        await s._override_source_descriptor([], repo)
        repo.find_with_filter.assert_not_called()


class TestOriginId:
    def test_known_prefix_is_foreign(self):
        assert s._origin_id("/acme/github", {"acme"}) == "acme"

    def test_unknown_prefix_is_local(self):
        assert s._origin_id("/acme/github", set()) is None

    def test_single_segment_is_local(self):
        assert s._origin_id("/local-srv", {"acme"}) is None
