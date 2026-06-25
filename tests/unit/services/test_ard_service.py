"""Unit tests for the ARD catalog service (build_catalog with mocked repos)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jsonschema
import pytest

from registry.services import ard_service

# The authoritative ARD schema (ards-project/ard-spec), vendored into the
# committed test fixtures so it is available in CI.
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "ard"
    / "ai-catalog.schema.json"
)


def _fake_request(host="registry.example.com", proto="https"):
    return SimpleNamespace(
        headers={"host": host, "x-forwarded-proto": proto},
        url=SimpleNamespace(scheme=proto),
    )


def _skill(path, name, tags=None):
    return SimpleNamespace(
        path=path,
        name=name,
        description="A skill.",
        tags=tags or ["docs"],
        allowed_tools=[SimpleNamespace(tool_name="Read")],
        version=None,
        updated_at="2026-06-01T00:00:00",
    )


class TestResolvePublisherDomain:
    """Tests for _resolve_publisher_domain."""

    def test_explicit_wins(self):
        with patch.object(ard_service.settings, "ard_publisher_domain", "acme.com"):
            assert ard_service._resolve_publisher_domain() == "acme.com"

    def test_derives_from_registry_url(self):
        with (
            patch.object(ard_service.settings, "ard_publisher_domain", ""),
            patch.object(ard_service.settings, "registry_url", "https://reg.example.org"),
        ):
            assert ard_service._resolve_publisher_domain() == "reg.example.org"

    def test_localhost_falls_back_to_placeholder(self):
        with (
            patch.object(ard_service.settings, "ard_publisher_domain", ""),
            patch.object(ard_service.settings, "registry_url", "http://localhost:8000"),
        ):
            assert ard_service._resolve_publisher_domain() == "example.com"


@pytest.mark.asyncio
class TestBuildCatalog:
    """Tests for build_catalog assembly."""

    async def _build_with(self, servers, agents, skills):
        server_repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=servers))
        agent_repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=agents))
        skill_repo = SimpleNamespace(list_filtered=AsyncMock(return_value=skills))
        with (
            patch.object(ard_service, "get_server_repository", return_value=server_repo),
            patch.object(ard_service, "get_agent_repository", return_value=agent_repo),
            patch.object(ard_service, "get_skill_repository", return_value=skill_repo),
            patch.object(ard_service.settings, "ard_publisher_domain", "registry.example.com"),
            patch.object(ard_service.settings, "registry_name", "Test Registry"),
            # Pin an HTTPS registry_url so the host trust-manifest identity is
            # deterministic. Without this the test inherits the ambient default
            # (http://localhost:8000), which fails the Phase-1 https identity
            # assertion in environments that do not override REGISTRY_URL.
            patch.object(ard_service.settings, "registry_url", "https://registry.example.com"),
            # Gate off the Phase 2 self ai-registry entry so these publisher-mapping
            # tests assert only on the server/agent/skill entries. The self-entry is
            # covered by test_includes_registry_self_entry_when_enabled.
            patch.object(ard_service.settings, "ard_registry_enabled", False),
        ):
            manifest = await ard_service.build_catalog(_fake_request())
        return manifest, server_repo, agent_repo, skill_repo

    async def test_includes_registry_self_entry_when_enabled(self):
        server_repo = SimpleNamespace(find_with_filter=AsyncMock(return_value={}))
        agent_repo = SimpleNamespace(find_with_filter=AsyncMock(return_value={}))
        skill_repo = SimpleNamespace(list_filtered=AsyncMock(return_value=[]))
        with (
            patch.object(ard_service, "get_server_repository", return_value=server_repo),
            patch.object(ard_service, "get_agent_repository", return_value=agent_repo),
            patch.object(ard_service, "get_skill_repository", return_value=skill_repo),
            patch.object(ard_service.settings, "ard_publisher_domain", "registry.example.com"),
            patch.object(ard_service.settings, "registry_url", "https://registry.example.com"),
            patch.object(ard_service.settings, "ard_registry_enabled", True),
        ):
            manifest = await ard_service.build_catalog(_fake_request())
        registry_entries = [
            e for e in manifest.entries if e.type == "application/ai-registry+json"
        ]
        assert len(registry_entries) == 1
        assert registry_entries[0].identifier == "urn:air:registry.example.com:registry:self"
        assert registry_entries[0].url.endswith("/api/ard")

    async def test_three_bulk_reads_only(self):
        manifest, sr, ar, kr = await self._build_with({}, {}, [])
        sr.find_with_filter.assert_awaited_once()
        ar.find_with_filter.assert_awaited_once()
        kr.list_filtered.assert_awaited_once()
        assert manifest.entries == []

    async def test_entries_from_all_three_types(self):
        servers = {"/github/": {"server_name": "GitHub", "tags": ["vcs"]}}
        agents = {"/agents/trav": {"name": "Trav", "tags": ["travel"]}}
        skills = [_skill("/skills/pdf", "pdf")]
        manifest, *_ = await self._build_with(servers, agents, skills)
        types = {e.type for e in manifest.entries}
        assert types == {
            "application/mcp-server-card+json",
            "application/a2a-agent-card+json",
            "application/ai-skill",
        }

    async def test_one_bad_record_does_not_fail(self):
        # An unsanitizable server name is skipped, others still rendered.
        servers = {
            "/!!!/": {"server_name": "bad"},
            "/good/": {"server_name": "Good", "tags": ["x"]},
        }
        manifest, *_ = await self._build_with(servers, {}, [])
        idents = [e.identifier for e in manifest.entries]
        assert idents == ["urn:air:registry.example.com:server:good"]

    async def test_manifest_validates_against_schema(self):
        servers = {"/github/": {"server_name": "GitHub", "tags": ["vcs", "git"]}}
        agents = {"/agents/trav": {"name": "Trav", "tags": ["travel", "trip"]}}
        skills = [_skill("/skills/pdf", "pdf", tags=["pdf", "docs"])]
        manifest, *_ = await self._build_with(servers, agents, skills)
        payload = manifest.model_dump(by_alias=True, exclude_none=True)
        schema = json.loads(_SCHEMA_PATH.read_text())
        jsonschema.validate(instance=payload, schema=schema)  # raises on failure

    async def test_host_has_only_allowed_keys(self):
        manifest, *_ = await self._build_with({}, {}, [])
        host = manifest.host.model_dump(by_alias=True, exclude_none=True)
        allowed = {
            "displayName",
            "identifier",
            "documentationUrl",
            "logoUrl",
            "trustManifest",
        }
        assert set(host).issubset(allowed)

    async def test_host_omits_did_web_identifier_in_phase1(self):
        # Phase 1 publishes no key material, so an unresolvable did:web would be
        # a false claim. Identity is asserted via the https trust manifest only.
        manifest, *_ = await self._build_with({}, {}, [])
        host = manifest.host.model_dump(by_alias=True, exclude_none=True)
        assert "identifier" not in host
        assert host["trustManifest"]["identityType"] == "https"
        assert host["trustManifest"]["identity"].startswith("https://")
