"""Unit tests for the ARD ingestion service mapping/trust gating (issue #1296)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from registry.schemas.ard_models import (
    AICatalogManifest,
    ArdCatalogEntry,
    ArdHost,
    ArdTrustManifest,
)
from registry.schemas.federation_schema import AiCatalogFederationConfig, AiCatalogSourceConfig
from registry.services import ard_ingestion_service as ingest
from registry.services.ard_ingestion_service import get_ard_ingestion_service


def _manifest(entries, identity="https://acme.com"):
    return AICatalogManifest(
        host=ArdHost(display_name="Acme",
                     trust_manifest=ArdTrustManifest(identity=identity, identity_type="https")),
        entries=entries,
    )


def _entry(identifier, type_="application/mcp-server-card+json"):
    return ArdCatalogEntry(identifier=identifier, display_name="X", type=type_, url="https://acme.com/x")


_SRC = AiCatalogSourceConfig(source_id="acme", domain="acme.com")


class TestMapDocuments:
    def test_accepts_matching_publisher(self):
        svc = get_ard_ingestion_service()
        cfg = AiCatalogFederationConfig(trust_enforcement="reject")
        docs = [(_manifest([_entry("urn:air:acme.com:server:github")]), "https://acme.com/x")]
        servers, agents, skills, rejected = svc._map_documents(docs, _SRC, cfg)
        assert len(servers) == 1
        assert servers[0]["path"] == "/github"
        assert rejected == 0

    def test_rejects_publisher_mismatch_under_reject(self):
        svc = get_ard_ingestion_service()
        cfg = AiCatalogFederationConfig(trust_enforcement="reject")
        docs = [(_manifest([_entry("urn:air:victim.com:server:x")]), "https://acme.com/x")]
        servers, _agents, _skills, rejected = svc._map_documents(docs, _SRC, cfg)
        assert servers == []
        assert rejected == 1

    def test_flag_policy_accepts_with_flag(self):
        svc = get_ard_ingestion_service()
        cfg = AiCatalogFederationConfig(trust_enforcement="flag")
        docs = [(_manifest([_entry("urn:air:victim.com:server:x")]), "https://acme.com/x")]
        servers, _agents, _skills, rejected = svc._map_documents(docs, _SRC, cfg)
        assert len(servers) == 1
        assert servers[0]["trust_flag"]
        assert rejected == 0

    def test_separates_servers_agents_and_skills(self):
        svc = get_ard_ingestion_service()
        cfg = AiCatalogFederationConfig(trust_enforcement="off")
        entries = [
            _entry("urn:air:acme.com:server:a", "application/mcp-server-card+json"),
            _entry("urn:air:acme.com:agent:b", "application/a2a-agent-card+json"),
            _entry("urn:air:acme.com:skill:c", "application/ai-skill"),
            _entry("urn:air:acme.com:registry:self", "application/ai-registry+json"),
        ]
        servers, agents, skills, _rejected = svc._map_documents([(_manifest(entries), "u")], _SRC, cfg)
        assert len(servers) == 1
        assert len(agents) == 1
        assert len(skills) == 1  # skill mapped to a SkillCard dict; registry entry ignored
        assert skills[0]["path"] == "/skills/acme/c"
        assert skills[0]["registry_name"] == "acme"


class TestSkillStorage:
    async def test_store_skills_creates_each(self):
        svc = get_ard_ingestion_service()
        repo = AsyncMock()
        skills = [
            {"path": "/skills/acme/a", "name": "acme-a", "description": "d",
             "skill_md_url": "https://acme.com/a", "registry_name": "acme"},
        ]
        with patch.object(ingest, "get_skill_repository", return_value=repo):
            stored = await svc._store_skills(skills)
        assert stored == 1
        repo.create.assert_awaited_once()

    async def test_reconcile_removes_orphans_for_source(self):
        svc = get_ard_ingestion_service()
        repo = AsyncMock()
        repo.list_filtered = AsyncMock(return_value=[
            SimpleNamespace(path="/skills/acme/keep"),
            SimpleNamespace(path="/skills/acme/gone"),
        ])
        with patch.object(ingest, "get_skill_repository", return_value=repo):
            removed = await svc._reconcile_skill_orphans("acme", {"/skills/acme/keep"})
        assert removed == 1
        repo.delete.assert_awaited_once_with("/skills/acme/gone")
