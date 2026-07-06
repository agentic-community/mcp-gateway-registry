"""ARD ai-catalog.json ingestion service (issue #1296, Phase 3).

Reads the enabled ``ai_catalog`` sources from the DB-backed federation config,
crawls each source's catalog (``AiCatalogFederationClient``), applies the
domain-anchored trust gate, reverse-maps accepted entries to internal records,
and stores them by **reusing the peer-federation storage + orphan machinery**
(treating each ``source_id`` as a peer_id). That reuse means ingested items are
indexed for semantic search, origin-tagged via ``sync_metadata`` (so the search
federation filter and the UI "federated" badge work unchanged), and orphan-
reconciled exactly like peer-synced items.

A per-source in-process lock prevents overlapping runs within a replica; the
scheduler should run on a single replica (parity with peer sync). Run state
(last run, generation, counts, failures) is kept in memory and surfaced via
``get_status`` for the admin endpoint/CLI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from ..observability.meters import (
    ard_ingestion_duration_ms,
    ard_ingestion_entries_total,
    ard_ingestion_runs_total,
    ard_trust_mismatch_total,
)
from ..repositories.factory import get_federation_config_repository, get_skill_repository
from ..schemas.federation_schema import AiCatalogFederationConfig, AiCatalogSourceConfig
from ..schemas.peer_federation_schema import SyncResult
from ..schemas.skill_models import SkillCard
from .ard_ingest_mapping import entry_to_record, entry_to_skill_data
from .ard_trust import host_identity_domain, verify_entry_trust
from .federation.ai_catalog_client import AiCatalogFederationClient
from .peer_federation_service import get_peer_federation_service

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_ID = "default"


class ArdIngestionService:
    """Singleton orchestrator for ARD ai-catalog.json ingestion."""

    _instance: ArdIngestionService | None = None

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._state: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> ArdIngestionService:
        if cls._instance is None:
            cls._instance = ArdIngestionService()
        return cls._instance

    def _lock_for(self, source_id: str) -> asyncio.Lock:
        if source_id not in self._locks:
            self._locks[source_id] = asyncio.Lock()
        return self._locks[source_id]

    async def get_config(self) -> AiCatalogFederationConfig:
        """Return the ai_catalog block from the DB federation config (or defaults)."""
        repo = get_federation_config_repository()
        config = await repo.get_config(_DEFAULT_CONFIG_ID)
        if config is None:
            return AiCatalogFederationConfig()
        return config.ai_catalog

    def get_status(self) -> list[dict[str, Any]]:
        """Return per-source ingestion state for the admin endpoint/CLI."""
        return [{"source_id": sid, **state} for sid, state in self._state.items()]

    async def ingest_all(self) -> list[SyncResult]:
        """Ingest every enabled source in the current ai_catalog config."""
        cfg = await self.get_config()
        if not cfg.enabled:
            logger.info("ARD ingestion is disabled; skipping ingest_all")
            return []
        results: list[SyncResult] = []
        for source in cfg.sources:
            results.append(await self.ingest_source(source, cfg))
        return results

    async def ingest_source(
        self,
        source: AiCatalogSourceConfig,
        cfg: AiCatalogFederationConfig,
    ) -> SyncResult:
        """Ingest a single source. Never raises; failures return a failed SyncResult."""
        lock = self._lock_for(source.source_id)
        if lock.locked():
            logger.info("ARD ingestion for source %s already in progress; skipping", source.source_id)
            return SyncResult(
                success=False,
                peer_id=source.source_id,
                error_message="ingestion already in progress",
                new_generation=self._state.get(source.source_id, {}).get("generation", 0),
            )
        async with lock:
            return await self._ingest_locked(source, cfg)

    async def _ingest_locked(
        self,
        source: AiCatalogSourceConfig,
        cfg: AiCatalogFederationConfig,
    ) -> SyncResult:
        start = time.time()
        prev = self._state.get(source.source_id, {})
        generation = int(prev.get("generation", 0)) + 1
        peer_svc = get_peer_federation_service()

        try:
            client = AiCatalogFederationClient(
                timeout_seconds=cfg.fetch_timeout_seconds,
                max_depth=cfg.max_depth,
                polite_interval_ms=cfg.polite_interval_ms,
                same_domain_only=cfg.same_domain_only,
            )
            # httpx is synchronous; run the whole crawl off the event loop.
            docs = await asyncio.to_thread(client.fetch_catalog, source.resolve_uri())

            servers, agents, skills, rejected = self._map_documents(docs, source, cfg)

            servers_stored = await peer_svc._store_synced_servers(source.source_id, servers)
            agents_stored = await peer_svc._store_synced_agents(source.source_id, agents)
            skills_stored = await self._store_skills(skills)

            orphaned_servers, orphaned_agents = await peer_svc.detect_orphaned_items(
                source.source_id,
                [s["path"] for s in servers],
                [a["path"] for a in agents],
            )
            if orphaned_servers or orphaned_agents:
                await peer_svc.handle_orphaned_items(
                    source.source_id, orphaned_servers, orphaned_agents, action="mark"
                )
            skills_orphaned = await self._reconcile_skill_orphans(
                source.source_id, {s["path"] for s in skills}
            )

            duration = time.time() - start
            sid = source.source_id
            ard_ingestion_runs_total.add(1, {"source_id": sid, "status": "success"})
            ard_ingestion_duration_ms.record(duration * 1000.0, {"source_id": sid})
            indexed = servers_stored + agents_stored + skills_stored
            ard_ingestion_entries_total.add(indexed, {"source_id": sid, "outcome": "indexed"})
            if rejected:
                ard_ingestion_entries_total.add(rejected, {"source_id": sid, "outcome": "rejected"})
                ard_trust_mismatch_total.add(rejected, {"source_id": sid, "policy": cfg.trust_enforcement})
            orphaned_total = len(orphaned_servers) + len(orphaned_agents) + skills_orphaned
            if orphaned_total:
                ard_ingestion_entries_total.add(orphaned_total, {"source_id": sid, "outcome": "orphaned"})
            self._state[source.source_id] = {
                "generation": generation,
                "last_synced_at": datetime.now(UTC).isoformat(),
                "servers_synced": servers_stored,
                "agents_synced": agents_stored,
                "skills_synced": skills_stored,
                "rejected": rejected,
                "servers_orphaned": len(orphaned_servers),
                "agents_orphaned": len(orphaned_agents),
                "skills_orphaned": skills_orphaned,
                "consecutive_failures": 0,
                "last_error": None,
            }
            logger.info(
                "ARD ingestion source=%s gen=%d servers=%d agents=%d skills=%d rejected=%d "
                "orphaned=%d/%d/%d elapsed_ms=%.1f",
                source.source_id, generation, servers_stored, agents_stored, skills_stored, rejected,
                len(orphaned_servers), len(orphaned_agents), skills_orphaned, duration * 1000,
            )
            return SyncResult(
                success=True,
                peer_id=source.source_id,
                servers_synced=servers_stored,
                agents_synced=agents_stored,
                servers_orphaned=len(orphaned_servers),
                agents_orphaned=len(orphaned_agents),
                duration_seconds=duration,
                new_generation=generation,
            )
        except Exception as e:  # noqa: BLE001 - never let one source kill the scheduler
            logger.error("ARD ingestion failed for source %s: %s", source.source_id, e, exc_info=True)
            ard_ingestion_runs_total.add(1, {"source_id": source.source_id, "status": "error"})
            failures = int(prev.get("consecutive_failures", 0)) + 1
            self._state[source.source_id] = {
                **prev,
                "generation": prev.get("generation", 0),
                "consecutive_failures": failures,
                "last_error": str(e),
                "last_attempt_at": datetime.now(UTC).isoformat(),
            }
            return SyncResult(
                success=False,
                peer_id=source.source_id,
                error_message=str(e),
                duration_seconds=time.time() - start,
                new_generation=prev.get("generation", 0),
            )

    def _map_documents(
        self,
        docs: list[tuple[Any, str]],
        source: AiCatalogSourceConfig,
        cfg: AiCatalogFederationConfig,
    ) -> tuple[list[dict], list[dict], list[dict], int]:
        """Trust-gate + reverse-map crawled manifests into record dicts.

        Returns (server_dicts, agent_dicts, skill_dicts, rejected). Servers/agents
        are stored via the peer-sync layer; skills via the skill repository.
        """
        servers: list[dict] = []
        agents: list[dict] = []
        skills: list[dict] = []
        rejected = 0
        for manifest, _doc_uri in docs:
            trust = manifest.host.trust_manifest
            host_domain = host_identity_domain(trust.identity) if trust else None
            for entry in manifest.entries:
                accept, reason = verify_entry_trust(entry, host_domain, source, cfg.trust_enforcement)
                if not accept:
                    rejected += 1
                    continue
                mapped = entry_to_record(entry, source.source_id)
                if mapped is None:
                    continue
                kind, path, record = mapped
                record["path"] = path
                if reason:  # flagged-but-accepted under the "flag" policy
                    record["trust_flag"] = reason
                if kind == "server":
                    servers.append(record)
                elif kind == "agent":
                    agents.append(record)
                elif kind == "skill":
                    skill_data = entry_to_skill_data(entry, source.source_id)
                    if skill_data is not None:
                        skills.append(skill_data)
        return servers, agents, skills, rejected

    async def _store_skills(
        self,
        skills: list[dict],
    ) -> int:
        """Create or update ingested skills via the skill repository."""
        repo = get_skill_repository()
        stored = 0
        for skill_data in skills:
            path = skill_data["path"]
            try:
                try:
                    await repo.create(SkillCard(**skill_data))
                except Exception as create_err:  # noqa: BLE001 - exists -> update
                    logger.debug("Skill create failed for %s, trying update: %s", path, create_err)
                    update_fields = {
                        k: v for k, v in skill_data.items() if k not in ("path", "id", "created_at")
                    }
                    await repo.update(path, update_fields)
                stored += 1
            except Exception as e:  # noqa: BLE001 - one bad skill must not fail the run
                logger.error("Failed to ingest skill %s: %s", path, e)
        return stored

    async def _reconcile_skill_orphans(
        self,
        source_id: str,
        fetched_paths: set[str],
    ) -> int:
        """Delete previously-ingested skills for this source no longer advertised."""
        repo = get_skill_repository()
        try:
            existing = await repo.list_filtered(include_disabled=True, registry_name=source_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Skill orphan reconcile failed to list for %s: %s", source_id, e)
            return 0
        removed = 0
        for skill in existing:
            if skill.path not in fetched_paths:
                try:
                    await repo.delete(skill.path)
                    removed += 1
                except Exception as e:  # noqa: BLE001
                    logger.error("Failed to remove orphaned skill %s: %s", skill.path, e)
        return removed


def get_ard_ingestion_service() -> ArdIngestionService:
    """Return the ARD ingestion service singleton."""
    return ArdIngestionService.get_instance()
