"""Persistence-level regression tests for PATCH /servers writes.

Exercises DocumentDBServerRepository.update — the write primitive under
PATCH /api/servers/{path} — against a real MongoDB: the trailing-slash
_id asymmetry that made a description-only PATCH fail with a 500, the
field-scoped $set that stops a metadata PATCH from clobbering a
concurrent credential/egress write, and the atomic revision predicate
that backs If-Match. Requires a running MongoDB (the test harness
points DOCUMENTDB_HOST at localhost); skipped automatically if
unreachable.
"""

import uuid

import pytest

from registry.repositories.documentdb.server_repository import (
    DocumentDBServerRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _card() -> dict:
    return {
        "server_name": "legacy",
        "description": "initial",
        "tags": ["old"],
        "license": "MIT",
        "deployment": "remote",
        "proxy_pass_url": "http://upstream:9000",
        "registered_by": "alice",
        "id": f"asset-{uuid.uuid4().hex[:8]}",
        "is_enabled": True,
        "is_active": True,
        "version": "v1.0.0",
        "num_tools": 1,
        "tool_list": [{"name": "search", "inputSchema": {"$schema": "x", "type": "object"}}],
        "auth_credential_encrypted": "ENC::keepme",
        "egress_oauth": {"provider": "github", "scopes": ["repo"]},
        "registered_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


@pytest.fixture
async def repo():
    """Repository backed by a real collection; cleans up cards it created."""
    r = DocumentDBServerRepository()
    try:
        col = await r._get_collection()
        await col.database.command("ping")
    except Exception as e:
        pytest.skip(f"MongoDB not reachable: {e}")
    created: list[str] = []
    r._created = created
    yield r
    col = await r._get_collection()
    if created:
        await col.delete_many({"_id": {"$in": created}})


class TestSlashVariantWrite:
    async def test_update_resolves_slash_variant_id(self, repo):
        """A card stored under '/x/' must be writable as '/x'."""
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/legacy-variant/", **card})
        repo._created.append("/legacy-variant/")

        existing = await repo.get("/legacy-variant")
        assert existing is not None

        merged = {**existing, "description": "patched"}
        assert await repo.update("/legacy-variant", merged, updated_fields=["description"])

        doc = await col.find_one({"_id": "/legacy-variant/"})
        assert doc["description"] == "patched"


class TestFieldScopedWrite:
    async def test_scoped_update_preserves_concurrent_credential_write(self, repo):
        """A PATCH must not overwrite fields another writer owns."""
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/racy-scope", **card})
        repo._created.append("/racy-scope")

        stale_read = await repo.get("/racy-scope")
        await col.update_one(
            {"_id": "/racy-scope"},
            {"$set": {"auth_credential_encrypted": "ENC::rotated"}},
        )

        merged = {**stale_read, "description": "patched"}
        assert await repo.update("/racy-scope", merged, updated_fields=["description"])

        doc = await col.find_one({"_id": "/racy-scope"})
        assert doc["description"] == "patched"
        assert doc["auth_credential_encrypted"] == "ENC::rotated"
        assert doc["egress_oauth"] == card["egress_oauth"]
        assert doc["tool_list"] == card["tool_list"]

    async def test_repeated_identical_patch_is_harmless(self, repo):
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/idem-scope", **card})
        repo._created.append("/idem-scope")

        for _ in range(2):
            existing = await repo.get("/idem-scope")
            merged = {**existing, "description": "same"}
            assert await repo.update("/idem-scope", merged, updated_fields=["description"])

        doc = await col.find_one({"_id": "/idem-scope"})
        assert doc["description"] == "same"
        assert doc["num_tools"] == card["num_tools"]


class TestRevisionGuardedWrite:
    async def test_stale_revision_fails_atomically(self, repo):
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/cas-race", **card})
        repo._created.append("/cas-race")

        existing = await repo.get("/cas-race")
        # Another writer lands between our read and our write.
        await col.update_one(
            {"_id": "/cas-race"},
            {
                "$set": {
                    "auth_credential_encrypted": "ENC::theirs",
                    "updated_at": "2030-01-01T00:00:00",
                }
            },
        )

        merged = {**existing, "description": "must not land"}
        result = await repo.update(
            "/cas-race",
            merged,
            updated_fields=["description"],
            expected_updated_at=existing["updated_at"],
        )
        assert result is False

        doc = await col.find_one({"_id": "/cas-race"})
        assert doc["description"] == "initial"
        assert doc["auth_credential_encrypted"] == "ENC::theirs"

    async def test_fresh_revision_writes(self, repo):
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/cas-race-2", **card})
        repo._created.append("/cas-race-2")

        existing = await repo.get("/cas-race-2")
        merged = {**existing, "description": "landed"}
        assert await repo.update(
            "/cas-race-2",
            merged,
            updated_fields=["description"],
            expected_updated_at=existing["updated_at"],
        )
        doc = await col.find_one({"_id": "/cas-race-2"})
        assert doc["description"] == "landed"

    async def test_stale_revision_does_not_retarget_to_variant_card(self, repo):
        """A lost race must fail even when a different card sits at the variant _id."""
        col = await repo._get_collection()
        card = _card()
        await col.insert_one({"_id": "/cas-variant-race", **card})
        variant_card = {**_card(), "server_name": "variant occupant"}
        await col.insert_one({"_id": "/cas-variant-race/", **variant_card})
        repo._created.extend(["/cas-variant-race", "/cas-variant-race/"])

        existing = await repo.get("/cas-variant-race")
        # Another writer lands on the exact card between our read and write.
        await col.update_one(
            {"_id": "/cas-variant-race"},
            {"$set": {"updated_at": "2030-01-01T00:00:00"}},
        )

        merged = {**existing, "description": "must not land"}
        result = await repo.update(
            "/cas-variant-race",
            merged,
            updated_fields=["description"],
            expected_updated_at=existing["updated_at"],
        )
        assert result is False

        doc = await col.find_one({"_id": "/cas-variant-race"})
        assert doc["description"] == "initial"
        variant = await col.find_one({"_id": "/cas-variant-race/"})
        assert variant["description"] == "initial"
        assert variant["server_name"] == "variant occupant"
