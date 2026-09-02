"""Unit tests for list_proxied() across the five DocumentDB entity repos.

The Mongo collection is mocked (no live DB). These lock in that list_proxied:
- issues an indexed {"is_proxied": True} filter (not a full scan);
- projects only the render/resolve-relevant fields (incl. sync_metadata for the
  federation guard, and the per-type native fallback field);
- maps _id -> path and returns raw dicts (never reconstructs a model, so a
  bypass-written invalid row cannot crash the config-reload hot path);
- degrades to [] on a driver error rather than raising into the reload.

The ABC default (list[]) is exercised for the non-overriding contract.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class _AsyncCursor:
    """Minimal async cursor over a fixed doc list (mirrors motor's find())."""

    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        async def _gen():
            for d in self._docs:
                yield d

        return _gen()


def _collection(docs):
    coll = MagicMock()
    coll.find = MagicMock(return_value=_AsyncCursor([dict(d) for d in docs]))
    return coll


async def _run_list_proxied(repo, docs):
    coll = _collection(docs)
    with patch.object(repo, "_get_collection", new=AsyncMock(return_value=coll)):
        # skill/custom call ensure_indexes() which also hits _get_collection; the
        # patched _get_collection returns the same mock, and _indexes_created guards.
        repo._indexes_created = True
        result = await repo.list_proxied()
    return result, coll


class TestServerListProxied:
    async def test_query_projection_and_mapping(self):
        from registry.repositories.documentdb.server_repository import (
            DocumentDBServerRepository,
        )

        docs = [
            {
                "_id": "/s1",
                "is_proxied": True,
                "proxy_pass_url": "https://b/",
                "deployment": "remote",
            }
        ]
        rows, coll = await _run_list_proxied(DocumentDBServerRepository(), docs)

        filt, projection = coll.find.call_args[0]
        assert filt == {"is_proxied": True}
        assert projection["proxy_pass_url"] == 1  # native fallback
        assert projection["deployment"] == 1
        assert projection["sync_metadata"] == 1  # federation guard
        assert projection["is_enabled"] == 1  # disabled -> no route
        # negative: heavy fields must NOT be pulled on the hot path
        assert "tool_list" not in projection
        assert rows[0]["path"] == "/s1"
        assert "_id" not in rows[0]


class TestAgentListProxied:
    async def test_projects_url_fallback(self):
        from registry.repositories.documentdb.agent_repository import (
            DocumentDBAgentRepository,
        )

        docs = [{"_id": "/a1", "is_proxied": True, "url": "https://a/"}]
        rows, coll = await _run_list_proxied(DocumentDBAgentRepository(), docs)

        _filt, projection = coll.find.call_args[0]
        assert projection["url"] == 1  # agent native fallback
        assert projection["sync_metadata"] == 1  # federation guard
        assert projection["is_enabled"] == 1
        assert rows[0]["path"] == "/a1"


class TestSkillListProxied:
    async def test_projects_no_native_fallback(self):
        from registry.repositories.documentdb.skill_repository import (
            DocumentDBSkillRepository,
        )

        docs = [{"_id": "/skills/x", "is_proxied": True, "proxy_target_url": "https://t/"}]
        rows, coll = await _run_list_proxied(DocumentDBSkillRepository(), docs)

        _filt, projection = coll.find.call_args[0]
        assert projection["proxy_target_url"] == 1
        assert "url" not in projection and "proxy_pass_url" not in projection
        assert projection["sync_metadata"] == 1  # federation guard
        assert projection["is_enabled"] == 1
        assert rows[0]["path"] == "/skills/x"


class TestVirtualServerListProxied:
    async def test_alias_only_projection(self):
        from registry.repositories.documentdb.virtual_server_repository import (
            DocumentDBVirtualServerRepository,
        )

        docs = [{"_id": "/virtual/v1", "is_proxied": True, "is_enabled": True}]
        rows, coll = await _run_list_proxied(DocumentDBVirtualServerRepository(), docs)

        _filt, projection = coll.find.call_args[0]
        assert "proxy_target_url" not in projection  # alias-only
        assert projection["sync_metadata"] == 1  # federation guard
        assert projection["is_enabled"] == 1
        assert rows[0]["path"] == "/virtual/v1"


class TestCustomEntityListProxied:
    async def test_projects_entity_type(self):
        from registry.repositories.documentdb.custom_entity_repository import (
            DocumentDBCustomEntityRepository,
        )

        docs = [
            {
                "_id": "/workflow/uuid",
                "entity_type": "workflow",
                "is_proxied": True,
                "proxy_target_url": "https://t/",
            }
        ]
        rows, coll = await _run_list_proxied(DocumentDBCustomEntityRepository(), docs)

        _filt, projection = coll.find.call_args[0]
        assert projection["entity_type"] == 1  # type token spans records
        assert projection["sync_metadata"] == 1  # federation guard
        assert projection["is_enabled"] == 1
        assert rows[0]["path"] == "/workflow/uuid"
        assert rows[0]["entity_type"] == "workflow"


class TestListProxiedFailClosed:
    async def test_driver_error_returns_empty_not_raises(self):
        from registry.repositories.documentdb.server_repository import (
            DocumentDBServerRepository,
        )

        repo = DocumentDBServerRepository()
        coll = MagicMock()
        coll.find = MagicMock(side_effect=RuntimeError("db down"))
        with patch.object(repo, "_get_collection", new=AsyncMock(return_value=coll)):
            assert await repo.list_proxied() == []  # must not raise into the reload

    async def test_collection_acquisition_error_returns_empty(self):
        """A connection/acquisition error must ALSO fail closed (acquisition is
        inside the try, per the widened boundary)."""
        from registry.repositories.documentdb.server_repository import (
            DocumentDBServerRepository,
        )

        repo = DocumentDBServerRepository()
        with patch.object(
            repo, "_get_collection", new=AsyncMock(side_effect=RuntimeError("no db"))
        ):
            assert await repo.list_proxied() == []


class TestIsProxiedIndexResilient:
    """ensure_is_proxied_index never raises and falls back to a plain index when
    the partial form is rejected (AWS DocumentDB)."""

    async def test_partial_rejected_falls_back_to_plain(self):
        from registry.repositories.documentdb._identity_url_sidecar import (
            ensure_is_proxied_index,
        )

        calls = []

        async def _create_index(*args, **kwargs):
            calls.append(kwargs)
            if "partialFilterExpression" in kwargs:
                raise RuntimeError("partial indexes not supported")  # DocumentDB
            return "idx"

        coll = MagicMock()
        coll.create_index = _create_index
        # Must not raise, and must attempt the plain fallback after the partial fails.
        await ensure_is_proxied_index(coll, "mcp_servers")
        assert any("partialFilterExpression" in c for c in calls)
        assert any("partialFilterExpression" not in c for c in calls)

    async def test_both_rejected_still_no_raise(self):
        from registry.repositories.documentdb._identity_url_sidecar import (
            ensure_is_proxied_index,
        )

        async def _create_index(*args, **kwargs):
            raise RuntimeError("no indexes at all")

        coll = MagicMock()
        coll.create_index = _create_index
        await ensure_is_proxied_index(coll, "mcp_servers")  # must not raise


class TestAbcDefaultIsEmpty:
    @pytest.mark.parametrize(
        "base_name",
        [
            "ServerRepositoryBase",
            "AgentRepositoryBase",
            "SkillRepositoryBase",
            "VirtualServerRepositoryBase",
            "CustomEntityRepositoryBase",
        ],
    )
    async def test_default_returns_empty(self, base_name):
        # The non-abstract default returns [] (safe no-op for non-overriding
        # backends). Call the unbound coroutine with a dummy self to avoid ABC
        # instantiation of the many other abstractmethods.
        import registry.repositories.interfaces as interfaces

        base = getattr(interfaces, base_name)
        assert await base.list_proxied(object()) == []
