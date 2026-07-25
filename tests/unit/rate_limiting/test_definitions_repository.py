"""Unit tests for the definitions repository read cache and parsing.

These patch the collection accessor so no real DocumentDB is needed; the focus is
the cache behavior (hit / miss / invalidation) and malformed-doc resilience.
"""

from registry.rate_limiting.definitions_repository import DefinitionsRepository


class _FakeCursor:
    """Async-iterable cursor over a fixed list of docs."""

    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for doc in self._docs:
            yield dict(doc)


class _FakeCollection:
    """Records find() calls and returns queued doc sets, to prove caching."""

    def __init__(self, docs):
        self._docs = docs
        self.find_calls = 0

    def find(self, query):
        self.find_calls += 1
        return _FakeCursor(self._docs)


def _make_repo_with_docs(docs, ttl=30.0):
    """Build a repository whose collection is a fake with the given docs."""
    repo = DefinitionsRepository(cache_ttl_seconds=ttl)
    collection = _FakeCollection(docs)

    async def _fake_get_collection():
        return collection

    repo._get_collection = _fake_get_collection  # type: ignore[assignment]
    return repo, collection


def _doc_matches(doc, query):
    """Minimal Mongo-query matcher: equality, top-level ``$or``, and array-contains.

    Only supports the operators the definitions repo actually builds, enough to
    exercise the server_group ``$or`` expansion in list_target_limits.
    """
    for key, cond in query.items():
        if key == "$or":
            if not any(_doc_matches(doc, sub) for sub in cond):
                return False
            continue
        value = doc.get(key)
        if isinstance(value, list):
            # array-contains match (e.g. {"members": "mcpgw"})
            if cond not in value:
                return False
        elif value != cond:
            return False
    return True


class _FilteringCollection:
    """A fake collection whose find() actually applies the query (for $or tests)."""

    def __init__(self, docs):
        self._docs = docs
        self.find_calls = 0

    def find(self, query):
        self.find_calls += 1
        matched = [d for d in self._docs if _doc_matches(d, query)]
        return _FakeCursor(matched)


def _make_repo_with_filtering(docs, ttl=30.0):
    """Build a repository whose fake collection filters by the query."""
    repo = DefinitionsRepository(cache_ttl_seconds=ttl)
    collection = _FilteringCollection(docs)

    async def _fake_get_collection():
        return collection

    repo._get_collection = _fake_get_collection  # type: ignore[assignment]
    return repo, collection


class TestDefinitionsCache:
    """Tests for the in-process read cache."""

    async def test_caller_limits_parsed(self):
        """A caller-limit doc is parsed into a model."""
        docs = [
            {
                "_id": "caller:group:dev:60",
                "axis": "caller",
                "entity_type": "group",
                "name": "dev",
                "user_max_requests": 5,
                "window_seconds": 60,
                "enabled": True,
            }
        ]
        repo, _ = _make_repo_with_docs(docs)
        result = await repo.list_caller_limits("group", ["dev"])
        assert len(result) == 1
        assert result[0].user_max_requests == 5

    async def test_second_read_is_cached(self):
        """A repeated identical query does not hit the collection again."""
        docs = [
            {
                "_id": "caller:group:dev:60",
                "axis": "caller",
                "entity_type": "group",
                "name": "dev",
                "user_max_requests": 5,
                "window_seconds": 60,
                "enabled": True,
            }
        ]
        repo, collection = _make_repo_with_docs(docs)
        await repo.list_caller_limits("group", ["dev"])
        await repo.list_caller_limits("group", ["dev"])
        assert collection.find_calls == 1

    async def test_empty_names_short_circuits(self):
        """No group names => no query, no limits."""
        repo, collection = _make_repo_with_docs([])
        result = await repo.list_caller_limits("group", [])
        assert result == []
        assert collection.find_calls == 0

    async def test_invalidate_cache_forces_reread(self):
        """After invalidation the next read hits the collection again."""
        docs = [
            {
                "_id": "target:mcp_server:mcpgw:60",
                "axis": "target",
                "entity_type": "mcp_server",
                "name": "mcpgw",
                "max_requests": 500,
                "window_seconds": 60,
                "enabled": True,
            }
        ]
        repo, collection = _make_repo_with_docs(docs)
        await repo.list_target_limits("mcp_server", "mcpgw")
        repo.invalidate_cache()
        await repo.list_target_limits("mcp_server", "mcpgw")
        assert collection.find_calls == 2

    async def test_malformed_doc_is_skipped(self):
        """A malformed definition must not break the read (auth path resilience)."""
        docs = [
            {
                "_id": "caller:group:dev:60",
                "axis": "caller",
                "entity_type": "group",
                "name": "dev",
                "user_max_requests": 5,
                "window_seconds": 60,
                "enabled": True,
            },
            {
                "_id": "caller:group:bad:60",
                "axis": "caller",
                "entity_type": "group",
                "name": "bad",
                "user_max_requests": -1,  # invalid: ge=1
                "window_seconds": 60,
                "enabled": True,
            },
        ]
        repo, _ = _make_repo_with_docs(docs)
        result = await repo.list_caller_limits("group", ["dev", "bad"])
        assert len(result) == 1
        assert result[0].name == "dev"


class TestServerGroupTargetLimits:
    """Tests for server_group expansion in list_target_limits."""

    def _docs(self):
        return [
            {
                "_id": "target:server_group:fragile:60",
                "axis": "target",
                "entity_type": "server_group",
                "name": "fragile",
                "max_requests": 100,
                "window_seconds": 60,
                "members": ["mcpgw", "atlas"],
                "enabled": True,
            },
            {
                "_id": "target:mcp_server:mcpgw:60",
                "axis": "target",
                "entity_type": "mcp_server",
                "name": "mcpgw",
                "max_requests": 500,
                "window_seconds": 60,
                "enabled": True,
            },
        ]

    async def test_member_server_picks_up_group_limit(self):
        """A server in a group's members list gets the group definition."""
        repo, _ = _make_repo_with_filtering(self._docs())
        result = await repo.list_target_limits("mcp_server", "atlas")
        # atlas has no individual def, only the group -> exactly the group def.
        assert [d.entity_type for d in result] == ["server_group"]
        assert result[0].max_requests == 100

    async def test_non_member_server_gets_nothing(self):
        """A server not in any group and with no individual def gets no limits."""
        repo, _ = _make_repo_with_filtering(self._docs())
        result = await repo.list_target_limits("mcp_server", "other")
        assert result == []

    async def test_server_with_both_individual_and_group_limits(self):
        """A server with its own def AND group membership gets BOTH (independent buckets)."""
        repo, _ = _make_repo_with_filtering(self._docs())
        result = await repo.list_target_limits("mcp_server", "mcpgw")
        entity_types = sorted(d.entity_type for d in result)
        assert entity_types == ["mcp_server", "server_group"]

    async def test_agent_lookup_never_matches_server_group(self):
        """The group arm is scoped to mcp_server; an a2a_agent lookup ignores server_groups."""
        docs = self._docs() + [
            {
                "_id": "target:a2a_agent:mcpgw:60",
                "axis": "target",
                "entity_type": "a2a_agent",
                "name": "mcpgw",
                "max_requests": 10,
                "window_seconds": 60,
                "enabled": True,
            }
        ]
        repo, _ = _make_repo_with_filtering(docs)
        result = await repo.list_target_limits("a2a_agent", "mcpgw")
        assert [d.entity_type for d in result] == ["a2a_agent"]

    async def test_two_member_servers_use_distinct_cache_keys(self):
        """Different member servers are cached separately (per-server keys)."""
        repo, collection = _make_repo_with_filtering(self._docs())
        await repo.list_target_limits("mcp_server", "mcpgw")
        await repo.list_target_limits("mcp_server", "atlas")
        assert collection.find_calls == 2
