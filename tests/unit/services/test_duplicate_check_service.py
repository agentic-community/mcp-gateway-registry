"""Unit tests for the shared ``DuplicateCheckService``.

The service is exercised against in-memory fakes for its dependencies
(a stubbed ``SemanticSearchService`` plus the three entity repository
factories patched at the module level) so the tests run without any
database. Visibility helpers in ``search_routes`` are patched so each
test can dictate access.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.services.duplicate_check_service import DuplicateCheckService


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    server_match: dict | None = None,
    agent_match: dict | None = None,
    skill_match: dict | None = None,
    search_results: dict | None = None,
    search_raises: Exception | None = None,
    threshold: float = 0.7,
    max_suggestions: int = 3,
) -> DuplicateCheckService:
    """Construct a service with stubbed-out dependencies.

    The three entity repository factories, the SemanticSearchService
    class, and the global ``settings`` singleton are patched at the
    duplicate_check_service module level so
    ``DuplicateCheckService.__init__`` picks up the fakes when it
    resolves them.
    """
    server_repository = MagicMock()
    server_repository.find_by_identity_url = AsyncMock(return_value=server_match)
    agent_repository = MagicMock()
    agent_repository.find_by_identity_url = AsyncMock(return_value=agent_match)
    skill_repository = MagicMock()
    skill_repository.find_by_identity_url = AsyncMock(return_value=skill_match)

    monkeypatch.setattr(
        "registry.services.duplicate_check_service.get_server_repository",
        lambda: server_repository,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.get_agent_repository",
        lambda: agent_repository,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.get_skill_repository",
        lambda: skill_repository,
    )

    semantic_search_service = MagicMock()
    if search_raises is not None:
        semantic_search_service.search = AsyncMock(side_effect=search_raises)
    else:
        semantic_search_service.search = AsyncMock(return_value=search_results or {})
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.SemanticSearchService",
        lambda: semantic_search_service,
    )

    fake_settings = MagicMock()
    fake_settings.dedup_score_threshold = threshold
    fake_settings.dedup_max_suggestions = max_suggestions
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.settings",
        fake_settings,
    )

    return DuplicateCheckService()


@pytest.fixture(autouse=True)
def _allow_all_visibility(monkeypatch):
    """By default, allow every visibility check to pass.

    ``user_can_access_server`` / ``user_can_access_skill`` are
    coroutines; ``user_can_access_agent_from_doc`` is synchronous (it
    reads visibility fields from the candidate dict instead of
    fetching). The fakes need to match those signatures or the
    service code awaits a non-coroutine and raises RuntimeWarning.
    """

    async def _allow_async(*_args, **_kwargs) -> bool:
        return True

    def _allow_sync(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_server",
        _allow_async,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_agent_from_doc",
        _allow_sync,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_skill",
        _allow_async,
    )


def _check_kwargs(
    name: str = "Test",
    description: str | None = None,
    identity_url: str | None = None,
    self_path: str | None = None,
    user_context: dict | None = None,
) -> dict:
    """Default kwargs for service.check() — tests override what they care about."""
    return {
        "name": name,
        "description": description,
        "identity_url": identity_url,
        "self_path": self_path,
        "user_context": user_context if user_context is not None else {"is_admin": True},
    }


@pytest.mark.asyncio
class TestBothChecksRunIndependently:
    """Both checks run on every call; neither short-circuits the other."""

    async def test_collision_and_advisory_populate_independently(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/exact",
                "server_name": "ExactMatch",
                "owner": "team-a",
            },
            search_results={
                "servers": [
                    {"path": "/sim", "server_name": "Similar", "relevance_score": 0.85},
                ]
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="My New Server",
                description="A server",
                identity_url="https://api.example.com/mcp",
            )
        )
        assert len(result.collision_with) == 1
        assert result.collision_with[0].path == "/exact"
        assert [m.path for m in result.advisory_matches] == ["/sim"]

    async def test_url_match_excluded_from_advisory_to_avoid_duplication(self, monkeypatch) -> None:
        """An exact-URL hit must not also appear in the advisory list."""
        service = _build_service(
            monkeypatch,
            server_match={"path": "/exact", "server_name": "ExactMatch"},
            search_results={
                "servers": [
                    # Same path as the URL match — should be filtered.
                    {"path": "/exact", "server_name": "ExactMatch", "relevance_score": 0.95},
                    {"path": "/other", "server_name": "Other", "relevance_score": 0.85},
                ]
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="My Server",
                description="...",
                identity_url="https://api.example.com/mcp",
            )
        )
        advisory_paths = [m.path for m in result.advisory_matches]
        assert "/exact" not in advisory_paths
        assert "/other" in advisory_paths


@pytest.mark.asyncio
class TestExactMatchCheck:
    async def test_server_url_collision_returns_existing_entity(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/foo",
                "server_name": "Foo",
                "registered_by": "team-a",
                "registered_at": "2026-04-12T10:30:00Z",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="My New Server",
                description="...",
                identity_url="https://api.example.com/mcp",
            )
        )
        assert len(result.collision_with) == 1
        entity = result.collision_with[0]
        assert entity.entity_type == "mcp_server"
        assert entity.path == "/foo"
        assert entity.name == "Foo"
        assert entity.owner == "team-a"
        assert entity.match_reason == "exact URL match"

    async def test_agent_url_collision(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            agent_match={
                "path": "/agents/foo",
                "name": "Agent Foo",
                "registered_by": "alice",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Agent Bar",
                identity_url="https://api.example.com/agent",
            )
        )
        assert len(result.collision_with) == 1
        entity = result.collision_with[0]
        assert entity.entity_type == "a2a_agent"
        assert entity.path == "/agents/foo"
        assert entity.owner == "alice"

    async def test_skill_url_collision(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            skill_match={
                "path": "/skills/foo",
                "skill_name": "Foo Skill",
                "owner": "bob",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="My Skill",
                identity_url="https://github.com/org/repo/blob/main/SKILL.md",
            )
        )
        assert len(result.collision_with) == 1
        entity = result.collision_with[0]
        assert entity.entity_type == "skill"
        assert entity.path == "/skills/foo"
        assert entity.name == "Foo Skill"

    async def test_url_collision_is_cross_entity(self, monkeypatch) -> None:
        """Collision check scans all three entity repos.

        With the indexed sidecar lookup, scanning the other repos is
        essentially free (one indexed $eq each). A registering
        server's URL that already exists as a skill (e.g. both point
        at the same GitHub repo) surfaces here so the user is alerted
        about the cross-type duplicate.
        """
        service = _build_service(
            monkeypatch,
            server_match={"path": "/server-alpha", "server_name": "Alpha"},
            skill_match={"path": "/skills/alpha", "skill_name": "AlphaSkill"},
        )
        result = await service.check(
            **_check_kwargs(
                name="New Alpha",
                identity_url="https://api.example.com/mcp",
            )
        )
        # All matching repos contribute their hits, each tagged with
        # the entity_type appropriate to the repo that produced it.
        types = sorted(e.entity_type for e in result.collision_with)
        assert types == ["mcp_server", "skill"]
        # All three repos were queried — agent had no match, skill and
        # server did.
        service._repositories["a2a_agent"].find_by_identity_url.assert_called_once()
        service._repositories["skill"].find_by_identity_url.assert_called_once()
        service._repositories["mcp_server"].find_by_identity_url.assert_called_once()

    async def test_url_collision_normalizes_per_target_repo(self, monkeypatch) -> None:
        """Each repo gets the URL normalized with its own rule.

        The skill rule strips ``.git``; server/agent rules don't.
        Normalizing per target repo keeps cross-type lookups
        apples-to-apples with whatever each repo stored at write
        time.
        """
        from registry.utils.url_normalize import (
            ENTITY_TYPE_AGENT,
            ENTITY_TYPE_SERVER,
            ENTITY_TYPE_SKILL,
            normalize_identity_url,
        )

        service = _build_service(monkeypatch)
        await service.check(
            **_check_kwargs(
                name="X",
                identity_url="https://github.com/org/repo.git",
            )
        )
        server_url = service._repositories["mcp_server"].find_by_identity_url.await_args.args[0]
        agent_url = service._repositories["a2a_agent"].find_by_identity_url.await_args.args[0]
        skill_url = service._repositories["skill"].find_by_identity_url.await_args.args[0]
        assert server_url == normalize_identity_url(
            "https://github.com/org/repo.git", ENTITY_TYPE_SERVER
        )
        assert agent_url == normalize_identity_url(
            "https://github.com/org/repo.git", ENTITY_TYPE_AGENT
        )
        assert skill_url == normalize_identity_url(
            "https://github.com/org/repo.git", ENTITY_TYPE_SKILL
        )
        # Server and skill rules produce different strings; this test
        # would be vacuous if they didn't.
        assert server_url != skill_url

    async def test_self_path_excluded_from_collision(self, monkeypatch) -> None:
        """Re-registering the same entity should not collide with itself."""
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo", "server_name": "Foo"},
        )
        result = await service.check(
            **_check_kwargs(
                name="Foo",
                identity_url="https://api.example.com/mcp",
                self_path="/foo",
            )
        )
        assert result.collision_with == []

    async def test_no_identity_url_skips_exact_match_check(self, monkeypatch) -> None:
        """The exact-match check must short-circuit when the identity URL is missing."""
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo", "server_name": "Foo"},
        )
        result = await service.check(
            **_check_kwargs(
                name="Foo",
                identity_url=None,
            )
        )
        assert result.collision_with == []

    async def test_unparseable_identity_url_skips_exact_match_check(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo", "server_name": "Foo"},
        )
        result = await service.check(
            **_check_kwargs(
                name="Foo",
                identity_url="not a url",
            )
        )
        assert result.collision_with == []

    async def test_repo_exception_yields_empty_collision_but_advisory_still_runs(
        self, monkeypatch
    ) -> None:
        """A repo throwing on find_by_identity_url must not break the request.

        The service logs and returns an empty collision list; the
        similarity check runs independently and still populates
        advisory_matches. Treating a transient repo failure as
        "no collision found" is the right call for an advisory check
        that never blocks registration.
        """
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo"},  # would have matched if repo were healthy
            search_results={
                "servers": [
                    {"path": "/sim", "server_name": "Similar", "relevance_score": 0.85},
                ]
            },
        )
        # Make the server repo throw on find_by_identity_url.
        service._repositories["mcp_server"].find_by_identity_url = AsyncMock(
            side_effect=RuntimeError("db unavailable"),
        )
        result = await service.check(
            **_check_kwargs(
                name="X",
                description="Y",
                identity_url="https://api.example.com/mcp",
            )
        )
        assert result.collision_with == []
        assert [m.path for m in result.advisory_matches] == ["/sim"]
        assert result.similarity_search_available is True

    async def test_own_entry_still_surfaces_as_collision(self, monkeypatch) -> None:
        """Re-registering one's own server still surfaces as collision.

        Users should see their own duplicates so they can choose to
        update rather than create a second entry at the same URL.
        The self_path exclusion is the only way to suppress a match
        (used by the edit flow where the user IS updating that entry).
        """
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/me",
                "server_name": "Mine",
                "registered_by": "alice",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Mine",
                identity_url="https://api.example.com/mcp",
                user_context={"username": "alice", "is_admin": True},
            )
        )
        assert len(result.collision_with) == 1
        assert result.collision_with[0].path == "/me"

    async def test_own_skill_still_surfaces_as_collision(self, monkeypatch) -> None:
        """Skills owned by the caller still surface as collisions."""
        service = _build_service(
            monkeypatch,
            skill_match={
                "path": "/skills/mine",
                "skill_name": "MySkill",
                "owner": "alice",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="MySkill",
                identity_url="https://github.com/alice/repo/blob/main/SKILL.md",
                user_context={"username": "alice", "is_admin": True},
            )
        )
        assert len(result.collision_with) == 1
        assert result.collision_with[0].path == "/skills/mine"

    async def test_collision_surfaces_when_owner_differs(self, monkeypatch) -> None:
        """Auto-exclusion fires only on exact username match.

        A different user registering the same URL still gets the
        collision — that's the whole point of the dedup signal.
        """
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/theirs",
                "server_name": "Theirs",
                "registered_by": "bob",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Mine",
                identity_url="https://api.example.com/mcp",
                user_context={"username": "alice", "is_admin": True},
            )
        )
        assert len(result.collision_with) == 1
        assert result.collision_with[0].path == "/theirs"


@pytest.mark.asyncio
class TestExactMatchVisibilityRedaction:
    async def test_collision_redacted_when_caller_cannot_view(self, monkeypatch) -> None:
        """Owner/path/name blanked when the caller can't see the entity.

        Existence is still exposed (entry appears in collision_with) so
        the frontend can render a generic "URL is already registered"
        hint, but no ownership leak.
        """

        async def _deny(*_args, **_kwargs) -> bool:
            return False

        monkeypatch.setattr(
            "registry.services.duplicate_check_service.user_can_access_server",
            _deny,
        )
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/private-entity",
                "server_name": "Private",
                "owner": "secret-team",
                "visibility": "private",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Anything",
                identity_url="https://api.example.com/mcp",
                user_context={"username": "attacker"},
            )
        )
        assert len(result.collision_with) == 1
        entity = result.collision_with[0]
        assert entity.path == ""
        assert entity.name == ""
        assert entity.owner is None


@pytest.mark.asyncio
class TestSimilarityAdvisory:
    async def test_returns_filtered_results_above_threshold(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [
                    {"path": "/alpha", "server_name": "Alpha", "relevance_score": 0.91},
                    {"path": "/beta", "server_name": "Beta", "relevance_score": 0.55},  # below
                    {"path": "/gamma", "server_name": "Gamma", "relevance_score": 0.81},
                ]
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="My New Server",
                description="A server that does X",
            )
        )
        assert result.collision_with == []
        names = [m.name for m in result.advisory_matches]
        assert "Beta" not in names
        assert "Alpha" in names
        assert "Gamma" in names
        assert all(
            m.match_reason == "similar name and description" for m in result.advisory_matches
        )

    async def test_caps_at_max_suggestions(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [
                    {"path": f"/s{i}", "server_name": f"S{i}", "relevance_score": 0.95}
                    for i in range(10)
                ]
            },
            max_suggestions=3,
        )
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert len(result.advisory_matches) == 3

    async def test_self_path_excluded_from_advisory(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [
                    {"path": "/me", "server_name": "Me", "relevance_score": 0.99},
                    {"path": "/other", "server_name": "Other", "relevance_score": 0.85},
                ]
            },
        )
        result = await service.check(**_check_kwargs(name="X", description="Y", self_path="/me"))
        paths = [m.path for m in result.advisory_matches]
        assert "/me" not in paths
        assert "/other" in paths

    async def test_visibility_filter_drops_private_results(self, monkeypatch) -> None:
        async def _deny_unless_alpha(path, *_args, **_kwargs) -> bool:
            return path == "/alpha"

        monkeypatch.setattr(
            "registry.services.duplicate_check_service.user_can_access_server",
            _deny_unless_alpha,
        )
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [
                    {"path": "/alpha", "server_name": "Alpha", "relevance_score": 0.95},
                    {"path": "/beta", "server_name": "Beta", "relevance_score": 0.92},
                ]
            },
        )
        result = await service.check(
            **_check_kwargs(name="X", description="Y", user_context={"username": "user"})
        )
        paths = [m.path for m in result.advisory_matches]
        assert paths == ["/alpha"]

    async def test_search_unavailable_returns_available_false(self, monkeypatch) -> None:
        service = _build_service(monkeypatch, search_raises=RuntimeError("FAISS down"))
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert result.collision_with == []
        assert result.advisory_matches == []
        assert result.similarity_search_available is False

    async def test_unexpected_search_error_treated_as_unavailable(self, monkeypatch) -> None:
        service = _build_service(monkeypatch, search_raises=ValueError("bad query"))
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert result.similarity_search_available is False
        assert result.advisory_matches == []

    async def test_overfetch_factor_passed_to_search(self, monkeypatch) -> None:
        """The similarity check requests more candidates than the visible cap."""
        service = _build_service(
            monkeypatch,
            search_results={"servers": []},
            max_suggestions=5,
        )
        await service.check(**_check_kwargs(name="X", description="Y"))
        called_kwargs = service._semantic_search_service.search.await_args.kwargs
        assert called_kwargs["max_results"] == 5 * 10  # _SIMILARITY_OVERFETCH_FACTOR

    async def test_search_call_runs_cross_entity(self, monkeypatch) -> None:
        """The similarity check searches all entity types, not just one."""
        service = _build_service(monkeypatch, search_results={"servers": []})
        await service.check(**_check_kwargs(name="X", description="Y"))
        called_kwargs = service._semantic_search_service.search.await_args.kwargs
        # entity_types=None means cross-entity (no filter).
        assert called_kwargs["entity_types"] is None

    async def test_advisory_includes_results_from_all_entity_types(self, monkeypatch) -> None:
        """Cross-entity advisory: server registration can surface skills/agents too."""
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [{"path": "/s1", "server_name": "S1", "relevance_score": 0.9}],
                "agents": [{"path": "/a1", "name": "A1", "relevance_score": 0.95}],
                "skills": [{"path": "/k1", "skill_name": "K1", "relevance_score": 0.85}],
            },
        )
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        types = sorted(m.entity_type for m in result.advisory_matches)
        assert types == ["a2a_agent", "mcp_server", "skill"]


@pytest.mark.asyncio
class TestQueryComposition:
    async def test_empty_query_skips_search(self, monkeypatch) -> None:
        service = _build_service(monkeypatch, search_results={"servers": []})
        result = await service.check(**_check_kwargs(name="", description=None))
        service._semantic_search_service.search.assert_not_called()
        assert result.advisory_matches == []
        assert result.similarity_search_available is True

    async def test_query_combines_name_and_description(self, monkeypatch) -> None:
        service = _build_service(monkeypatch, search_results={"servers": []})
        await service.check(**_check_kwargs(name="Github Tools", description="Wraps gh API"))
        called_kwargs = service._semantic_search_service.search.await_args.kwargs
        assert called_kwargs["query"] == "Github Tools Wraps gh API"

    async def test_long_description_is_truncated_before_search(self, monkeypatch) -> None:
        """Bound the per-call cost of long descriptions.

        The registration form's description field is unbounded (10K
        chars at the schema layer); without truncation a 10K-char
        description would be embedded verbatim every time a user types
        a character into the form. The leading ~500 chars carry the
        bulk of the distinguishing similarity signal.
        """
        from registry.services.duplicate_check_service import _QUERY_TEXT_CHAR_CAP

        service = _build_service(monkeypatch, search_results={"servers": []})
        long_description = "x" * 1000
        await service.check(**_check_kwargs(name="N", description=long_description))
        called_kwargs = service._semantic_search_service.search.await_args.kwargs
        query = called_kwargs["query"]
        assert len(query) == _QUERY_TEXT_CHAR_CAP
        # The leading portion (name + space + start of description) is preserved.
        assert query.startswith("N x")


@pytest.mark.asyncio
class TestExtractorFallbacks:
    """Defensive coverage for the schema-mismatch extractors.

    The repo doc shape and the search hit shape diverge slightly per
    entity type (skill_name vs name, agent_card nesting, etc.). These
    tests document the fallback ordering so a future schema change
    fails loudly rather than silently producing blank fields.
    """

    async def test_agent_search_hit_with_agent_card_nesting(self, monkeypatch) -> None:
        """Search hits expose name on agent_card; the extractor must follow it."""
        service = _build_service(
            monkeypatch,
            search_results={
                "agents": [
                    {
                        "path": "/agents/foo",
                        "agent_card": {
                            "name": "AgentFromCard",
                            "registered_by": "alice",
                            "visibility": "public",
                        },
                        "relevance_score": 0.9,
                    }
                ]
            },
        )
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert len(result.advisory_matches) == 1
        entity = result.advisory_matches[0]
        assert entity.entity_type == "a2a_agent"
        assert entity.name == "AgentFromCard"
        assert entity.owner == "alice"

    async def test_agent_repo_dump_with_top_level_name(self, monkeypatch) -> None:
        """Repo dumps put name at the top level (no agent_card nesting)."""
        service = _build_service(
            monkeypatch,
            agent_match={
                "path": "/agents/bar",
                "name": "AgentTopLevel",
                "registered_by": "bob",
                "visibility": "public",
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Y",
                identity_url="https://api.example.com/agent",
            )
        )
        agent_collisions = [c for c in result.collision_with if c.entity_type == "a2a_agent"]
        assert len(agent_collisions) == 1
        assert agent_collisions[0].name == "AgentTopLevel"
        assert agent_collisions[0].owner == "bob"

    async def test_skill_hit_with_name_fallback(self, monkeypatch) -> None:
        """Skill hits typically have skill_name; ``name`` is the fallback."""
        service = _build_service(
            monkeypatch,
            search_results={
                "skills": [
                    {
                        "path": "/skills/foo",
                        "name": "FallbackName",  # no skill_name
                        "relevance_score": 0.85,
                    }
                ]
            },
        )
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert len(result.advisory_matches) == 1
        assert result.advisory_matches[0].name == "FallbackName"

    async def test_server_hit_with_name_fallback(self, monkeypatch) -> None:
        """Server hits typically have server_name; ``name`` is the fallback."""
        service = _build_service(
            monkeypatch,
            search_results={
                "servers": [
                    {
                        "path": "/server/foo",
                        "name": "FallbackName",  # no server_name
                        "relevance_score": 0.85,
                    }
                ]
            },
        )
        result = await service.check(**_check_kwargs(name="X", description="Y"))
        assert len(result.advisory_matches) == 1
        assert result.advisory_matches[0].name == "FallbackName"

    async def test_owner_falls_back_through_registered_by(self, monkeypatch) -> None:
        """When owner is missing, the extractor falls back to registered_by."""
        service = _build_service(
            monkeypatch,
            server_match={
                "path": "/foo",
                "server_name": "Foo",
                "registered_by": "carol",  # no explicit owner field
            },
        )
        result = await service.check(
            **_check_kwargs(
                name="Foo",
                identity_url="https://api.example.com/mcp",
            )
        )
        assert result.collision_with[0].owner == "carol"

    async def test_missing_registered_at_handled_gracefully(self, monkeypatch) -> None:
        """No registration timestamp on the doc -> registered_at is None."""
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo", "server_name": "Foo"},
        )
        result = await service.check(
            **_check_kwargs(
                name="Foo",
                identity_url="https://api.example.com/mcp",
            )
        )
        assert result.collision_with[0].registered_at is None


@pytest.mark.asyncio
class TestComputedHasCollision:
    """The ``has_collision`` computed field is True iff collision_with is non-empty."""

    async def test_has_collision_false_when_empty(self, monkeypatch) -> None:
        service = _build_service(monkeypatch)
        result = await service.check(**_check_kwargs(name="X"))
        assert result.has_collision is False

    async def test_has_collision_true_with_one_match(self, monkeypatch) -> None:
        service = _build_service(
            monkeypatch,
            server_match={"path": "/foo", "server_name": "Foo"},
        )
        result = await service.check(
            **_check_kwargs(name="X", identity_url="https://api.example.com/mcp")
        )
        assert result.has_collision is True

    async def test_has_collision_true_when_other_entity_type_repo_matches(
        self, monkeypatch
    ) -> None:
        """Cross-entity URL collisions surface through the same envelope.

        A registering server with the same URL as an existing skill
        produces a collision tagged with ``entity_type="skill"``.
        ``has_collision`` is the convenience flag for the frontend
        and is True when ``collision_with`` is non-empty regardless
        of which repo produced the hit.
        """
        service = _build_service(
            monkeypatch,
            server_match=None,
            skill_match={"path": "/skills/foo", "skill_name": "FooSkill"},
        )
        result = await service.check(
            **_check_kwargs(
                name="X",
                identity_url="https://github.com/org/repo",
            )
        )
        assert result.has_collision is True
        assert len(result.collision_with) == 1
        assert result.collision_with[0].entity_type == "skill"
