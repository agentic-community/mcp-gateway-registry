"""Unit tests for registry/auth/resource_binding.py (Issue #944)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.auth import resource_binding
from registry.auth.resource_binding import ResourceType

pytestmark = [pytest.mark.unit, pytest.mark.auth]


class TestIsIntrospectionPath:
    """The edge guard uses is_resource_token_introspection_path to let
    resource-bound tokens hit /api/auth/me, which doesn't classify to any
    (type, id) but is safe for every token."""

    def test_auth_me_is_introspection(self) -> None:
        assert resource_binding.is_resource_token_introspection_path("/api/auth/me") is True

    def test_trailing_slash_still_introspection(self) -> None:
        assert resource_binding.is_resource_token_introspection_path("/api/auth/me/") is True

    def test_root_path_prefix_handled(self) -> None:
        assert (
            resource_binding.is_resource_token_introspection_path(
                "/registry/api/auth/me", root_path="/registry"
            )
            is True
        )

    def test_non_allow_listed_path_not_introspection(self) -> None:
        assert resource_binding.is_resource_token_introspection_path("/api/agents/foo") is False
        # Even paths under the allow-listed prefix but not the exact path
        # must not be treated as introspection.
        assert resource_binding.is_resource_token_introspection_path("/api/auth/logout") is False


class TestAllowList:
    def test_auth_me_is_allowed(self) -> None:
        assert "/api/auth/me" in resource_binding.RESOURCE_TOKEN_ALLOWED_PATHS

    def test_allow_list_beats_block_list(self) -> None:
        # /api/auth/me is under the blocked /api/auth prefix but carved out.
        assert resource_binding.check_resource_token_allowed("/api/auth/me") is True
        # Other /api/auth/* paths remain blocked.
        assert resource_binding.check_resource_token_allowed("/api/auth/logout") is False


class TestSharedConstantsExported:
    def test_blocked_prefixes_cover_known_escalation_paths(self) -> None:
        # Everything on this list must remain blocked. Removing one is a
        # security decision that should require an explicit test change.
        required = {
            "/api/tokens",
            "/api/admin",
            "/api/search",
            "/api/auth",
        }
        assert required.issubset(set(resource_binding.RESOURCE_TOKEN_BLOCKED_PREFIXES))

    def test_api_prefix_map_lists_agent_and_skill(self) -> None:
        prefixes = {p for p, _ in resource_binding.API_PREFIX_TO_TYPE}
        assert "/api/agents/" in prefixes
        assert "/api/skills/" in prefixes

    def test_transport_segments_cover_mcp_variants(self) -> None:
        assert resource_binding.MCP_TRANSPORT_SEGMENTS == frozenset({"mcp", "sse", "messages"})

    def test_classify_honors_module_transport_segments(self, monkeypatch) -> None:
        """classify_request_url must reference the module-level
        constant so adding a new transport segment flows through.
        If the function re-declares its own local copy, this test
        fails.
        """
        monkeypatch.setattr(
            resource_binding,
            "MCP_TRANSPORT_SEGMENTS",
            frozenset({"mcp", "sse", "messages", "ws"}),
        )
        # "ws" is a transport segment, it should be stripped from the resource id.
        assert resource_binding.classify_request_url("/cloudflare-docs/ws") == (
            ResourceType.SERVER,
            "cloudflare-docs",
        )


class TestResourceTypeEnum:
    def test_values_stable(self) -> None:
        # These string values are persisted in JWT claims; renaming breaks
        # in-flight tokens. Guard against accidental changes.
        assert ResourceType.SERVER.value == "server"
        assert ResourceType.VIRTUAL_SERVER.value == "virtual_server"
        assert ResourceType.AGENT.value == "agent"
        assert ResourceType.SKILL.value == "skill"

    def test_tuple_matches_enum(self) -> None:
        assert set(resource_binding.RESOURCE_TYPES) == {rt.value for rt in ResourceType}

    def test_string_comparison(self) -> None:
        # str Enum allows direct comparison to string wire values.
        assert ResourceType.SERVER == "server"


class TestTokenKindEnum:
    def test_values_stable(self) -> None:
        # Persisted in JWT claims; renaming is a breaking wire change.
        assert resource_binding.TokenKind.USER.value == "user"
        assert resource_binding.TokenKind.RESOURCE.value == "resource"

    def test_string_comparison(self) -> None:
        # str Enum — equality against the wire value must work without .value.
        assert resource_binding.TokenKind.USER == "user"
        assert resource_binding.TokenKind.RESOURCE == "resource"

    def test_claim_name_constants_stable(self) -> None:
        # Claim-name spelling is wire format. Enforced here so the mint
        # side and the edge guard can never accidentally diverge.
        assert resource_binding.TOKEN_KIND_CLAIM == "token_kind"
        assert resource_binding.RESOURCE_TYPE_CLAIM == "resource_type"
        assert resource_binding.RESOURCE_ID_CLAIM == "resource_id"


class TestNormalizeResourceId:
    def test_strips_leading_slash(self) -> None:
        assert resource_binding.normalize_resource_id("/my-server") == "my-server"

    def test_no_leading_slash_preserved(self) -> None:
        assert resource_binding.normalize_resource_id("my-server") == "my-server"

    def test_strips_whitespace(self) -> None:
        assert resource_binding.normalize_resource_id("  /my-server  ") == "my-server"

    def test_preserves_internal_slashes(self) -> None:
        assert (
            resource_binding.normalize_resource_id("/peer/cloudflare-docs")
            == "peer/cloudflare-docs"
        )

    def test_strips_trailing_slash(self) -> None:
        # A resource id minted as "cloudflare-docs/" must normalize
        # identically to "cloudflare-docs" so the edge comparison against
        # classify_request_url's output (never has a trailing slash)
        # succeeds.
        assert resource_binding.normalize_resource_id("cloudflare-docs/") == "cloudflare-docs"
        assert resource_binding.normalize_resource_id("/cloudflare-docs/") == "cloudflare-docs"


class TestClassifyRequestUrl:
    @pytest.mark.parametrize(
        "path,expected",
        [
            # Agents (under /api/agents/)
            ("/api/agents/code-reviewer", (ResourceType.AGENT, "code-reviewer")),
            ("/api/agents/code-reviewer/health", (ResourceType.AGENT, "code-reviewer")),
            # Skills (under /api/skills/)
            ("/api/skills/python-linter", (ResourceType.SKILL, "skills/python-linter")),
            ("/api/skills/python-linter/rate", (ResourceType.SKILL, "skills/python-linter")),
            ("/api/skills/python-linter/content", (ResourceType.SKILL, "skills/python-linter")),
            # Virtual servers (under /virtual/)
            ("/virtual/my-agg", (ResourceType.VIRTUAL_SERVER, "virtual/my-agg")),
            ("/virtual/my-agg/mcp", (ResourceType.VIRTUAL_SERVER, "virtual/my-agg")),
            ("/virtual/my-agg/sse", (ResourceType.VIRTUAL_SERVER, "virtual/my-agg")),
            # Plain MCP servers (flat paths)
            ("/cloudflare-docs", (ResourceType.SERVER, "cloudflare-docs")),
            ("/cloudflare-docs/mcp", (ResourceType.SERVER, "cloudflare-docs")),
            ("/cloudflare-docs/sse", (ResourceType.SERVER, "cloudflare-docs")),
            # Federated servers (peer/server)
            (
                "/peer-registry-lob-1/cloudflare-docs",
                (ResourceType.SERVER, "peer-registry-lob-1/cloudflare-docs"),
            ),
            (
                "/peer-registry-lob-1/cloudflare-docs/mcp",
                (ResourceType.SERVER, "peer-registry-lob-1/cloudflare-docs"),
            ),
            # REST metadata endpoint: a server-bound token should reach
            # /api/servers/<slug> alongside /<slug>/mcp. Classifier must
            # produce the same resource_id as mint-time normalization.
            ("/api/servers/cloudflare-docs", (ResourceType.SERVER, "cloudflare-docs")),
            (
                "/api/servers/cloudflare-docs/tools",
                (ResourceType.SERVER, "cloudflare-docs"),
            ),
        ],
    )
    def test_classifies_known_paths(self, path: str, expected: tuple[ResourceType, str]) -> None:
        assert resource_binding.classify_request_url(path) == expected

    def test_federated_server_rest_metadata_mismatches_mcp_binding(self) -> None:
        """Documented restriction: federated server ids are multi-segment
        (``peer-registry/cf-docs``) on the MCP path but the REST
        metadata classifier takes only the first segment.

        A federated-server-bound token minted from the MCP path
        classifies as ``peer-registry/cf-docs`` but the REST endpoint
        ``/api/servers/peer-registry/cf-docs`` classifies as
        ``peer-registry``. The mismatch fails closed (403) at the edge
        — federated bound tokens reach the MCP transport only, not the
        REST metadata endpoint. This test locks in the intentional
        asymmetry noted in the ``API_PREFIX_TO_TYPE`` docstring so a
        future refactor does not silently change the behavior.
        """
        mcp_path = "/peer-registry/cf-docs/mcp"
        rest_path = "/api/servers/peer-registry/cf-docs"
        mcp_result = resource_binding.classify_request_url(mcp_path)
        rest_result = resource_binding.classify_request_url(rest_path)
        assert mcp_result == (ResourceType.SERVER, "peer-registry/cf-docs")
        # The REST classifier stops at the first segment, so these
        # resource_ids differ — a bound token cannot traverse both.
        assert rest_result == (ResourceType.SERVER, "peer-registry")
        assert mcp_result != rest_result

    @pytest.mark.parametrize(
        "path",
        [
            # Registry API paths that are not themselves a resource binding
            "/api/tokens/generate",
            "/api/admin/users",
            "/api/search/semantic",
            "/api/health",
            "/api/stats",
            "/api/",
            "/api",
            "",
            "/",
        ],
    )
    def test_unclassifiable_paths_return_none(self, path: str) -> None:
        assert resource_binding.classify_request_url(path) is None

    def test_strips_query_string(self) -> None:
        assert resource_binding.classify_request_url("/api/agents/code-reviewer?foo=bar") == (
            ResourceType.AGENT,
            "code-reviewer",
        )

    def test_adds_leading_slash_if_missing(self) -> None:
        assert resource_binding.classify_request_url("api/agents/foo") == (
            ResourceType.AGENT,
            "foo",
        )


class TestCheckResourceTokenAllowed:
    @pytest.mark.parametrize(
        "path,expected",
        [
            # Resource paths are allowed
            ("/api/agents/foo", True),
            ("/api/skills/foo", True),
            ("/virtual/foo", True),
            ("/cloudflare-docs/mcp", True),
            # Blocked prefixes: token mint, admin, search, federation, auth...
            ("/api/tokens/generate", False),
            ("/api/tokens", False),
            ("/api/admin/anything", False),
            ("/api/admin", False),
            ("/api/search/semantic", False),
            ("/api/federation/peers", False),
            ("/api/auth/logout", False),
            # /api/auth/me is explicitly carved out (introspection endpoint).
            ("/api/auth/me", True),
            # Prefix specificity — '/api/administration' is NOT '/api/admin'
            ("/api/administration", True),
        ],
    )
    def test_blocked_prefixes(self, path: str, expected: bool) -> None:
        assert resource_binding.check_resource_token_allowed(path) is expected

    def test_trailing_slash_on_allow_list_path(self) -> None:
        """A trailing slash on an allow-listed path must still match so that
        clients hitting /api/auth/me/ (common redirect behavior) are not
        accidentally blocked."""
        assert resource_binding.check_resource_token_allowed("/api/auth/me/") is True
        assert resource_binding.check_resource_token_allowed("/api/auth/me") is True

    @pytest.mark.parametrize(
        "path",
        [
            # Semicolon path parameter — urlparse keeps this in .path so
            # the exact-match allow-list does not match. Fail-closed.
            "/api/auth/me;evil=1",
            # URL fragments belong in .fragment, not .path — but explicit
            # lock-in test in case a future refactor uses raw URL parsing.
            "/api/auth/me#frag",
        ],
    )
    def test_introspection_check_rejects_smuggled_paths(self, path: str) -> None:
        # None of these should be treated as the /api/auth/me introspection
        # path. Fail-closed is correct here — resource-bound tokens get
        # routed to the general classify-and-match logic.
        assert resource_binding.is_resource_token_introspection_path(path) is False

    @pytest.mark.parametrize(
        "path",
        [
            "/api//auth/me",
            "//api/auth/me",
            "///api///auth///me",
        ],
    )
    def test_introspection_check_accepts_double_slash_variants(self, path: str) -> None:
        # After slash-collapse normalization, double-slash variants of
        # /api/auth/me reach the introspection endpoint. This is safe:
        # the endpoint is public-by-design for introspection, and the
        # normalization is applied symmetrically to the deny-list so
        # ``//api/tokens/generate`` cannot use the same trick to bypass
        # blocking.
        assert resource_binding.is_resource_token_introspection_path(path) is True

    def test_double_slash_on_deny_list_path_still_blocked(self) -> None:
        """If the upstream proxy does not merge slashes
        (Traefik, Envoy, nginx with ``merge_slashes off``), a path like
        ``//api/tokens/generate`` must still hit the deny-list. The
        helpers canonicalize the path with a slash-collapse so the prefix
        match sees ``/api/tokens/generate`` regardless of upstream
        sanitization.
        """
        # Well-formed single-slash path is blocked.
        assert resource_binding.check_resource_token_allowed("/api/tokens/generate") is False
        # Double-slash variants must also be blocked after normalization.
        assert resource_binding.check_resource_token_allowed("//api/tokens/generate") is False
        assert resource_binding.check_resource_token_allowed("/api//tokens//generate") is False
        assert resource_binding.check_resource_token_allowed("///api///tokens") is False

    def test_root_path_containing_blocked_prefix(self) -> None:
        """Pathological case: registry mounted under a sub-path that
        itself contains a blocked prefix substring (e.g. /api/admin).
        The prefix strip must run first so the deny-list check sees the
        canonical path.
        """
        # If the registry is hosted at /api/admin,
        # a request to /api/admin/api/agents/foo has root_path stripped
        # first, leaving /api/agents/foo — a normal agent path, allowed.
        assert (
            resource_binding.check_resource_token_allowed(
                "/api/admin/api/agents/foo", root_path="/api/admin"
            )
            is True
        )
        # And the nominally-blocked /api/admin prefix under that root
        # becomes a simple /api/agents path too.
        assert (
            resource_binding.check_resource_token_allowed(
                "/api/admin/api/tokens/generate", root_path="/api/admin"
            )
            is False
        )

    def test_root_path_with_whitespace_ignored(self) -> None:
        """Whitespace-only root_path (misconfiguration) must not corrupt
        the prefix match. Should behave as if no root_path was given."""
        assert resource_binding.check_resource_token_allowed("/api/admin", root_path="  ") is False
        assert (
            resource_binding.check_resource_token_allowed("/api/agents/foo", root_path="  ") is True
        )

    def test_root_path_prefix_stripped(self) -> None:
        """Registry hosted under a sub-path: /registry/api/admin/x should be
        blocked the same way /api/admin/x is."""
        assert (
            resource_binding.check_resource_token_allowed(
                "/registry/api/admin/users", root_path="/registry"
            )
            is False
        )
        assert (
            resource_binding.check_resource_token_allowed(
                "/registry/api/auth/me", root_path="/registry"
            )
            is True
        )


class TestValidateUserCanBindResource:
    @pytest.mark.asyncio
    async def test_admin_can_bind_any_resource(self) -> None:
        result = await resource_binding.validate_user_can_bind_resource(
            resource_type="server",
            resource_id="/any-server",
            user_context={"is_admin": True},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_server_allow_when_in_accessible(self) -> None:
        result = await resource_binding.validate_user_can_bind_resource(
            resource_type="server",
            resource_id="/cloudflare-docs",
            user_context={
                "is_admin": False,
                "accessible_servers": ["cloudflare-docs", "github-docs"],
            },
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_server_path_grant_allows_exact_binding_only(self) -> None:
        context = {"is_admin": False, "accessible_servers": ["/journey-mcp"]}
        own = await resource_binding.validate_user_can_bind_resource(
            resource_type="server", resource_id="/journey-mcp", user_context=context
        )
        other = await resource_binding.validate_user_can_bind_resource(
            resource_type="server", resource_id="/other-mcp", user_context=context
        )
        assert own is True and other is False

    @pytest.mark.asyncio
    async def test_server_deny_when_not_in_accessible(self) -> None:
        result = await resource_binding.validate_user_can_bind_resource(
            resource_type="server",
            resource_id="/github-docs",
            user_context={
                "is_admin": False,
                "accessible_servers": ["cloudflare-docs"],
            },
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_server_wildcard_all_allows(self) -> None:
        result = await resource_binding.validate_user_can_bind_resource(
            resource_type="server",
            resource_id="/anything",
            user_context={"accessible_servers": ["all"]},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_virtual_server_checked_against_ui_permission(self) -> None:
        ctx = {
            "is_admin": False,
            "ui_permissions": {"list_virtual_server": ["/virtual/my-agg"]},
        }
        assert (
            await resource_binding.validate_user_can_bind_resource(
                "virtual_server", "virtual/my-agg", ctx
            )
            is True
        )
        assert (
            await resource_binding.validate_user_can_bind_resource(
                "virtual_server", "virtual/other", ctx
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_virtual_server_all_permission(self) -> None:
        ctx = {
            "is_admin": False,
            "ui_permissions": {"list_virtual_server": ["all"]},
        }
        assert (
            await resource_binding.validate_user_can_bind_resource(
                "virtual_server", "virtual/anything", ctx
            )
            is True
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wildcard", ["all", "*"])
    async def test_wildcard_access_consistent_across_types(self, wildcard: str) -> None:
        """Different scope/permission stores have used both ``"all"`` and
        ``"*"`` as the "no restriction" sentinel. All three resource
        types must accept both to avoid denials when one config
        format happens to be used in one place."""
        server_ctx = {"is_admin": False, "accessible_servers": [wildcard]}
        virtual_ctx = {
            "is_admin": False,
            "ui_permissions": {"list_virtual_server": [wildcard]},
        }
        agent_ctx = {
            "is_admin": False,
            "accessible_agents": [wildcard],
            "username": "me",
            "groups": [],
        }
        assert (
            await resource_binding.validate_user_can_bind_resource("server", "anything", server_ctx)
            is True
        )
        assert (
            await resource_binding.validate_user_can_bind_resource(
                "virtual_server", "virtual/anything", virtual_ctx
            )
            is True
        )
        # Agent still has to pass the visibility check — mock the service
        # to return a public agent so the wildcard check is what gates.
        from registry.services import agent_service as agent_mod

        fake_agent = MagicMock()
        fake_agent.visibility = "public"
        fake_agent.registered_by = "someone-else"
        fake_agent.allowed_groups = []
        with patch.object(
            agent_mod.agent_service,
            "get_agent_info",
            new=AsyncMock(return_value=fake_agent),
        ):
            assert (
                await resource_binding.validate_user_can_bind_resource(
                    "agent", "anything", agent_ctx
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_agent_public_visibility_allowed_when_in_list(self) -> None:
        from registry.services import agent_service as agent_mod

        fake_agent = MagicMock()
        fake_agent.visibility = "public"
        fake_agent.registered_by = "someone-else"
        fake_agent.allowed_groups = []
        with patch.object(
            agent_mod.agent_service,
            "get_agent_info",
            new=AsyncMock(return_value=fake_agent),
        ):
            ctx = {
                "is_admin": False,
                "accessible_agents": ["code-reviewer"],
                "username": "me",
                "groups": [],
            }
            assert (
                await resource_binding.validate_user_can_bind_resource(
                    "agent", "code-reviewer", ctx
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_agent_not_in_accessible_denied(self) -> None:
        ctx = {
            "is_admin": False,
            "accessible_agents": ["other-agent"],
            "username": "me",
            "groups": [],
        }
        assert (
            await resource_binding.validate_user_can_bind_resource("agent", "code-reviewer", ctx)
            is False
        )

    @pytest.mark.asyncio
    async def test_skill_public_allowed(self) -> None:
        from registry.schemas.skill_models import VisibilityEnum

        fake_skill = MagicMock()
        fake_skill.visibility = VisibilityEnum.PUBLIC
        fake_skill.owner = "someone-else"
        fake_skill.allowed_groups = []
        fake_service = MagicMock()
        fake_service.get_skill = AsyncMock(return_value=fake_skill)
        with patch(
            "registry.services.skill_service.get_skill_service",
            return_value=fake_service,
        ):
            ctx = {"is_admin": False, "username": "me", "groups": []}
            assert (
                await resource_binding.validate_user_can_bind_resource(
                    "skill", "python-linter", ctx
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_skill_private_only_owner(self) -> None:
        from registry.schemas.skill_models import VisibilityEnum

        fake_skill = MagicMock()
        fake_skill.visibility = VisibilityEnum.PRIVATE
        fake_skill.owner = "owner-user"
        fake_skill.allowed_groups = []
        fake_service = MagicMock()
        fake_service.get_skill = AsyncMock(return_value=fake_skill)
        with patch(
            "registry.services.skill_service.get_skill_service",
            return_value=fake_service,
        ):
            owner_ctx = {"is_admin": False, "username": "owner-user", "groups": []}
            stranger_ctx = {"is_admin": False, "username": "stranger", "groups": []}
            assert (
                await resource_binding.validate_user_can_bind_resource(
                    "skill", "/my-skill", owner_ctx
                )
                is True
            )
            assert (
                await resource_binding.validate_user_can_bind_resource(
                    "skill", "/my-skill", stranger_ctx
                )
                is False
            )

    @pytest.mark.asyncio
    async def test_unknown_resource_type_rejected(self) -> None:
        assert (
            await resource_binding.validate_user_can_bind_resource(
                "totally-unknown", "foo", {"is_admin": True}
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_accepts_enum_or_string(self) -> None:
        ctx = {"is_admin": False, "accessible_servers": ["foo"]}
        # Same input, once as enum, once as string — both must behave
        # identically.
        assert (
            await resource_binding.validate_user_can_bind_resource(ResourceType.SERVER, "foo", ctx)
            is True
        )
        assert await resource_binding.validate_user_can_bind_resource("server", "foo", ctx) is True

    @pytest.mark.asyncio
    async def test_agent_lookup_timeout_denies_bind(self) -> None:
        """If the DB hangs on get_agent_info, the mint request must not
        hang indefinitely — we bound the lookup with asyncio.wait_for
        and a timeout treats the bind as denied (fail-closed)."""
        import asyncio as _asyncio

        async def _hang(*_a, **_k):
            await _asyncio.sleep(10)

        from registry.services import agent_service as agent_mod

        ctx = {
            "is_admin": False,
            "accessible_agents": ["code-reviewer"],
            "username": "me",
            "groups": [],
        }
        with (
            patch.object(agent_mod.agent_service, "get_agent_info", new=_hang),
            patch.object(resource_binding, "_BIND_CHECK_LOOKUP_TIMEOUT_SECONDS", 0.05),
        ):
            result = await resource_binding.validate_user_can_bind_resource(
                "agent", "code-reviewer", ctx
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_skill_lookup_timeout_denies_bind(self) -> None:
        import asyncio as _asyncio

        async def _hang(*_a, **_k):
            await _asyncio.sleep(10)

        fake_service = MagicMock()
        fake_service.get_skill = _hang
        ctx = {"is_admin": False, "username": "me", "groups": []}
        with (
            patch(
                "registry.services.skill_service.get_skill_service",
                return_value=fake_service,
            ),
            patch.object(resource_binding, "_BIND_CHECK_LOOKUP_TIMEOUT_SECONDS", 0.05),
        ):
            result = await resource_binding.validate_user_can_bind_resource(
                "skill", "python-linter", ctx
            )
        assert result is False


class TestNormalizePathTraversal:
    """The classifier and deny-list consume an attacker-controlled raw request
    URI (nginx forwards $request_uri percent-encoded). _normalize_path must
    percent-decode and resolve ../. segments so an encoded traversal
    classifies identically to its canonical form and cannot bypass the
    byte-exact deny-list."""

    def test_encoded_and_literal_traversal_match_canonical(self) -> None:
        # Encoded and literal dot-dot must normalize to the same canonical path
        # as the fully-decoded equivalent.
        encoded = resource_binding._normalize_path("/api/agents/%2e%2e/tokens/generate")
        literal = resource_binding._normalize_path("/api/agents/../tokens/generate")
        canonical = resource_binding._normalize_path("/api/tokens/generate")
        assert encoded == literal == canonical == "/api/tokens/generate"

    def test_mixed_encoding_matches_canonical(self) -> None:
        mixed = resource_binding._normalize_path("/api/agents/%2e./tokens/generate")
        assert mixed == "/api/tokens/generate"

    def test_encoded_traversal_to_blocked_prefix_is_denied(self) -> None:
        # An encoded traversal that resolves onto a deny-listed prefix must be
        # denied, exactly like the literal form.
        assert (
            resource_binding.check_resource_token_allowed("/api/agents/%2e%2e/tokens/generate")
            is False
        )
        assert resource_binding.check_resource_token_allowed("/api/x/%2e%2e/admin/config") is False

    def test_encoded_traversal_classifies_like_canonical(self) -> None:
        assert resource_binding.classify_request_url(
            "/api/agents/%2e%2e/skills/foo"
        ) == resource_binding.classify_request_url("/api/skills/foo")

    def test_escape_above_root_is_clamped_not_allowed(self) -> None:
        # A traversal trying to escape above root is clamped to an absolute
        # path, never a relative/ambiguous one.
        assert resource_binding._normalize_path("/api/%2e%2e/%2e%2e/etc/passwd") == "/etc/passwd"

    def test_double_encoding_stays_literal(self) -> None:
        # Decode exactly once: a doubly-encoded %252e stays literal and does
        # NOT resolve to a traversal (fail closed toward the more restrictive
        # outcome -- it will not silently reach a blocked prefix).
        normalized = resource_binding._normalize_path("/api/agents/foo/%252e%252e/tokens")
        assert "%2e%2e" in normalized
        assert normalized == "/api/agents/foo/%2e%2e/tokens"

    def test_benign_path_unchanged(self) -> None:
        assert (
            resource_binding._normalize_path("/api/agents/code-reviewer")
            == "/api/agents/code-reviewer"
        )

    def test_consecutive_slashes_still_collapsed(self) -> None:
        assert resource_binding._normalize_path("//api//tokens//generate") == "/api/tokens/generate"
        assert resource_binding.check_resource_token_allowed("//api//tokens//generate") is False
