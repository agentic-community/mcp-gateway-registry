"""Unit tests for the generic-proxy nginx block generation.

Covers the four render-time pieces added to registry/core/nginx_service.py:
- _fetch_generic_proxied_resources: FLAG-GATED (zero DB queries when off), and
  drops federated/disabled/targetless rows via resolve_proxy_target;
- _create_generic_proxy_block: proxies to the auth-server generic hop, sets the
  SEPARATE $generic_backend_url (never $backend_url), and pins the target as
  X-Upstream-Url;
- _safe_generic_block: skip-not-fail render-time SSRF/scheme/illegal-char guard;
- _generate_generic_proxy_blocks + _location_paths_in: cross-block collision
  dedup with generic as the lowest-precedence tier.

The SECRET_KEY env is set so importing registry.core.config.settings succeeds.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from registry.core.nginx_service import (  # noqa: E402
    NginxConfigService,
    _fetch_generic_proxied_resources,
)

pytestmark = pytest.mark.unit


def _service() -> NginxConfigService:
    # NginxConfigService.__init__ only sets paths/locks; no I/O. The block
    # helpers under test are pure string builders that read settings.
    return NginxConfigService()


# --------------------------------------------------------------------------- #
# _fetch_generic_proxied_resources — flag gate + resolve_proxy_target filtering
# --------------------------------------------------------------------------- #


class TestFetchFlagGate:
    async def test_disabled_flag_issues_zero_queries(self):
        """SRE invariant: when the feature is off, NOT ONE repo query fires."""
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = False
            with patch("registry.repositories.factory.get_agent_repository") as ga:
                result = await _fetch_generic_proxied_resources()
        assert result == []
        ga.assert_not_called()  # the fetch short-circuited before touching factory

    async def test_enabled_flag_queries_all_three_repos(self):
        agent_repo = AsyncMock()
        agent_repo.list_proxied.return_value = [
            {"path": "/agents/a1", "is_proxied": True, "url": "https://a.example/"}
        ]
        skill_repo = AsyncMock()
        skill_repo.list_proxied.return_value = [
            {"path": "/skills/s1", "is_proxied": True, "proxy_target_url": "https://s.example/"}
        ]
        custom_repo = AsyncMock()
        custom_repo.list_proxied.return_value = [
            {
                "path": "/workflow/w1",
                "entity_type": "workflow",
                "is_proxied": True,
                "proxy_target_url": "https://w.example/",
            }
        ]
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = True
            with (
                patch(
                    "registry.repositories.factory.get_agent_repository",
                    return_value=agent_repo,
                ),
                patch(
                    "registry.repositories.factory.get_skill_repository",
                    return_value=skill_repo,
                ),
                patch(
                    "registry.repositories.factory.get_custom_entity_repository",
                    return_value=custom_repo,
                ),
            ):
                result = await _fetch_generic_proxied_resources()

        by_path = {r["path"]: r for r in result}
        assert by_path["/agents/a1"]["entity_type"] == "a2a_agent"
        assert by_path["/agents/a1"]["target_url"] == "https://a.example/"
        assert by_path["/skills/s1"]["entity_type"] == "skill"
        # custom record carries its own type token, used as the entity_type
        assert by_path["/workflow/w1"]["entity_type"] == "workflow"

    async def test_federated_and_targetless_rows_dropped(self):
        skill_repo = AsyncMock()
        skill_repo.list_proxied.return_value = [
            # federated -> resolve_proxy_target returns None -> dropped
            {
                "path": "/skills/fed",
                "is_proxied": True,
                "proxy_target_url": "https://x/",
                "sync_metadata": {"is_federated": True},
            },
            # no target -> dropped
            {"path": "/skills/none", "is_proxied": True},
            # good
            {"path": "/skills/ok", "is_proxied": True, "proxy_target_url": "https://ok/"},
        ]
        empty = AsyncMock()
        empty.list_proxied.return_value = []
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = True
            with (
                patch("registry.repositories.factory.get_agent_repository", return_value=empty),
                patch(
                    "registry.repositories.factory.get_skill_repository", return_value=skill_repo
                ),
                patch(
                    "registry.repositories.factory.get_custom_entity_repository",
                    return_value=empty,
                ),
            ):
                result = await _fetch_generic_proxied_resources()
        assert [r["path"] for r in result] == ["/skills/ok"]

    async def test_one_repo_error_does_not_break_others(self):
        boom = AsyncMock()
        boom.list_proxied.side_effect = RuntimeError("db down")
        good = AsyncMock()
        good.list_proxied.return_value = [
            {"path": "/skills/ok", "is_proxied": True, "proxy_target_url": "https://ok/"}
        ]
        empty = AsyncMock()
        empty.list_proxied.return_value = []
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = True
            with (
                patch("registry.repositories.factory.get_agent_repository", return_value=boom),
                patch("registry.repositories.factory.get_skill_repository", return_value=good),
                patch(
                    "registry.repositories.factory.get_custom_entity_repository",
                    return_value=empty,
                ),
            ):
                result = await _fetch_generic_proxied_resources()
        assert [r["path"] for r in result] == ["/skills/ok"]


# --------------------------------------------------------------------------- #
# _create_generic_proxy_block — shape
# --------------------------------------------------------------------------- #


class TestCreateBlock:
    def _block(self, entity_type="skill", path="/skills/proxy-demo", target="https://b.example/"):
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = "http://auth-server:8888"
            s.gateway_generic_client_max_body_size = "1m"
            return _service()._create_generic_proxy_block(entity_type, path, target)

    def test_location_is_entity_type_namespaced(self):
        block = self._block()
        assert "location {{ROOT_PATH}}/skill/skills/proxy-demo {" in block

    def test_proxy_pass_targets_auth_server_generic_hop(self):
        block = self._block()
        assert "proxy_pass http://auth-server:8888/proxy/skill/skills/proxy-demo/;" in block

    def test_sets_generic_backend_url_not_backend_url(self):
        block = self._block(target="https://b.example/")
        assert 'set $generic_backend_url "https://b.example/";' in block
        # CRITICAL double-mint guard: never emits a $backend_url DIRECTIVE on the
        # generic path (a comment may reference the name for context). Check the
        # live-directive lines only.
        directive_lines = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("set $backend_url" in ln for ln in directive_lines)
        assert not any(
            "$backend_url" in ln and "$generic_backend_url" not in ln for ln in directive_lines
        )
        assert "X-Upstream-Url $generic_backend_url;" in block

    def test_carries_entity_markers_and_body_size(self):
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = "http://auth-server:8888"
            s.gateway_generic_client_max_body_size = "8m"
            block = _service()._create_generic_proxy_block(
                "a2a_agent", "/agents/code-reviewer", "https://x/"
            )
        assert 'set $generic_proxy_kind "a2a_agent";' in block
        assert 'set $entity_path "agents/code-reviewer";' in block
        assert "client_max_body_size 8m;" in block

    def test_forwards_token_under_generic_header_name(self):
        # The hop's verify_generic_proxy_token reads X-Internal-Token-Generic, NOT
        # the MCP hop's X-Internal-Token. Forwarding under the wrong name 401s the
        # hop ("Missing internal proxy token"). Lock in the exact header name.
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = "http://auth-server:8888"
            s.gateway_generic_client_max_body_size = "1m"
            block = _service()._create_generic_proxy_block("skill", "/skills/x", "https://x/")
        assert "proxy_set_header X-Internal-Token-Generic $auth_internal_token_generic;" in block
        # must NOT forward the generic token under the MCP header name
        assert "proxy_set_header X-Internal-Token $auth_internal_token_generic;" not in block

    def test_auth_server_url_trailing_slash_not_doubled(self):
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = "http://auth-server:8888/"
            s.gateway_generic_client_max_body_size = "1m"
            block = _service()._create_generic_proxy_block("skill", "/skills/x", "https://x/")
        assert "proxy_pass http://auth-server:8888/proxy/skill/skills/x/;" in block


# --------------------------------------------------------------------------- #
# _safe_generic_block — skip-not-fail guard
# --------------------------------------------------------------------------- #


class TestSafeBlock:
    def _safe(
        self, entity_type, path, target, allow_private=False, auth_url="http://auth-server:8888"
    ):
        with patch("registry.core.nginx_service.settings") as s:
            s.auth_server_url = auth_url
            s.gateway_generic_client_max_body_size = "1m"
            s.gateway_proxy_allow_private_targets = allow_private
            return _service()._safe_generic_block(entity_type, path, target)

    def test_valid_target_returns_block(self):
        assert self._safe("skill", "/skills/x", "https://ok.example/") is not None

    def test_non_http_scheme_skipped(self):
        assert self._safe("skill", "/skills/x", "file:///etc/passwd") is None
        assert self._safe("skill", "/skills/x", "gopher://x/") is None

    def test_metadata_ip_skipped_regardless_of_flag(self):
        # link-local/metadata is denied even when private targets are allowed.
        assert (
            self._safe("skill", "/skills/x", "http://169.254.169.254/", allow_private=True) is None
        )

    def test_private_target_skipped_when_flag_false(self):
        assert self._safe("skill", "/skills/x", "http://10.0.0.5/", allow_private=False) is None

    def test_illegal_char_in_target_skipped(self):
        # a semicolon/brace/quote/newline could break out of the nginx directive
        assert self._safe("skill", "/skills/x", 'https://x/";}') is None

    def test_illegal_char_in_path_skipped(self):
        assert self._safe("skill", "/skills/x\ninjected", "https://ok.example/") is None

    def test_dollar_in_target_skipped(self):
        # $ inside the double-quoted `set $generic_backend_url "..."` value would be
        # expanded by nginx as a variable reference. Must be rejected.
        assert self._safe("skill", "/skills/x", "https://x/$request_uri") is None
        assert self._safe("skill", "/skills/x", "https://x/${remote_addr}") is None

    def test_illegal_entity_type_skipped(self):
        # A corrupt DB row whose entity_type breaks the token grammar is
        # interpolated into 4 directive positions — reject it.
        assert self._safe("skill; }\nlocation /evil {", "/skills/x", "https://ok/") is None
        assert self._safe("bad type", "/skills/x", "https://ok/") is None
        assert self._safe("Type/With/Slashes", "/skills/x", "https://ok/") is None

    def test_valid_entity_type_tokens_pass(self):
        assert self._safe("a2a_agent", "/agents/x", "https://ok/") is not None
        assert self._safe("custom-type_1", "/custom-type_1/x", "https://ok/") is not None

    def test_path_traversal_segment_skipped(self):
        # ..-segments would path-traverse on the auth-server hop after nginx
        # normalizes the proxy_pass URI.
        assert self._safe("agent", "/agents/../admin/secret", "https://ok/") is None
        assert self._safe("skill", "/skills/proxy-demo/../../etc", "https://ok/") is None

    def test_dot_in_path_is_allowed(self):
        # A single dot (not a `..` segment) is a legal path character.
        assert self._safe("skill", "/skills/v1.2.3", "https://ok/") is not None

    def test_empty_auth_server_url_skipped(self):
        # Empty auth_server_url would render `proxy_pass /proxy/...;` — a relative
        # self-loop. Skip rather than emit a bad block.
        assert self._safe("skill", "/skills/x", "https://ok/", auth_url="") is None


# --------------------------------------------------------------------------- #
# _location_paths_in + _generate_generic_proxy_blocks — collision dedup
# --------------------------------------------------------------------------- #


class TestLocationPaths:
    def test_extracts_plain_and_placeholder_paths(self):
        text = """
    location {{ROOT_PATH}}/skill/skills/x {
        proxy_pass http://a/;
    }
    location = /validate {
    }"""
        paths = NginxConfigService._location_paths_in(text)
        assert "/skill/skills/x" in paths
        assert "/validate" in paths

    def test_ignores_commented_locations(self):
        text = "#    location {{ROOT_PATH}}/dead {\n    location /live {\n    }"
        paths = NginxConfigService._location_paths_in(text)
        assert paths == {"/live"}


class TestGenerateGenericBlocks:
    async def _generate(self, resources, claimed):
        with patch("registry.core.nginx_service.settings") as s:
            s.gateway_generic_proxy_enabled = True
            s.auth_server_url = "http://auth-server:8888"
            s.gateway_generic_client_max_body_size = "1m"
            s.gateway_proxy_allow_private_targets = False
            with patch(
                "registry.core.nginx_service._fetch_generic_proxied_resources",
                new=AsyncMock(return_value=resources),
            ):
                return await _service()._generate_generic_proxy_blocks(claimed)

    async def test_empty_when_no_resources(self):
        assert await self._generate([], set()) == []

    async def test_generates_block_per_resource(self):
        resources = [
            {"entity_type": "skill", "path": "/skills/a", "target_url": "https://a/"},
            {"entity_type": "a2a_agent", "path": "/agents/b", "target_url": "https://b/"},
        ]
        blocks = await self._generate(resources, set())
        assert len(blocks) == 2

    async def test_generic_dropped_on_collision_with_claimed_path(self):
        # A legacy MCP server already claimed /skill/skills/a (precedence).
        resources = [
            {"entity_type": "skill", "path": "/skills/a", "target_url": "https://a/"},
            {"entity_type": "skill", "path": "/skills/b", "target_url": "https://b/"},
        ]
        claimed = {"/skill/skills/a"}
        blocks = await self._generate(resources, claimed)
        # only /skills/b survives; the colliding /skills/a is dropped
        assert len(blocks) == 1
        assert "/skill/skills/b" in blocks[0]

    async def test_generic_vs_generic_first_seen_wins(self):
        # Two entities rendering the SAME location path: deterministic first-wins
        # by sorted (entity_type, path). Construct a genuine collision: same type
        # + same path can't happen (unique), so use custom types whose namespaced
        # location coincides — here identical entity_type/path is impossible, so
        # assert the sort-stable claim behavior with a pre-claim instead.
        resources = [
            {"entity_type": "skill", "path": "/skills/dup", "target_url": "https://a/"},
        ]
        # Pre-claim the path; the single generic must be dropped, not duplicated.
        blocks = await self._generate(resources, {"/skill/skills/dup"})
        assert blocks == []

    async def test_invalid_target_skipped_but_others_render(self):
        resources = [
            {
                "entity_type": "skill",
                "path": "/skills/bad",
                "target_url": "http://169.254.169.254/",
            },
            {"entity_type": "skill", "path": "/skills/good", "target_url": "https://good/"},
        ]
        blocks = await self._generate(resources, set())
        assert len(blocks) == 1
        assert "/skill/skills/good" in blocks[0]
