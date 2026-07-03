"""
Unit tests for resource discovery (issue #938 follow-up).

Covers:
  - _resolve_tree_api: blob URLs, raw URLs, GHES URLs, malformed input.
  - _classify_resource: subfolder convention + flat-skill fallback.
  - _discover_skill_resources: GitHub Trees API response shape, auth
    header merge, flat-skill happy path, mixed-classification, hidden
    file exclusion.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.schemas.skill_models import SkillResourceManifest
from registry.services.skill_service import (
    _classify_resource,
    _discover_skill_resources,
    _resolve_tree_api,
)

logger = logging.getLogger(__name__)


# =============================================================================
# _resolve_tree_api
# =============================================================================


class TestResolveTreeApi:
    """URL parsing for the GitHub/GHES Trees API."""

    def test_public_blob_url(self):
        url = "https://github.com/anthropics/skills/blob/main/example/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        tree_url, project, ref, skill_dir = result
        assert (
            tree_url == "https://api.github.com/repos/anthropics/skills/git/trees/main?recursive=1"
        )
        assert project == "anthropics/skills"
        assert ref == "main"
        assert skill_dir == "example"

    def test_public_raw_url(self):
        url = "https://raw.githubusercontent.com/anthropics/skills/refs/heads/main/example/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        tree_url, _, ref, skill_dir = result
        assert (
            tree_url == "https://api.github.com/repos/anthropics/skills/git/trees/main?recursive=1"
        )
        assert ref == "main"
        assert skill_dir == "example"

    def test_skill_at_repo_root_yields_empty_skill_dir(self):
        url = "https://github.com/foo/bar/blob/main/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        _, _, _, skill_dir = result
        assert skill_dir == ""

    def test_nested_skill_path(self):
        url = (
            "https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/"
            "refs/heads/main/.claude/skills/usage-report/SKILL.md"
        )
        result = _resolve_tree_api(url)
        assert result is not None
        _, project, _, skill_dir = result
        assert project == "agentic-community/mcp-gateway-registry"
        assert skill_dir == ".claude/skills/usage-report"

    def test_branch_name_with_special_chars_is_preserved(self):
        url = "https://github.com/foo/bar/blob/feature%2Fnew/skill/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        _, _, ref, _ = result
        # Branch is taken verbatim from the URL path (already URL-encoded).
        assert ref == "feature%2Fnew"

    def test_non_github_returns_none(self):
        url = "https://gitlab.com/foo/bar/blob/main/SKILL.md"
        assert _resolve_tree_api(url) is None

    def test_malformed_path_returns_none(self):
        url = "https://github.com/just/two-segments/SKILL.md"
        assert _resolve_tree_api(url) is None

    def test_missing_hostname_returns_none(self):
        assert _resolve_tree_api("not a url at all") is None

    @patch("registry.services.skill_service.settings")
    def test_ghes_blob_url_with_matching_api_base(
        self,
        mock_settings,
    ):
        mock_settings.github_api_base_url = "https://github.mycompany.com/api/v3"
        url = "https://github.mycompany.com/team/repo/blob/develop/skills/x/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        tree_url, project, ref, skill_dir = result
        assert tree_url == (
            "https://github.mycompany.com/api/v3/repos/team/repo/git/trees/develop?recursive=1"
        )
        assert project == "team/repo"
        assert ref == "develop"
        assert skill_dir == "skills/x"

    @patch("registry.services.skill_service.settings")
    def test_ghes_raw_url_with_matching_api_base(
        self,
        mock_settings,
    ):
        mock_settings.github_api_base_url = "https://github.mycompany.com/api/v3"
        url = "https://raw.github.mycompany.com/team/repo/refs/heads/main/skills/x/SKILL.md"
        result = _resolve_tree_api(url)
        assert result is not None
        tree_url, _, _, _ = result
        assert tree_url.startswith("https://github.mycompany.com/api/v3/")

    @patch("registry.services.skill_service.settings")
    def test_ghes_url_when_api_base_does_not_match(
        self,
        mock_settings,
    ):
        # Configured api base host (github.acme.com) doesn't match the
        # SKILL.md host (github.mycompany.com) -- we cannot safely guess
        # the API endpoint, so return None.
        mock_settings.github_api_base_url = "https://github.acme.com/api/v3"
        url = "https://github.mycompany.com/team/repo/blob/main/SKILL.md"
        assert _resolve_tree_api(url) is None


# =============================================================================
# _classify_resource
# =============================================================================


class TestClassifyResource:
    """Two-tier classification: subfolder convention then extension fallback."""

    def test_subfolder_references(self):
        assert _classify_resource("example/references/arch.md", "example") == "reference"

    def test_subfolder_scripts(self):
        assert _classify_resource("example/scripts/run.sh", "example") == "script"

    def test_subfolder_agents(self):
        assert _classify_resource("example/agents/coder.md", "example") == "agent"

    def test_subfolder_assets(self):
        assert _classify_resource("example/assets/diagram.png", "example") == "asset"

    def test_flat_skill_python_is_script(self):
        assert _classify_resource("usage-report/run.py", "usage-report") == "script"

    def test_flat_skill_shell_is_script(self):
        assert _classify_resource("usage-report/install.sh", "usage-report") == "script"

    def test_flat_skill_markdown_is_reference(self):
        assert _classify_resource("usage-report/notes.md", "usage-report") == "reference"

    def test_flat_skill_unknown_extension_is_asset(self):
        assert _classify_resource("usage-report/style.css", "usage-report") == "asset"

    def test_flat_skill_root_skill_dir(self):
        # SKILL.md is at the repo root; flat fallback uses skill_dir == "".
        assert _classify_resource("run.py", "") == "script"

    def test_nested_unrecognised_subfolder_returns_none(self):
        # tests/test_run.py is not a recognised resource subfolder, and it's
        # not directly under the skill_dir, so it should not be classified.
        assert _classify_resource("usage-report/tests/test_run.py", "usage-report") is None

    def test_file_without_extension_returns_none(self):
        assert _classify_resource("usage-report/Makefile", "usage-report") is None


# =============================================================================
# _discover_skill_resources (integration-style with mocked HTTP)
# =============================================================================


def _trees_payload(items: list[dict], truncated: bool = False) -> dict:
    """Build a GitHub Trees API response envelope."""
    return {
        "sha": "deadbeef",
        "url": "https://api.github.com/...",
        "tree": items,
        "truncated": truncated,
    }


class TestDiscoverSkillResources:
    """End-to-end discovery: URL -> tree fetch -> classified manifest."""

    @pytest.fixture
    def mock_github_auth(self):
        with patch(
            "registry.services.skill_service._github_auth",
        ) as auth:
            auth.get_auth_headers = AsyncMock(return_value={})
            yield auth

    @pytest.fixture
    def mock_async_client(self):
        """Patch the guarded async client and yield the mock response object."""
        with patch("registry.services.skill_service.guarded_async_client") as client_cls:
            response = MagicMock()
            response.status_code = 200
            response.json = MagicMock()
            # Empty dict so the GitLab pagination loop never starts: a bare
            # MagicMock would make .get("x-next-page", "") return a truthy
            # sentinel, and because json() returns one shared list the loop's
            # payload.extend(payload) would double the list every iteration and
            # OOM the worker.
            response.headers = {}

            client_instance = MagicMock()
            client_instance.get = AsyncMock(return_value=response)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            client_cls.return_value = client_instance

            yield response

    async def test_returns_none_when_url_unrecognised(self):
        result = await _discover_skill_resources("https://gitlab.com/foo/bar/SKILL.md")
        assert result is None

    async def test_subfolder_skill_classified_correctly(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": "example/SKILL.md", "size": 1000},
                {"type": "blob", "path": "example/references/arch.md", "size": 200},
                {"type": "blob", "path": "example/scripts/run.sh", "size": 300},
                {"type": "blob", "path": "example/agents/coder.md", "size": 400},
                {"type": "blob", "path": "example/assets/diagram.png", "size": 500},
                # Tree entries other than the skill folder must be ignored.
                {"type": "blob", "path": "other-skill/SKILL.md", "size": 999},
                {"type": "tree", "path": "example/references", "size": 0},
            ]
        )

        manifest = await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/example/SKILL.md",
        )
        assert isinstance(manifest, SkillResourceManifest)
        assert [r.path for r in manifest.references] == ["references/arch.md"]
        assert [r.path for r in manifest.scripts] == ["scripts/run.sh"]
        assert [r.path for r in manifest.agents] == ["agents/coder.md"]
        assert [r.path for r in manifest.assets] == ["assets/diagram.png"]

    async def test_flat_skill_extension_fallback(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        # Mirrors the agentic-community/mcp-gateway-registry usage-report layout.
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": ".claude/skills/usage-report/SKILL.md", "size": 1000},
                {"type": "blob", "path": ".claude/skills/usage-report/analyze.py", "size": 200},
                {"type": "blob", "path": ".claude/skills/usage-report/install.sh", "size": 50},
                {"type": "blob", "path": ".claude/skills/usage-report/notes.md", "size": 80},
                {"type": "blob", "path": ".claude/skills/usage-report/style.css", "size": 60},
                # Hidden / pycache content should be filtered.
                {
                    "type": "blob",
                    "path": ".claude/skills/usage-report/__pycache__/x.pyc",
                    "size": 90,
                },
                {"type": "blob", "path": ".claude/skills/usage-report/.DS_Store", "size": 5},
            ]
        )

        manifest = await _discover_skill_resources(
            "https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/"
            "refs/heads/main/.claude/skills/usage-report/SKILL.md",
        )
        assert isinstance(manifest, SkillResourceManifest)
        assert {r.path for r in manifest.scripts} == {"analyze.py", "install.sh"}
        assert {r.path for r in manifest.references} == {"notes.md"}
        assert {r.path for r in manifest.assets} == {"style.css"}
        assert manifest.agents == []

    async def test_excludes_skill_md_readme_license(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": "x/SKILL.md", "size": 100},
                {"type": "blob", "path": "x/README.md", "size": 100},
                {"type": "blob", "path": "x/LICENSE", "size": 100},
                {"type": "blob", "path": "x/run.py", "size": 100},
            ]
        )
        manifest = await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
        )
        assert manifest is not None
        assert {r.path for r in manifest.scripts} == {"run.py"}
        assert manifest.references == []

    async def test_returns_none_when_no_classified_resources(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": "x/SKILL.md", "size": 100},
                {"type": "blob", "path": "x/Makefile", "size": 100},  # no extension
            ]
        )
        result = await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
        )
        assert result is None

    async def test_truncated_response_logs_warning(
        self,
        mock_github_auth,
        mock_async_client,
        caplog,
    ):
        mock_async_client.json.return_value = _trees_payload(
            [{"type": "blob", "path": "x/r.py", "size": 1}],
            truncated=True,
        )
        with caplog.at_level(logging.WARNING):
            manifest = await _discover_skill_resources(
                "https://github.com/foo/bar/blob/main/x/SKILL.md",
            )
        assert manifest is not None
        assert any("truncated" in m for m in caplog.messages)

    async def test_http_error_returns_none(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        mock_async_client.status_code = 502
        result = await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
        )
        assert result is None

    async def test_top_level_array_response_still_works(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        # Forward-compat: a hosting platform that returns a bare array
        # rather than the GitHub envelope shape should still work.
        mock_async_client.json.return_value = [
            {"type": "blob", "path": "x/SKILL.md", "size": 100},
            {"type": "blob", "path": "x/r.py", "size": 100},
        ]
        manifest = await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
        )
        assert manifest is not None
        assert {r.path for r in manifest.scripts} == {"r.py"}

    async def test_gitlab_pagination_is_bounded_by_page_cap(
        self,
        mock_github_auth,
    ):
        # A GitLab server that always echoes x-next-page must not loop forever:
        # discovery stops after MAX_GITLAB_TREE_PAGES fetches.
        from registry.services.skill_service import MAX_GITLAB_TREE_PAGES

        response = MagicMock()
        response.status_code = 200
        # Fresh list per call: real httpx parses a new list each response, so
        # the loop's payload.extend(page_payload) appends distinct objects. A
        # shared return_value list would make payload.extend(payload) double the
        # list every iteration (exponential blow-up) and OOM before the cap.
        response.json = MagicMock(
            side_effect=lambda: [{"type": "blob", "path": "x/r.py", "size": 100}]
        )
        response.headers = {"x-next-page": "9"}  # never empty -> would loop forever without the cap

        client_instance = MagicMock()
        client_instance.get = AsyncMock(return_value=response)
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "registry.services.skill_service.guarded_async_client",
            return_value=client_instance,
        ):
            await _discover_skill_resources(
                "https://gitlab.example.com/g/r/-/raw/main/skills/demo/SKILL.md",
            )

        # 1 initial fetch + MAX_GITLAB_TREE_PAGES paginated fetches, no more.
        assert client_instance.get.await_count == MAX_GITLAB_TREE_PAGES + 1

    async def test_github_auth_headers_merged_into_request(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        mock_github_auth.get_auth_headers.return_value = {"Authorization": "Bearer ghs_xxx"}
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": "x/SKILL.md", "size": 100},
                {"type": "blob", "path": "x/r.py", "size": 100},
            ]
        )

        await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
        )

        # Confirm get_auth_headers was awaited against the exact Trees API URL
        # (full-equality assertion -- avoids CodeQL's
        # py/incomplete-url-substring-sanitization rule, and is also stricter
        # because it would catch a regression that called the wrong endpoint).
        # The shared credentials are only offered when the caller opted into the
        # (admin-gated) global_credentials scheme; the default "none" scheme must
        # NOT opt in, so allow_global_credentials is False here.
        mock_github_auth.get_auth_headers.assert_awaited_once_with(
            "https://api.github.com/repos/foo/bar/git/trees/main?recursive=1",
            allow_global_credentials=False,
        )

    async def test_global_credentials_scheme_opts_into_shared_token(
        self,
        mock_github_auth,
        mock_async_client,
    ):
        """auth_scheme=global_credentials is the only scheme that opts in."""
        mock_github_auth.get_auth_headers.return_value = {"Authorization": "Bearer ghs_xxx"}
        mock_async_client.json.return_value = _trees_payload(
            [
                {"type": "blob", "path": "x/SKILL.md", "size": 100},
                {"type": "blob", "path": "x/r.py", "size": 100},
            ]
        )

        await _discover_skill_resources(
            "https://github.com/foo/bar/blob/main/x/SKILL.md",
            auth_scheme="global_credentials",
        )

        mock_github_auth.get_auth_headers.assert_awaited_once_with(
            "https://api.github.com/repos/foo/bar/git/trees/main?recursive=1",
            allow_global_credentials=True,
        )
