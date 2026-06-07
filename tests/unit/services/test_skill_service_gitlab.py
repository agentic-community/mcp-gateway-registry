"""
Unit tests for the GitLab integration hooks in registry.services.skill_service.

These cover the call-site branches added alongside the optional
registry.utils.gitlab_url_utils helper:

  * _build_fetch_headers — translates a /-/raw/ URL to API v4 only when
    auth headers are populated and the URL is GitLab-shaped.
  * _resolve_tree_api    — tries GitLab translation first, falls through
    to GitHub / GHES handling when the URL is not GitLab.

Both hooks wrap their import in try/except ImportError so the upstream
GitHub paths keep working when gitlab_url_utils is removed.  Those
fall-through paths are exercised by patching sys.modules.
"""

import sys
from unittest.mock import patch

import pytest

from registry.services.skill_service import (
    _build_fetch_headers,
    _resolve_tree_api,
)


# =============================================================================
# _build_fetch_headers — GitLab URL translation
# =============================================================================


class TestBuildFetchHeadersGitlab:
    """Translation of GitLab /-/raw/ URLs to API v4 raw-file endpoints."""

    def test_gitlab_raw_url_with_api_key_is_translated(self):
        url = "https://gitlab.example.com/g/r/-/raw/main/skills/demo/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="api_key", auth_credential="glpat-xxx"
        )

        assert headers == {"PRIVATE-TOKEN": "glpat-xxx"}
        assert fetch_url == (
            "https://gitlab.example.com/api/v4/projects/g%2Fr"
            "/repository/files/skills%2Fdemo%2FSKILL.md/raw?ref=main"
        )

    def test_gitlab_url_with_bearer_auth_also_translates(self):
        url = "https://gitlab.example.com/g/r/-/raw/main/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="bearer", auth_credential="tok"
        )

        assert headers == {"Authorization": "Bearer tok"}
        assert fetch_url == (
            "https://gitlab.example.com/api/v4/projects/g%2Fr"
            "/repository/files/SKILL.md/raw?ref=main"
        )

    def test_github_url_is_not_translated(self):
        url = "https://github.com/owner/repo/blob/main/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="api_key", auth_credential="ghp-xxx"
        )

        assert fetch_url == url
        assert headers == {"PRIVATE-TOKEN": "ghp-xxx"}

    def test_gitlab_url_without_credentials_skips_translation(self):
        url = "https://gitlab.example.com/g/r/-/raw/main/SKILL.md"

        fetch_url, headers = _build_fetch_headers(url, auth_scheme="none")

        assert fetch_url == url
        assert headers == {}

    def test_self_hosted_gitlab_without_gitlab_in_hostname(self):
        """Self-hosted GitLab with no 'gitlab' in hostname still translates."""
        url = "https://code.internal.corp/team/project/-/raw/main/skills/demo/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="api_key", auth_credential="glpat-xxx"
        )

        assert headers == {"PRIVATE-TOKEN": "glpat-xxx"}
        assert fetch_url == (
            "https://code.internal.corp/api/v4/projects/team%2Fproject"
            "/repository/files/skills%2Fdemo%2FSKILL.md/raw?ref=main"
        )

    def test_self_hosted_gitlab_bearer_without_gitlab_in_hostname(self):
        """Self-hosted GitLab with bearer auth and no 'gitlab' in hostname."""
        url = "https://scm.mycompany.io/ops/infra/-/raw/develop/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="bearer", auth_credential="token123"
        )

        assert headers == {"Authorization": "Bearer token123"}
        assert fetch_url == (
            "https://scm.mycompany.io/api/v4/projects/ops%2Finfra"
            "/repository/files/SKILL.md/raw?ref=develop"
        )

    def test_gitlab_url_with_unrecognised_shape_falls_through(self):
        # Contains "gitlab" but is not a /-/raw/ or API v4 file URL,
        # so translate_gitlab_to_api_url returns None and fetch_url
        # is left unchanged.
        url = "https://gitlab.example.com/g/r/blob/main/SKILL.md"

        fetch_url, headers = _build_fetch_headers(
            url, auth_scheme="api_key", auth_credential="glpat-xxx"
        )

        assert fetch_url == url
        assert headers == {"PRIVATE-TOKEN": "glpat-xxx"}

    def test_importerror_is_swallowed_and_url_unchanged(self):
        # Simulate gitlab_url_utils being absent (upstream-only deployment).
        url = "https://gitlab.example.com/g/r/-/raw/main/SKILL.md"

        with patch.dict(
            sys.modules, {"registry.utils.gitlab_url_utils": None}
        ):
            fetch_url, headers = _build_fetch_headers(
                url, auth_scheme="api_key", auth_credential="glpat-xxx"
            )

        assert fetch_url == url
        assert headers == {"PRIVATE-TOKEN": "glpat-xxx"}


# =============================================================================
# _resolve_tree_api — GitLab branch + ImportError fall-through
# =============================================================================


class TestResolveTreeApiGitlab:
    """GitLab path through the tree-API resolver."""

    def test_gitlab_url_returns_api_v4_tree_url(self):
        skill_md_url = (
            "https://gitlab.example.com/g/r/-/raw/main/skills/demo/SKILL.md"
        )

        result = _resolve_tree_api(skill_md_url)

        assert result is not None
        tree_url, project_encoded, ref, skill_dir = result
        assert project_encoded == "g%2Fr"
        assert ref == "main"
        assert skill_dir == "skills/demo"
        assert tree_url == (
            "https://gitlab.example.com/api/v4/projects/g%2Fr/repository/tree"
            "?path=skills%2Fdemo&ref=main&recursive=true&per_page=100"
        )

    def test_self_hosted_gitlab_without_gitlab_in_hostname(self):
        """Self-hosted GitLab without 'gitlab' in hostname resolves tree URL."""
        skill_md_url = (
            "https://code.internal.corp/team/project/-/raw/main/skills/demo/SKILL.md"
        )

        result = _resolve_tree_api(skill_md_url)

        assert result is not None
        tree_url, project_encoded, ref, skill_dir = result
        assert project_encoded == "team%2Fproject"
        assert ref == "main"
        assert skill_dir == "skills/demo"
        assert tree_url == (
            "https://code.internal.corp/api/v4/projects/team%2Fproject/repository/tree"
            "?path=skills%2Fdemo&ref=main&recursive=true&per_page=100"
        )

    def test_gitlab_skill_md_at_repo_root_falls_through_to_github_branch(self):
        # translate_gitlab_tree_api_url returns None for root-level files,
        # so the function continues to the GitHub matcher which then also
        # returns None (no /blob/ or /refs/heads/ in the path).
        skill_md_url = "https://gitlab.example.com/g/r/-/raw/main/SKILL.md"

        assert _resolve_tree_api(skill_md_url) is None

    def test_importerror_falls_through_to_github_path(self):
        # With gitlab_url_utils unavailable, a GitHub URL must still resolve.
        github_url = "https://github.com/anthropics/skills/blob/main/x/SKILL.md"

        with patch.dict(
            sys.modules, {"registry.utils.gitlab_url_utils": None}
        ):
            result = _resolve_tree_api(github_url)

        assert result is not None
        tree_url, project, ref, skill_dir = result
        assert tree_url == (
            "https://api.github.com/repos/anthropics/skills/git/trees/main?recursive=1"
        )
        assert project == "anthropics/skills"
        assert ref == "main"
        assert skill_dir == "x"


# =============================================================================
# Parametrised non-GitLab auth schemes (kept short — for regression only)
# =============================================================================


# =============================================================================
# _append_page_param helper
# =============================================================================


class TestAppendPageParam:
    """Tests for the _append_page_param URL helper."""

    def test_appends_page_to_url_with_existing_params(self):
        from registry.services.skill_service import _append_page_param

        url = "https://host/api/v4/projects/x/repository/tree?path=a&ref=main&recursive=true&per_page=100"

        result = _append_page_param(url, "2")

        assert result == url + "&page=2"

    def test_replaces_existing_page_param(self):
        from registry.services.skill_service import _append_page_param

        url = "https://host/api/tree?path=a&ref=main&page=2&per_page=100"

        result = _append_page_param(url, "3")

        assert "page=3" in result
        assert "page=2" not in result

    def test_appends_page_to_url_without_query_string(self):
        from registry.services.skill_service import _append_page_param

        url = "https://host/api/tree"

        result = _append_page_param(url, "1")

        assert result == "https://host/api/tree?page=1"


# =============================================================================
# Parametrised non-GitLab auth schemes (kept short - for regression only)
# =============================================================================


@pytest.mark.parametrize(
    "scheme,credential,expected_header",
    [
        ("bearer", "abc", {"Authorization": "Bearer abc"}),
        ("api_key", "abc", {"PRIVATE-TOKEN": "abc"}),
        ("none", "abc", {}),
        ("global_credentials", "abc", {}),
    ],
)
def test_build_fetch_headers_non_gitlab_url_leaves_url_intact(
    scheme, credential, expected_header
):
    url = "https://example.com/path/SKILL.md"

    fetch_url, headers = _build_fetch_headers(
        url, auth_scheme=scheme, auth_credential=credential
    )

    assert fetch_url == url
    assert headers == expected_header
