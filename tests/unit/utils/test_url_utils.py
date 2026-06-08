"""
Unit tests for registry.utils.url_utils.

Covers extract_repository_url (GitHub / GHES extraction) and
derive_resource_url, including the optional GitLab translation hook
and its ImportError fall-through.
"""

import sys
from unittest.mock import patch

from registry.utils.url_utils import derive_resource_url, extract_repository_url


class TestExtractRepositoryUrl:
    """Tests for extract_repository_url utility function."""

    def test_github_blob_url(self):
        """Should extract repo URL from a standard GitHub blob URL."""
        # Arrange
        url = "https://github.com/anthropics/skills/blob/main/skills/art/SKILL.md"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result == "https://github.com/anthropics/skills"

    def test_raw_githubusercontent_url(self):
        """Should extract repo URL from a raw.githubusercontent.com URL."""
        # Arrange
        url = (
            "https://raw.githubusercontent.com/anthropics/skills"
            "/refs/heads/main/skills/art/SKILL.md"
        )

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result == "https://github.com/anthropics/skills"

    def test_enterprise_github_blob_url(self):
        """Should extract repo URL from an enterprise GitHub blob URL."""
        # Arrange
        url = "https://github.mycompany.com/org/repo/blob/main/SKILL.md"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result == "https://github.mycompany.com/org/repo"

    def test_enterprise_raw_url(self):
        """Should extract repo URL from an enterprise raw GitHub URL."""
        # Arrange
        url = "https://raw.github.mycompany.com/org/repo/refs/heads/main/SKILL.md"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result == "https://github.mycompany.com/org/repo"

    def test_non_github_url_returns_none(self):
        """Should return None for non-GitHub URLs."""
        # Arrange
        url = "https://gitlab.com/org/repo/raw/main/SKILL.md"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result is None

    def test_empty_string_returns_none(self):
        """Should return None for an empty string."""
        # Arrange
        url = ""

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result is None

    def test_url_with_no_path_returns_none(self):
        """Should return None when the URL has no path segments."""
        # Arrange
        url = "https://github.com"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result is None

    def test_url_with_only_owner_returns_none(self):
        """Should return None when the URL has only an owner, no repo."""
        # Arrange
        url = "https://github.com/anthropics"

        # Act
        result = extract_repository_url(url)

        # Assert
        assert result is None


class TestDeriveResourceUrl:
    """Tests for derive_resource_url, including the GitLab hook."""

    def test_github_url_uses_skill_md_stripping(self):
        """GitHub URLs should fall through to the SKILL.md path-replace path."""
        skill_md_url = (
            "https://raw.githubusercontent.com/o/r/refs/heads/main/dir/SKILL.md"
        )

        result = derive_resource_url(skill_md_url, "ref/note.md")

        assert result == (
            "https://raw.githubusercontent.com/o/r/refs/heads/main/dir/ref/note.md"
        )

    def test_non_skill_md_url_falls_back_to_basename_replace(self):
        """When the URL doesn't end in /SKILL.md, replace the basename."""
        skill_md_url = "https://example.com/a/b/index.html"

        result = derive_resource_url(skill_md_url, "asset.png")

        assert result == "https://example.com/a/b/asset.png"

    def test_gitlab_url_is_translated_to_api_v4(self):
        """GitLab SKILL.md URLs go through derive_gitlab_resource_url."""
        skill_md_url = "https://gitlab.example.com/g/r/-/raw/main/dir/SKILL.md"

        result = derive_resource_url(skill_md_url, "ref/note.md")

        assert result == (
            "https://gitlab.example.com/api/v4/projects/g%2Fr"
            "/repository/files/dir%2Fref%2Fnote.md/raw?ref=main"
        )

    def test_importerror_falls_back_to_skill_md_stripping(self):
        """With gitlab_url_utils absent, GitHub-style fallback still works."""
        skill_md_url = (
            "https://raw.githubusercontent.com/o/r/refs/heads/main/SKILL.md"
        )

        with patch.dict(
            sys.modules, {"registry.utils.gitlab_url_utils": None}
        ):
            result = derive_resource_url(skill_md_url, "asset.png")

        assert result == (
            "https://raw.githubusercontent.com/o/r/refs/heads/main/asset.png"
        )
