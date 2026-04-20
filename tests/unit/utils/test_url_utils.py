"""
Unit tests for registry.utils.url_utils.extract_repository_url.

Validates extraction of GitHub repository URLs from SKILL.md URLs,
including public GitHub, raw.githubusercontent.com, and enterprise instances.
"""

from registry.utils.url_utils import extract_repository_url


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
