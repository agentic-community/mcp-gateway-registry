"""Tests that GitHub auth headers are injected into skill service httpx calls."""

from unittest.mock import AsyncMock, MagicMock, patch


class TestValidateSkillMdUrlAuth:
    """Tests for auth header injection in _validate_skill_md_url."""

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    async def test_auth_headers_passed_to_get(self, mock_safe_url, mock_auth):
        """Auth headers from GitHubAuthProvider are passed to httpx.get."""
        mock_auth.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer ghp_test"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"# Test Skill"
        mock_response.url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"

        with patch("registry.services.skill_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.services.skill_service import _validate_skill_md_url

            result = await _validate_skill_md_url(
                "https://raw.githubusercontent.com/o/r/main/SKILL.md"
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer ghp_test"}
            assert result["valid"] is True

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    async def test_empty_headers_when_no_credentials(self, mock_safe_url, mock_auth):
        """Empty headers passed when no credentials configured."""
        mock_auth.get_auth_headers = AsyncMock(return_value={})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"# Test Skill"
        mock_response.url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"

        with patch("registry.services.skill_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.services.skill_service import _validate_skill_md_url

            await _validate_skill_md_url("https://raw.githubusercontent.com/o/r/main/SKILL.md")

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {}


class TestParseSkillMdContentAuth:
    """Tests for auth header injection in _parse_skill_md_content."""

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    @patch("registry.services.skill_service.translate_skill_url")
    async def test_auth_headers_passed_to_get(self, mock_translate, mock_safe_url, mock_auth):
        """global_credentials sends GitHub auth headers when parsing SKILL.md."""
        mock_auth.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer ghp_test"})
        mock_translate.return_value = (
            "https://github.com/o/r/blob/main/SKILL.md",
            "https://raw.githubusercontent.com/o/r/refs/heads/main/SKILL.md",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"---\nname: test\n---\n# Test Skill"
        mock_response.text = "---\nname: test\n---\n# Test Skill"
        mock_response.url = "https://raw.githubusercontent.com/o/r/refs/heads/main/SKILL.md"

        with patch("registry.services.skill_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.services.skill_service import _parse_skill_md_content

            result = await _parse_skill_md_content(
                "https://github.com/o/r/blob/main/SKILL.md",
                auth_scheme="global_credentials",
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer ghp_test"}
            assert result["name"] == "test"

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    @patch("registry.services.skill_service.translate_skill_url")
    async def test_none_scheme_sends_no_headers(self, mock_translate, mock_safe_url, mock_auth):
        """auth_scheme=none sends no auth headers when parsing SKILL.md."""
        mock_auth.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer ghp_should_not_appear"})
        mock_translate.return_value = (
            "https://github.com/o/r/blob/main/SKILL.md",
            "https://raw.githubusercontent.com/o/r/refs/heads/main/SKILL.md",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"---\nname: test\n---\n# Test Skill"
        mock_response.text = "---\nname: test\n---\n# Test Skill"
        mock_response.url = "https://raw.githubusercontent.com/o/r/refs/heads/main/SKILL.md"

        with patch("registry.services.skill_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.services.skill_service import _parse_skill_md_content

            result = await _parse_skill_md_content(
                "https://github.com/o/r/blob/main/SKILL.md",
                auth_scheme="none",
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {}
            assert result["name"] == "test"


class TestCheckSkillHealthAuth:
    """Tests for auth header injection in _check_skill_health."""

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    async def test_auth_headers_passed_to_head(self, mock_safe_url, mock_auth):
        """Auth headers are passed to httpx.head in health check."""
        mock_auth.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer ghp_test"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"

        with patch("registry.services.skill_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.services.skill_service import _check_skill_health

            result = await _check_skill_health(
                "https://raw.githubusercontent.com/o/r/main/SKILL.md"
            )

            call_kwargs = mock_client.head.call_args
            assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer ghp_test"}
            assert result["healthy"] is True
