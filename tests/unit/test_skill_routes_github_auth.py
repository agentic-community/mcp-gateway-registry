"""Tests that GitHub auth headers are injected into skill routes httpx calls."""

from unittest.mock import AsyncMock, MagicMock, patch


class TestGetSkillContentAuth:
    """Tests for auth header injection in get_skill_content."""

    @patch("registry.api.skill_routes._github_auth")
    @patch("registry.api.skill_routes._is_safe_url", return_value=True)
    @patch("registry.api.skill_routes._user_can_access_skill", return_value=True)
    @patch("registry.api.skill_routes.get_skill_service")
    async def test_auth_headers_passed_to_get(
        self, mock_get_service, mock_access, mock_safe_url, mock_auth
    ):
        """Auth headers from GitHubAuthProvider are passed to httpx.get."""
        mock_auth.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer ghp_test"})

        # Mock skill service to return a skill with raw URL
        mock_skill = MagicMock()
        mock_skill.skill_md_raw_url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        mock_skill.skill_md_url = "https://github.com/o/r/blob/main/SKILL.md"
        mock_skill.skill_md_content = None  # Force httpx path, not inline content
        # Drift-detection guard in get_skill_content() reads .content_integrity;
        # MagicMock's default truthy attribute would otherwise trigger 409.
        mock_skill.content_integrity = None
        mock_skill.resource_manifest = None
        mock_skill.tags = []
        mock_skill.auth_scheme = "none"
        mock_skill.auth_credential_encrypted = None
        mock_skill.auth_header_name = None

        mock_service = AsyncMock()
        mock_service.get_skill.return_value = mock_skill
        mock_get_service.return_value = mock_service

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# My Skill"
        mock_response.url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from registry.api.skill_routes import get_skill_content

            result = await get_skill_content(
                user_context={"sub": "test-user"},
                skill_path="test/skill",
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer ghp_test"}
            assert result["content"] == "# My Skill"
