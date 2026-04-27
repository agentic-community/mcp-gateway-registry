"""Tests that GitHub auth headers are injected into skill routes httpx calls."""

from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_skill(
    auth_scheme: str = "none",
) -> MagicMock:
    """Create a mock SkillCard with sensible defaults."""
    mock_skill = MagicMock()
    mock_skill.skill_md_raw_url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"
    mock_skill.skill_md_url = "https://github.com/o/r/blob/main/SKILL.md"
    mock_skill.skill_md_content = None
    mock_skill.content_integrity = None
    mock_skill.resource_manifest = None
    mock_skill.tags = []
    mock_skill.auth_scheme = auth_scheme
    mock_skill.auth_credential_encrypted = None
    mock_skill.auth_header_name = None
    return mock_skill


class TestGetSkillContentAuth:
    """Tests for auth header injection in get_skill_content."""

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    @patch("registry.api.skill_routes._user_can_access_skill", return_value=True)
    @patch("registry.api.skill_routes.get_skill_service")
    async def test_global_credentials_sends_github_headers(
        self, mock_get_service, mock_access, mock_safe_url, mock_auth
    ):
        """auth_scheme=global_credentials sends global GitHub auth headers."""
        mock_auth.get_auth_headers = AsyncMock(
            return_value={"Authorization": "Bearer ghp_test"},
        )

        mock_skill = _make_mock_skill(auth_scheme="global_credentials")
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
                resource=None,
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer ghp_test"}
            assert result["content"] == "# My Skill"

    @patch("registry.services.skill_service._github_auth")
    @patch("registry.services.skill_service._is_safe_url", return_value=True)
    @patch("registry.api.skill_routes._user_can_access_skill", return_value=True)
    @patch("registry.api.skill_routes.get_skill_service")
    async def test_none_scheme_sends_no_auth_headers(
        self, mock_get_service, mock_access, mock_safe_url, mock_auth
    ):
        """auth_scheme=none sends no auth headers at all."""
        mock_auth.get_auth_headers = AsyncMock(
            return_value={"Authorization": "Bearer ghp_should_not_appear"},
        )

        mock_skill = _make_mock_skill(auth_scheme="none")
        mock_service = AsyncMock()
        mock_service.get_skill.return_value = mock_skill
        mock_get_service.return_value = mock_service

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Public Skill"
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
                resource=None,
            )

            call_kwargs = mock_client.get.call_args
            assert call_kwargs.kwargs.get("headers") == {}
            assert result["content"] == "# Public Skill"
