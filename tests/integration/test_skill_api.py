"""Integration tests for skill API endpoints."""
import pytest
from unittest.mock import AsyncMock, patch


# Sample skill data for testing
SAMPLE_SKILL_DATA = {
    "name": "test-skill",
    "description": "A test skill for integration testing",
    "skill_md_url": "https://raw.githubusercontent.com/test/skills/main/SKILL.md",
    "tags": ["test", "integration"],
    "visibility": "public",
}


@pytest.fixture
def skill_data():
    """Sample skill data for testing."""
    return SAMPLE_SKILL_DATA.copy()


@pytest.fixture
def mock_url_validation():
    """Mock SKILL.md URL validation to avoid network requests."""
    with patch(
        "registry.services.skill_service._validate_skill_md_url",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = {
            "valid": True,
            "content_version": "abc123def456",
            "content_updated_at": None
        }
        yield mock


@pytest.fixture
def mock_auth_admin():
    """Mock authentication returning admin user context."""
    admin_context = {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"],
        "is_admin": True,
        "can_modify_servers": True,
    }
    with patch(
        "registry.api.skill_routes.nginx_proxied_auth",
        return_value=admin_context,
    ):
        yield admin_context


@pytest.fixture
def mock_auth_user():
    """Mock authentication returning regular user context."""
    user_context = {
        "username": "testuser",
        "groups": ["mcp-registry-user"],
        "scopes": ["mcp-servers-unrestricted/read"],
        "is_admin": False,
        "can_modify_servers": False,
    }
    with patch(
        "registry.api.skill_routes.nginx_proxied_auth",
        return_value=user_context,
    ):
        yield user_context


@pytest.fixture
def mock_skill_repository():
    """Mock skill repository."""
    mock_repo = AsyncMock()
    mock_repo.ensure_indexes = AsyncMock()
    mock_repo.create = AsyncMock()
    mock_repo.get = AsyncMock(return_value=None)
    mock_repo.list_all = AsyncMock(return_value=[])
    mock_repo.list_filtered = AsyncMock(return_value=[])
    mock_repo.update = AsyncMock()
    mock_repo.delete = AsyncMock(return_value=True)
    mock_repo.set_state = AsyncMock(return_value=True)
    mock_repo.get_state = AsyncMock(return_value=True)
    return mock_repo


@pytest.fixture
def mock_search_repository():
    """Mock search repository."""
    mock_repo = AsyncMock()
    mock_repo.index_skill = AsyncMock()
    mock_repo.remove_entity = AsyncMock()
    return mock_repo


class TestSkillModels:
    """Test skill data model validation."""

    def test_skill_card_creation(self):
        """Test SkillCard model creation."""
        from registry.schemas.skill_models import SkillCard

        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="A test skill",
            skill_md_url="https://test.com/SKILL.md"
        )

        assert skill.path == "/skills/test-skill"
        assert skill.name == "test-skill"
        assert skill.is_enabled is True

    def test_skill_registration_request_validation(self, skill_data):
        """Test SkillRegistrationRequest validation."""
        from registry.schemas.skill_models import SkillRegistrationRequest

        request = SkillRegistrationRequest(**skill_data)
        assert request.name == "test-skill"
        assert "test" in request.tags

    def test_skill_info_from_card(self):
        """Test creating SkillInfo from SkillCard."""
        from registry.schemas.skill_models import SkillCard, SkillInfo

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test skill",
            skill_md_url="https://test.com/SKILL.md",
            tags=["tag1", "tag2"]
        )

        info = SkillInfo(
            path=skill.path,
            name=skill.name,
            description=skill.description,
            skill_md_url=str(skill.skill_md_url),
            tags=skill.tags,
            is_enabled=skill.is_enabled,
            visibility=skill.visibility,
        )

        assert info.path == "/skills/test"
        assert len(info.tags) == 2


class TestSkillService:
    """Test skill service functionality."""

    @pytest.mark.asyncio
    async def test_register_skill(
        self,
        skill_data,
        mock_url_validation,
        mock_skill_repository,
        mock_search_repository,
    ):
        """Test skill registration."""
        from registry.schemas.skill_models import SkillCard, SkillRegistrationRequest
        from registry.services.skill_service import SkillService

        # Setup mock
        created_skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="A test skill for integration testing",
            skill_md_url="https://raw.githubusercontent.com/test/skills/main/SKILL.md",
            tags=["test", "integration"],
        )
        mock_skill_repository.create.return_value = created_skill

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        request = SkillRegistrationRequest(**skill_data)
        result = await service.register_skill(request, owner="testuser")

        assert result.name == "test-skill"
        assert result.path == "/skills/test-skill"
        mock_skill_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_skill(self, mock_skill_repository, mock_search_repository):
        """Test getting a skill by path."""
        from registry.schemas.skill_models import SkillCard
        from registry.services.skill_service import SkillService

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md"
        )
        mock_skill_repository.get.return_value = skill

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        result = await service.get_skill("/skills/test")
        assert result is not None
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_list_skills(self, mock_skill_repository, mock_search_repository):
        """Test listing skills."""
        from registry.schemas.skill_models import SkillCard
        from registry.services.skill_service import SkillService

        skills = [
            SkillCard(
                path="/skills/skill1",
                name="skill1",
                description="Skill 1",
                skill_md_url="https://test.com/SKILL.md"
            ),
            SkillCard(
                path="/skills/skill2",
                name="skill2",
                description="Skill 2",
                skill_md_url="https://test.com/SKILL.md"
            ),
        ]
        mock_skill_repository.list_filtered.return_value = skills

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        result = await service.list_skills()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_toggle_skill(self, mock_skill_repository, mock_search_repository):
        """Test toggling skill enabled state."""
        from registry.schemas.skill_models import SkillCard
        from registry.services.skill_service import SkillService

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            is_enabled=True
        )
        mock_skill_repository.set_state.return_value = True
        mock_skill_repository.get.return_value = skill

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        result = await service.toggle_skill("/skills/test", False)
        assert result is True
        mock_skill_repository.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_skill(self, mock_skill_repository, mock_search_repository):
        """Test skill deletion."""
        from registry.services.skill_service import SkillService

        mock_skill_repository.delete.return_value = True

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        result = await service.delete_skill("/skills/test")
        assert result is True
        mock_skill_repository.delete.assert_called_once()


class TestSkillVisibility:
    """Test skill visibility filtering."""

    @pytest.mark.asyncio
    async def test_public_skill_visible_to_anonymous(
        self,
        mock_skill_repository,
        mock_search_repository,
    ):
        """Test that public skills are visible to anonymous users."""
        from registry.schemas.skill_models import SkillCard, VisibilityEnum
        from registry.services.skill_service import SkillService

        public_skill = SkillCard(
            path="/skills/public",
            name="public",
            description="Public skill",
            skill_md_url="https://test.com/SKILL.md",
            visibility=VisibilityEnum.PUBLIC,
        )
        mock_skill_repository.list_filtered.return_value = [public_skill]

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        result = await service.list_skills_for_user(user_context=None)
        assert len(result) == 1
        assert result[0].name == "public"

    @pytest.mark.asyncio
    async def test_private_skill_hidden_from_others(
        self,
        mock_skill_repository,
        mock_search_repository,
    ):
        """Test that private skills are hidden from non-owners."""
        from registry.schemas.skill_models import SkillCard, VisibilityEnum
        from registry.services.skill_service import SkillService

        private_skill = SkillCard(
            path="/skills/private",
            name="private",
            description="Private skill",
            skill_md_url="https://test.com/SKILL.md",
            visibility=VisibilityEnum.PRIVATE,
            owner="other_user",
        )
        mock_skill_repository.list_filtered.return_value = [private_skill]
        mock_skill_repository.get.return_value = private_skill

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        user_context = {"username": "testuser", "groups": [], "is_admin": False}
        result = await service.list_skills_for_user(user_context=user_context)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_admin_sees_all_skills(
        self,
        mock_skill_repository,
        mock_search_repository,
    ):
        """Test that admin users see all skills."""
        from registry.schemas.skill_models import SkillCard, VisibilityEnum
        from registry.services.skill_service import SkillService

        skills = [
            SkillCard(
                path="/skills/public",
                name="public",
                description="Public skill",
                skill_md_url="https://test.com/SKILL.md",
                visibility=VisibilityEnum.PUBLIC,
            ),
            SkillCard(
                path="/skills/private",
                name="private",
                description="Private skill",
                skill_md_url="https://test.com/SKILL.md",
                visibility=VisibilityEnum.PRIVATE,
                owner="other_user",
            ),
        ]
        mock_skill_repository.list_filtered.return_value = skills

        service = SkillService()
        service._repo = mock_skill_repository
        service._search_repo = mock_search_repository

        admin_context = {"username": "admin", "groups": [], "is_admin": True}
        result = await service.list_skills_for_user(user_context=admin_context)
        assert len(result) == 2


class TestToolValidation:
    """Test tool validation service."""

    @pytest.mark.asyncio
    async def test_validate_tools_all_available(self):
        """Test validation when all tools are available."""
        from registry.schemas.skill_models import SkillCard, ToolReference
        from registry.services.tool_validation_service import ToolValidationService

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            allowed_tools=[
                ToolReference(tool_name="Read"),
                ToolReference(tool_name="Write"),
            ]
        )

        # Mock server repository
        mock_server_repo = AsyncMock()
        mock_server_repo.list_all.return_value = {
            "/filesystem": {
                "server_name": "filesystem",
                "tool_list": [
                    {"name": "Read"},
                    {"name": "Write"},
                ]
            }
        }
        mock_server_repo.get_state.return_value = True

        service = ToolValidationService()
        service._server_repo = mock_server_repo

        result = await service.validate_tools_available(skill)
        assert result.all_available is True
        assert len(result.missing_tools) == 0
        assert len(result.available_tools) == 2

    @pytest.mark.asyncio
    async def test_validate_tools_some_missing(self):
        """Test validation when some tools are missing."""
        from registry.schemas.skill_models import SkillCard, ToolReference
        from registry.services.tool_validation_service import ToolValidationService

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            allowed_tools=[
                ToolReference(tool_name="Read"),
                ToolReference(tool_name="NonExistent"),
            ]
        )

        # Mock server repository
        mock_server_repo = AsyncMock()
        mock_server_repo.list_all.return_value = {
            "/filesystem": {
                "server_name": "filesystem",
                "tool_list": [
                    {"name": "Read"},
                ]
            }
        }
        mock_server_repo.get_state.return_value = True

        service = ToolValidationService()
        service._server_repo = mock_server_repo

        result = await service.validate_tools_available(skill)
        assert result.all_available is False
        assert "NonExistent" in result.missing_tools
        assert "Read" in result.available_tools

    @pytest.mark.asyncio
    async def test_validate_no_tools_required(self):
        """Test validation when skill has no required tools."""
        from registry.schemas.skill_models import SkillCard
        from registry.services.tool_validation_service import ToolValidationService

        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            allowed_tools=[]
        )

        service = ToolValidationService()

        result = await service.validate_tools_available(skill)
        assert result.all_available is True
        assert len(result.missing_tools) == 0
        assert len(result.available_tools) == 0


class TestPathUtils:
    """Test path utility functions."""

    def test_normalize_skill_path_basic(self):
        """Test basic path normalization."""
        from registry.utils.path_utils import normalize_skill_path

        assert normalize_skill_path("test") == "/skills/test"
        assert normalize_skill_path("/test") == "/skills/test"
        assert normalize_skill_path("/skills/test") == "/skills/test"

    def test_normalize_skill_path_duplicate_slashes(self):
        """Test path normalization removes duplicate slashes."""
        from registry.utils.path_utils import normalize_skill_path

        assert normalize_skill_path("//test") == "/skills/test"
        assert normalize_skill_path("/skills//test") == "/skills/test"

    def test_extract_skill_name(self):
        """Test extracting skill name from path."""
        from registry.utils.path_utils import extract_skill_name

        assert extract_skill_name("/skills/test") == "test"
        assert extract_skill_name("test") == "test"

    def test_validate_skill_name(self):
        """Test skill name validation."""
        from registry.utils.path_utils import validate_skill_name

        assert validate_skill_name("test") is True
        assert validate_skill_name("test-skill") is True
        assert validate_skill_name("test-skill-v2") is True
        assert validate_skill_name("a1") is True

        assert validate_skill_name("TEST") is False
        assert validate_skill_name("test--skill") is False
        assert validate_skill_name("-test") is False
        assert validate_skill_name("test-") is False
        assert validate_skill_name("test_skill") is False
