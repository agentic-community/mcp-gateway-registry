"""Unit tests for skill models."""
import pytest

from registry.schemas.skill_models import (
    CompatibilityRequirement,
    SkillCard,
    SkillInfo,
    SkillRegistrationRequest,
    ToolReference,
    VisibilityEnum,
)


class TestSkillCard:
    """Tests for SkillCard model."""

    def test_valid_skill_name(self):
        """Test valid skill names are accepted."""
        valid_names = ["pdf-processing", "code-review", "data-analysis", "a1"]

        for name in valid_names:
            skill = SkillCard(
                path=f"/skills/{name}",
                name=name,
                description="Test description",
                skill_md_url="https://github.com/test/skill/SKILL.md"
            )
            assert skill.name == name

    def test_invalid_skill_name_uppercase(self):
        """Test uppercase names are rejected."""
        with pytest.raises(ValueError, match="lowercase"):
            SkillCard(
                path="/skills/PDF-Processing",
                name="PDF-Processing",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )

    def test_invalid_skill_name_consecutive_hyphens(self):
        """Test consecutive hyphens are rejected."""
        with pytest.raises(ValueError):
            SkillCard(
                path="/skills/pdf--processing",
                name="pdf--processing",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )

    def test_invalid_skill_name_leading_hyphen(self):
        """Test names starting with hyphen are rejected."""
        with pytest.raises(ValueError):
            SkillCard(
                path="/skills/-pdf-processing",
                name="-pdf-processing",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )

    def test_invalid_skill_name_trailing_hyphen(self):
        """Test names ending with hyphen are rejected."""
        with pytest.raises(ValueError):
            SkillCard(
                path="/skills/pdf-processing-",
                name="pdf-processing-",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )

    def test_invalid_path_format(self):
        """Test path must start with /skills/."""
        with pytest.raises(ValueError, match="/skills/"):
            SkillCard(
                path="/agents/test",
                name="test",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )

    def test_visibility_enum_default(self):
        """Test visibility defaults to public."""
        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md"
        )
        assert skill.visibility == VisibilityEnum.PUBLIC

    def test_visibility_enum_private(self):
        """Test visibility can be set to private."""
        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            visibility=VisibilityEnum.PRIVATE
        )
        assert skill.visibility == VisibilityEnum.PRIVATE

    def test_visibility_enum_group(self):
        """Test visibility can be set to group."""
        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md",
            visibility=VisibilityEnum.GROUP,
            allowed_groups=["developers"]
        )
        assert skill.visibility == VisibilityEnum.GROUP
        assert "developers" in skill.allowed_groups

    def test_default_values(self):
        """Test default values are set correctly."""
        skill = SkillCard(
            path="/skills/test",
            name="test",
            description="Test",
            skill_md_url="https://test.com/SKILL.md"
        )
        assert skill.is_enabled is True
        assert skill.registry_name == "local"
        assert skill.tags == []
        assert skill.allowed_tools == []
        assert skill.requirements == []
        assert skill.target_agents == []


class TestToolReference:
    """Tests for ToolReference model."""

    def test_tool_reference_minimal(self):
        """Test minimal ToolReference."""
        tool = ToolReference(tool_name="Bash")
        assert tool.tool_name == "Bash"
        assert tool.capabilities == []
        assert tool.server_path is None

    def test_tool_reference_with_capabilities(self):
        """Test ToolReference with capabilities."""
        tool = ToolReference(
            tool_name="Bash",
            capabilities=["git:*", "docker:*"]
        )
        assert tool.tool_name == "Bash"
        assert len(tool.capabilities) == 2
        assert "git:*" in tool.capabilities

    def test_tool_reference_with_server_path(self):
        """Test ToolReference with server path."""
        tool = ToolReference(
            tool_name="Read",
            server_path="/servers/filesystem"
        )
        assert tool.server_path == "/servers/filesystem"


class TestCompatibilityRequirement:
    """Tests for CompatibilityRequirement model."""

    def test_compatibility_requirement_product(self):
        """Test product type requirement."""
        req = CompatibilityRequirement(
            type="product",
            target="claude-code",
            min_version="1.0.0"
        )
        assert req.type == "product"
        assert req.target == "claude-code"
        assert req.required is True

    def test_compatibility_requirement_tool(self):
        """Test tool type requirement."""
        req = CompatibilityRequirement(
            type="tool",
            target="python>=3.10",
            required=True
        )
        assert req.type == "tool"
        assert req.required is True

    def test_compatibility_requirement_optional(self):
        """Test optional requirement."""
        req = CompatibilityRequirement(
            type="api",
            target="openai-api",
            required=False
        )
        assert req.required is False


class TestSkillRegistrationRequest:
    """Tests for SkillRegistrationRequest model."""

    def test_valid_request(self):
        """Test valid registration request."""
        request = SkillRegistrationRequest(
            name="pdf-processing",
            description="Extract text from PDFs",
            skill_md_url="https://github.com/org/skills/SKILL.md",
            tags=["pdf", "extraction"],
            visibility=VisibilityEnum.PUBLIC
        )
        assert request.name == "pdf-processing"
        assert len(request.tags) == 2

    def test_request_with_tools(self):
        """Test request with allowed tools."""
        request = SkillRegistrationRequest(
            name="git-workflow",
            description="Git workflow automation",
            skill_md_url="https://github.com/org/skills/SKILL.md",
            allowed_tools=[
                ToolReference(tool_name="Bash", capabilities=["git:*"]),
                ToolReference(tool_name="Read")
            ]
        )
        assert len(request.allowed_tools) == 2

    def test_url_validation_valid(self):
        """Test valid URL is accepted."""
        request = SkillRegistrationRequest(
            name="test",
            description="Test",
            skill_md_url="https://raw.githubusercontent.com/org/repo/main/SKILL.md"
        )
        assert str(request.skill_md_url).startswith("https://")

    def test_url_validation_invalid(self):
        """Test invalid URL is rejected."""
        with pytest.raises(ValueError):
            SkillRegistrationRequest(
                name="test",
                description="Test",
                skill_md_url="not-a-url"
            )

    def test_name_validation(self):
        """Test name validation in request."""
        with pytest.raises(ValueError, match="lowercase"):
            SkillRegistrationRequest(
                name="INVALID",
                description="Test",
                skill_md_url="https://test.com/SKILL.md"
            )


class TestSkillInfo:
    """Tests for SkillInfo model."""

    def test_skill_info_minimal(self):
        """Test minimal SkillInfo."""
        info = SkillInfo(
            path="/skills/test",
            name="test",
            description="Test skill",
            skill_md_url="https://test.com/SKILL.md"
        )
        assert info.path == "/skills/test"
        assert info.is_enabled is True
        assert info.visibility == VisibilityEnum.PUBLIC

    def test_skill_info_with_author(self):
        """Test SkillInfo with author."""
        info = SkillInfo(
            path="/skills/test",
            name="test",
            description="Test skill",
            skill_md_url="https://test.com/SKILL.md",
            author="John Doe",
            version="1.0.0"
        )
        assert info.author == "John Doe"
        assert info.version == "1.0.0"


class TestVisibilityEnum:
    """Tests for VisibilityEnum."""

    def test_enum_values(self):
        """Test enum values."""
        assert VisibilityEnum.PUBLIC.value == "public"
        assert VisibilityEnum.PRIVATE.value == "private"
        assert VisibilityEnum.GROUP.value == "group"

    def test_enum_string_comparison(self):
        """Test enum string comparison."""
        assert VisibilityEnum.PUBLIC == "public"
        assert VisibilityEnum.PRIVATE == "private"
        assert VisibilityEnum.GROUP == "group"
