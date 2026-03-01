"""
Factory Boy factories for generating test data.

This module provides factories for creating test instances of domain models
with realistic default data.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import factory
from factory import fuzzy

from registry.schemas import (
    AgentCard,
    AgentInfo,
    Package,
    Repository,
    SecurityScheme,
    ServerDetail,
    Skill,
    StdioTransport,
    StreamableHttpTransport,
)
from registry.schemas.agent_models import AgentProvider
from tests.fixtures.constants import (
    DEFAULT_CAPABILITIES,
    MIME_APPLICATION_JSON,
    MIME_TEXT_PLAIN,
    PROTOCOL_VERSION_1_0,
    TEST_PACKAGE_IDENTIFIER,
    TEST_PACKAGE_REGISTRY_TYPE,
    TEST_PACKAGE_VERSION,
    TEST_REPO_SOURCE,
    TEST_REPO_URL,
    TEST_SKILL_ID_1,
    TEST_SKILL_NAME_1,
    TEST_TAGS_DATA,
    TEST_TOOL_DESCRIPTION_1,
    TEST_TOOL_NAME_1,
    TRUST_UNVERIFIED,
    VISIBILITY_PUBLIC,
)

logger = logging.getLogger(__name__)


class RepositoryFactory(factory.Factory):
    """Factory for creating Repository instances."""

    class Meta:
        model = Repository

    url = TEST_REPO_URL
    source = TEST_REPO_SOURCE
    id = factory.Sequence(lambda n: f"test-repo-{n}")
    subfolder = None


class StdioTransportFactory(factory.Factory):
    """Factory for creating StdioTransport instances."""

    class Meta:
        model = StdioTransport

    type = "stdio"
    command = "uvx"
    args = factory.LazyAttribute(lambda _: ["test-server"])
    env = None


class StreamableHttpTransportFactory(factory.Factory):
    """Factory for creating StreamableHttpTransport instances."""

    class Meta:
        model = StreamableHttpTransport

    type = "streamable-http"
    url = factory.Sequence(lambda n: f"http://localhost:8080/server-{n}")
    headers = None


class PackageFactory(factory.Factory):
    """Factory for creating Package instances."""

    class Meta:
        model = Package

    registryType = TEST_PACKAGE_REGISTRY_TYPE
    identifier = TEST_PACKAGE_IDENTIFIER
    version = TEST_PACKAGE_VERSION
    registryBaseUrl = "https://registry.npmjs.org"
    transport = factory.LazyAttribute(lambda _: StdioTransportFactory().model_dump())
    runtimeHint = "uvx"


class ServerDetailFactory(factory.Factory):
    """Factory for creating ServerDetail instances."""

    class Meta:
        model = ServerDetail

    name = factory.Sequence(lambda n: f"com.example.server-{n}")
    description = factory.Faker("sentence")
    version = fuzzy.FuzzyChoice(["1.0.0", "1.1.0", "2.0.0"])
    title = factory.Faker("word")
    repository = factory.SubFactory(RepositoryFactory)
    websiteUrl = factory.Faker("url")
    packages = factory.LazyAttribute(lambda _: [PackageFactory()])
    meta = None


class SecuritySchemeFactory(factory.Factory):
    """Factory for creating SecurityScheme instances."""

    class Meta:
        model = SecurityScheme

    type = "http"
    scheme = "bearer"
    in_ = None
    name = None
    bearer_format = "JWT"
    flows = None
    openid_connect_url = None


class AgentProviderFactory(factory.Factory):
    """Factory for creating AgentProvider instances."""

    class Meta:
        model = AgentProvider

    organization = factory.Faker("company")
    url = factory.Faker("url")


class SkillFactory(factory.Factory):
    """Factory for creating Skill instances."""

    class Meta:
        model = Skill

    id = factory.Sequence(lambda n: f"skill-{n}")
    name = factory.Faker("word")
    description = factory.Faker("sentence")
    tags = factory.LazyAttribute(lambda _: TEST_TAGS_DATA.copy())
    examples = factory.LazyAttribute(lambda _: ["Example usage of this skill"])
    input_modes = factory.LazyAttribute(lambda _: [MIME_TEXT_PLAIN])
    output_modes = factory.LazyAttribute(lambda _: [MIME_TEXT_PLAIN, MIME_APPLICATION_JSON])
    security = None


class AgentCardFactory(factory.Factory):
    """Factory for creating AgentCard instances."""

    class Meta:
        model = AgentCard

    # Required A2A fields
    protocol_version = PROTOCOL_VERSION_1_0
    name = factory.Sequence(lambda n: f"test-agent-{n}")
    description = factory.Faker("sentence")
    url = factory.Sequence(lambda n: f"http://localhost:9000/agent-{n}")
    version = fuzzy.FuzzyChoice(["1.0", "1.1", "2.0"])
    capabilities = factory.LazyAttribute(lambda _: DEFAULT_CAPABILITIES.copy())
    default_input_modes = factory.LazyAttribute(lambda _: [MIME_TEXT_PLAIN])
    default_output_modes = factory.LazyAttribute(lambda _: [MIME_TEXT_PLAIN])
    skills = factory.LazyAttribute(lambda _: [SkillFactory()])

    # Optional A2A fields
    preferred_transport = "JSONRPC"
    provider = factory.SubFactory(AgentProviderFactory)
    icon_url = factory.Faker("url")
    documentation_url = factory.Faker("url")
    security_schemes = factory.Dict({})
    security = None
    supports_authenticated_extended_card = False
    metadata = factory.Dict({})

    # MCP Gateway Registry extensions
    path = factory.Sequence(lambda n: f"/agents/test-agent-{n}")
    tags = factory.LazyAttribute(lambda _: TEST_TAGS_DATA.copy())
    # Note: AgentCard model does not have a 'streaming' attribute. Streaming capability
    # should be accessed via capabilities.get("streaming", False). See bug documentation:
    # .scratchpad/fixes/registry/fix-agent-streaming-attribute.md
    is_enabled = True
    rating_details = factory.List([])
    license = "MIT"

    # Registry metadata
    registered_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
    registered_by = factory.Faker("user_name")

    # Access control
    visibility = VISIBILITY_PUBLIC
    allowed_groups = factory.List([])

    # Validation and trust
    signature = None
    trust_level = TRUST_UNVERIFIED


class AgentInfoFactory(factory.Factory):
    """Factory for creating AgentInfo instances."""

    class Meta:
        model = AgentInfo

    name = factory.Sequence(lambda n: f"test-agent-{n}")
    description = factory.Faker("sentence")
    path = factory.Sequence(lambda n: f"/agents/test-agent-{n}")
    url = factory.Sequence(lambda n: f"http://localhost:9000/agent-{n}")
    tags = factory.LazyAttribute(lambda _: TEST_TAGS_DATA.copy())
    skills = factory.LazyAttribute(lambda _: [TEST_SKILL_NAME_1])
    num_skills = 1
    is_enabled = True
    provider = factory.Faker("company")
    streaming = False
    trust_level = TRUST_UNVERIFIED


# Helper functions for creating multiple instances


def create_server_with_tools(
    name: str | None = None, num_tools: int = 3, **kwargs: Any
) -> ServerDetail:
    """
    Create a ServerDetail with multiple tools in metadata.

    Args:
        name: Server name (auto-generated if not provided)
        num_tools: Number of tools to create
        **kwargs: Additional ServerDetail attributes

    Returns:
        ServerDetail instance with tools in metadata
    """
    server = ServerDetailFactory(name=name, **kwargs)

    # Add tools to metadata
    tools = []
    for i in range(num_tools):
        tools.append(
            {
                "name": f"{TEST_TOOL_NAME_1}_{i}",
                "description": f"{TEST_TOOL_DESCRIPTION_1} {i}",
                "inputSchema": {"type": "object", "properties": {}},
            }
        )

    server.meta = {"tools": tools, "prompts": [], "resources": []}

    return server


def create_agent_with_skills(
    name: str | None = None, num_skills: int = 3, **kwargs: Any
) -> AgentCard:
    """
    Create an AgentCard with multiple skills.

    Args:
        name: Agent name (auto-generated if not provided)
        num_skills: Number of skills to create
        **kwargs: Additional AgentCard attributes

    Returns:
        AgentCard instance with multiple skills
    """
    skills = [
        SkillFactory(id=f"{TEST_SKILL_ID_1}_{i}", name=f"{TEST_SKILL_NAME_1} {i}")
        for i in range(num_skills)
    ]

    return AgentCardFactory(name=name, skills=skills, **kwargs)


def create_multiple_servers(count: int = 5, **kwargs: Any) -> list[ServerDetail]:
    """
    Create multiple ServerDetail instances.

    Args:
        count: Number of servers to create
        **kwargs: Additional ServerDetail attributes

    Returns:
        List of ServerDetail instances
    """
    return [ServerDetailFactory(**kwargs) for _ in range(count)]


def create_multiple_agents(count: int = 5, **kwargs: Any) -> list[AgentCard]:
    """
    Create multiple AgentCard instances.

    Args:
        count: Number of agents to create
        **kwargs: Additional AgentCard attributes

    Returns:
        List of AgentCard instances
    """
    return [AgentCardFactory(**kwargs) for _ in range(count)]


def create_server_dict(name: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Create a server dictionary (not a Pydantic model).

    Useful for testing JSON serialization/deserialization.

    Args:
        name: Server name
        **kwargs: Additional server attributes

    Returns:
        Server dictionary
    """
    server = ServerDetailFactory(name=name, **kwargs)
    return server.model_dump(by_alias=True, exclude_none=True)


def create_agent_dict(name: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Create an agent dictionary (not a Pydantic model).

    Useful for testing JSON serialization/deserialization.

    Args:
        name: Agent name
        **kwargs: Additional agent attributes

    Returns:
        Agent dictionary
    """
    agent = AgentCardFactory(name=name, **kwargs)
    return agent.model_dump(by_alias=True, exclude_none=True)
