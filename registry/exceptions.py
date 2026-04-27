"""
Domain-specific exceptions for the MCP Gateway Registry.

This module contains custom exception classes for various operations
including skill management, agent management, and server operations.
"""


class RegistryError(Exception):
    """Base exception for all registry operations."""

    pass


# Skill-specific exceptions


class SkillRegistryError(RegistryError):
    """Base exception for skill operations."""

    pass


class SkillNotFoundError(SkillRegistryError):
    """Skill does not exist."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Skill not found: {path}")


class SkillAlreadyExistsError(SkillRegistryError):
    """Skill with this name already exists."""

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        super().__init__(f"Skill '{name}' already exists")


class SkillValidationError(SkillRegistryError):
    """Skill data failed validation."""

    pass


class SkillServiceError(SkillRegistryError):
    """Internal service error during skill operation."""

    pass


class SkillUrlValidationError(SkillRegistryError):
    """SKILL.md URL validation failed."""

    def __init__(
        self,
        url: str,
        reason: str,
    ):
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid SKILL.md URL '{url}': {reason}")


# Agent-specific exceptions


class AgentRegistryError(RegistryError):
    """Base exception for agent operations."""

    pass


class AgentNotFoundError(AgentRegistryError):
    """Agent does not exist."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Agent not found: {path}")


class AgentAlreadyExistsError(AgentRegistryError):
    """Agent with this path already exists."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Agent already exists at path: {path}")


# Server-specific exceptions


class ServerRegistryError(RegistryError):
    """Base exception for server operations."""

    pass


class ServerNotFoundError(ServerRegistryError):
    """Server does not exist."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Server not found: {path}")


class ServerAlreadyExistsError(ServerRegistryError):
    """Server with this path already exists."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Server already exists at path: {path}")


# Virtual Server-specific exceptions


class VirtualServerRegistryError(RegistryError):
    """Base exception for virtual server operations."""

    pass


class VirtualServerNotFoundError(VirtualServerRegistryError):
    """Virtual server does not exist."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Virtual server not found: {path}")


class VirtualServerAlreadyExistsError(VirtualServerRegistryError):
    """Virtual server with this path already exists."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Virtual server already exists at path: {path}")


class VirtualServerValidationError(VirtualServerRegistryError):
    """Virtual server data failed validation."""

    pass


class VirtualServerServiceError(VirtualServerRegistryError):
    """Internal service error during virtual server operation."""

    pass


# Skill content fetch exceptions


class SkillContentFetchError(SkillRegistryError):
    """Failed to fetch skill content from a remote URL."""

    def __init__(
        self,
        url: str,
        reason: str,
        status_code: int = 502,
    ):
        self.url = url
        self.reason = reason
        self.status_code = status_code
        super().__init__(f"Failed to fetch content from '{url}': {reason}")


class SkillContentSSRFError(SkillRegistryError):
    """URL failed SSRF validation."""

    def __init__(
        self,
        url: str,
    ):
        self.url = url
        super().__init__(
            f"URL failed SSRF validation: {url}"
        )


class SkillContentTooLargeError(SkillRegistryError):
    """Fetched content exceeds the size limit."""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size
        super().__init__(
            f"Content exceeds {max_size // 1024} KB limit"
        )


# Registration Gate exceptions


class RegistrationGateError(RegistryError):
    """Base exception for registration gate operations."""

    pass


class RegistrationGateDeniedError(RegistrationGateError):
    """Registration was denied by the gate endpoint."""

    def __init__(
        self,
        reason: str,
    ):
        self.reason = reason
        super().__init__(f"Registration denied by policy gate: {reason}")


class RegistrationGateUnavailableError(RegistrationGateError):
    """Gate endpoint is unreachable or returned an unexpected error."""

    def __init__(
        self,
        detail: str,
    ):
        self.detail = detail
        super().__init__(
            f"Registration gate is unavailable: {detail}. "
            f"Registration blocked (fail-closed policy)."
        )
