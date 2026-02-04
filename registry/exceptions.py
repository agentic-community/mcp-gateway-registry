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
