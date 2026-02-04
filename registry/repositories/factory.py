"""
Repository factory - creates concrete implementations based on configuration.
"""

import logging
from typing import Optional

from ..core.config import settings
from .interfaces import (
    ServerRepositoryBase,
    AgentRepositoryBase,
    ScopeRepositoryBase,
    SecurityScanRepositoryBase,
    SearchRepositoryBase,
    FederationConfigRepositoryBase,
    SkillRepositoryBase,
)

logger = logging.getLogger(__name__)

# Singleton instances
_server_repo: Optional[ServerRepositoryBase] = None
_agent_repo: Optional[AgentRepositoryBase] = None
_scope_repo: Optional[ScopeRepositoryBase] = None
_security_scan_repo: Optional[SecurityScanRepositoryBase] = None
_search_repo: Optional[SearchRepositoryBase] = None
_federation_config_repo: Optional[FederationConfigRepositoryBase] = None
_skill_repo: Optional[SkillRepositoryBase] = None


def get_server_repository() -> ServerRepositoryBase:
    """Get server repository singleton."""
    global _server_repo

    if _server_repo is not None:
        return _server_repo

    backend = settings.storage_backend
    logger.info(f"Creating server repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.server_repository import DocumentDBServerRepository
        _server_repo = DocumentDBServerRepository()
    else:
        from .file.server_repository import FileServerRepository
        _server_repo = FileServerRepository()

    return _server_repo


def get_agent_repository() -> AgentRepositoryBase:
    """Get agent repository singleton."""
    global _agent_repo

    if _agent_repo is not None:
        return _agent_repo

    backend = settings.storage_backend
    logger.info(f"Creating agent repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.agent_repository import DocumentDBAgentRepository
        _agent_repo = DocumentDBAgentRepository()
    else:
        from .file.agent_repository import FileAgentRepository
        _agent_repo = FileAgentRepository()

    return _agent_repo


def get_scope_repository() -> ScopeRepositoryBase:
    """Get scope repository singleton."""
    global _scope_repo

    if _scope_repo is not None:
        return _scope_repo

    backend = settings.storage_backend
    logger.info(f"Creating scope repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.scope_repository import DocumentDBScopeRepository
        _scope_repo = DocumentDBScopeRepository()
    else:
        from .file.scope_repository import FileScopeRepository
        _scope_repo = FileScopeRepository()

    return _scope_repo


def get_security_scan_repository() -> SecurityScanRepositoryBase:
    """Get security scan repository singleton."""
    global _security_scan_repo

    if _security_scan_repo is not None:
        return _security_scan_repo

    backend = settings.storage_backend
    logger.info(f"Creating security scan repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.security_scan_repository import DocumentDBSecurityScanRepository
        _security_scan_repo = DocumentDBSecurityScanRepository()
    else:
        from .file.security_scan_repository import FileSecurityScanRepository
        _security_scan_repo = FileSecurityScanRepository()

    return _security_scan_repo


def get_search_repository() -> SearchRepositoryBase:
    """Get search repository singleton."""
    global _search_repo

    if _search_repo is not None:
        return _search_repo

    backend = settings.storage_backend
    logger.info(f"Creating search repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.search_repository import DocumentDBSearchRepository
        _search_repo = DocumentDBSearchRepository()
    else:
        from .file.search_repository import FaissSearchRepository
        _search_repo = FaissSearchRepository()

    return _search_repo


def get_federation_config_repository() -> FederationConfigRepositoryBase:
    """Get federation config repository singleton."""
    global _federation_config_repo

    if _federation_config_repo is not None:
        return _federation_config_repo

    backend = settings.storage_backend
    logger.info(f"Creating federation config repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.federation_config_repository import DocumentDBFederationConfigRepository
        _federation_config_repo = DocumentDBFederationConfigRepository()
    else:
        from .file.federation_config_repository import FileFederationConfigRepository
        _federation_config_repo = FileFederationConfigRepository()

    return _federation_config_repo


def get_skill_repository() -> SkillRepositoryBase:
    """Get skill repository singleton."""
    global _skill_repo

    if _skill_repo is not None:
        return _skill_repo

    backend = settings.storage_backend
    logger.info(f"Creating skill repository with backend: {backend}")

    if backend in ("documentdb", "mongodb-ce"):
        from .documentdb.skill_repository import DocumentDBSkillRepository
        _skill_repo = DocumentDBSkillRepository()
    else:
        # File-based skill repository not implemented yet
        # Fall back to DocumentDB repository for now
        from .documentdb.skill_repository import DocumentDBSkillRepository
        _skill_repo = DocumentDBSkillRepository()

    return _skill_repo


def reset_repositories() -> None:
    """Reset all repository singletons. USE ONLY IN TESTS."""
    global _server_repo, _agent_repo, _scope_repo, _security_scan_repo, _search_repo, _federation_config_repo, _skill_repo
    _server_repo = None
    _agent_repo = None
    _scope_repo = None
    _security_scan_repo = None
    _search_repo = None
    _federation_config_repo = None
    _skill_repo = None
