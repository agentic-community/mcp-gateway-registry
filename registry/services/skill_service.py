"""
Service layer for skill management.

Simplified design:
- No in-memory state duplication
- Database as source of truth
- SKILL.md URL validation on registration
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import httpx

from ..exceptions import (
    SkillNotFoundError,
    SkillServiceError,
    SkillUrlValidationError,
    SkillValidationError,
)
from ..repositories.factory import (
    get_search_repository,
    get_skill_repository,
)
from ..repositories.interfaces import (
    SearchRepositoryBase,
    SkillRepositoryBase,
)
from ..schemas.skill_models import (
    SkillCard,
    SkillInfo,
    SkillMetadata,
    SkillRegistrationRequest,
    VisibilityEnum,
)
from ..utils.path_utils import normalize_skill_path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Constants
URL_VALIDATION_TIMEOUT: int = 10


async def _validate_skill_md_url(
    url: str,
) -> Dict[str, Any]:
    """Validate SKILL.md URL is accessible and get content hash.

    Args:
        url: URL to SKILL.md file

    Returns:
        Dict with validation result and content hash

    Raises:
        SkillUrlValidationError: If URL is not accessible
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                str(url),
                follow_redirects=True,
                timeout=URL_VALIDATION_TIMEOUT
            )

            if response.status_code >= 400:
                raise SkillUrlValidationError(
                    url,
                    f"HTTP {response.status_code}"
                )

            # Generate content hash for versioning
            content_hash = hashlib.sha256(
                response.content
            ).hexdigest()[:16]

            return {
                "valid": True,
                "content_version": content_hash,
                "content_updated_at": datetime.now(timezone.utc)
            }

    except httpx.RequestError as e:
        raise SkillUrlValidationError(url, str(e)) from e


def _build_skill_card(
    request: SkillRegistrationRequest,
    path: str,
    owner: Optional[str],
    content_version: Optional[str],
    content_updated_at: Optional[datetime],
) -> SkillCard:
    """Build SkillCard from registration request.

    Args:
        request: Registration request
        path: Skill path
        owner: Owner username/email
        content_version: Content hash
        content_updated_at: Content update timestamp

    Returns:
        SkillCard instance
    """
    # Convert metadata dict to SkillMetadata if provided
    metadata = None
    if request.metadata:
        metadata = SkillMetadata(
            author=request.metadata.get("author"),
            version=request.metadata.get("version"),
            extra={
                k: v for k, v in request.metadata.items()
                if k not in ("author", "version")
            }
        )

    return SkillCard(
        path=path,
        name=request.name,
        description=request.description,
        skill_md_url=request.skill_md_url,
        repository_url=request.repository_url,
        license=request.license,
        compatibility=request.compatibility,
        requirements=request.requirements,
        target_agents=request.target_agents,
        metadata=metadata,
        allowed_tools=request.allowed_tools,
        tags=request.tags,
        visibility=request.visibility,
        allowed_groups=request.allowed_groups,
        owner=owner,
        is_enabled=True,
        content_version=content_version,
        content_updated_at=content_updated_at,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class SkillService:
    """Service for skill CRUD operations.

    Simplified design with no in-memory state duplication.
    Database is the source of truth.
    """

    def __init__(self):
        self._repo: Optional[SkillRepositoryBase] = None
        self._search_repo: Optional[SearchRepositoryBase] = None


    def _get_repo(self) -> SkillRepositoryBase:
        """Lazy initialization of repository."""
        if self._repo is None:
            self._repo = get_skill_repository()
        return self._repo


    def _get_search_repo(self) -> SearchRepositoryBase:
        """Lazy initialization of search repository."""
        if self._search_repo is None:
            self._search_repo = get_search_repository()
        return self._search_repo


    async def register_skill(
        self,
        request: SkillRegistrationRequest,
        owner: Optional[str] = None,
        validate_url: bool = True,
    ) -> SkillCard:
        """Register a new skill.

        Args:
            request: Skill registration request
            owner: Owner username/email for access control
            validate_url: Whether to validate SKILL.md URL

        Returns:
            Created SkillCard

        Raises:
            SkillUrlValidationError: If URL validation fails
            SkillAlreadyExistsError: If skill name exists
        """
        # Generate path
        path = normalize_skill_path(request.name)

        # Validate URL and get content hash
        content_version = None
        content_updated_at = None

        if validate_url:
            validation = await _validate_skill_md_url(str(request.skill_md_url))
            content_version = validation["content_version"]
            content_updated_at = validation["content_updated_at"]

        # Build SkillCard
        skill = _build_skill_card(
            request=request,
            path=path,
            owner=owner,
            content_version=content_version,
            content_updated_at=content_updated_at
        )

        # Save to repository
        repo = self._get_repo()
        created_skill = await repo.create(skill)

        # Index for search
        try:
            search_repo = self._get_search_repo()
            await search_repo.index_skill(
                path=path,
                skill=created_skill,
                is_enabled=True,
            )
        except Exception as e:
            logger.warning(f"Failed to index skill for search: {e}")

        logger.info(f"Registered skill: {path}")
        return created_skill


    async def get_skill(
        self,
        path: str,
    ) -> Optional[SkillCard]:
        """Get a skill by path."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        return await repo.get(normalized)


    async def list_skills(
        self,
        include_disabled: bool = False,
        tag: Optional[str] = None,
        visibility: Optional[str] = None,
        registry_name: Optional[str] = None,
    ) -> List[SkillInfo]:
        """List skills with optional filtering.

        Uses database-level filtering for performance.

        Args:
            include_disabled: Whether to include disabled skills
            tag: Filter by tag
            visibility: Filter by visibility
            registry_name: Filter by registry

        Returns:
            List of SkillInfo summaries
        """
        repo = self._get_repo()
        skills = await repo.list_filtered(
            include_disabled=include_disabled,
            tag=tag,
            visibility=visibility,
            registry_name=registry_name,
        )

        return [
            SkillInfo(
                path=s.path,
                name=s.name,
                description=s.description,
                skill_md_url=str(s.skill_md_url),
                tags=s.tags,
                author=s.metadata.author if s.metadata else None,
                version=s.metadata.version if s.metadata else None,
                compatibility=s.compatibility,
                target_agents=s.target_agents,
                is_enabled=s.is_enabled,
                visibility=s.visibility,
                allowed_groups=s.allowed_groups,
                registry_name=s.registry_name,
            )
            for s in skills
        ]


    async def list_skills_for_user(
        self,
        user_context: Optional[Dict[str, Any]],
        include_disabled: bool = False,
        tag: Optional[str] = None,
    ) -> List[SkillInfo]:
        """List skills filtered by user's visibility access.

        Args:
            user_context: User context with groups and username
            include_disabled: Whether to include disabled skills
            tag: Filter by tag

        Returns:
            List of SkillInfo visible to user
        """
        all_skills = await self.list_skills(
            include_disabled=include_disabled,
            tag=tag,
        )

        if not user_context:
            # Anonymous - only public
            return [s for s in all_skills if s.visibility == VisibilityEnum.PUBLIC]

        if user_context.get("is_admin"):
            return all_skills

        user_groups = set(user_context.get("groups", []))
        username = user_context.get("username", "")

        filtered = []
        for skill in all_skills:
            if skill.visibility == VisibilityEnum.PUBLIC:
                filtered.append(skill)
            elif skill.visibility == VisibilityEnum.PRIVATE:
                # Check owner - need to get full skill
                full_skill = await self.get_skill(skill.path)
                if full_skill and full_skill.owner == username:
                    filtered.append(skill)
            elif skill.visibility == VisibilityEnum.GROUP:
                if user_groups & set(skill.allowed_groups):
                    filtered.append(skill)

        return filtered


    async def update_skill(
        self,
        path: str,
        updates: Dict[str, Any],
    ) -> Optional[SkillCard]:
        """Update a skill."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        updated = await repo.update(normalized, updates)

        if updated:
            # Update search index
            try:
                search_repo = self._get_search_repo()
                await search_repo.index_skill(
                    path=normalized,
                    skill=updated,
                    is_enabled=updated.is_enabled,
                )
            except Exception as e:
                logger.warning(f"Failed to update skill in search index: {e}")
            logger.info(f"Updated skill: {normalized}")

        return updated


    async def delete_skill(
        self,
        path: str,
    ) -> bool:
        """Delete a skill."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        success = await repo.delete(normalized)

        if success:
            # Remove from search index
            try:
                search_repo = self._get_search_repo()
                await search_repo.remove_entity(normalized)
            except Exception as e:
                logger.warning(f"Failed to remove skill from search index: {e}")
            logger.info(f"Deleted skill: {normalized}")

        return success


    async def toggle_skill(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Toggle skill enabled state."""
        normalized = normalize_skill_path(path)
        repo = self._get_repo()
        success = await repo.set_state(normalized, enabled)

        if success:
            # Update search index
            skill = await repo.get(normalized)
            if skill:
                try:
                    search_repo = self._get_search_repo()
                    await search_repo.index_skill(
                        path=normalized,
                        skill=skill,
                        is_enabled=enabled,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update skill in search index: {e}")
            logger.info(f"Toggled skill {normalized} to enabled={enabled}")

        return success


# Singleton instance
_skill_service: Optional[SkillService] = None


def get_skill_service() -> SkillService:
    """Get or create skill service singleton."""
    global _skill_service
    if _skill_service is None:
        _skill_service = SkillService()
    return _skill_service
