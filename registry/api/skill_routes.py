"""
API routes for skill management.

All recommendations implemented:
- Authentication required on all endpoints
- Visibility filtering in list operations
- Path normalization via dependency
- Domain-specific exception handling
- Discovery endpoint for coding assistants
- Resource listing endpoints
"""

import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from pydantic import BaseModel

from ..audit.context import set_audit_action
from ..auth.csrf import verify_csrf_token_flexible
from ..auth.dependencies import nginx_proxied_auth
from ..exceptions import (
    SkillAlreadyExistsError,
    SkillServiceError,
    SkillUrlValidationError,
    SkillValidationError,
)
from ..schemas.skill_models import (
    DiscoveryResponse,
    SkillCard,
    SkillMetadata,
    SkillRegistrationRequest,
    SkillTier1_Metadata,
    ToggleStateRequest,
    ToolValidationResult,
    VisibilityEnum,
)
from ..services.skill_service import (
    _is_safe_url,
    get_skill_service,
)
from ..services.tool_validation_service import get_tool_validation_service
from ..utils.path_utils import normalize_skill_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class RatingRequest(BaseModel):
    """Request model for rating a skill."""

    rating: int


router = APIRouter(prefix="/skills", tags=["skills"])


# Dependency for normalized path
def get_normalized_path(
    skill_path: str = Path(..., description="Skill path or name"),
) -> str:
    """Normalize skill path."""
    return normalize_skill_path(skill_path)


@router.get(
    "/discovery",
    response_model=DiscoveryResponse,
    summary="Discovery endpoint for coding assistants",
)
async def discover_skills(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    query: str | None = Query(None, description="Search query"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
    compatibility: str | None = Query(None, description="Filter by compatibility"),
    page: int = Query(0, ge=0),
    page_size: int = Query(100, ge=1, le=500),
) -> DiscoveryResponse:
    """Discovery endpoint optimized for coding assistants.

    Returns lightweight metadata for efficient loading.
    """
    service = get_skill_service()
    skills = await service.list_skills_for_user(user_context)

    # Apply filters
    if tags:
        skills = [s for s in skills if any(t in s.tags for t in tags)]

    if compatibility:
        skills = [
            s
            for s in skills
            if s.compatibility and compatibility.lower() in s.compatibility.lower()
        ]

    # Pagination
    total = len(skills)
    start = page * page_size
    end = start + page_size
    paginated = skills[start:end]

    # Convert to Tier1 metadata
    tier1_skills = [
        SkillTier1_Metadata(
            path=s.path,
            name=s.name,
            description=s.description,
            skill_md_url=s.skill_md_url,
            skill_md_raw_url=s.skill_md_raw_url,
            tags=s.tags,
            compatibility=s.compatibility,
            target_agents=s.target_agents,
        )
        for s in paginated
    ]

    return DiscoveryResponse(
        skills=tier1_skills,
        total_count=total,
        page=page,
        page_size=page_size,
    )


@router.get("", summary="List all skills")
async def list_skills(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    include_disabled: bool = Query(False, description="Include disabled skills"),
    tag: str | None = Query(None, description="Filter by tag"),
) -> dict:
    """List all registered skills with visibility filtering."""
    service = get_skill_service()
    skills = await service.list_skills_for_user(
        user_context=user_context,
        include_disabled=include_disabled,
        tag=tag,
    )
    logger.info(
        f"Returning {len(skills)} skills for user {user_context.get('username', 'unknown')}"
    )
    return {
        "skills": [skill.model_dump(mode="json") for skill in skills],
        "total_count": len(skills),
    }


@router.post("/parse-skill-md", summary="Parse SKILL.md content from URL")
async def parse_skill_md(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    url: str = Query(..., description="URL to SKILL.md file"),
) -> dict:
    """Parse SKILL.md content and extract metadata.

    Returns name, description, version, and tags from the SKILL.md file.
    Useful for auto-populating the skill registration form.
    """
    service = get_skill_service()
    try:
        result = await service.parse_skill_md(url)
        return {
            "success": True,
            "name": result.get("name"),
            "name_slug": result.get("name_slug"),
            "description": result.get("description"),
            "version": result.get("version"),
            "tags": result.get("tags", []),
            "content_version": result.get("content_version"),
            "skill_md_url": result.get("skill_md_url"),
            "skill_md_raw_url": result.get("skill_md_raw_url"),
        }
    except SkillUrlValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse SKILL.md: {e.reason}"
        )


@router.get("/search", summary="Search skills")
async def search_skills(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    q: str = Query(..., description="Search query"),
    tags: str | None = Query(None, description="Comma-separated tags to filter by"),
) -> dict:
    """Search for skills by name, description, or tags.

    Returns skills matching the query with basic relevance scoring.
    """
    service = get_skill_service()
    skills = await service.list_skills_for_user(user_context)

    query_lower = q.lower()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    matching_skills = []
    for skill in skills:
        score = 0.0

        # Match in name (highest priority)
        if query_lower in skill.name.lower():
            score += 0.5

        # Match in description
        if skill.description and query_lower in skill.description.lower():
            score += 0.3

        # Match in tags
        skill_tags_lower = [t.lower() for t in (skill.tags or [])]
        if any(query_lower in t for t in skill_tags_lower):
            score += 0.2

        # Filter by specified tags
        if tag_list:
            if not all(t.lower() in skill_tags_lower for t in tag_list):
                continue

        if score > 0:
            matching_skills.append(
                {
                    "path": skill.path,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "visibility": skill.visibility,
                    "is_enabled": skill.is_enabled,
                    "relevance_score": score,
                }
            )

    # Sort by relevance score descending
    matching_skills.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "query": q,
        "skills": matching_skills,
        "total_count": len(matching_skills),
    }


@router.get("/{skill_path:path}/health", summary="Check skill health")
async def check_skill_health(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Check skill health by performing HEAD request to SKILL.md URL.

    Returns health status, HTTP status code, and response time.
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    result = await service.check_skill_health(normalized_path)
    return {
        "path": normalized_path,
        "healthy": result["healthy"],
        "status_code": result["status_code"],
        "error": result["error"],
        "response_time_ms": result["response_time_ms"],
    }


@router.get("/{skill_path:path}/content", summary="Get SKILL.md content")
async def get_skill_content(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Fetch SKILL.md content from the raw URL.

    This endpoint proxies the SKILL.md content to avoid CORS issues
    when the frontend tries to fetch from GitHub directly.
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {normalized_path}",
        )

    # Check visibility
    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Fetch content from raw URL
    raw_url = skill.skill_md_raw_url or skill.skill_md_url
    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No SKILL.md URL configured for this skill",
        )

    # SSRF protection: validate URL before making request
    if not _is_safe_url(str(raw_url)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL failed SSRF validation - private/internal addresses are not allowed",
        )

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(str(raw_url), follow_redirects=True, timeout=30.0)

            # SSRF protection: validate final URL after redirects
            final_url = str(response.url)
            if final_url != str(raw_url) and not _is_safe_url(final_url):
                logger.warning(
                    f"SSRF protection: Blocked redirect from {raw_url} to unsafe URL {final_url}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Redirect to unsafe URL blocked: {final_url}",
                )

            if response.status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch SKILL.md: HTTP {response.status_code}",
                )

            return {
                "content": response.text,
                "url": str(raw_url),
            }
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch SKILL.md from {raw_url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch SKILL.md: {e}",
        )


@router.get(
    "/{skill_path:path}/tools",
    response_model=ToolValidationResult,
    summary="Get required tools with availability",
)
async def get_skill_tools(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> ToolValidationResult:
    """Get required tools for a skill with availability status."""
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    tool_service = get_tool_validation_service()
    return await tool_service.validate_tools_available(skill)


@router.get("/{skill_path:path}/rating", response_model=dict, summary="Get skill rating")
async def get_skill_rating(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Get rating information for a skill.

    Returns the average rating and list of individual ratings.
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()

    # Check skill exists and user has access
    skill = await service.get_skill(normalized_path)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this skill"
        )

    return {
        "num_stars": skill.num_stars,
        "rating_details": skill.rating_details,
    }


# ---------------------------------------------------------------------------
# Security scan endpoints (must be before catch-all GET /{skill_path:path})
# ---------------------------------------------------------------------------


@router.get(
    "/{skill_path:path}/security-scan",
    response_model=dict,
    summary="Get skill security scan results",
)
async def get_skill_security_scan(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path"),
) -> dict:
    """Get the latest security scan results for a skill."""
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()

    skill = await service.get_skill(normalized_path)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {normalized_path}",
        )

    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    from ..services.skill_scanner import skill_scanner_service

    scan_result = await skill_scanner_service.get_scan_result(normalized_path)
    if not scan_result:
        return {"message": "No security scan results available", "skill_path": normalized_path}

    return scan_result


@router.post(
    "/{skill_path:path}/rescan",
    response_model=dict,
    summary="Trigger manual security scan",
)
async def rescan_skill(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path"),
) -> dict:
    """Trigger a manual security scan for a skill. Admin only."""
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()

    skill = await service.get_skill(normalized_path)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {normalized_path}",
        )

    set_audit_action(
        http_request,
        "rescan",
        "skill",
        resource_id=normalized_path,
        description=f"Manual security scan for skill {normalized_path}",
    )

    from ..services.skill_scanner import skill_scanner_service

    try:
        result = await skill_scanner_service.scan_skill(
            skill_path=normalized_path,
            skill_md_url=str(skill.skill_md_raw_url or skill.skill_md_url),
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"Manual security scan failed for skill '{normalized_path}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security scan failed: {str(e)}",
        )


@router.get("/{skill_path:path}", response_model=SkillCard, summary="Get a skill by path")
async def get_skill(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> SkillCard:
    """Get a specific skill by its path."""
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    # Check visibility
    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return skill


@router.post(
    "",
    response_model=SkillCard,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new skill",
)
async def register_skill(
    http_request: Request,
    request: SkillRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> SkillCard:
    """Register a new skill in the registry."""
    # Set audit action for skill registration
    # Note: path is derived from name, so use name as resource_id
    set_audit_action(
        http_request,
        "create",
        "skill",
        resource_id=request.name,
        description=f"Register skill {request.name}",
    )

    service = get_skill_service()
    owner = user_context.get("username")

    try:
        skill = await service.register_skill(request=request, owner=owner, validate_url=True)
        logger.info(f"Registered skill: {skill.name} by {owner}")

        # Perform security scan on registration (non-blocking on failure)
        await _perform_skill_security_scan_on_registration(skill, service)

        return skill

    except SkillUrlValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid SKILL.md URL: {e.reason}"
        )
    except SkillAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except SkillValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SkillServiceError as e:
        logger.error(f"Failed to register skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register skill"
        )


@router.put("/{skill_path:path}", response_model=SkillCard, summary="Update a skill")
async def update_skill(
    http_request: Request,
    request: SkillRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> SkillCard:
    """Update an existing skill."""
    normalized_path = normalize_skill_path(skill_path)

    # Set audit action for skill update
    set_audit_action(
        http_request,
        "update",
        "skill",
        resource_id=normalized_path,
        description=f"Update skill {request.name}",
    )

    service = get_skill_service()

    # Check ownership
    existing = await service.get_skill(normalized_path)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_modify_skill(existing, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    updates = request.model_dump(exclude_unset=True, mode="json")

    # Convert raw metadata dict to SkillMetadata structure for consistent storage
    if "metadata" in updates and updates["metadata"] is not None:
        raw_meta = updates["metadata"]
        updates["metadata"] = SkillMetadata(
            author=raw_meta.get("author"),
            version=raw_meta.get("version"),
            extra={k: v for k, v in raw_meta.items() if k not in ("author", "version")},
        ).model_dump(mode="json")

    updated = await service.update_skill(normalized_path, updates)

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    return updated


@router.delete(
    "/{skill_path:path}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a skill"
)
async def delete_skill(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> None:
    """Delete a skill from the registry."""
    normalized_path = normalize_skill_path(skill_path)

    # Set audit action for skill deletion
    set_audit_action(
        http_request,
        "delete",
        "skill",
        resource_id=normalized_path,
        description=f"Delete skill at {normalized_path}",
    )

    service = get_skill_service()

    # Check ownership
    existing = await service.get_skill(normalized_path)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_modify_skill(existing, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    success = await service.delete_skill(normalized_path)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )


@router.post("/{skill_path:path}/toggle", response_model=dict, summary="Toggle skill enabled state")
async def toggle_skill(
    http_request: Request,
    request: ToggleStateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Toggle a skill's enabled state."""
    normalized_path = normalize_skill_path(skill_path)

    # Set audit action for skill toggle
    set_audit_action(
        http_request,
        "toggle",
        "skill",
        resource_id=normalized_path,
        description=f"Toggle skill to {request.enabled}",
    )

    service = get_skill_service()

    # Check ownership
    existing = await service.get_skill(normalized_path)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_modify_skill(existing, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    success = await service.toggle_skill(normalized_path, request.enabled)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    return {"path": normalized_path, "is_enabled": request.enabled}


@router.post("/{skill_path:path}/rate", response_model=dict, summary="Rate a skill")
async def rate_skill(
    http_request: Request,
    rating_request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Submit a rating for a skill.

    Users can rate skills from 1-5 stars. Each user can only have one
    rating per skill - submitting a new rating updates the previous one.
    """
    normalized_path = normalize_skill_path(skill_path)

    # Set audit action for skill rating
    set_audit_action(
        http_request,
        "rate",
        "skill",
        resource_id=normalized_path,
        description=f"Rate skill with {rating_request.rating}",
    )

    service = get_skill_service()

    # Check skill exists and user has access
    skill = await service.get_skill(normalized_path)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_access_skill(skill, user_context):
        logger.warning(
            f"User {user_context.get('username')} attempted to rate skill "
            f"{normalized_path} without permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this skill"
        )

    try:
        avg_rating = await service.update_rating(
            normalized_path, user_context["username"], rating_request.rating
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {
        "message": "Rating added successfully",
        "average_rating": avg_rating,
    }


# Helper functions


def _user_can_access_skill(
    skill: SkillCard,
    user_context: dict,
) -> bool:
    """Check if user can access skill based on visibility."""
    if user_context.get("is_admin"):
        return True

    visibility = skill.visibility

    if visibility == VisibilityEnum.PUBLIC:
        return True

    if visibility == VisibilityEnum.PRIVATE:
        return skill.owner == user_context.get("username")

    if visibility == VisibilityEnum.GROUP:
        user_groups = set(user_context.get("groups", []))
        return bool(user_groups & set(skill.allowed_groups))

    return False


def _user_can_modify_skill(
    skill: SkillCard,
    user_context: dict,
) -> bool:
    """Check if user can modify skill."""
    if user_context.get("is_admin"):
        return True

    return skill.owner == user_context.get("username")


async def _perform_skill_security_scan_on_registration(
    skill: SkillCard,
    service,
) -> None:
    """
    Perform security scan on skill registration.

    Args:
        skill: The registered skill card
        service: The skill service instance
    """
    from ..services.skill_scanner import skill_scanner_service

    config = skill_scanner_service.get_scan_config()

    if not config.enabled or not config.scan_on_registration:
        logger.info("Skill security scanning disabled, skipping")
        return

    logger.info(f"Performing security scan for skill: {skill.path}")

    try:
        result = await skill_scanner_service.scan_skill(
            skill_path=skill.path,
            skill_md_url=str(skill.skill_md_raw_url or skill.skill_md_url),
        )

        if not result.is_safe and config.block_unsafe_skills:
            logger.warning(f"Disabling unsafe skill: {skill.path}")
            await service.toggle_skill(skill.path, enabled=False)

            if config.add_security_pending_tag:
                current_tags = skill.tags or []
                if "security-pending" not in current_tags:
                    await service.update_skill(
                        skill.path, {"tags": current_tags + ["security-pending"]}
                    )

    except Exception as e:
        logger.error(f"Security scan failed for skill {skill.path}: {e}")
        if config.add_security_pending_tag:
            try:
                current_tags = skill.tags or []
                if "security-pending" not in current_tags:
                    await service.update_skill(
                        skill.path, {"tags": current_tags + ["security-pending"]}
                    )
            except Exception as tag_err:
                logger.error(f"Failed to add security-pending tag: {tag_err}")
