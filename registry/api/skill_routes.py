"""
API routes for skill management.

All recommendations implemented:
- Authentication required on all endpoints
- Visibility filtering in list operations
- Path normalization via dependency
- Domain-specific exception handling
- Discovery endpoint for coding assistants
- Resource content served via the /content endpoint
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import (
    Annotated,
    Any,
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
    SkillInfo,
    SkillMetadata,
    SkillRegistrationRequest,
    SkillTier1_Metadata,
    ToggleStateRequest,
    ToolValidationResult,
    VisibilityEnum,
)
from ..exceptions import (
    SkillContentFetchError,
    SkillContentSSRFError,
    SkillContentTooLargeError,
)
from ..services.skill_service import (
    _build_fetch_headers,
    _check_drift_inline,
    _decrypt_skill_auth,
    _discover_skill_resources,
    _fetch_authenticated_content,
    _is_safe_url,
    get_skill_service,
)
from ..services.registration_gate_service import check_registration_gate
from ..services.tool_validation_service import get_tool_validation_service
from ..services.webhook_service import send_registration_webhook
from ..utils.metadata import flatten_metadata_to_text
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

_SKILL_CARD_EXCLUDE = {"auth_credential_encrypted"}


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
    limit: int = Query(20, ge=1, le=500, description="Number of skills to return (max 500)"),
    offset: int = Query(0, ge=0, description="Number of skills to skip"),
) -> dict:
    """List all registered skills with visibility filtering and pagination."""
    logger.debug(
        f"list_skills called: limit={limit}, offset={offset}, "
        f"tag={tag!r}, include_disabled={include_disabled}"
    )

    service = get_skill_service()

    # Determine if user has unrestricted access (no skills will be filtered out)
    is_admin = user_context.get("is_admin", False) if user_context else False
    accessible_agent_list = user_context.get("accessible_agents", []) if user_context else []
    is_unrestricted = is_admin or "all" in accessible_agent_list
    # include_disabled=False (default) means "exclude disabled" which IS a filter.
    # Only include_disabled=True (show all) with no tag requires no filtering.
    has_field_filters = bool(tag or not include_disabled)

    # Dual-path pagination:
    # - Fast path: DB-level skip/limit for unrestricted users without field filters
    # - Fallback: full fetch + Python filter + slice for restricted users or field filters
    if is_unrestricted and not has_field_filters:
        # FAST PATH: DB-level pagination -- correct because no skills are filtered out
        # and no field filters need a full scan for accurate total_count
        skill_cards, db_total = await service.get_skills_paginated(
            skip=offset,
            limit=limit,
        )
        skills = [
            SkillInfo(
                id=s.id,
                path=s.path,
                name=s.name,
                description=s.description,
                skill_md_url=str(s.skill_md_url),
                skill_md_raw_url=str(s.skill_md_raw_url) if s.skill_md_raw_url else None,
                repository_url=s.repository_url,
                tags=s.tags,
                author=s.metadata.author if s.metadata else None,
                version=s.metadata.version if s.metadata else None,
                metadata=s.metadata,
                compatibility=s.compatibility,
                target_agents=s.target_agents,
                is_enabled=s.is_enabled,
                visibility=s.visibility,
                allowed_groups=s.allowed_groups,
                registry_name=s.registry_name,
                owner=s.owner,
                auth_scheme=s.auth_scheme,
                auth_header_name=s.auth_header_name,
                num_stars=s.num_stars,
                health_status=s.health_status,
                last_checked_time=s.last_checked_time,
                status=s.status,
            )
            for s in skill_cards
        ]
        total_count = db_total
        page_skills = skills
    else:
        # FALLBACK PATH: full fetch needed for filtering or restricted users
        all_skills = await service.list_skills_for_user(
            user_context=user_context,
            include_disabled=include_disabled,
            tag=tag,
        )
        total_count = len(all_skills)
        page_skills = all_skills[offset : offset + limit]

    has_next = (offset + limit) < total_count

    logger.info(
        f"Returning {len(page_skills)} skills for user "
        f"{user_context.get('username', 'unknown')} "
        f"(total: {total_count}, offset: {offset}, limit: {limit})"
    )
    return {
        "skills": [skill.model_dump(mode="json") for skill in page_skills],
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "has_next": has_next,
    }


@router.post("/parse-skill-md", summary="Parse SKILL.md content from URL")
async def parse_skill_md(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    url: str = Query(..., description="URL to SKILL.md file"),
    auth_scheme: str = Query("none", description="Auth scheme: none, global_credentials, bearer, api_key"),
    auth_credential: str | None = Query(None, description="Plaintext credential for bearer/api_key"),
    auth_header_name: str | None = Query(None, description="Custom header name for api_key scheme"),
) -> dict:
    """Parse SKILL.md content and extract metadata.

    Returns name, description, version, and tags from the SKILL.md file.
    Useful for auto-populating the skill registration form.
    Accepts optional auth parameters for parsing private repo SKILL.md files.
    """
    service = get_skill_service()
    try:
        result = await service.parse_skill_md(
            url,
            auth_scheme=auth_scheme,
            auth_credential=auth_credential,
            auth_header_name=auth_header_name,
        )
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
            "repository_url": result.get("repository_url"),
        }
    except SkillUrlValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse SKILL.md: {e.reason}"
        )


@router.get("/search", summary="Search skills")
async def search_skills(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    q: str = Query(
        ...,
        description="Lexical substring search across skill name, description, tags, and metadata",
    ),
    tags: str | None = Query(None, description="Comma-separated tags to filter by"),
    include_deprecated: bool = Query(False, description="Include deprecated skills in results"),
    include_draft: bool = Query(False, description="Include draft skills in results"),
) -> dict:
    """Search for skills by name, description, tags, and metadata.

    Uses lexical (substring) search with basic relevance scoring, not
    hybrid/semantic. For vector-based search, use POST /api/search/semantic instead.
    Deprecated and draft skills are excluded by default.
    """
    service = get_skill_service()
    skills = await service.list_skills_for_user(user_context)

    query_lower = q.lower()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    # Build set of excluded lifecycle statuses
    excluded_statuses: set[str] = set()
    if not include_deprecated:
        excluded_statuses.add("deprecated")
    if not include_draft:
        excluded_statuses.add("draft")

    matching_skills = []
    for skill in skills:
        # Filter by lifecycle status
        skill_status = getattr(skill, "status", "active") or "active"
        if skill_status in excluded_statuses:
            continue

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

        # Match in metadata (author, version, extra key-value pairs)
        skill_meta_dict: dict[str, Any] = {}
        if skill.metadata:
            if skill.metadata.author:
                skill_meta_dict["author"] = skill.metadata.author
            if skill.metadata.version:
                skill_meta_dict["version"] = skill.metadata.version
            if skill.metadata.extra:
                skill_meta_dict.update(skill.metadata.extra)
        metadata_text = flatten_metadata_to_text(skill_meta_dict)
        if metadata_text and query_lower in metadata_text.lower():
            score += 0.1

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
                    "status": skill_status,
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


@router.get("/{skill_path:path}/integrity", summary="Get content integrity status")
async def get_integrity_status(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Return the stored content integrity record for a skill.

    This is a read-only view of the baseline hashes and drift state
    that were computed at registration and updated on every content fetch.
    No external requests are made.
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {normalized_path}",
        )

    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    integrity = skill.content_integrity
    if not integrity:
        return {
            "path": normalized_path,
            "has_baseline": False,
            "message": "No integrity baseline. Re-register the skill to compute one.",
        }

    return {
        "path": normalized_path,
        "has_baseline": True,
        "composite_hash": integrity.composite_hash,
        "computed_at": integrity.computed_at.isoformat() if integrity.computed_at else None,
        "drift_detected": integrity.drift_detected,
        "last_drift_check": integrity.last_drift_check.isoformat() if integrity.last_drift_check else None,
        "drifted_files": integrity.drifted_files,
        "file_count": len(integrity.file_hashes),
        "files": [
            {"path": fh.path, "sha256": fh.sha256, "size_bytes": fh.size_bytes}
            for fh in integrity.file_hashes
        ],
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


MAX_RESOURCE_SIZE = 512 * 1024  # 512 KB


@router.get("/{skill_path:path}/content", summary="Get SKILL.md content")
async def get_skill_content(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
    resource: str | None = Query(
        None,
        description="Optional relative path to a companion resource file. "
        "When omitted, returns SKILL.md content.",
    ),
) -> dict:
    """Fetch skill content.

    Without ``resource``: returns SKILL.md markdown and the resource manifest.
    With ``resource``: returns the content of the specified companion file
    (validated against the stored manifest to prevent path traversal).
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {normalized_path}",
        )

    if not _user_can_access_skill(skill, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if (
        skill.content_integrity
        and skill.content_integrity.drift_detected
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Content drift detected for {normalized_path}. "
                f"Drifted files: {', '.join(skill.content_integrity.drifted_files)}. "
                "The skill has been disabled. Re-register to update the baseline."
            ),
        )

    # For federated skills with inline content, serve directly from DB
    if skill.skill_md_content:
        return {
            "content": skill.skill_md_content,
            "source": "inline",
            "path": normalized_path,
        }

    raw_url = skill.skill_md_raw_url or skill.skill_md_url
    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No SKILL.md URL configured for this skill",
        )

    try:
        if resource is not None:
            manifest = skill.resource_manifest
            if not manifest:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No resource manifest available for this skill",
                )
            all_resources = manifest.references + manifest.scripts + manifest.agents + manifest.assets
            matched = [r for r in all_resources if r.path == resource]
            if not matched:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Resource '{resource}' not found in manifest",
                )

            from ..utils.url_utils import derive_resource_url

            resource_url = derive_resource_url(str(raw_url), resource)
            response = await _fetch_authenticated_content(
                resource_url, skill, max_size=MAX_RESOURCE_SIZE,
            )

            drift_task = asyncio.create_task(
                _check_drift_inline(service, normalized_path, skill, resource, response.content)
            )
            drift_task.add_done_callback(_log_task_exception)

            return {
                "content": response.text,
                "path": resource,
                "type": matched[0].type,
                "url": resource_url,
            }

        response = await _fetch_authenticated_content(str(raw_url), skill)

        drift_task = asyncio.create_task(
            _check_drift_inline(service, normalized_path, skill, "SKILL.md", response.content)
        )
        drift_task.add_done_callback(_log_task_exception)

        result: dict[str, Any] = {
            "content": response.text,
            "url": str(raw_url),
        }
        if skill.resource_manifest:
            result["resource_manifest"] = skill.resource_manifest.model_dump()
        if skill.content_integrity:
            result["drift_detected"] = skill.content_integrity.drift_detected
        return result

    except SkillContentSSRFError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL failed SSRF validation: {e.url}",
        )
    except SkillContentTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )
    except SkillContentFetchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
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



@router.post(
    "/{skill_path:path}/refresh-resources",
    response_model=dict,
    summary="Refresh skill resource manifest",
)
async def refresh_skill_resources(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    skill_path: str = Path(..., description="Skill path or name"),
) -> dict:
    """Re-discover companion resource files and update the stored manifest.

    Useful when new files have been added to the skill's repository directory
    without re-registering the skill.
    """
    normalized_path = normalize_skill_path(skill_path)
    service = get_skill_service()
    skill = await service.get_skill(normalized_path)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {normalized_path}"
        )

    if not _user_can_modify_skill(skill, user_context):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    set_audit_action(
        http_request,
        "refresh_resources",
        "skill",
        resource_id=normalized_path,
        description=f"Refresh resource manifest for {normalized_path}",
    )

    raw_url = str(skill.skill_md_raw_url or skill.skill_md_url)
    auth_scheme, credential, auth_header_name = _decrypt_skill_auth(skill)

    manifest = await _discover_skill_resources(
        raw_url,
        auth_scheme=auth_scheme,
        auth_credential=credential,
        auth_header_name=auth_header_name,
    )

    updates = {"resource_manifest": manifest.model_dump() if manifest else None}
    await service.update_skill(normalized_path, updates)

    total = 0
    if manifest:
        total = len(manifest.references) + len(manifest.scripts) + len(manifest.agents) + len(manifest.assets)

    return {
        "path": normalized_path,
        "resources_discovered": total,
        "resource_manifest": manifest.model_dump() if manifest else None,
    }


@router.get("/{skill_path:path}", response_model=SkillCard, response_model_exclude=_SKILL_CARD_EXCLUDE, summary="Get a skill by path")
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
    response_model_exclude=_SKILL_CARD_EXCLUDE,
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

    # Registration gate check (admission control, issue #809)
    gate_result = await check_registration_gate(
        asset_type="skill",
        operation="register",
        source_api="/api/skills",
        registration_payload=request.model_dump(mode="json"),
        raw_headers=http_request.scope.get("headers", []),
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied skill '{request.name}': "
            f"{gate_result.error_message}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Registration denied by policy gate: {gate_result.error_message}",
        )

    service = get_skill_service()
    owner = user_context.get("username")

    try:
        skill = await service.register_skill(request=request, owner=owner, validate_url=True)
        logger.info(f"Registered skill: {skill.name} by {owner}")

        # Security scanning if enabled (non-blocking — mirrors server registration pattern)
        scan_task = asyncio.create_task(
            _perform_skill_security_scan_on_registration(skill, service)
        )
        scan_task.add_done_callback(_log_task_exception)

        asyncio.create_task(
            send_registration_webhook(
                event_type="registration",
                registration_type="skill",
                card_data=skill.model_dump(mode="json"),
                performed_by=owner,
            )
        )

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


@router.put("/{skill_path:path}", response_model=SkillCard, response_model_exclude=_SKILL_CARD_EXCLUDE, summary="Update a skill")
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

    # Registration gate check for update (admission control, issue #809)
    gate_result = await check_registration_gate(
        asset_type="skill",
        operation="update",
        source_api=f"/api/skills/{normalized_path}",
        registration_payload=request.model_dump(mode="json"),
        raw_headers=http_request.scope.get("headers", []),
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied skill update '{request.name}': "
            f"{gate_result.error_message}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Registration denied by policy gate: {gate_result.error_message}",
        )

    updates = request.model_dump(exclude_unset=True, mode="json")

    # Convert raw metadata dict to SkillMetadata structure for consistent storage
    if "metadata" in updates and updates["metadata"] is not None:
        raw_meta = updates["metadata"]
        updates["metadata"] = SkillMetadata(
            author=raw_meta.get("author"),
            version=raw_meta.get("version"),
            extra={k: v for k, v in raw_meta.items() if k not in ("author", "version")},
        ).model_dump(mode="json")

    # Encrypt credential if provided on update
    auth_credential = updates.pop("auth_credential", None)
    auth_scheme = updates.get("auth_scheme", existing.auth_scheme)
    if auth_credential and auth_scheme not in ("none", "global_credentials"):
        from ..utils.credential_encryption import encrypt_credential

        updates["auth_credential_encrypted"] = encrypt_credential(auth_credential)
        updates["credential_updated_at"] = datetime.now(UTC).isoformat()
    elif auth_scheme in ("none", "global_credentials"):
        updates["auth_credential_encrypted"] = None
        updates["auth_header_name"] = None
        updates["credential_updated_at"] = None

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

    asyncio.create_task(
        send_registration_webhook(
            event_type="deletion",
            registration_type="skill",
            card_data=existing.model_dump(mode="json"),
            performed_by=user_context.get("username"),
        )
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


def _log_task_exception(task: asyncio.Task) -> None:
    """Done-callback that surfaces unhandled exceptions from background tasks.

    Without this, exceptions raised inside ``asyncio.create_task(...)``
    fire-and-forget calls are only visible when the garbage collector
    finalises the task, which is too late for production debugging.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background task failed: %s", exc, exc_info=exc)


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
    """Perform security scan on newly registered skill.

    Mirrors the MCP server registration scan pattern:
    - Builds auth headers from the skill's encrypted credential
    - Passes headers to the scanner for authenticated downloads
    - Adds security-pending tag if scan fails
    - Disables skill if configured and scan fails
    - All scan failures are non-fatal and logged but not raised.

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
        raw_url = str(skill.skill_md_raw_url or skill.skill_md_url)
        auth_scheme, credential, auth_header_name = _decrypt_skill_auth(skill)
        fetch_headers: dict[str, str] = {}
        if credential:
            raw_url, fetch_headers = _build_fetch_headers(
                raw_url, auth_scheme, credential, auth_header_name,
            )

        result = await skill_scanner_service.scan_skill(
            skill_path=skill.path,
            skill_md_url=raw_url,
            headers=fetch_headers or None,
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


