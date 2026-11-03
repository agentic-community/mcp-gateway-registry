"""Semantic search reports skill health from the skills collection (issue #1952).

A health check writes only the skills collection, so the search route must read
health from there rather than from the search index.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request

from registry.api.search_routes import (
    SemanticSearchRequest,
    semantic_search,
)

SKILL_PATH: str = "/skills/pdf"
CHECKED_AT: str = "2026-10-01T00:00:00+00:00"


def _admin_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "accessible_servers": ["*"],
        "accessible_agents": ["all"],
    }


def _search_repo_with_skill() -> AsyncMock:
    # A search document written before issue #1952 still carries health fields.
    # The record must win, so the hit holds a stale value that differs from it.
    repo = AsyncMock()
    repo.search = AsyncMock(
        return_value={
            "skills": [
                {
                    "path": SKILL_PATH,
                    "skill_name": "pdf",
                    "visibility": "public",
                    "is_enabled": True,
                    "relevance_score": 0.9,
                    "health_status": "healthy",
                    "last_checked_time": "2020-01-01T00:00:00+00:00",
                }
            ]
        }
    )
    return repo


async def _search(
    skill_records: dict[str, dict[str, Any]],
    metadata_fields: str | None = None,
) -> Any:
    skill_repo = Mock()
    skill_repo.list_by_paths = AsyncMock(return_value=skill_records)
    http_request = Mock(spec=Request)
    http_request.state = Mock()
    with patch("registry.api.search_routes.get_skill_repository", return_value=skill_repo):
        response = await semantic_search(
            http_request,
            SemanticSearchRequest(query="pdf", metadata_fields=metadata_fields),
            _admin_context(),
            _search_repo_with_skill(),
        )
    return response


@pytest.mark.asyncio
async def test_search_returns_health_recorded_in_skills_collection():
    response = await _search(
        {SKILL_PATH: {"health_status": "unhealthy", "last_checked_time": CHECKED_AT}}
    )

    skill = response.skills[0]
    assert (skill.health_status, skill.last_checked_time) == ("unhealthy", CHECKED_AT)


@pytest.mark.asyncio
async def test_search_reports_unknown_when_skill_record_is_missing():
    response = await _search({})

    skill = response.skills[0]
    assert (skill.health_status, skill.last_checked_time) == ("unknown", None)


@pytest.mark.asyncio
async def test_search_reports_unknown_for_an_unrecognised_stored_value():
    response = await _search({SKILL_PATH: {"health_status": "checking"}})

    assert response.skills[0].health_status == "unknown"


@pytest.mark.asyncio
async def test_search_projects_metadata_alongside_health():
    response = await _search(
        {
            SKILL_PATH: {
                "health_status": "healthy",
                "metadata": {"author": "someone", "extra": {"team": "docs"}},
            }
        },
        metadata_fields="extra.team",
    )

    skill = response.skills[0]
    assert (skill.health_status, skill.metadata) == ("healthy", {"extra": {"team": "docs"}})
