"""Unit tests for ``metadata_fields`` projection on ``GET /api/skills``.

The list endpoint has two fetch paths: a DB-level fast path (admin, no field
filters) and a filtered fallback (every other caller, including an admin on a
default call). Projection must apply to the page whichever path produced it.
"""

import uuid
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

from registry.schemas.skill_models import (
    SkillInfo,
    SkillMetadata,
    VisibilityEnum,
)


def _make_skill_info(
    path: str,
) -> SkillInfo:
    """Build a public SkillInfo carrying author, version and extra metadata."""
    return SkillInfo(
        id=str(uuid.uuid4()),
        path=path,
        name=path.rsplit("/", 1)[-1],
        description="test skill",
        skill_md_url="https://example.com/SKILL.md",
        tags=["demo"],
        author="alice",
        version="1.0.0",
        metadata=SkillMetadata(
            author="alice",
            version="1.0.0",
            extra={"team": "data", "cost_center": "cc-1"},
        ),
        is_enabled=True,
        visibility=VisibilityEnum.PUBLIC,
        registry_name="local",
        owner="alice",
    )


def _admin_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "auth_method": "session",
    }


def _user_context() -> dict[str, Any]:
    return {
        "username": "bob",
        "is_admin": False,
        "groups": [],
        "scopes": [],
        "auth_method": "session",
    }


def _make_service(
    skills: list[SkillInfo],
) -> MagicMock:
    """Mock skill service returning fresh copies so tests never share state."""

    async def _list_for_user(
        user_context: dict[str, Any] | None,
        include_disabled: bool = False,
        tag: str | None = None,
    ) -> list[SkillInfo]:
        return [s.model_copy(deep=True) for s in skills]

    service = MagicMock()
    service.list_skills_for_user = AsyncMock(side_effect=_list_for_user)
    service.get_skills_paginated = AsyncMock(
        return_value=([s.model_copy(deep=True) for s in skills], len(skills))
    )
    return service


def _get_skills(
    service: MagicMock,
    user_context: dict[str, Any],
    url: str,
) -> list[dict[str, Any]]:
    """Call the list endpoint with the mock service and return the skills."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context

    scanner = MagicMock()
    scanner.get_scan_summaries = AsyncMock(return_value={})

    try:
        with (
            patch(
                "registry.api.skill_routes.get_skill_service",
                return_value=service,
            ),
            patch("registry.services.skill_scanner.skill_scanner_service", scanner),
            patch("registry.health.service.health_service", MagicMock()),
            patch("registry.core.nginx_service.nginx_service", MagicMock()),
        ):
            client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
            response = client.get(url)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    return response.json()["skills"]


@pytest.fixture
def sample_skills() -> list[SkillInfo]:
    return [
        _make_skill_info("/skills/one"),
        _make_skill_info("/skills/two"),
    ]


@pytest.mark.unit
class TestSkillListMetadataProjection:
    """``metadata_fields`` narrows metadata on both fetch paths."""

    @pytest.mark.parametrize(
        ("context_factory", "query"),
        [
            (_user_context, ""),
            (_admin_context, ""),
            (_admin_context, "&tag=demo"),
        ],
        ids=["non-admin", "admin-default", "admin-tag-filter"],
    )
    def test_fallback_path_projects_metadata(
        self,
        mock_settings,
        sample_skills,
        context_factory,
        query,
    ):
        """Callers routed through the filtered fallback get the projection."""
        service = _make_service(sample_skills)

        skills = _get_skills(
            service,
            context_factory(),
            f"/api/skills?metadata_fields=author{query}",
        )

        service.list_skills_for_user.assert_awaited()
        service.get_skills_paginated.assert_not_awaited()
        assert [s["metadata"] for s in skills] == [
            {"author": "alice", "version": None, "extra": {}},
            {"author": "alice", "version": None, "extra": {}},
        ]

    def test_fast_path_projects_metadata(
        self,
        mock_settings,
        sample_skills,
    ):
        """An admin on the DB-level fast path still gets the projection."""
        service = _make_service(sample_skills)

        skills = _get_skills(
            service,
            _admin_context(),
            "/api/skills?include_disabled=true&metadata_fields=author",
        )

        service.get_skills_paginated.assert_awaited()
        assert [s["metadata"] for s in skills] == [
            {"author": "alice", "version": None, "extra": {}},
            {"author": "alice", "version": None, "extra": {}},
        ]

    def test_nested_path_keeps_only_requested_extra_key(
        self,
        mock_settings,
        sample_skills,
    ):
        """A dot-notation path keeps one nested key and drops its siblings."""
        service = _make_service(sample_skills)

        skills = _get_skills(
            service,
            _user_context(),
            "/api/skills?metadata_fields=extra.team",
        )

        assert skills[0]["metadata"] == {
            "author": None,
            "version": None,
            "extra": {"team": "data"},
        }

    def test_without_metadata_fields_returns_full_metadata(
        self,
        mock_settings,
        sample_skills,
    ):
        """Omitting ``metadata_fields`` leaves metadata untouched."""
        service = _make_service(sample_skills)

        skills = _get_skills(service, _user_context(), "/api/skills")

        assert skills[0]["metadata"] == {
            "author": "alice",
            "version": "1.0.0",
            "extra": {"team": "data", "cost_center": "cc-1"},
        }
