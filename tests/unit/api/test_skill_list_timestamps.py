"""Unit tests for the registry timestamps on GET /api/skills list items.

Server and agent list items carry ``updated_at``, so a client can tell when a
record changed without fetching each one. Skill list items are built as
``SkillInfo``, which did not carry the SkillCard's ``created_at`` and
``updated_at``: every list item came back without them, while
``GET /api/skills/{path}`` returned both. These tests cover both list paths,
the admin fast path and the filtered path every non-admin takes.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

from registry.schemas.skill_models import SkillCard, VisibilityEnum
from registry.services.skill_service import SkillService

logger = logging.getLogger(__name__)


CREATED = datetime(2026, 10, 16, 8, 10, 52, tzinfo=UTC)
UPDATED = datetime(2026, 10, 16, 9, 30, 0, tzinfo=UTC)


def _skill_card() -> SkillCard:
    return SkillCard(
        path="/skills/demo-skill",
        id="demo-skill",
        name="demo-skill",
        description="A demo skill.",
        skill_md_url="https://example.com/SKILL.md",
        visibility=VisibilityEnum.PUBLIC,
        created_at=CREATED,
        updated_at=UPDATED,
    )


def _service_with_repo(cards: list[SkillCard]) -> SkillService:
    service = SkillService()
    repo = AsyncMock()
    repo.list_filtered.return_value = cards
    service._repo = repo
    return service


def _client(service: Any, user_context: dict[str, Any]):
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context

    scanner = MagicMock()
    scanner.get_scan_summaries = AsyncMock(return_value={})

    with (
        patch("registry.api.skill_routes.get_skill_service", return_value=service),
        patch("registry.services.skill_scanner.skill_scanner_service", scanner),
        patch("registry.health.service.health_service", MagicMock()),
        patch("registry.core.nginx_service.nginx_service", MagicMock()),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

    app.dependency_overrides.clear()


def _get_skills(service: Any, user_context: dict[str, Any], query: str = "") -> dict:
    gen = _client(service, user_context)
    client = next(gen)
    try:
        response = client.get(f"/api/skills{query}")
    finally:
        gen.close()
    assert response.status_code == 200
    return response.json()


def _parsed(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSkillServiceListTimestamps:
    async def test_list_skills_carries_card_timestamps(self):
        service = _service_with_repo([_skill_card()])

        [info] = await service.list_skills()

        assert info.created_at == CREATED
        assert info.updated_at == UPDATED


@pytest.mark.unit
class TestSkillListEndpointTimestamps:
    def test_non_admin_list_items_carry_timestamps(self, mock_settings):
        """The filtered path (every non-admin) returns both timestamps."""
        service = _service_with_repo([_skill_card()])
        context = {
            "username": "reader",
            "is_admin": False,
            "groups": [],
            "scopes": [],
            "accessible_servers": [],
            "accessible_services": [],
            "accessible_agents": [],
            "accessible_skills": ["all"],
            "auth_method": "session",
        }

        data = _get_skills(service, context)

        [item] = data["skills"]
        assert _parsed(item["created_at"]) == CREATED
        assert _parsed(item["updated_at"]) == UPDATED

    def test_admin_fast_path_list_items_carry_timestamps(self, mock_settings):
        """The admin fast path builds its items separately; it returns them too."""
        service = MagicMock()
        service.get_skills_paginated = AsyncMock(return_value=([_skill_card()], 1))
        service.list_skills_for_user = AsyncMock(return_value=[])
        context = {
            "username": "admin",
            "is_admin": True,
            "groups": ["mcp-registry-admin"],
            "scopes": [],
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "auth_method": "session",
        }

        data = _get_skills(service, context, "?include_disabled=true")

        service.get_skills_paginated.assert_awaited()
        [item] = data["skills"]
        assert _parsed(item["created_at"]) == CREATED
        assert _parsed(item["updated_at"]) == UPDATED
