"""The content route records SKILL.md reachability as skill health (issue #1952).

Clients pick skills from search results; recording health from the fetch they
depend on keeps search in step with what get_skill_content returns.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from registry.api.skill_routes import get_skill_content
from registry.exceptions import SkillContentFetchError

SKILL_PATH: str = "/skills/pdf"
RAW_URL: str = "https://raw.githubusercontent.com/o/r/main/SKILL.md"


def _skill(
    health_status: str,
    skill_md_content: str | None = None,
) -> SimpleNamespace:
    resource = SimpleNamespace(path="references/guide.md", type="reference")
    manifest = SimpleNamespace(
        references=[resource],
        scripts=[],
        agents=[],
        assets=[],
        model_dump=lambda: {},
    )
    return SimpleNamespace(
        health_status=health_status,
        skill_md_content=skill_md_content,
        skill_md_url=RAW_URL,
        skill_md_raw_url=RAW_URL,
        content_integrity=None,
        resource_manifest=manifest,
    )


async def _fetch(
    skill: SimpleNamespace,
    fetch: AsyncMock,
    expected_status: int,
    resource: str | None = None,
    allowed: bool = True,
) -> AsyncMock:
    """Call the content route, check its status, and return the health recorder.

    Checking the status keeps a not-awaited assertion from passing because the
    route failed for some unrelated reason.
    """
    service = Mock()
    service.get_skill = AsyncMock(return_value=skill)
    service.record_skill_health = AsyncMock()
    status_code = 200
    with (
        patch("registry.api.skill_routes.get_skill_service", return_value=service),
        patch("registry.api.skill_routes._user_can_access_skill", return_value=allowed),
        patch("registry.api.skill_routes._fetch_authenticated_content", new=fetch),
        patch("registry.api.skill_routes._check_drift_inline", new=AsyncMock()),
    ):
        try:
            await get_skill_content(user_context={}, skill_path=SKILL_PATH, resource=resource)
        except HTTPException as exc:
            status_code = exc.status_code
        # Let the background health task run.
        await asyncio.sleep(0)
    assert status_code == expected_status
    return service.record_skill_health


def _failing_fetch(
    upstream_status: int | None = 404,
) -> AsyncMock:
    return AsyncMock(
        side_effect=SkillContentFetchError(
            RAW_URL, f"HTTP {upstream_status}", upstream_status=upstream_status
        )
    )


def _working_fetch() -> AsyncMock:
    return AsyncMock(return_value=SimpleNamespace(text="# PDF\n", content=b"# PDF\n"))


@pytest.mark.asyncio
async def test_skill_md_fetch_failure_records_unhealthy():
    recorder = await _fetch(_skill("healthy"), _failing_fetch(), expected_status=502)

    recorder.assert_awaited_once_with(SKILL_PATH, healthy=False)


@pytest.mark.asyncio
async def test_resource_fetch_failure_leaves_health_alone():
    """A missing companion file says nothing about SKILL.md itself."""
    recorder = await _fetch(
        _skill("healthy"),
        _failing_fetch(),
        expected_status=502,
        resource="references/guide.md",
    )

    recorder.assert_not_awaited()


@pytest.mark.parametrize("upstream_status", [403, 429, 503, None])
@pytest.mark.asyncio
async def test_transient_fetch_failure_leaves_health_alone(upstream_status):
    """Rate limits, server errors and timeouts must not flip health.

    A caller can provoke these by hammering the route; only 404/410 counts.
    """
    recorder = await _fetch(_skill("healthy"), _failing_fetch(upstream_status), expected_status=502)

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_denied_caller_cannot_change_health():
    recorder = await _fetch(_skill("healthy"), _failing_fetch(), expected_status=403, allowed=False)

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_fetch_clears_unhealthy():
    recorder = await _fetch(_skill("unhealthy"), _working_fetch(), expected_status=200)

    recorder.assert_awaited_once_with(SKILL_PATH, healthy=True)


@pytest.mark.asyncio
async def test_successful_fetch_of_healthy_skill_writes_nothing():
    """The hot read path must not write on every call."""
    recorder = await _fetch(_skill("healthy"), _working_fetch(), expected_status=200)

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_gone_skill_md_writes_nothing():
    """Retrying a skill already recorded unhealthy must not write on every call."""
    recorder = await _fetch(_skill("unhealthy"), _failing_fetch(404), expected_status=502)

    recorder.assert_not_awaited()


@pytest.mark.parametrize("health_status", ["unknown", "unhealthy"])
@pytest.mark.asyncio
async def test_serving_inline_content_marks_skill_healthy(health_status):
    """AgentCore skills never reach the URL fetch, so serving them is the success."""
    recorder = await _fetch(
        _skill(health_status, skill_md_content="# PDF\n"),
        _working_fetch(),
        expected_status=200,
    )

    recorder.assert_awaited_once_with(SKILL_PATH, healthy=True)


@pytest.mark.asyncio
async def test_serving_inline_content_of_healthy_skill_writes_nothing():
    recorder = await _fetch(
        _skill("healthy", skill_md_content="# PDF\n"),
        _working_fetch(),
        expected_status=200,
    )

    recorder.assert_not_awaited()
