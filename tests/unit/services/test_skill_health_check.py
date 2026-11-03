"""Skill health checks: what the probe reports and what the service stores (issue #1952).

AgentCore federation stores SKILL.md inline and points skill_md_url at the
registry's own content route, which answers an unauthenticated HEAD with 405.
Probing that URL would mark a working skill unhealthy. Any caller with view
access can run the check, so only a conclusive result may be stored.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from registry.services.skill_service import (
    SkillService,
    _check_skill_health,
)

INLINE_SKILL_PATH: str = "/skills/agentcore-demo"
URL_SKILL_PATH: str = "/skills/pdf"
SKILL_MD_URL: str = "https://skills.example.com/pdf/SKILL.md"

Handler = Callable[[httpx.Request], httpx.Response]


def _status_handler(
    head_status: int,
    get_status: int,
    methods: list[str] | None = None,
) -> Handler:
    """Answer HEAD and GET with fixed statuses, recording each method seen."""

    def handler(request: httpx.Request) -> httpx.Response:
        if methods is not None:
            methods.append(request.method)
        return httpx.Response(head_status if request.method == "HEAD" else get_status)

    return handler


@contextmanager
def _skill_md_server(
    handler: Handler,
    unsafe_hop: str | None = None,
    url_is_safe: bool = True,
) -> Iterator[None]:
    """Route the health probe to an in-process HTTP handler.

    The real redirect check runs unless ``unsafe_hop`` forces a failure.
    """
    auth = Mock()
    auth.get_auth_headers = AsyncMock(return_value={})
    with ExitStack() as stack:
        stack.enter_context(
            patch("registry.services.skill_service._is_safe_url", return_value=url_is_safe)
        )
        stack.enter_context(patch("registry.services.skill_service._github_auth", auth))
        stack.enter_context(
            patch(
                "registry.services.skill_service.guarded_async_client",
                side_effect=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
        )
        if unsafe_hop is not None:
            stack.enter_context(
                patch(
                    "registry.services.skill_service._unsafe_redirect_target",
                    return_value=unsafe_hop,
                )
            )
        yield


async def _probe(
    head_status: int,
    get_status: int,
    methods: list[str] | None = None,
    unsafe_hop: str | None = None,
) -> dict:
    with _skill_md_server(_status_handler(head_status, get_status, methods), unsafe_hop):
        return await _check_skill_health(SKILL_MD_URL)


def _service_for(
    skill: SimpleNamespace,
) -> tuple[SkillService, Mock]:
    repo = Mock()
    repo.get = AsyncMock(return_value=skill)
    repo.update = AsyncMock()
    service = SkillService()
    service._repo = repo
    return service, repo


def _url_skill() -> SimpleNamespace:
    return SimpleNamespace(
        skill_md_content=None,
        skill_md_url=SKILL_MD_URL,
        skill_md_raw_url=None,
        auth_scheme="none",
        auth_credential_encrypted=None,
        auth_header_name=None,
    )


def _stored_status(
    repo: Mock,
) -> str | None:
    """The health_status the service wrote, or None when it wrote nothing."""
    if not repo.update.await_count:
        return None
    return repo.update.await_args.args[1]["health_status"]


@pytest.mark.asyncio
async def test_inline_skill_is_healthy_without_probing_its_url():
    skill = SimpleNamespace(
        skill_md_content="# Demo\n",
        skill_md_url="https://registry.example.com/api/skills/agentcore-demo/content",
        skill_md_raw_url=None,
    )
    service, repo = _service_for(skill)

    with patch("registry.services.skill_service._check_skill_health", new=AsyncMock()) as probe:
        result = await service.check_skill_health(INLINE_SKILL_PATH)

    probe.assert_not_awaited()
    assert (result["healthy"], _stored_status(repo)) == (True, "healthy")


@pytest.mark.parametrize(
    ("upstream_status", "expected_stored"),
    [
        (200, "healthy"),
        (404, "unhealthy"),
        (410, "unhealthy"),
        (403, None),
        (429, None),
        (503, None),
    ],
)
@pytest.mark.asyncio
async def test_health_check_stores_only_conclusive_results(upstream_status, expected_stored):
    """A rate limit or server error is reported to the caller but not stored."""
    service, repo = _service_for(_url_skill())

    with _skill_md_server(_status_handler(upstream_status, upstream_status)):
        result = await service.check_skill_health(URL_SKILL_PATH)

    assert (result["status_code"], _stored_status(repo)) == (upstream_status, expected_stored)


@pytest.mark.asyncio
async def test_health_check_timeout_is_reported_but_not_stored():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    service, repo = _service_for(_url_skill())

    with _skill_md_server(timeout):
        result = await service.check_skill_health(URL_SKILL_PATH)

    assert (result["healthy"], _stored_status(repo)) == (False, None)


@pytest.mark.asyncio
async def test_health_check_stores_a_url_blocked_by_the_ssrf_guard():
    service, repo = _service_for(_url_skill())

    with _skill_md_server(_status_handler(200, 200), url_is_safe=False):
        result = await service.check_skill_health(URL_SKILL_PATH)

    assert (result["healthy"], _stored_status(repo)) == (False, "unhealthy")


@pytest.mark.parametrize("head_status", [405, 501])
@pytest.mark.asyncio
async def test_unsupported_head_falls_back_to_get(head_status):
    """A server that rejects HEAD on a GET-only route must not read as unhealthy."""
    result = await _probe(head_status=head_status, get_status=200)

    assert (result["healthy"], result["status_code"]) == (True, 200)


@pytest.mark.parametrize("head_status", [200, 404])
@pytest.mark.asyncio
async def test_supported_head_decides_without_get(head_status):
    """The GET answers differently here, so any fallback would change the result."""
    result = await _probe(head_status=head_status, get_status=500)

    assert result["status_code"] == head_status


@pytest.mark.asyncio
async def test_get_fallback_still_reports_missing_skill_md():
    result = await _probe(head_status=405, get_status=404)

    assert (result["healthy"], result["status_code"]) == (False, 404)


@pytest.mark.asyncio
async def test_unsafe_head_redirect_never_issues_get():
    """A HEAD chain that fails the SSRF check must not be retried as a GET."""
    methods: list[str] = []

    result = await _probe(
        head_status=405,
        get_status=200,
        methods=methods,
        unsafe_hop="http://169.254.169.254/",
    )

    assert (result["healthy"], methods) == (False, ["HEAD"])
