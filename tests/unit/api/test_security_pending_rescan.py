"""Issue #2032: a passing rescan clears the security-pending tag a failing scan set,
and a failed scan's message points at the likely cause.

The scanner does not record that it added the tag, so the rule comes from scan
history: clear on a passing rescan unless the scan before it also passed (then a
person added the tag after it).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.schemas.security import SecurityScanConfig, SecurityScanResult
from registry.services.security_pending import tags_after_passing_scan
from registry.services.security_scanner import (
    _scan_failure_message,
    scan_credential_expected,
)

pytestmark = [pytest.mark.unit]

SERVER_PATH = "/pending-server"
FAILED_SCAN: dict = {"is_safe": False, "scan_failed": True}
PASSED_SCAN: dict = {"is_safe": True, "scan_failed": False}


class TestTagsAfterPassingScan:
    """The pure rule in registry/services/security_pending.py."""

    def test_clears_tag_set_by_a_failing_scan(self):
        assert tags_after_passing_scan(["a", "security-pending", "b"], FAILED_SCAN) == ["a", "b"]

    def test_clears_tag_when_no_earlier_scan_was_stored(self):
        """A scan that raised stores no result but can still have tagged the entity."""
        assert tags_after_passing_scan(["security-pending"], None) == []

    def test_keeps_tag_added_after_a_passing_scan(self):
        """The previous scan passed, so a person added the tag; leave it."""
        assert tags_after_passing_scan(["security-pending"], PASSED_SCAN) is None

    def test_no_tag_means_no_change(self):
        assert tags_after_passing_scan(["a"], FAILED_SCAN) is None
        assert tags_after_passing_scan(None, FAILED_SCAN) is None


def _scan(
    is_safe: bool,
) -> SecurityScanResult:
    return SecurityScanResult(
        server_url="https://example.com/mcp",
        server_path=SERVER_PATH,
        scan_timestamp="2026-11-13T00:00:00Z",
        is_safe=is_safe,
        critical_issues=0,
        high_severity=0,
        raw_output={"analysis_results": {}, "tool_results": []},
        scan_failed=False,
    )


async def _rescan_server(
    scan_is_safe: bool,
    previous_scan: dict | None,
    tags: list[str],
) -> AsyncMock:
    """Run the real rescan_server route with mocked services; return server_service."""
    from registry.api.server_routes import rescan_server

    scanner = AsyncMock()
    scanner.scan_server = AsyncMock(return_value=_scan(scan_is_safe))
    scanner.get_scan_result = AsyncMock(return_value=previous_scan)
    scanner.get_scan_config = lambda: SecurityScanConfig(
        enabled=True,
        block_unsafe_servers=False,
        allow_unsafe_servers=False,
    )
    svc = AsyncMock()
    svc.get_server_info = AsyncMock(
        return_value={
            "server_name": "S",
            "path": SERVER_PATH,
            "proxy_pass_url": "http://x/",
            "tags": tags,
        }
    )
    svc.update_server = AsyncMock(return_value=True)
    with (
        patch("registry.api.server_routes.security_scanner_service", scanner),
        patch("registry.api.server_routes.server_service", svc),
        patch("registry.api.server_routes._build_scan_headers_from_credentials", lambda *_: None),
    ):
        await rescan_server(SERVER_PATH, user_context={"is_admin": True, "username": "a"})
    return svc


@pytest.mark.asyncio
class TestServerRescan:
    """The server rescan route applies the rule with a tags-only write."""

    async def test_passing_rescan_removes_scanner_tag(self):
        svc = await _rescan_server(True, FAILED_SCAN, ["mcp", "security-pending"])

        args, kwargs = svc.update_server.await_args
        assert args[0] == SERVER_PATH
        assert args[1]["tags"] == ["mcp"]
        assert kwargs["updated_fields"] == ["tags"]

    async def test_passing_rescan_keeps_tag_added_after_a_pass(self):
        svc = await _rescan_server(True, PASSED_SCAN, ["security-pending"])

        svc.update_server.assert_not_awaited()

    async def test_failing_rescan_never_touches_tags(self):
        svc = await _rescan_server(False, FAILED_SCAN, ["security-pending"])

        svc.update_server.assert_not_awaited()


@pytest.mark.asyncio
class TestAgentAndSkillRescan:
    """Agents and skills follow the same rule."""

    async def test_agent_tag_removed_after_failing_scan(self):
        from registry.api.agent_routes import _clear_agent_security_pending

        agent = MagicMock()
        agent.tags = ["security-pending", "a2a"]
        service = AsyncMock()
        service.get_agent_info = AsyncMock(return_value=agent)
        with patch("registry.api.agent_routes.agent_service", service):
            cleared = await _clear_agent_security_pending("/agent", FAILED_SCAN)

        assert cleared is True
        service.update_agent.assert_awaited_once_with("/agent", {"tags": ["a2a"]})

    async def test_agent_tag_kept_after_passing_scan(self):
        from registry.api.agent_routes import _clear_agent_security_pending

        agent = MagicMock()
        agent.tags = ["security-pending"]
        service = AsyncMock()
        service.get_agent_info = AsyncMock(return_value=agent)
        with patch("registry.api.agent_routes.agent_service", service):
            cleared = await _clear_agent_security_pending("/agent", PASSED_SCAN)

        assert cleared is False
        service.update_agent.assert_not_awaited()

    async def test_skill_rescan_removes_scanner_tag(self):
        from registry.api.skill_routes import rescan_skill

        skill = MagicMock()
        skill.path = "/test-skill"
        skill.tags = ["security-pending", "docs"]
        skill.skill_md_raw_url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        skill.auth_scheme = "none"
        safe = MagicMock()
        safe.is_safe = True
        safe.model_dump = MagicMock(return_value={"is_safe": True})
        scanner = MagicMock()
        scanner.scan_skill = AsyncMock(return_value=safe)
        scanner.get_scan_result = AsyncMock(return_value=FAILED_SCAN)
        service = AsyncMock()
        service.get_skill = AsyncMock(return_value=skill)
        with (
            patch("registry.api.skill_routes.get_skill_service", return_value=service),
            patch("registry.services.skill_scanner.skill_scanner_service", scanner),
            patch(
                "registry.api.skill_routes._build_skill_fetch_request",
                AsyncMock(return_value=("https://raw.githubusercontent.com/x", None)),
            ),
        ):
            await rescan_skill(
                http_request=MagicMock(),
                user_context={"is_admin": True, "username": "admin", "groups": []},
                skill_path="test-skill",
            )

        service.update_skill.assert_awaited_once()
        path, updates = service.update_skill.await_args.args
        assert path.endswith("test-skill")
        assert updates == {"tags": ["docs"]}


class TestScanFailureMessage:
    """The message shown in the scan modal points at the likely cause."""

    def test_no_credential_hint_for_server_without_credential(self):
        """A server configured with no credential is not told to check one."""
        message = _scan_failure_message(RuntimeError("x"), None, credential_expected=False)

        assert "without a credential" not in message

    def test_credential_hint_kept_when_one_is_configured(self):
        message = _scan_failure_message(RuntimeError("x"), None, credential_expected=True)

        assert "without a credential" in message

    def test_appended_mcp_path_is_named_without_the_url(self):
        """The guessed /mcp suffix is named; nothing from the URL is included."""
        message = _scan_failure_message(
            RuntimeError("x"),
            None,
            credential_expected=False,
            appended_mcp_path=True,
        )

        assert "added /mcp" in message
        assert "mcp_endpoint" in message
        assert "http" not in message

    def test_credential_expected_follows_the_configured_auth(self):
        assert scan_credential_expected({"auth_scheme": "bearer"}) is True
        assert scan_credential_expected({"oauth_discovery": {"enabled": True}}) is True
        assert scan_credential_expected({"egress_auth_mode": "obo_exchange"}) is True
        assert scan_credential_expected({"auth_scheme": "none"}) is False
        assert scan_credential_expected(None) is False


@pytest.mark.asyncio
async def test_scan_server_names_the_appended_mcp_path():
    """scan_server flags the /mcp guess when no mcp_endpoint is set and the scan fails."""
    from registry.services.security_scanner import SecurityScannerService

    service = SecurityScannerService.__new__(SecurityScannerService)
    service._scan_repo = AsyncMock()
    service.get_scan_config = lambda: SecurityScanConfig(enabled=True)
    with (
        patch("registry.utils.url_guard.validate_url", lambda *a, **k: None),
        patch(
            "registry.services.security_scanner.ensure_llm_analyzer_available",
            lambda *a, **k: None,
        ),
        patch.object(service, "_run_mcp_scanner", side_effect=RuntimeError("cancelled")),
    ):
        result = await service.scan_server(
            server_url="https://public.example/app-server",
            server_path="/x",
            credential_expected=False,
        )

    assert result.scan_failed is True
    assert "added /mcp" in result.error_message
    assert "without a credential" not in result.error_message
    assert "public.example" not in result.error_message
