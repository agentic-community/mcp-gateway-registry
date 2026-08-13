"""Unit tests for SSRF re-validation in the security-scanner service.

The scanner runs an external subprocess (mcp-scanner) against a target URL.
That URL may be an operator-supplied ``mcp_endpoint`` override, which is a
different host than the ``proxy_pass_url`` validated earlier. Because the
external subprocess cannot use the registry's IP-pinned guarded client, the
FINAL resolved URL must be re-validated before the subprocess is spawned, and
the scan must fail closed (no subprocess) when it points at a
private/metadata/loopback target.
"""

import logging
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_service():
    """Build a SecurityScannerService with its repository dependency mocked."""
    with patch("registry.services.security_scanner.get_security_scan_repository"):
        from registry.services.security_scanner import SecurityScannerService

        service = SecurityScannerService()
        service._scan_repo = MagicMock()
        service._scan_repo.create = AsyncMock(return_value=None)
        return service


def _resolve_to(*ips: str):
    def _stub(host, port, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return _stub


@pytest.mark.asyncio
async def test_scan_blocked_when_mcp_endpoint_targets_private_host():
    """An mcp_endpoint override resolving to a private host does NOT run the scanner."""
    service = _make_service()

    def _resolve(host, port, **kw):
        # proxy_pass_url host resolves public; the mcp_endpoint override is private.
        if "internal.evil.example" in host:
            return [(2, 1, 6, "", ("10.0.0.5", port))]
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    with (
        patch("registry.services.security_scanner.subprocess.run") as mock_run,
        patch("registry.utils.url_guard.socket.getaddrinfo", side_effect=_resolve),
    ):
        # proxy_pass_url is public, but the mcp_endpoint override is private.
        # The block must be attributable to the mcp_endpoint re-check.
        result = await service.scan_server(
            server_url="https://public.example",
            server_path="/x",
            mcp_endpoint="https://internal.evil.example/mcp",
        )

        # Fail closed: subprocess never runs, scan is recorded as unsafe/failed.
        mock_run.assert_not_called()
        assert result.is_safe is False
        assert result.scan_failed is True


@pytest.mark.asyncio
async def test_scan_blocked_when_mcp_endpoint_is_metadata_literal():
    """A metadata-IP literal mcp_endpoint is refused with no subprocess."""
    service = _make_service()

    with (
        patch("registry.services.security_scanner.subprocess.run") as mock_run,
        patch(
            "registry.utils.url_guard.socket.getaddrinfo",
            side_effect=_resolve_to("93.184.216.34"),
        ),
    ):
        result = await service.scan_server(
            server_url="https://public.example",
            server_path="/x",
            mcp_endpoint="http://169.254.169.254/latest/meta-data/",
        )

        mock_run.assert_not_called()
        assert result.is_safe is False
        assert result.scan_failed is True


@pytest.mark.asyncio
async def test_scan_runs_for_valid_public_endpoint():
    """A valid public mcp_endpoint proceeds to run the scanner subprocess."""
    service = _make_service()

    completed = MagicMock()
    completed.stdout = "[]"
    completed.stderr = ""

    with (
        patch(
            "registry.services.security_scanner.subprocess.run", return_value=completed
        ) as mock_run,
        patch(
            "registry.utils.url_guard.socket.getaddrinfo",
            side_effect=_resolve_to("93.184.216.34"),
        ),
    ):
        result = await service.scan_server(
            server_url="https://public.example",
            server_path="/x",
            mcp_endpoint="https://good.example.com/mcp",
        )

        mock_run.assert_called_once()
        # The subprocess was invoked against the validated public endpoint.
        cmd = mock_run.call_args.args[0]
        assert "https://good.example.com/mcp" in cmd
        assert result.scan_failed is False


@pytest.mark.asyncio
async def test_scan_failure_omits_query_and_raw_exception_detail(caplog):
    service = _make_service()

    with (
        patch(
            "registry.utils.url_guard.socket.getaddrinfo",
            side_effect=_resolve_to("93.184.216.34"),
        ),
        patch.object(
            service,
            "_run_mcp_scanner",
            side_effect=RuntimeError("raw-exception-secret"),
        ),
        caplog.at_level(logging.INFO, logger="registry.services.security_scanner"),
    ):
        result = await service.scan_server(
            server_url="https://public.example/mcp?api_key=query-secret",
            server_path="/x",
        )

    assert "query-secret" not in caplog.text
    assert "raw-exception-secret" not in caplog.text
    assert "raw-exception-secret" not in str(result.model_dump())
    assert result.error_message == "security scan failed (RuntimeError)"


def test_scanner_does_not_log_stdout_body_or_query(caplog):
    service = _make_service()
    completed = MagicMock()
    completed.stdout = '[{"analyzer": "test", "secret": "raw-body-secret"}]'
    completed.stderr = ""

    with (
        patch("registry.services.security_scanner.subprocess.run", return_value=completed),
        caplog.at_level(logging.INFO, logger="registry.services.security_scanner"),
    ):
        output = service._run_mcp_scanner(
            server_url="https://public.example/mcp?api_key=query-secret",
            analyzers="test",
            api_key=None,
            headers=None,
            timeout=5,
        )

    assert output["tool_results"]
    assert "query-secret" not in caplog.text
    assert "raw-body-secret" not in caplog.text


@pytest.mark.asyncio
async def test_scan_builtin_server_uses_exact_identity_profile():
    """The bundled airegistry-tools server (internal mcpgw-server host) is scannable
    via its exact-identity built-in profile, even though PROXY_PROFILE deliberately
    does not allowlist the internal host. Regression test for the built-in-scan 500
    (issue surfaced by the release smoke test)."""
    service = _make_service()

    completed = MagicMock()
    completed.stdout = "[]"
    completed.stderr = ""

    with (
        patch(
            "registry.services.security_scanner.subprocess.run", return_value=completed
        ) as mock_run,
        patch(
            "registry.utils.url_guard.socket.getaddrinfo",
            side_effect=_resolve_to("172.18.0.9"),
        ),
    ):
        result = await service.scan_server(
            server_url="http://mcpgw-server:8003/",
            server_path="/airegistry-tools",
        )

        # The scan proceeds against the exact built-in endpoint identity.
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert "http://mcpgw-server:8003/mcp" in cmd
        assert result.scan_failed is False


@pytest.mark.asyncio
async def test_scan_blocked_for_non_builtin_internal_target():
    """The built-in exemption is exact-identity only, not a general internal bypass:
    a server at the same internal host but a DIFFERENT registered path gets
    PROXY_PROFILE and is refused (no subprocess)."""
    from registry.exceptions import UrlValidationError

    service = _make_service()

    with (
        patch("registry.services.security_scanner.subprocess.run") as mock_run,
        patch(
            "registry.utils.url_guard.socket.getaddrinfo",
            side_effect=_resolve_to("172.18.0.9"),
        ),
        pytest.raises(UrlValidationError),
    ):
        await service.scan_server(
            server_url="http://mcpgw-server:8003/",
            server_path="/not-the-builtin",
        )

    mock_run.assert_not_called()


def test_scanner_does_not_log_or_raise_stderr(caplog):
    service = _make_service()
    error = subprocess.CalledProcessError(
        returncode=2,
        cmd=["mcp-scanner"],
        stderr="stderr-secret",
    )

    with (
        patch("registry.services.security_scanner.subprocess.run", side_effect=error),
        caplog.at_level(logging.INFO, logger="registry.services.security_scanner"),
        pytest.raises(RuntimeError) as exc_info,
    ):
        service._run_mcp_scanner(
            server_url="https://public.example/mcp?api_key=query-secret",
            analyzers="test",
            api_key=None,
            headers=None,
            timeout=5,
        )

    assert str(exc_info.value) == "Security scanner command failed"
    assert "query-secret" not in caplog.text
    assert "stderr-secret" not in caplog.text
