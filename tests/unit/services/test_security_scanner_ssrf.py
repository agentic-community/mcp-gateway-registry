"""Unit tests for SSRF re-validation in the security-scanner service.

The scanner runs an external subprocess (mcp-scanner) against a target URL.
That URL may be an operator-supplied ``mcp_endpoint`` override, which is a
different host than the ``proxy_pass_url`` validated earlier. Because the
external subprocess cannot use the registry's IP-pinned guarded client, the
FINAL resolved URL must be re-validated before the subprocess is spawned, and
the scan must fail closed (no subprocess) when it points at a
private/metadata/loopback target.
"""

import ast
import json
import logging
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.services import security_scanner


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


def test_scan_credential_is_never_placed_in_argv():
    """A scan credential must reach the child through the environment, not the command line.

    `mcp-scanner remote` only takes a credential as `--bearer-token` / `--header`, both of
    which land in the child's argv and are readable by anything that can see `ps` or
    /proc/<pid>/cmdline for the life of the scan. That was tolerable while the value was
    an operator's static scan token. It is not now that the resolver chain can hand this
    function a *delegated human* OAuth token (a borrowed discovery identity) or the
    gateway's own app-only token -- argv is a far lower bar than the vault those come
    from, needing no key and no decryption.

    `X-Authorization` is the only spelling asserted because it is the only one the
    registry produces: both `_build_scan_auth_headers` (resolved OAuth) and
    `_build_scan_headers_from_credentials` (static bearer) emit it.
    """
    service = _make_service()
    token = "delegated-user-token-must-not-appear"  # nosec B105 - test fixture
    completed = MagicMock(stdout="[]", stderr="", returncode=0)

    with patch(
        "registry.services.security_scanner.subprocess.run", return_value=completed
    ) as mock_run:
        service._run_mcp_scanner(
            server_url="https://public.example/mcp",
            analyzers="test",
            api_key=None,
            headers=json.dumps({"X-Authorization": f"Bearer {token}"}),
            timeout=5,
        )

    mock_run.assert_called_once()
    argv = mock_run.call_args.args[0]
    env = mock_run.call_args.kwargs["env"]

    assert not any(token in str(a) for a in argv), f"credential leaked into argv: {argv}"
    assert "--bearer-token" not in argv, "the flag must be re-attached inside the child"
    # It still has to actually get there, or the scan silently loses authentication and
    # the whole discovery chain becomes pointless for an authed server.
    assert env[security_scanner._SCANNER_BEARER_ENV] == token


def test_scan_without_a_credential_sets_no_bearer_env():
    """No credential resolved -> nothing in the environment for the shim to attach."""
    service = _make_service()
    completed = MagicMock(stdout="[]", stderr="", returncode=0)

    with patch(
        "registry.services.security_scanner.subprocess.run", return_value=completed
    ) as mock_run:
        service._run_mcp_scanner(
            server_url="https://public.example/mcp",
            analyzers="test",
            api_key=None,
            headers=None,
            timeout=5,
        )

    assert security_scanner._SCANNER_BEARER_ENV not in mock_run.call_args.kwargs["env"]


def test_shim_reattaches_the_credential_flag_inside_the_child():
    """The shim must move the env value back onto mcp-scanner's own flag.

    Asserting only that the real CLI is reachable is vacuous -- `--bearer-token` appears
    in its `--help` output whether or not the shim attached anything, so dropping the
    re-attachment would pass while silently degrading every authenticated scan to
    unauthenticated. This stubs mcpscanner.cli and captures the argv the shim actually
    hands over.
    """
    import subprocess as _sp
    import sys as _sys

    probe = (
        "import sys, types;"
        "m = types.ModuleType('mcpscanner.cli');"
        "m.cli_entry_point = lambda: (print(repr(sys.argv)), 0)[1];"
        "pkg = types.ModuleType('mcpscanner'); pkg.cli = m;"
        "sys.modules['mcpscanner'] = pkg; sys.modules['mcpscanner.cli'] = m;"
        + security_scanner._SCANNER_SHIM
    )
    env = dict(os.environ)
    env[security_scanner._SCANNER_BEARER_ENV] = "shim-probe-token"  # nosec B105
    result = _sp.run(
        [_sys.executable, "-c", probe, "remote", "--server-url", "https://x.example/mcp"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-400:]
    argv = ast.literal_eval(result.stdout.strip())
    assert argv[0] == "mcp-scanner"
    assert "--bearer-token" in argv, "the shim did not re-attach the credential flag"
    assert argv[argv.index("--bearer-token") + 1] == "shim-probe-token"
    # ...and it must be consumed from the environment, not left for a grandchild.
    assert "MCP_GATEWAY_SCAN_BEARER" not in result.stdout


def test_shim_reaches_the_real_scanner_entry_point():
    """Separately: the import path in the shim must actually exist."""
    import subprocess as _sp
    import sys as _sys

    result = _sp.run(
        [_sys.executable, "-c", security_scanner._SCANNER_SHIM, "remote", "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-400:]
    assert "--server-url" in result.stdout
