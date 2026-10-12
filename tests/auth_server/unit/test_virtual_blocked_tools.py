"""Unit tests for virtual-server tools/call enforcement at /validate.

A virtual server exposes aggregated tools under a virtual alias, while both the
scanner block state and the enable/quarantine state live on the OWNING backend
server. These tests lock in that the tools/call gate resolves the alias to that
backend and enforces, at request time:
  * a scanner-blocked tool (keyed on the ORIGINAL name) is denied;
  * a disabled or security-quarantined backend is denied;
and that an indeterminate lookup fails closed (denies).
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from auth_server.server import (  # noqa: E402
    _tool_call_denied,
    validate_server_tool_access,
)
from registry.schemas.virtual_server_models import (  # noqa: E402
    ToolMapping,
    VirtualServerConfig,
)

pytestmark = pytest.mark.unit


def _vs_config(tool_mappings):
    return VirtualServerConfig(
        path="/virtual/dev",
        server_name="Dev",
        tool_mappings=tool_mappings,
        is_enabled=True,
    )


_DEFAULT_BACKEND = object()


def _patch_repos(config, blocked, backend=_DEFAULT_BACKEND):
    """Patch the virtual-server and server repositories used by the gate.

    ``backend`` is the backend server document returned by ``get`` (defaults to
    an enabled, healthy backend; pass ``None`` to simulate a deleted backend).
    Returns the two patch context managers plus the server-repo mock so a test
    can assert which backend path was consulted.
    """
    if backend is _DEFAULT_BACKEND:
        backend = {"is_enabled": True}
    vs_repo = AsyncMock()
    vs_repo.get.return_value = config
    srv_repo = AsyncMock()
    srv_repo.get.return_value = backend
    srv_repo.get_blocked_tools.return_value = set(blocked)
    return (
        patch(
            "registry.repositories.factory.get_virtual_server_repository",
            return_value=vs_repo,
        ),
        patch("auth_server.server.get_server_repository", return_value=srv_repo),
        srv_repo,
    )


class TestToolCallDeniedVirtual:
    async def test_alias_resolves_to_backend_and_blocks(self):
        """A tool blocked on the backend is denied when called by its alias."""
        config = _vs_config(
            [
                ToolMapping(
                    tool_name="search",
                    alias="gh-search",
                    backend_server_path="/github",
                ),
            ]
        )
        vs_patch, srv_patch, srv_repo = _patch_repos(config, {"search"})
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "gh-search")

        assert reason == "blocked tool"
        # The block set must be read from the OWNING backend, not the virtual path.
        srv_repo.get_blocked_tools.assert_awaited_once_with("/github")

    async def test_backend_block_keyed_on_original_name(self):
        """Backend blocks are keyed on the ORIGINAL tool name, not the alias.

        Blocking the alias on the backend must NOT deny (the backend never sees
        the alias); only a block on the original name bites.
        """
        config = _vs_config(
            [
                ToolMapping(
                    tool_name="search",
                    alias="gh-search",
                    backend_server_path="/github",
                ),
            ]
        )
        vs_patch, srv_patch, _ = _patch_repos(config, {"gh-search"})
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "gh-search")

        assert reason is None

    async def test_unblocked_tool_allowed(self):
        """A tool that is not blocked on an enabled backend is allowed through."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, _ = _patch_repos(config, set())
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "search")

        assert reason is None

    async def test_disabled_backend_denied(self):
        """A disabled backend is denied at request time (before nginx catches up)."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, srv_repo = _patch_repos(config, set(), backend={"is_enabled": False})
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "search")

        assert reason == "disabled or quarantined backend"
        # Denied on state alone -- the block set is never consulted.
        srv_repo.get_blocked_tools.assert_not_awaited()

    async def test_security_quarantined_backend_denied(self):
        """A security-quarantined backend is denied at request time."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, _ = _patch_repos(
            config, set(), backend={"is_enabled": True, "is_disabled_for_security": True}
        )
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "search")

        assert reason == "disabled or quarantined backend"

    async def test_missing_backend_denied(self):
        """A deleted / unknown backend document is treated as unreachable."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, _ = _patch_repos(config, set(), backend=None)
        with vs_patch, srv_patch:
            reason = await _tool_call_denied("virtual/dev", "search")

        assert reason == "disabled or quarantined backend"

    async def test_unknown_virtual_server_fails_closed(self):
        """An unknown virtual server raises so the caller denies (fail closed)."""
        vs_patch, srv_patch, _ = _patch_repos(None, set())
        with vs_patch, srv_patch, pytest.raises(LookupError):
            await _tool_call_denied("virtual/dev", "search")

    async def test_unmapped_tool_fails_closed(self):
        """A tool not in the virtual mapping raises so the caller denies."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, _ = _patch_repos(config, {"search"})
        with vs_patch, srv_patch, pytest.raises(LookupError):
            await _tool_call_denied("virtual/dev", "does-not-exist")

    async def test_duplicate_exposed_name_denied_if_any_backend_blocks(self):
        """If two mappings share the exposed name, deny when ANY resolved backend
        blocks the tool -- the data plane's last-wins routing must not diverge
        from a first-match-only authorization check."""
        config = _vs_config(
            [
                ToolMapping(tool_name="a", alias="foo", backend_server_path="/clean"),
                ToolMapping(tool_name="b", alias="foo", backend_server_path="/blocked"),
            ]
        )
        vs_repo = AsyncMock()
        vs_repo.get.return_value = config
        srv_repo = AsyncMock()
        srv_repo.get.return_value = {"is_enabled": True}

        async def _blocked(path):
            return {"b"} if path == "/blocked" else set()

        srv_repo.get_blocked_tools.side_effect = _blocked
        with (
            patch(
                "registry.repositories.factory.get_virtual_server_repository",
                return_value=vs_repo,
            ),
            patch("auth_server.server.get_server_repository", return_value=srv_repo),
        ):
            reason = await _tool_call_denied("virtual/dev", "foo")

        assert reason == "blocked tool"

    async def test_duplicate_exposed_name_denied_if_any_backend_disabled(self):
        """Same guard for a disabled/quarantined duplicate backend."""
        config = _vs_config(
            [
                ToolMapping(tool_name="a", alias="foo", backend_server_path="/clean"),
                ToolMapping(tool_name="b", alias="foo", backend_server_path="/down"),
            ]
        )
        vs_repo = AsyncMock()
        vs_repo.get.return_value = config
        srv_repo = AsyncMock()

        async def _get(path):
            return {"is_enabled": path != "/down"}

        srv_repo.get.side_effect = _get
        srv_repo.get_blocked_tools.return_value = set()
        with (
            patch(
                "registry.repositories.factory.get_virtual_server_repository",
                return_value=vs_repo,
            ),
            patch("auth_server.server.get_server_repository", return_value=srv_repo),
        ):
            reason = await _tool_call_denied("virtual/dev", "foo")

        assert reason == "disabled or quarantined backend"


class TestToolCallDeniedDirect:
    async def test_non_virtual_uses_server_block_set(self):
        """A normal server reads its own block set with the tool name unchanged."""
        with patch(
            "auth_server.server._get_blocked_tools",
            new=AsyncMock(return_value={"search"}),
        ):
            assert await _tool_call_denied("github", "search") == "blocked tool"
            assert await _tool_call_denied("github", "issues") is None


class TestValidateServerToolAccessVirtual:
    async def test_blocked_backend_tool_denied_via_virtual_server(self):
        """End to end: a backend-blocked tool is denied through the virtual server
        before scope evaluation, so it can never reach the backend."""
        config = _vs_config(
            [
                ToolMapping(
                    tool_name="search",
                    alias="gh-search",
                    backend_server_path="/github",
                ),
            ]
        )
        vs_patch, srv_patch, _ = _patch_repos(config, {"search"})
        with vs_patch, srv_patch:
            allowed = await validate_server_tool_access(
                "virtual/dev", "tools/call", "gh-search", ["some-scope"]
            )

        assert allowed is False

    async def test_quarantined_backend_denied_via_virtual_server(self):
        """A quarantined backend is unreachable through the virtual server."""
        config = _vs_config([ToolMapping(tool_name="search", backend_server_path="/github")])
        vs_patch, srv_patch, _ = _patch_repos(
            config, set(), backend={"is_enabled": True, "is_disabled_for_security": True}
        )
        with vs_patch, srv_patch:
            allowed = await validate_server_tool_access(
                "virtual/dev", "tools/call", "search", ["some-scope"]
            )

        assert allowed is False

    async def test_unknown_virtual_server_denied(self):
        """A tools/call for an unknown virtual server fails closed (denied)."""
        vs_patch, srv_patch, _ = _patch_repos(None, set())
        with vs_patch, srv_patch:
            allowed = await validate_server_tool_access(
                "virtual/dev", "tools/call", "gh-search", ["some-scope"]
            )

        assert allowed is False
