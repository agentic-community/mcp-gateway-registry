"""Unit tests for the privileged-scope write guard in scope_service.import_group.

Defense-in-depth: even if a caller reaches the service layer without a
route-level admin check, import_group must reject a non-admin actor that tries
to create or map a privileged scope (e.g. mcp-registry-admin). This is the last
line of defense against self-assignment of admin via group import.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.services import scope_service
from registry.services.scope_service import (
    PrivilegedScopeWriteError,
    _import_touches_privileged_scope,
    import_group,
)


@pytest.mark.unit
class TestImportTouchesPrivilegedScope:
    """The detector flags privileged scope, mapping, or all-server grant."""

    def test_privileged_scope_name(self) -> None:
        assert _import_touches_privileged_scope("mcp-registry-admin", None, None) is True

    def test_privileged_group_mapping(self) -> None:
        assert _import_touches_privileged_scope("eng", ["mcp-registry-admin"], None) is True

    def test_all_server_grant_is_privileged(self) -> None:
        assert _import_touches_privileged_scope("eng", ["eng"], {"toggle_service": ["all"]}) is True

    def test_string_shaped_all_grant_is_privileged(self) -> None:
        """A bare-string "all" (not a list) must still be flagged.

        _user_is_admin treats {"register_service": "all"} as admin-conferring
        (`"all" in "all"` is True via substring match). The guard must agree, or
        a string-shaped grant slips past this last line of defense.
        """
        assert _import_touches_privileged_scope("eng", ["eng"], {"register_service": "all"}) is True

    def test_benign_scope_not_flagged(self) -> None:
        assert (
            _import_touches_privileged_scope(
                "eng", ["eng"], {"list_service": ["currenttime", "mcpgw"]}
            )
            is False
        )

    def test_benign_string_value_not_flagged(self) -> None:
        """A non-"all" string value must not be over-flagged."""
        assert (
            _import_touches_privileged_scope("eng", ["eng"], {"list_service": "currenttime"})
            is False
        )


@pytest.mark.unit
class TestImportGroupPrivilegedGuard:
    """import_group fails closed for non-admin privileged writes."""

    @pytest.mark.asyncio
    async def test_non_admin_privileged_mapping_rejected(self) -> None:
        """A non-admin mapping engineering -> mcp-registry-admin is rejected."""
        with pytest.raises(PrivilegedScopeWriteError):
            await import_group(
                scope_name="engineering",
                group_mappings=["engineering", "mcp-registry-admin"],
                allow_privileged=False,
            )

    @pytest.mark.asyncio
    async def test_non_admin_string_shaped_all_grant_rejected(self) -> None:
        """A non-admin write granting a mutating action "all" (string) is rejected.

        Regression for the guard's type-narrowness: a bare-string "all" (vs a
        ["all"] list) must still trip the privileged-scope guard, because
        _user_is_admin would treat it as admin-conferring.
        """
        with pytest.raises(PrivilegedScopeWriteError):
            await import_group(
                scope_name="engineering",
                group_mappings=["engineering"],
                ui_permissions={"register_service": "all"},
                allow_privileged=False,
            )

    @pytest.mark.asyncio
    async def test_admin_privileged_mapping_allowed(self) -> None:
        """An admin may write the same privileged mapping; repo is invoked."""
        mock_repo = AsyncMock()
        mock_repo.import_group = AsyncMock(return_value=True)
        with patch.object(scope_service, "get_scope_repository", return_value=mock_repo):
            result = await import_group(
                scope_name="engineering",
                group_mappings=["engineering", "mcp-registry-admin"],
                allow_privileged=True,
            )
        assert result is True
        mock_repo.import_group.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_admin_benign_write_allowed(self) -> None:
        """A non-admin benign group write is not blocked by the guard."""
        mock_repo = AsyncMock()
        mock_repo.import_group = AsyncMock(return_value=True)
        with patch.object(scope_service, "get_scope_repository", return_value=mock_repo):
            result = await import_group(
                scope_name="engineering",
                group_mappings=["engineering"],
                ui_permissions={"list_service": ["currenttime"]},
                allow_privileged=False,
            )
        assert result is True
        mock_repo.import_group.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_default_actor_is_not_admin(self) -> None:
        """allow_privileged defaults to False (fail-closed) for privileged writes."""
        with pytest.raises(PrivilegedScopeWriteError):
            await import_group(
                scope_name="mcp-registry-admin",
                group_mappings=["mcp-registry-admin"],
            )
