"""Unit tests for the admin-conferring ui_permissions guard in scope_repository.

Covers _grants_admin (the content check) and the allow_privileged gate on
DocumentDBScopeRepository.import_group, which is the defense-in-depth layer of
the /api/servers/* privilege-escalation fix (see
.scratchpad/cve/design-privesc-fix-servers-api.md, section 3.4).

_grants_admin must agree with _user_is_admin in registry/auth/dependencies.py:
both decide admin status from the same mutating-action-with-all/* rule.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.auth.dependencies import _user_is_admin
from registry.repositories.documentdb.scope_repository import (
    _PRIVILEGED_SCOPE_NAMES,
    DocumentDBScopeRepository,
    _grants_admin,
    _is_privileged_write,
)


class TestGrantsAdmin:
    """Direct tests for the _grants_admin content check."""

    def test_register_service_all_grants_admin(self):
        assert _grants_admin({"register_service": ["all"]}) is True

    def test_register_service_star_does_not_grant_admin(self):
        # "*" grants all-server access WITHOUT admin (issue #663); it must not
        # be treated as admin-conferring, or this guard would block a
        # legitimate non-admin permission.
        assert _grants_admin({"register_service": ["*"]}) is False

    def test_other_mutating_actions_grant_admin(self):
        assert _grants_admin({"modify_service": ["all"]}) is True
        assert _grants_admin({"toggle_service": ["all"]}) is True
        assert _grants_admin({"delete_service": ["all"]}) is True

    def test_read_only_action_with_all_does_not_grant_admin(self):
        assert _grants_admin({"list_service": ["all"]}) is False

    def test_scoped_mutating_action_does_not_grant_admin(self):
        # Scoped to a specific server, not "all"/"*".
        assert _grants_admin({"register_service": ["currenttime"]}) is False

    def test_empty_and_none(self):
        assert _grants_admin({}) is False
        assert _grants_admin(None) is False

    def test_agrees_with_user_is_admin(self):
        """The repo guard and the live admin check must agree.

        If they drift, an attacker could write permissions the guard allows but
        that still promote to admin (or vice-versa).
        """
        cases = [
            {"register_service": ["all"]},
            {"register_service": ["*"]},
            {"modify_service": ["all"]},
            {"list_service": ["all"]},
            {"register_service": ["currenttime"]},
            {},
        ]
        for ui_permissions in cases:
            assert _grants_admin(ui_permissions) == _user_is_admin(ui_permissions), (
                f"disagreement on {ui_permissions}"
            )


def _make_repo() -> DocumentDBScopeRepository:
    """Build a repo whose _get_collection returns a mock collection."""
    repo = DocumentDBScopeRepository()
    mock_collection = MagicMock()
    mock_collection.replace_one = AsyncMock(
        return_value=MagicMock(upserted_id="x", matched_count=1)
    )
    repo._get_collection = AsyncMock(return_value=mock_collection)
    repo._scopes_cache = {}
    return repo, mock_collection


class TestImportGroupPrivilegedGuard:
    """allow_privileged gate on DocumentDBScopeRepository.import_group."""

    @pytest.mark.asyncio
    async def test_blocks_admin_permissions_by_default(self):
        repo, collection = _make_repo()

        result = await repo.import_group(
            group_name="attacker-group",
            ui_permissions={"register_service": ["all"]},
        )

        assert result is False
        collection.replace_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_admin_permissions_when_flagged(self):
        repo, collection = _make_repo()

        result = await repo.import_group(
            group_name="legit-admin-group",
            ui_permissions={"register_service": ["all"]},
            allow_privileged=True,
        )

        assert result is True
        collection.replace_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_non_privileged_permissions_by_default(self):
        repo, collection = _make_repo()

        result = await repo.import_group(
            group_name="normal-group",
            ui_permissions={"list_service": ["all"]},
        )

        assert result is True
        collection.replace_one.assert_called_once()


class TestIsPrivilegedWrite:
    """The repo guard catches all three admin-conferring vectors, not just
    ui_permissions -- so the deepest layer is also the most complete."""

    def test_privileged_group_name(self):
        assert _is_privileged_write("mcp-registry-admin", None, None) is True

    def test_privileged_group_mapping_is_the_original_privesc_vector(self):
        # Mapping a benign group to mcp-registry-admin -- _grants_admin alone
        # (ui_permissions only) would MISS this; _is_privileged_write catches it.
        assert _is_privileged_write("eng", ["mcp-registry-admin"], None) is True
        assert _grants_admin(None) is False  # the gap _is_privileged_write closes

    def test_admin_conferring_ui_permissions(self):
        assert _is_privileged_write("eng", ["eng"], {"register_service": ["all"]}) is True

    def test_benign_write_not_flagged(self):
        assert (
            _is_privileged_write("eng", ["eng"], {"list_service": ["currenttime"]}) is False
        )


class TestImportGroupMappingGuard:
    """The allow_privileged gate also covers group_mappings, not just ui_perms."""

    @pytest.mark.asyncio
    async def test_blocks_privileged_group_mapping_by_default(self):
        repo, collection = _make_repo()

        result = await repo.import_group(
            group_name="attacker-group",
            group_mappings=["attacker-group", "mcp-registry-admin"],
        )

        assert result is False
        collection.replace_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_privileged_group_mapping_when_flagged(self):
        repo, collection = _make_repo()

        result = await repo.import_group(
            group_name="legit",
            group_mappings=["legit", "mcp-registry-admin"],
            allow_privileged=True,
        )

        assert result is True
        collection.replace_one.assert_called_once()


def test_repo_privileged_names_match_service_layer():
    """The repo and service privileged-name sets must be the one leaf constant.

    These were previously duplicated across layers and kept aligned only by
    comments; they are now both re-exports of the single source of truth in
    registry.auth.privileged_constants. Asserting identity (``is``) pins that
    structural guarantee: if anyone re-introduces a local copy at either layer,
    the ``is`` check fails even when the values happen to be equal.
    """
    from registry.auth.privileged_constants import PRIVILEGED_SCOPE_NAMES as leaf
    from registry.services.scope_service import PRIVILEGED_SCOPE_NAMES

    assert _PRIVILEGED_SCOPE_NAMES is leaf
    assert PRIVILEGED_SCOPE_NAMES is leaf
