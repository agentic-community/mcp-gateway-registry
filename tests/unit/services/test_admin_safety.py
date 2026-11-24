"""Unit tests for registry/services/admin_safety.py.

Covers the intra-admin safety helpers: self-target detection, admin-group
resolution, admin-user counting, and the last-admin / demotion guards. Each
guard must fail closed (deny) when the admin population cannot be determined.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.services import admin_safety
from registry.services.admin_safety import AdminSafetyError


class TestIsSelfTarget:
    """Tests for the self-deletion detection helper."""

    def test_matches_same_username(self):
        assert admin_safety.is_self_target("admin", "admin") is True

    def test_matches_case_insensitively(self):
        assert admin_safety.is_self_target("Admin", "admin") is True
        assert admin_safety.is_self_target("  admin ", "ADMIN") is True

    def test_different_user_is_not_self(self):
        assert admin_safety.is_self_target("admin", "someone-else") is False

    def test_empty_actor_never_matches(self):
        assert admin_safety.is_self_target("", "admin") is False
        assert admin_safety.is_self_target(None, "admin") is False


class TestGroupsConferAdmin:
    """Tests for the group->admin predicate."""

    def test_matches_admin_group(self):
        admins = {"mcp-registry-admin"}
        assert admin_safety.groups_confer_admin(["mcp-registry-admin"], admins) is True

    def test_case_insensitive(self):
        admins = {"mcp-registry-admin"}
        assert admin_safety.groups_confer_admin(["MCP-Registry-Admin"], admins) is True

    def test_no_match(self):
        admins = {"mcp-registry-admin"}
        assert admin_safety.groups_confer_admin(["some-group"], admins) is False

    def test_empty_groups(self):
        assert admin_safety.groups_confer_admin(None, {"mcp-registry-admin"}) is False
        assert admin_safety.groups_confer_admin([], {"mcp-registry-admin"}) is False


@pytest.mark.asyncio
class TestResolveAdminGroupNames:
    """Tests for admin-conferring group resolution from scope docs."""

    @pytest.mark.parametrize("value", ["all", "wall", {"all": True}])
    async def test_non_list_grant_is_not_admin(self, value):
        # get_ui_permissions_for_user reads only list values, so a bare
        # string grants the user nothing. "all" in "wall" is True, which made
        # the guard count such a holder as another admin.
        groups = {
            "registry-admins": {
                "ui_scopes": {"register_service": ["all"]},
                "mappings": ["registry-admins"],
            },
            "ops": {"ui_scopes": {"register_service": value}, "mappings": ["ops"]},
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert result == {"registry-admins"}

    async def test_scoped_grant_is_not_admin(self):
        # "readers" grants only scoped (non-"all") access, so it is NOT
        # admin-conferring.
        groups = {
            "mcp-registry-admin": {
                "ui_scopes": {"register_service": ["all"]},
                "mappings": [],
            },
            "readers": {"ui_scopes": {"list_service": ["some-server"]}, "mappings": []},
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert result == {"mcp-registry-admin"}

    async def test_read_only_all_grant_is_not_admin(self):
        # A read-only "all" grant (the shipped federation-service seed) must
        # not count as admin, or the last real admin could be removed.
        groups = {
            "registry-admins": {
                "ui_scopes": {"modify_service": ["all"]},
                "mappings": ["registry-admins"],
            },
            "federation-service": {
                "ui_scopes": {
                    "list_agents": ["all"],
                    "get_agent": ["all"],
                    "list_service": ["all"],
                    "health_check_service": ["all"],
                },
                "mappings": ["federation-service"],
            },
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert result == {"registry-admins"}

    async def test_non_conferring_mutating_all_grant_is_not_admin(self):
        # Skill and per-type entity scopes are excluded by the request-time rule.
        groups = {
            "admins": {"ui_scopes": {"register_service": ["all"]}, "mappings": []},
            "skill-managers": {
                "ui_scopes": {"publish_skill": ["all"], "create_dataset_entity": ["all"]},
                "mappings": [],
            },
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert result == {"admins"}

    async def test_privileged_name_without_admin_grant_is_not_admin(self):
        # Mirrors _user_is_admin: the name alone confers nothing at request time.
        groups = {
            "mcp-servers-unrestricted/read": {
                "ui_scopes": {"list_service": ["all"]},
                "mappings": [],
            },
            "admins": {"ui_scopes": {"delete_service": ["all"]}, "mappings": []},
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert result == {"admins"}

    async def test_identifies_admin_via_ui_permissions(self):
        groups = {
            "custom-admins": {
                "ui_scopes": {"register_service": ["all"]},
                "mappings": ["idp-admin-group"],
            },
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            result = await admin_safety.resolve_admin_group_names()
        assert "custom-admins" in result
        # Mapped IdP group name is also treated as admin-conferring.
        assert "idp-admin-group" in result

    async def test_fails_closed_on_error_result(self):
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value={"error": "boom", "groups": {}}),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.resolve_admin_group_names()
        assert exc.value.status_code == 503

    async def test_fails_closed_on_exception(self):
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            with pytest.raises(AdminSafetyError):
                await admin_safety.resolve_admin_group_names()

    async def test_fails_closed_on_empty_admin_group_set(self):
        # Catalogue lists groups but NONE match the privileged-scope predicate
        # (predicate/catalogue drift). Must refuse rather than return an empty set
        # that would silently disable the last-admin guard.
        groups = {
            "readers": {"ui_scopes": {"list_service": ["some-server"]}, "mappings": []},
            "viewers": {"ui_scopes": {"list_service": ["another-server"]}, "mappings": []},
        }
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=groups),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.resolve_admin_group_names()
        assert exc.value.status_code == 503


def _iam_mock(users):
    mock = MagicMock()
    mock.list_users = AsyncMock(return_value=users)
    return mock


def _mongo_m2m_mock(docs):
    """Patch target: get_documentdb_client returning a db whose idp_m2m_clients
    collection yields ``docs`` from ``find({}).to_list()``."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    collection = MagicMock()
    collection.find = MagicMock(return_value=cursor)
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return AsyncMock(return_value=db)


@pytest.mark.asyncio
class TestListAdminIdentities:
    """Tests for enumerating admin accounts (one alias-set per admin)."""

    async def test_counts_only_admin_users(self):
        users = [
            {"username": "admin1", "groups": ["mcp-registry-admin"]},
            {"username": "admin2", "groups": ["mcp-registry-admin"]},
            {"username": "regular", "groups": ["readers"]},
        ]
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_m2m_mock([]),
            ),
        ):
            admins = await admin_safety.list_admin_identities()
        assert set(admins) == {frozenset({"admin1"}), frozenset({"admin2"})}

    async def test_counts_m2m_only_admin_from_mongo(self):
        # The IdP listing has NO admin (M2M group membership lives only in
        # idp_m2m_clients for non-Keycloak IdPs). The M2M admin must be counted so
        # the population is not falsely empty and the guard does not 503-brick.
        users = [{"username": "regular", "groups": ["readers"]}]
        m2m_docs = [
            {"client_id": "svc-admin", "name": "svc-admin", "groups": ["mcp-registry-admin"]},
            {"client_id": "svc-reader", "name": "svc-reader", "groups": ["readers"]},
        ]
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_m2m_mock(m2m_docs),
            ),
        ):
            admins = await admin_safety.list_admin_identities()
        assert admins == [frozenset({"svc-admin"})]

    async def test_m2m_admin_aliases_are_ONE_admin_not_two(self):
        # Okta/Auth0 case: client_id (opaque IdP id) != name (friendly label), and
        # the M2M account is NOT in the IdP listing. Both aliases identify ONE
        # admin, so the entry is a single frozenset carrying both keys. This is
        # what lets the guard (a) recognise a delete by EITHER key AND (b) count it
        # as one admin — a flat two-element identifier set would double-count and
        # re-open the last-admin fail-open.
        users = [{"username": "regular", "groups": ["readers"]}]
        m2m_docs = [
            {
                "client_id": "0oa1b2c3d4e5",
                "name": "break-glass-admin",
                "groups": ["mcp-registry-admin"],
            },
        ]
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_m2m_mock(m2m_docs),
            ),
        ):
            admins = await admin_safety.list_admin_identities()
        # ONE admin, addressable by either identifier.
        assert admins == [frozenset({"0oa1b2c3d4e5", "break-glass-admin"})]

    async def test_m2m_store_error_does_not_raise_when_idp_admin_present(self):
        # A datastore error reading idp_m2m_clients is best-effort: it must not
        # raise as long as the IdP listing already surfaced an admin.
        users = [{"username": "admin1", "groups": ["mcp-registry-admin"]}]
        broken = AsyncMock(side_effect=Exception("mongo down"))
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=broken,
            ),
        ):
            admins = await admin_safety.list_admin_identities()
        assert admins == [frozenset({"admin1"})]

    async def test_fails_closed_when_users_unavailable(self):
        mock = MagicMock()
        mock.list_users = AsyncMock(side_effect=Exception("keycloak down"))
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=mock,
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.list_admin_identities()
        assert exc.value.status_code == 503

    async def test_fails_closed_when_no_admins_found(self):
        # Admin-conferring groups exist, but neither the user listing NOR the M2M
        # store surfaces an admin (e.g. an IdP adapter that returned groupless
        # users, or a truncated listing). Deriving "no admins, nothing to guard"
        # would bypass the last-admin guard, so this must fail closed.
        users = [
            {"username": "regular1", "groups": ["readers"]},
            {"username": "regular2", "groups": []},
        ]
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_m2m_mock([]),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.list_admin_identities()
        assert exc.value.status_code == 503


def _mongo_by_collection_mock(docs_by_name):
    """Like _mongo_m2m_mock, but each collection name yields its own docs."""

    def _collection(name):
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=docs_by_name.get(name, []))
        collection = MagicMock()
        collection.find = MagicMock(return_value=cursor)
        return collection

    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=_collection)
    return AsyncMock(return_value=db)


@pytest.mark.asyncio
class TestGroupStoreAdmins:
    """Admins from idp_user_groups, disabled records, and IdPs with no listing."""

    def _patches(self, users_mock, docs_by_name, fallback_active=True):
        return (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch("registry.services.admin_safety.get_iam_manager", return_value=users_mock),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_by_collection_mock(docs_by_name),
            ),
            patch(
                "registry.services.admin_safety.user_group_fallback_active",
                return_value=fallback_active,
            ),
        )

    async def test_idp_without_listing_uses_user_group_store(self):
        # PingFederate: list_users raises NotImplementedError; admins live in
        # idp_user_groups. Must not 503, and disabled records are skipped.
        iam = MagicMock()
        iam.list_users = AsyncMock(side_effect=NotImplementedError("pf"))
        docs = {
            "idp_user_groups": [
                {"username": "pf-admin", "groups": ["mcp-registry-admin"]},
                {"username": "gone", "groups": ["mcp-registry-admin"], "enabled": False},
                {"username": "reader", "groups": ["readers"]},
            ]
        }
        p1, p2, p3, p4 = self._patches(iam, docs)
        with p1, p2, p3, p4:
            admins = await admin_safety.list_admin_identities()
        assert admins == [frozenset({"pf-admin"})]

    async def test_user_in_both_sources_counts_once(self):
        users = [{"username": "Admin1", "groups": ["mcp-registry-admin"]}]
        docs = {"idp_user_groups": [{"username": "admin1", "groups": ["mcp-registry-admin"]}]}
        p1, p2, p3, p4 = self._patches(_iam_mock(users), docs)
        with p1, p2, p3, p4:
            admins = await admin_safety.list_admin_identities()
        assert admins == [frozenset({"admin1"})]

    async def test_user_group_store_ignored_when_fallback_inactive(self):
        users = [{"username": "admin1", "groups": ["mcp-registry-admin"]}]
        docs = {"idp_user_groups": [{"username": "x", "groups": ["mcp-registry-admin"]}]}
        p1, p2, p3, p4 = self._patches(_iam_mock(users), docs, fallback_active=False)
        with p1, p2, p3, p4:
            admins = await admin_safety.list_admin_identities()
        assert admins == [frozenset({"admin1"})]

    async def test_disabled_m2m_admin_is_not_counted(self):
        users = [{"username": "admin1", "groups": ["mcp-registry-admin"]}]
        docs = {
            "idp_m2m_clients": [
                {"client_id": "svc", "name": "svc", "groups": ["mcp-registry-admin"]},
                {
                    "client_id": "off",
                    "name": "off",
                    "groups": ["mcp-registry-admin"],
                    "enabled": False,
                },
            ]
        }
        p1, p2, p3, p4 = self._patches(_iam_mock(users), docs)
        with p1, p2, p3, p4:
            admins = await admin_safety.list_admin_identities()
        assert set(admins) == {frozenset({"admin1"}), frozenset({"svc"})}


@pytest.mark.asyncio
class TestAssertScopeUpdateKeepsAnAdmin:
    """Scope rewrites that drop an admin-conferring grant or mapping."""

    CATALOGUE = {
        "registry-admins": {
            "ui_scopes": {"register_service": ["all"]},
            "mappings": ["idp-admins"],
        },
    }

    async def test_unchanged_admin_grant_skips_population_read(self):
        identities = AsyncMock()
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch("registry.services.admin_safety.list_admin_identities", new=identities),
        ):
            await admin_safety.assert_scope_update_keeps_an_admin(
                "registry-admins", {"register_service": ["all"]}, ["idp-admins"]
            )
        identities.assert_not_awaited()

    async def test_dropping_the_only_admin_grant_is_refused(self):
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=[frozenset({"alice"})]),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_update_keeps_an_admin(
                    "registry-admins", {"list_service": ["all"]}, ["idp-admins"]
                )
        assert exc.value.status_code == 409

    async def test_dropping_the_admin_mapping_is_refused_when_nobody_remains(self):
        users = [{"username": "alice", "groups": ["idp-admins"]}]
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=[frozenset({"alice"})]),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.services.admin_safety._m2m_admin_identities",
                new=AsyncMock(return_value=[]),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_update_keeps_an_admin(
                    "registry-admins", {"register_service": ["all"]}, ["renamed-group"]
                )
        assert exc.value.status_code == 409

    async def test_fails_closed_when_population_unknown(self):
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(side_effect=AdminSafetyError(503, "unknown")),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_update_keeps_an_admin(
                    "registry-admins", {}, ["idp-admins"]
                )
        assert exc.value.status_code == 503

    @pytest.mark.parametrize(
        "catalogue",
        [
            {},
            {"readers": {"ui_scopes": {"list_service": ["all"]}, "mappings": ["readers"]}},
        ],
    )
    async def test_fails_closed_when_catalogue_has_no_admin_scope(self, catalogue):
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value=catalogue),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_update_keeps_an_admin("registry-admins", {}, [])
        assert exc.value.status_code == 503


@pytest.mark.asyncio
class TestAssertScopeDeleteKeepsAnAdmin:
    """Scope deletes that drop an admin-conferring scope or IdP group."""

    CATALOGUE = {
        "registry-admins": {
            "ui_scopes": {"register_service": ["all"]},
            "mappings": ["idp-admins"],
        },
        "idp-admins": {"ui_scopes": {"list_service": ["all"]}, "mappings": ["idp-admins"]},
        "ops-admins": {"ui_scopes": {"delete_service": ["all"]}, "mappings": ["ops"]},
    }

    def _patches(self, users):
        return (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.services.admin_safety._m2m_admin_identities",
                new=AsyncMock(return_value=[]),
            ),
        )

    async def test_deleting_the_only_admins_scope_is_refused(self):
        p1, p2, p3 = self._patches([{"username": "alice", "groups": ["idp-admins"]}])
        with p1, p2, p3:
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_delete_keeps_an_admin(
                    "registry-admins", idp_group_removed=False
                )
        assert exc.value.status_code == 409
        assert "delete" in exc.value.detail

    async def test_deleting_an_admin_scope_is_allowed_when_another_admin_remains(self):
        users = [
            {"username": "alice", "groups": ["idp-admins"]},
            {"username": "bob", "groups": ["ops"]},
        ]
        p1, p2, p3 = self._patches(users)
        with p1, p2, p3:
            await admin_safety.assert_scope_delete_keeps_an_admin(
                "registry-admins", idp_group_removed=True
            )

    async def test_deleting_a_non_admin_scope_skips_population_read(self):
        identities = AsyncMock()
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=self.CATALOGUE),
            ),
            patch("registry.services.admin_safety.list_admin_identities", new=identities),
        ):
            await admin_safety.assert_scope_delete_keeps_an_admin(
                "idp-admins", idp_group_removed=False
            )
        identities.assert_not_awaited()

    async def test_deleting_the_idp_group_drops_its_mapping_in_other_scopes(self):
        # "idp-admins" confers nothing itself, but registry-admins maps it.
        # Deleting the IdP group removes alice's membership, so alice stops
        # being an admin even though the registry-admins scope stays.
        p1, p2, p3 = self._patches([{"username": "alice", "groups": ["idp-admins"]}])
        with p1, p2, p3:
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_delete_keeps_an_admin(
                    "idp-admins", idp_group_removed=True
                )
        assert exc.value.status_code == 409

    async def test_fails_closed_on_empty_catalogue(self):
        with patch(
            "registry.services.admin_safety.scope_service.list_groups",
            new=AsyncMock(return_value={}),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_scope_delete_keeps_an_admin(
                    "registry-admins", idp_group_removed=True
                )
        assert exc.value.status_code == 503


@pytest.mark.asyncio
class TestAssertRecordPatchKeepsAdmin:
    """PATCH bodies for the M2M and user-group record stores."""

    @pytest.mark.parametrize(
        "fields",
        [{"groups": []}, {"groups": None}, {"enabled": False}, {"enabled": None}],
    )
    async def test_demoting_body_is_checked(self, fields):
        with patch(
            "registry.services.admin_safety.would_remove_last_admin_via_groups",
            new=AsyncMock(),
        ) as guard:
            await admin_safety.assert_record_patch_keeps_admin("t", fields)
        guard.assert_awaited_once_with("t", [])

    async def test_new_groups_are_passed_through(self):
        with patch(
            "registry.services.admin_safety.would_remove_last_admin_via_groups",
            new=AsyncMock(),
        ) as guard:
            await admin_safety.assert_record_patch_keeps_admin(
                "t", {"enabled": True, "groups": ["readers"]}
            )
        guard.assert_awaited_once_with("t", ["readers"])

    async def test_body_without_groups_or_enabled_is_not_checked(self):
        with patch(
            "registry.services.admin_safety.would_remove_last_admin_via_groups",
            new=AsyncMock(),
        ) as guard:
            await admin_safety.assert_record_patch_keeps_admin("t", {"description": "d"})
        guard.assert_not_awaited()


@pytest.mark.asyncio
class TestAssertNotLastAdmin:
    """Tests for the last-admin deletion guard."""

    async def test_allows_when_other_admins_remain(self):
        with patch(
            "registry.services.admin_safety.list_admin_identities",
            new=AsyncMock(return_value=[frozenset({"admin1"}), frozenset({"admin2"})]),
        ):
            # Should not raise.
            await admin_safety.assert_not_last_admin("admin1")

    async def test_rejects_last_admin(self):
        with patch(
            "registry.services.admin_safety.list_admin_identities",
            new=AsyncMock(return_value=[frozenset({"admin1"})]),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_not_last_admin("admin1")
        assert exc.value.status_code == 409

    async def test_noop_when_target_not_admin(self):
        with patch(
            "registry.services.admin_safety.list_admin_identities",
            new=AsyncMock(return_value=[frozenset({"admin1"})]),
        ):
            # Deleting a non-admin cannot empty the admin population.
            await admin_safety.assert_not_last_admin("regular-user")

    async def test_fails_closed_when_population_unknown(self):
        with patch(
            "registry.services.admin_safety.list_admin_identities",
            new=AsyncMock(side_effect=AdminSafetyError(503, "unknown")),
        ):
            with pytest.raises(AdminSafetyError):
                await admin_safety.assert_not_last_admin("admin1")

    async def test_read_only_all_holder_does_not_keep_last_admin_deletable(self):
        # End to end: a federation-service principal (read-only "all") is not
        # "another admin", so deleting the only real admin is refused.
        groups = {
            "registry-admins": {"ui_scopes": {"modify_service": ["all"]}, "mappings": []},
            "federation-service": {"ui_scopes": {"list_service": ["all"]}, "mappings": []},
        }
        users = [
            {"username": "admin1", "groups": ["registry-admins"]},
            {"username": "fed-bot", "groups": ["federation-service"]},
        ]
        with (
            patch(
                "registry.services.admin_safety.scope_service.list_groups",
                new=AsyncMock(return_value=groups),
            ),
            patch(
                "registry.services.admin_safety.get_iam_manager",
                return_value=_iam_mock(users),
            ),
            patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new=_mongo_m2m_mock([]),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.assert_not_last_admin("admin1")
        assert exc.value.status_code == 409

    async def test_rejects_last_admin_deleted_by_either_m2m_key(self):
        # The sole admin is ONE M2M client whose alias-set carries both client_id
        # and name. Deleting by EITHER identifier must be refused. This is the case
        # a flat identifier set got wrong: two aliases must count as one admin so
        # removing either empties the population.
        sole_admin = [frozenset({"0oa1b2c3d4e5", "break-glass-admin"})]
        for target in ("0oa1b2c3d4e5", "break-glass-admin"):
            with patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=list(sole_admin)),
            ):
                with pytest.raises(AdminSafetyError) as exc:
                    await admin_safety.assert_not_last_admin(target)
            assert exc.value.status_code == 409

    async def test_allows_when_another_distinct_admin_remains_alongside_m2m(self):
        # Two DISTINCT admins: a two-alias M2M client and a regular user. Deleting
        # the M2M admin (by either alias) is allowed because the regular admin
        # remains. Guards against over-counting the aliases as "the only admin".
        admins = [
            frozenset({"0oa1b2c3d4e5", "break-glass-admin"}),
            frozenset({"human-admin"}),
        ]
        for target in ("0oa1b2c3d4e5", "break-glass-admin"):
            with patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=list(admins)),
            ):
                # Must NOT raise — another distinct admin remains.
                await admin_safety.assert_not_last_admin(target)


@pytest.mark.asyncio
class TestWouldRemoveLastAdminViaGroups:
    """Tests for the last-admin demotion guard."""

    async def test_allows_when_new_groups_still_admin(self):
        with patch(
            "registry.services.admin_safety.resolve_admin_group_names",
            new=AsyncMock(return_value={"mcp-registry-admin"}),
        ):
            # Keeping the admin group means no demotion; must not raise.
            await admin_safety.would_remove_last_admin_via_groups("admin1", ["mcp-registry-admin"])

    async def test_rejects_demoting_last_admin(self):
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=[frozenset({"admin1"})]),
            ),
        ):
            with pytest.raises(AdminSafetyError) as exc:
                await admin_safety.would_remove_last_admin_via_groups("admin1", ["readers"])
        assert exc.value.status_code == 409

    async def test_allows_demoting_when_other_admins_remain(self):
        with (
            patch(
                "registry.services.admin_safety.resolve_admin_group_names",
                new=AsyncMock(return_value={"mcp-registry-admin"}),
            ),
            patch(
                "registry.services.admin_safety.list_admin_identities",
                new=AsyncMock(return_value=[frozenset({"admin1"}), frozenset({"admin2"})]),
            ),
        ):
            await admin_safety.would_remove_last_admin_via_groups("admin1", ["readers"])


@pytest.mark.asyncio
class TestDesiredGroupsGrantAdmin:
    """Tests for the admin-grant detection used by the audit hook."""

    async def test_true_when_granting_admin_group(self):
        with patch(
            "registry.services.admin_safety.resolve_admin_group_names",
            new=AsyncMock(return_value={"mcp-registry-admin"}),
        ):
            assert await admin_safety.desired_groups_grant_admin(["mcp-registry-admin"])

    async def test_false_for_non_admin_groups(self):
        with patch(
            "registry.services.admin_safety.resolve_admin_group_names",
            new=AsyncMock(return_value={"mcp-registry-admin"}),
        ):
            assert not await admin_safety.desired_groups_grant_admin(["readers"])
