"""Unit tests for config-driven M2M client-group assignment in the Okta sync.

Security property under test (TM2-6 class): the Okta M2M sync must never
auto-assign group memberships — least of all privileged/admin groups — from a
code-shipped hardcoded mapping. An attacker who can influence the Okta API
response, point the sync at a rogue tenant, or write to MongoDB must not be able
to have an injected client_id auto-granted RBAC. Group assignment is fail-closed:
a client_id receives groups ONLY if it is explicitly listed in the operator's
``OKTA_M2M_CLIENT_GROUPS`` configuration.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.services.auth0_m2m_sync import (
    AUTH0_M2M_CLIENT_GROUPS_ENV,
    Auth0M2MSync,
)
from registry.services.auth0_m2m_sync import (
    _load_client_groups_mapping as _load_auth0_mapping,
)
from registry.services.okta_m2m_sync import (
    OKTA_M2M_CLIENT_GROUPS_ENV,
    OktaM2MSync,
    _load_client_groups_mapping,
)

pytestmark = pytest.mark.unit


def _make_sync() -> OktaM2MSync:
    """Build a sync instance with a mocked database (no I/O)."""
    db = MagicMock()
    db.__getitem__.return_value = MagicMock()
    return OktaM2MSync(db=db, okta_domain="dev-123.okta.com", okta_api_token="token")  # noqa: S106


def _make_auth0_sync() -> Auth0M2MSync:
    """Build an Auth0 sync instance with a mocked database (no I/O)."""
    db = MagicMock()
    db.__getitem__.return_value = MagicMock()
    return Auth0M2MSync(
        db=db,
        auth0_domain="dev-abc.us.auth0.com",
        m2m_client_id="mgmt-client",
        m2m_client_secret="mgmt-secret",  # noqa: S106 - test fixture
    )


class TestLoadClientGroupsMapping:
    """The mapping is loaded from config and fails closed on bad input."""

    def test_unset_env_yields_empty_mapping(self, monkeypatch):
        monkeypatch.delenv(OKTA_M2M_CLIENT_GROUPS_ENV, raising=False)
        assert _load_client_groups_mapping() == {}

    def test_empty_env_yields_empty_mapping(self, monkeypatch):
        monkeypatch.setenv(OKTA_M2M_CLIENT_GROUPS_ENV, "   ")
        assert _load_client_groups_mapping() == {}

    def test_valid_json_mapping_loaded(self, monkeypatch):
        monkeypatch.setenv(
            OKTA_M2M_CLIENT_GROUPS_ENV,
            '{"0oaABC": ["public-mcp-users"], "0oaDEF": ["registry-admins"]}',
        )
        mapping = _load_client_groups_mapping()
        assert mapping == {
            "0oaABC": ["public-mcp-users"],
            "0oaDEF": ["registry-admins"],
        }

    def test_malformed_json_fails_closed(self, monkeypatch):
        monkeypatch.setenv(OKTA_M2M_CLIENT_GROUPS_ENV, "{not json")
        assert _load_client_groups_mapping() == {}

    def test_non_object_json_fails_closed(self, monkeypatch):
        monkeypatch.setenv(OKTA_M2M_CLIENT_GROUPS_ENV, '["registry-admins"]')
        assert _load_client_groups_mapping() == {}

    def test_malformed_entry_dropped(self, monkeypatch):
        # groups value must be a list of strings; a non-list entry is dropped.
        monkeypatch.setenv(
            OKTA_M2M_CLIENT_GROUPS_ENV,
            '{"0oaGOOD": ["public-mcp-users"], "0oaBAD": "registry-admins"}',
        )
        mapping = _load_client_groups_mapping()
        assert mapping == {"0oaGOOD": ["public-mcp-users"]}


class TestDetermineGroups:
    """No admin (or any) group is granted unless explicitly configured."""

    def test_unconfigured_client_gets_no_groups(self, monkeypatch):
        # The historically hardcoded admin client_id must no longer be granted
        # admin (or any) groups when it is not in the configured mapping.
        monkeypatch.delenv(OKTA_M2M_CLIENT_GROUPS_ENV, raising=False)
        sync = _make_sync()
        assert sync._determine_groups("0oa1100req1AzfKaY698") == []
        assert sync._determine_groups("any-injected-client-id") == []

    def test_configured_client_gets_configured_groups(self, monkeypatch):
        monkeypatch.setenv(
            OKTA_M2M_CLIENT_GROUPS_ENV,
            '{"0oaTRUSTED": ["public-mcp-users"]}',
        )
        sync = _make_sync()
        assert sync._determine_groups("0oaTRUSTED") == ["public-mcp-users"]

    def test_injected_client_not_in_config_denied_admin(self, monkeypatch):
        # Only the explicitly-listed client gets groups; an attacker-injected
        # client_id, even if it collides with a name, gets nothing.
        monkeypatch.setenv(
            OKTA_M2M_CLIENT_GROUPS_ENV,
            '{"0oaTRUSTED": ["registry-admins"]}',
        )
        sync = _make_sync()
        assert sync._determine_groups("0oaATTACKER") == []


class TestAuth0SyncGroups:
    """The same fail-closed, config-driven mapping applies to the Auth0 sync."""

    def test_unconfigured_client_gets_no_groups(self, monkeypatch):
        monkeypatch.delenv(AUTH0_M2M_CLIENT_GROUPS_ENV, raising=False)
        sync = _make_auth0_sync()
        assert sync._determine_groups("any-client-id") == []

    def test_configured_client_gets_configured_groups(self, monkeypatch):
        monkeypatch.setenv(
            AUTH0_M2M_CLIENT_GROUPS_ENV,
            '{"abc123": ["public-mcp-users"]}',
        )
        sync = _make_auth0_sync()
        assert sync._determine_groups("abc123") == ["public-mcp-users"]

    def test_malformed_json_fails_closed(self, monkeypatch):
        monkeypatch.setenv(AUTH0_M2M_CLIENT_GROUPS_ENV, "{bad json")
        assert _load_auth0_mapping() == {}


class _Auth0Collection:
    """Store documents and apply MongoDB's insert-only update semantics."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def find_one(self, query):
        return deepcopy(self.docs.get(query["client_id"]))

    async def insert_one(self, document):
        self.docs[document["client_id"]] = deepcopy(document)
        return SimpleNamespace(inserted_id=document["client_id"])

    async def update_one(self, query, update, upsert=False):
        client_id = query["client_id"]
        existing = self.docs.get(client_id)
        if existing is None:
            if not upsert:
                return SimpleNamespace(upserted_id=None, matched_count=0)
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.docs[client_id] = document
        else:
            document = existing
        document.update(deepcopy(update.get("$set", {})))
        return SimpleNamespace(
            upserted_id=client_id if existing is None else None,
            matched_count=0 if existing is None else 1,
        )


@pytest.fixture
def auth0_sync_store(monkeypatch):
    monkeypatch.delenv(AUTH0_M2M_CLIENT_GROUPS_ENV, raising=False)
    db = {
        "auth0_m2m_clients": _Auth0Collection(),
        "idp_m2m_clients": _Auth0Collection(),
    }
    sync = Auth0M2MSync(
        db=db,
        auth0_domain="dev-abc.us.auth0.com",
        m2m_client_id="mgmt-client",
        m2m_client_secret="mgmt-secret",  # noqa: S106 - test fixture
    )
    sync._get_management_api_token = AsyncMock(return_value="management-token")
    return sync, db


def _auth0_application(is_first_party):
    return {
        "client_id": "client-id",
        "name": "service-account",
        "app_type": "non_interactive",
        "is_first_party": is_first_party,
    }


class TestAuth0SyncEnablement:
    """Consent classification cannot override local enablement."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_first_party", [True, False])
    async def test_new_application_is_enabled_in_both_collections(
        self, auth0_sync_store, is_first_party
    ):
        sync, db = auth0_sync_store
        sync._get_auth0_clients = AsyncMock(return_value=[_auth0_application(is_first_party)])

        result = await sync.sync_from_auth0()

        assert result["errors"] == []
        assert result["added_count"] == 1
        for collection in db.values():
            assert collection.docs["client-id"]["enabled"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_first_party", [True, False])
    @pytest.mark.parametrize("force_full_sync", [True, False])
    @pytest.mark.parametrize("mirror_enabled", [True, False])
    @pytest.mark.parametrize("idp_enabled", [True, False])
    async def test_sync_preserves_each_collections_enablement(
        self, auth0_sync_store, is_first_party, force_full_sync, mirror_enabled, idp_enabled
    ):
        sync, db = auth0_sync_store
        expected = {"auth0_m2m_clients": mirror_enabled, "idp_m2m_clients": idp_enabled}
        for name, enabled in expected.items():
            db[name].docs["client-id"] = {
                "client_id": "client-id",
                "enabled": enabled,
                "name": "previous-name",
            }
        sync._get_auth0_clients = AsyncMock(return_value=[_auth0_application(is_first_party)])

        result = await sync.sync_from_auth0(force_full_sync=force_full_sync)

        assert result["errors"] == []
        assert result["updated_count"] == 1
        for name, enabled in expected.items():
            assert db[name].docs["client-id"]["enabled"] is enabled
            assert db[name].docs["client-id"]["name"] == "service-account"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("existing_collection", ["auth0_m2m_clients", "idp_m2m_clients"])
    @pytest.mark.parametrize("is_first_party", [True, False])
    @pytest.mark.parametrize("force_full_sync", [True, False])
    async def test_missing_counterpart_inherits_disabled_state(
        self, auth0_sync_store, existing_collection, is_first_party, force_full_sync
    ):
        sync, db = auth0_sync_store
        db[existing_collection].docs["client-id"] = {
            "client_id": "client-id",
            "enabled": False,
        }
        sync._get_auth0_clients = AsyncMock(return_value=[_auth0_application(is_first_party)])

        result = await sync.sync_from_auth0(force_full_sync=force_full_sync)

        assert result["errors"] == []
        for collection in db.values():
            assert collection.docs["client-id"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_legacy_disabled_rows_require_explicit_reconciliation(self, auth0_sync_store):
        sync, db = auth0_sync_store
        for collection in db.values():
            collection.docs["client-id"] = {"client_id": "client-id", "enabled": False}
        sync._get_auth0_clients = AsyncMock(return_value=[_auth0_application(True)])

        result = await sync.sync_from_auth0(force_full_sync=True)

        assert result["errors"] == []
        for collection in db.values():
            assert collection.docs["client-id"]["enabled"] is False
            await collection.update_one({"client_id": "client-id"}, {"$set": {"enabled": True}})

        result = await sync.sync_from_auth0(force_full_sync=True)

        assert result["errors"] == []
        for collection in db.values():
            assert collection.docs["client-id"]["enabled"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("disabled_collection", ["auth0_m2m_clients", "idp_m2m_clients"])
    @pytest.mark.parametrize("existing", [True, False])
    async def test_concurrent_disable_survives_sync_update(
        self, auth0_sync_store, disabled_collection, existing
    ):
        sync, db = auth0_sync_store
        if existing:
            for collection in db.values():
                collection.docs["client-id"] = {"client_id": "client-id", "enabled": True}
        collection = db[disabled_collection]
        original_update = collection.update_one

        async def disable_before_update(query, update, upsert=False):
            await original_update(
                {"client_id": "client-id"}, {"$set": {"enabled": False}}, upsert=True
            )
            return await original_update(query, update, upsert=upsert)

        collection.update_one = disable_before_update
        sync._get_auth0_clients = AsyncMock(return_value=[_auth0_application(False)])

        result = await sync.sync_from_auth0(force_full_sync=True)

        assert result["errors"] == []
        assert collection.docs["client-id"]["enabled"] is False
        assert collection.docs["client-id"]["name"] == "service-account"
