"""Unit tests for config-driven M2M client-group assignment in the Okta sync.

Security property under test (TM2-6 class): the Okta M2M sync must never
auto-assign group memberships — least of all privileged/admin groups — from a
code-shipped hardcoded mapping. An attacker who can influence the Okta API
response, point the sync at a rogue tenant, or write to MongoDB must not be able
to have an injected client_id auto-granted RBAC. Group assignment is fail-closed:
a client_id receives groups ONLY if it is explicitly listed in the operator's
``OKTA_M2M_CLIENT_GROUPS`` configuration.
"""

from unittest.mock import MagicMock

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
