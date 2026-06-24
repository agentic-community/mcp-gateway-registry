"""Unit tests for the token_mint branch of the audit query builder (#1308).

The token_mint stream stores a hashed username at the top level
(``username_hash``) and flat resource fields (not nested under ``action.*``),
so ``_build_query`` must route filters differently than the other streams.
"""

from registry.audit.routes import _build_query


def _q(**overrides):
    """Call _build_query with token_mint defaults, overriding as needed."""
    kwargs = {
        "stream": "token_mint",
        "from_time": None,
        "to_time": None,
        "username": None,
        "operation": None,
        "resource_type": None,
        "resource_id": None,
        "status_min": None,
        "status_max": None,
        "auth_decision": None,
    }
    kwargs.update(overrides)
    return _build_query(**kwargs)


class TestTokenMintQuery:
    def test_stream_maps_to_token_mint_log_type(self):
        assert _q() == {"log_type": "token_mint"}

    def test_username_filters_on_hash_not_identity(self):
        query = _q(username="user_abcd1234")
        assert "username_hash" in query
        assert query["username_hash"]["$regex"] == "user_abcd1234"
        assert query["username_hash"]["$options"] == "i"
        # Must NOT use the registry_api/mcp_access identity field.
        assert "identity.username" not in query

    def test_operation_filters_on_token_kind(self):
        query = _q(operation="resource")
        assert query["token_kind"] == "resource"
        # Must NOT use the registry_api action field.
        assert "action.operation" not in query

    def test_resource_fields_are_flat(self):
        query = _q(resource_type="server", resource_id="fininfo")
        assert query["resource_type"] == "server"
        assert query["resource_id"] == "fininfo"
        # Flat, not nested under action.*
        assert "action.resource_type" not in query
        assert "action.resource_id" not in query

    def test_combined_filters(self):
        query = _q(username="user_dead", operation="user", resource_type="agent")
        assert query["log_type"] == "token_mint"
        assert query["username_hash"]["$regex"] == "user_dead"
        assert query["token_kind"] == "user"
        assert query["resource_type"] == "agent"
