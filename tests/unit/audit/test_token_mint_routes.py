"""Unit tests for the token_mint branch of the audit query builder (#1308).

The token_mint stream stores the raw, human-readable identity at the top level
(``username``; ``username_hash`` is deprecated) and flat resource fields (not
nested under ``action.*``), so ``_build_query`` must route filters differently
than the other streams.
"""

import re

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

    def test_username_filters_on_raw_username_not_identity(self):
        query = _q(username="alice@example.com")
        # Filters on the raw, human-readable `username` field (not the
        # deprecated `username_hash`, and not the nested identity field), plus
        # the top-level identity claims. The value is regex-escaped (the "." in
        # an email is a metachar).
        expected = {"$regex": re.escape("alice@example.com"), "$options": "i"}
        assert {"username": expected} in query["$or"]
        # Opaque identity claims match exactly (no regex): they are pasted whole.
        assert {"canonical_id": "alice@example.com"} in query["$or"]
        assert "username_hash" not in query
        # token_mint has no nested identity block.
        assert not any(field.startswith("identity.") for clause in query["$or"] for field in clause)

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
        query = _q(username="alice@example.com", operation="user", resource_type="agent")
        assert query["log_type"] == "token_mint"
        assert {"username": {"$regex": re.escape("alice@example.com"), "$options": "i"}} in query[
            "$or"
        ]
        assert query["token_kind"] == "user"
        assert query["resource_type"] == "agent"
