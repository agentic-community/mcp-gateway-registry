"""Tests for issue #956 ETag and batch hashing helpers in agent_routes.

Covers the pure helpers:
- _weak_etag_for / _agent_updated_ms (derive a weak validator from updated_at)
- _parse_if_match (parse and reject malformed/strong If-Match headers)
- _hash_items (stable SHA-256 over canonicalized batch items)
"""

from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

from registry.api.agent_routes import (
    _agent_updated_ms,
    _hash_items,
    _parse_if_match,
    _weak_etag_for,
)
from registry.schemas.agent_models import AgentBatchItem
from tests.fixtures.factories import AgentCardFactory

_ITEM_ADAPTER = TypeAdapter(list[AgentBatchItem])


@pytest.mark.unit
class TestWeakEtag:
    """Tests for _weak_etag_for and _agent_updated_ms."""

    def test_etag_uses_updated_at_epoch_ms(self):
        ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        card = AgentCardFactory(updated_at=ts)
        expected_ms = int(ts.timestamp() * 1000)
        assert _weak_etag_for(card) == f'W/"{expected_ms}"'
        assert _agent_updated_ms(card) == expected_ms

    def test_etag_falls_back_to_registered_at(self):
        ts = datetime(2025, 6, 1, tzinfo=UTC)
        card = AgentCardFactory(updated_at=None, registered_at=ts)
        expected_ms = int(ts.timestamp() * 1000)
        assert _weak_etag_for(card) == f'W/"{expected_ms}"'
        assert _agent_updated_ms(card) == expected_ms

    def test_etag_zero_when_no_timestamps(self):
        card = AgentCardFactory(updated_at=None, registered_at=None)
        assert _weak_etag_for(card) == 'W/"0"'
        assert _agent_updated_ms(card) == 0

    def test_etag_roundtrips_through_parse(self):
        """An emitted ETag parses back to the same epoch-ms it was built from."""
        card = AgentCardFactory(updated_at=datetime(2026, 3, 4, tzinfo=UTC))
        etag = _weak_etag_for(card)
        assert _parse_if_match(etag) == _agent_updated_ms(card)


@pytest.mark.unit
class TestParseIfMatch:
    """Tests for _parse_if_match."""

    def test_none_returns_none(self):
        assert _parse_if_match(None) is None

    def test_valid_weak_etag_parsed(self):
        assert _parse_if_match('W/"1700000000000"') == 1700000000000

    def test_surrounding_whitespace_tolerated(self):
        assert _parse_if_match('  W/"42"  ') == 42

    def test_strong_etag_rejected_with_message(self):
        with pytest.raises(HTTPException) as exc:
            _parse_if_match('"1700000000000"')
        assert exc.value.status_code == 400
        assert "Strong ETag not supported" in exc.value.detail

    @pytest.mark.parametrize(
        "bad",
        [
            "1700000000000",  # bare number, no W/"..."
            'W/"abc"',  # non-numeric
            'W/""',  # empty inner value
            "garbage",
            "W/1700",  # missing quotes
        ],
    )
    def test_malformed_rejected(self, bad):
        with pytest.raises(HTTPException) as exc:
            _parse_if_match(bad)
        assert exc.value.status_code == 400


@pytest.mark.unit
class TestHashItems:
    """Tests for _hash_items idempotency hashing."""

    def _items(self, data):
        return _ITEM_ADAPTER.validate_python(data)

    def test_hash_is_deterministic(self):
        data = [{"op": "delete", "path": "/agents/a"}]
        assert _hash_items(self._items(data)) == _hash_items(self._items(data))

    def test_hash_is_sha256_hex(self):
        h = _hash_items(self._items([{"op": "delete", "path": "/agents/a"}]))
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_different_items_differ(self):
        h1 = _hash_items(self._items([{"op": "delete", "path": "/agents/a"}]))
        h2 = _hash_items(self._items([{"op": "delete", "path": "/agents/b"}]))
        assert h1 != h2

    def test_order_sensitive(self):
        """Reordering items changes the hash (list order is part of identity)."""
        a = {"op": "delete", "path": "/agents/a"}
        b = {"op": "delete", "path": "/agents/b"}
        assert _hash_items(self._items([a, b])) != _hash_items(self._items([b, a]))
