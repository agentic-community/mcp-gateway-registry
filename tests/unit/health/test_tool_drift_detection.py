"""Saving a fetched tool list only when a tool definition changed (issue #2050)."""

import copy
import logging
from unittest.mock import AsyncMock, patch

import pytest

from registry.health.service import HealthMonitoringService
from registry.utils.tool_list_hash import tool_list_hash

SERVER_PATH: str = "/notes"
STORED_TOOLS: list[dict] = [
    {
        "name": "read_note",
        "description": "Read a note.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
    {
        "name": "delete_note",
        "description": "Delete a note.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
]
SERVER_INFO: dict = {"name": "notes", "version": "1.0.0"}


class _StoredServerService:
    """Stand-in for server_service backed by one stored card, with field-scoped writes."""

    def __init__(self, card: dict):
        self.card = copy.deepcopy(card)
        self.writes: list[list[str] | None] = []

    async def get_server_info(self, path: str, include_credentials: bool = False):
        return copy.deepcopy(self.card)

    async def update_server(self, path: str, server_info: dict, *, updated_fields=None, **_):
        self.writes.append(updated_fields)
        doc = {k: v for k, v in server_info.items() if k != "path"}
        if updated_fields is not None:
            doc = {k: v for k, v in doc.items() if k in set(updated_fields)}
        self.card.update(copy.deepcopy(doc))
        return True


@pytest.fixture
def service() -> HealthMonitoringService:
    """A health service whose broadcasts do nothing."""
    instance = HealthMonitoringService()
    instance.broadcast_health_update = AsyncMock()
    return instance


@pytest.fixture
def store() -> _StoredServerService:
    """A stored server with two tools and version 1.0.0, from before the hash existed."""
    return _StoredServerService(
        {
            "server_name": "notes",
            "path": SERVER_PATH,
            "tool_list": copy.deepcopy(STORED_TOOLS),
            "num_tools": 2,
            "mcp_server_version": "1.0.0",
        }
    )


async def _save(
    service: HealthMonitoringService,
    store: _StoredServerService,
    tool_list: list[dict],
    mcp_server_info: dict | None = SERVER_INFO,
    blocked: set[str] | None = None,
) -> tuple[str, AsyncMock]:
    """Run _save_discovered_tools against the store; return the result and the scope mock."""
    scope_mock = AsyncMock()
    with (
        patch("registry.services.server_service.server_service", store),
        patch("registry.services.scope_service.update_server_scopes", scope_mock),
        patch(
            "registry.services.tool_blocks.blocked_names_for",
            AsyncMock(return_value=blocked or set()),
        ),
    ):
        result = await service._save_discovered_tools(SERVER_PATH, tool_list, mcp_server_info)
    return result, scope_mock


@pytest.mark.unit
@pytest.mark.asyncio
class TestToolDriftDetection:
    """_save_discovered_tools writes on a real change and stays quiet otherwise."""

    async def test_unchanged_list_writes_nothing(self, service, store):
        result, scope_mock = await _save(service, store, copy.deepcopy(STORED_TOOLS))
        assert result == "unchanged"
        assert store.writes == []
        scope_mock.assert_not_awaited()

    async def test_unchanged_list_without_stored_hash_writes_nothing(self, service, store):
        assert "tool_list_hash" not in store.card
        await _save(service, store, copy.deepcopy(STORED_TOOLS))
        assert "tool_list_hash" not in store.card

    async def test_missing_server_version_is_not_a_change(self, service, store):
        result, _ = await _save(service, store, copy.deepcopy(STORED_TOOLS), mcp_server_info=None)
        assert result == "unchanged"
        assert store.writes == []

    async def test_description_change_rewrites_list(self, service, store, caplog):
        fetched = copy.deepcopy(STORED_TOOLS)
        fetched[0]["description"] = "Read one note by id."
        with caplog.at_level(logging.INFO, logger="registry.health.service"):
            result, scope_mock = await _save(service, store, fetched)
        assert result == "changed"
        assert store.card["tool_list"][0]["description"] == "Read one note by id."
        assert store.card["tool_list_hash"] == tool_list_hash(fetched)
        assert store.card["tool_list_changed_at"]
        assert "changed=['read_note']" in caplog.text
        scope_mock.assert_not_awaited()

    async def test_input_schema_change_rewrites_list(self, service, store):
        fetched = copy.deepcopy(STORED_TOOLS)
        fetched[0]["inputSchema"]["properties"]["format"] = {"type": "string"}
        result, _ = await _save(service, store, fetched)
        assert result == "changed"
        assert "format" in store.card["tool_list"][0]["inputSchema"]["properties"]

    async def test_write_is_field_scoped(self, service, store):
        fetched = copy.deepcopy(STORED_TOOLS)
        fetched[1]["description"] = "Delete a note forever."
        await _save(service, store, fetched)
        written = set(store.writes[0])
        assert {"tool_list", "num_tools", "tool_list_hash", "tool_list_changed_at"} <= written
        assert "server_name" not in written

    async def test_scope_update_on_name_change(self, service, store):
        fetched = [*copy.deepcopy(STORED_TOOLS), {"name": "list_notes", "description": "List."}]
        _, scope_mock = await _save(service, store, fetched)
        scope_mock.assert_awaited_once()
        assert scope_mock.await_args.args[2] == ["read_note", "delete_note", "list_notes"]

    async def test_first_discovery_updates_scopes_without_changed_at(self, service, store):
        store.card["tool_list"] = []
        store.card["num_tools"] = 0
        result, scope_mock = await _save(service, store, copy.deepcopy(STORED_TOOLS))
        assert result == "changed"
        assert "tool_list_changed_at" not in store.card
        assert store.card["tool_list_hash"] == tool_list_hash(STORED_TOOLS)
        scope_mock.assert_awaited_once()

    async def test_version_change_alone_rewrites(self, service, store):
        result, _ = await _save(
            service, store, copy.deepcopy(STORED_TOOLS), {"name": "notes", "version": "1.1.0"}
        )
        assert result == "changed"
        assert store.card["mcp_server_version"] == "1.1.0"
        assert store.card["mcp_server_version_previous"] == "1.0.0"
        assert "tool_list_changed_at" not in store.card

    async def test_removed_blocked_tool_warns(self, service, store, caplog):
        fetched = [copy.deepcopy(STORED_TOOLS[0]), {"name": "remove_note", "description": "x"}]
        with caplog.at_level(logging.WARNING, logger="registry.health.service"):
            await _save(service, store, fetched, blocked={"delete_note"})
        assert "Blocked tools no longer listed by /notes: ['delete_note']" in caplog.text

    async def test_removed_unblocked_tool_does_not_warn(self, service, store, caplog):
        fetched = [copy.deepcopy(STORED_TOOLS[0])]
        with caplog.at_level(logging.WARNING, logger="registry.health.service"):
            await _save(service, store, fetched, blocked=set())
        assert "Blocked tools" not in caplog.text


@pytest.mark.unit
def test_log_names_escape_newlines():
    """An upstream tool name with a newline cannot forge a second log line."""
    from registry.health.service import _names_for_log

    assert "\n" not in _names_for_log(["evil\nINFO,forged line"])


@pytest.mark.unit
def test_log_names_are_capped():
    from registry.health.service import (
        MAX_LOGGED_TOOL_NAMES,
        _names_for_log,
    )

    names = [f"tool_{index}" for index in range(MAX_LOGGED_TOOL_NAMES + 5)]
    assert _names_for_log(names).endswith("and 5 more")
