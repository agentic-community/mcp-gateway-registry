"""Tool list hashing and comparison for drift detection (issue #2050)."""

import copy

import pytest

from registry.utils.tool_list_hash import (
    ToolListChanges,
    compare_tool_lists,
    tool_definition_hash,
    tool_list_hash,
)

SEARCH_TOOL: dict = {
    "name": "search",
    "description": "Search the catalog.",
    "parsed_description": {"main": "Search the catalog.", "args": None},
    "schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
}
DELETE_TOOL: dict = {
    "name": "delete",
    "description": "Delete an item.",
    "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
    "annotations": {"readOnlyHint": False, "destructiveHint": True},
}


def _tools() -> list[dict]:
    """Return a fresh two-tool list."""
    return [copy.deepcopy(SEARCH_TOOL), copy.deepcopy(DELETE_TOOL)]


@pytest.mark.unit
class TestToolListHash:
    """tool_list_hash covers what a client acts on and nothing else."""

    def test_hash_is_hex_sha256(self):
        assert len(tool_list_hash(_tools())) == 64

    def test_hash_ignores_tool_order(self):
        tools = _tools()
        assert tool_list_hash(tools) == tool_list_hash(list(reversed(tools)))

    def test_hash_changes_on_description_edit(self):
        edited = _tools()
        edited[0]["description"] = "Search the catalog by keyword."
        assert tool_list_hash(edited) != tool_list_hash(_tools())

    def test_hash_changes_on_input_schema_edit(self):
        edited = _tools()
        edited[0]["inputSchema"]["properties"]["limit"] = {"type": "integer"}
        assert tool_list_hash(edited) != tool_list_hash(_tools())

    def test_hash_changes_on_annotation_flip(self):
        edited = _tools()
        edited[1]["annotations"]["destructiveHint"] = False
        assert tool_list_hash(edited) != tool_list_hash(_tools())

    def test_hash_changes_on_title_and_output_schema(self):
        titled = _tools()
        titled[0]["title"] = "Catalog search"
        with_output = _tools()
        with_output[0]["outputSchema"] = {"type": "object"}
        assert tool_list_hash(titled) != tool_list_hash(_tools())
        assert tool_list_hash(with_output) != tool_list_hash(_tools())

    def test_hash_changes_on_rename(self):
        renamed = _tools()
        renamed[0]["name"] = "find"
        assert tool_list_hash(renamed) != tool_list_hash(_tools())

    def test_hash_ignores_parsed_description(self):
        edited = _tools()
        edited[0]["parsed_description"] = {"main": "something else"}
        assert tool_list_hash(edited) == tool_list_hash(_tools())

    def test_hash_reads_legacy_schema_key(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        legacy = {"name": "search", "description": "d", "schema": schema}
        modern = {"name": "search", "description": "d", "inputSchema": schema}
        assert tool_definition_hash(legacy) == tool_definition_hash(modern)

    def test_missing_description_equals_empty(self):
        assert tool_definition_hash({"name": "a"}) == tool_definition_hash(
            {"name": "a", "description": None}
        )

    def test_empty_and_none_lists_hash_equal(self):
        assert tool_list_hash([]) == tool_list_hash(None)

    def test_non_dict_entries_are_skipped(self):
        assert tool_list_hash([*_tools(), "junk", 7]) == tool_list_hash(_tools())

    def test_invalid_annotation_values_do_not_change_hash(self):
        """Annotations go through the #2047 validator, so a non-boolean hint is dropped."""
        noisy = _tools()
        noisy[0]["annotations"] = {"readOnlyHint": "yes"}
        assert tool_list_hash(noisy) == tool_list_hash(_tools())


@pytest.mark.unit
class TestCompareToolLists:
    """compare_tool_lists names what changed, for the drift log line."""

    def test_identical_lists_report_nothing(self):
        changes = compare_tool_lists(_tools(), _tools())
        assert changes == ToolListChanges()
        assert not changes.any_change

    def test_reports_added_removed_changed(self):
        new = _tools()
        new[0]["description"] = "Changed."
        new[1] = {"name": "archive", "description": "Archive an item."}
        changes = compare_tool_lists(_tools(), new)
        assert changes.added == ["archive"]
        assert changes.removed == ["delete"]
        assert changes.changed == ["search"]

    def test_description_only_change_keeps_names(self):
        new = _tools()
        new[0]["description"] = "Changed."
        changes = compare_tool_lists(_tools(), new)
        assert changes.any_change
        assert not changes.names_changed

    def test_first_discovery_reports_all_added(self):
        changes = compare_tool_lists([], _tools())
        assert changes.added == ["delete", "search"]
        assert changes.names_changed
