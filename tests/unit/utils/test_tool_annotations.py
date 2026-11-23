"""MCP tool annotations, title and outputSchema: capture, validation, signature."""

import subprocess
import sys

import pytest
from mcp.types import (
    Tool,
    ToolAnnotations,
)

from registry.core.mcp_client import _extract_tool_details
from registry.utils.tool_annotations import (
    MAX_TOOL_TITLE_LENGTH,
    stored_tool_annotations,
    stored_tool_metadata,
    tool_metadata_fields,
    tool_metadata_signature,
)
from registry.utils.tool_schema import stored_tool_to_mcp

IMPORT_TIMEOUT_SECONDS: int = 60

OUTPUT_SCHEMA: dict = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _sdk_tool(
    payload: dict,
) -> Tool:
    """Build an SDK Tool the way the client parses tools/list (wire names)."""
    return Tool.model_validate({"name": "t", "inputSchema": {"type": "object"}, **payload})


@pytest.mark.unit
class TestToolMetadataFields:
    """tool_metadata_fields reads an SDK Tool."""

    def test_all_fields_captured_with_wire_names(self):
        tool = _sdk_tool(
            {
                "title": "Delete note",
                "annotations": {"readOnlyHint": False, "destructiveHint": True},
                "outputSchema": OUTPUT_SCHEMA,
            }
        )

        fields = tool_metadata_fields(tool)

        assert fields == {
            "title": "Delete note",
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
            "outputSchema": OUTPUT_SCHEMA,
        }

    def test_explicit_false_survives(self):
        tool = _sdk_tool({"annotations": {"readOnlyHint": False}})

        assert tool_metadata_fields(tool) == {"annotations": {"readOnlyHint": False}}

    def test_tool_without_fields_yields_nothing(self):
        assert tool_metadata_fields(_sdk_tool({})) == {}

    def test_null_hint_is_treated_as_absent(self):
        tool = _sdk_tool({"annotations": {"readOnlyHint": None, "idempotentHint": True}})

        assert tool_metadata_fields(tool) == {"annotations": {"idempotentHint": True}}

    def test_empty_annotations_object_is_not_stored(self):
        assert tool_metadata_fields(_sdk_tool({"annotations": {}})) == {}

    def test_annotation_title_kept(self):
        tool = _sdk_tool({"annotations": {"title": "Read notes", "readOnlyHint": True}})

        assert tool_metadata_fields(tool)["annotations"] == {
            "title": "Read notes",
            "readOnlyHint": True,
        }

    def test_annotations_built_by_python_name(self):
        tool = Tool(
            name="t",
            input_schema={"type": "object"},
            annotations=ToolAnnotations(open_world_hint=True),
        )

        assert tool_metadata_fields(tool) == {"annotations": {"openWorldHint": True}}

    def test_oversized_title_dropped(self):
        tool = _sdk_tool({"title": "x" * (MAX_TOOL_TITLE_LENGTH + 1)})

        assert tool_metadata_fields(tool) == {}


@pytest.mark.unit
class TestStoredToolAnnotations:
    """Stored values are validated before any reader forwards them."""

    def test_valid_annotations_returned(self):
        tool = {"name": "t", "annotations": {"readOnlyHint": True}}

        assert stored_tool_annotations(tool) == {"readOnlyHint": True}

    @pytest.mark.parametrize(
        "raw",
        [
            {"readOnlyHint": "true"},
            {"readOnlyHint": 1},
            {"destructiveHint": {"nested": True}},
            {"unknownHint": True},
            {"title": ""},
        ],
    )
    def test_wrong_types_and_unknown_keys_dropped(self, raw):
        assert stored_tool_annotations({"name": "t", "annotations": raw}) is None

    @pytest.mark.parametrize("raw", [["readOnlyHint"], "readOnlyHint", 5])
    def test_non_dict_annotations_rejected(self, raw):
        assert stored_tool_annotations({"name": "t", "annotations": raw}) is None

    def test_mixed_valid_and_invalid_keeps_valid(self):
        raw = {"readOnlyHint": True, "destructiveHint": "yes", "x": 1}

        assert stored_tool_annotations({"name": "t", "annotations": raw}) == {"readOnlyHint": True}

    def test_non_dict_tool(self):
        assert stored_tool_annotations("not a tool") is None
        assert stored_tool_metadata(None) == {}

    def test_stored_metadata_drops_non_dict_output_schema(self):
        tool = {"name": "t", "title": "T", "outputSchema": "not a schema"}

        assert stored_tool_metadata(tool) == {"title": "T"}


@pytest.mark.unit
class TestToolMetadataSignature:
    """The health check compares signatures to decide whether to rewrite."""

    def test_list_without_metadata_has_empty_signature(self):
        old_record = [{"name": "a", "description": "x", "schema": {}}]

        assert tool_metadata_signature(old_record) == []
        assert tool_metadata_signature(None) == []

    def test_backfill_changes_signature(self):
        stored = [{"name": "a"}]
        fetched = [{"name": "a", "annotations": {"readOnlyHint": True}}]

        assert tool_metadata_signature(stored) != tool_metadata_signature(fetched)

    def test_hint_flip_changes_signature(self):
        before = [{"name": "a", "annotations": {"destructiveHint": True}}]
        after = [{"name": "a", "annotations": {"destructiveHint": False}}]

        assert tool_metadata_signature(before) != tool_metadata_signature(after)

    def test_key_and_tool_order_do_not_matter(self):
        first = [
            {"name": "a", "annotations": {"readOnlyHint": True, "openWorldHint": False}},
            {"name": "b", "outputSchema": {"type": "object", "properties": {}}},
        ]
        second = [
            {"name": "b", "outputSchema": {"properties": {}, "type": "object"}},
            {"name": "a", "annotations": {"openWorldHint": False, "readOnlyHint": True}},
        ]

        assert tool_metadata_signature(first) == tool_metadata_signature(second)


@pytest.mark.unit
class TestExtractToolDetails:
    """_extract_tool_details stores the fields from real SDK Tool objects."""

    def test_three_tool_kinds_stay_distinct(self):
        tools = [
            _sdk_tool({"name": "read_notes", "annotations": {"readOnlyHint": True}}),
            _sdk_tool(
                {
                    "name": "delete_note",
                    "annotations": {"readOnlyHint": False, "destructiveHint": True},
                    "outputSchema": OUTPUT_SCHEMA,
                }
            ),
            _sdk_tool({"name": "ping"}),
        ]

        by_name = {tool["name"]: tool for tool in _extract_tool_details(tools)}

        assert by_name["read_notes"]["annotations"] == {"readOnlyHint": True}
        assert by_name["delete_note"]["annotations"] == {
            "readOnlyHint": False,
            "destructiveHint": True,
        }
        assert by_name["delete_note"]["outputSchema"] == OUTPUT_SCHEMA
        assert "annotations" not in by_name["ping"]
        assert "outputSchema" not in by_name["ping"]
        assert "title" not in by_name["ping"]

    def test_existing_fields_unchanged(self):
        result = _extract_tool_details([_sdk_tool({"name": "ping", "description": "Ping."})])

        assert set(result[0]) == {
            "name",
            "description",
            "parsed_description",
            "schema",
            "inputSchema",
        }


@pytest.mark.unit
class TestStoredToolToMcp:
    """The pre-consent tool list (issue #1986) keeps the fields."""

    def test_fields_copied(self):
        tool = {
            "name": "delete_note",
            "description": "Delete",
            "title": "Delete note",
            "annotations": {"destructiveHint": True},
            "outputSchema": OUTPUT_SCHEMA,
        }

        result = stored_tool_to_mcp(tool)

        assert result["annotations"] == {"destructiveHint": True}
        assert result["title"] == "Delete note"
        assert result["outputSchema"] == OUTPUT_SCHEMA

    def test_invalid_annotations_not_forwarded(self):
        tool = {"name": "t", "annotations": {"readOnlyHint": "true"}}

        assert "annotations" not in stored_tool_to_mcp(tool)

    def test_plain_tool_shape_unchanged(self):
        result = stored_tool_to_mcp({"name": "t", "description": "d"})

        assert set(result) == {"name", "description", "inputSchema"}


BLOCK_MCP_AND_IMPORT: str = """
import sys
sys.modules["mcp"] = None
import registry.utils.tool_schema
import registry.utils.tool_annotations
print("imported")
"""


@pytest.mark.unit
def test_helpers_import_without_mcp_sdk():
    """The auth-server image has no MCP SDK and imports tool_schema."""
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-authored code
        [sys.executable, "-c", BLOCK_MCP_AND_IMPORT],
        capture_output=True,
        text=True,
        timeout=IMPORT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "imported" in result.stdout
