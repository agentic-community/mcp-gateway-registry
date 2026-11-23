"""server-search prints the MCP annotation hints a server declared for each tool."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "api"))

import registry_management as rm  # noqa: E402


@pytest.mark.unit
class TestFormatToolHints:
    """_format_tool_hints turns annotations into one readable line."""

    def test_read_only_tool(self):
        annotations = {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }

        assert rm._format_tool_hints(annotations) == "read-only, idempotent"

    def test_destructive_tool(self):
        annotations = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}

        assert rm._format_tool_hints(annotations) == "not read-only, destructive, open-world"

    def test_no_annotations(self):
        assert rm._format_tool_hints(None) == ""
        assert rm._format_tool_hints({}) == ""

    def test_absent_read_only_hint_is_not_reported(self):
        assert rm._format_tool_hints({"idempotentHint": True}) == "idempotent"
