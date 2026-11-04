"""Tests for _ensure_mcp_compliant_schema, which every virtual-server tool passes through.

MCP 2026-07-28 requires a tool's inputSchema root to have "type": "object".
The rewrite must produce that without moving a client's arguments or breaking
"#/$defs/..." references.
"""

from registry.core.nginx_service import _ensure_mcp_compliant_schema

ITEM_DEFS: dict = {"Item": {"type": "string"}}


def test_object_schema_is_unchanged():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    assert _ensure_mcp_compliant_schema(schema) is schema


def test_empty_schema_becomes_an_empty_object():
    assert _ensure_mcp_compliant_schema({}) == {"type": "object", "properties": {}}


def test_type_list_allowing_object_is_narrowed_not_nested():
    schema = {"type": ["object", "null"], "properties": {"q": {"type": "string"}}}

    result = _ensure_mcp_compliant_schema(schema)

    assert result == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert schema["type"] == ["object", "null"]  # the stored copy is not modified


def test_root_ref_without_type_keeps_its_reference():
    schema = {"$ref": "#/$defs/Item", "$defs": ITEM_DEFS}

    result = _ensure_mcp_compliant_schema(schema)

    assert result["type"] == "object"
    assert result["$ref"] == "#/$defs/Item"
    assert result["$defs"] == ITEM_DEFS


def test_wrapped_schema_moves_definitions_to_the_new_root():
    schema = {"type": "array", "items": {"$ref": "#/$defs/Item"}, "$defs": ITEM_DEFS}

    result = _ensure_mcp_compliant_schema(schema)

    assert result["type"] == "object"
    assert result["$defs"] == ITEM_DEFS
    assert result["properties"]["value"] == {"type": "array", "items": {"$ref": "#/$defs/Item"}}
    assert "$defs" in schema  # the stored copy is not modified


def test_wrapped_schema_moves_draft07_definitions_too():
    schema = {"type": "string", "definitions": ITEM_DEFS}

    result = _ensure_mcp_compliant_schema(schema)

    assert result["definitions"] == ITEM_DEFS
    assert "definitions" not in result["properties"]["value"]
