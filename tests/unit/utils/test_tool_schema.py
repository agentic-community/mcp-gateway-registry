"""Tests for reading a stored tool's input schema under any of its keys."""

import pytest

from registry.utils.tool_schema import stored_input_schema

SCHEMA: dict = {"type": "object", "properties": {"q": {"type": "string"}}}


@pytest.mark.parametrize("key", ["inputSchema", "input_schema", "schema"])
def test_reads_each_key(key):
    assert stored_input_schema({"name": "t", key: SCHEMA}) == SCHEMA


def test_mcp_key_wins_over_the_discovery_key():
    other = {"type": "object"}
    assert stored_input_schema({"inputSchema": SCHEMA, "schema": other}) == SCHEMA


def test_empty_mcp_key_falls_back_to_the_discovery_key():
    """A registration may carry an empty inputSchema next to a discovered schema."""
    assert stored_input_schema({"inputSchema": {}, "schema": SCHEMA}) == SCHEMA


@pytest.mark.parametrize("tool", [{}, {"schema": None}, {"inputSchema": "not a dict"}])
def test_missing_or_invalid_schema_is_empty(tool):
    assert stored_input_schema(tool) == {}
