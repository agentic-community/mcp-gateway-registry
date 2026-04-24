"""Unit tests for registry.utils.metadata module."""

import pytest

from registry.utils.metadata import flatten_metadata_to_text


class TestFlattenMetadataToText:
    """Tests for the metadata flattening utility."""

    def test_simple_string_values(self):
        """Flat dict with string values produces key-value tokens."""
        metadata = {"team": "finance", "region": "us-east"}
        result = flatten_metadata_to_text(metadata)
        assert "team" in result
        assert "finance" in result
        assert "region" in result
        assert "us-east" in result

    def test_list_values_flattened(self):
        """List values are expanded into individual tokens."""
        metadata = {"langs": ["python", "go", "rust"]}
        result = flatten_metadata_to_text(metadata)
        assert "langs" in result
        assert "python" in result
        assert "go" in result
        assert "rust" in result

    def test_nested_dict_values_flattened(self):
        """Nested dict values are included."""
        metadata = {"contact": {"name": "Alice", "role": "lead"}}
        result = flatten_metadata_to_text(metadata)
        assert "contact" in result
        assert "Alice" in result
        assert "lead" in result

    def test_empty_dict_returns_empty_string(self):
        """Empty dict returns empty string."""
        assert flatten_metadata_to_text({}) == ""

    def test_none_returns_empty_string(self):
        """None input returns empty string."""
        assert flatten_metadata_to_text(None) == ""

    def test_non_dict_returns_empty_string(self):
        """Non-dict input returns empty string."""
        assert flatten_metadata_to_text("not a dict") == ""

    def test_numeric_values_converted_to_string(self):
        """Numeric values are converted to strings."""
        metadata = {"version": 3, "priority": 1.5}
        result = flatten_metadata_to_text(metadata)
        assert "version" in result
        assert "3" in result
        assert "priority" in result
        assert "1.5" in result

    def test_boolean_values_converted_to_string(self):
        """Boolean values are converted to strings."""
        metadata = {"active": True, "deprecated": False}
        result = flatten_metadata_to_text(metadata)
        assert "True" in result
        assert "False" in result

    def test_mixed_value_types(self):
        """Mixed value types all appear in output."""
        metadata = {
            "team": "platform",
            "tags": ["internal", "v2"],
            "config": {"timeout": 30},
            "priority": 1,
        }
        result = flatten_metadata_to_text(metadata)
        assert "team" in result
        assert "platform" in result
        assert "internal" in result
        assert "v2" in result
        assert "30" in result
        assert "priority" in result
        assert "1" in result
