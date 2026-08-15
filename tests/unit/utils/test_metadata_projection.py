"""Unit tests for metadata field projection helpers (Issue #1277)."""

import pytest
from fastapi import HTTPException

from registry.utils.metadata import (
    build_metadata_set_stage,
    normalize_metadata_paths,
    parse_and_validate_metadata_fields,
    parse_metadata_fields,
    project_metadata,
)


class TestParseMetadataFields:
    """Tests for parse_metadata_fields()."""

    def test_none_returns_none(self) -> None:
        assert parse_metadata_fields(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_metadata_fields("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert parse_metadata_fields("   ") is None

    def test_single_field(self) -> None:
        assert parse_metadata_fields("owner") == ["owner"]

    def test_multiple_fields(self) -> None:
        result = parse_metadata_fields("owner,config.region,limits.rps")
        assert result == ["owner", "config.region", "limits.rps"]

    def test_strips_whitespace(self) -> None:
        result = parse_metadata_fields(" owner , config.region ")
        assert result == ["owner", "config.region"]

    def test_ignores_empty_segments_in_list(self) -> None:
        result = parse_metadata_fields("owner,,config")
        assert result == ["owner", "config"]

    def test_too_many_paths_raises(self) -> None:
        paths = ",".join(f"field{i}" for i in range(21))
        with pytest.raises(ValueError, match="Too many paths"):
            parse_metadata_fields(paths)

    def test_exactly_max_paths_ok(self) -> None:
        paths = ",".join(f"field{i}" for i in range(20))
        result = parse_metadata_fields(paths)
        assert len(result) == 20

    def test_dollar_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must not start with '\\$'"):
            parse_metadata_fields("$set.injection")

    def test_dollar_in_nested_raises(self) -> None:
        with pytest.raises(ValueError, match="must not start with '\\$'"):
            parse_metadata_fields("config.$unwind")

    def test_empty_segment_raises(self) -> None:
        with pytest.raises(ValueError, match="empty parts"):
            parse_metadata_fields("config..region")

    def test_path_too_deep_raises(self) -> None:
        with pytest.raises(ValueError, match="Path too deep"):
            parse_metadata_fields("a.b.c.d.e.f")

    def test_max_depth_ok(self) -> None:
        result = parse_metadata_fields("a.b.c.d.e")
        assert result == ["a.b.c.d.e"]

    def test_segment_too_long_raises(self) -> None:
        long_segment = "x" * 65
        with pytest.raises(ValueError, match="Path segment too long"):
            parse_metadata_fields(long_segment)

    def test_max_segment_length_ok(self) -> None:
        segment = "x" * 64
        result = parse_metadata_fields(segment)
        assert result == [segment]


class TestNormalizeMetadataPaths:
    """Tests for normalize_metadata_paths()."""

    def test_no_duplicates_unchanged(self) -> None:
        paths = ["owner", "config.region", "limits.rps"]
        assert normalize_metadata_paths(paths) == paths

    def test_ancestor_removes_descendant(self) -> None:
        result = normalize_metadata_paths(["config", "config.region"])
        assert result == ["config"]

    def test_descendant_before_ancestor(self) -> None:
        result = normalize_metadata_paths(["config.region", "config"])
        assert result == ["config"]

    def test_multiple_descendants_removed(self) -> None:
        result = normalize_metadata_paths(["config", "config.region", "config.tier"])
        assert result == ["config"]

    def test_unrelated_paths_kept(self) -> None:
        result = normalize_metadata_paths(["owner", "config.region"])
        assert result == ["owner", "config.region"]

    def test_exact_duplicate_removed(self) -> None:
        result = normalize_metadata_paths(["owner", "owner"])
        assert result == ["owner"]

    def test_deep_ancestor(self) -> None:
        result = normalize_metadata_paths(["a.b", "a.b.c.d"])
        assert result == ["a.b"]


class TestProjectMetadata:
    """Tests for project_metadata()."""

    def test_none_paths_returns_full_metadata(self) -> None:
        metadata = {"owner": "team", "config": {"region": "us-east-1"}}
        result = project_metadata(metadata, None)
        assert result == metadata

    def test_none_metadata_returns_none(self) -> None:
        assert project_metadata(None, ["owner"]) is None

    def test_empty_paths_returns_empty_dict(self) -> None:
        assert project_metadata({"owner": "team"}, []) == {}

    def test_single_top_level_key(self) -> None:
        metadata = {"owner": "team", "contact": "email@test.com"}
        result = project_metadata(metadata, ["owner"])
        assert result == {"owner": "team"}

    def test_multiple_top_level_keys(self) -> None:
        metadata = {"owner": "team", "contact": "email", "env": "prod"}
        result = project_metadata(metadata, ["owner", "env"])
        assert result == {"owner": "team", "env": "prod"}

    def test_nested_dot_path(self) -> None:
        metadata = {"config": {"region": "us-east-1", "tier": "prod"}}
        result = project_metadata(metadata, ["config.region"])
        assert result == {"config": {"region": "us-east-1"}}

    def test_multiple_nested_paths(self) -> None:
        metadata = {
            "owner": "team",
            "config": {"region": "us-east-1", "tier": "prod"},
            "limits": {"rps": 5000, "burst": 10000},
        }
        result = project_metadata(metadata, ["owner", "config.region", "limits.rps"])
        assert result == {
            "owner": "team",
            "config": {"region": "us-east-1"},
            "limits": {"rps": 5000},
        }

    def test_missing_path_produces_no_value(self) -> None:
        metadata = {"owner": "team"}
        result = project_metadata(metadata, ["nonexistent"])
        assert result == {}

    def test_missing_nested_path(self) -> None:
        metadata = {"config": {"region": "us-east-1"}}
        result = project_metadata(metadata, ["config.nonexistent"])
        assert result == {}

    def test_path_into_scalar(self) -> None:
        """Traversing into a scalar (owner.x where owner is a string) = missing."""
        metadata = {"owner": "team-platform"}
        result = project_metadata(metadata, ["owner.x"])
        assert result == {}

    def test_ancestor_path_returns_subtree(self) -> None:
        metadata = {"config": {"region": "us-east-1", "tier": "prod", "nested": {"deep": True}}}
        result = project_metadata(metadata, ["config"])
        assert result == {"config": {"region": "us-east-1", "tier": "prod", "nested": {"deep": True}}}

    def test_mixed_found_and_missing(self) -> None:
        metadata = {"owner": "team", "config": {"region": "us-east-1"}}
        result = project_metadata(metadata, ["owner", "nonexistent", "config.region"])
        assert result == {"owner": "team", "config": {"region": "us-east-1"}}

    def test_preserves_various_value_types(self) -> None:
        metadata = {
            "string": "hello",
            "number": 42,
            "boolean": True,
            "null_val": None,
            "array": [1, 2, 3],
            "nested": {"key": "val"},
        }
        result = project_metadata(metadata, ["string", "number", "boolean", "null_val", "array", "nested"])
        assert result == metadata

    def test_empty_metadata_with_paths(self) -> None:
        result = project_metadata({}, ["owner"])
        assert result == {}


class TestBuildMetadataSetStage:
    """Tests for build_metadata_set_stage()."""

    def test_empty_paths_produces_empty_metadata(self) -> None:
        result = build_metadata_set_stage([])
        assert result == {"$set": {"metadata": {}}}

    def test_single_top_level_field(self) -> None:
        result = build_metadata_set_stage(["owner"])
        assert result == {
            "$set": {
                "metadata": {
                    "owner": {"$ifNull": ["$metadata.owner", "$$REMOVE"]},
                }
            }
        }

    def test_nested_field(self) -> None:
        result = build_metadata_set_stage(["config.region"])
        assert result == {
            "$set": {
                "metadata": {
                    "config": {
                        "region": {"$ifNull": ["$metadata.config.region", "$$REMOVE"]},
                    }
                }
            }
        }

    def test_multiple_fields(self) -> None:
        result = build_metadata_set_stage(["owner", "config.region"])
        stage = result["$set"]["metadata"]
        assert "owner" in stage
        assert stage["owner"] == {"$ifNull": ["$metadata.owner", "$$REMOVE"]}
        assert "config" in stage
        assert stage["config"]["region"] == {"$ifNull": ["$metadata.config.region", "$$REMOVE"]}


class TestParseAndValidateMetadataFields:
    """Tests for the convenience wrapper that raises HTTPException."""

    def test_none_returns_none(self) -> None:
        assert parse_and_validate_metadata_fields(None) is None

    def test_valid_input_returns_normalized(self) -> None:
        result = parse_and_validate_metadata_fields("config,config.region")
        # normalize removes config.region since config is ancestor
        assert result == ["config"]

    def test_invalid_input_raises_422(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            parse_and_validate_metadata_fields("$set.attack")
        assert exc_info.value.status_code == 422
        assert "Invalid metadata_fields" in exc_info.value.detail


class TestParseMetadataFieldsEdgeCases:
    """Edge case tests for parse_metadata_fields()."""

    def test_empty_string_with_commas_only(self) -> None:
        """Commas with no content between them."""
        assert parse_metadata_fields(",,,") is None

    def test_single_character_field(self) -> None:
        result = parse_metadata_fields("a")
        assert result == ["a"]

    def test_hyphenated_field_names(self) -> None:
        result = parse_metadata_fields("my-field,another-one")
        assert result == ["my-field", "another-one"]

    def test_underscored_field_names(self) -> None:
        result = parse_metadata_fields("my_field,field_123")
        assert result == ["my_field", "field_123"]

    def test_numeric_segment(self) -> None:
        """Numeric segments are valid (could be array-like keys in a dict)."""
        result = parse_metadata_fields("items.0.name")
        assert result == ["items.0.name"]

    def test_unicode_field_names(self) -> None:
        result = parse_metadata_fields("café,région")
        assert result == ["café", "région"]

    def test_trailing_comma(self) -> None:
        result = parse_metadata_fields("owner,config,")
        assert result == ["owner", "config"]

    def test_leading_comma(self) -> None:
        result = parse_metadata_fields(",owner,config")
        assert result == ["owner", "config"]


class TestProjectMetadataEdgeCases:
    """Edge case tests for project_metadata()."""

    def test_deeply_nested_5_levels(self) -> None:
        """Project at maximum supported depth."""
        metadata = {"a": {"b": {"c": {"d": {"e": "deep_value"}}}}}
        result = project_metadata(metadata, ["a.b.c.d.e"])
        assert result == {"a": {"b": {"c": {"d": {"e": "deep_value"}}}}}

    def test_numeric_dict_keys(self) -> None:
        """Metadata with numeric string keys (e.g. from arrays stored as dicts)."""
        metadata = {"items": {"0": {"name": "first"}, "1": {"name": "second"}}}
        result = project_metadata(metadata, ["items.0.name"])
        assert result == {"items": {"0": {"name": "first"}}}

    def test_hyphenated_keys(self) -> None:
        metadata = {"my-field": "value", "other-field": "other"}
        result = project_metadata(metadata, ["my-field"])
        assert result == {"my-field": "value"}

    def test_metadata_with_none_values(self) -> None:
        """None values in metadata should be preserved when projected."""
        metadata = {"owner": None, "team": "platform"}
        result = project_metadata(metadata, ["owner", "team"])
        assert result == {"owner": None, "team": "platform"}

    def test_metadata_with_empty_dict_value(self) -> None:
        metadata = {"config": {}, "owner": "team"}
        result = project_metadata(metadata, ["config"])
        assert result == {"config": {}}

    def test_metadata_with_empty_list_value(self) -> None:
        metadata = {"tags": [], "owner": "team"}
        result = project_metadata(metadata, ["tags"])
        assert result == {"tags": []}

    def test_overlapping_paths_after_normalization(self) -> None:
        """After normalization removes descendants, projection still works."""
        metadata = {"config": {"region": "us-east-1", "tier": "prod"}}
        # normalize_metadata_paths(["config", "config.region"]) -> ["config"]
        paths = normalize_metadata_paths(["config", "config.region"])
        result = project_metadata(metadata, paths)
        assert result == {"config": {"region": "us-east-1", "tier": "prod"}}

    def test_large_metadata_many_keys(self) -> None:
        """Project one key from metadata with many keys."""
        metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = project_metadata(metadata, ["key_50"])
        assert result == {"key_50": "value_50"}


class TestBuildMetadataSetStageEquivalence:
    """Verify that build_metadata_set_stage produces structures that,
    when evaluated conceptually, would yield the same result as project_metadata.

    This pins the contract: both layers must agree on semantics."""

    def _simulate_set_stage(
        self, metadata: dict, stage: dict
    ) -> dict:
        """Simulate what MongoDB's $set would produce by resolving $ifNull refs."""
        set_expr = stage["$set"]["metadata"]
        return self._resolve_expr(set_expr, metadata)

    def _resolve_expr(self, expr: dict, metadata: dict) -> dict:
        """Recursively resolve a $set expression against source metadata."""
        result = {}
        for key, value in expr.items():
            if isinstance(value, dict) and "$ifNull" in value:
                # Leaf: resolve the path reference
                path_ref = value["$ifNull"][0]  # e.g. "$metadata.owner"
                path = path_ref.replace("$metadata.", "")
                resolved = self._resolve_path(metadata, path)
                if resolved is not None:
                    result[key] = resolved
            elif isinstance(value, dict):
                # Branch: recurse
                nested = self._resolve_expr(value, metadata)
                if nested:
                    result[key] = nested
            else:
                result[key] = value
        return result

    def _resolve_path(self, metadata: dict, path: str):
        """Navigate a dot-path into metadata, return None if missing."""
        current = metadata
        for segment in path.split("."):
            if not isinstance(current, dict) or segment not in current:
                return None
            current = current[segment]
        return current

    def test_equivalence_top_level(self) -> None:
        metadata = {"owner": "team", "contact": "email@test.com", "env": "prod"}
        paths = ["owner", "env"]
        normalized = normalize_metadata_paths(paths)

        python_result = project_metadata(metadata, normalized)
        stage = build_metadata_set_stage(normalized)
        db_result = self._simulate_set_stage(metadata, stage)

        assert python_result == db_result

    def test_equivalence_nested(self) -> None:
        metadata = {
            "config": {"region": "us-east-1", "tier": "prod"},
            "limits": {"rps": 5000},
        }
        paths = ["config.region", "limits.rps"]
        normalized = normalize_metadata_paths(paths)

        python_result = project_metadata(metadata, normalized)
        stage = build_metadata_set_stage(normalized)
        db_result = self._simulate_set_stage(metadata, stage)

        assert python_result == db_result

    def test_equivalence_missing_paths(self) -> None:
        metadata = {"owner": "team"}
        paths = ["owner", "nonexistent", "also.missing"]
        normalized = normalize_metadata_paths(paths)

        python_result = project_metadata(metadata, normalized)
        stage = build_metadata_set_stage(normalized)
        db_result = self._simulate_set_stage(metadata, stage)

        assert python_result == db_result

    def test_equivalence_ancestor_path(self) -> None:
        metadata = {"config": {"region": "us-east-1", "tier": "prod", "nested": {"deep": True}}}
        paths = ["config"]
        normalized = normalize_metadata_paths(paths)

        python_result = project_metadata(metadata, normalized)
        stage = build_metadata_set_stage(normalized)
        db_result = self._simulate_set_stage(metadata, stage)

        assert python_result == db_result
