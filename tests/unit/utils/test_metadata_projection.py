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
