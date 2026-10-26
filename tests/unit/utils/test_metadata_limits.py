"""Unit tests for registry.utils.metadata_limits (issue #1811).

Pins how the metadata size and nesting limits are measured, so every write path
that calls ``validate_metadata_limits`` agrees on where the boundary sits.
"""

import json

import pytest

from registry.utils.metadata_limits import (
    MAX_METADATA_BYTES,
    MAX_METADATA_DEPTH,
    MetadataLimitError,
    metadata_depth,
    metadata_serialized_size,
    validate_metadata_limits,
)

# '{"k":""}' is 8 bytes of compact JSON around the value.
_ENVELOPE_BYTES = 8


def _nested_dicts(depth: int) -> dict:
    """Return a dict nested exactly ``depth`` levels deep."""
    doc: dict = {"leaf": 1}
    for _ in range(depth - 1):
        doc = {"n": doc}
    return doc


def _nested_lists(depth: int) -> list:
    """Return a list nested exactly ``depth`` levels deep."""
    doc: list = [1]
    for _ in range(depth - 1):
        doc = [doc]
    return doc


@pytest.mark.unit
class TestLimitValues:
    def test_size_limit_matches_the_registry_card_cap(self):
        """10 KB is the cap registry_card.py already applies to its metadata."""
        assert MAX_METADATA_BYTES == 10240

    def test_depth_limit_is_eight(self):
        """Depth 5 would reject two existing server records (measured in #1811)."""
        assert MAX_METADATA_DEPTH == 8


@pytest.mark.unit
class TestSerializedSize:
    def test_counts_compact_json_without_whitespace(self):
        assert metadata_serialized_size({"a": 1, "b": [1, 2]}) == len('{"a":1,"b":[1,2]}')

    def test_none_is_zero_bytes(self):
        assert metadata_serialized_size(None) == 0

    @pytest.mark.parametrize(
        ("char", "utf8_width"),
        [("é", 2), ("中", 3), ("🔒", 4)],
        ids=["latin-2-bytes", "cjk-3-bytes", "emoji-4-bytes"],
    )
    def test_counts_real_utf8_width_not_escape_length(self, char, utf8_width):
        """A non-ASCII character is its UTF-8 width, not a 6-byte \\uXXXX escape."""
        assert metadata_serialized_size({"k": char}) == _ENVELOPE_BYTES + utf8_width


@pytest.mark.unit
class TestDepth:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("scalar", 0),
            ({}, 1),
            ({"a": 1}, 1),
            ({"a": {"b": 1}}, 2),
            ({"a": [{"b": 1}]}, 3),
            ({"wide": 1, "deep": {"x": {"y": 1}}}, 3),
        ],
        ids=["scalar", "empty", "flat", "nested-dict", "dict-list-dict", "uneven"],
    )
    def test_measures_the_deepest_branch(self, value, expected):
        assert metadata_depth(value) == expected

    def test_hostile_nesting_does_not_exhaust_the_stack(self):
        """The walk is iterative, so 50,000 levels is measured, not a RecursionError."""
        assert metadata_depth(_nested_lists(50_000)) == 50_000


@pytest.mark.unit
class TestValidateMetadataLimits:
    @pytest.mark.parametrize("value", [None, {}], ids=["none", "empty"])
    def test_missing_or_empty_metadata_passes(self, value):
        validate_metadata_limits(value)

    def test_exactly_the_byte_limit_passes(self):
        metadata = {"k": "x" * (MAX_METADATA_BYTES - _ENVELOPE_BYTES)}
        assert metadata_serialized_size(metadata) == MAX_METADATA_BYTES

        validate_metadata_limits(metadata)

    def test_one_byte_over_the_limit_is_rejected(self):
        metadata = {"k": "x" * (MAX_METADATA_BYTES - _ENVELOPE_BYTES + 1)}

        with pytest.raises(MetadataLimitError, match="10241 bytes .* limit of 10240 bytes"):
            validate_metadata_limits(metadata)

    def test_multibyte_characters_count_toward_the_byte_limit(self):
        """5,117 two-byte characters fit in 10,240 bytes; 5,117 plus one ASCII byte do not."""
        fits = {"k": "é" * ((MAX_METADATA_BYTES - _ENVELOPE_BYTES) // 2)}
        over = {"k": "é" * ((MAX_METADATA_BYTES - _ENVELOPE_BYTES) // 2) + "x"}

        validate_metadata_limits(fits)
        with pytest.raises(MetadataLimitError, match="10241 bytes"):
            validate_metadata_limits(over)

    def test_unicode_that_fits_as_utf8_is_not_rejected_for_its_escaped_length(self):
        """2,000 emoji are 8,008 UTF-8 bytes and pass.

        Measured the way registry_card.py measures (ASCII-escaped JSON, each
        emoji a 12-character surrogate pair), the same document is over 24,000
        characters and would be wrongly rejected.
        """
        metadata = {"k": "🔒" * 2000}
        assert len(json.dumps(metadata)) > MAX_METADATA_BYTES

        validate_metadata_limits(metadata)

    def test_exactly_eight_levels_of_dicts_passes(self):
        validate_metadata_limits(_nested_dicts(MAX_METADATA_DEPTH))

    def test_nine_levels_of_dicts_is_rejected(self):
        with pytest.raises(MetadataLimitError, match="nests 9 levels deep, over the limit of 8"):
            validate_metadata_limits(_nested_dicts(MAX_METADATA_DEPTH + 1))

    def test_lists_count_as_levels(self):
        """Hiding nesting in lists must not get around the depth limit."""
        validate_metadata_limits({"a": _nested_lists(MAX_METADATA_DEPTH - 1)})
        with pytest.raises(MetadataLimitError, match="nests 9 levels deep"):
            validate_metadata_limits({"a": _nested_lists(MAX_METADATA_DEPTH)})

    def test_mixed_dicts_and_lists(self):
        mixed = {"a": [{"b": [{"c": [{"d": [{"e": 1}]}]}]}]}
        assert metadata_depth(mixed) == 9

        with pytest.raises(MetadataLimitError, match="nests 9 levels deep"):
            validate_metadata_limits(mixed)

    def test_many_keys_are_allowed_within_the_byte_limit(self):
        """There is no key-count cap: 500 small keys are fine if they fit in bytes."""
        metadata = {f"k{i}": i for i in range(500)}
        assert metadata_serialized_size(metadata) < MAX_METADATA_BYTES

        validate_metadata_limits(metadata)

    def test_error_names_the_subject(self):
        with pytest.raises(MetadataLimitError, match="^skill metadata nests"):
            validate_metadata_limits(_nested_dicts(9), subject="skill metadata")

    def test_error_is_a_value_error_for_pydantic(self):
        """Pydantic turns a ValueError raised in a validator into a 422."""
        assert issubclass(MetadataLimitError, ValueError)
