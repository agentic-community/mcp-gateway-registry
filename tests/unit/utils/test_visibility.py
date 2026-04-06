"""Unit tests for the shared visibility normalization utilities."""

import pytest

from registry.utils.visibility import (
    VALID_VISIBILITY_VALUES,
    _normalize_visibility,
    validate_visibility,
)


# ---------------------------------------------------------------------------
# _normalize_visibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeVisibility:
    """Tests for the _normalize_visibility helper."""

    def test_internal_normalized_to_private(self):
        """'internal' should be normalized to 'private'."""
        assert _normalize_visibility("internal") == "private"

    def test_group_normalized_to_group_restricted(self):
        """'group' should be normalized to 'group-restricted'."""
        assert _normalize_visibility("group") == "group-restricted"

    def test_public_unchanged(self):
        """'public' should remain 'public'."""
        assert _normalize_visibility("public") == "public"

    def test_private_unchanged(self):
        """'private' should remain 'private'."""
        assert _normalize_visibility("private") == "private"

    def test_group_restricted_unchanged(self):
        """'group-restricted' should remain 'group-restricted'."""
        assert _normalize_visibility("group-restricted") == "group-restricted"

    def test_case_insensitive_internal(self):
        """'Internal' (mixed case) should normalize to 'private'."""
        assert _normalize_visibility("Internal") == "private"

    def test_case_insensitive_public(self):
        """'PUBLIC' (uppercase) should normalize to 'public'."""
        assert _normalize_visibility("PUBLIC") == "public"

    def test_unknown_value_passed_through_lowered(self):
        """Unknown values are lowered but not aliased."""
        assert _normalize_visibility("CUSTOM") == "custom"


# ---------------------------------------------------------------------------
# validate_visibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateVisibility:
    """Tests for the validate_visibility function."""

    def test_all_canonical_values_accepted(self):
        """All three canonical visibility values should be accepted."""
        for value in VALID_VISIBILITY_VALUES:
            assert validate_visibility(value) == value

    def test_internal_alias_accepted(self):
        """'internal' should be accepted and normalized to 'private'."""
        assert validate_visibility("internal") == "private"

    def test_group_alias_accepted(self):
        """'group' should be accepted and normalized to 'group-restricted'."""
        assert validate_visibility("group") == "group-restricted"

    def test_case_insensitive(self):
        """Mixed case input should be accepted."""
        assert validate_visibility("INTERNAL") == "private"
        assert validate_visibility("Public") == "public"
        assert validate_visibility("GROUP") == "group-restricted"

    def test_invalid_value_rejected(self):
        """Invalid visibility values should raise ValueError."""
        with pytest.raises(ValueError, match="Visibility must be one of"):
            validate_visibility("secret")

    def test_empty_string_rejected(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Visibility must be one of"):
            validate_visibility("")

    def test_unknown_value_rejected(self):
        """Unknown value that isn't an alias should be rejected."""
        with pytest.raises(ValueError, match="Visibility must be one of"):
            validate_visibility("hidden")


# ---------------------------------------------------------------------------
# VALID_VISIBILITY_VALUES constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidVisibilityValues:
    """Tests for the VALID_VISIBILITY_VALUES constant."""

    def test_contains_three_values(self):
        """Constant should contain exactly three values."""
        assert len(VALID_VISIBILITY_VALUES) == 3

    def test_contains_expected_values(self):
        """Constant should contain public, private, and group-restricted."""
        assert "public" in VALID_VISIBILITY_VALUES
        assert "private" in VALID_VISIBILITY_VALUES
        assert "group-restricted" in VALID_VISIBILITY_VALUES

    def test_does_not_contain_internal(self):
        """Constant should NOT contain 'internal' (it's an alias, not canonical)."""
        assert "internal" not in VALID_VISIBILITY_VALUES
