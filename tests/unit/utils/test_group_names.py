"""Unit tests for registry.utils.group_names.

IdP group identifiers are compared against scope ``group_mappings`` in
canonical form. Keycloak's Group Membership mapper emits full paths
(``/mcp-admins``) when *Full group path* is enabled, and such a claim matched
nothing before normalisation existed (issue #1689).
"""

import pytest

from registry.utils.group_names import (
    group_name_variants,
    groups_intersect,
    normalize_group_name,
    normalize_group_names,
)

pytestmark = [pytest.mark.unit]


class TestNormalizeGroupName:
    """Canonical form of a single identifier."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("mcp-admins", "mcp-admins"),
            ("/mcp-admins", "mcp-admins"),
            ("mcp-admins/", "mcp-admins"),
            ("/mcp-admins/", "mcp-admins"),
            ("  mcp-admins  ", "mcp-admins"),
            (" / mcp-admins / ", "mcp-admins"),
        ],
    )
    def test_strips_surrounding_slashes_and_whitespace(self, raw: str, expected: str):
        assert normalize_group_name(raw) == expected

    def test_nested_path_keeps_inner_segments(self):
        """``/parent/child`` must not collapse to ``child``.

        Two nested groups can share a leaf name; reducing to the leaf would
        grant one group's scopes to the other.
        """
        assert normalize_group_name("/parent/child") == "parent/child"

    def test_case_is_preserved(self):
        """Keycloak group names are case-sensitive; Entra Object IDs are opaque."""
        assert normalize_group_name("/MCP-Admins") == "MCP-Admins"

    def test_entra_object_id_is_unchanged(self):
        guid = "138989a5-0510-442b-b909-c02d7b8c580a"
        assert normalize_group_name(guid) == guid

    @pytest.mark.parametrize("raw", ["", "/", "//", "   ", " / "])
    def test_empty_forms_become_empty(self, raw: str):
        assert normalize_group_name(raw) == ""

    def test_non_string_becomes_empty(self):
        assert normalize_group_name(None) == ""  # type: ignore[arg-type]


class TestNormalizeGroupNames:
    """Canonical list form."""

    def test_preserves_first_seen_order(self):
        assert normalize_group_names(["b", "a", "c"]) == ["b", "a", "c"]

    def test_collapses_duplicates_that_differ_only_by_slashes(self):
        assert normalize_group_names(["/team", "team", "team/"]) == ["team"]

    def test_drops_empties(self):
        assert normalize_group_names(["", "/", "team", None]) == ["team"]  # type: ignore[list-item]

    def test_empty_and_none_input(self):
        assert normalize_group_names([]) == []
        assert normalize_group_names(None) == []  # type: ignore[arg-type]


class TestGroupNameVariants:
    """Query-side forms for matching stored mappings in either shape."""

    def test_bare_name_also_yields_full_path_form(self):
        """A bare claim must find a mapping stored as ``/team``."""
        assert group_name_variants(["team"]) == ["/team", "team"]

    def test_nested_path_yields_full_path_form(self):
        assert group_name_variants(["parent/child"]) == ["/parent/child", "parent/child"]

    def test_full_path_yields_raw_and_canonical(self):
        assert group_name_variants(["/team"]) == ["/team", "team"]

    def test_sorted_and_deduplicated(self):
        assert group_name_variants(["/b", "a", "b", "/a"]) == ["/a", "/b", "a", "b"]

    def test_empties_removed(self):
        assert group_name_variants(["", "/", None, "x"]) == ["/x", "x"]  # type: ignore[list-item]


class TestGroupsIntersect:
    """Group-restricted visibility compares canonical names on both sides."""

    def test_full_path_allowed_group_matches_bare_session_group(self):
        """The login filter stores ``team``; an operator may have written ``/team``."""
        assert groups_intersect(["team"], ["/team"]) is True

    def test_bare_allowed_group_matches_full_path_claim(self):
        assert groups_intersect(["/team"], ["team"]) is True

    def test_disjoint_groups_do_not_match(self):
        assert groups_intersect(["team"], ["/other"]) is False

    def test_nested_path_does_not_match_its_leaf(self):
        """``/org/team`` is a different group from ``team``."""
        assert groups_intersect(["team"], ["/org/team"]) is False

    def test_nested_path_matches_its_own_canonical_form(self):
        assert groups_intersect(["org/team"], ["/org/team"]) is True

    def test_case_must_match(self):
        assert groups_intersect(["Team"], ["team"]) is False

    def test_blank_entries_never_match(self):
        """``""`` and ``"/"`` canonicalise to nothing and grant nothing."""
        assert groups_intersect(["", "/"], ["/", " "]) is False

    def test_none_and_empty_inputs_deny(self):
        assert groups_intersect(None, ["team"]) is False
        assert groups_intersect(["team"], None) is False
        assert groups_intersect([], []) is False

    def test_accepts_sets(self):
        assert groups_intersect({"team"}, {"/team"}) is True
