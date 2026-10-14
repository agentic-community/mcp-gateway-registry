"""Tests for registry.utils.path_utils."""

import pytest

from registry.utils.path_utils import make_version_id


@pytest.mark.unit
class TestMakeVersionId:
    """The version-document id format has one definition.

    It was built with an f-string in seven places across three modules before
    this helper existed. PR #1869 fixed four separate defects that were all two
    components disagreeing about an identifier's canonical form, so the format
    itself is worth centralizing.
    """

    def test_appends_the_version_suffix(self):
        assert make_version_id("/github", "v2") == "/github:v2"

    def test_is_idempotent_for_a_path_that_already_carries_the_suffix(self):
        """A caller that cannot tell an active path from a version id is safe.

        Replaces a hand-rolled ``path.endswith(f":{version}")`` guard in
        nginx_service, which was the only site that got this right.
        """
        assert make_version_id("/github:v2", "v2") == "/github:v2"

    def test_a_nested_path_keeps_its_slashes(self):
        assert make_version_id("/peer/github", "v1.0.0") == "/peer/github:v1.0.0"

    def test_a_different_version_suffix_is_still_appended(self):
        """Only this version's own suffix is treated as already present."""
        assert make_version_id("/github:v1", "v2") == "/github:v1:v2"
