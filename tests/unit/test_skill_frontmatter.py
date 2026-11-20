import pytest

from registry.services.skill_service import _load_skill_frontmatter


@pytest.mark.parametrize(
    ("style", "expected"),
    [(" >", "first second\n"), (" |", "first\nsecond\n"), (" |-", "first\nsecond")],
)
def test_multiline_description(style: str, expected: str) -> None:
    assert (
        _load_skill_frontmatter(f"description:{style}\n  first\n  second")["description"]
        == expected
    )


@pytest.mark.parametrize(
    "source",
    [
        r'description: "\uD800"',
        "description: !custom text",
        "name: &x text\ndescription: *x",
        "name: " + "[" * 33 + "]" * 33,
        "name: " + "x" * 65536,
    ],
)
def test_unsafe_frontmatter_is_ignored(source: str) -> None:
    assert _load_skill_frontmatter(source) == {}
