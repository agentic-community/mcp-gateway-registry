from unittest.mock import patch

import httpx
import pytest

from registry.services.skill_service import (
    _load_skill_frontmatter,
    _parse_skill_md_content,
)


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


async def _parse_document(document: str) -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=document))
    with patch(
        "registry.services.skill_service.guarded_async_client",
        return_value=httpx.AsyncClient(transport=transport),
    ):
        return await _parse_skill_md_content("https://8.8.8.8/SKILL.md")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("opening", "closing", "tags"),
    [
        ("---\n", "\n---", "[alpha, 'beta, gamma', [], ' ']"),
        ("```YML\n---\n", "\n---\n```", "[alpha, 'beta, gamma']"),
        ("```yaml\n", "\n```", "alpha, beta, , gamma"),
    ],
)
async def test_document_metadata(opening: str, closing: str, tags: str) -> None:
    document = (
        f"{opening}NAME: Sample Skill\nversion: 01\ndescription: |-\n"
        f"  first\n  second\ntags: {tags}{closing}\n\n# Body Title\n\nBody text.\n"
    )

    result = await _parse_document(document)

    assert result["name_slug"] == "sample-skill"
    assert result["description"] == "first\nsecond"
    assert result["version"] == "01"
    assert result["tags"] == (
        ["alpha", "beta", "gamma"] if tags.startswith("alpha") else ["alpha", "beta, gamma"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    ["[name, description]", "name: [unfinished", "name: [nested]\ndescription: {nested: value}"],
)
async def test_invalid_metadata_falls_back_to_body(metadata: str) -> None:
    result = await _parse_document(f"---\n{metadata}\n---\n\n# Body Title\n\nBody text.\n")

    assert (result["name"], result["description"]) == ("Body Title", "Body text.")
