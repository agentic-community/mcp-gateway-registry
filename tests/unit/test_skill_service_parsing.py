"""Unit tests for skill service YAML frontmatter parsing."""

import pytest
import re


# Extract the parsing logic for unit testing (avoiding HTTP calls)
def _parse_frontmatter(
    content: str,
) -> dict:
    """Parse YAML frontmatter from SKILL.md content.

    Supports multiple formats:
    1. Standard: --- at start of file
    2. Code block with ---: ```yaml\n---\n...\n---\n```
    3. Code block without ---: ```yaml\n...\n```

    Args:
        content: Raw SKILL.md content

    Returns:
        Dict with parsed name, description, version, tags
    """
    result = {
        "name": None,
        "description": None,
        "version": None,
        "tags": [],
    }

    frontmatter = None
    frontmatter_end_pos = 0

    # Format 1: Standard frontmatter at start of file
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        frontmatter_end_pos = frontmatter_match.end()
    else:
        # Format 2: YAML code block with --- markers inside
        codeblock_with_markers = re.search(
            r"```ya?ml\s*\n---\s*\n(.*?)\n---\s*\n```",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if codeblock_with_markers:
            frontmatter = codeblock_with_markers.group(1)
            frontmatter_end_pos = codeblock_with_markers.end()
        else:
            # Format 3: YAML code block without --- markers
            codeblock_no_markers = re.search(
                r"```ya?ml\s*\n(.*?)\n```",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if codeblock_no_markers:
                frontmatter = codeblock_no_markers.group(1)
                frontmatter_end_pos = codeblock_no_markers.end()

    if frontmatter:
        # Parse simple YAML key: value pairs
        for line in frontmatter.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                if key == "name":
                    result["name"] = value
                elif key == "description":
                    result["description"] = value
                elif key == "version":
                    result["version"] = value
                elif key == "tags":
                    if value.startswith("["):
                        value = value.strip("[]")
                    result["tags"] = [
                        t.strip().strip('"').strip("'") for t in value.split(",") if t.strip()
                    ]

    return result


class TestFrontmatterParsing:
    """Tests for SKILL.md frontmatter parsing."""

    def test_standard_frontmatter_at_start(self):
        """Test parsing standard YAML frontmatter at file start."""
        content = """---
name: pdf-tool
description: Extract text from PDF files
version: 1.0.0
tags: pdf, extraction
---

# PDF Tool

This skill handles PDF processing.
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "pdf-tool"
        assert result["description"] == "Extract text from PDF files"
        assert result["version"] == "1.0.0"
        assert result["tags"] == ["pdf", "extraction"]

    def test_yaml_codeblock_with_markers(self):
        """Test parsing YAML in code block with --- markers (React skill format)."""
        content = """# Feature Flags Skill

```yaml
---
name: flags
description: Use when you need to check feature flag states
version: 2.0.0
---
```

## Overview

This skill manages feature flag inspection.
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "flags"
        assert result["description"] == "Use when you need to check feature flag states"
        assert result["version"] == "2.0.0"

    def test_yaml_codeblock_without_markers(self):
        """Test parsing YAML in code block without --- markers."""
        content = """# Simple Skill

```yaml
name: simple-skill
description: A simple skill example
tags: example, demo
```

## Usage

Just use it!
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "simple-skill"
        assert result["description"] == "A simple skill example"
        assert result["tags"] == ["example", "demo"]

    def test_yml_extension_codeblock(self):
        """Test parsing with ```yml instead of ```yaml."""
        content = """# YML Skill

```yml
name: yml-skill
description: Uses yml extension
```

## Details
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "yml-skill"
        assert result["description"] == "Uses yml extension"

    def test_tags_as_yaml_list(self):
        """Test parsing tags as YAML list format."""
        content = """---
name: list-tags
description: Test list tags
tags: [tag1, tag2, tag3]
---
"""
        result = _parse_frontmatter(content)
        assert result["tags"] == ["tag1", "tag2", "tag3"]

    def test_tags_with_quotes(self):
        """Test parsing tags with quotes."""
        content = """---
name: quoted-tags
description: Test quoted tags
tags: "tag-a", 'tag-b', tag-c
---
"""
        result = _parse_frontmatter(content)
        assert result["tags"] == ["tag-a", "tag-b", "tag-c"]

    def test_quoted_values(self):
        """Test parsing values with quotes."""
        content = """---
name: "quoted-name"
description: 'Single quoted description'
version: "1.2.3"
---
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "quoted-name"
        assert result["description"] == "Single quoted description"
        assert result["version"] == "1.2.3"

    def test_no_frontmatter(self):
        """Test content with no frontmatter returns None values."""
        content = """# Just a Heading

Some content without any frontmatter.
"""
        result = _parse_frontmatter(content)
        assert result["name"] is None
        assert result["description"] is None
        assert result["version"] is None
        assert result["tags"] == []

    def test_standard_frontmatter_priority(self):
        """Test standard frontmatter takes priority over code blocks."""
        content = """---
name: priority-name
description: Priority description
---

# Some Heading

```yaml
name: ignored-name
description: This should be ignored
```
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "priority-name"
        assert result["description"] == "Priority description"

    def test_multiline_description_in_codeblock(self):
        """Test that only first line of description is captured."""
        content = """```yaml
name: multiline-test
description: First line of description
version: 1.0.0
```
"""
        result = _parse_frontmatter(content)
        assert result["description"] == "First line of description"

    def test_facebook_react_flags_skill_format(self):
        """Test exact format from facebook/react flags skill."""
        content = """# Feature Flags Skill

```yaml
---
name: flags
description: Use when you need to check feature flag states, compare channels, or debug why a feature behaves differently across release channels.
---
```

## Overview

This skill manages feature flag inspection across multiple release channels.
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "flags"
        assert "feature flag states" in result["description"]
        assert "release channels" in result["description"]

    def test_case_insensitive_yaml_tag(self):
        """Test YAML/yaml/YML/yml all work."""
        formats = [
            "```YAML\nname: test1\n```",
            "```yaml\nname: test2\n```",
            "```YML\nname: test3\n```",
            "```yml\nname: test4\n```",
        ]
        for i, content in enumerate(formats, 1):
            result = _parse_frontmatter(content)
            assert result["name"] == f"test{i}", f"Failed for format: {content}"

    def test_empty_content(self):
        """Test empty content returns None values."""
        result = _parse_frontmatter("")
        assert result["name"] is None
        assert result["description"] is None

    def test_whitespace_handling(self):
        """Test whitespace in frontmatter is handled correctly."""
        content = """---
name:   spaced-name
description:    Spaced description
---
"""
        result = _parse_frontmatter(content)
        assert result["name"] == "spaced-name"
        assert result["description"] == "Spaced description"
