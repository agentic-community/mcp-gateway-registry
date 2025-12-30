from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from ..schemas.agent_models import (
    AgentCard,
)


def _get_text_for_embedding(
    server_info: Dict[str, Any],
) -> str:
    """Prepare text string from server info (including tools) for embedding."""
    name = server_info.get("server_name", "")
    description = server_info.get("description", "")
    tags = server_info.get("tags", [])
    tag_string = ", ".join(tags)
    tool_list = server_info.get("tool_list") or []
    tool_snippets = []
    for tool in tool_list:
        tool_name = tool.get("name", "")
        parsed_description = tool.get("parsed_description", {}) or {}
        tool_desc = parsed_description.get("main") or tool.get("description", "")
        tool_args = parsed_description.get("args", "")
        snippet = f"Tool: {tool_name}. Description: {tool_desc}. Args: {tool_args}"
        tool_snippets.append(snippet.strip())

    tools_section = "\n".join(tool_snippets)
    return (
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"Tags: {tag_string}\n"
        f"Tools:\n{tools_section}"
    ).strip()


def _get_text_for_agent(
    agent_card: AgentCard,
) -> str:
    """Prepare text string from agent card for embedding."""
    name = agent_card.name
    description = agent_card.description

    skills_text = ""
    if agent_card.skills:
        skill_names = [skill.name for skill in agent_card.skills]
        skill_descriptions = [
            f"{skill.name}: {skill.description}"
            for skill in agent_card.skills
        ]
        skills_text = "Skills: " + ", ".join(skill_names)
        skills_text += "\nSkill Details: " + " | ".join(skill_descriptions)

    tags = agent_card.tags
    tag_string = ", ".join(tags) if tags else ""

    text_parts = [
        f"Name: {name}",
        f"Description: {description}",
    ]

    if skills_text:
        text_parts.append(skills_text)

    if tag_string:
        text_parts.append(f"Tags: {tag_string}")

    return "\n".join(text_parts)

