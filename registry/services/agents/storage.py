from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
)

from ...schemas.agent_models import (
    AgentCard,
)

logger = logging.getLogger(__name__)


def _path_to_filename(
    path: str,
) -> str:
    """Convert agent path to safe filename."""
    normalized = path.lstrip("/").replace("/", "_")
    if not normalized.endswith("_agent.json"):
        if normalized.endswith(".json"):
            normalized = normalized.replace(".json", "_agent.json")
        else:
            normalized += "_agent.json"
    return normalized


def _load_agent_from_file(
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    """Load agent card from JSON file."""
    try:
        with open(file_path, "r") as file_handle:
            agent_data = json.load(file_handle)

        if not isinstance(agent_data, dict):
            logger.warning("Invalid agent data format in %s", file_path)
            return None

        if "path" not in agent_data or "name" not in agent_data:
            logger.warning("Missing required fields in %s", file_path)
            return None

        return agent_data

    except FileNotFoundError:
        logger.error("Agent file not found: %s", file_path)
        return None
    except json.JSONDecodeError as exc:
        logger.error("Could not parse JSON from %s: %s", file_path, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error loading %s: %s", file_path, exc, exc_info=True)
        return None


def _save_agent_to_disk(
    agent_card: AgentCard,
    agents_dir: Path,
) -> bool:
    """Save agent card to individual JSON file."""
    try:
        agents_dir.mkdir(parents=True, exist_ok=True)

        filename = _path_to_filename(agent_card.path)
        file_path = agents_dir / filename

        agent_dict = agent_card.model_dump(mode="json")

        with open(file_path, "w") as file_handle:
            json.dump(agent_dict, file_handle, indent=2)

        logger.info("Successfully saved agent '%s' to %s", agent_card.name, file_path)
        return True

    except Exception as exc:
        logger.error(
            "Failed to save agent '%s' to disk: %s",
            agent_card.name,
            exc,
            exc_info=True,
        )
        return False

