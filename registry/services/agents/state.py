from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import (
    Dict,
    List,
)

logger = logging.getLogger(__name__)


def _load_state_file(
    state_file: Path,
) -> Dict[str, List[str]]:
    """Load agent state from disk.

    Returns:
        Dictionary with 'enabled' and 'disabled' lists.
    """
    logger.info("Loading agent state from %s...", state_file)

    try:
        if state_file.exists():
            with open(state_file, "r") as file_handle:
                state_data = json.load(file_handle)

            if not isinstance(state_data, dict):
                logger.warning("Invalid state format in %s", state_file)
                return {"enabled": [], "disabled": []}

            if "enabled" not in state_data:
                state_data["enabled"] = []
            if "disabled" not in state_data:
                state_data["disabled"] = []

            logger.info(
                "Loaded state: %s enabled, %s disabled",
                len(state_data["enabled"]),
                len(state_data["disabled"]),
            )
            return state_data

        logger.info("No state file found at %s, initializing empty state", state_file)
        return {"enabled": [], "disabled": []}

    except json.JSONDecodeError as exc:
        logger.error("Could not parse JSON from %s: %s", state_file, exc)
        return {"enabled": [], "disabled": []}
    except Exception as exc:
        logger.error("Failed to read state file %s: %s", state_file, exc, exc_info=True)
        return {"enabled": [], "disabled": []}


def _persist_state_to_disk(
    state_data: Dict[str, List[str]],
    state_file: Path,
) -> None:
    """Persist agent state to disk."""
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w") as file_handle:
            json.dump(state_data, file_handle, indent=2)

        logger.info("Persisted agent state to %s", state_file)

    except Exception as exc:
        logger.error("Failed to persist state to %s: %s", state_file, exc)

