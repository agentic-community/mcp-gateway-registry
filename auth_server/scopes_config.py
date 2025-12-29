from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_scopes_config() -> dict[str, Any]:
    """Load the scopes configuration from scopes.yml."""
    try:
        scopes_path = os.environ.get("SCOPES_CONFIG_PATH")
        if scopes_path:
            scopes_file = Path(scopes_path)
        else:
            scopes_file = Path(__file__).parent / "scopes.yml"

        config = yaml.safe_load(scopes_file.read_text()) or {}
        logger.info(
            "Loaded scopes configuration from %s with %s group mappings",
            scopes_file,
            len(config.get("group_mappings", {})),
        )
        return config
    except Exception as exc:
        logger.error("Failed to load scopes configuration: %s", exc)
        return {}


SCOPES_CONFIG: dict[str, Any] = load_scopes_config()


def get_scopes_config() -> dict[str, Any]:
    return SCOPES_CONFIG


def reload_scopes_config() -> dict[str, Any]:
    global SCOPES_CONFIG
    SCOPES_CONFIG = load_scopes_config()
    return SCOPES_CONFIG

