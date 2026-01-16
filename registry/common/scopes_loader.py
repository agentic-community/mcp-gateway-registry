"""
Shared scopes loader module for loading authorization scopes from repository.

This module is used by both the auth server and registry to load scopes
from either DocumentDB or YAML file backends.
"""

import asyncio
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


logger = logging.getLogger(__name__)


async def load_scopes_from_repository(
    max_retries: int = 5,
    initial_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Load scopes configuration from repository with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (exponential backoff)

    Returns:
        Dict with "group_mappings", scope definitions, and "UI-Scopes"
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Import here to avoid circular dependencies
            from ..core.config import settings
            from ..repositories.factory import get_scope_repository

            if attempt == 0:
                logger.info(
                    f"Repository settings - backend: {settings.storage_backend}"
                )

            scope_repo = get_scope_repository()

            # Load all scopes
            await scope_repo.load_all()

            # Get all groups and build scopes configuration
            groups_dict = await scope_repo.list_groups()

            group_mappings = {}
            scopes_config = {}
            ui_scopes = {}

            # Build scopes config from repository
            for group_name in groups_dict.keys():
                # Get full group details
                group_data = await scope_repo.get_group(group_name)
                if not group_data:
                    continue

                # Group mappings: Keycloak group → list of scope names
                keycloak_groups = group_data.get("group_mappings", [])
                for keycloak_group in keycloak_groups:
                    if keycloak_group not in group_mappings:
                        group_mappings[keycloak_group] = []
                    if group_name not in group_mappings[keycloak_group]:
                        group_mappings[keycloak_group].append(group_name)

                # Server access scopes: scope_name → server_access list
                server_access = group_data.get("server_access", [])
                if server_access:
                    scopes_config[group_name] = server_access

                # UI permissions: scope_name → ui_permissions dict
                ui_permissions = group_data.get("ui_permissions", {})
                if ui_permissions:
                    ui_scopes[group_name] = ui_permissions

            logger.info(
                f"Loaded from repository: {len(group_mappings)} group mappings, "
                f"{len(scopes_config)} scope definitions, {len(ui_scopes)} UI scopes"
            )

            # Build the complete config structure
            config = {"group_mappings": group_mappings, "UI-Scopes": ui_scopes}
            config.update(scopes_config)

            return config

        except (ConnectionRefusedError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    f"Repository not ready (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Failed to connect to repository after {max_retries} attempts: {e}",
                    exc_info=True
                )
        except Exception as e:
            # Other exceptions should also be retried (might be transient repository errors)
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    f"Error loading scopes (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Failed to load scopes after {max_retries} attempts: {e}",
                    exc_info=True
                )

    # If we get here, all retries failed
    logger.error("Returning empty scopes configuration due to failures")
    return {"group_mappings": {}}


def load_scopes_from_yaml(scopes_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load scopes configuration from YAML file.

    Args:
        scopes_path: Optional path to scopes.yml file

    Returns:
        Dict with scopes configuration
    """
    try:
        if scopes_path:
            scopes_file = Path(scopes_path)
        else:
            # Default to auth_server/scopes.yml
            scopes_file = Path(__file__).parent.parent.parent / "auth_server" / "scopes.yml"

        # Check alternative location (EFS mount)
        if not scopes_file.exists():
            alt_scopes_file = Path(__file__).parent.parent.parent / "auth_server" / "auth_config" / "scopes.yml"
            if alt_scopes_file.exists():
                scopes_file = alt_scopes_file

        if not scopes_file.exists():
            logger.warning(f"Scopes config file not found at {scopes_file}")
            return {"group_mappings": {}}

        with open(scopes_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(
                f"Loaded scopes from YAML with "
                f"{len(config.get('group_mappings', {}))} group mappings"
            )
            return config

    except Exception as e:
        logger.error(f"Failed to load scopes from YAML: {e}")
        return {"group_mappings": {}}


async def reload_scopes_config(storage_backend: Optional[str] = None) -> Dict[str, Any]:
    """
    Reload scopes configuration from configured backend (async version).

    Args:
        storage_backend: Override storage backend (defaults to env var)

    Returns:
        Dict with scopes configuration
    """
    if storage_backend is None:
        from ..core.config import settings
        storage_backend = settings.storage_backend

    logger.info(f"Reloading scopes with storage backend: {storage_backend}")

    if storage_backend in ("documentdb", "mongodb-ce"):
        return await load_scopes_from_repository()
    else:
        # For file backend, also load into the repository so get_ui_scopes works
        from ..repositories.factory import get_scope_repository
        scope_repo = get_scope_repository()
        await scope_repo.load_all()

        return load_scopes_from_yaml(os.getenv("SCOPES_CONFIG_PATH"))
