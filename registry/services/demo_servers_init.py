"""Initialize built-in demo servers on registry startup.

This module automatically registers essential demo servers directly into the
database during registry startup, eliminating the need for external registration
scripts and authentication tokens.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _load_server_config(config_path: str) -> dict[str, Any]:
    """Load server configuration from JSON file.

    Args:
        config_path: Relative path to config file from project root (/app in container)

    Returns:
        Server configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Get project root (3 levels up from this file: registry/services -> registry -> app)
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / config_path

    logger.debug(f"Loading server config from: {full_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"Server config not found: {full_path}")

    with open(full_path) as f:
        return json.load(f)


async def initialize_airegistry_server() -> bool:
    """Initialize AI Registry Tools server on startup.

    This function registers the airegistry-tools server (mcpgw) directly into
    the database, making it immediately available after deployment without
    requiring external registration scripts.

    The server provides essential registry management tools including:
    - Semantic search across all registered servers
    - Server discovery and listing
    - Intelligent tool finder

    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        logger.info("Initializing AI Registry Tools server...")

        # Load configuration from file
        config = _load_server_config("cli/examples/airegistry.json")

        # Get server repository (works with any backend: DocumentDB, MongoDB, or file)
        from registry.repositories.factory import get_server_repository

        server_repo = get_server_repository()

        # Check if server already exists
        path = config["path"]
        existing = await server_repo.get(path)

        if existing:
            logger.info(f"AI Registry Tools server already exists at {path}, updating...")

            # Update with new configuration
            config["updated_at"] = datetime.utcnow().isoformat()
            config["is_enabled"] = True  # Ensure it's enabled

            success = await server_repo.update(path, config)
            if success:
                logger.info(f"✅ AI Registry Tools server updated at {path}")
            else:
                logger.error(f"Failed to update AI Registry Tools server at {path}")
                return False
        else:
            logger.info(f"Creating AI Registry Tools server at {path}...")

            # Set metadata for new server
            config["registered_at"] = datetime.utcnow().isoformat()
            config["updated_at"] = datetime.utcnow().isoformat()
            config["is_enabled"] = True  # Enable by default
            config["source"] = "builtin"  # Mark as built-in server

            # Ensure UUID id field exists
            if "id" not in config or not config["id"]:
                config["id"] = str(uuid4())

            success = await server_repo.create(config)
            if success:
                logger.info(f"✅ AI Registry Tools server created at {path}")
            else:
                logger.error(f"Failed to create AI Registry Tools server at {path}")
                return False

        # Trigger immediate health check and security scan for the registered server
        logger.info(f"Triggering health check and security scan for {path}...")
        from registry.health.service import health_service
        from registry.services.security_scanner import security_scanner_service

        # Trigger health check asynchronously
        asyncio.create_task(health_service.perform_immediate_health_check(path))

        # Trigger security scan asynchronously
        proxy_pass_url = config.get("proxy_pass_url")
        if proxy_pass_url:
            asyncio.create_task(
                security_scanner_service.scan_server(server_url=proxy_pass_url, server_path=path)
            )
            logger.info(f"Security scan scheduled for {path}")
        else:
            logger.warning(f"No proxy_pass_url found for {path}, skipping security scan")

        return True

    except FileNotFoundError as e:
        logger.error(f"Server config file not found: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in server config: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize AI Registry Tools server: {e}", exc_info=True)
        return False


async def initialize_demo_servers() -> None:
    """Initialize all built-in demo servers on registry startup.

    This is called during FastAPI lifespan initialization to ensure demo
    servers are available immediately after deployment.

    Currently initializes:
    - AI Registry Tools (mcpgw server) at /airegistry-tools/
    """
    logger.info("🔧 Initializing built-in demo servers...")

    # Initialize AI Registry Tools
    success = await initialize_airegistry_server()

    if success:
        logger.info("✅ Built-in demo servers initialized successfully")
    else:
        logger.warning("⚠️ Failed to initialize some demo servers (registry will continue)")
