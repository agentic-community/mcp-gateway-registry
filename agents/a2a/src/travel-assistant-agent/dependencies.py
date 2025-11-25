"""Dependency injection module for Travel Assistant Agent."""

import logging
from functools import lru_cache
from typing import Optional

from database import FlightDatabaseManager
from env_settings import EnvSettings
from registry_discovery_client import RegistryDiscoveryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Simple singleton providers
@lru_cache()
def get_env() -> EnvSettings:
    """Get environment settings singleton."""
    logger.debug("Getting environment settings")
    return EnvSettings()


@lru_cache()
def get_db_manager() -> FlightDatabaseManager:
    """Get database manager singleton."""
    env = get_env()
    logger.debug(f"Getting database manager with db_path: {env.db_path}")
    return FlightDatabaseManager(env.db_path)


@lru_cache()
def get_registry_client() -> Optional[RegistryDiscoveryClient]:
    """Get registry discovery client singleton.

    Returns:
        RegistryDiscoveryClient if configured, None otherwise
    """
    env = get_env()

    if not env.m2m_client_secret:
        logger.warning("M2M_CLIENT_SECRET not configured, discovery will not work")
        return None

    if not env.m2m_client_id:
        logger.warning("M2M_CLIENT_ID not configured, discovery will not work")
        return None

    logger.info("Creating RegistryDiscoveryClient")
    return RegistryDiscoveryClient(
        registry_url=env.mcp_registry_url,
        keycloak_url=env.keycloak_url,
        client_id=env.m2m_client_id,
        client_secret=env.m2m_client_secret,
        realm=env.keycloak_realm,
    )