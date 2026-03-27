"""CLI script to sync Okta M2M clients to MongoDB.

This script connects to MongoDB and syncs all Okta M2M applications,
storing their client IDs and group mappings for authorization decisions.
"""

import asyncio
import logging
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient

# Add parent directory to path so we can import registry modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry.services.okta_m2m_sync import OktaM2MSync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    """Main function to sync Okta M2M clients."""
    # Get configuration from environment
    mongo_uri = os.getenv("DOCUMENTDB_URI", "mongodb://localhost:27017")
    mongo_db_name = os.getenv("DOCUMENTDB_DB_NAME", "mcp_registry")
    okta_domain = os.getenv("OKTA_DOMAIN")
    okta_api_token = os.getenv("OKTA_API_TOKEN")

    if not okta_domain or not okta_api_token:
        logger.error("ERROR: OKTA_DOMAIN and OKTA_API_TOKEN environment variables must be set")
        logger.error("Example:")
        logger.error("  export OKTA_DOMAIN=integrator-9917255.okta.com")
        logger.error("  export OKTA_API_TOKEN=your_api_token_here")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Okta M2M Client Sync")
    logger.info("=" * 60)
    logger.info(f"MongoDB URI: {mongo_uri}")
    logger.info(f"Database: {mongo_db_name}")
    logger.info(f"Okta Domain: {okta_domain}")
    logger.info("=" * 60)

    # Connect to MongoDB
    try:
        mongo_client = AsyncIOMotorClient(mongo_uri)
        db = mongo_client[mongo_db_name]

        # Test connection
        await db.command("ping")
        logger.info("✓ Connected to MongoDB")

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # Initialize Okta sync service
    try:
        okta_sync = OktaM2MSync(
            db=db,
            okta_domain=okta_domain,
            okta_api_token=okta_api_token,
        )

        # Perform sync
        logger.info("\nStarting sync from Okta...")
        result = await okta_sync.sync_from_okta(force_full_sync=True)

        logger.info("\n" + "=" * 60)
        logger.info("SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Added: {result['added_count']} clients")
        logger.info(f"Updated: {result['updated_count']} clients")
        logger.info(f"Total synced: {result['synced_count']} clients")

        if result["errors"]:
            logger.warning(f"\nErrors encountered: {len(result['errors'])}")
            for error in result["errors"]:
                logger.warning(f"  - {error}")

        # Display synced clients
        logger.info("\nSynced clients:")
        clients = await okta_sync.get_all_clients()
        for client in clients:
            logger.info(f"  - {client.name} (ID: {client.client_id}, Groups: {client.groups})")

        logger.info("\n✓ Sync successful!")

    except Exception as e:
        logger.exception(f"Sync failed: {e}")
        sys.exit(1)
    finally:
        mongo_client.close()


if __name__ == "__main__":
    asyncio.run(main())
