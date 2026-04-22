"""Auth0 M2M Client Sync Service.

This service syncs M2M applications from Auth0 to MongoDB, allowing the registry
to track service accounts and their group mappings without hardcoding them in
authorization server expressions.
"""

import logging
import os
from datetime import datetime

import requests
from motor.motor_asyncio import AsyncIOMotorDatabase

from registry.schemas.idp_m2m_client import IdPM2MClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Auth0 client ID to groups mapping
# TODO: Make this configurable via database or config file
DEFAULT_CLIENT_GROUPS = {
    # Add Auth0 M2M client IDs and their default groups here
    # Example: "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd": ["registry-admins"],
}


class Auth0M2MSync:
    """Service for syncing Auth0 M2M applications to MongoDB."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        auth0_domain: str,
        m2m_client_id: str,
        m2m_client_secret: str,
    ):
        """Initialize Auth0 M2M sync service.

        Args:
            db: MongoDB database instance
            auth0_domain: Auth0 tenant domain (e.g., dev-abc123.us.auth0.com)
            m2m_client_id: Auth0 M2M client ID for Management API
            m2m_client_secret: Auth0 M2M client secret for Management API
        """
        self.db = db
        self.auth0_domain = auth0_domain.replace("https://", "").rstrip("/")
        self.m2m_client_id = m2m_client_id
        self.m2m_client_secret = m2m_client_secret
        self.collection = db["auth0_m2m_clients"]
        self.idp_collection = db["idp_m2m_clients"]

        logger.info(f"Initialized Auth0 M2M sync for domain: {self.auth0_domain}")

    async def _get_management_api_token(self) -> str:
        """Get Auth0 Management API access token.

        Returns:
            Access token string

        Raises:
            ValueError: If token request fails
        """
        token_url = f"https://{self.auth0_domain}/oauth/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": self.m2m_client_id,
            "client_secret": self.m2m_client_secret,
            "audience": f"https://{self.auth0_domain}/api/v2/",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            logger.debug(f"Requesting Management API token from {token_url}")
            response = requests.post(token_url, data=data, headers=headers, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            return token_data["access_token"]

        except requests.RequestException as e:
            logger.error(f"Failed to get Management API token: {e}")
            raise ValueError(f"Management API token request failed: {e}")

    async def _get_auth0_clients(self, access_token: str) -> list[dict]:
        """Fetch all clients from Auth0 Management API.

        Args:
            access_token: Auth0 Management API access token

        Returns:
            List of Auth0 client dictionaries

        Raises:
            ValueError: If Auth0 API request fails
        """
        url = f"https://{self.auth0_domain}/api/v2/clients"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        try:
            logger.info(f"Fetching clients from Auth0: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            clients = response.json()
            logger.info(f"Retrieved {len(clients)} clients from Auth0")
            return clients

        except requests.RequestException as e:
            logger.error(f"Failed to fetch Auth0 clients: {e}")
            raise ValueError(f"Auth0 API request failed: {e}")

    def _filter_m2m_clients(self, clients: list[dict]) -> list[dict]:
        """Filter to only M2M (non-interactive) clients.

        Args:
            clients: List of all Auth0 clients

        Returns:
            Filtered list of M2M clients
        """
        m2m_clients = []

        for client in clients:
            # M2M clients have app_type "non_interactive" or "machine_to_machine"
            app_type = client.get("app_type", "")
            name = client.get("name", "")
            client_id = client.get("client_id", "")

            if app_type in ["non_interactive", "machine_to_machine"]:
                logger.debug(f"Found M2M client: {name} (ID: {client_id})")
                m2m_clients.append(client)

        logger.info(f"Filtered to {len(m2m_clients)} M2M clients")
        return m2m_clients

    def _determine_groups(self, client_id: str) -> list[str]:
        """Determine groups for a client ID.

        This checks the hardcoded mapping (DEFAULT_CLIENT_GROUPS) to determine
        which groups a client should have. In the future, this could query a
        configuration table or use other logic.

        Args:
            client_id: Auth0 client ID

        Returns:
            List of group names for this client
        """
        groups = DEFAULT_CLIENT_GROUPS.get(client_id, [])
        logger.debug(f"Client {client_id} assigned groups: {groups}")
        return groups

    async def sync_from_auth0(self, force_full_sync: bool = False) -> dict:
        """Sync M2M clients from Auth0 to MongoDB.

        Args:
            force_full_sync: If True, update all clients. Otherwise incremental.

        Returns:
            Dictionary with sync statistics
        """
        logger.info(f"Starting Auth0 M2M sync (force_full_sync={force_full_sync})")

        added_count = 0
        updated_count = 0
        error_count = 0
        errors = []

        try:
            # Get Management API token
            access_token = await self._get_management_api_token()

            # Fetch all clients from Auth0
            all_clients = await self._get_auth0_clients(access_token)

            # Filter to M2M clients
            m2m_clients = self._filter_m2m_clients(all_clients)

            # Process each M2M client
            for client in m2m_clients:
                try:
                    client_id = client.get("client_id")

                    if not client_id:
                        logger.warning(f"Client {client.get('name')} has no client_id, skipping")
                        continue

                    # Check if client already exists in database
                    existing = await self.collection.find_one({"client_id": client_id})

                    # Determine groups for this client
                    groups = self._determine_groups(client_id)

                    # Auth0-specific collection document
                    client_doc = {
                        "client_id": client_id,
                        "name": client.get("name", client_id),
                        "description": client.get("description"),
                        "groups": groups,
                        "enabled": not client.get("is_first_party", False),
                        "auth0_client_id": client.get("client_id"),
                        "app_type": client.get("app_type"),
                        "last_synced": datetime.utcnow(),
                    }

                    if existing:
                        # Update existing record
                        client_doc["updated_at"] = datetime.utcnow()
                        await self.collection.update_one(
                            {"client_id": client_id}, {"$set": client_doc}
                        )
                        updated_count += 1
                        logger.info(f"Updated client: {client_id}")
                    else:
                        # Insert new record
                        client_doc["created_at"] = datetime.utcnow()
                        client_doc["updated_at"] = datetime.utcnow()
                        await self.collection.insert_one(client_doc)
                        added_count += 1
                        logger.info(f"Added new client: {client_id}")

                    # Also sync to generic idp_m2m_clients collection for groups enrichment
                    idp_doc = {
                        "client_id": client_id,
                        "name": client.get("name", client_id),
                        "description": client.get("description"),
                        "groups": groups,
                        "enabled": not client.get("is_first_party", False),
                        "provider": "auth0",
                        "idp_app_id": client.get("client_id"),
                        "updated_at": datetime.utcnow(),
                    }

                    existing_idp = await self.idp_collection.find_one({"client_id": client_id})
                    if existing_idp:
                        await self.idp_collection.update_one(
                            {"client_id": client_id}, {"$set": idp_doc}
                        )
                    else:
                        idp_doc["created_at"] = datetime.utcnow()
                        await self.idp_collection.insert_one(idp_doc)

                except Exception as e:
                    error_msg = f"Failed to process client {client.get('name')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    error_count += 1

            logger.info(
                f"Sync completed: {added_count} added, {updated_count} updated, "
                f"{error_count} errors"
            )

            return {
                "synced_count": added_count + updated_count,
                "added_count": added_count,
                "updated_count": updated_count,
                "removed_count": 0,
                "errors": errors,
            }

        except Exception as e:
            logger.exception(f"Auth0 sync failed: {e}")
            return {
                "synced_count": 0,
                "added_count": 0,
                "updated_count": 0,
                "removed_count": 0,
                "errors": [str(e)],
            }

    async def get_all_clients(self) -> list[IdPM2MClient]:
        """Get all M2M clients from MongoDB.

        Returns:
            List of IdPM2MClient objects
        """
        cursor = self.collection.find({})
        docs = await cursor.to_list(length=None)

        clients = []
        for doc in docs:
            try:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                # Convert to generic IdPM2MClient format
                client = IdPM2MClient(
                    client_id=doc["client_id"],
                    name=doc.get("name", doc["client_id"]),
                    description=doc.get("description"),
                    groups=doc.get("groups", []),
                    enabled=doc.get("enabled", True),
                    provider="auth0",
                    created_at=doc.get("created_at", datetime.utcnow()),
                    updated_at=doc.get("updated_at", datetime.utcnow()),
                    idp_app_id=doc.get("auth0_client_id"),
                )
                clients.append(client)
            except Exception as e:
                logger.warning(f"Failed to parse client document: {e}")

        return clients

    async def get_client_groups(self, client_id: str) -> list[str]:
        """Get groups for a specific client ID.

        Args:
            client_id: Auth0 client ID

        Returns:
            List of group names, empty if client not found
        """
        doc = await self.collection.find_one({"client_id": client_id})
        if doc:
            return doc.get("groups", [])
        return []

    async def update_client_groups(
        self,
        client_id: str,
        groups: list[str],
    ) -> bool:
        """Update groups for a specific client.

        Args:
            client_id: Auth0 client ID
            groups: New list of groups

        Returns:
            True if updated, False if client not found
        """
        # Update in Auth0-specific collection
        result = await self.collection.update_one(
            {"client_id": client_id},
            {
                "$set": {
                    "groups": groups,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        # Also update in generic idp_m2m_clients collection
        await self.idp_collection.update_one(
            {"client_id": client_id},
            {
                "$set": {
                    "groups": groups,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        if result.modified_count > 0:
            logger.info(f"Updated groups for client {client_id}: {groups}")
            return True

        logger.warning(f"Client {client_id} not found for update")
        return False


def get_auth0_m2m_sync(db: AsyncIOMotorDatabase) -> Auth0M2MSync | None:
    """Factory function to create Auth0M2MSync instance.

    Args:
        db: MongoDB database instance

    Returns:
        Auth0M2MSync instance if Auth0 is configured, None otherwise
    """
    auth0_domain = os.getenv("AUTH0_DOMAIN")
    m2m_client_id = os.getenv("AUTH0_M2M_CLIENT_ID")
    m2m_client_secret = os.getenv("AUTH0_M2M_CLIENT_SECRET")

    if not auth0_domain or not m2m_client_id or not m2m_client_secret:
        logger.warning(
            "Auth0 M2M sync not configured (missing AUTH0_DOMAIN, "
            "AUTH0_M2M_CLIENT_ID, or AUTH0_M2M_CLIENT_SECRET)"
        )
        return None

    return Auth0M2MSync(
        db=db,
        auth0_domain=auth0_domain,
        m2m_client_id=m2m_client_id,
        m2m_client_secret=m2m_client_secret,
    )
