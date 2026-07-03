"""Okta M2M Client Sync Service.

This service syncs M2M applications from Okta to MongoDB, allowing the registry
to track service accounts and their group mappings without hardcoding them in
authorization server expressions.
"""

import json
import logging
import os
from datetime import datetime

import requests
from motor.motor_asyncio import AsyncIOMotorDatabase

from registry.schemas.okta_m2m_client import OktaM2MClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Environment variable holding the operator-supplied Okta M2M client-id -> groups
# mapping. Format: a JSON object, e.g.
#   {"0oa1100req1AzfKaY698": ["public-mcp-users"]}
# There is intentionally NO hardcoded/default mapping: a client ID that is not
# explicitly listed here is synced with NO group memberships (fail closed). This
# prevents an attacker who can influence the Okta API response, point OKTA_DOMAIN
# at a rogue tenant, or write to MongoDB from having an injected client_id
# auto-granted privileged (e.g. admin) RBAC via a code-shipped mapping.
OKTA_M2M_CLIENT_GROUPS_ENV = "OKTA_M2M_CLIENT_GROUPS"


def _load_client_groups_mapping() -> dict[str, list[str]]:
    """Load the Okta M2M client-id -> groups mapping from the environment.

    Reads ``OKTA_M2M_CLIENT_GROUPS`` as a JSON object mapping each client_id to
    a list of group names. Fails closed: a malformed or unset value yields an
    empty mapping (no client is auto-assigned any group). Entries whose value is
    not a list of strings are dropped with a warning rather than trusted.

    Returns:
        Mapping of client_id to the list of group names to assign.
    """
    raw = os.environ.get(OKTA_M2M_CLIENT_GROUPS_ENV, "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.error(
            "%s is not valid JSON; no M2M client groups will be assigned: %s",
            OKTA_M2M_CLIENT_GROUPS_ENV,
            exc,
        )
        return {}

    if not isinstance(parsed, dict):
        logger.error(
            "%s must be a JSON object of client_id -> [groups]; got %s. "
            "No M2M client groups will be assigned.",
            OKTA_M2M_CLIENT_GROUPS_ENV,
            type(parsed).__name__,
        )
        return {}

    mapping: dict[str, list[str]] = {}
    for client_id, groups in parsed.items():
        if (
            isinstance(client_id, str)
            and isinstance(groups, list)
            and all(isinstance(g, str) for g in groups)
        ):
            mapping[client_id] = groups
        else:
            logger.warning(
                "Ignoring malformed %s entry for client_id=%r (expected list[str])",
                OKTA_M2M_CLIENT_GROUPS_ENV,
                client_id,
            )
    return mapping


class OktaM2MSync:
    """Service for syncing Okta M2M applications to MongoDB."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        okta_domain: str,
        okta_api_token: str,
    ):
        """Initialize Okta M2M sync service.

        Args:
            db: MongoDB database instance
            okta_domain: Okta org domain (e.g., YOUR_ORG.okta.com)
            okta_api_token: Okta API token for Admin API access
        """
        self.db = db
        self.okta_domain = okta_domain.replace("https://", "").rstrip("/")
        self.okta_api_token = okta_api_token
        self.collection = db["okta_m2m_clients"]
        self.idp_collection = db["idp_m2m_clients"]
        # Config-driven client_id -> groups mapping (fail closed: unset means no
        # client is auto-assigned any group). Never derived from Okta responses.
        self.client_groups = _load_client_groups_mapping()

        logger.info(
            "Initialized Okta M2M sync for domain: %s (%d configured client-group mappings)",
            self.okta_domain,
            len(self.client_groups),
        )

    async def _get_okta_applications(self) -> list[dict]:
        """Fetch all applications from Okta Admin API.

        Returns:
            List of Okta application dictionaries

        Raises:
            ValueError: If Okta API request fails
        """
        url = f"https://{self.okta_domain}/api/v1/apps"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"SSWS {self.okta_api_token}",
        }

        try:
            logger.info(f"Fetching applications from Okta: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            apps = response.json()
            logger.info(f"Retrieved {len(apps)} applications from Okta")
            return apps

        except requests.RequestException as e:
            logger.error(f"Failed to fetch Okta applications: {e}")
            raise ValueError(f"Okta API request failed: {e}")

    def _filter_m2m_applications(self, apps: list[dict]) -> list[dict]:
        """Filter to only M2M service applications.

        Args:
            apps: List of all Okta applications

        Returns:
            Filtered list of M2M applications
        """
        m2m_apps = []

        for app in apps:
            # M2M apps have signOnMode as "OPENID_CONNECT" and specific settings
            sign_on_mode = app.get("signOnMode")
            app_name = app.get("name", "")
            label = app.get("label", "")

            # Filter for service apps (API Services type in Okta)
            # or apps that have client_credentials grant type
            settings = app.get("settings", {})
            oauth_client = settings.get("oauthClient", {})
            grant_types = oauth_client.get("grant_types", [])

            if "client_credentials" in grant_types:
                logger.debug(f"Found M2M app: {label} (ID: {app.get('id')})")
                m2m_apps.append(app)

        logger.info(f"Filtered to {len(m2m_apps)} M2M applications")
        return m2m_apps

    def _determine_groups(self, client_id: str) -> list[str]:
        """Determine groups for a client ID from the configured mapping.

        Consults the operator-supplied mapping loaded from
        ``OKTA_M2M_CLIENT_GROUPS`` only. A client ID that is not explicitly
        listed receives NO groups (fail closed), so an injected/attacker-chosen
        client ID cannot be auto-granted privileged RBAC.

        Args:
            client_id: Okta client ID

        Returns:
            List of group names for this client (empty if not configured)
        """
        groups = self.client_groups.get(client_id, [])
        masked_id = f"{client_id[:8]}..." if client_id else "<none>"
        logger.debug(f"Client {masked_id} assigned groups: {groups}")
        return groups

    async def sync_from_okta(self, force_full_sync: bool = False) -> dict:
        """Sync M2M clients from Okta to MongoDB.

        Args:
            force_full_sync: If True, update all clients. Otherwise incremental.

        Returns:
            Dictionary with sync statistics
        """
        logger.info(f"Starting Okta M2M sync (force_full_sync={force_full_sync})")

        added_count = 0
        updated_count = 0
        error_count = 0
        errors = []

        try:
            # Fetch all applications from Okta
            all_apps = await self._get_okta_applications()

            # Filter to M2M applications
            m2m_apps = self._filter_m2m_applications(all_apps)

            # Process each M2M app
            for app in m2m_apps:
                try:
                    client_id = app.get("credentials", {}).get("oauthClient", {}).get("client_id")

                    if not client_id:
                        logger.warning(f"App {app.get('label')} has no client_id, skipping")
                        continue

                    # Check if client already exists in database
                    existing = await self.collection.find_one({"client_id": client_id})

                    # Determine groups for this client
                    groups = self._determine_groups(client_id)

                    client_doc = {
                        "client_id": client_id,
                        "name": app.get("label", client_id),
                        "description": app.get("_embedded", {})
                        .get("user", {})
                        .get("profile", {})
                        .get("description"),
                        "groups": groups,
                        "enabled": app.get("status") == "ACTIVE",
                        "okta_app_id": app.get("id"),
                        "last_synced": datetime.utcnow(),
                    }

                    masked_cid = f"{client_id[:8]}..." if client_id else "<none>"

                    if existing:
                        # Update existing record
                        client_doc["updated_at"] = datetime.utcnow()
                        await self.collection.update_one(
                            {"client_id": client_id}, {"$set": client_doc}
                        )
                        updated_count += 1
                        logger.info(f"Updated client: {masked_cid}")
                    else:
                        # Insert new record
                        client_doc["created_at"] = datetime.utcnow()
                        client_doc["updated_at"] = datetime.utcnow()
                        await self.collection.insert_one(client_doc)
                        added_count += 1
                        logger.info(f"Added new client: {masked_cid}")

                    # Also sync to generic idp_m2m_clients collection for groups enrichment
                    idp_doc = {
                        "client_id": client_id,
                        "name": app.get("label", client_id),
                        "description": client_doc.get("description"),
                        "groups": groups,
                        "enabled": client_doc["enabled"],
                        "provider": "okta",
                        "idp_app_id": app.get("id"),
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
                    error_msg = f"Failed to process app {app.get('label')}: {e}"
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
            logger.exception(f"Okta sync failed: {e}")
            return {
                "synced_count": 0,
                "added_count": 0,
                "updated_count": 0,
                "removed_count": 0,
                "errors": [str(e)],
            }

    async def get_all_clients(self) -> list[OktaM2MClient]:
        """Get all M2M clients from MongoDB.

        Returns:
            List of OktaM2MClient objects
        """
        cursor = self.collection.find({})
        docs = await cursor.to_list(length=None)

        clients = []
        for doc in docs:
            try:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                client = OktaM2MClient(**doc)
                clients.append(client)
            except Exception as e:
                logger.warning(f"Failed to parse client document: {e}")

        return clients

    async def get_client_groups(self, client_id: str) -> list[str]:
        """Get groups for a specific client ID.

        Args:
            client_id: Okta client ID

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
            client_id: Okta client ID
            groups: New list of groups

        Returns:
            True if updated, False if client not found
        """
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


def get_okta_m2m_sync(db: AsyncIOMotorDatabase) -> OktaM2MSync | None:
    """Factory function to create OktaM2MSync instance.

    Args:
        db: MongoDB database instance

    Returns:
        OktaM2MSync instance if Okta is configured, None otherwise
    """
    okta_domain = os.getenv("OKTA_DOMAIN")
    okta_api_token = os.getenv("OKTA_API_TOKEN")

    if not okta_domain or not okta_api_token:
        logger.warning("Okta not configured (missing OKTA_DOMAIN or OKTA_API_TOKEN)")
        return None

    return OktaM2MSync(
        db=db,
        okta_domain=okta_domain,
        okta_api_token=okta_api_token,
    )
