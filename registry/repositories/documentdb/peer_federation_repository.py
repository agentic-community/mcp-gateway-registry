"""DocumentDB repository for peer federation configuration storage.

Works with both MongoDB Community Edition (storage_backend=mongodb-ce)
and AWS DocumentDB (storage_backend=documentdb). The client.py handles
authentication differences automatically.

Collections:
- mcp_peers_{namespace}: Peer registry configurations (_id = peer_id)
- mcp_peer_sync_state_{namespace}: Sync status (_id = peer_id)
"""

import logging
from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ...schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
)
from ...utils.federation_encryption import (
    decrypt_token_in_peer_dict,
    encrypt_token_in_peer_dict,
)
from ..interfaces import PeerFederationRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class DocumentDBPeerFederationRepository(PeerFederationRepositoryBase):
    """DocumentDB implementation of peer federation repository.

    Uses two collections:
    - Peers collection: stores PeerRegistryConfig documents
    - Sync state collection: stores PeerSyncStatus documents

    Both use peer_id as the _id field for efficient lookups.
    """

    def __init__(self):
        self._peers_collection: AsyncIOMotorCollection | None = None
        self._sync_state_collection: AsyncIOMotorCollection | None = None
        self._peers_collection_name = get_collection_name("mcp_peers")
        self._sync_state_collection_name = get_collection_name("mcp_peer_sync_state")
        logger.info(
            f"Initialized DocumentDB PeerFederationRepository with collections: "
            f"{self._peers_collection_name}, {self._sync_state_collection_name}"
        )

    async def _get_peers_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB peers collection."""
        if self._peers_collection is None:
            db = await get_documentdb_client()
            self._peers_collection = db[self._peers_collection_name]
        return self._peers_collection

    async def _get_sync_state_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB sync state collection."""
        if self._sync_state_collection is None:
            db = await get_documentdb_client()
            self._sync_state_collection = db[self._sync_state_collection_name]
        return self._sync_state_collection

    async def get_peer(
        self,
        peer_id: str,
    ) -> PeerRegistryConfig | None:
        """Get peer configuration by ID."""
        try:
            collection = await self._get_peers_collection()

            doc = await collection.find_one({"_id": peer_id})

            if not doc:
                logger.debug(f"Peer not found: {peer_id}")
                return None

            # Remove MongoDB _id before creating Pydantic model
            doc.pop("_id", None)

            # Decrypt federation token if present
            decrypt_token_in_peer_dict(doc)

            peer_config = PeerRegistryConfig(**doc)
            logger.debug(f"Retrieved peer config: {peer_id}")
            return peer_config

        except Exception as e:
            logger.error(f"Failed to get peer {peer_id}: {e}", exc_info=True)
            return None

    async def list_peers(
        self,
        enabled: bool | None = None,
    ) -> list[PeerRegistryConfig]:
        """List all peer configurations with optional filtering."""
        try:
            collection = await self._get_peers_collection()

            # Build query based on enabled filter
            query: dict[str, Any] = {}
            if enabled is not None:
                query["enabled"] = enabled

            cursor = collection.find(query)

            peers = []
            async for doc in cursor:
                doc.pop("_id", None)
                # Decrypt federation token if present
                decrypt_token_in_peer_dict(doc)
                try:
                    peer_config = PeerRegistryConfig(**doc)
                    peers.append(peer_config)
                except Exception as e:
                    logger.error(
                        f"Failed to parse peer config {doc.get('peer_id', 'unknown')}: {e}"
                    )

            logger.info(f"Listed {len(peers)} peers (enabled={enabled})")
            return peers

        except Exception as e:
            logger.error(f"Failed to list peers: {e}", exc_info=True)
            return []

    async def create_peer(
        self,
        config: PeerRegistryConfig,
    ) -> PeerRegistryConfig:
        """Create a new peer configuration."""
        try:
            collection = await self._get_peers_collection()
            peer_id = config.peer_id

            # Check if peer already exists
            existing = await collection.find_one({"_id": peer_id})
            if existing:
                raise ValueError(f"Peer ID '{peer_id}' already exists")

            # Set timestamps
            now = datetime.now(UTC)
            config.created_at = now
            config.updated_at = now

            # Convert to document with _id
            doc = config.model_dump(mode="json")
            doc["_id"] = peer_id

            # Encrypt federation token before storage
            encrypt_token_in_peer_dict(doc)

            await collection.insert_one(doc)

            # Also create initial sync status
            initial_status = PeerSyncStatus(peer_id=peer_id)
            await self.update_sync_status(peer_id, initial_status)

            logger.info(f"Created peer: {peer_id} ({config.name})")
            return config

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create peer {config.peer_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to create peer: {e}")

    async def update_peer(
        self,
        peer_id: str,
        updates: dict[str, Any],
    ) -> PeerRegistryConfig:
        """Update an existing peer configuration."""
        try:
            collection = await self._get_peers_collection()

            # Get existing peer
            existing_doc = await collection.find_one({"_id": peer_id})
            if not existing_doc:
                raise ValueError(f"Peer not found: {peer_id}")

            # Remove _id before merging
            existing_doc.pop("_id", None)

            # Decrypt federation token before constructing Pydantic model.
            # Without this, federation_token_encrypted (unknown to Pydantic)
            # is silently dropped during model_dump(), permanently losing the
            # token on any peer update. Fixes issue #561.
            decrypt_token_in_peer_dict(existing_doc)

            # Merge updates with existing data
            existing_doc.update(updates)

            # Ensure peer_id is consistent
            existing_doc["peer_id"] = peer_id

            # Update timestamp
            existing_doc["updated_at"] = datetime.now(UTC).isoformat()

            # Validate updated peer
            try:
                updated_peer = PeerRegistryConfig(**existing_doc)
            except Exception as e:
                raise ValueError(f"Invalid peer update: {e}")

            # Save to database
            doc = updated_peer.model_dump(mode="json")
            doc["_id"] = peer_id

            # Encrypt federation token before storage
            encrypt_token_in_peer_dict(doc)

            await collection.replace_one({"_id": peer_id}, doc, upsert=False)

            logger.info(f"Updated peer: {peer_id}")
            return updated_peer

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update peer {peer_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to update peer: {e}")

    async def delete_peer(
        self,
        peer_id: str,
    ) -> bool:
        """Delete a peer configuration and its sync status."""
        try:
            peers_collection = await self._get_peers_collection()
            sync_collection = await self._get_sync_state_collection()

            # Check if peer exists
            existing = await peers_collection.find_one({"_id": peer_id})
            if not existing:
                raise ValueError(f"Peer not found: {peer_id}")

            # Delete peer config
            result = await peers_collection.delete_one({"_id": peer_id})

            if result.deleted_count == 0:
                logger.warning(f"Peer not found for deletion: {peer_id}")
                return False

            # Also delete sync status (cascade delete)
            await sync_collection.delete_one({"_id": peer_id})

            logger.info(f"Deleted peer and sync status: {peer_id}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete peer {peer_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to delete peer: {e}")

    async def get_sync_status(
        self,
        peer_id: str,
    ) -> PeerSyncStatus | None:
        """Get sync status for a peer."""
        try:
            collection = await self._get_sync_state_collection()

            doc = await collection.find_one({"_id": peer_id})

            if not doc:
                logger.debug(f"Sync status not found for peer: {peer_id}")
                return None

            # Remove MongoDB _id and metadata before creating Pydantic model
            doc.pop("_id", None)
            doc.pop("updated_at", None)

            sync_status = PeerSyncStatus(**doc)
            logger.debug(f"Retrieved sync status for peer: {peer_id}")
            return sync_status

        except Exception as e:
            logger.error(f"Failed to get sync status for {peer_id}: {e}", exc_info=True)
            return None

    async def update_sync_status(
        self,
        peer_id: str,
        status: PeerSyncStatus,
    ) -> PeerSyncStatus:
        """Update sync status for a peer (upsert)."""
        try:
            collection = await self._get_sync_state_collection()

            # Convert to document with _id
            doc = status.model_dump(mode="json")
            doc["_id"] = peer_id
            doc["updated_at"] = datetime.now(UTC).isoformat()

            await collection.replace_one({"_id": peer_id}, doc, upsert=True)

            logger.debug(f"Updated sync status for peer: {peer_id}")
            return status

        except Exception as e:
            logger.error(f"Failed to update sync status for {peer_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to update sync status: {e}")

    async def list_sync_statuses(self) -> list[PeerSyncStatus]:
        """List all peer sync statuses."""
        try:
            collection = await self._get_sync_state_collection()

            cursor = collection.find({})

            statuses = []
            async for doc in cursor:
                doc.pop("_id", None)
                doc.pop("updated_at", None)  # Remove metadata field
                try:
                    sync_status = PeerSyncStatus(**doc)
                    statuses.append(sync_status)
                except Exception as e:
                    logger.error(
                        f"Failed to parse sync status {doc.get('peer_id', 'unknown')}: {e}"
                    )

            logger.info(f"Listed {len(statuses)} sync statuses")
            return statuses

        except Exception as e:
            logger.error(f"Failed to list sync statuses: {e}", exc_info=True)
            return []

    async def load_all(self) -> None:
        """Load/reload all peers and sync states from storage.

        For DocumentDB, this is a no-op since data is loaded on-demand.
        However, we verify connectivity and log collection stats.
        """
        try:
            peers_collection = await self._get_peers_collection()
            sync_collection = await self._get_sync_state_collection()

            # Count documents to verify connectivity
            peer_count = await peers_collection.count_documents({})
            sync_count = await sync_collection.count_documents({})

            logger.info(
                f"Loaded peer federation data: {peer_count} peers, {sync_count} sync statuses"
            )

        except Exception as e:
            logger.error(f"Failed to load peer federation data: {e}", exc_info=True)
            raise
