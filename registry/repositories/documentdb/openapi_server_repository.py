"""DocumentDB implementation of the OpenAPI MCP server and spec repositories (issue #2022)."""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ...schemas.openapi_server_models import (
    OpenApiServer,
    StoredSpec,
)
from ..interfaces import (
    OpenApiServerRepositoryBase,
    OpenApiSpecRepositoryBase,
)
from .client import (
    get_collection_name,
    get_documentdb_client,
)

logger = logging.getLogger(__name__)


SERVERS_COLLECTION: str = "openapi_servers"
SPECS_COLLECTION: str = "openapi_specs"


def _server_to_document(
    server: OpenApiServer,
) -> dict[str, Any]:
    doc = server.model_dump(mode="json")
    doc["_id"] = server.slug
    return doc


def _document_to_server(
    doc: dict[str, Any],
) -> OpenApiServer:
    data = dict(doc)
    data.pop("_id", None)
    return OpenApiServer(**data)


def _spec_to_document(
    spec: StoredSpec,
) -> dict[str, Any]:
    doc = spec.model_dump(mode="json")
    doc["_id"] = spec.sha256
    return doc


def _document_to_spec(
    doc: dict[str, Any],
) -> StoredSpec:
    data = dict(doc)
    data.pop("_id", None)
    return StoredSpec(**data)


class DocumentDBOpenApiServerRepository(OpenApiServerRepositoryBase):
    """Stores one document per OpenAPI MCP server, keyed by slug, targets embedded."""

    def __init__(self) -> None:
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name(SERVERS_COLLECTION)

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
            try:
                await self._collection.create_index("targets.spec_sha256")
            except Exception as e:
                logger.warning(f"Could not create indexes for {self._collection_name}: {e}")
        return self._collection

    async def count(self) -> int:
        collection = await self._get_collection()
        return await collection.count_documents({})

    async def list_all(self) -> list[OpenApiServer]:
        collection = await self._get_collection()
        return [_document_to_server(doc) async for doc in collection.find({})]

    async def get(
        self,
        slug: str,
    ) -> OpenApiServer | None:
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": slug})
        return _document_to_server(doc) if doc else None

    async def create(
        self,
        server: OpenApiServer,
    ) -> bool:
        collection = await self._get_collection()
        try:
            await collection.insert_one(_server_to_document(server))
        except DuplicateKeyError:
            return False
        return True

    async def replace(
        self,
        server: OpenApiServer,
        expected_generation: int,
    ) -> bool:
        collection = await self._get_collection()
        result = await collection.replace_one(
            {"_id": server.slug, "generation": expected_generation},
            _server_to_document(server),
        )
        return result.matched_count == 1

    async def delete(
        self,
        slug: str,
    ) -> bool:
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": slug})
        return result.deleted_count == 1

    async def count_spec_references(
        self,
        sha256: str,
    ) -> int:
        collection = await self._get_collection()
        return await collection.count_documents({"targets.spec_sha256": sha256})

    async def count_targets(self) -> int:
        collection = await self._get_collection()
        total = 0
        async for doc in collection.find({}, {"targets.name": 1}):
            total += len(doc.get("targets") or [])
        return total


class DocumentDBOpenApiSpecRepository(OpenApiSpecRepositoryBase):
    """Stores one document per distinct spec, keyed by the SHA-256 of its bytes."""

    def __init__(self) -> None:
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name(SPECS_COLLECTION)

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def upsert(
        self,
        spec: StoredSpec,
    ) -> None:
        collection = await self._get_collection()
        await collection.replace_one({"_id": spec.sha256}, _spec_to_document(spec), upsert=True)

    async def get_many(
        self,
        sha256s: set[str],
    ) -> dict[str, StoredSpec]:
        if not sha256s:
            return {}
        collection = await self._get_collection()
        cursor = collection.find({"_id": {"$in": sorted(sha256s)}})
        return {doc["_id"]: _document_to_spec(doc) async for doc in cursor}

    async def delete(
        self,
        sha256: str,
    ) -> None:
        collection = await self._get_collection()
        await collection.delete_one({"_id": sha256})
