#!/usr/bin/env python3
"""
Initialize MongoDB CE for local development.

This script:
1. Initializes replica set (rs0)
2. Creates collections and indexes
3. Loads default admin scope from registry-admins.json

Usage:
    python init-mongodb-ce.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure, ServerSelectionTimeoutError

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Collection names
COLLECTION_SERVERS = "mcp_servers"
COLLECTION_AGENTS = "mcp_agents"
COLLECTION_SCOPES = "mcp_scopes"
COLLECTION_EMBEDDINGS = "mcp_embeddings_1536"
COLLECTION_SECURITY_SCANS = "mcp_security_scans"
COLLECTION_FEDERATION_CONFIG = "mcp_federation_config"
COLLECTION_AUDIT_EVENTS = "audit_events"
COLLECTION_SKILLS = "agent_skills"

# Identity claim fields persisted on audit records (issue #1642). Keep in step
# with registry/audit/models.py::IdentityClaims and with the DocumentDB init
# (scripts/init-documentdb-indexes.py), which builds the same index set.
AUDIT_CLAIM_FIELDS = ("principal_name", "subject", "canonical_id", "object_id")
# Streams that nest the claims under `identity`. `token_mint` instead carries the
# same values at the TOP level, plus its readable identity in a flat `username`.
AUDIT_NESTED_LOG_TYPES = ("registry_api_access", "mcp_server_access")
AUDIT_FLAT_LOG_TYPE = "token_mint"
# Short stream tokens keep index names well inside Amazon DocumentDB's limit.
_AUDIT_LOG_TYPE_ABBREV = {
    "registry_api_access": "api",
    "mcp_server_access": "mcp",
    "token_mint": "mint",  # nosec B105 - audit stream name, not a secret
}
# Claim indexes created by the first cut of #1642 with auto-generated names, now
# superseded by the log_type-led partial indexes. Dropped only AFTER the
# replacements exist, so no query is ever left without an index.
LEGACY_AUDIT_CLAIM_INDEXES = tuple(
    [f"identity.{field}_1_timestamp_-1" for field in AUDIT_CLAIM_FIELDS]
    + [f"{field}_1_timestamp_-1" for field in ("username", *AUDIT_CLAIM_FIELDS)]
)


def _get_config_from_env() -> dict:
    """Get MongoDB CE configuration from environment variables.

    When MONGODB_CONNECTION_STRING is set, the caller owns the full URI
    (Atlas, externally-managed replica set, etc.) and discrete
    host/port/user/password values are ignored.
    """
    return {
        "connection_string": os.getenv("MONGODB_CONNECTION_STRING", ""),
        "host": os.getenv("DOCUMENTDB_HOST", "mongodb"),
        "port": int(os.getenv("DOCUMENTDB_PORT", "27017")),
        "database": os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        "namespace": os.getenv("DOCUMENTDB_NAMESPACE", "default"),
        "username": os.getenv("DOCUMENTDB_USERNAME", ""),
        "password": os.getenv("DOCUMENTDB_PASSWORD", ""),
        "replicaset": os.getenv("DOCUMENTDB_REPLICA_SET", "rs0"),
    }


def _initialize_replica_set(
    host: str,
    port: int,
    username: str,
    password: str,
) -> None:
    """Initialize MongoDB replica set using pymongo (synchronous)."""
    from pymongo import MongoClient

    logger.info("Initializing MongoDB replica set...")

    try:
        # Connect without the replica set for initialization. _bootstrap_uri adds
        # credentials only when a username is configured, because MongoDB CE runs
        # without authentication by default.
        connection_uri = _bootstrap_uri(host, port, username, password)
        if not (username and password):
            logger.info("Connecting without authentication (MongoDB CE no-auth mode)")

        client = MongoClient(
            connection_uri,
            serverSelectionTimeoutMS=5000,
            directConnection=True,
        )

        # Check if already initialized
        try:
            status = client.admin.command("replSetGetStatus")
            logger.info("Replica set already initialized")
            client.close()
            return
        except OperationFailure as e:
            if "no replset config has been received" in str(e).lower():
                # Not initialized, proceed
                pass
            else:
                raise

        # Initialize replica set
        config = {"_id": "rs0", "members": [{"_id": 0, "host": f"{host}:{port}"}]}

        result = client.admin.command("replSetInitiate", config)
        logger.info(f"Replica set initialized: {result}")
        client.close()

        # Wait for replica set to elect primary
        logger.info("Waiting for replica set to elect primary...")
        time.sleep(10)

    except Exception as e:
        logger.error(f"Error initializing replica set: {e}")
        raise


def _redact_uri(message: object) -> str:
    """Strip any Mongo URI -- and so any embedded password -- from a message."""
    return re.sub(r"mongodb(?:\+srv)?://\S*", "<redacted-uri>", str(message))


def _bootstrap_uri(host: str, port: int, username: str, password: str) -> str:
    """Direct URI used for admin commands before the replica set is usable.

    MongoDB CE runs without authentication by default, so credentials are added
    only when a username is actually configured: authenticating against a
    no-user MongoDB fails outright.
    """
    if username and password:
        return (
            f"mongodb://{username}:{password}@{host}:{port}/"
            "?authMechanism=SCRAM-SHA-256&authSource=admin"
        )
    return f"mongodb://{host}:{port}/"


def _audit_ttl_days() -> int:
    """Audit retention in days, from AUDIT_LOG_MONGODB_TTL_DAYS."""
    raw = os.getenv("AUDIT_LOG_MONGODB_TTL_DAYS", "7").strip() or "7"
    try:
        days = int(raw)
    except ValueError as exc:
        raise ValueError(f"AUDIT_LOG_MONGODB_TTL_DAYS must be an integer, got {raw!r}") from exc
    if days < 1:
        raise ValueError(f"AUDIT_LOG_MONGODB_TTL_DAYS must be >= 1, got {days}")
    return days


def _wait_for_mongodb(config: dict, override: str) -> None:
    """Block until MongoDB is usable, or raise once the deadline passes.

    This replaces two things that used to sit outside this script: a blind
    ``time.sleep(10)`` here, and the Helm chart's ``wait.py`` init container,
    which looped ``while True`` with no deadline and always built a credentialed
    URI (so it could never connect to a no-auth MongoDB CE).

    Phase 1 waits for the server to answer ``ping``.

    Phase 2, skipped when the caller owns the topology, waits for every replica
    set member to reach PRIMARY or SECONDARY. If the set is still uninitialized
    once the server has been reachable for
    ``MONGODB_REPLICA_SET_INITIATE_GRACE_SECONDS``, this initializes it. The
    grace period exists so that a managed control plane -- the MongoDB
    Kubernetes operator -- gets to configure its own members first; racing it
    would produce a single-member set under the wrong name.
    """
    from pymongo import MongoClient

    timeout_s = float(os.getenv("MONGODB_WAIT_TIMEOUT_SECONDS", "300"))
    grace_s = float(os.getenv("MONGODB_REPLICA_SET_INITIATE_GRACE_SECONDS", "30"))
    deadline = time.monotonic() + timeout_s

    if override:
        uri = override
        target = urlsplit(override).hostname or "(override)"
    else:
        uri = _bootstrap_uri(config["host"], config["port"], config["username"], config["password"])
        target = f"{config['host']}:{config['port']}"

    logger.info(f"Waiting up to {timeout_s:.0f}s for MongoDB at {target} to be ready...")
    reachable_at: float | None = None
    initiated = False
    last_error = "not reachable yet"

    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"MongoDB at {target} was not ready within {timeout_s:.0f}s "
                f"(last state: {last_error}). Raise MONGODB_WAIT_TIMEOUT_SECONDS if the "
                "cluster is simply slow to start."
            )
        client = None
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                directConnection=not override,
            )
            client.admin.command("ping")
            if reachable_at is None:
                reachable_at = time.monotonic()
                logger.info("MongoDB is accepting connections")
            if override:
                return

            status = client.admin.command("replSetGetStatus")
            members = status.get("members", [])
            ready = [m for m in members if m.get("state") in (1, 2)]
            if members and len(ready) == len(members):
                logger.info(f"Replica set ready ({len(ready)}/{len(members)} members)")
                return
            last_error = f"replica set members ready {len(ready)}/{len(members)}"
            logger.info(f"Waiting for replica set: {last_error}")
        except OperationFailure as exc:
            uninitialized = exc.code == 94 or "no replset config" in str(exc).lower()
            if not override and uninitialized:
                waited = 0.0 if reachable_at is None else time.monotonic() - reachable_at
                if not initiated and waited >= grace_s:
                    logger.info(
                        f"Replica set still uninitialized {waited:.0f}s after MongoDB became "
                        "reachable; initializing it from here"
                    )
                    _initialize_replica_set(
                        config["host"], config["port"], config["username"], config["password"]
                    )
                    initiated = True
                last_error = "replica set not initialized yet"
            else:
                last_error = _redact_uri(exc)
        except Exception as exc:
            last_error = _redact_uri(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as close_exc:
                    # Closing a probe client is best effort; the retry loop makes
                    # a fresh one either way. Recorded rather than swallowed.
                    logger.debug(f"Ignoring MongoDB probe close error: {close_exc}")
        time.sleep(max(0.5, min(5.0, deadline - time.monotonic())))


def _audit_claim_index_name(log_type: str, field: str) -> str:
    """Explicit, engine-independent name for one identity-claim index.

    Named explicitly so MongoDB CE and Amazon DocumentDB agree: auto-generated
    names differ between the two, which is what forced an earlier migration to
    guess at several name variants when dropping an index.
    """
    return f"audit_claim_{_AUDIT_LOG_TYPE_ABBREV[log_type]}_{field}_idx"


async def _create_audit_claim_indexes(collection, full_name: str) -> None:
    """Create the identity-claim indexes behind the audit username filter.

    An operator pastes an IdP-side value (upn / sub / oid@tid / oid) and the
    audit API searches every stored claim for it, so each claim needs an index
    or the whole ``$or`` degrades to a collection scan.

    Shape, and why it is this shape:

    * ``log_type`` LEADS every key, because every audit query filters on it
      (``registry/audit/routes.py`` builds ``{"log_type": ...}`` before adding
      the identity ``$or``). It also matches the neighbouring
      ``log_type_resource_type_resource_id_timestamp_idx``. Changing the leading
      field later would mean rebuilding every one of these on a hot collection.
    * ``partialFilterExpression`` pins each index to the ONE stream whose record
      shape it serves. Without it, every index stores an entry for every audit
      record -- including the ones that carry no claim at all. Measured on a
      representative 30k-record mix: 2.09 MB across these 13 partial indexes
      versus 3.46 MB across 9 unpartitioned ones (-40%), and an insert updates
      4-5 claim indexes instead of all 9.
    * ``sparse`` is deliberately NOT used. On a compound index it keeps a
      document when ANY indexed field exists, and ``log_type`` always exists, so
      it would be inert here (measured: byte-identical to passing no option).
    * The filter is a single-stream EQUALITY rather than an ``$in`` because
      Amazon DocumentDB only uses a partial index when the query predicate
      matches its filter expression, and audit queries are always scoped to one
      stream at a time.
    * Descending timestamp matches how the audit API reads: newest first.

    Falls back to unpartitioned indexes if the engine rejects
    ``partialFilterExpression`` (Amazon DocumentDB before 5.0, Elastic
    Clusters), so an old cluster degrades to "larger index" rather than "init
    job fails".
    """
    targets: list[tuple[str, str]] = [
        (log_type, f"identity.{field}")
        for log_type in AUDIT_NESTED_LOG_TYPES
        for field in AUDIT_CLAIM_FIELDS
    ]
    targets += [(AUDIT_FLAT_LOG_TYPE, field) for field in ("username", *AUDIT_CLAIM_FIELDS)]

    partial_supported = True
    for log_type, key in targets:
        name = _audit_claim_index_name(log_type, key.rsplit(".", 1)[-1])
        keys = [("log_type", ASCENDING), (key, ASCENDING), ("timestamp", DESCENDING)]
        if partial_supported:
            partial = {"log_type": log_type}
            try:
                await collection.create_index(keys, name=name, partialFilterExpression=partial)
                continue
            except OperationFailure as exc:
                if exc.code == 85:
                    # Same name, different options: an earlier run created this
                    # index unpartitioned (or with another filter). Replace it.
                    await collection.drop_index(name)
                    await collection.create_index(keys, name=name, partialFilterExpression=partial)
                    continue
                partial_supported = False
                logger.warning(
                    f"{full_name}: engine rejected partialFilterExpression "
                    f"(code {exc.code}); creating unpartitioned identity-claim indexes "
                    "instead. They are correct, just larger."
                )
        await collection.create_index(keys, name=name)


async def _existing_ttl_seconds(collection, name: str = "timestamp_ttl") -> int | None:
    """Current expireAfterSeconds of the audit TTL index, or None if absent."""
    async for spec in collection.list_indexes():
        if spec.get("name") == name:
            value = spec.get("expireAfterSeconds")
            return int(value) if value is not None else None
    return None


async def _reconcile_audit_ttl(collection, full_name: str, ttl_days: int) -> int:
    """Point the audit TTL index at ``ttl_days``, refusing to shorten silently.

    Shortening a TTL makes MongoDB's background TTL monitor delete every audit
    record older than the new window, normally within a minute, irreversibly.
    That must never happen as a side effect of an upgrade picking up a default:
    this job now runs on every ``helm upgrade``, and the deployment that most
    needs a long retention is exactly the one that set it out-of-band.

    So a REDUCTION is refused unless AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK is set.
    Refusing leaves the longer retention in place, which costs storage but
    destroys nothing.

    Returns the retention actually in effect afterwards, which is NOT always the
    requested value -- callers must log what they got, not what they asked for.
    """
    desired = ttl_days * 24 * 60 * 60
    existing = await _existing_ttl_seconds(collection)

    if existing is None:
        await collection.create_index(
            [("timestamp", ASCENDING)], expireAfterSeconds=desired, name="timestamp_ttl"
        )
        logger.info(f"Created audit TTL index for {full_name} ({ttl_days} days)")
        return ttl_days

    if existing == desired:
        logger.info(f"Audit TTL index for {full_name} already {ttl_days} days")
        return ttl_days

    allow_shrink = os.getenv("AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if desired < existing and not allow_shrink:
        logger.error(
            f"{full_name}: REFUSING to shorten audit retention from "
            f"{existing // 86400} days to {ttl_days} days. MongoDB would delete every "
            f"audit record older than {ttl_days} days within about a minute, and it "
            "cannot be undone. The existing retention has been left in place. Set "
            "AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK=true to allow this on purpose, or set "
            "AUDIT_LOG_MONGODB_TTL_DAYS to the retention you actually want."
        )
        return existing // 86400

    logger.info(f"{full_name}: changing audit retention {existing // 86400} -> {ttl_days} days")
    await collection.drop_index("timestamp_ttl")
    await collection.create_index(
        [("timestamp", ASCENDING)], expireAfterSeconds=desired, name="timestamp_ttl"
    )
    return ttl_days


async def _create_standard_indexes(
    collection,
    collection_name: str,
    namespace: str,
) -> None:
    """Create standard indexes for collections."""
    full_name = f"{collection_name}_{namespace}"

    if collection_name == COLLECTION_SERVERS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("enabled", ASCENDING)])
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("manifest.serverInfo.name", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_AGENTS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("enabled", ASCENDING)])
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("card.name", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_SCOPES:
        # No additional indexes needed - scopes use _id as primary key
        # group_mappings is an array, not indexed
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_EMBEDDINGS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("entity_type", ASCENDING)])
        logger.info(f"Created indexes for {full_name} (vector search via app code)")

    elif collection_name == COLLECTION_SECURITY_SCANS:
        await collection.create_index([("server_path", ASCENDING)])
        await collection.create_index([("scan_status", ASCENDING)])
        await collection.create_index([("scanned_at", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_FEDERATION_CONFIG:
        await collection.create_index([("registry_name", ASCENDING)], unique=True)
        await collection.create_index([("enabled", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_AUDIT_EVENTS:
        # Indexes for audit event queries (Requirements 6.2)
        # Note: timestamp index is created as TTL index below, so we use compound indexes here
        await collection.create_index([("identity.username", ASCENDING), ("timestamp", ASCENDING)])
        await collection.create_index([("action.operation", ASCENDING), ("timestamp", ASCENDING)])
        await collection.create_index(
            [("action.resource_type", ASCENDING), ("timestamp", ASCENDING)]
        )

        # Identity lookups behind the audit username filter (issue #1642).
        # See _create_audit_claim_indexes for the shape and the reasoning.
        await _create_audit_claim_indexes(collection, full_name)

        # Index for MCP server name distinct/filter queries
        await collection.create_index([("mcp_server.name", ASCENDING)])

        # Composite unique index on (request_id, log_type). One request
        # legitimately writes both an MCPServerAccessRecord and a
        # RegistryApiAccessRecord, and they share a request_id; the single-field
        # unique index this supersedes rejected the second write, and the audit
        # sink logged CRITICAL "AUDIT RECORD DROPPED" for it.
        #
        # Order matters: create the replacement BEFORE dropping the old index.
        # The new index is strictly more permissive, so the collection is never
        # left without a uniqueness constraint on request_id, and a concurrent
        # duplicate cannot slip through a gap. Dropping first would open exactly
        # that window on a live cluster -- and a duplicate landing in it would
        # then fail this unique build on every subsequent upgrade.
        try:
            await collection.create_index(
                [("request_id", ASCENDING), ("log_type", ASCENDING)],
                name="request_id_log_type_idx",
                unique=True,
            )
        except DuplicateKeyError:
            finder = (
                "db." + full_name + ".aggregate([{$group: {_id: {request_id: "
                "'$request_id', log_type: '$log_type'}, n: {$sum: 1}}}, "
                "{$match: {n: {$gt: 1}}}])"
            )
            logger.error(
                f"{full_name}: cannot build the unique (request_id, log_type) index because "
                "the collection already holds a true duplicate. List the offending pairs "
                f"with:  {finder}  -- then delete the surplus copies and re-run this job. "
                "The old index has deliberately NOT been dropped, so uniqueness is still "
                "enforced meanwhile."
            )
            raise

        # Only now retire the superseded single-field index. Tolerates absence: a
        # fresh install never had it. Both the auto-generated and the explicitly
        # named variant are tried, because the two storage backends historically
        # named this index differently.
        for old_index_name in ("request_id_1", "request_id_idx"):
            try:
                await collection.drop_index(old_index_name)
                logger.info(
                    f"Dropped superseded single-field index '{old_index_name}' from {full_name}"
                )
            except OperationFailure:
                logger.debug(f"No '{old_index_name}' index to drop from {full_name}")

        # Retire the first-cut claim indexes now that their log_type-led partial
        # replacements exist (created above, so there is never a gap).
        for legacy_name in LEGACY_AUDIT_CLAIM_INDEXES:
            try:
                await collection.drop_index(legacy_name)
                logger.info(f"Dropped superseded claim index '{legacy_name}' from {full_name}")
            except OperationFailure:
                logger.debug(f"No '{legacy_name}' index to drop from {full_name}")

        # Compound index for token_mint flat-field queries (resource_type/
        # resource_id at the top level, not nested under action.*). Mirrors the
        # DocumentDB init (scripts/init-documentdb-indexes.py) so MongoDB CE
        # deployments get the same query performance for the token_mint stream.
        await collection.create_index(
            [
                ("log_type", ASCENDING),
                ("resource_type", ASCENDING),
                ("resource_id", ASCENDING),
                ("timestamp", DESCENDING),
            ],
            name="log_type_resource_type_resource_id_timestamp_idx",
        )

        # TTL index for automatic expiration (Requirements 6.3). Doubles as the
        # timestamp index for sorting. Default 7 days, set by
        # AUDIT_LOG_MONGODB_TTL_DAYS.
        ttl_days = _audit_ttl_days()
        effective_ttl_days = await _reconcile_audit_ttl(collection, full_name, ttl_days)
        logger.info(f"Created indexes for {full_name} (TTL: {effective_ttl_days} days)")

    elif collection_name == COLLECTION_SKILLS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("name", ASCENDING)], unique=True)
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("visibility", ASCENDING)])
        await collection.create_index([("is_enabled", ASCENDING)])
        await collection.create_index([("registry_name", ASCENDING)])
        await collection.create_index([("owner", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")


async def _load_default_scopes(
    db,
    namespace: str,
) -> None:
    """Load default scopes from JSON files into scopes collection.

    This loads all scope JSON files from the scripts directory:
    - registry-admins.json: Bootstrap admin scope with full permissions
    - mcp-registry-admin.json: MCP registry admin scope (Keycloak group)
    - mcp-servers-unrestricted-read.json: Read-only access to all servers
    - mcp-servers-unrestricted-execute.json: Full CRUD access to all servers
    """
    collection_name = f"{COLLECTION_SCOPES}_{namespace}"
    collection = db[collection_name]

    # Find scope files in the same directory as this script
    script_dir = Path(__file__).parent

    # List of scope files to load (order matters - base scopes first)
    scope_files = [
        "registry-admins.json",
        "mcp-registry-admin.json",
        "mcp-servers-unrestricted-read.json",
        "mcp-servers-unrestricted-execute.json",
        "federation-service.json",
    ]

    loaded_count = 0
    for scope_filename in scope_files:
        scope_file = script_dir / scope_filename

        if not scope_file.exists():
            logger.warning(f"Scope file not found: {scope_file}")
            continue

        try:
            with open(scope_file) as f:
                scope_data = json.load(f)

            logger.info(f"Loading scope from {scope_filename}")

            # For registry-admins scope, add Entra admin group ID from env if configured
            if scope_data["_id"] == "registry-admins":
                entra_admin_group_id = os.getenv("ENTRA_GROUP_ADMIN_ID", "").strip()
                if entra_admin_group_id:
                    group_mappings = scope_data.get("group_mappings", [])
                    if entra_admin_group_id not in group_mappings:
                        group_mappings.append(entra_admin_group_id)
                        scope_data["group_mappings"] = group_mappings
                        logger.info(f"  Added Entra admin group ID: {entra_admin_group_id}")

            # Upsert the scope document
            result = await collection.update_one(
                {"_id": scope_data["_id"]}, {"$set": scope_data}, upsert=True
            )

            if result.upserted_id:
                logger.info(f"Inserted scope: {scope_data['_id']}")
                loaded_count += 1
            elif result.modified_count > 0:
                logger.info(f"Updated scope: {scope_data['_id']}")
                loaded_count += 1
            else:
                logger.info(f"Scope already up-to-date: {scope_data['_id']}")

            if "group_mappings" in scope_data:
                logger.info(f"  group_mappings: {scope_data.get('group_mappings', [])}")

        except Exception as e:
            logger.error(f"Failed to load scope from {scope_filename}: {e}", exc_info=True)

    logger.info(f"Loaded {loaded_count} scopes into {collection_name}")


async def _initialize_mongodb_ce() -> None:
    """Main initialization function."""
    config = _get_config_from_env()
    override = config["connection_string"]

    logger.info("=" * 60)
    logger.info("MongoDB CE Initialization for MCP Gateway")
    logger.info("=" * 60)
    if override:
        logger.info(
            f"Host: {urlsplit(override).hostname or '(override)'} (connection string override)"
        )
    else:
        logger.info(f"Host: {config['host']}:{config['port']}")
    logger.info(f"Database: {config['database']}")
    logger.info(f"Namespace: {config['namespace']}")
    logger.info("")

    # Wait for MongoDB to be reachable and, unless the caller owns the topology,
    # for its replica set to be ready -- initializing the set here only if
    # nothing else has after a grace period. Bounded: raises on timeout rather
    # than hanging. This supersedes both the blind `time.sleep(10)` that used to
    # be here and the unbounded `wait.py` init container the Helm chart mounted.
    _wait_for_mongodb(config, override)

    if override:
        # Caller owns the topology (Atlas, externally-managed replica set, etc.).
        # Skip replSetInitiate -- we lack admin rights and the replica set is
        # already configured by the provider.
        logger.info("Skipping replica-set initialization (connection string override in use)")
        connection_string = override
    else:
        # Connect with motor for async operations
        # Use auth only if username is provided (MongoDB CE runs without auth by default)
        if config["username"] and config["password"]:
            connection_string = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?replicaSet={config['replicaset']}&authMechanism=SCRAM-SHA-256&authSource=admin"
        else:
            connection_string = f"mongodb://{config['host']}:{config['port']}/{config['database']}?replicaSet={config['replicaset']}"
            logger.info("Using no-auth connection for async client")

    try:
        client = AsyncIOMotorClient(
            connection_string,
            serverSelectionTimeoutMS=10000,
        )

        # Verify connection
        await client.admin.command("ping")
        logger.info("Connected to MongoDB successfully")

        db = client[config["database"]]
        namespace = config["namespace"]

        # Create collections and indexes
        logger.info("Creating collections and indexes...")

        collections = [
            COLLECTION_SERVERS,
            COLLECTION_AGENTS,
            COLLECTION_SCOPES,
            COLLECTION_EMBEDDINGS,
            COLLECTION_SECURITY_SCANS,
            COLLECTION_FEDERATION_CONFIG,
            COLLECTION_AUDIT_EVENTS,
            COLLECTION_SKILLS,
        ]

        for coll_name in collections:
            full_name = f"{coll_name}_{namespace}"

            # Check if collection already exists
            existing_collections = await db.list_collection_names()

            if full_name in existing_collections:
                logger.info(f"Collection {full_name} already exists, skipping creation")
            else:
                logger.info(f"Creating collection: {full_name}")
                await db.create_collection(full_name)

            # Create indexes (idempotent - MongoDB handles duplicates)
            collection = db[full_name]
            await _create_standard_indexes(collection, coll_name, namespace)

        # Load default admin scope
        await _load_default_scopes(db, namespace)

        logger.info("")
        logger.info("=" * 60)
        logger.info("MongoDB CE Initialization Complete!")
        logger.info("=" * 60)
        logger.info("Collections created:")
        for coll_name in collections:
            if coll_name == COLLECTION_EMBEDDINGS:
                logger.info(f"  - {coll_name}_{namespace} (with vector search)")
            elif coll_name == COLLECTION_AUDIT_EVENTS:
                ttl_days = _audit_ttl_days()
                logger.info(f"  - {coll_name}_{namespace} (TTL: {ttl_days} days requested)")
            else:
                logger.info(f"  - {coll_name}_{namespace}")
        logger.info("")
        logger.info("To use MongoDB CE:")
        logger.info("  export STORAGE_BACKEND=mongodb-ce")
        logger.info("  docker-compose up registry")
        logger.info("")
        logger.info("Or for AWS DocumentDB:")
        logger.info("  export STORAGE_BACKEND=documentdb")
        logger.info("  docker-compose up registry")
        logger.info("=" * 60)

        client.close()

    except ServerSelectionTimeoutError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Make sure MongoDB is running and accessible")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise


def main() -> None:
    """Entry point."""
    asyncio.run(_initialize_mongodb_ce())


if __name__ == "__main__":
    main()
