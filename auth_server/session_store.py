"""Server-side OAuth session store.

The browser cookie holds only an opaque, signed `session_id`. The full session
record (username, groups, optional id_token) lives in MongoDB/DocumentDB so it
cannot blow the 4 KB browser cookie limit when an external IdP returns a large
groups claim.

Sensitivity model:
    - Username, email, name, groups, provider, auth_method are stored as-is.
      They were already client-visible in the previous signed-cookie payload.
    - `id_token` is the only true bearer credential and is encrypted with
      AES-GCM using a key derived from `SECRET_KEY` via HKDF. Rotating
      `SECRET_KEY` invalidates stored id_tokens — the same blast radius as
      cookie rotation today.

Failure semantics:
    - `resolve_session` returns None on any read failure (network blip, replica
      lag, missing document). Callers must treat None as 401. We never raise
      500 from a transient store failure; the client retries.
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING, ReadPreference, WriteConcern

from registry.auth.session_crypto import decrypt_id_token, encrypt_id_token

logger = logging.getLogger(__name__)

COLLECTION_BASE_NAME: str = "oauth_sessions"
SESSION_ID_BYTES: int = 32

_collection: AsyncIOMotorCollection | None = None
_indexes_created: bool = False


async def _get_collection() -> AsyncIOMotorCollection:
    """Get the oauth_sessions collection, creating indexes on first access.

    Configures w:majority writes and primaryPreferred reads to mitigate the
    multi-replica read-after-write window: the cookie is set on the client
    after the write returns, so subsequent reads must see the document.

    The underlying connection comes from the registry's shared MongoDB
    client. w:majority and primaryPreferred are no-ops on a standalone
    single-node MongoDB but are required on multi-node clusters.
    """
    global _collection, _indexes_created

    if _collection is not None:
        return _collection

    # Reuse the registry's namespaced collection naming so this collection
    # follows the same `<base>_<namespace>` convention as the rest of the
    # registry's MongoDB collections (multi-tenant deployments rely on this).
    from mongodb_groups_enrichment import _get_mongodb

    try:
        from registry.repositories.documentdb.client import get_collection_name

        collection_name = get_collection_name(COLLECTION_BASE_NAME)
    except Exception:
        # If the registry module is not importable from the auth_server's
        # process (unusual — we already share its DocumentDB client), fall
        # back to the unnamespaced base name. This matches the behavior of
        # other auth-server-side reads of registry collections.
        collection_name = COLLECTION_BASE_NAME

    db = await _get_mongodb()
    collection = db.get_collection(
        collection_name,
        write_concern=WriteConcern(w="majority"),
        read_preference=ReadPreference.PRIMARY_PREFERRED,
    )

    if not _indexes_created:
        try:
            await collection.create_index(
                [("session_id", ASCENDING)],
                unique=True,
                name="ux_session_id",
            )
            # TTL on a single date field. expireAfterSeconds=0 means "expire
            # documents at the time stored in the field" — supported by both
            # MongoDB and DocumentDB.
            await collection.create_index(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_expires_at",
            )
            _indexes_created = True
            logger.info(f"Created indexes for {collection_name} collection")
        except Exception as e:
            logger.warning(f"Could not create indexes for {collection_name}: {e}")

    _collection = collection
    return _collection


def generate_session_id() -> str:
    """Generate a fresh opaque session identifier."""
    return secrets.token_hex(SESSION_ID_BYTES)


async def create_session(
    username: str,
    email: str | None,
    name: str | None,
    groups: list[str],
    provider: str,
    auth_method: str,
    max_age_seconds: int,
    id_token: str | None = None,
    subject: str | None = None,
) -> str:
    """Persist a new session and return its opaque session_id.

    Args:
        username: Authenticated user identity (display / login id).
        email: Optional user email.
        name: Optional user display name.
        groups: External IdP groups (may be empty).
        provider: OAuth provider key (e.g. "keycloak", "entra").
        auth_method: Always "oauth2" today.
        max_age_seconds: Session lifetime — drives the TTL document field.
        id_token: Optional OIDC id_token, encrypted at rest. Required for
            SSO logout via id_token_hint; only pass None when the IdP did
            not return one.
        subject: The OIDC ``sub`` claim from the id_token. Persisted so the
            per-user egress vault can key on a stable subject that is present in
            BOTH id_tokens and access_tokens across providers — unlike
            ``username`` (preferred_username), which some IdPs (e.g. Entra) omit
            from access tokens, causing the browser-consent and bearer-vend paths
            to key on different identities. See canonical_egress_user.

    Returns:
        Opaque session_id to be embedded in the signed session cookie.
    """
    session_id = generate_session_id()
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=max_age_seconds)

    document: dict[str, Any] = {
        "session_id": session_id,
        "username": username,
        "email": email,
        "name": name,
        "groups": groups,
        "provider": provider,
        "auth_method": auth_method,
        "created_at": now,
        "expires_at": expires_at,
    }
    if subject:
        document["subject"] = subject
    if id_token:
        document["id_token_encrypted"] = encrypt_id_token(id_token)

    collection = await _get_collection()
    await collection.insert_one(document)
    logger.debug(f"Created session for user {username} (provider={provider})")
    return session_id


async def resolve_session(session_id: str) -> dict[str, Any] | None:
    """Look up a session by id, hydrating id_token if encrypted.

    Returns None on any failure (missing, expired, store unavailable). Callers
    must translate None into 401 and never propagate exceptions.
    """
    if not session_id:
        return None

    try:
        collection = await _get_collection()
        doc = await collection.find_one({"session_id": session_id})
    except Exception as e:
        logger.warning(f"Session store read failed: {e}")
        return None

    if not doc:
        return None

    expires_at = doc.get("expires_at")
    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= datetime.now(UTC):
            return None

    result: dict[str, Any] = {
        "session_id": doc["session_id"],
        "username": doc.get("username"),
        "email": doc.get("email"),
        "name": doc.get("name"),
        "groups": doc.get("groups", []),
        "provider": doc.get("provider"),
        "auth_method": doc.get("auth_method"),
        # OIDC sub, when the callback persisted it. The egress vault keys on this
        # stable subject (present in id_tokens and access_tokens) rather than
        # username, so the browser-consent and bearer-vend paths agree. See
        # canonical_egress_user.
        "subject": doc.get("subject"),
    }

    encrypted = doc.get("id_token_encrypted")
    if encrypted:
        try:
            result["id_token"] = decrypt_id_token(encrypted)
        except Exception as e:
            logger.warning(f"Failed to decrypt id_token for session: {e}")

    return result


async def delete_session(session_id: str) -> bool:
    """Delete a session record. Returns True if the document existed.

    Logout calls this before clearing the cookie so a stolen cookie cannot be
    replayed against the still-valid server record.
    """
    if not session_id:
        return False
    try:
        collection = await _get_collection()
        result = await collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.warning(f"Session store delete failed: {e}")
        return False
