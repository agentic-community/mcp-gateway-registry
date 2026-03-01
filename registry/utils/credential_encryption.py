"""
Backend MCP server credential encryption utilities.

Provides Fernet-based encryption and decryption for backend server auth
credentials (Bearer tokens, API keys) stored in server configurations.
Uses the application SECRET_KEY (via PBKDF2 key derivation) for encryption.

Follows the same pattern as federation_encryption.py but derives the Fernet
key from SECRET_KEY instead of requiring a separate environment variable.
"""

import logging
import base64
import hashlib
from datetime import datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# Salt for PBKDF2 key derivation (purpose-specific to avoid key reuse)
_KEY_DERIVATION_SALT: bytes = b"mcp-gateway-credential-encryption"

# PBKDF2 iteration count
_KEY_DERIVATION_ITERATIONS: int = 100_000

# Field names in server config dicts
PLAINTEXT_FIELD: str = "auth_credential"
ENCRYPTED_FIELD: str = "auth_credential_encrypted"


def _derive_fernet_key(
    secret_key: str,
) -> bytes:
    """Derive a Fernet-compatible key from the application SECRET_KEY using PBKDF2.

    Args:
        secret_key: Application SECRET_KEY string.

    Returns:
        32-byte url-safe base64-encoded key suitable for Fernet.
    """
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        secret_key.encode(),
        _KEY_DERIVATION_SALT,
        _KEY_DERIVATION_ITERATIONS,
    )
    return base64.urlsafe_b64encode(derived)


def _get_fernet() -> Optional[Fernet]:
    """Get a Fernet instance derived from the application SECRET_KEY.

    Returns:
        Fernet instance, or None if SECRET_KEY is not available.
    """
    try:
        from ..core.config import settings

        secret_key = settings.secret_key
    except Exception as e:
        logger.error(f"Could not load SECRET_KEY from settings: {e}")
        return None

    if not secret_key:
        return None

    try:
        key = _derive_fernet_key(secret_key)
        return Fernet(key)
    except Exception as e:
        logger.error(f"Failed to derive Fernet key from SECRET_KEY: {e}")
        return None


def encrypt_credential(
    credential: str,
) -> str:
    """Encrypt a backend server credential for storage.

    Args:
        credential: Plaintext credential (Bearer token or API key).

    Returns:
        Fernet-encrypted credential string (base64-encoded).

    Raises:
        ValueError: If SECRET_KEY is not configured or encryption fails.
    """
    fernet = _get_fernet()
    if not fernet:
        raise ValueError(
            "SECRET_KEY is not configured. Cannot encrypt credentials. "
            "Set SECRET_KEY in your environment or .env file."
        )

    encrypted = fernet.encrypt(credential.encode())
    return encrypted.decode()


def decrypt_credential(
    encrypted_credential: str,
) -> Optional[str]:
    """Decrypt a backend server credential from storage.

    Args:
        encrypted_credential: Fernet-encrypted credential string.

    Returns:
        Plaintext credential, or None if decryption fails.
    """
    fernet = _get_fernet()
    if not fernet:
        logger.error("SECRET_KEY not configured. Cannot decrypt server credential.")
        return None

    try:
        decrypted = fernet.decrypt(encrypted_credential.encode())
        return decrypted.decode()
    except InvalidToken:
        logger.error(
            "Failed to decrypt server credential. "
            "SECRET_KEY may have changed since the credential was stored. "
            "Re-register the server with a new credential."
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error decrypting server credential: {e}")
        return None


def encrypt_credential_in_server_dict(
    server_dict: dict,
) -> dict:
    """Encrypt auth_credential in a server dict before storage.

    If auth_credential is present and non-empty, encrypts it into
    auth_credential_encrypted and removes the plaintext field.
    Also sets credential_updated_at timestamp.

    Args:
        server_dict: Server config dictionary.

    Returns:
        Modified dict with encrypted credential (original dict is mutated).

    Raises:
        ValueError: If credential is present but encryption fails.
    """
    credential = server_dict.get(PLAINTEXT_FIELD)
    if not credential:
        server_dict.pop(PLAINTEXT_FIELD, None)
        return server_dict

    encrypted = encrypt_credential(credential)
    server_dict[ENCRYPTED_FIELD] = encrypted
    server_dict["credential_updated_at"] = datetime.now(timezone.utc).isoformat()

    # Remove plaintext from storage dict
    server_dict.pop(PLAINTEXT_FIELD, None)

    logger.info(
        f"Server credential encrypted for storage (path: {server_dict.get('path', 'unknown')})"
    )
    return server_dict


def strip_credentials_from_dict(
    server_dict: dict,
) -> dict:
    """Remove encrypted credentials from a server dict before returning in API responses.

    Args:
        server_dict: Server config dictionary.

    Returns:
        Modified dict with credentials removed (original dict is mutated).
    """
    server_dict.pop(ENCRYPTED_FIELD, None)
    server_dict.pop(PLAINTEXT_FIELD, None)
    return server_dict


def _migrate_auth_type_to_auth_scheme(
    server_dict: dict,
) -> dict:
    """Migrate legacy auth_type to auth_scheme on read.

    Converts old auth_type values to the new auth_scheme enum values.
    Does nothing if auth_scheme already exists.

    Args:
        server_dict: Server info dictionary from storage.

    Returns:
        Modified dict with auth_scheme populated from auth_type if needed.
    """
    if "auth_scheme" in server_dict:
        return server_dict

    auth_type = server_dict.get("auth_type")
    if not auth_type:
        return server_dict

    migration_map = {
        "none": "none",
        "oauth": "bearer",
        "api-key": "api_key",
        "api_key": "api_key",
        "custom": "bearer",
    }

    server_dict["auth_scheme"] = migration_map.get(auth_type, "none")
    return server_dict
