"""
Backend MCP server credential encryption utilities.

Provides Fernet-based encryption and decryption for backend server auth
credentials (Bearer tokens, API keys) stored in server configurations.
Uses the application SECRET_KEY (via PBKDF2 key derivation) for encryption.

Follows the same pattern as federation_encryption.py but derives the Fernet
key from SECRET_KEY instead of requiring a separate environment variable.
"""

import base64
import hashlib
import logging
import re
from datetime import UTC, datetime

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# Salt for PBKDF2 key derivation (purpose-specific to avoid key reuse)
_KEY_DERIVATION_SALT: bytes = b"mcp-gateway-credential-encryption"

# PBKDF2 iteration count
_KEY_DERIVATION_ITERATIONS: int = 100_000

# Field names in server config dicts
PLAINTEXT_FIELD: str = "auth_credential"
ENCRYPTED_FIELD: str = "auth_credential_encrypted"

# Custom headers field names
CUSTOM_HEADERS_PLAINTEXT_FIELD: str = "custom_headers"
CUSTOM_HEADERS_ENCRYPTED_FIELD: str = "custom_headers_encrypted"
CUSTOM_HEADER_NAMES_FIELD: str = "custom_header_names"

MAX_CUSTOM_HEADER_NAME_LENGTH: int = 256
MAX_CUSTOM_HEADER_VALUE_LENGTH: int = 4096
_RFC_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def validate_custom_header_name(name: object) -> str:
    """Return a bounded RFC token header name or raise ``ValueError``."""
    if not isinstance(name, str):
        raise ValueError("custom_headers entry name must be a string")
    if not name:
        raise ValueError("custom_headers entry requires a non-empty name")
    if len(name) > MAX_CUSTOM_HEADER_NAME_LENGTH:
        raise ValueError(
            f"custom_headers entry name exceeds {MAX_CUSTOM_HEADER_NAME_LENGTH} characters"
        )
    if not _RFC_TOKEN_RE.fullmatch(name):
        raise ValueError("custom_headers entry name must be a valid RFC token string")
    return name


def _validate_custom_header_value(value: object, *, allow_empty: bool = False) -> str:
    """Return a bounded, control-free string header value."""
    if not isinstance(value, str):
        raise ValueError("custom_headers entry value must be a string")
    if not value and not allow_empty:
        raise ValueError("custom_headers entry requires a non-empty value")
    if len(value) > MAX_CUSTOM_HEADER_VALUE_LENGTH:
        raise ValueError(
            f"custom_headers entry value exceeds {MAX_CUSTOM_HEADER_VALUE_LENGTH} characters"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError("custom_headers entry value cannot contain control characters")
    return value


def validate_custom_headers(
    raw: list[dict] | None,
    *,
    allow_empty_values: bool = False,
) -> list[dict] | None:
    """Validate existing MCP custom headers before storage or use."""
    if raw is None:
        return None

    from registry.constants import MAX_CUSTOM_HEADERS_PER_SERVER, RESERVED_CUSTOM_HEADER_NAMES

    if not isinstance(raw, list):
        raise ValueError("custom_headers must be a list")
    if len(raw) > MAX_CUSTOM_HEADERS_PER_SERVER:
        raise ValueError(
            f"Too many custom headers: got {len(raw)}, maximum is {MAX_CUSTOM_HEADERS_PER_SERVER}"
        )

    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("custom_headers entry must be an object")
        name = validate_custom_header_name(item.get("name"))
        _validate_custom_header_value(item.get("value"), allow_empty=allow_empty_values)
        lower = name.lower()
        if lower in RESERVED_CUSTOM_HEADER_NAMES:
            raise ValueError(
                f"Header '{name}' is managed by the gateway and cannot be set as a custom header"
            )
        if lower in seen:
            raise ValueError(f"Duplicate custom header name: {name}")
        seen.add(lower)
    return raw


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


def _get_fernet() -> Fernet | None:
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
) -> str | None:
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
    server_dict["credential_updated_at"] = datetime.now(UTC).isoformat()

    # Remove plaintext from storage dict
    server_dict.pop(PLAINTEXT_FIELD, None)

    logger.info(
        f"Server credential encrypted for storage (path: {server_dict.get('path', 'unknown')})"
    )
    return server_dict


_SERVER_RESPONSE_SECRET_FIELDS: frozenset[str] = frozenset(
    {
        ENCRYPTED_FIELD,
        PLAINTEXT_FIELD,
        CUSTOM_HEADERS_ENCRYPTED_FIELD,
        CUSTOM_HEADERS_PLAINTEXT_FIELD,
        "client_secret",
        "client_secret_encrypted",
    }
)


def _token_free_projection(value: object) -> object:
    """Recursively copy a response value while omitting known secret fields."""
    if isinstance(value, dict):
        return {
            key: _token_free_projection(item)
            for key, item in value.items()
            if key not in _SERVER_RESPONSE_SECRET_FIELDS
        }
    if isinstance(value, list):
        return [_token_free_projection(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_token_free_projection(item) for item in value)
    return value


def strip_credentials_from_dict(
    server_dict: dict,
) -> dict:
    """Return a recursive token-free copy for server API responses.

    Top-level backend credentials, encrypted custom-header values, nested
    per-version credentials, and ``egress_oauth.client_secret_encrypted`` are
    removed. The input and all shared nested dictionaries/lists are left
    untouched so redacting one response cannot corrupt repository/cache state.
    """
    projected = _token_free_projection(server_dict)
    if not isinstance(projected, dict):  # pragma: no cover - input type contract
        return {}
    return projected


def encrypt_custom_headers_in_server_dict(
    server_dict: dict,
) -> dict:
    """Validate and encrypt custom-header values before storage."""
    raw = server_dict.get(CUSTOM_HEADERS_PLAINTEXT_FIELD)
    if raw is None:
        return server_dict

    validate_custom_headers(raw)
    encrypted_list: list[dict[str, str]] = []
    names: list[str] = []
    for item in raw:
        name = validate_custom_header_name(item.get("name"))
        value = _validate_custom_header_value(item.get("value"))
        encrypted_list.append({"name": name, "value_encrypted": encrypt_credential(value)})
        names.append(name)

    server_dict[CUSTOM_HEADERS_ENCRYPTED_FIELD] = encrypted_list
    server_dict[CUSTOM_HEADER_NAMES_FIELD] = names
    server_dict["custom_headers_updated_at"] = datetime.now(UTC).isoformat()
    server_dict.pop(CUSTOM_HEADERS_PLAINTEXT_FIELD, None)

    logger.info(
        f"Custom headers encrypted for storage (path: {server_dict.get('path', 'unknown')}, count: {len(names):d})",
    )
    return server_dict


def decrypt_custom_headers(
    encrypted_list: list[dict] | None,
) -> list[dict]:
    """Best-effort decrypt safe stored custom headers, skipping bad entries."""
    from registry.constants import RESERVED_CUSTOM_HEADER_NAMES

    if not encrypted_list:
        return []
    if not isinstance(encrypted_list, list):
        logger.warning("Stored custom headers are not a list; skipping.")
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in encrypted_list:
        if not isinstance(item, dict):
            logger.warning("Stored custom header entry is not an object; skipping.")
            continue
        try:
            name = validate_custom_header_name(item.get("name"))
        except ValueError:
            logger.warning("Stored custom header has an invalid name; skipping.")
            continue
        lower = name.lower()
        if lower in RESERVED_CUSTOM_HEADER_NAMES:
            logger.warning(f"Stored custom header '{name}' is gateway-managed; skipping.")
            continue
        if lower in seen:
            logger.warning(f"Stored custom header '{name}' is duplicated; skipping.")
            continue
        encrypted = item.get("value_encrypted")
        if not isinstance(encrypted, str) or not encrypted:
            logger.warning(f"Stored custom header '{name}' has no ciphertext; skipping.")
            continue
        value = decrypt_credential(encrypted)
        if value is None:
            logger.warning(f"Failed to decrypt custom header '{name}'; skipping.")
            continue
        try:
            _validate_custom_header_value(value, allow_empty=True)
        except ValueError:
            logger.warning(f"Stored custom header '{name}' has an unsafe value; skipping.")
            continue
        seen.add(lower)
        out.append({"name": name, "value": value})
    return out


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
