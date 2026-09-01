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
CUSTOM_HEADER_OVERRIDABLE_NAMES_FIELD: str = "custom_header_overridable_names"

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


_CALLER_OVERRIDABLE_RESERVED_NAMES: frozenset[str] = frozenset({"authorization"})


def _validate_custom_header_overridable(value: object) -> bool:
    """Require an explicit boolean override flag; reject ambiguous truthy values."""
    if not isinstance(value, bool):
        raise ValueError("custom_headers entry overridable must be a boolean")
    return value


def validate_custom_headers(
    raw: list[dict] | None,
    *,
    allow_empty_values: bool = False,
) -> list[dict] | None:
    """Validate bounded, control-free upstream headers and override metadata.

    A fixed header requires an operator value. An ``overridable`` header may
    instead be caller-only (no default value). Gateway-managed names remain
    forbidden except ``Authorization``, which is accepted only as caller-
    overridable; fixed credentials belong in the egress credential vault.
    """
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
        value = item.get("value")
        overridable = _validate_custom_header_overridable(item.get("overridable", False))
        if value in (None, ""):
            if not overridable and not allow_empty_values:
                raise ValueError(
                    f"custom_headers entry '{name}' has no value and is not overridable"
                )
        else:
            _validate_custom_header_value(value)

        lower = name.lower()
        if lower in RESERVED_CUSTOM_HEADER_NAMES:
            if lower not in _CALLER_OVERRIDABLE_RESERVED_NAMES:
                raise ValueError(
                    f"Header '{name}' is managed by the gateway and cannot be set as a custom header"
                )
            if not overridable:
                raise ValueError(
                    f"Header '{name}' may only be a caller-overridable header; "
                    "fixed credentials belong in the egress credential vault"
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
    """Validate, encrypt defaults, and store caller-override header metadata."""
    raw = server_dict.get(CUSTOM_HEADERS_PLAINTEXT_FIELD)
    if raw is None:
        return server_dict

    # Validate before mutating the input so malformed data cannot leave a
    # partially encrypted record behind.
    validate_custom_headers(raw)
    encrypted_list: list[dict[str, str]] = []
    names: list[str] = []
    overridable_names: list[str] = []
    for item in raw:
        name = validate_custom_header_name(item.get("name"))
        value = item.get("value")
        overridable = _validate_custom_header_overridable(item.get("overridable", False))
        if value not in (None, ""):
            value = _validate_custom_header_value(value)
            encrypted_list.append({"name": name, "value_encrypted": encrypt_credential(value)})
        names.append(name)
        if overridable:
            overridable_names.append(name)

    server_dict[CUSTOM_HEADERS_ENCRYPTED_FIELD] = encrypted_list
    server_dict[CUSTOM_HEADER_NAMES_FIELD] = names
    server_dict[CUSTOM_HEADER_OVERRIDABLE_NAMES_FIELD] = overridable_names
    server_dict["custom_headers_updated_at"] = datetime.now(UTC).isoformat()
    server_dict.pop(CUSTOM_HEADERS_PLAINTEXT_FIELD, None)

    logger.info(
        f"Custom headers encrypted for storage (path: {server_dict.get('path', 'unknown')}, "
        f"count: {len(names):d}, overridable: {len(overridable_names):d})"
    )
    return server_dict


def build_custom_headers_storage_fields(
    raw: list[dict] | None,
    existing_encrypted: list[dict] | None = None,
) -> dict:
    """Validate + encrypt a plaintext header list into the four storage fields.

    Shared by the dedicated header-rotation endpoints (skill + custom entity),
    which -- unlike create -- must produce a self-contained ``$set`` of ALL header
    storage fields, including the CLEAR case (an empty/None list removes every
    stored header).

    Write-only value convention (mirrors the 3LO egress ``client_secret`` and the
    MCP-server custom-header edit path): stored header VALUES are never returned to
    the client, so on edit each row arrives with a BLANK value. A blank value on a
    row whose name already has a stored ciphertext means "keep the existing value"
    -- the prior ciphertext is decrypted and carried forward, so an unchanged edit
    does not wipe the secret. A blank value with NO prior ciphertext is only legal
    when the row is ``overridable`` (a caller-only passthrough slot); otherwise it
    is rejected (nothing to inject, nothing to preserve). After the preserve-merge,
    the result is run through ``validate_custom_headers`` (full policy) and
    encrypted.

    Args:
        raw: The plaintext ``[{name, value?, overridable?}, ...]`` list. An empty
            list or None means "remove all upstream headers".
        existing_encrypted: The entity's current ``custom_headers_encrypted`` list
            (``[{name, value_encrypted}, ...]``), used to preserve a header whose
            submitted value is blank. None/absent = no priors (every blank
            non-overridable row is then a policy error).

    Returns:
        A dict with exactly these keys, safe to merge into an entity update:
        ``custom_headers_encrypted`` (list, [] when cleared),
        ``custom_header_names`` (list, [] when cleared),
        ``custom_header_overridable_names`` (list, [] when cleared),
        ``custom_headers_updated_at`` (ISO timestamp).

    Raises:
        ValueError: on any policy violation, a blank non-preservable value, or
            encryption failure (the caller maps it to a 400).
    """
    now = datetime.now(UTC).isoformat()
    if not raw:
        # Clear case: remove every stored header (and stamp the update time).
        return {
            CUSTOM_HEADERS_ENCRYPTED_FIELD: [],
            CUSTOM_HEADER_NAMES_FIELD: [],
            CUSTOM_HEADER_OVERRIDABLE_NAMES_FIELD: [],
            "custom_headers_updated_at": now,
        }

    if not isinstance(raw, list):
        raise ValueError("custom_headers must be a list")

    # Preserve-by-name merge: a blank value inherits the prior ciphertext's
    # plaintext so an unchanged edit keeps the secret (write-only value UX).
    existing_by_name: dict[str, dict] = {
        e["name"]: e for e in (existing_encrypted or []) if isinstance(e, dict) and e.get("name")
    }
    merged: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("custom_headers entry must be an object")
        name = item.get("name")
        value = item.get("value")
        overridable = _validate_custom_header_overridable(item.get("overridable", False))
        if name and not value:
            prior = existing_by_name.get(name)
            if prior is not None:
                plaintext = decrypt_credential(prior.get("value_encrypted", ""))
                if plaintext is None:
                    raise ValueError(f"Could not preserve the existing value for header '{name}'")
                value = plaintext
            # No prior: a blank overridable row is a legitimate caller-only slot
            # (validate_custom_headers accepts it); a blank non-overridable row is
            # rejected there. Leave value blank and let validation decide.
        merged.append({"name": name, "value": value, "overridable": overridable})

    validate_custom_headers(merged)
    tmp: dict = {CUSTOM_HEADERS_PLAINTEXT_FIELD: merged}
    encrypt_custom_headers_in_server_dict(tmp)
    return {
        CUSTOM_HEADERS_ENCRYPTED_FIELD: tmp.get(CUSTOM_HEADERS_ENCRYPTED_FIELD, []),
        CUSTOM_HEADER_NAMES_FIELD: tmp.get(CUSTOM_HEADER_NAMES_FIELD, []),
        CUSTOM_HEADER_OVERRIDABLE_NAMES_FIELD: tmp.get(CUSTOM_HEADER_OVERRIDABLE_NAMES_FIELD, []),
        "custom_headers_updated_at": now,
    }


def decrypt_custom_headers(
    encrypted_list: list[dict] | None,
    *,
    strict: bool = False,
) -> list[dict]:
    """Decrypt safe stored headers, optionally failing closed on any bad entry."""
    from registry.constants import RESERVED_CUSTOM_HEADER_NAMES

    if not encrypted_list:
        return []
    if not isinstance(encrypted_list, list):
        if strict:
            raise ValueError("custom_headers_encrypted must be a list")
        logger.warning("Stored custom headers are not a list; skipping.")
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in encrypted_list:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("stored custom header entry must be an object")
            logger.warning("Stored custom header entry is not an object; skipping.")
            continue
        try:
            name = validate_custom_header_name(item.get("name"))
        except ValueError:
            if strict:
                raise ValueError("stored custom header has an invalid name") from None
            logger.warning("Stored custom header has an invalid name; skipping.")
            continue
        lower = name.lower()
        if lower in RESERVED_CUSTOM_HEADER_NAMES:
            if strict:
                raise ValueError(f"stored custom header '{name}' is gateway-managed")
            logger.warning(f"Stored custom header '{name}' is gateway-managed; skipping.")
            continue
        if lower in seen:
            if strict:
                raise ValueError(f"stored custom header '{name}' is duplicated")
            logger.warning(f"Stored custom header '{name}' is duplicated; skipping.")
            continue
        encrypted = item.get("value_encrypted")
        if not isinstance(encrypted, str) or not encrypted:
            if strict:
                raise ValueError(f"stored custom header '{name}' has no ciphertext")
            logger.warning(f"Stored custom header '{name}' has no ciphertext; skipping.")
            continue
        value = decrypt_credential(encrypted)
        if value is None:
            if strict:
                raise ValueError(f"failed to decrypt stored custom header '{name}'")
            logger.warning(f"Failed to decrypt custom header '{name}'; skipping.")
            continue
        try:
            _validate_custom_header_value(value, allow_empty=True)
        except ValueError:
            if strict:
                raise ValueError(f"stored custom header '{name}' has an unsafe value") from None
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
