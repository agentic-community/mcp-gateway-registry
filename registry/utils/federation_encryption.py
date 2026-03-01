"""
Federation token encryption utilities.

Provides Fernet-based encryption and decryption for federation static tokens
stored in peer registry configurations (MongoDB/file). Uses the
FEDERATION_ENCRYPTION_KEY environment variable as the encryption key.

The encryption key must be a valid Fernet key (32 url-safe base64-encoded bytes).
Generate one with: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# Environment variable name for the encryption key
FEDERATION_ENCRYPTION_KEY_ENV: str = "FEDERATION_ENCRYPTION_KEY"

# Field names in peer config dicts
PLAINTEXT_FIELD: str = "federation_token"
ENCRYPTED_FIELD: str = "federation_token_encrypted"


def _get_fernet() -> Optional[Fernet]:
    """Get a Fernet instance from the FEDERATION_ENCRYPTION_KEY env var.

    Returns:
        Fernet instance, or None if key is not configured.
    """
    key = os.environ.get(FEDERATION_ENCRYPTION_KEY_ENV)
    if not key:
        return None

    try:
        return Fernet(key.encode())
    except Exception as e:
        logger.error(
            f"Invalid {FEDERATION_ENCRYPTION_KEY_ENV}: {e}. "
            "Generate a valid key with: python3 -c "
            '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )
        return None


def encrypt_federation_token(
    token: str,
) -> str:
    """Encrypt a federation token for storage.

    Args:
        token: Plaintext federation token.

    Returns:
        Fernet-encrypted token string (base64-encoded).

    Raises:
        ValueError: If FEDERATION_ENCRYPTION_KEY is not set or invalid.
    """
    fernet = _get_fernet()
    if not fernet:
        raise ValueError(
            f"{FEDERATION_ENCRYPTION_KEY_ENV} environment variable is not set or invalid. "
            "Cannot encrypt federation token for storage. "
            "Generate a key with: python3 -c "
            '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )

    encrypted = fernet.encrypt(token.encode())
    return encrypted.decode()


def decrypt_federation_token(
    encrypted_token: str,
) -> Optional[str]:
    """Decrypt a federation token from storage.

    Args:
        encrypted_token: Fernet-encrypted token string.

    Returns:
        Plaintext federation token, or None if decryption fails.
    """
    fernet = _get_fernet()
    if not fernet:
        logger.error(
            f"{FEDERATION_ENCRYPTION_KEY_ENV} not set. Cannot decrypt federation token. "
            "Peer sync will fail for peers using federation static tokens."
        )
        return None

    try:
        decrypted = fernet.decrypt(encrypted_token.encode())
        return decrypted.decode()
    except InvalidToken:
        logger.error(
            "Failed to decrypt federation token. The encryption key may have changed "
            "since the token was stored. Re-add the peer with the correct token."
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error decrypting federation token: {e}")
        return None


def encrypt_token_in_peer_dict(
    peer_dict: dict,
) -> dict:
    """Encrypt federation_token in a peer config dict before storage.

    If federation_token is present and non-empty, encrypts it into
    federation_token_encrypted and removes the plaintext field.

    If FEDERATION_ENCRYPTION_KEY is not set but a token is present,
    raises ValueError to prevent storing plaintext secrets.

    Args:
        peer_dict: Peer config dictionary (from model_dump).

    Returns:
        Modified dict with encrypted token (original dict is mutated).

    Raises:
        ValueError: If token is present but encryption key is not configured.
    """
    token = peer_dict.get(PLAINTEXT_FIELD)
    if not token:
        # No token to encrypt, remove plaintext field if present
        peer_dict.pop(PLAINTEXT_FIELD, None)
        return peer_dict

    # Encrypt the token
    encrypted = encrypt_federation_token(token)
    peer_dict[ENCRYPTED_FIELD] = encrypted

    # Remove plaintext from storage dict
    peer_dict.pop(PLAINTEXT_FIELD, None)

    logger.info("Federation token encrypted for storage")
    return peer_dict


def decrypt_token_in_peer_dict(
    peer_dict: dict,
) -> dict:
    """Decrypt federation_token_encrypted in a peer config dict after loading.

    If federation_token_encrypted is present, decrypts it into
    federation_token for use by PeerRegistryClient.

    Args:
        peer_dict: Peer config dictionary (from MongoDB/file).

    Returns:
        Modified dict with decrypted token (original dict is mutated).
    """
    encrypted_token = peer_dict.get(ENCRYPTED_FIELD)
    if not encrypted_token:
        return peer_dict

    # Decrypt the token
    decrypted = decrypt_federation_token(encrypted_token)
    if decrypted:
        peer_dict[PLAINTEXT_FIELD] = decrypted
    else:
        logger.warning(
            "Could not decrypt federation token. Peer sync will fall back to global OAuth2 auth."
        )

    # Remove encrypted field from the dict before constructing PeerRegistryConfig
    # (PeerRegistryConfig doesn't have a federation_token_encrypted field)
    peer_dict.pop(ENCRYPTED_FIELD, None)

    return peer_dict
