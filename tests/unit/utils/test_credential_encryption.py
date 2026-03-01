"""
Unit tests for registry.utils.credential_encryption.

Validates Fernet-based credential encryption, decryption, dict-level helpers,
credential stripping, and legacy auth_type to auth_scheme migration.
"""

import base64
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from registry.utils.credential_encryption import (
    _derive_fernet_key,
    _migrate_auth_type_to_auth_scheme,
    decrypt_credential,
    encrypt_credential,
    encrypt_credential_in_server_dict,
    strip_credentials_from_dict,
    ENCRYPTED_FIELD,
    PLAINTEXT_FIELD,
)


class TestDeriveFernetKey:
    """Tests for _derive_fernet_key."""

    def test_derive_fernet_key_produces_valid_key(self):
        """Verifies _derive_fernet_key returns a 44-byte base64-encoded key."""
        # Arrange
        secret = "test-secret-key-for-derivation"

        # Act
        key = _derive_fernet_key(secret)

        # Assert
        assert isinstance(key, bytes)
        assert len(key) == 44
        # Verify it is valid base64
        decoded = base64.urlsafe_b64decode(key)
        assert len(decoded) == 32

    def test_derive_fernet_key_deterministic(self):
        """Same secret must always produce the same key."""
        # Arrange
        secret = "reproducible-secret"

        # Act
        key_a = _derive_fernet_key(secret)
        key_b = _derive_fernet_key(secret)

        # Assert
        assert key_a == key_b

    def test_derive_fernet_key_different_secrets_produce_different_keys(self):
        """Different secrets must produce different keys."""
        # Act
        key_a = _derive_fernet_key("secret-one")
        key_b = _derive_fernet_key("secret-two")

        # Assert
        assert key_a != key_b


class TestEncryptDecryptRoundtrip:
    """Tests for encrypt_credential and decrypt_credential working together."""

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_encrypt_decrypt_roundtrip(self, mock_get_fernet):
        """Encrypt then decrypt returns original string."""
        # Arrange
        key = Fernet.generate_key()
        mock_get_fernet.return_value = Fernet(key)
        plaintext = "sk-abc123def456"

        # Act
        encrypted = encrypt_credential(plaintext)
        decrypted = decrypt_credential(encrypted)

        # Assert
        assert decrypted == plaintext
        assert encrypted != plaintext

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_encrypt_produces_different_ciphertext_each_time(self, mock_get_fernet):
        """Fernet includes a timestamp so each encryption is unique."""
        # Arrange
        key = Fernet.generate_key()
        mock_get_fernet.return_value = Fernet(key)
        plaintext = "my-api-key-value"

        # Act
        encrypted_a = encrypt_credential(plaintext)
        encrypted_b = encrypt_credential(plaintext)

        # Assert
        assert encrypted_a != encrypted_b


class TestEncryptCredentialErrors:
    """Tests for encrypt_credential error conditions."""

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_encrypt_credential_raises_without_secret_key(self, mock_get_fernet):
        """When no SECRET_KEY is available, encrypt raises ValueError."""
        # Arrange
        mock_get_fernet.return_value = None

        # Act / Assert
        with pytest.raises(ValueError, match="SECRET_KEY is not configured"):
            encrypt_credential("some-credential")


class TestDecryptCredentialErrors:
    """Tests for decrypt_credential error conditions."""

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_decrypt_credential_returns_none_without_secret_key(self, mock_get_fernet):
        """When no SECRET_KEY is available, decrypt returns None."""
        # Arrange
        mock_get_fernet.return_value = None

        # Act
        result = decrypt_credential("some-encrypted-token")

        # Assert
        assert result is None

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_decrypt_credential_returns_none_for_invalid_token(self, mock_get_fernet):
        """When token is garbage, decrypt returns None."""
        # Arrange
        key = Fernet.generate_key()
        mock_get_fernet.return_value = Fernet(key)

        # Act
        result = decrypt_credential("not-a-valid-fernet-token")

        # Assert
        assert result is None

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_decrypt_credential_returns_none_for_wrong_key(self, mock_get_fernet):
        """Token encrypted with a different key cannot be decrypted."""
        # Arrange - encrypt with one key
        key_a = Fernet.generate_key()
        fernet_a = Fernet(key_a)
        encrypted = fernet_a.encrypt(b"secret-data").decode()

        # Arrange - try to decrypt with a different key
        key_b = Fernet.generate_key()
        mock_get_fernet.return_value = Fernet(key_b)

        # Act
        result = decrypt_credential(encrypted)

        # Assert
        assert result is None


class TestEncryptCredentialInServerDict:
    """Tests for encrypt_credential_in_server_dict dict-level helper."""

    @patch("registry.utils.credential_encryption._get_fernet")
    def test_encrypt_credential_in_server_dict(self, mock_get_fernet):
        """Encrypts credential, removes plaintext, adds timestamp."""
        # Arrange
        key = Fernet.generate_key()
        mock_get_fernet.return_value = Fernet(key)
        server_dict = {
            "path": "/test-server",
            PLAINTEXT_FIELD: "bearer-token-12345",
        }

        # Act
        result = encrypt_credential_in_server_dict(server_dict)

        # Assert
        assert PLAINTEXT_FIELD not in result
        assert ENCRYPTED_FIELD in result
        assert "credential_updated_at" in result
        assert result["path"] == "/test-server"

        # Verify the encrypted value can be decrypted back
        decrypted = decrypt_credential(result[ENCRYPTED_FIELD])
        assert decrypted == "bearer-token-12345"

    def test_encrypt_credential_in_server_dict_no_credential(self):
        """Dict without credential is unchanged."""
        # Arrange
        server_dict = {
            "path": "/test-server",
            "transport": "streamable-http",
        }
        original_keys = set(server_dict.keys())

        # Act
        result = encrypt_credential_in_server_dict(server_dict)

        # Assert
        assert set(result.keys()) == original_keys
        assert ENCRYPTED_FIELD not in result
        assert PLAINTEXT_FIELD not in result

    def test_encrypt_credential_in_server_dict_empty_credential(self):
        """Dict with empty string credential has the plaintext field removed."""
        # Arrange
        server_dict = {
            "path": "/test-server",
            PLAINTEXT_FIELD: "",
        }

        # Act
        result = encrypt_credential_in_server_dict(server_dict)

        # Assert
        assert PLAINTEXT_FIELD not in result
        assert ENCRYPTED_FIELD not in result


class TestStripCredentialsFromDict:
    """Tests for strip_credentials_from_dict."""

    def test_strip_credentials_from_dict(self):
        """Removes both encrypted and plaintext credential fields."""
        # Arrange
        server_dict = {
            "path": "/test-server",
            PLAINTEXT_FIELD: "my-secret-token",
            ENCRYPTED_FIELD: "gAAAAABf_encrypted_data",
            "credential_updated_at": "2025-01-01T00:00:00+00:00",
        }

        # Act
        result = strip_credentials_from_dict(server_dict)

        # Assert
        assert PLAINTEXT_FIELD not in result
        assert ENCRYPTED_FIELD not in result
        assert result["path"] == "/test-server"
        assert "credential_updated_at" in result

    def test_strip_credentials_from_dict_no_credential_fields(self):
        """Dict without credential fields is returned unchanged."""
        # Arrange
        server_dict = {
            "path": "/test-server",
            "transport": "streamable-http",
        }

        # Act
        result = strip_credentials_from_dict(server_dict)

        # Assert
        assert result == {"path": "/test-server", "transport": "streamable-http"}


class TestMigrateAuthTypeToAuthScheme:
    """Tests for _migrate_auth_type_to_auth_scheme."""

    def test_migrate_auth_type_oauth(self):
        """auth_type='oauth' should map to auth_scheme='bearer'."""
        # Arrange
        server_dict = {"auth_type": "oauth"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "bearer"

    def test_migrate_auth_type_api_key(self):
        """auth_type='api-key' (hyphenated) should map to auth_scheme='api_key'."""
        # Arrange
        server_dict = {"auth_type": "api-key"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "api_key"

    def test_migrate_auth_type_api_key_underscore(self):
        """auth_type='api_key' (underscore) should map to auth_scheme='api_key'."""
        # Arrange
        server_dict = {"auth_type": "api_key"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "api_key"

    def test_migrate_auth_type_none(self):
        """auth_type='none' should map to auth_scheme='none'."""
        # Arrange
        server_dict = {"auth_type": "none"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "none"

    def test_migrate_auth_type_custom(self):
        """auth_type='custom' should map to auth_scheme='bearer'."""
        # Arrange
        server_dict = {"auth_type": "custom"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "bearer"

    def test_migrate_auth_type_unknown_defaults_to_none(self):
        """Unknown auth_type value should default to auth_scheme='none'."""
        # Arrange
        server_dict = {"auth_type": "something-unknown"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "none"

    def test_migrate_no_overwrite(self):
        """If auth_scheme already exists, migration does not overwrite."""
        # Arrange
        server_dict = {
            "auth_type": "oauth",
            "auth_scheme": "api_key",
        }

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert result["auth_scheme"] == "api_key"

    def test_migrate_no_auth_type(self):
        """Dict without auth_type is unchanged."""
        # Arrange
        server_dict = {"path": "/test-server"}

        # Act
        result = _migrate_auth_type_to_auth_scheme(server_dict)

        # Assert
        assert "auth_scheme" not in result
        assert result == {"path": "/test-server"}
