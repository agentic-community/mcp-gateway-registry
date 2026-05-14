"""
Unit tests for custom headers encryption/decryption in credential_encryption.py.
"""

from unittest.mock import patch

import pytest

from registry.utils.credential_encryption import (
    CUSTOM_HEADERS_ENCRYPTED_FIELD,
    CUSTOM_HEADERS_PLAINTEXT_FIELD,
    CUSTOM_HEADER_NAMES_FIELD,
    decrypt_custom_headers,
    encrypt_credential,
    encrypt_custom_headers_in_server_dict,
    strip_credentials_from_dict,
)


@pytest.fixture
def mock_secret_key():
    """Patch settings.secret_key for encryption tests."""
    with patch("registry.utils.credential_encryption._get_fernet") as mock_fernet:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        mock_fernet.return_value = Fernet(key)
        yield


class TestEncryptCustomHeaders:
    """Tests for encrypt_custom_headers_in_server_dict."""

    def test_encrypts_headers_successfully(self, mock_secret_key):
        server_dict = {
            "path": "/test-server",
            "custom_headers": [
                {"name": "X-Tenant-Id", "value": "42"},
                {"name": "X-Route-Cluster", "value": "prod-us-east"},
            ],
        }

        result = encrypt_custom_headers_in_server_dict(server_dict)

        assert CUSTOM_HEADERS_PLAINTEXT_FIELD not in result
        assert CUSTOM_HEADERS_ENCRYPTED_FIELD in result
        assert CUSTOM_HEADER_NAMES_FIELD in result
        assert result[CUSTOM_HEADER_NAMES_FIELD] == ["X-Tenant-Id", "X-Route-Cluster"]
        assert len(result[CUSTOM_HEADERS_ENCRYPTED_FIELD]) == 2
        assert result[CUSTOM_HEADERS_ENCRYPTED_FIELD][0]["name"] == "X-Tenant-Id"
        assert "value_encrypted" in result[CUSTOM_HEADERS_ENCRYPTED_FIELD][0]
        assert "custom_headers_updated_at" in result

    def test_no_custom_headers_field_is_noop(self, mock_secret_key):
        server_dict = {"path": "/test-server"}

        result = encrypt_custom_headers_in_server_dict(server_dict)

        assert CUSTOM_HEADERS_ENCRYPTED_FIELD not in result
        assert result == {"path": "/test-server"}

    def test_rejects_non_list(self, mock_secret_key):
        server_dict = {"custom_headers": "not-a-list"}

        with pytest.raises(ValueError, match="must be a list"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_rejects_non_dict_entry(self, mock_secret_key):
        server_dict = {"custom_headers": ["not-a-dict"]}

        with pytest.raises(ValueError, match="must be an object"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_rejects_empty_name(self, mock_secret_key):
        server_dict = {"custom_headers": [{"name": "", "value": "v"}]}

        with pytest.raises(ValueError, match="non-empty name and value"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_rejects_empty_value(self, mock_secret_key):
        server_dict = {"custom_headers": [{"name": "X-Foo", "value": ""}]}

        with pytest.raises(ValueError, match="non-empty name and value"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_rejects_duplicate_names(self, mock_secret_key):
        server_dict = {
            "custom_headers": [
                {"name": "X-Foo", "value": "a"},
                {"name": "x-foo", "value": "b"},
            ]
        }

        with pytest.raises(ValueError, match="Duplicate"):
            encrypt_custom_headers_in_server_dict(server_dict)


class TestDecryptCustomHeaders:
    """Tests for decrypt_custom_headers."""

    def test_round_trip(self, mock_secret_key):
        server_dict = {
            "path": "/test",
            "custom_headers": [
                {"name": "X-Tenant-Id", "value": "42"},
                {"name": "X-Secret", "value": "abc123"},
            ],
        }
        encrypt_custom_headers_in_server_dict(server_dict)

        decrypted = decrypt_custom_headers(server_dict[CUSTOM_HEADERS_ENCRYPTED_FIELD])

        assert len(decrypted) == 2
        assert decrypted[0] == {"name": "X-Tenant-Id", "value": "42"}
        assert decrypted[1] == {"name": "X-Secret", "value": "abc123"}

    def test_empty_list(self, mock_secret_key):
        assert decrypt_custom_headers([]) == []

    def test_none_input(self, mock_secret_key):
        assert decrypt_custom_headers(None) == []

    def test_skips_invalid_entries(self, mock_secret_key):
        encrypted_list = [
            {"name": "X-Valid", "value_encrypted": encrypt_credential("good")},
            {"name": "X-Bad", "value_encrypted": "invalid-ciphertext"},
            {"name": "", "value_encrypted": "something"},
        ]

        result = decrypt_custom_headers(encrypted_list)

        assert len(result) == 1
        assert result[0]["name"] == "X-Valid"
        assert result[0]["value"] == "good"


class TestStripCredentials:
    """Tests that strip_credentials_from_dict removes custom header fields."""

    def test_strips_custom_headers_encrypted(self):
        server_dict = {
            "path": "/test",
            "custom_headers_encrypted": [{"name": "X-Foo", "value_encrypted": "abc"}],
            "custom_header_names": ["X-Foo"],
            "custom_headers": [{"name": "X-Foo", "value": "bar"}],
        }

        result = strip_credentials_from_dict(server_dict)

        assert "custom_headers_encrypted" not in result
        assert "custom_headers" not in result
        assert "custom_header_names" in result
