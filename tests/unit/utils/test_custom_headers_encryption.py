"""
Unit tests for custom headers encryption/decryption in credential_encryption.py.
"""

from unittest.mock import patch

import pytest

from registry.utils.credential_encryption import (
    CUSTOM_HEADER_NAMES_FIELD,
    CUSTOM_HEADERS_ENCRYPTED_FIELD,
    CUSTOM_HEADERS_PLAINTEXT_FIELD,
    build_custom_headers_storage_fields,
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

        with pytest.raises(ValueError, match="non-empty name"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_rejects_empty_value_when_not_overridable(self, mock_secret_key):
        # A value-less entry that is NOT overridable is meaningless (nothing to
        # inject, caller cannot supply it) -> rejected.
        server_dict = {"custom_headers": [{"name": "X-Foo", "value": ""}]}

        with pytest.raises(ValueError, match="no value and is not overridable"):
            encrypt_custom_headers_in_server_dict(server_dict)

    def test_accepts_value_less_overridable_slot(self, mock_secret_key):
        # A value-less overridable entry is a caller-only passthrough slot: it
        # contributes a name (+ overridable name) but no encrypted value.
        server_dict = {"custom_headers": [{"name": "X-Tenant", "overridable": True}]}

        result = encrypt_custom_headers_in_server_dict(server_dict)

        assert result[CUSTOM_HEADERS_ENCRYPTED_FIELD] == []
        assert result["custom_header_names"] == ["X-Tenant"]
        assert result["custom_header_overridable_names"] == ["X-Tenant"]

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

    def test_strict_mode_fails_on_invalid_ciphertext(self, mock_secret_key):
        with pytest.raises(ValueError, match="failed to decrypt"):
            decrypt_custom_headers(
                [{"name": "X-Bad", "value_encrypted": "invalid-ciphertext"}],
                strict=True,
            )

    def test_strict_mode_fails_on_malformed_entry(self, mock_secret_key):
        with pytest.raises(ValueError, match="invalid name"):
            decrypt_custom_headers(
                [{"name": "X-Bad\rInjected", "value_encrypted": "ciphertext"}],
                strict=True,
            )

    def test_non_list_default_returns_empty(self, mock_secret_key):
        # Non-strict default: a non-list (but truthy) input is logged and dropped.
        assert decrypt_custom_headers("x") == []

    def test_strict_mode_fails_on_non_list(self, mock_secret_key):
        with pytest.raises(ValueError, match="must be a list"):
            decrypt_custom_headers("x", strict=True)

    def test_strict_mode_fails_on_non_object_entry(self, mock_secret_key):
        with pytest.raises(ValueError, match="must be an object"):
            decrypt_custom_headers([123], strict=True)

    def test_strict_mode_fails_on_reserved_name(self, mock_secret_key):
        # "content-type" is a gateway-managed reserved header name.
        with pytest.raises(ValueError, match="gateway-managed"):
            decrypt_custom_headers(
                [{"name": "Content-Type", "value_encrypted": encrypt_credential("x")}],
                strict=True,
            )

    def test_strict_mode_fails_on_duplicate_name(self, mock_secret_key):
        cipher = encrypt_credential("v")
        with pytest.raises(ValueError, match="duplicated"):
            decrypt_custom_headers(
                [
                    {"name": "X-Dup", "value_encrypted": cipher},
                    {"name": "x-dup", "value_encrypted": encrypt_credential("w")},
                ],
                strict=True,
            )

    def test_strict_mode_fails_on_missing_ciphertext(self, mock_secret_key):
        with pytest.raises(ValueError, match="no ciphertext"):
            decrypt_custom_headers(
                [{"name": "X-NoCipher", "value_encrypted": ""}],
                strict=True,
            )

    def test_strict_mode_fails_on_unsafe_value(self, mock_secret_key):
        # A control character survives encryption but is rejected on decrypt.
        cipher = encrypt_credential("bad\nvalue")
        with pytest.raises(ValueError, match="unsafe value"):
            decrypt_custom_headers(
                [{"name": "X-Unsafe", "value_encrypted": cipher}],
                strict=True,
            )


class TestBuildCustomHeadersStorageFields:
    """Tests for build_custom_headers_storage_fields error paths."""

    def test_non_list_raw_raises(self, mock_secret_key):
        # A non-empty, non-list raw reaches the isinstance guard.
        with pytest.raises(ValueError, match="must be a list"):
            build_custom_headers_storage_fields("notalist")

    def test_non_dict_entry_raises(self, mock_secret_key):
        with pytest.raises(ValueError, match="entry must be an object"):
            build_custom_headers_storage_fields(["notadict"])

    def test_blank_value_with_undecryptable_prior_raises(self, mock_secret_key):
        # Blank value -> preserve-by-name, but the stored ciphertext is garbage
        # that decrypt_credential cannot recover, so preservation fails.
        with pytest.raises(ValueError, match="Could not preserve"):
            build_custom_headers_storage_fields(
                [{"name": "X-Keep", "value": ""}],
                existing_encrypted=[{"name": "X-Keep", "value_encrypted": "garbage"}],
            )


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
