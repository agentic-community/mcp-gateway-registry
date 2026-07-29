"""Focused tests for existing MCP custom-header storage hardening."""

from unittest.mock import patch

import pytest

from registry.utils.credential_encryption import (
    decrypt_custom_headers,
    encrypt_custom_headers_in_server_dict,
    validate_custom_headers,
)

pytestmark = pytest.mark.unit


def test_valid_headers_pass():
    headers = [
        {"name": "X-Tenant", "value": "acme"},
        {"name": "X-Key!#$%&'*+-.^_`|~09Az", "value": "safe value"},
    ]
    assert validate_custom_headers(headers) == headers


@pytest.mark.parametrize(
    "name",
    [
        "X Bad",
        "X:Bad",
        "X-Bad\rInjected",
        "X-Bad\nInjected",
        "X-Bad\x00Injected",
        "X-Bad\x7fInjected",
    ],
)
def test_name_must_be_rfc_token(name):
    with pytest.raises(ValueError, match="RFC token"):
        validate_custom_headers([{"name": name, "value": "safe"}])


def test_name_must_be_string_and_bounded():
    with pytest.raises(ValueError, match="name must be a string"):
        validate_custom_headers([{"name": 123, "value": "safe"}])
    with pytest.raises(ValueError, match="name exceeds 256"):
        validate_custom_headers([{"name": "X" * 257, "value": "safe"}])


@pytest.mark.parametrize(
    "value",
    ["bad\rvalue", "bad\nvalue", "bad\x00value", "bad\tvalue", "bad\x7fvalue"],
)
def test_value_rejects_all_controls(value):
    with pytest.raises(ValueError, match="control characters"):
        validate_custom_headers([{"name": "X-Safe", "value": value}])


def test_value_must_be_string_and_bounded():
    with pytest.raises(ValueError, match="value must be a string"):
        validate_custom_headers([{"name": "X-Safe", "value": 123}])
    with pytest.raises(ValueError, match="value exceeds 4096"):
        validate_custom_headers([{"name": "X-Safe", "value": "v" * 4097}])


def test_reserved_and_duplicate_names_rejected_case_insensitively():
    with pytest.raises(ValueError, match="managed by the gateway"):
        validate_custom_headers([{"name": "Authorization", "value": "secret"}])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_custom_headers(
            [
                {"name": "X-Tenant", "value": "a"},
                {"name": "x-tenant", "value": "b"},
            ]
        )


def test_encryption_defensively_validates_before_mutation():
    record = {"custom_headers": [{"name": "X-Bad", "value": "bad\tvalue"}]}
    with pytest.raises(ValueError, match="control characters"):
        encrypt_custom_headers_in_server_dict(record)
    assert "custom_headers" in record
    assert "custom_headers_encrypted" not in record


def test_non_strict_decrypt_skips_malformed_duplicate_and_unsafe_entries():
    entries = [
        "not-an-object",
        {"name": "X-Bad\rInjected", "value_encrypted": "bad-name"},
        {"name": "Authorization", "value_encrypted": "reserved"},
        {"name": "X-Good", "value_encrypted": "good"},
        {"name": "x-good", "value_encrypted": "duplicate"},
        {"name": "X-Control", "value_encrypted": "control"},
        {"name": "X-Missing"},
    ]

    def decrypt(ciphertext):
        return {
            "reserved": "must-not-emit",
            "good": "safe",
            "duplicate": "other",
            "control": "bad\nvalue",
        }.get(ciphertext)

    with patch(
        "registry.utils.credential_encryption.decrypt_credential",
        side_effect=decrypt,
    ):
        assert decrypt_custom_headers(entries) == [{"name": "X-Good", "value": "safe"}]


def test_non_strict_decrypt_rejects_non_list_without_raising():
    assert decrypt_custom_headers("not-a-list") == []
