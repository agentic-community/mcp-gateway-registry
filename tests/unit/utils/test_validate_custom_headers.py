"""Focused tests for existing MCP custom-header storage hardening."""

from unittest.mock import patch

import pytest

from registry.constants import MAX_CUSTOM_HEADERS_PER_SERVER
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


def test_none_passes_through():
    assert validate_custom_headers(None) is None


def test_valid_headers_accepted():
    hdrs = [{"name": "X-Api-Key", "value": "sk-123"}, {"name": "X-Tenant", "value": "acme"}]
    assert validate_custom_headers(hdrs) == hdrs


@pytest.mark.parametrize(
    "reserved",
    ["Authorization", "authorization", "Host", "Content-Length", "Cookie", "X-Forwarded-For"],
)
def test_reserved_header_rejected(reserved):
    with pytest.raises(ValueError, match="managed by the gateway"):
        validate_custom_headers([{"name": reserved, "value": "x"}])


def test_count_cap_enforced():
    too_many = [{"name": f"X-H{i}", "value": "v"} for i in range(MAX_CUSTOM_HEADERS_PER_SERVER + 1)]
    with pytest.raises(ValueError, match="Too many custom headers"):
        validate_custom_headers(too_many)


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_custom_headers([{"name": "X-A", "value": "1"}, {"name": "x-a", "value": "2"}])


def test_empty_name_or_value_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        validate_custom_headers([{"name": "X-A", "value": ""}])
    with pytest.raises(ValueError, match="non-empty"):
        validate_custom_headers([{"name": "", "value": "v"}])


def test_non_object_entry_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        validate_custom_headers(["X-A: v"])


class TestFederationStripCoversHeaders:
    """The upstream-header fields (encrypted + plaintext + bookkeeping) must all
    be stripped at the federation boundary so a peer can neither plant nor
    receive upstream credentials, in either direction."""

    def test_all_header_fields_in_proxy_field_names(self):
        from registry.schemas.proxy_mixin import PROXY_FIELD_NAMES

        for field in (
            "custom_headers",  # plaintext (defense-in-depth)
            "custom_headers_encrypted",
            "custom_header_names",
            "custom_headers_updated_at",
        ):
            assert field in PROXY_FIELD_NAMES, f"{field} not federation-stripped"

    def test_strip_removes_header_fields(self):
        from registry.schemas.proxy_mixin import strip_proxy_fields

        doc = {
            "name": "x",
            "custom_headers": [{"name": "X-A", "value": "secret"}],
            "custom_headers_encrypted": [{"name": "X-A", "value_encrypted": "enc"}],
            "custom_header_names": ["X-A"],
        }
        stripped = strip_proxy_fields(doc)
        assert "name" in stripped
        assert "custom_headers" not in stripped
        assert "custom_headers_encrypted" not in stripped
        assert "custom_header_names" not in stripped


class TestClearUpstreamHeadersOnRepoint:
    """Credential-misdirection guard: repointing an entity's proxy target to a
    different host must clear the create-time upstream headers, so the old host's
    secret is never injected at the new host."""

    def _updates(self):
        return {"proxy_target_url": "https://new.example/"}

    def test_host_change_clears_headers(self):
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd = self._updates()
        clear_upstream_headers_on_repoint(
            upd, existing_target="https://old.example/", new_target="https://new.example/"
        )
        assert upd["custom_headers_encrypted"] is None
        assert upd["custom_header_names"] == []
        assert upd["custom_headers_updated_at"] is None

    def test_same_host_different_path_preserves(self):
        # Same scheme+host+port -> same backend; a path/scheme-identical edit must
        # NOT clear the headers.
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd = {"proxy_target_url": "https://h.example/v2"}
        clear_upstream_headers_on_repoint(
            upd, existing_target="https://h.example/v1", new_target="https://h.example/v2"
        )
        assert "custom_headers_encrypted" not in upd

    def test_port_change_is_a_repoint(self):
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd = {}
        clear_upstream_headers_on_repoint(
            upd, existing_target="https://h.example:443/", new_target="https://h.example:8443/"
        )
        assert upd.get("custom_headers_encrypted") is None


class TestEffectiveProxyTargetRoutabilityAgnostic:
    """effective_proxy_target must derive the target WITHOUT the is_enabled /
    is_proxied / disabled gates, so the repoint guard fires even for a disabled
    or not-yet-enabled server (the round-3 Finding-1 bug: reusing the gated
    resolve_proxy_target let a repoint-before-enable keep the old host's secret).
    """

    def test_disabled_server_still_yields_target(self):
        from registry.schemas.proxy_mixin import effective_proxy_target, resolve_proxy_target

        doc = {"is_enabled": False, "is_proxied": True, "proxy_pass_url": "https://a.example/mcp"}
        # resolve_proxy_target is routability-gated -> None for a disabled server.
        assert resolve_proxy_target("mcp_server", doc) is None
        # effective_proxy_target ignores the gate -> the real backend.
        assert effective_proxy_target("mcp_server", doc) == "https://a.example/mcp"

    def test_disabled_server_repoint_clears_headers(self):
        # The exact Finding-1 scenario: repoint a DISABLED proxied server; the
        # guard (fed by effective_proxy_target) must still clear the headers.
        from registry.schemas.proxy_mixin import (
            clear_upstream_headers_on_repoint,
            effective_proxy_target,
        )

        old = {"is_enabled": False, "is_proxied": True, "proxy_pass_url": "https://legit-a/mcp"}
        new = {"is_enabled": False, "is_proxied": True, "proxy_pass_url": "https://attacker-b/mcp"}
        upd: dict = {}
        clear_upstream_headers_on_repoint(
            upd,
            existing_target=effective_proxy_target("mcp_server", old),
            new_target=effective_proxy_target("mcp_server", new),
        )
        assert upd["custom_headers_encrypted"] is None
        assert upd["custom_header_names"] == []

    def test_explicit_target_wins_over_fallback(self):
        from registry.schemas.proxy_mixin import effective_proxy_target

        doc = {"proxy_target_url": "https://explicit/", "proxy_pass_url": "https://fallback/"}
        assert effective_proxy_target("mcp_server", doc) == "https://explicit/"

    def test_local_mcp_server_has_no_target(self):
        from registry.schemas.proxy_mixin import effective_proxy_target

        doc = {"deployment": "local", "proxy_pass_url": "https://x/"}
        assert effective_proxy_target("mcp_server", doc) is None
