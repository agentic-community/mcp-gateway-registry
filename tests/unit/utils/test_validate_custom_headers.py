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
    with pytest.raises(ValueError, match="caller-overridable"):
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
    [
        "Host",
        "Content-Length",
        "Cookie",
        "X-Forwarded-For",
        "X-Authorization",
        # Gateway-internal identity / routing / signed-token headers: registering
        # any of these (esp. as overridable) would let a caller exfiltrate the
        # gateway's own internal token or the user's identity to the backend.
        "X-Internal-Token",
        "X-Internal-Token-Generic",
        "X-Internal-Token-Registry",
        "X-User",
        "X-Username",
        "X-Scopes",
        "X-Groups",
        "X-Auth-Method",
        "X-Client-Id",
        "X-Original-URL",
        "X-Generic-Has-Upstream-Auth",
        "X-Upstream-Url",
    ],
)
def test_reserved_header_rejected(reserved):
    # Every reserved gateway/hop-by-hop/internal name is rejected in ANY form --
    # EXCEPT Authorization (covered separately below), which is allowed as
    # overridable.
    with pytest.raises(ValueError, match="managed by the gateway"):
        validate_custom_headers([{"name": reserved, "value": "x"}])
    with pytest.raises(ValueError, match="managed by the gateway"):
        validate_custom_headers([{"name": reserved, "overridable": True}])


def test_count_cap_enforced():
    too_many = [{"name": f"X-H{i}", "value": "v"} for i in range(MAX_CUSTOM_HEADERS_PER_SERVER + 1)]
    with pytest.raises(ValueError, match="Too many custom headers"):
        validate_custom_headers(too_many)


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_custom_headers([{"name": "X-A", "value": "1"}, {"name": "x-a", "value": "2"}])


def test_empty_name_rejected():
    with pytest.raises(ValueError, match="non-empty name"):
        validate_custom_headers([{"name": "", "value": "v"}])


def test_non_object_entry_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        validate_custom_headers(["X-A: v"])


@pytest.mark.parametrize(
    "name",
    [
        "X Bad",
        "X:Bad",
        "X-Bad\rInjected",
        "X-Bad\nInjected",
        "X-Bad\x00Injected",
        "X-Bad\x1fInjected",
        "X-Bad\x7fInjected",
    ],
)
def test_header_name_must_be_rfc_token(name):
    with pytest.raises(ValueError, match="RFC token"):
        validate_custom_headers([{"name": name, "value": "safe"}])


def test_header_name_must_be_string_and_bounded():
    with pytest.raises(ValueError, match="name must be a string"):
        validate_custom_headers([{"name": 123, "value": "safe"}])
    with pytest.raises(ValueError, match="name exceeds 256"):
        validate_custom_headers([{"name": "X" * 257, "value": "safe"}])


def test_header_overridable_must_be_boolean():
    for invalid in ("true", 1, 0, [], {}):
        with pytest.raises(ValueError, match="overridable must be a boolean"):
            validate_custom_headers([{"name": "X-Caller-Only", "overridable": invalid}])


@pytest.mark.parametrize(
    "value",
    ["bad\rvalue", "bad\nvalue", "bad\x00value", "bad\tvalue", "bad\x7fvalue"],
)
def test_header_value_rejects_control_characters(value):
    with pytest.raises(ValueError, match="control characters"):
        validate_custom_headers([{"name": "X-Safe", "value": value}])


def test_header_value_must_be_string_and_bounded():
    with pytest.raises(ValueError, match="value must be a string"):
        validate_custom_headers([{"name": "X-Safe", "value": 123}])
    with pytest.raises(ValueError, match="value exceeds 4096"):
        validate_custom_headers([{"name": "X-Safe", "value": "v" * 4097}])


def test_valid_rfc_token_and_caller_only_slot_remain_allowed():
    headers = [
        {"name": "X-Key!#$%&'*+-.^_`|~09Az", "value": "safe visible value"},
        {"name": "X-Caller-Only", "overridable": True},
    ]
    assert validate_custom_headers(headers) == headers


class TestOverridablePolicy:
    """Per-header overridable flag: the three expressible shapes + rejections."""

    def test_fixed_header_with_value_accepted(self):
        # {name, value} (overridable defaults false) = FIXED operator credential.
        hdrs = [{"name": "X-Api-Key", "value": "sk-123"}]
        assert validate_custom_headers(hdrs) == hdrs

    def test_default_plus_override_accepted(self):
        hdrs = [{"name": "X-Api-Key", "value": "sk-default", "overridable": True}]
        assert validate_custom_headers(hdrs) == hdrs

    def test_caller_only_slot_accepted(self):
        # Value-less but overridable = caller-only passthrough slot.
        hdrs = [{"name": "X-Tenant", "overridable": True}]
        assert validate_custom_headers(hdrs) == hdrs

    def test_valueless_non_overridable_rejected(self):
        # Nothing to inject and the caller cannot supply it -> meaningless.
        with pytest.raises(ValueError, match="no value and is not overridable"):
            validate_custom_headers([{"name": "X-Tenant"}])
        with pytest.raises(ValueError, match="no value and is not overridable"):
            validate_custom_headers([{"name": "X-Tenant", "value": "", "overridable": False}])

    def test_authorization_allowed_only_when_overridable(self):
        # Caller-overridable Authorization (with or without a default) is allowed.
        assert validate_custom_headers([{"name": "Authorization", "overridable": True}]) is not None
        assert (
            validate_custom_headers(
                [{"name": "authorization", "value": "Bearer default", "overridable": True}]
            )
            is not None
        )

    def test_fixed_authorization_rejected(self):
        # A baked-in operator Authorization belongs in the egress vault, not here.
        with pytest.raises(ValueError, match="caller-overridable"):
            validate_custom_headers([{"name": "Authorization", "value": "Bearer secret"}])
        with pytest.raises(ValueError, match="caller-overridable"):
            validate_custom_headers(
                [{"name": "Authorization", "value": "Bearer secret", "overridable": False}]
            )


class TestEncryptBuildsNameSets:
    """encrypt_custom_headers_in_server_dict must encrypt only value-bearing
    entries, list every registered name, and separately list the overridable
    subset (the caller allowlist)."""

    def test_mixed_entries_produce_correct_sets(self):
        from registry.utils.credential_encryption import (
            decrypt_custom_headers,
            encrypt_custom_headers_in_server_dict,
        )

        d = {
            "custom_headers": [
                {"name": "X-Fixed", "value": "sk-fixed"},  # fixed default
                {"name": "X-Default", "value": "sk-def", "overridable": True},  # default+override
                {"name": "X-Caller", "overridable": True},  # caller-only, no value
            ]
        }
        encrypt_custom_headers_in_server_dict(d)

        # Every registered name is present; the value-less caller-only slot has no
        # encrypted entry but IS in the name + overridable sets.
        assert d["custom_header_names"] == ["X-Fixed", "X-Default", "X-Caller"]
        assert d["custom_header_overridable_names"] == ["X-Default", "X-Caller"]
        enc_names = [e["name"] for e in d["custom_headers_encrypted"]]
        assert enc_names == ["X-Fixed", "X-Default"]  # X-Caller has no value to store
        assert "custom_headers" not in d  # plaintext removed

        # The stored values round-trip.
        decrypted = {
            h["name"]: h["value"] for h in decrypt_custom_headers(d["custom_headers_encrypted"])
        }
        assert decrypted == {"X-Fixed": "sk-fixed", "X-Default": "sk-def"}

    def test_no_overridable_yields_empty_allowlist(self):
        from registry.utils.credential_encryption import encrypt_custom_headers_in_server_dict

        d = {"custom_headers": [{"name": "X-Api-Key", "value": "sk"}]}
        encrypt_custom_headers_in_server_dict(d)
        assert d["custom_header_overridable_names"] == []


class TestBuildCustomHeadersStorageFields:
    """The rotation helper produces a self-contained $set of all four header
    storage fields, including the CLEAR case, and validates like create."""

    def test_populates_all_four_fields(self):
        from registry.utils.credential_encryption import (
            build_custom_headers_storage_fields,
            decrypt_custom_headers,
        )

        out = build_custom_headers_storage_fields(
            [
                {"name": "X-Api-Key", "value": "sk-1"},
                {"name": "X-Tenant", "overridable": True},
            ]
        )
        assert out["custom_header_names"] == ["X-Api-Key", "X-Tenant"]
        assert out["custom_header_overridable_names"] == ["X-Tenant"]
        assert out["custom_headers_updated_at"]
        decrypted = {
            h["name"]: h["value"] for h in decrypt_custom_headers(out["custom_headers_encrypted"])
        }
        assert decrypted == {"X-Api-Key": "sk-1"}

    def test_empty_list_clears_all_fields(self):
        from registry.utils.credential_encryption import build_custom_headers_storage_fields

        out = build_custom_headers_storage_fields([])
        assert out["custom_headers_encrypted"] == []
        assert out["custom_header_names"] == []
        assert out["custom_header_overridable_names"] == []
        assert out["custom_headers_updated_at"]  # rotation is still stamped

    def test_none_clears_all_fields(self):
        from registry.utils.credential_encryption import build_custom_headers_storage_fields

        out = build_custom_headers_storage_fields(None)
        assert out["custom_header_names"] == []
        assert out["custom_headers_encrypted"] == []

    def test_policy_violation_raises(self):
        # Reserved / internal names are still rejected on rotation.
        from registry.utils.credential_encryption import build_custom_headers_storage_fields

        with pytest.raises(ValueError, match="managed by the gateway"):
            build_custom_headers_storage_fields([{"name": "X-Internal-Token", "value": "x"}])
        with pytest.raises(ValueError, match="caller-overridable"):
            build_custom_headers_storage_fields([{"name": "Authorization", "value": "Bearer x"}])

    def test_blank_value_preserves_existing_ciphertext(self):
        # Write-only value UX: on edit a fixed header arrives with a blank value;
        # the prior ciphertext is decrypted and carried forward unchanged.
        from registry.utils.credential_encryption import (
            build_custom_headers_storage_fields,
            decrypt_custom_headers,
            encrypt_credential,
        )

        existing = [{"name": "X-Api-Key", "value_encrypted": encrypt_credential("sk-original")}]
        out = build_custom_headers_storage_fields(
            [{"name": "X-Api-Key", "value": "", "overridable": False}],
            existing_encrypted=existing,
        )
        decrypted = {
            h["name"]: h["value"] for h in decrypt_custom_headers(out["custom_headers_encrypted"])
        }
        assert decrypted == {"X-Api-Key": "sk-original"}

    def test_blank_value_new_value_overwrites(self):
        from registry.utils.credential_encryption import (
            build_custom_headers_storage_fields,
            decrypt_custom_headers,
            encrypt_credential,
        )

        existing = [{"name": "X-Api-Key", "value_encrypted": encrypt_credential("sk-original")}]
        out = build_custom_headers_storage_fields(
            [{"name": "X-Api-Key", "value": "sk-rotated"}],
            existing_encrypted=existing,
        )
        decrypted = {
            h["name"]: h["value"] for h in decrypt_custom_headers(out["custom_headers_encrypted"])
        }
        assert decrypted == {"X-Api-Key": "sk-rotated"}

    def test_blank_value_no_prior_non_overridable_rejected(self):
        # A blank fixed header with no stored value to preserve is meaningless.
        from registry.utils.credential_encryption import build_custom_headers_storage_fields

        with pytest.raises(ValueError, match="no value and is not overridable"):
            build_custom_headers_storage_fields([{"name": "X-New", "value": ""}])

    def test_blank_value_no_prior_overridable_is_caller_slot(self):
        # A blank overridable row with no prior is a caller-only slot (allowed).
        from registry.utils.credential_encryption import build_custom_headers_storage_fields

        out = build_custom_headers_storage_fields(
            [{"name": "X-Tenant", "value": "", "overridable": True}]
        )
        assert out["custom_header_names"] == ["X-Tenant"]
        assert out["custom_header_overridable_names"] == ["X-Tenant"]
        assert out["custom_headers_encrypted"] == []  # no value stored


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
            "custom_header_overridable_names",
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
        assert upd["custom_header_overridable_names"] == []
        assert upd["custom_headers_updated_at"] is None

    def test_same_host_different_path_clears_headers(self):
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd = {"proxy_target_url": "https://h.example/v2"}
        clear_upstream_headers_on_repoint(
            upd, existing_target="https://h.example/v1", new_target="https://h.example/v2"
        )
        assert upd["custom_headers_encrypted"] is None
        assert upd["custom_header_names"] == []

    def test_same_path_different_query_clears_headers(self):
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd = {"proxy_target_url": "https://h.example/v1?tenant=b"}
        clear_upstream_headers_on_repoint(
            upd,
            existing_target="https://h.example/v1?tenant=a",
            new_target="https://h.example/v1?tenant=b",
        )
        assert upd["custom_headers_encrypted"] is None

    def test_normalization_equivalence_preserves_headers(self):
        from registry.schemas.proxy_mixin import clear_upstream_headers_on_repoint

        upd: dict = {}
        clear_upstream_headers_on_repoint(
            upd,
            existing_target="HTTPS://H.Example:443/v1?tenant=a",
            new_target="https://h.example/v1?tenant=a",
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
