"""Unit tests for registry.common.log_redaction.

These tests would FAIL against code that logs raw headers / mappings and PASS
against the shared redaction helpers.
"""

import pytest

from registry.common.log_redaction import (
    REDACTED,
    redact_headers,
    redact_mapping,
)


class _FakeHeaders:
    """Minimal stand-in for Starlette Headers exposing ``items()``."""

    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def items(self):
        return self._data.items()


@pytest.mark.unit
class TestRedactHeaders:
    """redact_headers masks credential-bearing headers."""

    def test_masks_authorization_and_cookie(self):
        headers = {
            "authorization": "Bearer secrettoken.aaa.bbb",
            "cookie": "session=abc123deadbeef",
            "content-type": "application/json",
        }
        result = redact_headers(headers)
        assert result["authorization"] == REDACTED
        assert result["cookie"] == REDACTED
        # Non-sensitive headers are preserved for diagnostics.
        assert result["content-type"] == "application/json"

    def test_masks_case_insensitively(self):
        headers = {"Authorization": "Bearer x", "X-Api-Key": "key-value-123"}
        result = redact_headers(headers)
        assert result["Authorization"] == REDACTED
        assert result["X-Api-Key"] == REDACTED

    def test_masks_federation_and_session_token_headers(self):
        headers = {
            "X-Federation-Token": "fed-secret-abc",
            "X-Session-Token": "sess-secret-xyz",
            "x-internal-token-registry": "int-secret",
        }
        result = redact_headers(headers)
        assert result["X-Federation-Token"] == REDACTED
        assert result["X-Session-Token"] == REDACTED
        assert result["x-internal-token-registry"] == REDACTED

    def test_no_token_substring_leaks(self):
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        headers = {"authorization": f"Bearer {token}"}
        result = redact_headers(headers)
        assert token not in str(result)

    def test_accepts_headers_like_object(self):
        headers = _FakeHeaders({"authorization": "Bearer y", "accept": "*/*"})
        result = redact_headers(headers)
        assert result["authorization"] == REDACTED
        assert result["accept"] == "*/*"

    def test_masks_skill_parse_credential_header(self):
        """The X-Auth-Credential header used by parse-skill-md is masked."""
        result = redact_headers({"X-Auth-Credential": "super-secret-token"})
        assert result["X-Auth-Credential"] == REDACTED
        assert "super-secret-token" not in str(result)

    def test_masks_variant_credential_headers_by_substring(self):
        """Variant credential header names not on the exact list still redact.

        Substring matching is the fail-closed layer: a header whose name is not
        enumerated but contains a credential marker (token/secret/auth/...) is
        redacted by default rather than leaking.
        """
        headers = {
            "X-Access-Token": "t1",
            "X-Client-Secret": "s1",
            "X-User-Password": "pw1",
            "X-Custom-Auth": "a1",
            "X-Some-Credential": "c1",
            # Exotic credential names with no auth/token/secret substring must
            # still redact (fail-closed): a JWT, bearer, session id, or key.
            "X-Jwt": "eyJhbGci.payload.sig",
            "X-Bearer": "opaque-value",
            "X-Session": "sess-abc",
            "X-Signing-Key": "kkkk",
        }
        result = redact_headers(headers)
        for name in headers:
            assert result[name] == REDACTED, name

    def test_leaves_nonsensitive_headers_intact(self):
        headers = {
            "User-Agent": "curl/8",
            "Accept": "application/json",
            "X-Forwarded-For": "10.0.0.1",
            "Content-Length": "42",
        }
        assert redact_headers(headers) == headers


@pytest.mark.unit
class TestRedactMapping:
    """redact_mapping recursively masks sensitive keys by value."""

    def test_masks_federation_token(self):
        data = {"peer_id": "p1", "enabled": True, "federation_token": "abc-secret-xyz"}
        result = redact_mapping(data)
        assert result["peer_id"] == "p1"
        assert result["enabled"] is True
        assert result["federation_token"] == REDACTED
        assert "abc-secret-xyz" not in str(result)

    def test_masks_common_secret_keys(self):
        data = {
            "password": "hunter2",
            "access_token": "tok",
            "client_secret": "shhh",
            "api_key": "k",
            "session": "sid",
            "username": "alice",
        }
        result = redact_mapping(data)
        assert result["password"] == REDACTED
        assert result["access_token"] == REDACTED
        assert result["client_secret"] == REDACTED
        assert result["api_key"] == REDACTED
        assert result["session"] == REDACTED
        # Non-sensitive identifier preserved.
        assert result["username"] == "alice"

    def test_masks_nested_dicts_and_lists(self):
        data = {
            "outer": {"inner_token": "deep-secret", "safe": 1},
            "items": [{"authorization": "Bearer z"}, {"name": "ok"}],
        }
        result = redact_mapping(data)
        assert result["outer"]["inner_token"] == REDACTED
        assert result["outer"]["safe"] == 1
        assert result["items"][0]["authorization"] == REDACTED
        assert result["items"][1]["name"] == "ok"
        assert "deep-secret" not in str(result)
        assert "Bearer z" not in str(result)

    def test_does_not_mutate_input(self):
        data = {"federation_token": "keepme"}
        redact_mapping(data)
        # Original is untouched.
        assert data["federation_token"] == "keepme"

    def test_non_mapping_returned_unchanged(self):
        assert redact_mapping("plain") == "plain"
        assert redact_mapping(42) == 42
        assert redact_mapping(None) is None
