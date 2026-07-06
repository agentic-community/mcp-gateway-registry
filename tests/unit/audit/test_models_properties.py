"""
Property-based tests for audit model masking and JSONL serialization.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 3.1
"""

import json
from datetime import UTC

from hypothesis import given, settings
from hypothesis import strategies as st

from registry.audit.models import (
    SENSITIVE_QUERY_PARAMS,
    Identity,
    RegistryApiAccessRecord,
    Request,
    Response,
    mask_credential,
)


class TestCredentialMasking:
    """Property 3: Credential masking consistency."""

    @given(st.text(min_size=0, max_size=6))
    @settings(max_examples=50)
    def test_short_credentials_masked_completely(self, credential: str):
        """Short credentials return the fixed marker."""
        assert mask_credential(credential) == "***"

    @given(st.text(min_size=7, max_size=100))
    @settings(max_examples=50)
    def test_long_credentials_emit_no_value(self, credential: str):
        """Long credentials return the fixed marker with NO substring of the value.

        Emitting even a suffix is a leak: for opaque tokens/cookies the trailing
        bytes are real key-space. The masker must reveal nothing.
        """
        result = mask_credential(credential)
        assert result == "***"
        # No non-trivial substring of the credential appears in the output.
        assert credential[-6:] not in result or len(credential[-6:].strip()) == 0


class TestSensitiveQueryParamMasking:
    """Property 4: Sensitive query parameter masking."""

    @given(
        st.dictionaries(
            keys=st.sampled_from(list(SENSITIVE_QUERY_PARAMS)),
            values=st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=3,
        )
    )
    @settings(max_examples=50)
    def test_sensitive_params_are_masked(self, sensitive_params: dict):
        """Query parameters with sensitive keys have their values masked."""
        request = Request(
            method="GET",
            path="/api/test",
            query_params=sensitive_params,
            client_ip="127.0.0.1",
        )
        for key, original_value in sensitive_params.items():
            assert request.query_params[key] == mask_credential(str(original_value))

    def test_auth_credential_is_masked(self):
        """The skill-parse ``auth_credential`` param is masked in the audit log."""
        secret = "super-secret-token-value"
        request = Request(
            method="POST",
            path="/api/skills/parse-skill-md",
            query_params={"auth_credential": secret, "url": "https://example.com"},
            client_ip="127.0.0.1",
        )
        assert request.query_params["auth_credential"] == mask_credential(secret)
        # Non-sensitive params are untouched.
        assert request.query_params["url"] == "https://example.com"

    @given(
        st.sampled_from(
            [
                "auth_credential",
                "AUTH_CREDENTIAL",
                "authCredential",
                "auth_token",
                "client_secret",
                "x_api_key",
                "X-Api-Key",
                "user_password",
                "session_token",
                "bearer_credential",
            ]
        ),
        st.text(min_size=7, max_size=50),
    )
    @settings(max_examples=50)
    def test_sensitive_variant_names_are_masked(self, key: str, value: str):
        """Variant / future sensitive param names are masked via substring match.

        Fail-closed: a parameter name that merely *contains* a sensitive token
        (case-insensitive) is masked even without an exact-match entry.
        """
        request = Request(
            method="GET",
            path="/api/test",
            query_params={key: value},
            client_ip="127.0.0.1",
        )
        assert request.query_params[key] == mask_credential(value)

    @given(
        st.sampled_from(["url", "page", "limit", "offset", "tag", "q", "include_disabled"]),
        st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_nonsensitive_params_are_not_masked(self, key: str, value: str):
        """Ordinary, non-credential query params are logged verbatim."""
        request = Request(
            method="GET",
            path="/api/test",
            query_params={key: value},
            client_ip="127.0.0.1",
        )
        assert request.query_params[key] == value


class TestJSONLFormatValidity:
    """Property 5: JSONL format validity."""

    @given(
        st.builds(
            RegistryApiAccessRecord,
            timestamp=st.datetimes(timezones=st.just(UTC)),
            request_id=st.uuids().map(str),
            identity=st.builds(
                Identity,
                username=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
                auth_method=st.sampled_from(["oauth2", "anonymous"]),
                credential_type=st.sampled_from(["bearer_token", "none"]),
            ),
            request=st.builds(
                Request,
                method=st.sampled_from(["GET", "POST"]),
                path=st.just("/api/test"),
                client_ip=st.just("127.0.0.1"),
            ),
            response=st.builds(
                Response,
                status_code=st.integers(min_value=200, max_value=500),
                duration_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            ),
        )
    )
    @settings(max_examples=50)
    def test_audit_record_round_trip(self, record: RegistryApiAccessRecord):
        """Serializing and deserializing produces an equivalent object."""
        json_str = record.model_dump_json()
        assert "\n" not in json_str  # Single line
        parsed = json.loads(json_str)
        reconstructed = RegistryApiAccessRecord.model_validate(parsed)
        assert reconstructed.request_id == record.request_id
