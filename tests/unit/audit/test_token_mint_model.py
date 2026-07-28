"""Unit tests for the TokenMintAuditRecord model (PR #1304 follow-up, #1308).

Covers defaults, required-field enforcement, serialization, and the
resource-bound / failure variants. These guard the audit contract: no raw
token material, a hashed username, and a stable ``token_mint`` discriminator.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from registry.audit.models import TokenMintAuditRecord


def _minimal(**overrides):
    """Build a record with only the required fields set, allowing overrides."""
    base = {
        "request_id": "req-123",
        "username_hash": "user_abcd1234",
        "auth_method": "oauth2",
        "internal_caller": "registry",
        "token_kind": "user",
        "token_path": "self_signed",
        "outcome": "success",
    }
    base.update(overrides)
    return TokenMintAuditRecord(**base)


class TestDefaults:
    def test_log_type_is_token_mint(self):
        assert _minimal().log_type == "token_mint"

    def test_version_default(self):
        assert _minimal().version == "1.0"

    def test_timestamp_is_timezone_aware_utc(self):
        ts = _minimal().timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None
        assert ts.utcoffset() == UTC.utcoffset(None)

    def test_optional_fields_default_none(self):
        r = _minimal()
        assert r.correlation_id is None
        assert r.provider is None
        assert r.resource_type is None
        assert r.resource_id is None
        assert r.expires_in_seconds is None
        assert r.failure_reason is None

    def test_requested_scopes_defaults_empty_list(self):
        assert _minimal().requested_scopes == []

    def test_username_defaults_anonymous(self):
        # The raw human-readable field is optional (back-compat with callers that
        # only set username_hash); it defaults to "anonymous", never blank.
        assert _minimal().username == "anonymous"

    def test_username_stores_raw_email(self):
        r = _minimal(username="alice@example.com")
        assert r.username == "alice@example.com"


class TestRequiredFields:
    @pytest.mark.parametrize(
        "missing",
        [
            "request_id",
            "username_hash",
            "auth_method",
            "internal_caller",
            "token_kind",
            "token_path",
            "outcome",
        ],
    )
    def test_missing_required_field_raises(self, missing):
        kwargs = {
            "request_id": "req-1",
            "username_hash": "user_abcd1234",
            "auth_method": "oauth2",
            "internal_caller": "registry",
            "token_kind": "user",
            "token_path": "self_signed",
            "outcome": "success",
        }
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            TokenMintAuditRecord(**kwargs)


class TestVariants:
    def test_resource_bound_record_carries_resource(self):
        r = _minimal(
            token_kind="resource",
            resource_type="server",
            resource_id="fininfo",
            token_path="self_signed",
        )
        assert r.token_kind == "resource"
        assert r.resource_type == "server"
        assert r.resource_id == "fininfo"

    def test_failure_record_carries_reason(self):
        r = _minimal(outcome="failure", failure_reason="rate_limited", token_path="unknown")
        assert r.outcome == "failure"
        assert r.failure_reason == "rate_limited"


class TestCorrelationIdCap:
    """The correlation_id field must be length-capped at the model boundary.

    Intent: even if a FUTURE producer forgets to run ``sanitize_correlation_id``,
    the durable audit record cannot store an unbounded, client-derived value that
    would bloat records or feed a log/audit-export injection. The cap mirrors the
    sanitizer's 200-char limit, so the model is a belt-and-suspenders backstop.
    """

    def test_correlation_id_at_cap_is_accepted(self):
        value = "a" * 200
        assert _minimal(correlation_id=value).correlation_id == value

    def test_correlation_id_over_cap_is_rejected(self):
        with pytest.raises(ValidationError):
            _minimal(correlation_id="a" * 201)


class TestSerialization:
    def test_round_trips_and_omits_raw_token(self):
        r = _minimal(correlation_id="corr-9", requested_scopes=["a", "b"])
        payload = r.model_dump_json()
        # Privacy contract: the hashed username is present, raw token never is.
        assert "user_abcd1234" in payload
        for forbidden in ("access_token", "refresh_token", "secret"):
            assert forbidden not in payload
        # token_mint discriminator must always be present for the audit store.
        assert '"log_type":"token_mint"' in payload
