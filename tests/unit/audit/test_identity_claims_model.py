"""Unit tests for the shared `IdentityClaims` definition.

The durable / IdP identity claims used to be hand-copied onto both `Identity`
and `TokenMintAuditRecord`, and the two copies' descriptions had already drifted
apart. They now come from one place. These tests guard the three things that
made the duplication dangerous:

1. The field NAMES are frozen (the frontend, the CSV export, the Mongo indexes
   and the auth server's claim resolver all key on them).
2. The type, default and description of each claim are identical everywhere.
3. The serialized SHAPE is unchanged -- same keys, same order, explicit `null`
   for an absent claim. The refactor must not rewrite records already stored.

Validates: Issue #1642 (audit identity claims).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from registry.audit.models import (
    IDENTITY_CLAIM_FIELDS,
    Identity,
    IdentityClaims,
    TokenMintAuditRecord,
)

# The frozen vocabulary, in canonical order.
EXPECTED_CLAIM_FIELDS = (
    "subject",
    "canonical_id",
    "principal_name",
    "object_id",
    "tenant_id",
    "app_id",
)

CLAIM_CARRIERS = (Identity, TokenMintAuditRecord)


def _identity() -> Identity:
    return Identity(username="alice@example.com", auth_method="oauth2", credential_type="none")


def _mint_record() -> TokenMintAuditRecord:
    return TokenMintAuditRecord(
        timestamp=datetime.now(UTC),
        request_id="req-1",
        username_hash="user_1234abcd",
        auth_method="oauth2",
        internal_caller="registry",
        token_kind="user",
        token_path="self_signed",
        outcome="success",
    )


class TestFrozenVocabulary:
    def test_claim_field_names_are_frozen(self):
        assert IDENTITY_CLAIM_FIELDS == EXPECTED_CLAIM_FIELDS

    def test_display_name_is_deliberately_not_captured(self):
        """The `name` claim is RESERVED, not stored: `principal_name` already
        makes the actor contactable, so a person's full name would add a PII
        category for no investigative gain (GDPR Art. 25(2))."""
        assert "display_name" not in IdentityClaims.model_fields
        for model in CLAIM_CARRIERS:
            assert "display_name" not in model.model_fields

    def test_every_carrier_declares_every_claim(self):
        for model in CLAIM_CARRIERS:
            assert set(EXPECTED_CLAIM_FIELDS) <= set(model.model_fields), model.__name__


class TestClaimFieldsComeFromIdentityClaims:
    """One definition: a hand-edited copy on either record fails here."""

    def test_claim_fields_come_from_identity_claims(self):
        for name in IDENTITY_CLAIM_FIELDS:
            source = IdentityClaims.model_fields[name]
            for model in CLAIM_CARRIERS:
                copied = model.model_fields[name]
                assert copied.annotation == source.annotation, (model.__name__, name)
                assert copied.default == source.default, (model.__name__, name)
                assert copied.description == source.description, (model.__name__, name)

    def test_descriptions_no_longer_differ_between_the_two_records(self):
        """The concrete drift this refactor closed: `app_id` was documented as
        "Calling application id" on one record and "... from the token" on the
        other."""
        for name in IDENTITY_CLAIM_FIELDS:
            descriptions = {model.model_fields[name].description for model in CLAIM_CARRIERS}
            assert len(descriptions) == 1, (name, descriptions)

    def test_field_definitions_are_not_shared_instances(self):
        """Each model owns a copy, so pydantic mutating one model's FieldInfo
        cannot leak into another's."""
        for name in IDENTITY_CLAIM_FIELDS:
            instances = {id(model.model_fields[name]) for model in CLAIM_CARRIERS}
            instances.add(id(IdentityClaims.model_fields[name]))
            assert len(instances) == len(CLAIM_CARRIERS) + 1, name


class TestSerializedShapeIsUnchanged:
    """A base class would have forced the claim block to serialize FIRST,
    reordering the keys of every record already in the audit store."""

    def test_identity_claim_block_stays_last_after_credential_hint(self):
        order = list(Identity.model_fields)
        start = order.index("credential_hint") + 1

        assert tuple(order[start:]) == EXPECTED_CLAIM_FIELDS

    def test_mint_claim_block_stays_between_internal_caller_and_token_kind(self):
        order = list(TokenMintAuditRecord.model_fields)
        start = order.index("internal_caller") + 1

        assert tuple(order[start : start + len(EXPECTED_CLAIM_FIELDS)]) == EXPECTED_CLAIM_FIELDS
        assert order[start + len(EXPECTED_CLAIM_FIELDS)] == "token_kind"

    def test_absent_claims_serialize_as_explicit_null(self):
        """Consumers rely on the key existing: a missing key and a null value are
        not the same thing to the CSV export or to a Mongo query."""
        for record in (_identity(), _mint_record()):
            payload = record.model_dump()
            for name in EXPECTED_CLAIM_FIELDS:
                assert name in payload, (type(record).__name__, name)
                assert payload[name] is None, (type(record).__name__, name)

    def test_populated_claims_round_trip(self):
        claims = {
            "subject": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "canonical_id": "11111111-2222-3333-4444-555555555555@66666666-7777-8888-9999-000000000000",
            "principal_name": "alice@example.com",
            "object_id": "11111111-2222-3333-4444-555555555555",
            "tenant_id": "66666666-7777-8888-9999-000000000000",
            "app_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        }
        identity = Identity(
            username="alice@example.com",
            auth_method="oauth2",
            credential_type="bearer_token",
            **claims,
        )

        payload = identity.model_dump()
        for name, value in claims.items():
            assert payload[name] == value

    def test_non_string_claim_is_still_rejected(self):
        """Validation behaviour is unchanged by the refactor."""
        with pytest.raises(ValidationError):
            Identity(
                username="alice@example.com",
                auth_method="oauth2",
                credential_type="none",
                subject=["not", "a", "string"],
            )
