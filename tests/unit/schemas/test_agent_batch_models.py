"""Tests for issue #956 batch and PATCH schemas in registry.schemas.agent_models.

Covers:
- AgentCardPatch: optional fields, alias support, registrant-only rejection, extra="forbid".
- The AgentBatchItem discriminated union (register/patch/replace/delete variants).
- AgentBatchRequest validation (min items, idempotency_key bounds, extra="forbid").
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from registry.schemas.agent_models import (
    REGISTRANT_ONLY_FIELDS,
    AgentBatchItem,
    AgentBatchRequest,
    AgentCardPatch,
    BatchItemOp,
)

REGISTER_CARD = {
    "name": "batch-agent",
    "url": "https://example.com/agent",
    "supported_protocol": "a2a",
}


@pytest.mark.unit
class TestAgentCardPatch:
    """Tests for the RFC 7396 merge-patch body model."""

    def test_empty_patch_is_valid(self):
        """A patch with no fields set is valid; exclude_unset yields nothing."""
        patch = AgentCardPatch()
        assert patch.model_dump(exclude_unset=True) == {}

    def test_single_field_only_that_field_is_set(self):
        """Only explicitly supplied fields appear in exclude_unset output."""
        patch = AgentCardPatch(description="new description")
        dumped = patch.model_dump(exclude_unset=True, by_alias=False)
        assert dumped == {"description": "new description"}

    def test_camel_case_alias_accepted(self):
        """camelCase aliases (protocolVersion, trustLevel) populate snake fields."""
        patch = AgentCardPatch(protocolVersion="2.0", trustLevel="verified")
        assert patch.protocol_version == "2.0"
        assert patch.trust_level == "verified"

    def test_snake_case_accepted_via_populate_by_name(self):
        """Snake-case names are accepted alongside aliases."""
        patch = AgentCardPatch(protocol_version="2.0", allowed_groups=["a"])
        assert patch.protocol_version == "2.0"
        assert patch.allowed_groups == ["a"]

    def test_extra_field_forbidden(self):
        """Unknown fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            AgentCardPatch(not_a_real_field="x")

    def test_empty_name_rejected(self):
        """name has min_length=1 when supplied."""
        with pytest.raises(ValidationError):
            AgentCardPatch(name="")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("registered_by", "someoneelse"),
            ("num_stars", 5),
            ("path", "/agents/hijacked"),
            ("health_status", "healthy"),
        ],
    )
    def test_registrant_only_field_rejected(self, field, value):
        """Supplying a registrant-only field raises a validation error.

        None of the registrant-only fields are declared on AgentCardPatch, so
        extra='forbid' rejects them before the dedicated model validator runs.
        The validator remains as defense-in-depth should any be added as a real
        field later.
        """
        assert field in REGISTRANT_ONLY_FIELDS
        with pytest.raises(ValidationError):
            AgentCardPatch(**{field: value})

    def test_tags_accept_string_or_list(self):
        """tags may be a list or comma-separated string per the union type."""
        assert AgentCardPatch(tags=["a", "b"]).tags == ["a", "b"]
        assert AgentCardPatch(tags="a,b").tags == "a,b"


@pytest.mark.unit
class TestAgentBatchItemUnion:
    """Tests for the op-discriminated AgentBatchItem union."""

    def setup_method(self):
        self.adapter = TypeAdapter(AgentBatchItem)

    def test_register_item_parsed(self):
        item = self.adapter.validate_python({"op": "register", "card": REGISTER_CARD})
        assert item.op == BatchItemOp.register
        assert item.card.name == "batch-agent"

    def test_patch_item_parsed(self):
        item = self.adapter.validate_python(
            {"op": "patch", "path": "/agents/x", "card": {"description": "d"}}
        )
        assert item.op == BatchItemOp.patch
        assert item.path == "/agents/x"
        assert item.card.description == "d"

    def test_replace_item_parsed(self):
        item = self.adapter.validate_python(
            {"op": "replace", "path": "/agents/x", "card": REGISTER_CARD}
        )
        assert item.op == BatchItemOp.replace
        assert item.card.name == "batch-agent"

    def test_delete_item_parsed(self):
        item = self.adapter.validate_python({"op": "delete", "path": "/agents/x"})
        assert item.op == BatchItemOp.delete
        assert item.path == "/agents/x"

    def test_unknown_op_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"op": "frobnicate", "path": "/agents/x"})

    def test_patch_item_requires_path(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"op": "patch", "card": {"description": "d"}})

    def test_patch_item_rejects_registrant_only_in_card(self):
        """A patch item's card still enforces registrant-only rejection."""
        with pytest.raises(ValidationError):
            self.adapter.validate_python(
                {"op": "patch", "path": "/agents/x", "card": {"registered_by": "z"}}
            )


@pytest.mark.unit
class TestAgentBatchRequest:
    """Tests for the batch submission request body."""

    def test_minimal_request_valid(self):
        req = AgentBatchRequest(items=[{"op": "delete", "path": "/agents/x"}])
        assert len(req.items) == 1
        assert req.idempotency_key is None

    def test_empty_items_rejected(self):
        """items has min_length=1."""
        with pytest.raises(ValidationError):
            AgentBatchRequest(items=[])

    def test_idempotency_key_length_capped(self):
        """idempotency_key has max_length=200."""
        with pytest.raises(ValidationError):
            AgentBatchRequest(
                idempotency_key="x" * 201,
                items=[{"op": "delete", "path": "/agents/x"}],
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            AgentBatchRequest(
                items=[{"op": "delete", "path": "/agents/x"}],
                unexpected="nope",
            )

    def test_mixed_ops_request(self):
        req = AgentBatchRequest(
            idempotency_key="key-1",
            items=[
                {"op": "register", "card": REGISTER_CARD},
                {"op": "patch", "path": "/agents/x", "card": {"description": "d"}},
                {"op": "delete", "path": "/agents/y"},
            ],
        )
        assert [i.op for i in req.items] == [
            BatchItemOp.register,
            BatchItemOp.patch,
            BatchItemOp.delete,
        ]


@pytest.mark.unit
def test_registrant_only_fields_constant_is_frozenset():
    """The shared constant is an immutable frozenset of expected anchors."""
    assert isinstance(REGISTRANT_ONLY_FIELDS, frozenset)
    assert {"registered_by", "path", "updated_at"} <= REGISTRANT_ONLY_FIELDS
