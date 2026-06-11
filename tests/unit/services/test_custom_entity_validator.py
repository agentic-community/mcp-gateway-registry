"""Unit tests for registry/services/custom_entity_validator.py.

Covers the descriptor-driven attribute validator: unknown-key rejection,
required-field enforcement, per-datatype checks (string/text/number/bool/
enum/date/array<string>), length/array bounds, and the multi-error
collection contract (ALL problems raised in one round-trip).
"""

import logging

import pytest

from registry.schemas.custom_entity_models import (
    MAX_ARRAY_ITEMS,
    MAX_STRING_LEN,
    MAX_TEXT_LEN,
    CustomFieldDescriptor,
    CustomFieldType,
    CustomTypeDescriptor,
)
from registry.services.custom_entity_errors import CustomEntityValidationError
from registry.services.custom_entity_validator import validate_attributes

logger = logging.getLogger(__name__)


def _descriptor(fields: list[CustomFieldDescriptor]) -> CustomTypeDescriptor:
    """Build a minimal descriptor wrapping the given fields."""
    return CustomTypeDescriptor(name="thing", fields=fields)


class TestValidateAttributesHappyPath:
    """Valid attributes pass and only known fields are returned."""

    def test_all_datatypes_valid(self):
        fields = [
            CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING),
            CustomFieldDescriptor(name="body", datatype=CustomFieldType.TEXT),
            CustomFieldDescriptor(name="count", datatype=CustomFieldType.NUMBER),
            CustomFieldDescriptor(name="active", datatype=CustomFieldType.BOOL),
            CustomFieldDescriptor(
                name="level",
                datatype=CustomFieldType.ENUM,
                enum_values=["low", "high"],
            ),
            CustomFieldDescriptor(name="due", datatype=CustomFieldType.DATE),
            CustomFieldDescriptor(name="labels", datatype=CustomFieldType.ARRAY_STRING),
        ]
        attrs = {
            "title": "hi",
            "body": "long text",
            "count": 3,
            "active": True,
            "level": "high",
            "due": "2026-01-15",
            "labels": ["a", "b"],
        }
        cleaned = validate_attributes(_descriptor(fields), attrs)
        assert cleaned == attrs

    def test_float_accepted_for_number(self):
        fields = [CustomFieldDescriptor(name="ratio", datatype=CustomFieldType.NUMBER)]
        cleaned = validate_attributes(_descriptor(fields), {"ratio": 1.5})
        assert cleaned["ratio"] == 1.5

    def test_optional_field_absent_is_omitted(self):
        fields = [
            CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING),
            CustomFieldDescriptor(name="note", datatype=CustomFieldType.STRING),
        ]
        cleaned = validate_attributes(_descriptor(fields), {"title": "x"})
        assert cleaned == {"title": "x"}
        assert "note" not in cleaned


class TestValidateAttributesErrors:
    """Invalid attributes raise CustomEntityValidationError with field detail."""

    def test_unknown_key_rejected(self):
        fields = [CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING)]
        with pytest.raises(CustomEntityValidationError) as exc:
            validate_attributes(_descriptor(fields), {"title": "x", "bogus": 1})
        assert any(e["field"] == "bogus" for e in exc.value.errors)

    def test_required_field_missing(self):
        fields = [
            CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING, required=True)
        ]
        with pytest.raises(CustomEntityValidationError) as exc:
            validate_attributes(_descriptor(fields), {})
        assert exc.value.errors[0]["field"] == "title"
        assert "required" in exc.value.errors[0]["message"]

    def test_bool_rejected_for_number(self):
        fields = [CustomFieldDescriptor(name="count", datatype=CustomFieldType.NUMBER)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"count": True})

    def test_enum_value_not_allowed(self):
        fields = [
            CustomFieldDescriptor(
                name="level",
                datatype=CustomFieldType.ENUM,
                enum_values=["low", "high"],
            )
        ]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"level": "medium"})

    def test_date_rejects_datetime(self):
        fields = [CustomFieldDescriptor(name="due", datatype=CustomFieldType.DATE)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"due": "2026-01-15T10:00:00"})

    def test_array_of_non_strings_rejected(self):
        fields = [CustomFieldDescriptor(name="labels", datatype=CustomFieldType.ARRAY_STRING)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"labels": ["a", 2]})

    def test_array_over_max_items_rejected(self):
        fields = [CustomFieldDescriptor(name="labels", datatype=CustomFieldType.ARRAY_STRING)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"labels": ["x"] * (MAX_ARRAY_ITEMS + 1)})

    def test_string_over_max_len_rejected(self):
        fields = [CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"title": "x" * (MAX_STRING_LEN + 1)})

    def test_text_over_max_len_rejected(self):
        fields = [CustomFieldDescriptor(name="body", datatype=CustomFieldType.TEXT)]
        with pytest.raises(CustomEntityValidationError):
            validate_attributes(_descriptor(fields), {"body": "x" * (MAX_TEXT_LEN + 1)})

    def test_all_errors_collected_in_one_raise(self):
        fields = [
            CustomFieldDescriptor(name="title", datatype=CustomFieldType.STRING, required=True),
            CustomFieldDescriptor(name="count", datatype=CustomFieldType.NUMBER),
        ]
        with pytest.raises(CustomEntityValidationError) as exc:
            validate_attributes(_descriptor(fields), {"count": "not-a-number", "extra": 1})
        fields_with_errors = {e["field"] for e in exc.value.errors}
        assert {"title", "count", "extra"} <= fields_with_errors
