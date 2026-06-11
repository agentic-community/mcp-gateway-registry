"""
Descriptor-driven validator for custom entity attributes.

``validate_attributes`` checks a record's ``attributes`` bag against its
type descriptor: rejects unknown keys, enforces required fields, and
validates each value's datatype/enum membership/length bounds. ALL errors
are collected and raised once so a client can fix everything in a single
round-trip.
"""

import logging
from datetime import date
from typing import Any

from ..schemas.custom_entity_models import (
    MAX_ARRAY_ITEMS,
    MAX_STRING_LEN,
    MAX_TEXT_LEN,
    CustomFieldDescriptor,
    CustomFieldType,
    CustomTypeDescriptor,
)
from .custom_entity_errors import CustomEntityValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _require_iso8601_date(
    field_name: str,
    value: Any,
) -> None:
    """Validate a v1 DATE value is a calendar date 'YYYY-MM-DD'.

    Rejects datetimes/times so the native ``<input type=date>`` widget output
    and the stored value always agree.
    """
    if not isinstance(value, str):
        raise CustomEntityValidationError(field_name, "expected a date string YYYY-MM-DD")
    try:
        date.fromisoformat(value)  # rejects 'YYYY-MM-DDTHH:MM:SS'
    except ValueError as e:
        raise CustomEntityValidationError(field_name, "expected ISO date YYYY-MM-DD") from e


def _coerce_and_check(
    f: CustomFieldDescriptor,
    value: Any,
) -> Any:
    """Validate a single attribute value against its field descriptor.

    Returns the (unchanged) value on success; raises
    ``CustomEntityValidationError`` for a single field on failure.
    """
    match f.datatype:
        case CustomFieldType.NUMBER:
            if not isinstance(value, (int | float)) or isinstance(value, bool):
                raise CustomEntityValidationError(f.name, "expected a number")
        case CustomFieldType.BOOL:
            if not isinstance(value, bool):
                raise CustomEntityValidationError(f.name, "expected a boolean")
        case CustomFieldType.ENUM:
            if value not in (f.enum_values or []):
                raise CustomEntityValidationError(f.name, f"must be one of {f.enum_values}")
        case CustomFieldType.ARRAY_STRING:
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise CustomEntityValidationError(f.name, "expected array of strings")
            if len(value) > MAX_ARRAY_ITEMS:
                raise CustomEntityValidationError(f.name, f"max {MAX_ARRAY_ITEMS} items")
            for x in value:
                if len(x) > MAX_STRING_LEN:
                    raise CustomEntityValidationError(
                        f.name, f"item exceeds {MAX_STRING_LEN} chars"
                    )
        case CustomFieldType.DATE:
            _require_iso8601_date(f.name, value)
        case CustomFieldType.TEXT:
            if not isinstance(value, str):
                raise CustomEntityValidationError(f.name, "expected a string")
            if len(value) > MAX_TEXT_LEN:
                raise CustomEntityValidationError(f.name, f"exceeds {MAX_TEXT_LEN} chars")
        case _:  # STRING
            if not isinstance(value, str):
                raise CustomEntityValidationError(f.name, "expected a string")
            if len(value) > MAX_STRING_LEN:
                raise CustomEntityValidationError(f.name, f"exceeds {MAX_STRING_LEN} chars")
    return value


def validate_attributes(
    descriptor: CustomTypeDescriptor,
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Validate + coerce a record's attributes against its type descriptor.

    Collects ALL errors and raises once — matches the Pydantic/FastAPI
    multi-error convention so API clients fix all issues in a single
    round-trip.

    Args:
        descriptor: The custom type descriptor defining the allowed fields.
        attributes: The record's attribute bag to validate.

    Returns:
        The cleaned attributes dict (only known, validated fields).

    Raises:
        CustomEntityValidationError: If any attribute is invalid; ``errors``
            holds one ``{"field", "message"}`` entry per problem.
    """
    known = {f.name: f for f in descriptor.fields}
    errors: list[dict[str, str]] = []

    # Reject unknown keys.
    for key in attributes:
        if key not in known:
            errors.append({"field": key, "message": "unknown field for this type"})

    cleaned: dict[str, Any] = {}
    for f in descriptor.fields:
        present = f.name in attributes
        if f.required and not present:
            errors.append({"field": f.name, "message": "required field missing"})
            continue
        if not present:
            continue
        try:
            cleaned[f.name] = _coerce_and_check(f, attributes[f.name])
        except CustomEntityValidationError as e:
            errors.append({"field": e.field or f.name, "message": e.message or "invalid value"})

    if errors:
        raise CustomEntityValidationError(errors=errors)
    return cleaned
