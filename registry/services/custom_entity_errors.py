"""
Domain exceptions for custom entity types.

Each exception maps to an HTTP status in the route layer's exception
handler. ``CustomEntityValidationError`` carries a list of per-field
errors so a client can fix all attribute issues in a single round-trip
(matching the Pydantic/FastAPI multi-error convention).
"""

from ..exceptions import RegistryError


class CustomEntityError(RegistryError):
    """Base exception for custom entity operations."""

    pass


class CustomEntityValidationError(CustomEntityError):
    """Attribute(s) failed datatype/enum/required/length validation (HTTP 400).

    Collects ALL field errors so the client sees every problem at once.
    Construct either with a list of error dicts (``errors=[...]``) or with a
    single ``(field, message)`` pair for convenience inside the validator.
    """

    def __init__(
        self,
        field: str | None = None,
        message: str | None = None,
        *,
        errors: list[dict[str, str]] | None = None,
    ):
        if errors is not None:
            self.errors = errors
        elif field is not None:
            self.errors = [{"field": field, "message": message or "invalid value"}]
        else:
            self.errors = []
        # Convenience accessors for the single-error construction path.
        self.field = field
        self.message = message
        super().__init__(f"Custom entity validation failed: {self.errors}")


class UnknownCustomTypeError(CustomEntityError):
    """A record operation referenced a type with no descriptor (HTTP 404)."""

    def __init__(
        self,
        type_name: str,
    ):
        self.type_name = type_name
        super().__init__(f"Unknown custom type: {type_name}")


class CustomEntityNotFoundError(CustomEntityError):
    """Update/delete of a non-existent (or non-visible) record (HTTP 404)."""

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        super().__init__(f"Custom entity record not found: {path}")


class CustomTypeAlreadyExistsError(CustomEntityError):
    """POST /api/custom-types with an existing name (HTTP 409)."""

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        super().__init__(f"Custom type '{name}' already exists")


class CustomTypeHasRecordsError(CustomEntityError):
    """DELETE type without ?force=true while records exist (HTTP 409)."""

    def __init__(
        self,
        name: str,
        count: int,
    ):
        self.name = name
        self.count = count
        super().__init__(
            f"Custom type '{name}' has {count} record(s); pass force=true to cascade-delete"
        )


class CustomTypeRecordCapError(CustomEntityError):
    """create_record when type is at MAX_CUSTOM_RECORDS_PER_TYPE soft cap (HTTP 409)."""

    def __init__(
        self,
        type_name: str,
        cap: int,
    ):
        self.type_name = type_name
        self.cap = cap
        super().__init__(f"Custom type '{type_name}' has reached the record cap of {cap}")


class CustomTypeLimitError(CustomEntityError):
    """create_type when the fleet is at MAX_CUSTOM_TYPES (HTTP 409)."""

    def __init__(
        self,
        cap: int,
    ):
        self.cap = cap
        super().__init__(f"Cannot create custom type: the limit of {cap} custom types has been reached")
