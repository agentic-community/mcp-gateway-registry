"""Shared HTTP header name and value validation.

Pure string validation with no encryption or storage concerns, so both the
encrypted per-server custom headers and plain-config headers (such as
EMBEDDINGS_EXTRA_HEADER_<N>) share one implementation. Copy-pasting these
checks is how header-injection findings get reopened.
"""

import re

MAX_HEADER_NAME_LENGTH: int = 256
MAX_HEADER_VALUE_LENGTH: int = 4096
DEFAULT_HEADER_CONTEXT: str = "custom_headers entry"
_RFC_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def validate_header_name(
    name: object,
    context: str = DEFAULT_HEADER_CONTEXT,
) -> str:
    """Return a bounded RFC 7230 token header name or raise ``ValueError``.

    Args:
        name: Candidate header name.
        context: Label used at the start of every error message, so the operator
            is told which setting or field holds the bad value.

    Returns:
        The validated header name, unchanged.

    Raises:
        ValueError: If the name is not a non-empty RFC token of bounded length.
    """
    if not isinstance(name, str):
        raise ValueError(f"{context} name must be a string")
    if not name:
        raise ValueError(f"{context} requires a non-empty name")
    if len(name) > MAX_HEADER_NAME_LENGTH:
        raise ValueError(f"{context} name exceeds {MAX_HEADER_NAME_LENGTH} characters")
    if not _RFC_TOKEN_RE.fullmatch(name):
        raise ValueError(f"{context} name must be a valid RFC token string")
    return name


def validate_header_value(
    value: object,
    allow_empty: bool = False,
    context: str = DEFAULT_HEADER_CONTEXT,
) -> str:
    """Return a bounded, control-free header value or raise ``ValueError``.

    Rejecting control characters is the CR/LF header-injection guard: a value
    containing a carriage return or line feed could otherwise split one header
    into several. Spaces are legal in HTTP header values and are allowed.

    Args:
        value: Candidate header value.
        allow_empty: Whether an empty string is accepted.
        context: Label used at the start of every error message.

    Returns:
        The validated header value, unchanged.

    Raises:
        ValueError: If the value is not a string, is empty when not allowed,
            is too long, or contains a control character.
    """
    if not isinstance(value, str):
        raise ValueError(f"{context} value must be a string")
    if not value and not allow_empty:
        raise ValueError(f"{context} requires a non-empty value")
    if len(value) > MAX_HEADER_VALUE_LENGTH:
        raise ValueError(f"{context} value exceeds {MAX_HEADER_VALUE_LENGTH} characters")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{context} value cannot contain control characters")
    return value
